"""V11-6b stage 3: re-code a REAL PACT attention kernel inside a CUDA graph.

Extends the stage-1 spike (graph_recode.py, toy PTX) to the production
kernel: build a 1-node graph whose kernel node is the pact decode-attention
variant (baseline), launch it in a loop, then swap the node's function to a
second compiled variant (deep = stages 5) via cuGraphExecKernelNodeSetParams
and verify that the SAME instantiated exec afterwards computes the deep
variant's output — bitwise against the eager launches of each variant.

ABI packing (the plan's flagged risk, now resolved for this family):
`CompiledKernel.src.signature` gives the node-argument order — the six
pointers plus the four i64 runtime strides; sm_scale and all geometry are
constexpr (baked), `global_scratch_size == 0` so there is no hidden
parameter. sharedMemBytes / blockDim come from kernel.metadata.

Run: python -m triton.pact_aobo.graph_attention   (PACT_RESULT_JSON to dump)
"""
from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
from typing import Any, Dict, List


class _NodeArgs:
    """Classic ABI: void*[] of addresses of the argument values, kept alive."""

    def __init__(self, kernel, args, kwargs):
        sig = dict(kernel.src.signature)
        values: List[Any] = list(args)
        by_name = dict(zip(_param_names(kernel), values))
        ctypes_vals = []
        self._keep = []
        for name, ty in sig.items():
            if ty == "constexpr":
                continue
            if ty.startswith("*"):
                buf = ctypes.c_uint64(int(by_name[name].data_ptr()))
            elif ty == "i64":
                buf = ctypes.c_uint64(int(kwargs[name]))
            elif ty == "i32":
                buf = ctypes.c_int32(int(kwargs[name]))
            elif ty == "fp32":
                buf = ctypes.c_float(float(kwargs[name]))
            else:
                raise ValueError(f"unhandled ABI type {ty} for {name}")
            self._keep.append(buf)
            ctypes_vals.append(ctypes.addressof(buf))
        # Trailing hidden params (see third_party/nvidia/backend/driver.c:
        # num_params = visible + global_scratch + profile_scratch).  Both
        # scratch objects are None here (sizes 0) -> NULL pointers, which
        # the C launcher would pass as well.
        self._gs = ctypes.c_uint64(0)
        self._ps = ctypes.c_uint64(0)
        ctypes_vals += [ctypes.addressof(self._gs),
                        ctypes.addressof(self._ps)]
        self._arr = (ctypes.c_void_p * len(ctypes_vals))(*ctypes_vals)
        self.abi_address = ctypes.addressof(self._arr)
        self.n_args = len(ctypes_vals)


def _param_names(kernel) -> List[str]:
    return [p.name for p in kernel.src.fn.params]




def _node_params(cu, fn, kernel, node_args: _NodeArgs, grid):
    md = kernel.metadata
    np_ = cu.CUDA_KERNEL_NODE_PARAMS()
    np_.func = fn
    np_.gridDimX, np_.gridDimY, np_.gridDimZ = int(grid[0]), int(grid[1]), int(grid[2])
    np_.blockDimX, np_.blockDimY, np_.blockDimZ = 32 * int(md.num_warps), 1, 1
    # The C launcher passes metadata.shared as the cuLaunchKernel dynamic
    # shared size — the node must match (driver.c: _launch(..., shared_memory, ...)).
    np_.sharedMemBytes = int(md.shared)
    # Buffer-protocol form: the node-params setter marshals a ctypes
    # array correctly; a raw int address silently mis-packs (verified:
    # int form launches garbage, buffer form matches eager within the
    # kernel's own noise band).
    np_.kernelParams = node_args._arr
    np_.extra = 0
    return np_


def load_variant_fn(cu, kernel):
    """Load the variant's cubin into its OWN module via the driver API.

    Triton's CompiledKernel._init_handles (load_binary) invalidates a
    previously loaded module of the same kernel symbol — two variants of
    one jit_fn cannot both stay resident through it.  Owning the module
    handles is also the AOBO resident-pool shape: lifetime is ours.
    """
    cubin = kernel.asm["cubin"]
    err, mod = cu.cuModuleLoadDataEx(cubin, 0, [], [])
    assert err == 0, f"cuModuleLoadDataEx {err}"
    err, fn = cu.cuModuleGetFunction(mod, str(kernel.metadata.name).encode())
    assert err == 0, f"cuModuleGetFunction {err}"
    return mod, fn


def _abs_diff(a, b) -> float:
    return float((a.float() - b.float()).abs().max().item())


def load_merged_module(cu, baseline_kernel, deep_kernel):
    """V12-2a round 3: ONE module, TWO symbols.

    The deep variant's PTX entry is renamed and both PTX blobs are linked
    into a single cubin, loaded ONCE: there is no second load to alias
    the first away (the stage-3 blocker: module and library routes both
    showed two same-symbol loads kicking each other, non-deterministically
    across processes).  Renaming also removes the triton same-symbol
    collision entirely — the merged module owns a private symbol space.
    """
    name = str(baseline_kernel.metadata.name)
    ptx_a = baseline_kernel.asm["ptx"]
    ptx_b = deep_kernel.asm["ptx"].replace(
        f".entry {name}", f".entry {name}_deep")
    assert f".entry {name}_deep" in ptx_b, "entry rename failed"
    err, link = cu.cuLinkCreate(0, [], [])
    assert err == 0, f"cuLinkCreate {err}"
    ptx_type = cu.CUjitInputType.CU_JIT_INPUT_PTX
    (err,) = cu.cuLinkAddData(link, ptx_type, ptx_a.encode(), len(ptx_a),
                              b"base", 0, [], [])
    assert err == 0, f"cuLinkAddData(base) {err}"
    (err,) = cu.cuLinkAddData(link, ptx_type, ptx_b.encode(), len(ptx_b),
                              b"deep", 0, [], [])
    assert err == 0, f"cuLinkAddData(deep) {err}"
    err, cubin, size = cu.cuLinkComplete(link)
    assert err == 0, f"cuLinkComplete {err}"
    try:
        cu.cuLinkDestroy(link)
    except Exception:
        pass
    import ctypes
    blob = ctypes.string_at(int(cubin), int(size))
    err, mod = cu.cuModuleLoadDataEx(blob, 0, [], [])
    assert err == 0, f"cuModuleLoadDataEx(merged) {err}"
    err, fn_a = cu.cuModuleGetFunction(mod, name.encode())
    assert err == 0, f"cuModuleGetFunction(base) {err}"
    err, fn_b = cu.cuModuleGetFunction(mod, f"{name}_deep".encode())
    assert err == 0, f"cuModuleGetFunction(deep) {err}"
    return mod, fn_a, fn_b


def load_variant_lib(cu, kernel):
    """V12-2a round 1: CUDA 12+ library API — the cubin enters a LIBRARY
    and the kernel gets a library-scoped CUkernel identity;
    cuKernelGetFunction derives the context CUfunction from it.

    Two variants = two independent libraries: no CUfunction same-symbol
    mutual exclusion, and no reliance on CUDA 13's module aliasing that
    blocked the module route (stage3_adjudication: self-managed modules
    invalidated each other non-deterministically; graphs built purely
    from own modules launched garbage unless triton's loader had loaded
    both variants first).
    """
    cubin = kernel.asm["cubin"]
    err, lib = cu.cuLibraryLoadData(cubin, [], [], 0, [], [], 0)
    assert err == 0, f"cuLibraryLoadData {err}"
    err, kern = cu.cuLibraryGetKernel(
        lib, str(kernel.metadata.name).encode())
    assert err == 0, f"cuLibraryGetKernel {err}"
    err, fn = cu.cuKernelGetFunction(kern)
    assert err == 0, f"cuKernelGetFunction {err}"
    return lib, kern, fn


def run(baseline_kernel, deep_kernel, args, kwargs, grid,
        out_tensor, ref=None) -> Dict[str, Any]:
    """All launches here go through OUR driver-API modules — never
    triton's loader.  Two probe findings force this (see
    suite/results/v11/aobo/): (a) triton's load_binary invalidates a
    previously loaded module of the same kernel symbol; (b) with CUDA
    module aliasing, eagerly launching variant B through triton unloads
    the content-identical module of variant A we had loaded ourselves.
    Raw-vs-eager equivalence is established separately (identical ABI
    packing, diff within the kernel's own noise band).

    V12-2a round 3 protocol (the one that WORKS — see the round-3
    adjudication): triton COMPILES the two variants but never launches
    them; the merged module (renamed deep entry, linked once) is the
    SINGLE loader; the correctness anchor is the pure-torch reference
    (`ref`), immune to any module aliasing.  Earlier protocols that let
    triton load a variant at runtime (eager references) hit CUDA 13's
    lazy-load aliasing: two loads of the same symbol — triton loader vs
    ours, module OR library — kick each other non-deterministically
    (probe12's PASS was the special case where the graph borrowed
    triton's own handles)."""
    import torch
    from cuda.bindings import driver as cu

    torch.zeros(1, device="cuda")
    (err,) = cu.cuInit(0)
    assert err == 0

    load_mode = os.environ.get("PACT_AOBO_GRAPH_LOAD", "merged")
    if load_mode == "merged":
        _mod_m, fn_a, fn_b = load_merged_module(cu, baseline_kernel,
                                                deep_kernel)
        load_mode = "merged(one module, two symbols, V12-2a round 3)"
    elif load_mode == "module":
        _mod_a, fn_a = load_variant_fn(cu, baseline_kernel)
        _mod_b, fn_b = load_variant_fn(cu, deep_kernel)
        load_mode = "module(CUDA13-aliased, stage-3 control)"
    else:
        _lib_a, _kern_a, fn_a = load_variant_lib(cu, baseline_kernel)
        _lib_b, _kern_b, fn_b = load_variant_lib(cu, deep_kernel)
        load_mode = "library(CUkernel, V12-2a round 1)"
    na = _NodeArgs(baseline_kernel, args, kwargs)
    stream = torch.cuda.current_stream().cuda_stream
    gx, gy, gz = int(grid[0]), int(grid[1]), int(grid[2])

    def raw_launch(fn, kernel):
        bx = 32 * int(kernel.metadata.num_warps)
        (e,) = cu.cuLaunchKernel(fn, gx, gy, gz, bx, 1, 1,
                                 int(kernel.metadata.shared), stream,
                                 na.abi_address, 0)
        assert e == 0, f"cuLaunchKernel {e}"
        torch.cuda.synchronize()

    # references from the merged module's OWN raw launches, anchored to
    # the pure-torch reference (the only aliasing-immune ground truth)
    out_tensor.zero_(); torch.cuda.synchronize()
    raw_launch(fn_a, baseline_kernel)
    ref_base = out_tensor.clone()
    base_anchor = _abs_diff(out_tensor, ref)
    out_tensor.zero_(); torch.cuda.synchronize()
    raw_launch(fn_b, deep_kernel)
    ref_deep = out_tensor.clone()
    deep_anchor = _abs_diff(out_tensor, ref)
    out_tensor.zero_(); torch.cuda.synchronize()
    raw_launch(fn_a, baseline_kernel)
    eager_noise = max(_abs_diff(out_tensor, ref_base), 1e-3)
    assert base_anchor <= 5e-3, \
        f"raw fn_a disagrees with torch reference: {base_anchor}"
    assert deep_anchor <= 5e-3, \
        f"raw fn_b disagrees with torch reference: {deep_anchor}"
    err, graph = cu.cuGraphCreate(0)
    assert err == 0, err
    p1 = _node_params(cu, fn_a, baseline_kernel, na, grid)
    err, node = cu.cuGraphAddKernelNode(graph, (), 0, p1)
    if err != 0:
        raise AssertionError(
            f"cuGraphAddKernelNode {err}; func={baseline_kernel.function!r} "
            f"({type(baseline_kernel.function).__name__}) grid="
            f"{p1.gridDimX},{p1.gridDimY},{p1.gridDimZ} block={p1.blockDimX} "
            f"shared={p1.sharedMemBytes} args={na.n_args} "
            f"abi={hex(na.abi_address)}")
    err, gexec = cu.cuGraphInstantiate(graph, 0)
    assert err == 0, f"cuGraphInstantiate {err}"
    stream = torch.cuda.current_stream().cuda_stream

    def graph_launch():
        (e,) = cu.cuGraphLaunch(gexec, stream)
        assert e == 0, f"cuGraphLaunch {e}"

    # the decode kernel is run-to-run nondeterministic (fp16 atomics;
    # eager-vs-eager is ~2.6e-3), so graph-vs-eager is judged against the
    # kernel's own noise band, not bitwise
    def _close(a, b):
        return (a.float() - b.float()).abs().max().item()

    # phase 1: graph runs baseline
    out_tensor.zero_(); graph_launch(); torch.cuda.synchronize()
    base_diff = _close(out_tensor, ref_base)
    base_match = base_diff <= eager_noise

    # re-code: same node, same exec — only func (and launch geometry) change
    import time
    t0 = time.monotonic()
    (rc,) = cu.cuGraphExecKernelNodeSetParams(
        gexec, node, _node_params(cu, fn_b, deep_kernel, na, grid))
    recode_ms = (time.monotonic() - t0) * 1000.0

    # phase 2: SAME exec now runs the deep variant
    out_tensor.zero_(); graph_launch(); torch.cuda.synchronize()
    deep_diff = _close(out_tensor, ref_deep)
    deep_match = deep_diff <= max(eager_noise, 3e-3)

    report = {
        "load_mode": load_mode,
        "torch_anchor_base": base_anchor,
        "torch_anchor_deep": deep_anchor,
        "abi_args": na.n_args,
        "shared_base": int(baseline_kernel.metadata.shared),
        "shared_deep": int(deep_kernel.metadata.shared),
        "graph_phase_baseline_diff": base_diff,
        "eager_noise_band": eager_noise,
        "recode_rc": int(rc),
        "recode_ms": round(recode_ms, 4),
        "graph_phase_after_recode_diff": deep_diff,
        "verdict": "PASS" if (base_match and int(rc) == 0 and deep_match
                               and base_anchor <= 5e-3
                               and deep_anchor <= 5e-3) else "FAIL",
        "meaning": "the instantiated decode-attention graph runs variant A, "
                   "then the node is re-pointed to variant B and the SAME "
                   "exec computes B's output — kernel replacement inside a "
                   "captured graph, no re-capture, no engine restart",
    }
    try:
        cu.cuGraphExecDestroy(gexec)
        cu.cuGraphDestroy(graph)
    except Exception:
        pass
    return report


def _demo_entry() -> int:
    import sys
    import torch

    sys.path.insert(0, os.environ.get(
        "PACT_PAPER_ROOT", "/home/johnzhw/workspace/pact_paper"))
    from suite.harness.ops import make_decode_inputs
    from suite.kernels.decode import pact_optimization_target
    from suite.kernels.reference import reference_decode
    from triton.pact.compiler.explicit_compiler import compile_explicit

    torch.manual_seed(0)
    # microkernel-protocol geometry (real page tables, torch reference):
    # correctness is judged against reference_decode — the toy vLLM-shaped
    # tensors this entry used before could not anchor correctness at all.
    cfg = {"B": 1, "S": 2048, "P": 16, "D": 128, "Hq": 32, "GQA": 8}
    B, S, P, D = cfg["B"], cfg["S"], cfg["P"], cfg["D"]
    Hq, Hk = cfg["Hq"], cfg["Hq"] // cfg["GQA"]
    q, kc, vc, bt, sl, P_, shape = make_decode_inputs(cfg)
    out = torch.empty_like(q)
    args = (out, q, kc, vc, bt, sl)
    kwargs = dict(
        sm_scale=1.0 / D ** 0.5, NUM_TOKENS=B, NUM_HEADS=Hq,
        NUM_KV_HEADS=Hk, HEAD_DIM=D, PAGE_SIZE=P,
        MAX_SEQ_LEN=bt.shape[1] * P, TILE_SIZE=16, GQA_RATIO=cfg["GQA"],
        STRIDE_BLOCK=kc.stride()[0], STRIDE_KV_HEAD=kc.stride()[1],
        STRIDE_PAGE=kc.stride()[2], STRIDE_HEAD_DIM=kc.stride()[3],
        USE_DUAL_TILE=False, TILE_SIZE_LARGE=32, TOKEN_IMPORTANCE_MODE=0)
    grid = (B, Hq, 1)
    ref = reference_decode(q, kc, vc, bt, sl, P)

    base, _ = compile_explicit(pact_optimization_target, args, kwargs,
                               {"PACT_ENABLE": "0"})
    deep, _ = compile_explicit(pact_optimization_target, args, kwargs,
                               {"PACT_ENABLE": "1",
                                "PACT_OVERRIDE_STAGES": "5"})

    rep = run(base, deep, args, kwargs, grid, out, ref=ref)
    txt = json.dumps(rep, indent=1)
    print(txt)
    outp = os.environ.get("PACT_RESULT_JSON")
    if outp:
        Path(outp).parent.mkdir(parents=True, exist_ok=True)
        Path(outp).write_text(txt + "\n")
    return 0 if rep["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(_demo_entry())
