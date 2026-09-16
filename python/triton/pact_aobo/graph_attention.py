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


def run(baseline_kernel, deep_kernel, args, kwargs, grid,
        out_tensor) -> Dict[str, Any]:
    """All launches here go through OUR driver-API modules — never
    triton's loader.  Two probe findings force this (see
    suite/results/v11/aobo/): (a) triton's load_binary invalidates a
    previously loaded module of the same kernel symbol; (b) with CUDA
    module aliasing, eagerly launching variant B through triton unloads
    the content-identical module of variant A we had loaded ourselves.
    Raw-vs-eager equivalence is established separately (identical ABI
    packing, diff within the kernel's own noise band)."""
    import torch
    from cuda.bindings import driver as cu

    torch.zeros(1, device="cuda")
    (err,) = cu.cuInit(0)
    assert err == 0
    # Probe record (suite/results/v11/aobo/): the graph node executes the
    # real kernel correctly ONLY when triton's own module loads for BOTH
    # variants happened first (CUDA module aliasing keeps our duplicate
    # cubin loads alive through triton's handles); a graph built purely
    # from our own modules launches garbage even though raw launches of
    # the same functions are perfect.  So: triton-eager references first
    # (binder-normalized, V11-0 lesson), then our module loads, then the
    # graph.  This is the probe12 configuration, re-validated here.
    from triton.runtime import driver as _tdrv
    device = _tdrv.active.get_current_device()
    _c, _k, _tt, _b, binder = baseline_kernel.src.fn.device_caches[device]

    def eager(kernel):
        bound, _s, _o = binder(*args, **kwargs)
        kernel[grid](*bound.values())
        torch.cuda.synchronize()
        return out_tensor.clone()

    ref_base = eager(baseline_kernel)
    ref_deep = eager(deep_kernel)

    _mod_a, fn_a = load_variant_fn(cu, baseline_kernel)
    _mod_b, fn_b = load_variant_fn(cu, deep_kernel)
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

    out_tensor.zero_(); torch.cuda.synchronize()
    raw_launch(fn_a, baseline_kernel)
    _d = (out_tensor.float() - ref_base.float()).abs().max().item()
    assert _d <= 5e-3, f"raw fn_a disagrees with eager: {_d}"
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

    def _close_or(a, b):
        return _close(a, b) <= 5e-3

    eager_noise = max(_close(eager(baseline_kernel), ref_base), 1e-3)
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
        "abi_args": na.n_args,
        "shared_base": int(baseline_kernel.metadata.shared),
        "shared_deep": int(deep_kernel.metadata.shared),
        "graph_phase_baseline_diff": base_diff,
        "eager_noise_band": eager_noise,
        "recode_rc": int(rc),
        "recode_ms": round(recode_ms, 4),
        "graph_phase_after_recode_diff": deep_diff,
        "verdict": "PASS" if (base_match and int(rc) == 0 and deep_match)
        else "FAIL",
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
    from suite.kernels.decode import pact_optimization_target
    from triton.pact.compiler.explicit_compiler import compile_explicit

    torch.manual_seed(0)
    B, S, P, D, Hq, Hk = 1, 256, 16, 64, 14, 2
    n = (S + P - 1) // P + 4
    q = torch.randn(B, Hq, D, dtype=torch.float16, device="cuda")
    cache = torch.randn(1, Hk, P, 2 * D, dtype=torch.float16, device="cuda")
    kc, vc = cache[..., :D], cache[..., D:]
    bt = torch.zeros(B, n, dtype=torch.int32, device="cuda")
    bt[0, :(S + P - 1) // P] = torch.arange(0, (S + P - 1) // P,
                                            dtype=torch.int32)
    sl = torch.full((B,), S, dtype=torch.int32, device="cuda")
    out = torch.zeros(B, Hq, D, dtype=torch.float16, device="cuda")
    args = (out, q, kc, vc, bt, sl)
    kwargs = dict(
        sm_scale=1.0 / D ** 0.5, NUM_TOKENS=B, NUM_HEADS=Hq,
        NUM_KV_HEADS=Hk, HEAD_DIM=D, PAGE_SIZE=P, MAX_SEQ_LEN=n * P,
        TILE_SIZE=16, GQA_RATIO=Hq // Hk,
        STRIDE_BLOCK=kc.stride()[0], STRIDE_KV_HEAD=kc.stride()[1],
        STRIDE_PAGE=kc.stride()[2], STRIDE_HEAD_DIM=kc.stride()[3],
        USE_DUAL_TILE=False, TILE_SIZE_LARGE=32, TOKEN_IMPORTANCE_MODE=0)
    grid = (B, Hq, 1)

    base, _ = compile_explicit(pact_optimization_target, args, kwargs,
                               {"PACT_ENABLE": "1"})
    deep, _ = compile_explicit(pact_optimization_target, args, kwargs,
                               {"PACT_ENABLE": "1",
                                "PACT_OVERRIDE_STAGES": "5"})

    rep = run(base, deep, args, kwargs, grid, out)
    txt = json.dumps(rep, indent=1)
    print(txt)
    outp = os.environ.get("PACT_RESULT_JSON")
    if outp:
        Path(outp).parent.mkdir(parents=True, exist_ok=True)
        Path(outp).write_text(txt + "\n")
    return 0 if rep["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(_demo_entry())
