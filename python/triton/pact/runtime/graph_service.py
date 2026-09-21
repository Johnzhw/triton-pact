"""V13 Phase0 (plan D1): in-graph kernel retarget service.

The graph-mode counterpart of the launch-boundary slot exchange.  Under a
torch-captured CUDA graph (vLLM FULL), a kernel node snapshots its
CUfunction at capture time: swapping in-process slots or re-loading modules
never reaches an already-captured node ("fake replacement" layer 3).  This
service retargets the captured node itself via
cuGraphExecKernelNodeSetParams — no re-capture, no engine restart.

Protocol (inherited from the 6b(3) merged single-loader winner,
graph_attention.py; adjudicated 20/20 fresh processes in V12-2a):
  * triton COMPILES the variants but the merged module is the only loader
    for graph retargeting — every variant PTX entry is renamed to
    ``{name}__{variant}``, so the module shares NO symbol with anything
    triton loaded (CUDA 13 lazy-load aliasing cannot kick the captured
    baseline node's module);
  * the baseline node keeps the triton-loaded CUfunction it was captured
    with; retarget only swaps func/blockDim/sharedMemBytes and REUSES the
    node's original kernelParams pointer array wholesale (ABI-identical
    variants of one jit_fn);
  * SetParams semantics guarantee in-flight replays are untouched — the
    retarget takes effect from the NEXT replay (verified by the Phase0
    probe, suite/probe/graph_phase0_probe.py, in-flight isolation PASS).

Threading (async-frame contract): ``offer`` (link+load, ms-scale, takes
the driver lock) runs on the frame's background worker; ``submit_retarget``
is a us-scale pending flag consumed at the replay boundary by
``before_replay`` — the decode step's operators are never blocked.

Handles come from the official torch 2.13 API: the engine process wraps
``torch.cuda.CUDAGraph.__new__`` so vLLM captures with ``keep_graph=True``
(see suite/harness/vllm_patch.py), which exposes raw_cuda_graph() /
raw_cuda_graph_exec().  Everything else is cuda.bindings.

Default-off: this module is imported only when PACT_GRAPH_SERVICE=1 arms
the harness; no other path touches it.
"""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

_LOCK = threading.RLock()
_SERVICE: Optional["GraphKernelService"] = None
_REPLAY_HOOKED = False

# One module, every entry renamed — the 6b(3) single-loader regime.  The
# graph-capture baseline is __base (PTX-identical to the dormant kernel the
# pre-Phase0 path recorded), __vanilla is PACT-off for A/B, the rest mirror
# the family vocabulary of both trees (pgo family names alias onto them).
VARIANT_ENVS: Dict[str, Dict[str, str]] = {
    "__base": {"PACT_ENABLE": "1"},
    "__vanilla": {"PACT_ENABLE": "0"},
    "theory": {"PACT_ENABLE": "1"},
    "occ": {"PACT_ENABLE": "1", "PACT_OVERRIDE_WARPS": "2"},
    "lat": {"PACT_ENABLE": "1", "PACT_OVERRIDE_STAGES": "2"},
    "deep": {"PACT_ENABLE": "1", "PACT_OVERRIDE_STAGES": "5"},
    "short": {"PACT_ENABLE": "1", "PACT_OVERRIDE_STAGES": "1"},
    "w1": {"PACT_ENABLE": "1", "PACT_OVERRIDE_WARPS": "1"},
    # V14-B (S1): unlock the deep-pipeline range; P6's L2-residency gate
    # itself keeps hot shapes at the theory decision, so this module entry
    # is the graph-side spelling of the "cold" auto family.
    "cold": {"PACT_ENABLE": "1", "PACT_MAX_PIPELINE_STAGES": "5"},
}
FAMILY_ALIAS: Dict[str, str] = {
    "baseline": "__vanilla", "vanilla": "__vanilla",
    "occupancy": "occ", "latency": "lat",
}


def get_service() -> "GraphKernelService":
    """Process-wide singleton (one CUDA context, one merged module set)."""
    global _SERVICE
    with _LOCK:
        if _SERVICE is None:
            _SERVICE = GraphKernelService()
        return _SERVICE


class _NodeBinding:
    """One pact kernel node inside one bound torch graph."""

    __slots__ = ("node", "func_before", "warps_before", "shared_before",
                 "kernel_params")

    def __init__(self, node, func_before, warps_before, shared_before,
                 kernel_params):
        self.node = node                # CUgraphNode
        self.func_before = int(func_before)   # triton-loaded CUfunction
        self.warps_before = warps_before
        self.shared_before = shared_before
        self.kernel_params = kernel_params    # original void** (reused)


class GraphKernelService:
    def __init__(self):
        self._cu = None                 # cuda.bindings.driver (lazy)
        self._mods: Dict[str, Any] = {}         # jit name -> CUmodule
        self._kerns: Dict[Tuple[str, str], Any] = {}  # CompiledKernel ref
        self._jit_args: Dict[str, tuple] = {}   # jit -> (args, kwargs, grid)
        self._variants: Dict[Tuple[str, str], dict] = {}
        # (jit, variant) -> {"fn": CUfunction, "warps": int, "shared": int}
        self._bindings: Dict[int, list] = {}    # id(graph) -> [_NodeBinding]
        self._keep: Dict[int, Any] = {}         # id(graph) -> graph ref
        self._execs: Dict[int, int] = {}        # id(graph) -> CUgraphExec
        self._jits: Dict[int, str] = {}         # id(graph) -> jit name
        self._pending: Dict[str, str] = {}      # jit -> variant (queued)
        self._active: Dict[str, str] = {}       # jit -> variant (installed)
        self._graph_active: Dict[int, str] = {}  # id(graph) -> variant
        self._sink: Optional[Callable[[dict], None]] = None
        self._boundary: Optional[Callable[[], None]] = None
        self._target_jit: Optional[str] = None
        self._skip: set = set()          # id(graph) that failed to bind
        self.state: Dict[str, Any] = {
            "offers": 0, "loads": 0, "binds": 0, "nodes": 0,
            "retargets": 0, "rollbacks": 0, "last_recode_ms": None,
            "last_offer_ms": None, "active": None, "errors": [],
            "replay_calls": 0, "bound_graphs": 0, "skipped_graphs": 0,
        }

    # ------------------------------------------------------------------
    def _driver(self):
        if self._cu is None:
            import torch  # noqa: F401 - context must exist first
            from cuda.bindings import driver as cu
            (err,) = cu.cuInit(0)
            if err != cu.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f"cuInit {err}")
            self._cu = cu
        return self._cu

    def _err(self, where: str, exc: Exception) -> None:
        msg = f"{where}: {exc}"[:200]
        with _LOCK:
            self.state["errors"] = (self.state["errors"] + [msg])[-8:]
        self._publish()  # silent failure is invisible across the boundary

    def set_state_sink(self, sink: Optional[Callable[[dict], None]]) -> None:
        """Bridge callback: publish service state into PACT_BRIDGE_STATE."""
        self._sink = sink

    def set_boundary(self, boundary: Optional[Callable[[], None]]) -> None:
        """Bridge callback fired at every replay boundary of a bound graph
        — the graph-mode analogue of the launch boundary: the frame's
        install_at_boundary + decide_async run there (us-scale each)."""
        self._boundary = boundary

    def set_target(self, jit_name: str) -> None:
        """Jit name whose kernel nodes get retargeted (e.g.
        'pact_optimization_target'); unseen graphs are auto-bound to it."""
        self._target_jit = jit_name

    def _publish(self) -> None:
        if self._sink is not None:
            try:
                self._sink(dict(self.state))
            except Exception:
                pass

    # -- first-forward arm: ONE module, ALL variants ----------------------
    def prepare(self, jit_fn, args, kwargs, grid,
                variants: Optional[Dict[str, Dict[str, str]]] = None) -> bool:
        """Compile the variant vocabulary, assemble ONE multi-entry cubin
        with TRITON'S OWN ptxas (each entry renamed — the 6b(3) single
        loader), load it once and materialize every function.

        Runs at the FIRST patched forward (the vLLM profile run: eager,
        before any capture) — never inside a capture window or a replay
        boundary.  After this, `capture_launch` records __base into the
        graphs, so every captured node is BORN inside this module and
        every later switch is an intra-module SetParams — the regime the
        6b(3) adjudication validated (20/20 processes).  (Cross-module
        SetParams — captured triton function -> separately loaded module
        — silently corrupts execution under CUDA 13 in the real vLLM
        graph while passing every micro reproduction: stable 896/896
        wrong outputs with identical SASS, correct eager behaviour of the
        same function, and a bit-identical token stream for a SetParams
        round-trip that pointed back at the original function.  Evidence:
        suite/results/v13/accept/.)
        """
        cu = self._driver()
        name = str(getattr(jit_fn, "__name__", None)
                   or jit_fn.fn.__name__)
        with _LOCK:
            if self.state.get("prepared"):
                return True
        t0 = time.monotonic()
        try:
            from triton.pact.compiler.explicit_compiler import compile_explicit
            vocab = dict(variants or VARIANT_ENVS)
            kerns: Dict[str, Any] = {}
            for vname, env in vocab.items():
                k, _ = compile_explicit(jit_fn, args, kwargs, dict(env))
                kerns[vname] = k
            blob = self._merge_ptxas(cu, kerns, name)
            err, mod = cu.cuModuleLoadDataEx(blob, 0, [], [])
            if err != cu.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f"cuModuleLoadDataEx {err}")
            with _LOCK:
                self._mods[name] = mod
                for vname, k in kerns.items():
                    err, fn = cu.cuModuleGetFunction(
                        mod, f"{name}__{vname}".encode())
                    if err != cu.CUresult.CUDA_SUCCESS:
                        raise RuntimeError(f"GetFunction({vname}) {err}")
                    self._variants[(name, vname)] = {
                        "fn": fn,
                        "warps": int(k.metadata.num_warps),
                        "shared": int(k.metadata.shared),
                    }
                    self._kerns[(name, vname)] = k
                self._jit_args[name] = (tuple(args), dict(kwargs),
                                        tuple(grid))
            # settle the lazy loads NOW (one raw launch per function on
            # cloned outputs, off any serving path)
            for vname, k in kerns.items():
                self._materialize(cu, k, self._variants[(name, vname)]["fn"],
                                  args, kwargs, grid)
            with _LOCK:
                self.state["prepared"] = name
                self.state["variants"] = sorted(kerns)
                self.state["prepare_ms"] = round(
                    (time.monotonic() - t0) * 1000.0, 1)
                self.state["loads"] += 1
            self._publish()
            return True
        except Exception as e:  # noqa: BLE001 - service must not kill engine
            self._err("prepare", e)
            self._publish()
            return False

    def _merge_ptxas(self, cu, kerns: Dict[str, Any], name: str) -> bytes:
        """Concatenate every renamed entry into one PTX (labels
        namespaced per entry to avoid duplicate definitions) and assemble
        with triton's ptxas flags — no driver JIT anywhere, the SASS is
        exactly what triton itself would emit."""
        import triton
        ptxas = Path(triton.__file__).parent / "backends" / "nvidia" / \
            "bin" / "ptxas"
        if not ptxas.exists():
            import shutil
            ptxas = Path(shutil.which("ptxas") or "ptxas")
        import torch
        maj, mnr = torch.cuda.get_device_capability()

        def entry_of(k, suffix: str, tag: str) -> str:
            ptx = k.asm["ptx"]
            idx = ptx.find(".visible .entry")
            if idx < 0:
                idx = ptx.find(".entry")
            tail = ptx[idx:]
            tail = tail.replace(f".entry {name}", f".entry {name}{suffix}",
                                1)
            tail = re.sub(r"\$L__", f"${tag}_L__", tail)
            return "\n".join(l for l in tail.splitlines()
                             if not l.strip().startswith(".file"))

        first = next(iter(kerns.values())).asm["ptx"]
        hdr_end = first.find(".visible .entry")
        if hdr_end < 0:
            hdr_end = first.find(".entry")
        hdr = "\n".join(l for l in first[:hdr_end].splitlines()
                        if not l.strip().startswith(".file"))
        parts = [hdr]
        for i, (vname, k) in enumerate(kerns.items()):
            parts.append(entry_of(k, f"__{vname}", f"V{i}"))
        merged = "\n".join(parts) + "\n"
        with tempfile.TemporaryDirectory(prefix="pact_gs_") as td:
            src = Path(td) / "m.ptx"
            src.write_text(merged)
            cub = Path(td) / "m.cubin"
            r = subprocess.run(
                [str(ptxas), "--regAllocOptLevel=2",
                 f"--gpu-name=sm_{maj}{mnr}", str(src), "-o", str(cub)],
                capture_output=True, timeout=300)
            if r.returncode != 0 or not cub.exists():
                raise RuntimeError("ptxas rc="
                                   f"{r.returncode}: "
                                   f"{r.stderr.decode(errors='ignore')[:200]}")
            return cub.read_bytes()

    def _materialize(self, cu, kernel, fn, args, kwargs, grid) -> None:
        """One raw launch on cloned tensors to settle the lazy load."""
        if not args or not kwargs or not grid:
            return
        import torch
        values = [torch.empty_like(args[0])] + list(args[1:])
        na = self._abi_args(kernel, tuple(values), kwargs)
        stream = torch.cuda.current_stream().cuda_stream
        (rc,) = cu.cuLaunchKernel(
            fn, int(grid[0]), int(grid[1]), int(grid[2]),
            32 * int(kernel.metadata.num_warps), 1, 1,
            int(kernel.metadata.shared), stream, na, 0)
        torch.cuda.synchronize()
        if rc != 0:
            raise RuntimeError(f"materialize launch rc={rc}")

    def capture_launch(self, args, kwargs, grid) -> bool:
        """Launch __base raw INSIDE the capture window so the recorded
        node is born pointing at the prepared module.  Capture-legal
        (plain cuLaunchKernel); falls back to False when unprepared."""
        with _LOCK:
            name = self.state.get("prepared")
            if not name:
                return False
            v = self._variants.get((name, "__base"))
            kern = self._kerns.get((name, "__base"))
        if v is None:
            return False
        na = self._abi_args(kern, tuple(args), kwargs)
        import torch
        (rc,) = self._cu.cuLaunchKernel(
            v["fn"], int(grid[0]), int(grid[1]), int(grid[2]),
            32 * v["warps"], 1, 1, v["shared"],
            torch.cuda.current_stream().cuda_stream, na, 0)
        return rc == 0

    @staticmethod
    def _abi_args(kernel, args, kwargs) -> int:
        """void*[] ABI pack (the 6b(3) _NodeArgs recipe: buffer-protocol
        ctypes array, two NULL scratch tail params)."""
        import ctypes
        sig = dict(kernel.src.signature)
        names = [p.name for p in kernel.src.fn.params]
        by_name = dict(zip(names, list(args)))
        ctypes_vals = []
        keep = []
        for pname, ty in sig.items():
            if ty == "constexpr":
                continue
            if ty.startswith("*"):
                buf = ctypes.c_uint64(int(by_name[pname].data_ptr()))
            elif ty == "i64":
                buf = ctypes.c_uint64(int(kwargs[pname]))
            elif ty == "i32":
                buf = ctypes.c_int32(int(kwargs[pname]))
            elif ty == "fp32":
                buf = ctypes.c_float(float(kwargs[pname]))
            else:
                raise ValueError(f"unhandled ABI type {ty} for {pname}")
            keep.append(buf)
            ctypes_vals.append(ctypes.addressof(buf))
        gs = ctypes.c_uint64(0)
        ps = ctypes.c_uint64(0)
        ctypes_vals += [ctypes.addressof(gs), ctypes.addressof(ps)]
        arr = (ctypes.c_void_p * len(ctypes_vals))(*ctypes_vals)
        GraphKernelService._ABI_KEEP.append((keep, gs, ps, arr))
        return ctypes.addressof(arr)

    _ABI_KEEP: list = []

    def has_variant(self, jit_name: str, variant: str) -> bool:
        v = FAMILY_ALIAS.get(variant, variant)
        with _LOCK:
            return (jit_name, v) in self._variants

    # -- graph binding ---------------------------------------------------
    def bind_graph(self, graph, jit_name: str) -> int:
        """Locate this jit's kernel nodes in a torch-captured graph
        (keep_graph=True required) and snapshot their original params."""
        cu = self._driver()
        gid = id(graph)
        with _LOCK:
            if gid in self._bindings:
                return len(self._bindings[gid])
        from cuda.bindings import runtime as rt
        graph_h = graph.raw_cuda_graph()
        err, _, num = rt.cudaGraphGetNodes(graph_h, numNodes=0)
        if int(err) != 0:
            raise RuntimeError(f"cudaGraphGetNodes(query) {err}")
        err, nodes, num = rt.cudaGraphGetNodes(graph_h, numNodes=int(num))
        if int(err) != 0:
            raise RuntimeError(f"cudaGraphGetNodes {err}")
        found: List[_NodeBinding] = []
        for i in range(int(num)):
            node = nodes[i]
            err, ntype = rt.cudaGraphNodeGetType(node)
            if ntype != rt.cudaGraphNodeType.cudaGraphNodeTypeKernel:
                continue
            cu_node = cu.CUgraphNode(init_value=int(node))
            err, params = cu.cuGraphKernelNodeGetParams(cu_node)
            if err != cu.CUresult.CUDA_SUCCESS:
                continue
            fhandle = int(params.func)
            if not fhandle:
                continue
            err2, fname = cu.cuFuncGetName(cu.CUfunction(init_value=fhandle))
            if err2 != cu.CUresult.CUDA_SUCCESS or not fname:
                continue
            if jit_name not in fname.decode(errors="ignore"):
                continue
            found.append(_NodeBinding(
                node=cu_node,
                func_before=fhandle,
                warps_before=int(params.blockDimX) // 32,
                shared_before=int(params.sharedMemBytes),
                kernel_params=params.kernelParams))
        with _LOCK:
            self._bindings[gid] = found
            self._keep[gid] = graph          # handles must outlive callers
            # the exec handle is fetched lazily (keep_graph=True graphs
            # only instantiate inside their first replay)
            self._execs.pop(gid, None)
            self._jits[gid] = jit_name
            # captured nodes are BORN at __base (capture_launch); every
            # switch after that is intra-module
            self._graph_active.setdefault(gid, "__base")
            self.state["binds"] += 1
            self.state["nodes"] += len(found)
        self._publish()
        return len(found)

    def _exec_of(self, gid: int) -> Any:
        """The CUgraphExec for a bound graph, instantiated on demand."""
        cu = self._driver()
        with _LOCK:
            h = self._execs.get(gid)
        if h is not None:
            return cu.CUgraphExec(init_value=h)
        graph = self._keep[gid]
        h = int(graph.raw_cuda_graph_exec())
        with _LOCK:
            self._execs[gid] = h
        return cu.CUgraphExec(init_value=h)

    # -- retarget ---------------------------------------------------------
    def submit_retarget(self, jit_name: str, variant: str) -> bool:
        """Queue a retarget (us-scale).  Consumed at the next replay
        boundary — never inside the decode step's operator stream."""
        variant = FAMILY_ALIAS.get(variant, variant)
        if not self.has_variant(jit_name, variant):
            return False
        with _LOCK:
            self._pending[jit_name] = variant
        return True

    def _consume_pending(self) -> None:
        with _LOCK:
            pending = dict(self._pending)
            self._pending.clear()
        for jit_name, variant in pending.items():
            self._apply(jit_name, variant)

    def _apply(self, jit_name: str, variant: str) -> bool:
        cu = self._driver()
        with _LOCK:
            v = self._variants.get((jit_name, variant))
        if v is None:
            self._err(f"apply({variant})", RuntimeError("variant not loaded"))
            return False
        t0 = time.monotonic()
        applied = 0
        with _LOCK:
            items = [(gid, nodes) for gid, nodes in self._bindings.items()
                     if self._jits.get(gid) == jit_name and
                     self._graph_active.get(gid) != variant]
        for gid, nodes in items:
            gexec = self._exec_of(gid)
            for nb in nodes:
                params = cu.CUDA_KERNEL_NODE_PARAMS()
                params.func = v["fn"]
                params.gridDimX, params.gridDimY, params.gridDimZ = 0, 0, 0
                params.blockDimX = 32 * v["warps"]
                params.blockDimY = 1
                params.blockDimZ = 1
                params.sharedMemBytes = v["shared"]
                params.kernelParams = nb.kernel_params
                params.extra = 0
                if os.environ.get("PACT_GRAPH_SELFCHECK") == "1":
                    # diagnostic: full SetParams round-trip but pointing at
                    # the ORIGINAL function — isolates the action from the
                    # module switch
                    params.func = cu.CUfunction(init_value=nb.func_before)
                    params.blockDimX = 32 * nb.warps_before
                    params.sharedMemBytes = nb.shared_before
                # grid dims must be preserved from the node's own snapshot
                err, cur = cu.cuGraphKernelNodeGetParams(nb.node)
                if err == cu.CUresult.CUDA_SUCCESS:
                    params.gridDimX = int(cur.gridDimX)
                    params.gridDimY = int(cur.gridDimY)
                    params.gridDimZ = int(cur.gridDimZ)
                    params.blockDimY = int(cur.blockDimY)
                    params.blockDimZ = int(cur.blockDimZ)
                if os.environ.get("PACT_GRAPH_DEBUG") == "1":
                    with _LOCK:
                        self.state.setdefault("verify", []).append({
                            "node": int(nb.node),
                            "orig": [int(cur.gridDimX), int(cur.gridDimY),
                                     int(cur.gridDimZ),
                                     int(cur.blockDimX),
                                     int(cur.sharedMemBytes)],
                            "variant": [32 * v["warps"], v["shared"]],
                            "kp_same": True,
                        })
                (rc,) = cu.cuGraphExecKernelNodeSetParams(
                    gexec, nb.node, params)
                if rc != 0:
                    self._err(f"SetParams({variant})",
                              RuntimeError(f"rc={rc}"))
                else:
                    applied += 1
            if applied:
                with _LOCK:
                    self._graph_active[gid] = variant
        with _LOCK:
            if applied:
                self.state["retargets"] += 1
                self.state["active"] = f"{jit_name}::{variant}"
                self._active[jit_name] = variant
                self.state["last_recode_ms"] = round(
                    (time.monotonic() - t0) * 1000.0, 4)
            elif not items:
                # idempotent re-submit: every bound graph already runs it
                self._active[jit_name] = variant
            else:
                self._pending[jit_name] = variant  # re-try next boundary
        self._publish()
        return applied > 0

    def rollback(self, jit_name: str) -> bool:
        """Point every node back at the module's __base function (the
        PTX-identical capture baseline) — an intra-module switch too."""
        cu = self._driver()
        with _LOCK:
            v = self._variants.get((jit_name, "__base"))
        if v is None:
            return False
        with _LOCK:
            items = [(gid, nodes) for gid, nodes in self._bindings.items()
                     if self._jits.get(gid) == jit_name and
                     self._graph_active.get(gid) != "__base"]
        applied = 0
        for gid, nodes in items:
            gexec = self._exec_of(gid)
            for nb in nodes:
                params = cu.CUDA_KERNEL_NODE_PARAMS()
                params.func = v["fn"]
                params.gridDimX, params.gridDimY, params.gridDimZ = 0, 0, 0
                params.blockDimX = 32 * v["warps"]
                params.blockDimY = 1
                params.blockDimZ = 1
                params.sharedMemBytes = v["shared"]
                params.kernelParams = nb.kernel_params
                params.extra = 0
                err, cur = cu.cuGraphKernelNodeGetParams(nb.node)
                if err == cu.CUresult.CUDA_SUCCESS:
                    params.gridDimX = int(cur.gridDimX)
                    params.gridDimY = int(cur.gridDimY)
                    params.gridDimZ = int(cur.gridDimZ)
                    params.blockDimY = int(cur.blockDimY)
                    params.blockDimZ = int(cur.blockDimZ)
                (rc,) = cu.cuGraphExecKernelNodeSetParams(
                    gexec, nb.node, params)
                if rc == 0:
                    applied += 1
            if applied:
                with _LOCK:
                    self._graph_active.pop(gid, None)
        with _LOCK:
            if applied:
                self.state["rollbacks"] += 1
                self.state["active"] = None
                self._active.pop(jit_name, None)
        self._publish()
        return applied > 0

    def active(self, jit_name: str) -> Optional[str]:
        with _LOCK:
            return self._active.get(jit_name)

    # -- replay boundary ---------------------------------------------------
    def after_replay(self, graph) -> None:
        """Hook body, called AFTER a replay completed enqueueing: bind
        unseen graphs (the exec is instantiated by now — keep_graph=True
        graphs instantiate lazily inside the first replay), fire the
        bridge boundary callback, then consume pending retargets.  Us-scale
        on the install path; a retarget applied here takes effect from the
        NEXT replay, which is exactly the async-frame contract."""
        try:
            gid = id(graph)
            if gid in self._skip:
                return
            if gid not in self._bindings:
                if not self._target_jit:
                    return
                try:
                    n = self.bind_graph(graph, self._target_jit)
                    if os.environ.get("PACT_GRAPH_DEBUG") == "1":
                        print(f"[pact graph_service] bind gid={id(graph):#x} "
                              f"jit={self._target_jit} nodes={n}", flush=True)
                    if n == 0:
                        self._skip.add(gid)   # no pact node: not our graph
                        with _LOCK:
                            self.state["skipped_graphs"] += 1
                    else:
                        with _LOCK:
                            self.state["bound_graphs"] += 1
                except Exception as e:  # noqa: BLE001
                    # keep_graph=False graphs raise in raw_cuda_graph()
                    self._skip.add(gid)
                    if os.environ.get("PACT_GRAPH_DEBUG") == "1":
                        print(f"[pact graph_service] bind gid={id(graph):#x} "
                              f"FAILED {e}", flush=True)
                    self._err("bind(auto)", e)
                    return
            if self._boundary is not None:
                try:
                    self._boundary()
                except Exception as e:  # noqa: BLE001
                    self._err("boundary", e)
            self._consume_pending()
        except Exception as e:  # noqa: BLE001 - hook must never kill replay
            self._err("after_replay", e)


def install_replay_hook() -> bool:
    """Wrap torch.cuda.CUDAGraph.replay so every graph replay passes the
    service's boundary check.  Idempotent; installed only by the harness
    when PACT_GRAPH_SERVICE=1."""
    global _REPLAY_HOOKED
    if _REPLAY_HOOKED:
        return True
    try:
        import torch

        orig = torch.cuda.CUDAGraph.replay
        svc = get_service()

        def _replay(self, *a, **kw):
            r = orig(self, *a, **kw)  # replay first: instantiates the exec
            try:
                with _LOCK:
                    svc.state["replay_calls"] += 1
                    n = svc.state["replay_calls"]
                if os.environ.get("PACT_GRAPH_DEBUG") == "1" and \
                        (n <= 3 or n % 25 == 0):
                    print(f"[pact graph_service] replay#{n} on "
                          f"{type(self).__name__} id={id(self):#x}",
                          flush=True)
            except Exception:
                pass
            svc.after_replay(self)   # retargets land from the NEXT replay
            return r

        torch.cuda.CUDAGraph.replay = _replay
        _REPLAY_HOOKED = True
        return True
    except Exception:
        return False
