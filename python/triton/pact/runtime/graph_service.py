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
    # V16-T3: pin 1 was silently rejected by P6's [2,8] domain (fake
    # config since day one); 2 is the honest "shallowest pipeline"
    "short": {"PACT_ENABLE": "1", "PACT_OVERRIDE_STAGES": "2"},
    "w1": {"PACT_ENABLE": "1", "PACT_OVERRIDE_WARPS": "1"},
    # V14-B (S1): unlock the deep-pipeline range; P6's L2-residency gate
    # itself keeps hot shapes at the theory decision, so this module entry
    # is the graph-side spelling of the "cold" auto family.
    "cold": {"PACT_ENABLE": "1", "PACT_MAX_PIPELINE_STAGES": "5"},
}
# V16-T4 lazy prepare: the capture-borne vocabulary is just the baseline
# pair + the theory default; every other family is compiled ON DEMAND as
# a SEPARATE single-entry cubin and swapped via the E3 path (source-node
# SetParams + whole-graph cudaGraphExecUpdate -- validated on the chain
# graph, 60us).  PACT_GRAPH_VOCAB=full restores the 9-entry eager module.
# V17 S2-4 B0: theory is PTX-byte-identical to __base (V16 ba6e266), so
# the mini vocab collapses 3->1 and "theory" becomes an ALIAS -- decider
# submits keep working, the graph simply never needs a second entry.
MINI_VOCAB = ("__base",)
FAMILY_ALIAS: Dict[str, str] = {
    "baseline": "__vanilla", "vanilla": "__vanilla",
    "theory": "__base",
    "occupancy": "occ", "latency": "lat",
}
# V19 N2 (PACT_PARAM_INDIRECTION=1, default off): the parameter-block
# layout mirrored from the kernel side (authoritative definition in
# suite/kernels/decode.py:IND_PARAM_BLOCK_LAYOUT).  The indirect ABI
# replaces ALL ten runtime params (6 pointers + 4 strides) with one
# 8B slot; variants of the indirect jit form a closed same-ABI family.
_IND_BLOCK_LAYOUT = (
    "out", "q", "k_cache", "v_cache", "block_table", "seq_lens",
    "STRIDE_BLOCK", "STRIDE_KV_HEAD", "STRIDE_PAGE", "STRIDE_HEAD_DIM",
)


def _vocab_mode() -> str:
    return "full" if os.environ.get("PACT_GRAPH_VOCAB") == "full" else "mini"


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
        self._jit_fns: Dict[str, Any] = {}      # jit -> jit_fn (offer path)
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
        # V17 S3-2 ②: consecutive-apply backoff + give-up blacklist
        self._fail_streak: Dict[Tuple[str, str], int] = {}
        self._blacklist: set = set()     # (jit, variant) abandoned
        # V17 S3-3 ①: func handles this service itself loaded (module
        # membership test for bind/apply -- the substring name match let
        # triton-loaded same-name handles through, the capture-fallback
        # danger shape)
        self._known_funcs: Dict[int, str] = {}
        # V19 N2: jit name -> {"slot": 8B device int64 tensor,
        # "block": int64[10] param block} — service-held for the whole
        # process life (NEVER in _ABI_KEEP: the tail-64 trim would drop
        # the very pointers captured graphs dereference)
        self._indirect: Dict[str, dict] = {}
        # V17 S3-2 D1: deterministic fault injection (PACT_FAULT_INJECT=1)
        self._fi_on = os.environ.get("PACT_FAULT_INJECT") == "1"
        self._fi_kinds = set(
            (os.environ.get("PACT_FAULT_INJECT_KINDS")
             or "setparams,execupdate").split(","))
        try:
            self._fi_every = max(
                int(os.environ.get("PACT_FAULT_INJECT_EVERY") or 7), 1)
        except ValueError:
            self._fi_every = 7
        self._fi_counts: Dict[str, int] = {}
        self.state: Dict[str, Any] = {
            "offers": 0, "loads": 0, "binds": 0, "nodes": 0,
            "retargets": 0, "rollbacks": 0, "last_recode_ms": None,
            "last_offer_ms": None, "active": None, "errors": [],
            "replay_calls": 0, "bound_graphs": 0, "skipped_graphs": 0,
            "skipped_nodes": 0, "bind_rejected": 0,
            "exec_update_failures": 0, "auto_rollbacks": 0,
            "canaries": 0, "submit_rejects": 0, "blacklisted": [],
            # V19 N4: content-hash variant catalogue (third evidence layer
            # against cross-module silent corruption, after handle
            # membership + structural audit).  "name::variant" ->
            # {cubin_sha256, ptx_sha256, kind}; pure bookkeeping, no
            # behaviour change on any path.
            "content_hash": {}, "hash_misses": 0,
            # V19 N2: indirect-ABI jit name once prepare_indirect ran
            "indirect": None,
        }

    def _fi(self, kind: str) -> bool:
        """V17 S3-2 D1 hook: deterministic fault injection.  Every K-th
        call of the given kind (setparams / execupdate) pretends a driver
        failure (rc=1 / rc=700) so the failure branches are exercised on
        the REAL service; PACT_FAULT_INJECT=1 gates the whole thing (the
        default path never enters here beyond one bool check)."""
        if not self._fi_on or kind not in self._fi_kinds:
            return False
        n = self._fi_counts.get(kind, 0) + 1
        self._fi_counts[kind] = n
        return n % self._fi_every == 0

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
                variants: Optional[Dict[str, Dict[str, str]]] = None,
                clone_first_arg: bool = True) -> bool:
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
            base_vocab = dict(variants or VARIANT_ENVS)
            if variants is None and _vocab_mode() == "mini":
                base_vocab = {k: VARIANT_ENVS[k] for k in MINI_VOCAB
                              if k in VARIANT_ENVS}
            kerns: Dict[str, Any] = {}
            for vname, env in base_vocab.items():
                k, _ = compile_explicit(jit_fn, args, kwargs, dict(env))
                kerns[vname] = k
            blob = self._merge_ptxas(cu, kerns, name)
            err, mod = cu.cuModuleLoadDataEx(blob, 0, [], [])
            if err != cu.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f"cuModuleLoadDataEx {err}")
            with _LOCK:
                self._mods[name] = mod
                # V19 N4: register every merged entry's content hashes
                # (all entries share the ONE merged cubin; per-variant
                # identity comes from its own PTX).
                import hashlib as _hl
                _cub_sha = _hl.sha256(blob).hexdigest()[:16]
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
                    self._known_funcs[int(fn)] = f"{name}::{vname}"
                    self.state["content_hash"][f"{name}::{vname}"] = {
                        "cubin_sha256": _cub_sha,
                        "ptx_sha256": _hl.sha256(
                            k.asm["ptx"].encode()).hexdigest()[:16],
                        "kind": "merged",
                    }
                self._jit_args[name] = (tuple(args), dict(kwargs),
                                        tuple(grid))
                self._jit_fns[name] = jit_fn
            # settle the lazy loads NOW (one raw launch per function on
            # cloned outputs, off any serving path)
            for vname, k in kerns.items():
                self._materialize(cu, k, self._variants[(name, vname)]["fn"],
                                  args, kwargs, grid,
                                  clone_first=clone_first_arg)
            with _LOCK:
                self.state["prepared"] = name
                self.state["vocab_mode"] = _vocab_mode()
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

    # -- V19 N2: indirect-ABI arming + the 8B write channel --------------
    def prepare_indirect(self, ind_jit_fn, orig_args, kwargs, grid) -> bool:
        """Arm the vocabulary on the INDIRECT kernel (the *_ind twin whose
        only runtime parameter is the 8B slot).  The per-GRAPH slots and
        param blocks are built at capture_launch time (one per captured
        graph -- vLLM captures one FULL graph PER BATCH TIER and a single
        shared block would have the later tiers overwrite the earlier
        tiers' pointers; that exact token-flip was caught by the a0
        indirect arm).  Here we only record the layout contract and let
        prepare() run unchanged -- compile/merge/load/materialise/N4
        catalogue all take the plain path (clone_first_arg=False keeps a
        bootstrap slot through materialise)."""
        import torch
        if len(orig_args) < 6:
            self._err("prepare_indirect",
                      ValueError(f"want 6 direct args, got {len(orig_args)}"))
            return False
        name = str(getattr(ind_jit_fn, "__name__", None)
                   or ind_jit_fn.fn.__name__)
        # bootstrap slot (pinned host, UVA): only used to settle lazy
        # loads through materialise; every real graph gets its own below
        block = torch.zeros(len(_IND_BLOCK_LAYOUT), dtype=torch.int64,
                            pin_memory=True)
        for i, t in enumerate(orig_args[:6]):
            block[i] = int(t.data_ptr())
        for j, key in enumerate(_IND_BLOCK_LAYOUT[6:]):
            if key not in kwargs:
                self._err("prepare_indirect",
                          KeyError(f"missing kwarg {key}"))
                return False
            block[6 + j] = int(kwargs[key])
        # the four STRIDE_* now live IN THE BLOCK: they are not signature
        # parameters of the indirect twin, and the JIT binder rejects
        # unknown kwargs -- hand prepare the pruned dict
        ind_kwargs = {k: v for k, v in kwargs.items()
                      if k not in _IND_BLOCK_LAYOUT[6:]}
        slot = torch.zeros(1, dtype=torch.int64, pin_memory=True)
        slot[0] = block.data_ptr()
        ok = self.prepare(ind_jit_fn, (slot,), ind_kwargs, grid,
                          clone_first_arg=False)
        if ok:
            with _LOCK:
                # "slots" grows one entry per captured graph (see
                # capture_launch); lifetime = service = process
                self._indirect[name] = {"slots": [slot],
                                        "blocks": [block]}
                self.state["indirect"] = name
                self._publish()
        return ok

    def _indirect_new_pair(self, args, kwargs):
        """Build ONE fresh (slot, block) pair in pinned host memory (UVA
        -- the device dereferences them over PCIe) from a DIRECT-ABI
        arg pack.  Host stores only: capture-legal, never recorded into
        the graph, refreshable at any later time."""
        import torch
        block = torch.zeros(len(_IND_BLOCK_LAYOUT), dtype=torch.int64,
                            pin_memory=True)
        for i, t in enumerate(args[:6]):
            block[i] = int(t.data_ptr())
        for j, key in enumerate(_IND_BLOCK_LAYOUT[6:]):
            block[6 + j] = int(kwargs[key])
        slot = torch.zeros(1, dtype=torch.int64, pin_memory=True)
        slot[0] = block.data_ptr()
        return slot, block

    def indirect_slot(self, jit_name: Optional[str] = None):
        """V19 N2: the newest 8B slot of the armed indirect jit (or the
        named one) -- capture-side/probe convenience."""
        with _LOCK:
            name = jit_name or self.state.get("indirect")
            ind = self._indirect.get(name) if name else None
            return ((ind or {}).get("slots") or [None])[-1]

    def write_param_slot(self, jit_name, block=None, values=None) -> bool:
        """V19 N2 8B-write channel (the PyGraph move): refresh block
        entries and/or repoint slots, for EVERY live pair (all captured
        batch tiers).  Slot and blocks are PINNED HOST memory -- every
        write is a plain host store, legal inside a capture window and
        effective for every subsequent launch/replay."""
        with _LOCK:
            ind = self._indirect.get(jit_name)
        if ind is None:
            return False
        pairs = list(zip(ind.get("slots") or [],
                         ind.get("blocks") or []))
        if values:
            for key, v in (values or {}).items():
                idx = _IND_BLOCK_LAYOUT.index(key)
                for _s, blk in pairs:
                    blk[idx] = int(v)
        if block is not None:
            for s, _blk in pairs:
                s[0] = int(block.data_ptr())
        return True

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
        # V16-T4 R1 (Foundry cubin persistence, scoped to the ptxas step):
        # the merged PTX hash + sm arch identifies the cubin; a pool hit
        # skips the ptxas subprocess entirely (compile_explicit keeps its
        # own triton cache, so a restart pays load+materialize only)
        import hashlib
        pkey = hashlib.sha256(merged.encode()).hexdigest()[:16]
        pool_dir = Path(os.environ.get("PACT_GRAPH_POOL_DIR")
                        or (Path.home() / ".triton" / "pact_graph_pool"))
        cached = pool_dir / f"{name}_{pkey}_sm{maj}{mnr}.cubin"
        if cached.exists():
            try:
                return cached.read_bytes()
            except Exception:
                pass
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
            blob = cub.read_bytes()
        try:
            pool_dir.mkdir(parents=True, exist_ok=True)
            (pool_dir / cached.name).write_bytes(blob)
        except Exception:
            pass  # pool is an accelerator, never a gate
        return blob

    def _materialize(self, cu, kernel, fn, args, kwargs, grid,
                     clone_first: bool = True) -> None:
        """One raw launch on cloned tensors to settle the lazy load."""
        if not args or not kwargs or not grid:
            return
        from triton.pact.runtime.capture_guard import (
            wait_out_of_capture, note_race)
        if not wait_out_of_capture():
            note_race("materialize")
        import torch
        # V19 N2: an INDIRECT-ABI kernel's args[0] is the 8B slot itself
        # -- empty_like would forge a garbage slot (wild deref).  The
        # slot/block are service-held, cloning is both wrong and unneeded
        first = torch.empty_like(args[0]) if clone_first else args[0]
        values = [first] + list(args[1:])
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
        (plain cuLaunchKernel); falls back to False when unprepared.

        V19 N2 (indirect mode): when the armed jit is the indirect twin,
        the caller passes the DIRECT six-tensor args; a FRESH pinned-host
        (slot, block) pair is built HERE per captured graph -- vLLM
        captures one FULL graph per batch tier and a shared block would
        let later tiers overwrite earlier tiers' pointers (a0 indirect
        arm, token-flip evidence).  The launched kernel's only runtime
        parameter is the slot, so each tier's node dereferences its own
        block forever after."""
        with _LOCK:
            name = self.state.get("prepared")
            if not name:
                return False
            v = self._variants.get((name, "__base"))
            kern = self._kerns.get((name, "__base"))
            indirect = (self.state.get("indirect") == name
                        and name in self._indirect)
        if v is None:
            return False
        if indirect:
            slot, block = self._indirect_new_pair(args, kwargs)
            with _LOCK:
                self._indirect[name]["slots"].append(slot)
                self._indirect[name]["blocks"].append(block)
            args = (slot,)
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
        # V17 S3-2 ④: bounded keep-alive (was an unbounded leak: every
        # prepare/offer/materialize/canary call pinned its staging arrays
        # forever).  Launches these backed are issued synchronously right
        # after the pack, so retaining a generous tail is safe.
        if len(GraphKernelService._ABI_KEEP) > 64:
            del GraphKernelService._ABI_KEEP[:-64]
        return ctypes.addressof(arr)

    _ABI_KEEP: list = []

    def has_variant(self, jit_name: str, variant: str) -> bool:
        v = FAMILY_ALIAS.get(variant, variant)
        with _LOCK:
            return (jit_name, v) in self._variants

    def offer(self, jit_name: str, variant: str,
              env: Optional[Dict[str, str]] = None) -> bool:
        """V16-T4: compile + load ONE more variant ON DEMAND as its own
        single-entry cubin in a SEPARATE module ("cross" variant).  Runs
        on background frames (ms..s: compile_explicit + pool ptxas +
        cuModuleLoadDataEx + materialize); never at a replay boundary.
        Swapping to a cross variant goes through the E3 path (source-node
        SetParams + whole-graph cudaGraphExecUpdate), so the module the
        captured nodes were born in never has to grow.
        V17 S2-4 B1: an explicit `env` (a normalize_preset extra_env dict)
        passes straight through -- VARIANT_ENVS is a compatibility layer,
        not a gate."""
        variant = FAMILY_ALIAS.get(variant, variant)
        with _LOCK:
            if (jit_name, variant) in self._variants:
                return True
            jit_fn = self._jit_fns.get(jit_name)
            args_kw = self._jit_args.get(jit_name)
            env = dict(env) if env is not None else VARIANT_ENVS.get(variant)
        if jit_fn is None or args_kw is None or env is None:
            return False
        args, kwargs, grid = args_kw
        t0 = time.monotonic()
        try:
            from triton.pact.runtime.capture_guard import (
                wait_out_of_capture, note_race)
            if not wait_out_of_capture():
                note_race(f"offer({variant})")
            cu = self._driver()
            # a caller on a fresh background thread has NO current CUDA
            # context -- driver-API cuModuleLoadDataEx below would fail
            # with CUDA_ERROR_INVALID_CONTEXT (201); bind the primary
            # context with one runtime-API op first (V17 坑 29, seen in
            # both the A0 EngineCore offer thread and the journey demos'
            # async cycles)
            import torch
            torch.zeros(1, device="cuda")
            from triton.pact.compiler.explicit_compiler import compile_explicit
            k, _ = compile_explicit(jit_fn, args, kwargs, dict(env))
            blob = self._merge_ptxas(cu, {variant: k}, jit_name)
            err, mod = cu.cuModuleLoadDataEx(blob, 0, [], [])
            if err != cu.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f"cuModuleLoadDataEx(offer) {err}")
            err, fn = cu.cuModuleGetFunction(
                mod, f"{jit_name}__{variant}".encode())
            if err != cu.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f"GetFunction(offer {variant}) {err}")
            # V19 N2: an indirect jit's args[0] IS the slot -- the
            # materialise clone would forge a garbage slot (wild deref,
            # async illegal access surfacing at the next replay)
            with _LOCK:
                _cf = jit_name not in self._indirect
            self._materialize(cu, k, fn, args, kwargs, grid,
                              clone_first=_cf)
            with _LOCK:
                self._variants[(jit_name, variant)] = {
                    "fn": fn,
                    "warps": int(k.metadata.num_warps),
                    "shared": int(k.metadata.shared),
                    "cross": mod,      # non-None => separate module
                }
                self._kerns[(jit_name, variant)] = k
                self._known_funcs[int(fn)] = f"{jit_name}::{variant}"
                # V19 N4: cross variants get their own single-entry cubin
                # hash (the catalogue must hit EVERY loaded variant)
                import hashlib as _hl
                self.state["content_hash"][f"{jit_name}::{variant}"] = {
                    "cubin_sha256": _hl.sha256(blob).hexdigest()[:16],
                    "ptx_sha256": _hl.sha256(
                        k.asm["ptx"].encode()).hexdigest()[:16],
                    "kind": "cross",
                }
                self.state["offers"] += 1
                self.state["last_offer_ms"] = round(
                    (time.monotonic() - t0) * 1e3, 1)
                self.state.setdefault("offered", []).append(variant)
            self._publish()
            return True
        except Exception as e:  # noqa: BLE001
            self._err(f"offer({variant})", e)
            return False

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
        with _LOCK:
            known = set(self._known_funcs)
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
            # V17 S3-3 ①: a name match is NOT enough -- accept only
            # handles this service itself loaded (module membership).  A
            # name-matching function from triton's own loader (the
            # capture-fallback shape, dynamic_bridge capture_launch
            # returned False) is rejected and counted: retargeting it is
            # the cross-module silent-corruption shape of v13.
            if fhandle not in known:
                with _LOCK:
                    self.state["bind_rejected"] = \
                        int(self.state["bind_rejected"] or 0) + 1
                continue
            # V19 N4: membership passed -- the third layer cross-checks the
            # handle's identity against the content-hash catalogue.  A miss
            # means load bookkeeping broke (state corruption shape); it is
            # COUNTERED + reported, never allowed to change bind behaviour
            # (pure-checkin contract: observable, no semantics change).
            ident = self._known_funcs.get(fhandle)
            if ident is not None and \
                    ident not in self.state.get("content_hash", {}):
                with _LOCK:
                    self.state["hash_misses"] = \
                        int(self.state.get("hash_misses") or 0) + 1
                self._err("content_hash(bind)",
                          RuntimeError(f"uncatalogued handle {ident}"))
            found.append(_NodeBinding(
                node=cu_node,
                func_before=fhandle,
                warps_before=int(params.blockDimX) // 32,
                shared_before=int(params.sharedMemBytes),
                kernel_params=params.kernelParams))
        with _LOCK:
            if not found:
                # V17 S3-2 fix: a graph with ZERO pact nodes (most vLLM
                # capture-size tiers) must not sit in _bindings as an
                # empty list -- the ok_g/all-or-nothing logic would count
                # it as a failed switch on every apply and 3-strike
                # blacklist the variant (seen live: theory AND deep
                # blacklisted with e3=2, 0/20 A0 rerun 20260928_083121)
                return 0
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
            # V18 G-2/BR-19: an out-of-vocab submit used to return
            # silently -- count it on the same channel as the blacklist
            # reject (behaviour unchanged, still False)
            with _LOCK:
                self.state["submit_rejects"] = \
                    int(self.state["submit_rejects"] or 0) + 1
            return False
        if (jit_name, variant) in self._blacklist:
            # V17 S3-2 ②: abandoned after 3 consecutive failed applies
            with _LOCK:
                self.state["submit_rejects"] = \
                    int(self.state["submit_rejects"] or 0) + 1
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
            known = set(self._known_funcs)
        if v is None:
            self._err(f"apply({variant})", RuntimeError("variant not loaded"))
            return False
        t0 = time.monotonic()
        use_e3 = bool(v.get("cross"))
        if use_e3:
            from cuda.bindings import runtime as _rt
        with _LOCK:
            items = [(gid, nodes) for gid, nodes in self._bindings.items()
                     if self._jits.get(gid) == jit_name and
                     self._graph_active.get(gid) != variant]
        total_applied = 0
        graphs_ok = True
        for gid, nodes in items:
            if not nodes:
                # empty binding (legacy/no-pact capture tier): a NO-OP,
                # never a failed switch
                continue
            gexec = self._exec_of(gid)
            graph_raw = self._keep[gid].raw_cuda_graph() if use_e3 else None
            applied_g = 0
            eligible = 0
            touched = []
            for nb in nodes:
                # V17 S3-3 ②: only nodes BORN from this service's module
                # (bind-time membership snapshot); anything else is a
                # different specialization loaded elsewhere -- skip+count
                if int(nb.func_before) not in known:
                    with _LOCK:
                        self.state["skipped_nodes"] = \
                            int(self.state.get("skipped_nodes") or 0) + 1
                    continue
                eligible += 1
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
                if use_e3:
                    # V16-T4 E3 path: mutate the SOURCE-graph node (func
                    # from a separate module), then one whole-graph
                    # cudaGraphExecUpdate adopts it into torch's own exec
                    rc = 1 if self._fi("setparams") else \
                        cu.cuGraphKernelNodeSetParams(nb.node, params)[0]
                else:
                    rc = 1 if self._fi("setparams") else \
                        cu.cuGraphExecKernelNodeSetParams(
                            gexec, nb.node, params)[0]
                if rc != 0:
                    self._err(f"SetParams({variant})",
                              RuntimeError(f"rc={rc}"))
                else:
                    applied_g += 1
                    touched.append(nb)
            # a graph counts as switched only when EVERY ELIGIBLE node was
            # rewritten (partial application = exec/state divergence;
            # born-outside nodes were skipped above and don't count)
            ok_g = eligible > 0 and applied_g == eligible
            # the CURRENT variant (what the exec is still running if this
            # apply fails halfway) -- needed by every restore path below
            with _LOCK:
                cur_v = self._graph_active.get(gid)
                v_cur = self._variants.get((jit_name, cur_v)) \
                    if cur_v else None
            if v_cur is None:
                v_cur = self._variants.get((jit_name, "__base"))
            if use_e3:
                if ok_g:
                    rc_u = 700 if self._fi("execupdate") else \
                        _rt.cudaGraphExecUpdate(
                            int(gexec), int(graph_raw))[0]
                    with _LOCK:
                        self.state["e3_updates"] = \
                            int(self.state.get("e3_updates") or 0) + 1
                    if rc_u != 0:
                        self._err(f"ExecUpdate({variant})",
                                  RuntimeError(f"rc={rc_u}"))
                        with _LOCK:
                            self.state["exec_update_failures"] = \
                                int(self.state.get("exec_update_failures")
                                    or 0) + 1
                        # V17 S3-2 ①: the exec still runs the CURRENT
                        # function while the source nodes now point at
                        # the variant -- rewrite the source back to the
                        # CURRENT variant (not the bind snapshot: the
                        # graph may have been on an in-module family)
                        # and do NOT set _graph_active
                        self._restore_nodes(cu, touched, v_cur,
                                            source=True)
                        ok_g = False
                elif applied_g:
                    # partial SetParams: the source is half-switched --
                    # restore the touched nodes to the CURRENT variant
                    self._restore_nodes(cu, touched, v_cur, source=True)
            elif not ok_g and applied_g:
                # intra-module partial: point the touched EXEC nodes back
                # at the CURRENT variant
                self._restore_nodes(cu, touched, v_cur, gexec=gexec,
                                    source=False)
            if ok_g:
                with _LOCK:
                    self._graph_active[gid] = variant
            else:
                graphs_ok = False
            total_applied += applied_g
        with _LOCK:
            if graphs_ok and total_applied:
                self.state["retargets"] += 1
                self.state["active"] = f"{jit_name}::{variant}"
                self._active[jit_name] = variant
                self.state["last_recode_ms"] = round(
                    (time.monotonic() - t0) * 1000.0, 4)
                self._fail_streak.pop((jit_name, variant), None)
            elif not items:
                # idempotent re-submit: every bound graph already runs it
                self._active[jit_name] = variant
            else:
                # V17 S3-2 ②: backoff + give-up — 3 consecutive failed
                # applies blacklist the variant instead of retrying forever
                key = (jit_name, variant)
                streak = self._fail_streak.get(key, 0) + 1
                self._fail_streak[key] = streak
                if streak >= 3:
                    self._blacklist.add(key)
                    self.state["blacklisted"] = sorted(
                        f"{j}::{w}" for j, w in self._blacklist)
                    self.state["audit_failures"] = (
                        self.state.get("audit_failures") or [])[-7:] + [
                        {"variant": variant, "n": streak,
                         "note": "give-up: 3 consecutive failed applies"}]
                    self._pending.pop(jit_name, None)
                else:
                    self._pending[jit_name] = variant  # re-try next boundary
        if graphs_ok and total_applied:
            # V16-T5 L1: post-swap structural audit (source-graph mirror
            # read-back; E3 path only, intra-module is exec-opaque)
            audit = None
            try:
                from triton.pact.runtime.graph_swap_validator import \
                    audit_after_apply
                audit = audit_after_apply(self, jit_name, variant)
                with _LOCK:
                    self.state["last_audit"] = {
                        "variant": variant, "ok": audit.get("ok"),
                        "nodes": audit.get("audited_nodes"),
                        "note": audit.get("note")}
            except Exception as e:  # noqa: BLE001
                self._err("audit", e)
            if audit is not None and audit.get("ok") is not True:
                # V17 S3-4 ④: audit failure -> automatic rollback to the
                # PTX-identical capture baseline + blacklist the variant
                self.rollback(jit_name)
                with _LOCK:
                    self.state["auto_rollbacks"] = \
                        int(self.state.get("auto_rollbacks") or 0) + 1
                    self._blacklist.add((jit_name, variant))
                    self.state["blacklisted"] = sorted(
                        f"{j}::{w}" for j, w in self._blacklist)
                    self.state["audit_failures"] = (
                        self.state.get("audit_failures") or [])[-7:] + [
                        {"variant": variant,
                         "note": "auto-rollback on audit failure"}]
            # V17 S3-4 ②: per-swap flip-anchor canary (opt-in,
            # PACT_SWAP_CANARY=1; side stream, budget-bound)
            if os.environ.get("PACT_SWAP_CANARY") == "1":
                can = self._canary(jit_name, variant)
                with _LOCK:
                    self.state["last_canary"] = can
                    self.state["canaries"] = \
                        int(self.state.get("canaries") or 0) + 1
                if can.get("anomaly"):
                    with _LOCK:
                        self.state["audit_failures"] = (
                            self.state.get("audit_failures") or [])[-7:] + [
                            {"variant": variant,
                             "note": f"canary anomaly: "
                                     f"{str(can)[:120]}"}]
                        self._blacklist.add((jit_name, variant))
                        self.state["blacklisted"] = sorted(
                            f"{j}::{w}" for j, w in self._blacklist)
                    self.rollback(jit_name)
        self._publish()
        return graphs_ok and total_applied > 0

    def _restore_nodes(self, cu, touched, v, gexec=None,
                       source: bool = True) -> None:
        """V17 S3-2 ①: point the touched nodes back at the CURRENT
        variant `v` so the graph mirrors the (unchanged) exec again --
        source-node SetParams for the E3 form, exec SetParams for the
        intra-module form.  Best-effort: failures land in state errors,
        never in the replay path."""
        if v is None:
            return
        for nb in touched:
            try:
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
                if source:
                    (rc,) = cu.cuGraphKernelNodeSetParams(nb.node, params)
                else:
                    (rc,) = cu.cuGraphExecKernelNodeSetParams(
                        gexec, nb.node, params)
                if rc != 0:
                    self._err("restore_nodes",
                              RuntimeError(f"rc={rc}"))
            except Exception as e:  # noqa: BLE001
                self._err("restore_nodes", e)

    def _canary(self, jit_name: str, variant: str) -> Dict[str, Any]:
        """V17 S3-4 ②: per-swap flip-anchor canary.  Launches the base
        and the variant once each on a SIDE stream with the prepare-time
        args (fresh output buffers) and records output-diff stats plus
        anomaly flags (NaN/Inf/all-zero/launch-rc).  Numerical difference
        between honest variant configs is EXPECTED and never gated on;
        the anomaly flags are the corruption signal (the v13 silent-bad
        shape).  Budget: two dummy launches, us-scale each, one
        side-stream sync — the decode stream never gains a sync point."""
        import torch
        args, kwargs, grid = self._jit_args.get(
            jit_name, (None, None, None))
        vb = self._variants.get((jit_name, "__base"))
        vv = self._variants.get((jit_name, variant))
        kb = self._kerns.get((jit_name, "__base"))
        kv = self._kerns.get((jit_name, variant))
        if not args or not kwargs or not grid or vb is None or vv is None \
                or kb is None or kv is None:
            return {"skipped": "missing args/variants"}
        try:
            out_b = torch.empty_like(args[0])
            out_v = torch.empty_like(args[0])
            stream = torch.cuda.Stream()
            rcs = []
            with torch.cuda.stream(stream):
                for out, vmeta, kern in ((out_b, vb, kb), (out_v, vv, kv)):
                    vals = (out,) + tuple(args[1:])
                    na = self._abi_args(kern, vals, kwargs)
                    (rc,) = self._cu.cuLaunchKernel(
                        vmeta["fn"], int(grid[0]), int(grid[1]),
                        int(grid[2]), 32 * int(vmeta["warps"]), 1, 1,
                        int(vmeta["shared"]), stream.cuda_stream, na, 0)
                    rcs.append(int(rc))
            stream.synchronize()
            d = (out_b.float() - out_v.float()).abs()
            res = {
                "variant": variant,
                "rc": rcs,
                "max_abs_diff": round(float(d.max().item()), 6),
                "mismatch_frac": round(
                    float((out_b != out_v).float().mean().item()), 6),
                "nan": bool(torch.isnan(out_v).any().item()),
                "inf": bool(torch.isinf(out_v).any().item()),
                "allzero": bool(int((out_v == 0).sum().item())
                                == out_v.numel()),
            }
            res["anomaly"] = (res["nan"] or res["inf"] or res["allzero"]
                              or any(r != 0 for r in rcs))
            return res
        except Exception as e:  # noqa: BLE001 - diagnostic only
            return {"error": str(e)[:120], "anomaly": False}

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
        all_ok = True
        for gid, nodes in items:
            if not nodes:
                # empty binding: a NO-OP, never a failed rollback
                continue
            gexec = self._exec_of(gid)
            # if the graph currently runs a CROSS variant, the source
            # nodes point into a separate module: pointing back at __base
            # is a cross-module change too -- take the E3 path (source
            # SetParams + whole-graph ExecUpdate), never a cross-module
            # exec SetParams (the known silent-corruption shape)
            cur_v = self._graph_active.get(gid)
            cur_meta = self._variants.get((jit_name, cur_v)) \
                if cur_v else None
            use_e3 = bool(cur_meta and cur_meta.get("cross"))
            graph_raw = self._keep[gid].raw_cuda_graph() if use_e3 else None
            applied_g = 0
            eligible = 0
            touched = []
            for nb in nodes:
                # V17 S3-3 ② (rollback side): never touch nodes that were
                # not born from this service's module -- writing our __base
                # handle into a triton-loaded node IS the cross-module
                # silent-corruption shape
                if int(nb.func_before) not in self._known_funcs:
                    with _LOCK:
                        self.state["skipped_nodes"] = \
                            int(self.state.get("skipped_nodes") or 0) + 1
                    continue
                eligible += 1
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
                if use_e3:
                    from cuda.bindings import runtime as _rt
                    rc = 1 if self._fi("setparams") else \
                        cu.cuGraphKernelNodeSetParams(nb.node, params)[0]
                else:
                    rc = 1 if self._fi("setparams") else \
                        cu.cuGraphExecKernelNodeSetParams(
                            gexec, nb.node, params)[0]
                if rc == 0:
                    applied_g += 1
                    touched.append(nb)
            ok_g = eligible > 0 and applied_g == eligible
            if eligible == 0:
                # defensive: no eligible node -> no-op graph, never a
                # failed rollback
                continue
            if not ok_g and applied_g:
                # V17 S3-2 ① (rollback side): partially rewritten nodes go
                # back to the CURRENT variant (the cross one we are
                # rolling back FROM)
                self._restore_nodes(cu, touched, cur_meta or v,
                                    gexec=gexec, source=use_e3)
            if use_e3 and ok_g:
                from cuda.bindings import runtime as _rt
                rc_u = 700 if self._fi("execupdate") else \
                    _rt.cudaGraphExecUpdate(int(gexec), int(graph_raw))[0]
                if rc_u != 0:
                    self._err("ExecUpdate(rollback)",
                              RuntimeError(f"rc={rc_u}"))
                    with _LOCK:
                        self.state["exec_update_failures"] = \
                            int(self.state.get("exec_update_failures")
                                or 0) + 1
                    # V17 S3-2 ③: the exec still runs the cross variant --
                    # _graph_active mirrors the EXEC, so it must stay, and
                    # the source nodes go back to the cross variant too
                    # (keep source == state)
                    self._restore_nodes(cu, touched, cur_meta or v,
                                        source=True)
                    ok_g = False
            if ok_g:
                with _LOCK:
                    self._graph_active.pop(gid, None)
            else:
                all_ok = False
            applied += applied_g
        with _LOCK:
            if all_ok and applied and items:
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
        # V19 X1/N16 trail instrumentation (PACT_TRAIL=1, default OFF --
        # three-principles discipline; probes open it explicitly).
        # Columns: replay host-submit path, whole-graph GPU span (events
        # hug the replay call, so the host gap is excluded), and the
        # boundary-callback host cost.  Rolling median of the last 256
        # reays lands in svc.state["trail"] -- pure observation, never a
        # decision input.
        # (rolling medians of the last 256 replays land in trail)
        trail_on = os.environ.get("PACT_TRAIL") == "1"
        _trail = {"host": [], "gpu": [], "bound": []}

        def _replay(self, *a, **kw):
            import time as _time
            ev_a = ev_b = None
            t0 = _time.perf_counter() if trail_on else None
            if trail_on:
                ev_a = torch.cuda.Event(enable_timing=True)
                ev_b = torch.cuda.Event(enable_timing=True)
                ev_a.record()
            r = orig(self, *a, **kw)  # replay first: instantiates the exec
            if trail_on:
                ev_b.record()
                t1 = _time.perf_counter()
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
            if trail_on:
                t2 = _time.perf_counter()
                _trail["host"].append((t1 - t0) * 1e6)
                _trail["gpu"].append(ev_a.elapsed_time(ev_b) * 1e3)
                _trail["bound"].append((t2 - t1) * 1e6)
                if len(_trail["host"]) > 256:
                    for k in _trail:
                        del _trail[k][:-256]
                if n % 32 == 0:
                    import statistics as _st
                    with _LOCK:
                        svc.state["trail"] = {
                            k: round(_st.median(v), 2)
                            for k, v in _trail.items() if v}
                        svc.state["trail"]["n"] = len(_trail["host"])
            return r

        torch.cuda.CUDAGraph.replay = _replay
        _REPLAY_HOOKED = True
        return True
    except Exception:
        return False
