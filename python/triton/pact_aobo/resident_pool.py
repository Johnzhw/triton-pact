"""Resident variant pool: compile once, stay loaded, swap = pointer exchange.

The AOBO "预载文件入内存" counterpart for GPUs: predicted variant sets are
compiled during an idle window and stay resident as CompiledKernel handles;
launching a pooled variant never recompiles.  All launches go through the
JIT binder normalization (the V11-0 root-cause fix — raw args must match the
specialized launcher signature; see pact_paper suite/results/v11/
race_fix_v11.md).

V12-2c grid pooling: kernels are specialized by constexpr geometry
(MAX_SEQ_LEN / strides / ...), so one flat name->kernel map breaks the
moment the workload moves to a new S bucket.  The pool now keys variants
per geometry: `self._geo[geo_key] = {name: kernel}` with an independent
active slot per geometry.  All public surface (prewarm/swap/launch/
__contains__/active/retarget) routes to the CURRENT geometry, so callers
that retarget per forward (the dynamic-e2e bridge shape) keep working
unchanged; a first-seen geometry compiles its baseline lazily on
retarget (one slow hit per bucket — same accounting as the pgo slow
path).
"""
from __future__ import annotations

import threading
from typing import Any, Dict, Optional, Tuple

from triton.pact.compiler.explicit_compiler import compile_explicit

# constexpr fields that specialize the compiled kernel: two workloads with
# the same key share variants; any difference needs its own pool
_GEO_KEYS = ("NUM_TOKENS", "NUM_HEADS", "NUM_KV_HEADS", "HEAD_DIM",
             "PAGE_SIZE", "MAX_SEQ_LEN", "TILE_SIZE", "GQA_RATIO",
             "STRIDE_BLOCK", "STRIDE_KV_HEAD", "STRIDE_PAGE",
             "STRIDE_HEAD_DIM", "USE_DUAL_TILE", "TILE_SIZE_LARGE",
             "TOKEN_IMPORTANCE_MODE")


def _geo_key(kwargs) -> tuple:
    return tuple((k, kwargs[k]) for k in _GEO_KEYS if k in kwargs)


def _normalize_grid(grid, bound_args) -> Tuple[int, int, int]:
    g = grid(bound_args) if callable(grid) else grid
    g = (g + (1, 1))[:3]
    return (int(g[0]), int(g[1]), int(g[2]))


class ResidentPool:
    """One jit_fn, many geometries; per geometry, many resident variants
    keyed by name.

    The pool owns the args form: every launch re-binds the *current* args
    through the jit binder, so retargeting to new tensors is free and the
    launcher signature always matches (no launch_with-style raw-args path).
    """

    def __init__(self, jit_fn, args, kwargs, grid,
                 baseline_env: Optional[Dict[str, str]] = None):
        self.jit_fn = jit_fn
        self.args = args
        self.kwargs = kwargs
        self.grid = grid
        self._baseline_env = dict(baseline_env or {"PACT_ENABLE": "0"})
        self._lock = threading.RLock()
        self._geo: Dict[tuple, Dict[str, Any]] = {}
        self._cur_key: Optional[tuple] = None
        self._ensure_geo(args, kwargs)

    # ---- geometry routing -------------------------------------------------
    def _ensure_geo(self, args, kwargs) -> tuple:
        """Compile the baseline for a first-seen geometry (blocking; call
        from retarget/init so launches never pay it)."""
        key = _geo_key(kwargs)
        with self._lock:
            if key not in self._geo:
                base, _bound = compile_explicit(
                    self.jit_fn, args, kwargs, dict(self._baseline_env))
                self._geo[key] = {"kernels": {"baseline": base},
                                  "active": "baseline"}
            self._cur_key = key
            return key

    @property
    def _kernels(self) -> Dict[str, Any]:
        """Kernels of the CURRENT geometry (AsyncKernelSwitch compat)."""
        with self._lock:
            return self._geo[self._cur_key]["kernels"]

    def geo_keys(self):
        with self._lock:
            return list(self._geo)

    # ---- pool management -------------------------------------------------
    def prewarm(self, variants: Dict[str, Dict[str, str]]) -> Dict[str, float]:
        """Compile the named variant set NOW (blocking; call in an idle
        window).  Returns per-variant wall-ms.  Existing names are no-ops."""
        import time
        spent = {}
        for name, extra_env in variants.items():
            if name in self._kernels:
                continue
            t0 = time.monotonic()
            k, _ = compile_explicit(self.jit_fn, self.args, self.kwargs,
                                    dict(extra_env))
            with self._lock:
                self._geo[self._cur_key]["kernels"][name] = k
            spent[name] = (time.monotonic() - t0) * 1000.0
        return spent

    def add_kernel(self, name: str, kernel) -> None:
        with self._lock:
            self._geo[self._cur_key]["kernels"].setdefault(name, kernel)

    def __contains__(self, name: str) -> bool:
        with self._lock:
            return name in self._geo[self._cur_key]["kernels"]

    def names(self):
        with self._lock:
            return sorted(self._geo[self._cur_key]["kernels"])

    @property
    def active(self) -> str:
        with self._lock:
            return self._geo[self._cur_key]["active"]

    def swap(self, name: str) -> None:
        """Slot-pointer exchange — the only switch-time operation."""
        with self._lock:
            if name not in self._geo[self._cur_key]["kernels"]:
                raise KeyError(f"variant {name!r} not resident")
            self._geo[self._cur_key]["active"] = name

    # ---- launch ----------------------------------------------------------
    def launch(self, name: Optional[str] = None, args=None, grid=None):
        """Binder-normalized launch of a pooled variant with CURRENT args."""
        from triton.runtime import driver
        a = tuple(self.args if args is None else args)
        device = driver.active.get_current_device()
        _cache, _k, _target, _backend, binder = self.jit_fn.device_caches[device]
        bound, _spec, _opts = binder(*a, **self.kwargs)
        with self._lock:
            sub = self._geo[self._cur_key]
            k = sub["kernels"][name or sub["active"]]
        g = _normalize_grid(grid or self.grid, bound)
        return k[g](*bound.values())

    def dry_launch(self, kernel, args, kwargs, grid=None):
        """Binder-normalized launch with an EXPLICIT args/kwargs snapshot
        (background warm-up path): the caller owns the tensors, so a
        dummy-output copy settles module lazy-load / first-launch cost
        off any serving forward."""
        from triton.runtime import driver
        device = driver.active.get_current_device()
        _cache, _k, _target, _backend, binder = \
            self.jit_fn.device_caches[device]
        bound, _spec, _opts = binder(*args, **kwargs)
        g = _normalize_grid(grid or self.grid, bound)
        return kernel[g](*bound.values())

    def retarget(self, args, kwargs, grid):
        """Point compiles/launches at a new tensor tuple.  A new constexpr
        geometry (e.g. a new MAX_SEQ_LEN bucket) transparently gets its own
        sub-pool with a lazily compiled baseline."""
        self.args = args
        self.kwargs = kwargs
        self.grid = grid
        self._ensure_geo(args, kwargs)
