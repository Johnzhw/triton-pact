"""Resident variant pool: compile once, stay loaded, swap = pointer exchange.

The AOBO "预载文件入内存" counterpart for GPUs: predicted variant sets are
compiled during an idle window and stay resident as CompiledKernel handles;
launching a pooled variant never recompiles.  All launches go through the
JIT binder normalization (the V11-0 root-cause fix — raw args must match the
specialized launcher signature; see pact_paper suite/results/v11/
race_fix_v11.md).
"""
from __future__ import annotations

import threading
from typing import Any, Dict, Optional, Tuple

from triton.pact.compiler.explicit_compiler import compile_explicit


def _normalize_grid(grid, bound_args) -> Tuple[int, int, int]:
    g = grid(bound_args) if callable(grid) else grid
    g = (g + (1, 1))[:3]
    return (int(g[0]), int(g[1]), int(g[2]))


class ResidentPool:
    """One jit_fn + geometry, many resident variants keyed by name.

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
        self._lock = threading.RLock()
        self._kernels: Dict[str, Any] = {}
        self._active: Optional[str] = None
        base, _bound = compile_explicit(
            jit_fn, args, kwargs,
            dict(baseline_env or {"PACT_ENABLE": "0"}))
        self._kernels["baseline"] = base
        self._active = "baseline"

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
                self._kernels[name] = k
            spent[name] = (time.monotonic() - t0) * 1000.0
        return spent

    def add_kernel(self, name: str, kernel) -> None:
        with self._lock:
            self._kernels.setdefault(name, kernel)

    def __contains__(self, name: str) -> bool:
        with self._lock:
            return name in self._kernels

    def names(self):
        with self._lock:
            return sorted(self._kernels)

    @property
    def active(self) -> str:
        with self._lock:
            return self._active

    def swap(self, name: str) -> None:
        """Slot-pointer exchange — the only switch-time operation."""
        with self._lock:
            if name not in self._kernels:
                raise KeyError(f"variant {name!r} not resident")
            self._active = name

    # ---- launch ----------------------------------------------------------
    def launch(self, name: Optional[str] = None, args=None, grid=None):
        """Binder-normalized launch of a pooled variant with CURRENT args."""
        from triton.runtime import driver
        a = tuple(self.args if args is None else args)
        device = driver.active.get_current_device()
        _cache, _k, _target, _backend, binder = self.jit_fn.device_caches[device]
        bound, _spec, _opts = binder(*a, **self.kwargs)
        k = self._kernels[name or self._active]
        g = _normalize_grid(grid or self.grid, bound)
        return k[g](*bound.values())

    def retarget(self, args, kwargs, grid):
        """Point compiles/launches at a new tensor tuple (same constexprs)."""
        self.args = args
        self.kwargs = kwargs
        self.grid = grid
