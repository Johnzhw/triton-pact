"""In-process dual-slot kernel pointer + G4 rollback.

Slot 0 is the live baseline.  Slot 1 is the candidate.  Swap is under a lock.
G4 re-measures after the swap and rolls back if the new kernel is not faster.
"""
from __future__ import annotations

import os
import statistics
import threading
from typing import Any, Callable, Dict, Optional, Tuple

from triton.pact.compiler.explicit_compiler import compile_explicit


def _normalize_grid(grid, bound_args) -> Tuple[int, int, int]:
    g = grid(bound_args) if callable(grid) else grid
    if len(g) == 1:
        return (int(g[0]), 1, 1)
    if len(g) == 2:
        return (int(g[0]), int(g[1]), 1)
    return (int(g[0]), int(g[1]), int(g[2]))


class HotSwapper:
    def __init__(self, jit_fn, args, kwargs, grid,
                 baseline_env: Optional[Dict[str, str]] = None):
        self.jit_fn = jit_fn
        self.args = args
        self.kwargs = kwargs
        self.grid = grid
        self.baseline_env = dict(baseline_env or {"PACT_ENABLE": "0"})
        self._lock = threading.Lock()
        baseline, bound = compile_explicit(
            jit_fn, args, kwargs, self.baseline_env)
        self.bound_args = bound
        self.slots = [baseline, None]
        self.active = 0
        self.last_g4: Dict[str, Any] = {}

    @property
    def baseline(self):
        return self.slots[0]

    @property
    def current(self):
        with self._lock:
            return self.slots[self.active] or self.slots[0]

    def compile_candidate(self, extra_env: Dict[str, str],
                          options_override: Optional[Dict[str, Any]] = None):
        kernel, bound = compile_explicit(
            self.jit_fn, self.args, self.kwargs, extra_env,
            options_override=options_override)
        self.bound_args = bound
        return kernel

    def launch(self, kernel=None, grid=None):
        k = kernel if kernel is not None else self.current
        g = _normalize_grid(grid or self.grid, self.bound_args)
        return k[g](*self.bound_args.values())

    def retarget(self, args, kwargs, grid):
        """Point the next compile/launch at a new tensor tuple (same constexprs)."""
        self.args = args
        self.kwargs = kwargs
        self.grid = grid

    def launch_with(self, args, grid=None):
        # V11-0: raw args must go through the same binder normalization the
        # kernel was compiled with (specialization drops/reorders params);
        # feeding the raw tuple to the CompiledKernel launcher mismatched
        # the C-side extraction and segfaulted ~25% of EngineCore runs
        # (launchKernel→extractI64; bisect: compile side 10/10 clean,
        # raw launch 2/2 crash — suite/results/v11/race_fix_v11.md).
        from triton.runtime import driver
        device = driver.active.get_current_device()
        _cache, _k, _target, _backend, binder = self.jit_fn.device_caches[device]
        bound, _spec, _opts = binder(*args, **self.kwargs)
        k = self.current
        g = _normalize_grid(grid or self.grid, bound)
        return k[g](*bound.values())

    def swap(self, kernel) -> int:
        with self._lock:
            if kernel is self.slots[0]:
                self.active = 0
            else:
                self.slots[1] = kernel
                self.active = 1
            return self.active

    def _measure(self, kernel, iters: int) -> float:
        import torch
        samples = []
        self.launch(kernel)
        torch.cuda.synchronize()
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            self.launch(kernel)
            end.record()
            torch.cuda.synchronize()
            samples.append(start.elapsed_time(end) * 1000.0)
        return statistics.median(samples)

    def g4_install(self, kernel, measure_iters: int = 10,
                   min_gain_percent: float = 0.0) -> Dict[str, Any]:
        """Install `kernel` into slot 1, swap, rollback on regression."""
        import time
        t0 = time.monotonic()
        measure_iters = int(os.environ.get("PACT_G4_ITERS", str(measure_iters)))
        base_us = self._measure(self.slots[0], measure_iters)
        cand_us = self._measure(kernel, measure_iters)
        gain_percent = 100.0 * (base_us - cand_us) / max(base_us, 1e-6)
        self.swap(kernel)
        post_us = self._measure(self.current, max(measure_iters // 2, 3))
        measure_time_ms = (time.monotonic() - t0) * 1000.0
        rolled = post_us >= base_us or gain_percent < min_gain_percent
        if rolled:
            self.swap(self.slots[0])
        plan = {
            "baseline_us": base_us,
            "candidate_us": cand_us,
            "post_us": post_us,
            "gain_percent": gain_percent,
            "swapped": not rolled,
            "rolled_back": rolled and post_us >= base_us,
            "measure_time_ms": measure_time_ms,
        }
        self.last_g4 = plan
        return plan
