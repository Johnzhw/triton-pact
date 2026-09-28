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
        """V17 S0-1 ②: event-pair timing with ZERO device-level sync.
        Per-iteration cuda events are recorded back-to-back and read only
        after event.synchronize() on the LAST event -- a device-wide
        torch.cuda.synchronize() here used to stall every other stream
        (the decode stream included) once per measured iteration."""
        import torch
        self.launch(kernel)          # settle, ordered before the first start
        pairs = []
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            self.launch(kernel)
            end.record()
            pairs.append((start, end))
        samples = []
        last_end = pairs[-1][1]
        last_end.synchronize()       # wait for the batch, not the device
        for start, end in pairs:
            samples.append(start.elapsed_time(end) * 1000.0)
        return statistics.median(samples)

    def measure_kernel(self, kernel, iters: int = 10) -> float:
        """Public wrapper over the G4 pair measurement (V12-P1: the async
        frame measures base/candidate on the background worker before the
        launch-boundary swap)."""
        return self._measure(kernel, iters)

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
        # V11-6a R4 (AOBO Table-1 style): segment the switch path so the
        # paper's overhead narrative has per-stage numbers and R2 pool hits
        # are distinguishable from compile slow paths.
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
            "slot_swap_ms": 0.0,
        }
        self.last_g4 = plan
        return plan

    # ---- V11-6a R2: resident variant pool (AOBO 预载思想) ------------------
    def prewarm_pool(self, variants: Dict[str, Dict[str, str]]
                     ) -> Dict[str, float]:
        """Compile the named variant set in an idle window and keep the
        kernels resident.  A later swap_from_pool(name) is a pure slot
        exchange — no compile, no module load on the switch path.

        V16-T3: additions go through the VariantRegistry (per-domain
        LRU cap, default 32, PACT_VARIANT_CAP); evicted names have their
        handles dropped here -- the triton disk cache keeps the cubin,
        so a later re-warm of an evicted name is a ms-level re-hit."""
        import time
        from triton.pact.runtime.variant_registry import default_registry
        if not hasattr(self, "_pool"):
            self._pool: Dict[str, Any] = {}
        if not hasattr(self, "evict_protected"):
            # V17 S2-4 B2: names the LRU wanted to evict but a bound
            # graph still runs (rollback+skip, never crash)
            self.evict_protected: list = []
        reg = default_registry()
        domain = (self._registry_model_tag(), self._registry_geo_key())
        spent = {}
        for name, extra_env in variants.items():
            if name in self._pool:
                reg.hit(domain, reg.variant_key(name, extra_env))
                continue
            t0 = time.monotonic()
            k, _ = compile_explicit(self.jit_fn, self.args, self.kwargs,
                                    dict(extra_env))
            if getattr(k, "_init_handles", None) and not getattr(
                    k, "function", None):
                try:
                    k._init_handles()
                except Exception:
                    pass
            key = reg.variant_key(name, extra_env)
            for _ev_key, ev_meta in reg.register(domain, key, {"name": name}):
                ev_name = (ev_meta or {}).get("name")
                if ev_name and ev_name in self._pool and ev_name != name:
                    if self._graph_runs_family(ev_name):
                        # V17 S2-4 B2: a bound graph still points at this
                        # family -- force the graphs back to __base and
                        # SKIP this eviction (keep the handle resident;
                        # recorded, never crashes)
                        self._rollback_bound_graph()
                        if ev_name not in self.evict_protected:
                            self.evict_protected.append(ev_name)
                        reg.hit(domain, _ev_key)
                        continue
                    del self._pool[ev_name]
            self._pool[name] = k
            spent[name] = (time.monotonic() - t0) * 1000.0
        return spent

    def _graph_runs_family(self, family: str) -> bool:
        """V17 S2-4 B2: True when any bound CUDA graph currently RUNS this
        family (the service is the single owner of that state; an unarmed
        process has an empty active map, which answers False on its own)."""
        try:
            from triton.pact.runtime.graph_service import (
                get_service, FAMILY_ALIAS)
            svc = get_service()
            jit = getattr(self.jit_fn, "__name__", None) \
                or self.jit_fn.fn.__name__
            v = FAMILY_ALIAS.get(family, family)
            return any(svc._jits.get(g) == jit and gv == v
                       for g, gv in svc._graph_active.items())
        except Exception:
            return False

    def _rollback_bound_graph(self) -> None:
        """V17 S2-4 B2: point every bound graph of this jit back at __base
        (the PTX-identical capture baseline) before protecting a handle."""
        try:
            from triton.pact.runtime.graph_service import get_service
            svc = get_service()
            jit = getattr(self.jit_fn, "__name__", None) \
                or self.jit_fn.fn.__name__
            svc.rollback(jit)
        except Exception:
            pass

    def _registry_model_tag(self) -> str:
        import os
        return os.environ.get("PACT_MODEL_TAG") or "default"

    def _registry_geo_key(self) -> str:
        kw = self.kwargs or {}
        return "|".join(str(kw.get(k, "")) for k in
                        ("NUM_TOKENS", "NUM_HEADS", "NUM_KV_HEADS",
                         "HEAD_DIM", "PAGE_SIZE", "MAX_SEQ_LEN"))

    def pool_contains(self, name: str) -> bool:
        return bool(getattr(self, "_pool", None)) and name in self._pool

    def pool_kernel(self, name: str):
        """V13 Phase0: read-only pool access so the graph service can
        link+load the pooled variant into its merged module — under a
        captured graph a slot exchange alone never reaches the node."""
        pool = getattr(self, "_pool", None)
        return pool.get(name) if pool else None

    def swap_from_pool(self, name: str) -> bool:
        """R2 fast path: slot-pointer exchange to a pooled variant.
        Returns False on a miss (caller falls back to the compile slow
        path).  Sub-millisecond by construction; measured in R4 segments."""
        import time
        pool = getattr(self, "_pool", None)
        if not pool or name not in pool:
            return False
        t0 = time.monotonic()
        self.swap(pool[name])
        self.last_g4 = dict(self.last_g4 or {})
        self.last_g4["pool_hit"] = name
        self.last_g4["slot_swap_ms"] = (time.monotonic() - t0) * 1000.0
        return True
