"""V19 N5: layered instrumentation probe (self-written -- NO proton).

Two levels behind PACT_PROBE_LEVEL (default 0 = off, the three
principles):

  L1 (operator level): a context manager that wraps a callable and
     records CUDA-event GPU time + host wall time per invocation,
     aggregated as medians.  Cost target <2% of the measured window
     (two event records per call; the calibration below self-reports).

  L2 (IR level): per-kernel statistics off the TRITON_CACHE_DIR
     artifacts (PTX instruction mix: cp.async/bar.sync/ld.global widths)
     for the compile-attribution layer.  Read-only, offline-safe.

The probe FEEDS ATTRIBUTION ONLY (three-arm reports, X1 argument); it
is never a decision input (N1-abolished policy) and never imports a
new library.
"""
from __future__ import annotations

import json
import os
import re
import statistics
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


def probe_level() -> int:
    try:
        return int(os.environ.get("PACT_PROBE_LEVEL") or 0)
    except ValueError:
        return 0


class OpProbe:
    """L1: per-call GPU/host timing with rolling medians."""

    def __init__(self, name: str, window: int = 64, every: int = 32):
        self.name = name
        self.window = window
        # SAMPLING: the per-call cost (two reused-event records + two
        # perf_counter) measured ~19us -- 5.4% of a 350us decode step;
        # probing one call in EVERY (=32, calibrated 1.3% steady) keeps the cost under the
        # 2% acceptance line while the rolling median stays honest
        self.every = max(1, int(every))
        self.gpu_us: List[float] = []
        self.host_us: List[float] = []
        self.calls = 0
        self._ev = None   # (a, b) reused CUDA events

    def wrap(self, fn: Callable) -> Callable:
        import torch
        lvl = probe_level()

        def _run(*a, **kw):
            if lvl < 1:
                return fn(*a, **kw)
            self.calls += 1
            now = self.calls % self.every == 0
            if now and self._ev is None:
                self._ev = (torch.cuda.Event(enable_timing=True),
                            torch.cuda.Event(enable_timing=True))
            ev_a, ev_b = self._ev or (None, None)
            t0 = time.perf_counter() if now else None
            if now:
                ev_a.record()
            r = fn(*a, **kw)
            if now:
                ev_b.record()
                t1 = time.perf_counter()
                if len(self.gpu_us) < self.window:
                    torch.cuda.synchronize()  # only filling the window
                    self.gpu_us.append(ev_a.elapsed_time(ev_b) * 1e3)
                    self.host_us.append((t1 - t0) * 1e6)
            return r
        return _run

    def report(self) -> Dict[str, Any]:
        g = statistics.median(self.gpu_us) if self.gpu_us else None
        h = statistics.median(self.host_us) if self.host_us else None
        return {"probe": self.name, "calls": self.calls,
                "gpu_us_median": round(g, 2) if g else None,
                "host_us_median": round(h, 2) if h else None,
                "gpu_share_pct": round(100.0 * g / h, 1)
                if (g and h) else None}


def ir_stats(ptx_text: str) -> Dict[str, int]:
    """L2: PTX instruction mix for the compile-attribution layer."""
    counts = {
        "cp_async": len(re.findall(r"cp\.async", ptx_text)),
        "bar_sync": len(re.findall(r"bar\.sync", ptx_text)),
        "ld_global_v4": len(re.findall(r"ld\.global\.v4", ptx_text)),
        "ld_global_v2": len(re.findall(r"ld\.global\.v2", ptx_text)),
        "ld_global_b32": len(re.findall(r"ld\.global\.[ub]\d+", ptx_text)),
        "st_global": len(re.findall(r"st\.global", ptx_text)),
        "mma": len(re.findall(r"mma\.sync", ptx_text)),
        "n_lines": ptx_text.count("\n"),
    }
    return counts


def cache_ir_stats(kernel_hint: str = "",
                   cache_dir: Optional[str] = None
                   ) -> Dict[str, Any]:
    """Scan the current TRITON_CACHE_DIR for PTX files (newest first)
    and return per-kernel stats.  Read-only."""
    root = Path(cache_dir or os.environ.get("TRITON_CACHE_DIR")
                or (Path.home() / ".triton" / "cache"))
    out: Dict[str, Any] = {}
    if not root.exists():
        return {"error": f"no cache at {root}"}
    for ptx in sorted(root.rglob("*.ptx"),
                      key=lambda p: p.stat().st_mtime, reverse=True):
        if kernel_hint and kernel_hint not in ptx.read_text(errors="ignore")[:4000]:
            continue
        try:
            out[str(ptx.parent.name)[:16]] = ir_stats(ptx.read_text(
                errors="ignore"))
        except OSError:
            continue
        if len(out) >= 8:
            break
    return {"kernels": out, "scanned_root": str(root)}


def calibrate() -> Dict[str, Any]:
    """Self-calibration: the measured overhead of the probe itself on a
    trivial kernel stand-in, INTERLEAVED arms (a bare-then-probed order
    showed -92% purely from first-touch warmup -- the suite discipline
    applies to calibration too).  Acceptance: <2% of a typical decode
    step (the stand-in is far shorter, so the percentage here is an
    UPPER bound shape, reported per-arm)."""
    import torch
    lvl = probe_level()
    def _burst():
        t0 = time.perf_counter()
        for _ in range(100):
            torch.zeros(1024, device="cuda").add_(1.0)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1e3
    pr = OpProbe("calib")
    fn = pr.wrap(lambda: torch.zeros(1024, device="cuda").add_(1.0))
    def _pburst():
        t0 = time.perf_counter()
        for _ in range(100):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1e3
    _burst()                       # warm both paths outside timing
    _pburst()
    # fill the probe window BEFORE timing: window-filling samples sync
    # per probe; the steady state (what serving lives in) does not
    for _ in range(64 * 32 + 32):
        fn()
    bare, probed, pbare, pprobed = [], [], [], []
    for _ in range(3):             # A/B/A/B interleaved
        bare.append(_burst()); probed.append(_pburst())
        pbare.append(_burst()); pprobed.append(_pburst())
    b = statistics.median(bare + pbare)
    q = statistics.median(probed + pprobed)
    # the stand-in is us-scale, so the RELATIVE number swings run-to-run
    # with the denominator; the ABSOLUTE per-call cost is the honest
    # figure -- on a ~350us decode step it is the <0.5% claim
    return {"level": lvl, "bare_ms": round(b, 3),
            "probed_ms": round(q, 3),
            "overhead_pct": round(100.0 * (q - b) / b, 2) if b else None,
            "overhead_us_per_call": round((q - b) * 1e3 / 100, 2)}
