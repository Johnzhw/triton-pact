"""
PACT Online Profiler — Lightweight runtime performance sampling.

Design principles:
  1. Low overhead: sample every N steps (default 50), not every step
  2. Non-blocking: background thread for analysis, main loop unblocked
  3. Fallback: CUDA event-based when Proton/CUPTI unavailable
  4. Windowed: keep rolling window of recent metrics for trend analysis

Usage:
  profiler = OnlineProfiler(sample_interval=50, window_size=20)
  for step_id in range(num_steps):
      output, metrics = profiler.profile_step(
          lambda: model.step(input), step_id)
"""

import time
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch

# Try Proton API (lightweight CUPTI-based profiling).
# M8a: triton.profiler has no Proton *class* (it exports a functional API), so
# use ProtonAdapter to bridge functional -> OOP.  Fall back to the raw import if
# ProtonAdapter is unavailable.
try:
    from triton.profiling.proton_adapter import ProtonAdapter as Proton
    HAS_PROTON = True
except ImportError:
    try:
        from triton.profiler import Proton
        HAS_PROTON = True
    except ImportError:
        HAS_PROTON = False


@dataclass
class StepMetrics:
    """Single decode-step performance metrics."""

    step_id: int
    timestamp: float

    # GPU hardware metrics
    sm_efficiency: float = 0.0
    achieved_occupancy: float = 0.0
    memory_throughput: float = 0.0
    l2_hit_rate: float = 0.0

    # Kernel-specific
    kernel_latency_us: float = 0.0
    num_stages: int = 3
    num_warps: int = 4
    tokens_per_sec: float = 0.0

    # Derived bottleneck flags
    is_compute_bound: bool = False
    is_memory_bound: bool = False
    is_occupancy_limited: bool = False

    def summary(self) -> str:
        flags = []
        if self.is_occupancy_limited: flags.append("OCC-LIMITED")
        if self.is_memory_bound: flags.append("MEM-BOUND")
        if self.is_compute_bound: flags.append("COMP-BOUND")
        return (f"step={self.step_id} "
                f"lat={self.kernel_latency_us:.0f}us "
                f"occ={self.achieved_occupancy:.0f}% "
                f"sm={self.sm_efficiency:.0f}% "
                f"{' '.join(flags)}")


class OnlineProfiler:
    """
    Lightweight online performance sampler for PACT.

    Samples GPU metrics periodically during decode, feeds a rolling
    window to the bottleneck analyzer for optimization decisions.
    """

    def __init__(
        self,
        sample_interval: int = 50,
        window_size: int = 20,
        enable_cupti: bool = True,
    ):
        self.sample_interval = sample_interval
        self.window_size = window_size
        self.enable_cupti = enable_cupti and HAS_PROTON

        # Rolling window of recent metrics
        self.metrics_history: deque[StepMetrics] = deque(maxlen=window_size)

        # Optimization tracking
        self.current_preset = "vanilla"
        self.optimization_count = 0

        # Background analysis
        self._analysis_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._pending_bottleneck: Optional[Dict] = None
        self._bottleneck_callback: Optional[Callable] = None

        # Proton profiler (if available)
        self._proton = None
        if self.enable_cupti:
            self._init_proton()

        # Start background thread
        self._start_background_analysis()

    def _init_proton(self):
        """Initialize Proton CUPTI profiler (via ProtonAdapter)."""
        try:
            self._proton = Proton()
            if not self._proton.available:
                print("[PACT OnlineProfiler] Proton adapter not available, "
                      "using CUDA event fallback.")
                self.enable_cupti = False
                return
            self._proton.start()
            print("[PACT OnlineProfiler] Proton CUPTI profiler initialized "
                  "(shadow mode, adapter).")
        except Exception as e:
            print(f"[PACT OnlineProfiler] Proton init failed: {e}")
            self.enable_cupti = False

    def _start_background_analysis(self):
        """Start background analysis thread."""

        def analysis_loop():
            while not self._stop_event.is_set():
                if len(self.metrics_history) >= 10:
                    bottleneck = self._analyze_bottleneck()
                    if bottleneck and self._bottleneck_callback:
                        self._pending_bottleneck = bottleneck
                time.sleep(2.0)

        self._analysis_thread = threading.Thread(
            target=analysis_loop, daemon=True
        )
        self._analysis_thread.start()

    def set_bottleneck_callback(self, callback: Callable[[Dict], None]):
        """Register callback for bottleneck detection."""
        self._bottleneck_callback = callback

    def profile_step(
        self, step_fn: Callable, step_id: int
    ) -> Tuple[any, StepMetrics]:
        """
        Execute one decode step and optionally sample metrics.

        Args:
            step_fn: callable that runs one decode step
            step_id: monotonic step counter

        Returns:
            (step_output, StepMetrics)
        """
        should_sample = step_id % self.sample_interval == 0

        if should_sample and self._proton:
            with self._proton.profile(
                name=f"decode_step_{step_id}",
                metrics=[
                    "sm_efficiency",
                    "occupancy",
                    "memory_throughput",
                    "l2_hit_rate",
                    "global_load",
                    "shared_load",
                ],
            ):
                start_ev = torch.cuda.Event(enable_timing=True)
                end_ev = torch.cuda.Event(enable_timing=True)
                start_ev.record()
                output = step_fn()
                end_ev.record()
                torch.cuda.synchronize()
                latency_us = start_ev.elapsed_time(end_ev) * 1000

            raw_metrics = self._proton.get_last_metrics()
            metrics = self._build_metrics(step_id, latency_us, raw_metrics)
        else:
            # Non-sampling step: CUDA event timing only
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)
            start_ev.record()
            output = step_fn()
            end_ev.record()
            torch.cuda.synchronize()
            latency_us = start_ev.elapsed_time(end_ev) * 1000
            metrics = self._build_metrics(step_id, latency_us, None)

        self.metrics_history.append(metrics)
        return output, metrics

    def _build_metrics(
        self, step_id: int, latency_us: float, raw_metrics: Optional[Dict]
    ) -> StepMetrics:
        """Build standardized StepMetrics from raw data."""
        m = StepMetrics(
            step_id=step_id,
            timestamp=time.monotonic(),
            kernel_latency_us=latency_us,
            tokens_per_sec=1e6 / max(latency_us, 1.0),
        )

        if raw_metrics:
            m.sm_efficiency = raw_metrics.get("sm_efficiency", 0)
            m.achieved_occupancy = raw_metrics.get("occupancy", 0)
            m.memory_throughput = raw_metrics.get("memory_throughput", 0)
            m.l2_hit_rate = raw_metrics.get("l2_hit_rate", 0)

        # Heuristic bottleneck classification
        if m.achieved_occupancy < 50:
            m.is_occupancy_limited = True
        if m.memory_throughput > 80:
            m.is_memory_bound = True
        if m.sm_efficiency > 80 and not m.is_memory_bound:
            m.is_compute_bound = True

        return m

    def _analyze_bottleneck(self) -> Optional[Dict]:
        """Analyze recent metrics window to identify bottleneck."""
        recent = list(self.metrics_history)[-10:]
        if not recent:
            return None

        avg_sm = sum(m.sm_efficiency for m in recent) / len(recent)
        avg_occ = sum(m.achieved_occupancy for m in recent) / len(recent)
        avg_mem = sum(m.memory_throughput for m in recent) / len(recent)
        avg_l2 = sum(m.l2_hit_rate for m in recent) / len(recent)
        avg_lat = sum(m.kernel_latency_us for m in recent) / len(recent)

        bottleneck = None

        if avg_occ < 50 and avg_sm < 60:
            bottleneck = {
                "type": "low_occupancy",
                "severity": 1.0 - avg_occ / 100,
                "suggested_passes": [
                    "P11_auto_warps",
                    "P6_reduce_stages",
                ],
                "metrics": {
                    "occupancy": avg_occ,
                    "sm_efficiency": avg_sm,
                },
            }
        elif avg_mem > 80 and avg_l2 < 70:
            bottleneck = {
                "type": "memory_bandwidth_bound",
                "severity": avg_mem / 100,
                "suggested_passes": [
                    "P3_axisinfo_override",
                    "P6_increase_stages",
                    "P9_coalescing",
                ],
                "metrics": {
                    "memory_util": avg_mem,
                    "l2_hit_rate": avg_l2,
                },
            }
        elif avg_sm > 80 and avg_mem < 60:
            bottleneck = {
                "type": "compute_bound",
                "severity": avg_sm / 100,
                "suggested_passes": ["P12_dot_promotion"],
                "metrics": {"sm_efficiency": avg_sm},
            }
        elif avg_l2 < 50:
            bottleneck = {
                "type": "poor_locality",
                "severity": 1.0 - avg_l2 / 100,
                "suggested_passes": [
                    "P7_page_major",
                    "P9_coalescing",
                ],
                "metrics": {"l2_hit_rate": avg_l2},
            }

        if bottleneck:
            print(
                f"[PACT OnlineProfiler] Bottleneck: {bottleneck['type']} "
                f"(severity={bottleneck['severity']:.2f})"
            )

        return bottleneck

    def get_summary(self) -> Dict:
        """Get profiler summary for reporting."""
        if not self.metrics_history:
            return {"status": "no_data"}
        recent = list(self.metrics_history)[-10:]
        return {
            "samples": len(self.metrics_history),
            "avg_latency_us": sum(
                m.kernel_latency_us for m in recent
            ) / len(recent),
            "avg_occupancy": sum(
                m.achieved_occupancy for m in recent
            ) / len(recent),
            "optimizations": self.optimization_count,
            "current_preset": self.current_preset,
        }

    def shutdown(self):
        """Stop profiler and background thread."""
        self._stop_event.set()
        if self._analysis_thread:
            self._analysis_thread.join(timeout=5.0)
        if self._proton:
            self._proton.stop()
        print("[PACT OnlineProfiler] Shutdown complete.")
