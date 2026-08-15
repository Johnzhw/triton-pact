"""PACT PGO controller.

Policy: collect facts -> derive numeric pass inputs -> compile ONE candidate ->
measure candidate vs baseline -> swap only if the amortized benefit pays for
the compile.  There is deliberately no bottleneck-to-pass-enable map here;
P6/P11 consume the numeric pact.pgo.* attributes and fall back to their
theory-only defaults when no facts exist.
"""
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from triton.profiling.pact_profile_db import PactProfileDB
from triton.profiling.pact_profile_collector import PactProfileCollector
from triton.profiling.pact_kernel_swapper import PactKernelSwapper


def facts_to_hints(facts: Dict[str, Any]) -> Dict[str, int]:
    hints: Dict[str, int] = {}
    if facts.get("measured_iterations", 0) > 0:
        hints["pact.pgo.measured_iterations"] = int(facts["measured_iterations"])
    if facts.get("active_warp_ratio_permille") is not None:
        hints["pact.pgo.active_warp_ratio_permille"] = \
            int(facts["active_warp_ratio_permille"])
    if facts.get("regs_per_thread", 0) > 0:
        hints["pact.pgo.regs_per_thread"] = int(facts["regs_per_thread"])
    if facts.get("pipeline_overlap_benefit_permille") is not None:
        hints["pact.pgo.pipeline_overlap_benefit_permille"] = \
            int(facts["pipeline_overlap_benefit_permille"])
    return hints


class PactPgoController:
    def __init__(self, db: Optional[PactProfileDB] = None,
                 collector: Optional[PactProfileCollector] = None,
                 hints_dir: Optional[Path] = None):
        self.db = db or PactProfileDB()
        self.collector = collector or PactProfileCollector()
        self.hints_dir = Path(hints_dir or (Path.home() / ".triton" / "pact_pgo"))
        self.hints_dir.mkdir(parents=True, exist_ok=True)
        self.last_plan: Dict[str, Any] = {}

    def collect(self, key: str, steps: int = 20) -> Dict[str, Any]:
        facts = self.collector.collect(steps=steps)
        self.db.put(key, facts)
        return facts

    def run(self, key: str, swapper: PactKernelSwapper, remaining_steps: int,
            measure_iters: int = 20, min_gain_percent: float = 5.0) -> Dict[str, Any]:
        """Collect facts, compile one candidate, measure, and swap if worth it."""
        facts = self.collector.collect(steps=1)
        self.db.put(key, facts)
        hints = facts_to_hints(facts)
        if not hints:
            return {"swapped": False, "reason": "no actionable facts"}

        hints_path = self.hints_dir / f"{key}.json"
        t0 = time.monotonic()
        try:
            candidate = swapper.compile_candidate(hints, str(hints_path))
        except Exception as e:
            return {"swapped": False, "reason": f"compile failed: {e}"}
        compile_seconds = time.monotonic() - t0

        # Small paired measurement: baseline and candidate, interleaved.
        def measure(kernel):
            lat = 0.0
            for _ in range(measure_iters):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                swapper.launch(kernel)
                end.record()
                torch.cuda.synchronize()
                lat += start.elapsed_time(end) * 1000.0
            return lat / measure_iters

        base_us = measure(swapper.baseline)
        cand_us = measure(candidate)
        gain_per_step_us = base_us - cand_us
        gain_percent = 100.0 * gain_per_step_us / max(base_us, 1e-6)
        predicted_us = gain_per_step_us * remaining_steps
        compile_cost_us = compile_seconds * 1e6 * 1.5

        plan = {
            "facts": facts,
            "hints": hints,
            "baseline_us": base_us,
            "candidate_us": cand_us,
            "gain_per_step_us": gain_per_step_us,
            "gain_percent": gain_percent,
            "compile_seconds": compile_seconds,
            "remaining_steps": remaining_steps,
        }

        if gain_per_step_us > 0 and gain_percent >= min_gain_percent and \
                predicted_us > compile_cost_us:
            swapper.swap(candidate)
            plan["swapped"] = True
        else:
            plan["swapped"] = False
        self.last_plan = plan
        return plan
