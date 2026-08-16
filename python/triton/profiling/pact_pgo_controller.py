"""PACT PGO controllers.

PactPgoController (legacy shape): collect -> one theory candidate -> measure ->
swap.  Kept for compatibility with the v2 staircase scripts.

PactPgoGatedController (v3): vanilla-first, three gates.
  G1 context gate  : re-run only when the kernel/shape/arch/knob context changes
                     (or a rejected context is explicitly re-probed).
  G2 theory gate   : pact.pgo.trigger computed in C++ from contiguity/stage/warp
                     opportunities.  trigger=false -> no measurement, no swap.
  G3 measure gate  : paired median measurements of vanilla, theory candidate and
                     2-warp candidate; positive, >=5% and amortized-gain gate.
  G4 rollback guard: after swap, a short re-measure must stay faster than the
                     saved vanilla baseline; otherwise swap back atomically.
"""
import hashlib
import os
import statistics
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from triton.profiling.pact_profile_db import PactProfileDB
from triton.profiling.pact_profile_collector import PactProfileCollector
from triton.profiling.pact_kernel_swapper import PactKernelSwapper
from triton.profiling.pact_cupti_metrics import probe_active_warp_permille


def facts_to_hints(facts: Dict[str, Any]) -> Dict[str, int]:
    """Numeric pass inputs.

    measured_iterations and regs_per_thread are directly usable.  The old
    participation ratio must never be published as active_warp_ratio_permille:
    only a hardware probe result (key active_warp_ratio_permille) is accepted.
    """
    hints: Dict[str, int] = {}
    if facts.get("measured_iterations", 0) > 0:
        hints["pact.pgo.measured_iterations"] = int(facts["measured_iterations"])
    if facts.get("active_warp_ratio_permille") is not None:
        hints["pact.pgo.active_warp_ratio_permille"] = \
            int(facts["active_warp_ratio_permille"])
    if facts.get("regs_per_thread", 0) > 0:
        hints["pact.pgo.regs_per_thread"] = int(facts["regs_per_thread"])
    return hints


def _measure(swapper: PactKernelSwapper, kernel, iters: int = 20) -> float:
    lat = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        swapper.launch(kernel)
        end.record()
        torch.cuda.synchronize()
        lat.append(start.elapsed_time(end) * 1000.0)
    return statistics.median(lat)


def context_key(swapper: PactKernelSwapper, shape: tuple) -> str:
    """Cheap signature of everything that should re-arm the PGO trigger."""
    material = {
        "kernel": f"{swapper.jit_fn.fn.__module__}.{swapper.jit_fn.fn.__name__}",
        "shape": list(shape),
        "pact_env": sorted(
            (k, v) for k, v in os.environ.items() if k.startswith("PACT_")),
        "target": str(getattr(swapper.baseline.metadata, "target", "")),
    }
    return hashlib.sha256(repr(material).encode()).hexdigest()[:16]


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
        facts = self.collector.collect(steps=1)
        self.db.put(key, facts)
        facts.update(probe_active_warp_permille())
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

        base_us = _measure(swapper, swapper.baseline, measure_iters)
        cand_us = _measure(swapper, candidate, measure_iters)
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


class PactPgoGatedController:
    def __init__(self, db: Optional[PactProfileDB] = None,
                 collector: Optional[PactProfileCollector] = None,
                 hints_dir: Optional[Path] = None,
                 min_gain_percent: float = 5.0,
                 reprobe_after_steps: int = 0):
        self.db = db or PactProfileDB()
        self.collector = collector or PactProfileCollector()
        self.hints_dir = Path(hints_dir or (Path.home() / ".triton" / "pact_pgo"))
        self.hints_dir.mkdir(parents=True, exist_ok=True)
        self.min_gain_percent = min_gain_percent
        self.reprobe_after_steps = reprobe_after_steps
        self.last_plan: Dict[str, Any] = {}

    def evaluate(self, key: str, swapper: PactKernelSwapper,
                 remaining_steps: int, measure_iters: int = 20,
                 force: bool = False) -> Dict[str, Any]:
        """Run the full gated pipeline.  `key` should embed the context_key."""
        prior = self.db.get(key)
        if not force and prior.get("decision") in ("rejected", "swapped", "rolled_back"):
            # G1 context gate: same context never re-runs.
            return {"triggered": False, "reason": f"context gate: {prior.get('decision')}",
                    "context_key": key, **prior}

        facts = self.collector.collect(steps=1)
        facts.update(probe_active_warp_permille())
        hints = facts_to_hints(facts)
        self.db.put(key, facts)

        t0 = time.monotonic()
        try:
            candidate = swapper.compile_candidate(hints, str(self.hints_dir / f"{key}.json"))
        except Exception as e:
            self.db.put(key, {"decision": "rejected", "reason": f"compile failed: {e}"})
            return {"triggered": False, "reason": f"compile failed: {e}",
                    "context_key": key}
        compile_seconds = time.monotonic() - t0

        # G2 theory trigger, computed in C++ and returned through metadata.
        md = candidate.metadata
        md_dict = md._asdict() if hasattr(md, "_asdict") else dict(md)
        trigger = int(md_dict.get("pact.pgo.trigger", 0) or 0)
        trigger_reason = md_dict.get("pact.pgo.trigger_reason")
        if not force and not trigger:
            plan = {"triggered": False, "reason": f"theory gate: {trigger_reason}",
                    "trigger_reason": trigger_reason, "context_key": key}
            self.db.put(key, {"decision": "rejected", **plan})
            self.last_plan = plan
            return plan

        # S3b: theory candidate + explicit 2-warp candidate (P3/M2 retained,
        # P6/P11 disabled so they cannot override the explicit choice).
        theory_stages = int(md_dict.get("pact.optimal_num_stages", 3) or 3)
        try:
            low_warp = swapper.compile_explicit(num_stages=theory_stages,
                                                num_warps=2)
        except Exception as e:
            low_warp = None
        variants = {"theory": candidate, "low_warp": low_warp}
        us = {"baseline": _measure(swapper, swapper.baseline, measure_iters)}
        for name, kernel in variants.items():
            if kernel is not None:
                us[name] = _measure(swapper, kernel, measure_iters)

        chosen_name = min((k for k, v in us.items() if k != "baseline"),
                          key=lambda k: us[k])
        chosen = variants[chosen_name]
        gain_us = us["baseline"] - us[chosen_name]
        gain_percent = 100.0 * gain_us / max(us["baseline"], 1e-6)
        compile_cost_us = compile_seconds * 1e6 * 1.5
        predicted_us = gain_us * max(remaining_steps, 1)
        passed = (gain_us > 0 and gain_percent >= self.min_gain_percent and
                  predicted_us > compile_cost_us)

        plan = {
            "triggered": True,
            "trigger_reason": trigger_reason,
            "facts": facts,
            "hints": hints,
            "median_us": us,
            "chosen": chosen_name,
            "theory_stages": theory_stages,
            "gain_percent": gain_percent,
            "compile_seconds": compile_seconds,
            "context_key": key,
        }
        if not passed:
            plan.update({"swapped": False, "reason":
                         ("gain <= 0" if gain_us <= 0 else
                          f"gain {gain_percent:.1f}% < {self.min_gain_percent}%"
                          if gain_percent < self.min_gain_percent else
                          "predicted gain does not cover compile cost")})
            self.db.put(key, {"decision": "rejected", **plan})
            self.last_plan = plan
            return plan

        swapper.swap(chosen)
        # G4 rollback guard: a short post-swap re-measure must stay faster than
        # the saved vanilla baseline, otherwise swap back atomically.
        post_us = _measure(swapper, chosen, max(measure_iters // 2, 5))
        if post_us >= us["baseline"]:
            swapper.swap(swapper.baseline)
            plan.update({"swapped": False, "rolled_back": True,
                         "post_us": post_us,
                         "reason": "rollback guard: post-swap regression"})
            self.db.put(key, {"decision": "rolled_back", **plan})
        else:
            plan.update({"swapped": True, "rolled_back": False,
                         "post_us": post_us})
            self.db.put(key, {"decision": "swapped", **plan})
        self.last_plan = plan
        return plan
