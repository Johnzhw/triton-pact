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


def _remaining_steps_bucket(remaining_steps):
    if remaining_steps is None:
        return "unknown"
    for cap in (64, 512, 2048, 8192):
        if remaining_steps <= cap:
            return f"le{cap}"
    return "gt8192"


def context_key(swapper: PactKernelSwapper, shape: tuple,
                remaining_steps: Optional[int] = None) -> str:
    """Cheap signature of everything that should re-arm the PGO trigger.

    ``remaining_steps`` is bucketed coarsely so an amortization rejection at a
    short remaining decode length does not permanently latch the same
    kernel/shape when a later request has enough steps to amortize the PGO
    compile cost.
    """
    material = {
        "kernel": f"{swapper.jit_fn.fn.__module__}.{swapper.jit_fn.fn.__name__}",
        "shape": list(shape),
        "pact_env": sorted(
            (k, v) for k, v in os.environ.items() if k.startswith("PACT_")),
        "target": str(getattr(swapper.baseline.metadata, "target", "")),
    }
    if remaining_steps is not None:
        material["remaining_bucket"] = _remaining_steps_bucket(remaining_steps)
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

    def _rejected(self, key: str, reason: str) -> Dict[str, Any]:
        plan = {"triggered": False, "swapped": False, "rolled_back": False,
                "reason": reason, "context_key": key}
        try:
            self.db.put(key, {"decision": "rejected", **plan})
        except Exception:
            pass
        self.last_plan = plan
        return plan

    def evaluate(self, key: str, swapper: PactKernelSwapper,
                 remaining_steps: int, measure_iters: int = 20,
                 force: bool = False) -> Dict[str, Any]:
        """Run the full gated pipeline.  `key` should embed the context_key.

        Any exception inside the gates (collect, compile, measure, swap or
        rollback) is converted into a persisted `rejected` decision and a
        non-triggered plan: the PGO layer must never propagate an exception to
        the vLLM/benchmark harness.
        """
        try:
            return self._evaluate(key, swapper, remaining_steps,
                                  measure_iters, force)
        except Exception as e:
            plan = {"triggered": False, "swapped": False, "rolled_back": False,
                    "reason": f"pgo error fallback: {e}", "context_key": key}
            try:
                self.db.put(key, {"decision": "rejected", **plan})
            except Exception:
                pass
            self.last_plan = plan
            return plan

    def _evaluate(self, key: str, swapper: PactKernelSwapper,
                  remaining_steps: int, measure_iters: int = 20,
                  force: bool = False) -> Dict[str, Any]:
        prior = self.db.get(key)
        if not force and prior.get("decision") in ("rejected", "swapped",
                                                   "rolled_back"):
            # G1 context gate: same context never re-runs.  One exception:
            # an amortization rejection may be retried after
            # reprobe_after_steps evaluate() calls (each call corresponds to a
            # harness prepare event), because the remaining decode length may
            # have changed enough to cover the compile cost.
            if (prior.get("decision") == "rejected" and
                    prior.get("amortization_rejected") and
                    self.reprobe_after_steps > 0):
                seen = int(prior.get("probe_skips", 0)) + 1
                updated = dict(prior, probe_skips=seen)
                self.db.put(key, updated)
                if seen < self.reprobe_after_steps:
                    return {**updated, "triggered": False,
                            "reason": (f"context gate: amortization rejection "
                                       f"(reprobe {seen}/"
                                       f"{self.reprobe_after_steps})"),
                            "context_key": key}
                # Reached the reprobe limit: fall through and re-run the gates.
            else:
                return {**prior, "triggered": False,
                        "reason": f"context gate: {prior.get('decision')}",
                        "context_key": key}

        # Total PGO overhead for the amortization gate: profile collection
        # (subprocess compile + instrumented run) plus every variant compile.
        # The gate may only count cost it actually incurred.
        overhead_wall_s = 0.0

        t0 = time.monotonic()
        try:
            facts = self.collector.collect(steps=1)
        except Exception as e:
            return self._rejected(key, f"collect failed: {e}")
        overhead_wall_s += time.monotonic() - t0
        try:
            facts.update(probe_active_warp_permille())
        except Exception as e:
            facts["active_warp_ratio_unavailable"] = f"probe raised: {e}"
        hints = facts_to_hints(facts)
        self.db.put(key, facts)

        t0 = time.monotonic()
        try:
            candidate = swapper.compile_candidate(
                hints, str(self.hints_dir / f"{key}.json"))
        except Exception as e:
            return self._rejected(key, f"compile failed: {e}")
        candidate_compile_seconds = time.monotonic() - t0
        overhead_wall_s += candidate_compile_seconds

        # G2 theory trigger, computed in C++ and returned through metadata.
        md = candidate.metadata
        md_dict = md._asdict() if hasattr(md, "_asdict") else dict(md)
        trigger = int(md_dict.get("pact_pgo_trigger", 0) or 0)
        trigger_reason = md_dict.get("pact_pgo_trigger_reason")
        if not force and not trigger:
            plan = {"triggered": False, "reason": f"theory gate: {trigger_reason}",
                    "trigger_reason": trigger_reason, "context_key": key}
            self.db.put(key, {"decision": "rejected", **plan})
            self.last_plan = plan
            return plan

        # S3b: theory candidate + explicit 2-warp candidate (P3/M2 retained,
        # P6/P11 disabled so they cannot override the explicit choice).
        # S4: stage_down at the theory warp count is added because the v3
        # oracle evidence gate (s4_stage_candidates_v3.json) showed it improves
        # >=5% on 3/7 shapes with no >5% regression.  stage_up failed the same
        # gate and is deliberately not compiled.
        theory_stages = int(md_dict.get("pact_optimal_num_stages", 3) or 3)
        theory_warps = int(md_dict.get("num_warps", 4) or 4)
        low_warp = None
        t0 = time.monotonic()
        try:
            low_warp = swapper.compile_explicit(num_stages=theory_stages,
                                                num_warps=2)
        except Exception:
            low_warp = None
        overhead_wall_s += time.monotonic() - t0

        stage_down = None
        t0 = time.monotonic()
        try:
            stage_down = swapper.compile_explicit(
                num_stages=max(2, theory_stages - 1), num_warps=theory_warps)
        except Exception:
            stage_down = None
        overhead_wall_s += time.monotonic() - t0

        variants = {"theory": candidate, "low_warp": low_warp,
                    "stage_down": stage_down}
        us = {"baseline": _measure(swapper, swapper.baseline, measure_iters)}
        for name, kernel in variants.items():
            if kernel is not None:
                us[name] = _measure(swapper, kernel, measure_iters)

        chosen_name = min((k for k, v in us.items() if k != "baseline"),
                          key=lambda k: us[k])
        chosen = variants[chosen_name]
        gain_us = us["baseline"] - us[chosen_name]
        gain_percent = 100.0 * gain_us / max(us["baseline"], 1e-6)
        compile_cost_us = 1.5 * overhead_wall_s * 1e6
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
            "compile_seconds": candidate_compile_seconds,
            "overhead_wall_seconds": overhead_wall_s,
            "context_key": key,
        }
        if not passed:
            reason = ("gain <= 0" if gain_us <= 0 else
                      f"gain {gain_percent:.1f}% < {self.min_gain_percent}%"
                      if gain_percent < self.min_gain_percent else
                      "predicted gain does not cover compile cost")
            plan.update({"swapped": False, "reason": reason})
            self.db.put(key, {"decision": "rejected", **plan,
                              "amortization_rejected":
                                  reason.startswith("predicted gain does not")})
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
