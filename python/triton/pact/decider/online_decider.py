"""Facts + family table -> one extra_env for a single JIT."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from triton.pact.decider.family_table import FamilyTable


def facts_to_hints(facts: Dict[str, Any]) -> Dict[str, int]:
    hints: Dict[str, int] = {}
    mapping = {
        "measured_iterations": "pact.hw.measured_iterations",
        "regs_per_thread": "pact.hw.regs_per_thread",
        "active_warp_ratio_permille": "pact.hw.active_warp_ratio_permille",
        "stall_memory_permille": "pact.hw.stall_memory_permille",
        "sm_efficiency_permille": "pact.hw.sm_efficiency_permille",
        # V14-B: scales the S1 L2-residency working set to the sequence's
        # full K+V stream (paged loads are per-KV-head tiles; the IR alone
        # cannot see how many heads stream through L2).
        "kv_heads": "pact.hw.kv_heads",
    }
    for src, dst in mapping.items():
        val = facts.get(src)
        if val is None:
            continue
        try:
            iv = int(val)
        except (TypeError, ValueError):
            continue
        if iv > 0:
            hints[dst] = iv
    return hints


def _family_env(family: str) -> tuple:
    """Family name -> (extra_env additions, options_override).

    V16-T3: 'short' used to pin PACT_OVERRIDE_STAGES=1, which P6's
    [2,8] validity domain silently REJECTS (AutoNumStages.cpp pins are
    clamped away; the reported config said 1 while the binary was P6's
    own pick) -- a fake config since day one.  The intent "shallowest
    pipeline" is pin 2, fixed here; every other family is unchanged.
    """
    extra_env: Dict[str, str] = {}
    options_override: Dict[str, Any] = {}
    if family == "occupancy":
        extra_env["PACT_OVERRIDE_WARPS"] = "2"
        options_override["num_warps"] = 2
    elif family == "latency":
        extra_env["PACT_OVERRIDE_STAGES"] = "2"
    elif family == "deep":
        extra_env["PACT_OVERRIDE_STAGES"] = "5"
    elif family == "short":
        extra_env["PACT_OVERRIDE_STAGES"] = "2"
    elif family == "w1":
        extra_env["PACT_OVERRIDE_WARPS"] = "1"
        options_override["num_warps"] = 1
    elif family == "cold":
        # S1 auto family: unlock deep pipelines; the C++ residency gate
        # decides per shape (hot shapes compile identically to theory).
        extra_env["PACT_MAX_PIPELINE_STAGES"] = "5"
    return extra_env, options_override


_WARPS_DOMAIN = (1, 2, 4, 8)

# ---- V16-T7a: runtime-truth-informed family adjustment ------------------
# The four CUPTI permilles (V16-T0 truths, see cupti_collector) describe
# the CURRENT kernel -- measured on the baseline replica inside the
# trigger cycle -- and steer the NEXT variant choice.  This is the
# closed feedback loop the paper narrative promises: hardware counters
# -> decider -> parametric compile.  Thresholds anchored on V16-T0
# measured points (memory-bound streaming/decode shapes: warp 700-840,
# stall 2300-2900, sm 2-18, l2 380-500; resident-friendly matmuls:
# warp 166, stall 2, sm 109, l2 893).  PACT_COUNTER_DECIDE=0 disables.
_COUNTER_THRESH = {
    "stall_heavy": 1500,     # permille of per-issue warps-stalled ratio x10
    "l2_missy": 400,         # l2 sector hit rate permille below this = cold
    "warp_rich": 800,        # warps resident but (see sm_starved) issue-poor
    "sm_starved": 150,       # sm throughput permille
    "stall_light": 300,
    "l2_resident": 700,
}

# V18 D-1 (PACT_COUNTER_V2, default OFF): the dram-stream proxy is the
# L2 MISS rate in permille (1000 - l2_hit), inverted per PLAN D-1.  The
# v2 path turns the three independent v1 rules into a 2-D quadrant
# decision on (dram_stream x stall) and adds the bandwidth-bound
# quadrant the v1 rules could never reach (dram high with issue
# headroom).  Activation requires the env; the v1 path stays
# bit-for-bit frozen.
_COUNTER_THRESH_V2 = {
    "dram_stream_hi": 300,   # miss-rate proxy permille (l2_hit < 700)
}


def counter_adjust(family: str, facts: Dict[str, Any],
                     thresholds: Optional[Dict[str, int]] = None
                     ) -> Dict[str, Any]:
    """Returns {'family': adjusted, 'counter_adjusted': {...}|None}.
    Conservative: only ever moves theory/vanilla (the defaults); a fit
    table's explicit winner is never overridden.  Missing counters ->
    no-op."""
    def _p(k):
        v = facts.get(k)
        try:
            return int(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    stall = _p("stall_memory_permille")
    warp = _p("active_warp_ratio_permille")
    sm = _p("sm_efficiency_permille")
    l2 = _p("l2_hit_permille")
    if stall is None or l2 is None:
        return {"family": family, "counter_adjusted": None}
    # V17 S1-2: optional threshold override (calibration grid); the
    # default path is bit-for-bit the frozen constants
    th = dict(_COUNTER_THRESH)
    if thresholds:
        th.update(thresholds)
    if family in ("theory", "vanilla") and \
            os.environ.get("PACT_COUNTER_V2", "0") == "1":
        # V18 D-1: 2-D (dram-stream x stall) quadrants, v2 semantics
        dram = 1000 - (l2 if l2 is not None else 1000)
        if dram >= _COUNTER_THRESH_V2["dram_stream_hi"]:
            if stall >= th["stall_heavy"]:
                return {"family": "cold",
                        "counter_adjusted": {"from": family, "to": "cold",
                                             "reason": f"v2 dram={dram} "
                                             f"stall={stall} -> deepen"}}
            # bandwidth-bound with issue headroom: the v1 chain never
            # reached this quadrant (it required stall to be heavy too)
            return {"family": "occ",
                    "counter_adjusted": {"from": family, "to": "occ",
                                         "reason": f"v2 dram={dram} "
                                         f"stall={stall} -> fewer warps"}}
        if l2 is not None and l2 >= th["l2_resident"] and \
                stall <= th["stall_light"]:
            return {"family": "lat",
                    "counter_adjusted": {"from": family, "to": "lat",
                                         "reason": f"v2 l2={l2} stall="
                                         f"{stall} -> shallow"}}
        return {"family": family, "counter_adjusted": None}
    if family in ("theory", "vanilla"):
        if stall >= th["stall_heavy"] and l2 <= th["l2_missy"]:
            # long-scoreboard-bound with a cold L2: deepen the pipeline
            # range and let P6's residency gate pick the depth
            return {"family": "cold",
                    "counter_adjusted": {"from": family, "to": "cold",
                                         "reason": f"stall={stall} "
                                         f"l2={l2} -> deepen"}}
        if warp is not None and sm is not None and \
                warp >= th["warp_rich"] and sm <= th["sm_starved"]:
            # warps resident but issue-starved: try fewer warps per CTA
            return {"family": "occ",
                    "counter_adjusted": {"from": family, "to": "occ",
                                         "reason": f"warp={warp} rich "
                                         f"sm={sm} starved -> fewer warps"}}
        if stall <= th["stall_light"] and l2 >= th["l2_resident"]:
            # latency regime, cache-resident: shallow pipeline wins
            return {"family": "lat",
                    "counter_adjusted": {"from": family, "to": "lat",
                                         "reason": f"stall={stall} light "
                                         f"l2={l2} resident -> shallow"}}
    return {"family": family, "counter_adjusted": None}


def _counter_gate_on() -> bool:
    return os.environ.get("PACT_COUNTER_DECIDE", "1") != "0"


def normalize_preset(spec) -> Dict[str, Any]:
    """V16-T3 generative-preset entry: a family NAME (compat) or a
    parameterized dict -> the same shape decide() returns for envs.

    Dict keys (all optional, validated/clamped, values outside the
    domains are dropped with the rest kept):
      stages     int in [2, 8]  -> PACT_OVERRIDE_STAGES
      warps      int in {1,2,4,8} -> PACT_OVERRIDE_WARPS + num_warps
      max_stages int in [2, 8]  -> PACT_MAX_PIPELINE_STAGES (cold-style)
      enable     bool           -> PACT_ENABLE (default 1)
    This is what lets deciders emit OUT-OF-VOCABULARY variants (user
    point 2: no fixed candidate cap) while the named families remain
    the compatibility anchors.
    """
    if isinstance(spec, str):
        extra_env = {"PACT_ENABLE": "0"} if spec == "vanilla" else {
            "PACT_ENABLE": "1",
            "PACT_ENABLE_AUTO_NUM_STAGES": "1",
            "PACT_ENABLE_AUTO_NUM_WARPS": "1",
        }
        add, opt = _family_env(spec) if spec != "vanilla" else ({}, {})
        extra_env.update(add)
        return {"family": spec, "extra_env": extra_env,
                "options_override": opt}
    if not isinstance(spec, dict):
        raise TypeError(f"preset must be str or dict, got {type(spec)}")
    enable = bool(spec.get("enable", True))
    extra_env: Dict[str, str] = {"PACT_ENABLE": "1" if enable else "0"}
    options_override: Dict[str, Any] = {}
    if not enable:
        return {"family": spec.get("name", "vanilla-preset"),
                "extra_env": extra_env, "options_override": options_override}
    extra_env["PACT_ENABLE_AUTO_NUM_STAGES"] = "1"
    extra_env["PACT_ENABLE_AUTO_NUM_WARPS"] = "1"
    stages = spec.get("stages")
    if isinstance(stages, int) and 2 <= stages <= 8:
        extra_env["PACT_OVERRIDE_STAGES"] = str(stages)
    warps = spec.get("warps")
    if isinstance(warps, int) and warps in _WARPS_DOMAIN:
        extra_env["PACT_OVERRIDE_WARPS"] = str(warps)
        options_override["num_warps"] = warps
    max_stages = spec.get("max_stages")
    if isinstance(max_stages, int) and 2 <= max_stages <= 8:
        extra_env["PACT_MAX_PIPELINE_STAGES"] = str(max_stages)
    name = spec.get("name") or "s{}w{}".format(
        extra_env.get("PACT_OVERRIDE_STAGES", "a"),
        extra_env.get("PACT_OVERRIDE_WARPS", "a"))
    return {"family": name, "extra_env": extra_env,
            "options_override": options_override}


def decide(facts: Dict[str, Any], batch: int, seq_len: int,
           table: Optional[FamilyTable] = None,
           hints_dir: Optional[Path] = None,
           head_dim: Optional[int] = None,
           gqa: Optional[int] = None,
           kv_heads: Optional[int] = None) -> Dict[str, Any]:
    """Facts + family table -> one extra_env for a single JIT.

    v8: the table may return ``"vanilla"`` (the fit's vanilla-anchored winner
    for buckets where no PACT variant beats vanilla by the 5% gate).  The
    vanilla family compiles the untouched default kernel -- there is no
    advantage in recompiling a PACT variant that measured slower.  Callers
    that know the kernel geometry pass head_dim/gqa so the d{D}g{gqa}|
    prefixed entries are addressable.

    V14-B: ``kv_heads`` (derivable as Hq/GQA from the geometry) enters the
    S1 residency hints; the ``cold`` family just lifts the stage cap —
    P6's L2-residency gate itself keeps hot shapes at the theory decision.
    """
    table = table or FamilyTable.load()
    family = table.lookup(
        batch, seq_len,
        facts.get("active_warp_ratio_permille"),
        facts.get("stall_memory_permille"),
        head_dim=head_dim, gqa=gqa,
    )
    counter_note = None
    if _counter_gate_on():
        adj = counter_adjust(family, facts)
        family, counter_note = adj["family"], adj["counter_adjusted"]
    preset = normalize_preset(family)
    extra_env: Dict[str, str] = preset["extra_env"]
    options_override: Dict[str, Any] = preset["options_override"]
    if family == "vanilla":
        return {
            "family": family,
            "extra_env": extra_env,
            "options_override": options_override,
            "hints": {},
            "counter_adjusted": counter_note,
        }
    if kv_heads and int(kv_heads) > 0:
        facts = {**facts, "kv_heads": int(kv_heads)}
    hints = facts_to_hints(facts)
    if hints:
        d = Path(hints_dir or (Path.home() / ".triton" / "pact_hw"))
        d.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(
            json.dumps(hints, sort_keys=True).encode()).hexdigest()[:12]
        path = d / f"hints_{digest}.json"
        path.write_text(json.dumps(hints))
        extra_env["PACT_HW_HINTS_JSON"] = str(path)
    return {
        "family": family,
        "extra_env": extra_env,
        "options_override": options_override,
        "hints": hints,
        "counter_adjusted": counter_note,
    }
