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
    preset = normalize_preset(family)
    extra_env: Dict[str, str] = preset["extra_env"]
    options_override: Dict[str, Any] = preset["options_override"]
    if family == "vanilla":
        return {
            "family": family,
            "extra_env": extra_env,
            "options_override": options_override,
            "hints": {},
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
    }
