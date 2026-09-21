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
    options_override: Dict[str, Any] = {}
    if family == "vanilla":
        extra_env = {"PACT_ENABLE": "0"}
        return {
            "family": family,
            "extra_env": extra_env,
            "options_override": options_override,
            "hints": {},
        }
    extra_env = {
        "PACT_ENABLE": "1",
        "PACT_ENABLE_AUTO_NUM_STAGES": "1",
        "PACT_ENABLE_AUTO_NUM_WARPS": "1",
    }
    if family == "occupancy":
        extra_env["PACT_OVERRIDE_WARPS"] = "2"
        options_override["num_warps"] = 2
    elif family == "latency":
        extra_env["PACT_OVERRIDE_STAGES"] = "2"
    elif family == "deep":
        extra_env["PACT_OVERRIDE_STAGES"] = "5"
    elif family == "short":
        extra_env["PACT_OVERRIDE_STAGES"] = "1"
    elif family == "w1":
        extra_env["PACT_OVERRIDE_WARPS"] = "1"
        options_override["num_warps"] = 1
    elif family == "cold":
        # S1 auto family: unlock deep pipelines; the C++ residency gate
        # decides per shape (hot shapes compile identically to theory).
        extra_env["PACT_MAX_PIPELINE_STAGES"] = "5"
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
