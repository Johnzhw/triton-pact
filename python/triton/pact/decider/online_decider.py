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
           hints_dir: Optional[Path] = None) -> Dict[str, Any]:
    table = table or FamilyTable.load()
    family = table.lookup(
        batch, seq_len,
        facts.get("active_warp_ratio_permille"),
        facts.get("stall_memory_permille"),
    )
    extra_env = {
        "PACT_ENABLE": "1",
        "PACT_ENABLE_AUTO_NUM_STAGES": "1",
        "PACT_ENABLE_AUTO_NUM_WARPS": "1",
    }
    options_override: Dict[str, Any] = {}
    if family == "occupancy":
        extra_env["PACT_OVERRIDE_WARPS"] = "2"
        options_override["num_warps"] = 2
    elif family == "latency":
        extra_env["PACT_OVERRIDE_STAGES"] = "2"
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
