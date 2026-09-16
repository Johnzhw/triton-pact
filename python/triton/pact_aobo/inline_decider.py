"""In-process decision: no socket, no service thread.

Reuses the pgo deciders (learned policy first when PACT_FAMILY_MODEL is set,
family-table/online_decider otherwise).  The result is a variant NAME in the
pool's vocabulary plus the extra_env needed to compile it — the caller
decides *when* to compile (async_switch does it off the forward path).
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

# Pool vocabulary: name -> extra_env for the pact target family.
VARIANTS: Dict[str, Dict[str, str]] = {
    "baseline": {"PACT_ENABLE": "0"},
    "theory": {"PACT_ENABLE": "1"},
    "occ": {"PACT_ENABLE": "1", "PACT_OVERRIDE_WARPS": "2"},
    "lat": {"PACT_ENABLE": "1", "PACT_OVERRIDE_STAGES": "2"},
    "deep": {"PACT_ENABLE": "1", "PACT_OVERRIDE_STAGES": "5"},
}


def decide_inline(facts: Optional[Dict[str, Any]], batch: int,
                  seq_len: int,
                  geometry: Optional[Dict[str, int]] = None
                  ) -> Tuple[str, Dict[str, str]]:
    """Return (variant_name, extra_env). Never raises past the fallbacks."""
    facts = dict(facts or {})
    if geometry:
        facts.setdefault("head_dim", geometry.get("D"))
        facts.setdefault("gqa", geometry.get("GQA"))
    try:
        from triton.pact.decider import learned_policy as lp
        cfg = dict(facts)
        if geometry:
            cfg.update({k: geometry[k] for k in ("S", "D", "P", "B")
                        if k in geometry})
        resp = lp.decide(facts, batch, seq_len, cfg=cfg)
        fam = (resp or {}).get("family") or "theory"
    except Exception:
        fam = os.environ.get("PACT_AOBO_DEFAULT_FAMILY", "theory")
    if fam not in VARIANTS:
        fam = "theory"
    return fam, dict(VARIANTS[fam])
