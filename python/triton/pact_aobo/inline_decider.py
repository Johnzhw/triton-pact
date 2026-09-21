"""In-process decision: no socket, no service thread.

Reuses the pgo deciders (learned policy first when PACT_FAMILY_MODEL is set,
family-table/online_decider otherwise).  The result is a variant NAME in the
pool's vocabulary plus the extra_env needed to compile it — the caller
decides *when* to compile (async_switch does it off the forward path).

V14-B (D4): S-bucket cold-aware reselection — when the sequence's K+V
working set exceeds the device L2 (the same inequality P6's S1 residency
gate uses), a plain 'theory' pick is upgraded to the 'cold' auto family
(lifts the stage cap; P6 itself keeps hot shapes at the theory decision).
Explicit measured families are never overridden.
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
    # V12-Q1: coverage for shapes where the original families regress
    "short": {"PACT_ENABLE": "1", "PACT_OVERRIDE_STAGES": "1"},
    "w1": {"PACT_ENABLE": "1", "PACT_OVERRIDE_WARPS": "1"},
    # V14-B (S1): unlock the deep-pipeline range; P6's L2-residency gate
    # decides per shape (hot shapes compile identically to theory).
    "cold": {"PACT_ENABLE": "1", "PACT_MAX_PIPELINE_STAGES": "5"},
}

_L2_BYTES: Optional[int] = None


def _l2_bytes() -> int:
    """Device L2 capacity (cached; 0 = unknown -> reselection disabled)."""
    global _L2_BYTES
    if _L2_BYTES is None:
        try:
            import torch
            _L2_BYTES = int(
                torch.cuda.get_device_properties(0).L2_cache_size)
        except Exception:
            _L2_BYTES = 0
    return _L2_BYTES


def _kv_cold(seq_len: int, geometry: Optional[Dict[str, int]],
             facts: Dict[str, Any]) -> bool:
    """Mirror of P6's S1 working-set test: K+V bytes of one sequence vs L2.

    kv_heads comes from the geometry (Hq//GQA) or the facts; default 1
    keeps pre-V14 shapes classified exactly as before.
    """
    l2 = _l2_bytes()
    if l2 <= 0 or not geometry:
        return False
    D = int(geometry.get("D") or 0)
    if D <= 0 or seq_len <= 0:
        return False
    kv_heads = 1
    try:
        hq, gqa = int(geometry.get("Hq") or 0), int(geometry.get("GQA") or 0)
        if hq > 0 and gqa > 0:
            kv_heads = hq // gqa
    except (TypeError, ValueError):
        pass
    if kv_heads <= 0:
        kv_heads = 1
    return seq_len * D * 2 * 2 * kv_heads > l2


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
    # D4 cold-aware reselection: unlock deep pipelines only when the S1
    # working set says the sequence streams from DRAM.
    if fam == "theory" and _kv_cold(seq_len, geometry, facts):
        fam = "cold"
    return fam, dict(VARIANTS[fam])
