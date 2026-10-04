"""Offline family table: (B,S,occ,stall) buckets -> family vocabulary.

v8: ``vanilla`` is a first-class family value -- the decider maps it to
``PACT_ENABLE=0`` (no recompile advantage, keep the default kernel).  Keys may
carry a ``d{D}g{gqa}|`` prefix; lookup prefers the prefixed entry when the
caller supplies head_dim/gqa and falls back to the unprefixed grammar.  In v7
the fit emitted prefixed keys that no runtime caller could address (dead
entries); the prefix is now part of the documented grammar.

V14-B: the vocabulary adds ``cold`` (S1 auto family: lifts the stage cap,
P6's L2-residency gate itself keeps hot shapes at the theory decision).
The aobo inline path can emit it today via its cold-aware reselection;
table/learned entries for it land with a future re-fit that measures
the cold variant directly.

V21-D (0a/0c mechanism C): the vocabulary adds the NAMED VARIANT arms
``p1rt`` (family A runtime page) and ``p1g`` (family B gather contig) —
a v3 table/learned model selects them per cell and the decider injects
their env via the learned-policy variants dict (zero C++; per-cell
default-on is expressed table-side, never as a global env flip).
"""
from __future__ import annotations

import json
import os
from typing import Dict, Optional

from triton.pact.runtime.workload_sniffer import bucket_bs

KNOWN_FAMILIES = ("theory", "occupancy", "latency", "vanilla",
                  "deep", "short", "w1", "cold",
                  # V21-D named variants (env-dict arms, see docstring)
                  "p1rt", "p1g",
                  # V21 model-vs-table experiment finding: the campaign
                  # TRAINING arms (stage/warp presets) were missing, so a
                  # distilled v3 table had those entries silently dropped
                  # by from_dict (lookup fell to default, agreement ~0.03
                  # vs its own source model) — the vocab must cover every
                  # deployable arm name the trainer can emit.
                  "s2", "s5", "w2",
                  # V22 1-1 kwargs/options dual-channel arms (PLAN_V22
                  # 阶段一；same vocab rule: every arm name the trainer
                  # or a deployed v3 model can emit must be known here).
                  # s3 = PACT_OVERRIDE_STAGES=3 constant arm (the
                  # num_stages OPTIONS channel is taken over by P6 under
                  # PACT_ENABLE=1, so constants pin via env — 20261004
                  # channel verification).
                  "blo16_swi8", "bloNone_evict", "s3",
                  # V22 R1 winner arms (3-1 free search: large-batch D128
                  # cells are won by DISABLING auto-stages or pinning V;
                  # w8 = the warps half of the D64/GQA4 winners — the
                  # swizzle/evict half is naive-kernel-only, production
                  # kernel has no kwargs face)
                  "w8", "p6off", "v8")


def _occ_bucket(permille: Optional[int]) -> str:
    if permille is None:
        return "occ_unk"
    if permille < 400:
        return "occ_lo"
    if permille < 700:
        return "occ_mid"
    return "occ_hi"


def _stall_bucket(permille: Optional[int]) -> str:
    if permille is None:
        return "stall_unk"
    if permille < 200:
        return "stall_lo"
    if permille < 400:
        return "stall_mid"
    return "stall_hi"


def make_key(batch: int, seq_len: int, occ_permille: Optional[int],
             stall_permille: Optional[int], head_dim: Optional[int] = None,
             gqa: Optional[int] = None) -> str:
    b, s = bucket_bs(batch, seq_len)
    key = f"{b}|{s}|{_occ_bucket(occ_permille)}|{_stall_bucket(stall_permille)}"
    if head_dim and gqa:
        key = f"d{int(head_dim)}g{int(gqa)}|{key}"
    return key


class FamilyTable:
    def __init__(self, enabled: bool, entries: Dict[str, str],
                 default: str = "theory"):
        self.enabled = enabled
        # Drop malformed entries loudly instead of carrying dead weight:
        # a value outside KNOWN_FAMILIES can never be acted on by the decider.
        dropped = {k: v for k, v in entries.items()
                   if not (isinstance(v, str) and v in KNOWN_FAMILIES)}
        self.entries = {k: v for k, v in entries.items() if k not in dropped}
        self.dropped = dropped
        # V21-D (F13, BR-17 "counted, never silent"): a dropped entry is
        # a vocabulary miss the decider can NEVER act on — one visible
        # line at load, so a stale table fails loudly upstream instead
        # of silently narrowing the action space.
        if dropped:
            print(f"[PACT FamilyTable] dropped {len(dropped)} entries "
                  f"outside KNOWN_FAMILIES: "
                  f"{sorted(dropped.values())[:6]}", flush=True)
        self.default = default if default in KNOWN_FAMILIES else "theory"

    @classmethod
    def from_dict(cls, data: Dict) -> "FamilyTable":
        return cls(bool(data.get("enabled", False)),
                   dict(data.get("entries") or {}),
                   str(data.get("default", "theory")))

    @classmethod
    def load(cls, path: Optional[str] = None) -> "FamilyTable":
        path = path or os.environ.get(
            "PACT_FAMILY_TABLE",
            os.path.join(os.environ.get("PACT_EVAL_DIR", ""), "offline",
                         "family_table.json"))
        if not path or not os.path.isfile(path):
            return cls(enabled=False, entries={}, default="theory")
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def lookup(self, batch: int, seq_len: int,
               occ_permille: Optional[int] = None,
               stall_permille: Optional[int] = None,
               head_dim: Optional[int] = None,
               gqa: Optional[int] = None) -> str:
        if not self.enabled:
            return self.default
        # Prefixed entries win when the caller knows (D, GQA).
        if head_dim and gqa:
            pref = f"d{int(head_dim)}g{int(gqa)}|"
            for cand in self._candidates(batch, seq_len, occ_permille,
                                         stall_permille):
                if (pref + cand) in self.entries:
                    return self.entries[pref + cand]
        for cand in self._candidates(batch, seq_len, occ_permille,
                                     stall_permille):
            if cand in self.entries:
                return self.entries[cand]
        return self.default

    @staticmethod
    def _candidates(batch: int, seq_len: int, occ_permille: Optional[int],
                    stall_permille: Optional[int]):
        b, s = bucket_bs(batch, seq_len)
        # exact, then relax occ/stall buckets to unknown
        yield f"{b}|{s}|{_occ_bucket(occ_permille)}|{_stall_bucket(stall_permille)}"
        yield f"{b}|{s}|{_occ_bucket(occ_permille)}|stall_unk"
        yield f"{b}|{s}|occ_unk|{_stall_bucket(stall_permille)}"
        yield f"{b}|{s}|occ_unk|stall_unk"
