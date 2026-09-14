"""Offline family table: (B,S,occ,stall) buckets -> {theory, occupancy, latency, vanilla}.

v8: ``vanilla`` is a first-class family value -- the decider maps it to
``PACT_ENABLE=0`` (no recompile advantage, keep the default kernel).  Keys may
carry a ``d{D}g{gqa}|`` prefix; lookup prefers the prefixed entry when the
caller supplies head_dim/gqa and falls back to the unprefixed grammar.  In v7
the fit emitted prefixed keys that no runtime caller could address (dead
entries); the prefix is now part of the documented grammar.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Optional

from triton.pact.runtime.workload_sniffer import bucket_bs

KNOWN_FAMILIES = ("theory", "occupancy", "latency", "vanilla")


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
