"""Offline family table: (B,S,occ,stall) buckets -> {theory, occupancy, latency}."""
from __future__ import annotations

import json
import os
from typing import Dict, Optional

from triton.pact.runtime.workload_sniffer import bucket_bs


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
             stall_permille: Optional[int]) -> str:
    b, s = bucket_bs(batch, seq_len)
    return f"{b}|{s}|{_occ_bucket(occ_permille)}|{_stall_bucket(stall_permille)}"


class FamilyTable:
    def __init__(self, enabled: bool, entries: Dict[str, str], default: str = "theory"):
        self.enabled = enabled
        self.entries = dict(entries)
        self.default = default or "theory"

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
               stall_permille: Optional[int] = None) -> str:
        if not self.enabled:
            return self.default
        key = make_key(batch, seq_len, occ_permille, stall_permille)
        if key in self.entries:
            return self.entries[key]
        # nearest: drop occ/stall then seq then batch
        b, s = bucket_bs(batch, seq_len)
        for cand in (
            f"{b}|{s}|{_occ_bucket(occ_permille)}|{_stall_bucket(stall_permille)}",
            f"{b}|{s}|{_occ_bucket(occ_permille)}|stall_unk",
            f"{b}|{s}|occ_unk|{_stall_bucket(stall_permille)}",
            f"{b}|{s}|occ_unk|stall_unk",
        ):
            if cand in self.entries:
                return self.entries[cand]
        return self.default
