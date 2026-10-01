"""(B, S) bucketing for the inference-process trigger.  No Proton required."""
from __future__ import annotations

import os
from typing import Optional, Tuple


def _cap_bucket(value: int, caps) -> str:
    for cap in caps:
        if value <= cap:
            return f"le{cap}"
    return f"gt{caps[-1]}"


def bucket_bs(batch: int, seq_len: int) -> Tuple[str, str]:
    """Coarse buckets so nearby shapes share a compiled variant."""
    b = _cap_bucket(int(batch), (1, 2, 4, 8, 16, 32, 64))
    s = _cap_bucket(int(seq_len), (128, 256, 512, 1024, 2048, 4096, 8192))
    return b, s


def should_trigger(prev: Optional[Tuple[str, str]],
                   batch: int, seq_len: int) -> Tuple[bool, Tuple[str, str]]:
    cur = bucket_bs(batch, seq_len)
    return (prev is None or cur != prev), cur


# V19 N9 (PACT_BUCKET_HYSTERESIS=1, default off): bucket-boundary
# hysteresis against switch oscillation.  A NEW bucket must persist
# K consecutive observations before the trigger fires; any return to the
# incumbent resets the window (a single boundary excursion never
# switches).  The +/-10% physical band stays the e2e judgement layer
# (unchanged); this is the trigger-side damper.
_HYST_STREAK = 4


class BucketHysteresis:
    """Stateful damper for the (B,S) bucket trigger.  Env-gated
    (PACT_BUCKET_HYSTERESIS=1); with the env off callers keep using the
    frozen stateless should_trigger and this class is never constructed."""

    def __init__(self, streak: int = _HYST_STREAK):
        self.streak = max(1, int(streak))
        self._cur: Optional[Tuple[str, str]] = None
        self._cand: Optional[Tuple[str, str]] = None
        self._n = 0
        self.switches_damped = 0

    def observe(self, batch: int, seq_len: int
                ) -> Tuple[bool, Tuple[str, str]]:
        """One observation; returns (trigger_fired, effective_bucket)."""
        b = bucket_bs(batch, seq_len)
        if self._cur is None:
            self._cur = b
            return True, b
        if b == self._cur:
            self._cand, self._n = None, 0
            return False, self._cur
        if b == self._cand:
            self._n += 1
        else:
            self._cand, self._n = b, 1
        if self._n >= self.streak:
            self._cur, self._cand, self._n = b, None, 0
            return True, b
        self.switches_damped += 1
        return False, self._cur


def gated_trigger(prev: Optional[Tuple[str, str]], batch: int, seq_len: int,
                  state: Optional[BucketHysteresis] = None
                  ) -> Tuple[bool, Tuple[str, str]]:
    """The env-aware entry point: hysteresis only when
    PACT_BUCKET_HYSTERESIS=1 (with the caller-owned state object);
    otherwise the frozen stateless path, bit-for-bit."""
    if os.environ.get("PACT_BUCKET_HYSTERESIS") == "1" and \
            state is not None:
        return state.observe(batch, seq_len)
    return should_trigger(prev, batch, seq_len)
