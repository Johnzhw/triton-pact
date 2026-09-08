"""(B, S) bucketing for the inference-process trigger.  No Proton required."""
from __future__ import annotations

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
