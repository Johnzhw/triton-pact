"""Post-hoc SASS footprint audit for compiled PACT variants (V14-A2).

Motivation (arXiv 2609.18662, "Automated Instruction Encoding Synthesis for
Modern GPU ISA Compression"): static SASS footprints of aggressively fused
kernels collide with the GPU front-end instruction-cache thresholds
(~32-128KiB, measured on B200/H100/4090) and cause CPI jumps.  PACT cannot
re-encode SASS on fixed silicon — the paper's CP-SAT slot assignment needs
a new decoder — but it can do the software half of the paper's
footprint-budget idea: *measure* every variant's static footprint and
refuse to select a variant whose code exceeds the target's instruction
budget.

Contract (pollution-free by construction):
- post-hoc only: disassembles an already-compiled cubin.  This module is
  never imported on the compile path, never touches codegen, and nothing
  here enters the JIT cache key (it reads bytes that were already
  produced);
- toolchain: the triton-bundled nvdisasm (same tool family as the ptxas
  that produced the cubin), falling back to PATH;
- verification methodology follows the paper's dual-path discipline: the
  instruction count and the address-span/16-byte-container figure must
  agree on sm<=90 (fixed 128-bit containers from Volta through Hopper);
  a disagreement is reported, not papered over.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Optional

# Instruction line, e.g. "        /*0060*/                   LOP3.LUT R0, ... ;"
_INSTR_RE = re.compile(r"^\s*/\*([0-9a-fA-F]+)\*/\s+(@?!?[A-Z0-9][A-Z0-9.]*)")

# Fixed 128-bit instruction containers from Volta through Hopper; Blackwell
# relaxes this (the encoding paper's variable-length motivation), so the
# address span stays the authoritative byte figure and the count x16 is the
# cross-check.
_BYTES_PER_INSTR = 16

# L0 instruction-cache budget per SM sub-partition, family-minimum
# convention (GA10x whitepaper: 16KiB per partition; Hopper: 32KiB).  These
# are vetoes, not targets: the decode-attention family measures in the low
# single-digit KiB, far below the budget — the gate exists so a future
# megakernel-shaped variant cannot be selected silently.
L0I_BUDGET_BYTES = {70: 16 * 1024, 80: 16 * 1024, 86: 16 * 1024,
                    89: 16 * 1024, 90: 32 * 1024, 100: 32 * 1024}


def _nvdisasm() -> str:
    cand = (Path(__file__).resolve().parents[2] / "backends" / "nvidia" /
            "bin" / "nvdisasm")
    if cand.exists():
        return str(cand)
    return shutil.which("nvdisasm") or "nvdisasm"


def parse_sass(text: str) -> Dict:
    """Count the static footprint of one disassembled kernel body."""
    n = 0
    last_off = -1
    hist: Dict[str, int] = {}
    mov32i = 0
    for line in text.splitlines():
        m = _INSTR_RE.match(line)
        if not m:
            continue
        off = int(m.group(1), 16)
        op = m.group(2).lstrip("@!")
        n += 1
        if off > last_off:
            last_off = off
        base = op.split(".")[0]
        hist[base] = hist.get(base, 0) + 1
        if base == "MOV32I":
            mov32i += 1
    span = last_off + _BYTES_PER_INSTR if n else 0
    cross_ok = (span == n * _BYTES_PER_INSTR)
    return {
        "ok": True,
        "n_instrs": n,
        "span_bytes": span,
        "count_x16_bytes": n * _BYTES_PER_INSTR,
        "container_crosscheck_ok": cross_ok,
        "mov32i": mov32i,
        "top_opcodes": dict(sorted(hist.items(), key=lambda kv: -kv[1])[:8]),
    }


def sass_footprint(cubin: bytes, timeout_s: float = 30.0) -> Dict:
    """Disassemble one cubin and return its static SASS footprint.

    Returns {"ok": False, "error": ...} on tool failure; callers treat a
    failed audit as "unknown" (never a veto, never a fabrication).
    """
    fd, path = tempfile.mkstemp(suffix=".cubin")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(cubin)
        out = subprocess.run([_nvdisasm(), "-c", path], capture_output=True,
                             text=True, timeout=timeout_s)
        if out.returncode != 0:
            return {"ok": False, "error": (out.stderr or "").strip()[:200]}
        return parse_sass(out.stdout)
    finally:
        os.unlink(path)


def budget_for(sm_version: int) -> int:
    best = 0
    for k in sorted(L0I_BUDGET_BYTES):
        if sm_version >= k:
            best = L0I_BUDGET_BYTES[k]
    return best


def within_budget(footprint: Optional[Dict], sm_version: int) -> bool:
    """Footprint budget veto.  Unknown/failed audits never veto."""
    if not footprint or not footprint.get("ok"):
        return True
    b = budget_for(sm_version)
    return b <= 0 or footprint.get("span_bytes", 0) <= b
