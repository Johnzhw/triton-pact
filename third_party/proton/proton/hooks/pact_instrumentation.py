"""
PACT Proton Instrumentation Hook
================================
Phase C (PGO branch): instruments paged-attention kernels for KPerfIR-style
profile-driven pass decisions.

Two complementary paths:
  1. instrument_ttgir_text(): textual TTGIR override — inserts proton.record
     start/end markers around paged loads, async copies, async waits, and dot
     ops, then the Triton override mechanism recompiles that TTGIR.
  2. PACTHook: kept as a registration placeholder for future pass-manager
     integration; do_patch() is intentionally a no-op in this branch.

Usage:
    from triton.profiler.hooks.pact_instrumentation import instrument_ttgir_text
    instrumented = instrument_ttgir_text(open(path).read())
    # place `instrumented` into the TRITON_OVERRIDE_DIR and compile with
    # TRITON_KERNEL_OVERRIDE=1 while proton.start(backend="instrumentation")
    # is active.
"""
import re
from typing import Dict, Optional, Any

from .hook import Hook
from ..mode import InstrumentationMode


class PACTInstrumentationMode(InstrumentationMode):
    """Mode placeholder kept for API compatibility with older hooks."""

    def __init__(self, options: Optional[Dict[str, str]] = None):
        super().__init__(options or {})
        self.overlap_data: Dict[int, Dict[str, float]] = {}
        self.kernel_count = 0

    def reset(self):
        self.overlap_data.clear()
        self.kernel_count = 0


_SCOPE_RULES = [
    (r"= tt\.load .*pact\.paged_load", "pact.load"),
    (r"ttg\.async_copy_global_to_local", "pact.async_copy"),
    (r"ttg\.async_wait", "pact.async_wait"),
    (r"= tt\.dot ", "pact.compute"),
]


def instrument_ttgir_text(ttgir: str) -> str:
    """Insert proton.record start/end markers around the interesting TTGIR ops.

    Scope names are made unique per textual op.  Inside loops the same
    proton.record op is executed once per iteration; the runtime records one
    event per warp per iteration under the same scope id.
    """
    lines = ttgir.splitlines()
    out = []
    counters: Dict[str, int] = {}
    for line in lines:
        matched = None
        for pattern, base_name in _SCOPE_RULES:
            if re.search(pattern, line):
                matched = base_name
                break
        if matched is None:
            out.append(line)
            continue

        idx = counters.get(matched, 0)
        counters[matched] = idx + 1
        scope = f"{matched}_{idx}"
        indent = line[: len(line) - len(line.lstrip())]
        out.append(f'{indent}proton.record start "{scope}"')
        out.append(line)
        out.append(f'{indent}proton.record end "{scope}"')
    return "\n".join(out) + "\n"


class PACTHook(Hook):
    """Placeholder hook.  Textual override is the supported integration path."""

    def __init__(self):
        super().__init__()
        self.mode_impl: Optional[PACTInstrumentationMode] = None

    def prepare_mode(self, mode_obj: Any) -> PACTInstrumentationMode:
        if isinstance(mode_obj, PACTInstrumentationMode):
            return mode_obj
        return PACTInstrumentationMode()

    def do_patch(self, ir: str, pm, context):
        # Pass-manager insertion is intentionally not implemented in this
        # branch; instrument_ttgir_text() + TRITON_KERNEL_OVERRIDE is used.
        return

    def post_process(self, data, output_format: str = ""):
        return {"phase": "PACT PGO", "status": "textual-override-only"}


def register():
    return {"pact_instrumentation": PACTHook}
