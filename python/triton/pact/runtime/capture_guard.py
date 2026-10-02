"""V17 fix (GuideLLM sweep x FULL failure): a global capture-busy flag.

Root cause (glm_on_full_gs_sweep.log, 2026-09-29): the sweep profile's
second phase (throughput@512) triggers vLLM's LAZY re-capture of a new
batch-size tier MID-RUN.  While the main thread is inside
torch.cuda.graph capture, our BACKGROUND threads (async-frame measure,
offer()'s cuModuleLoadDataEx + materialize raw launch, pool
_init_handles) issue CUDA calls that invalidate the capture
(cudaErrorStreamCaptureInvalidated -> "previous error during capture" at
the next extern_kernel).  The init-window hold gate cannot see mid-run
captures.

Protocol: the harness wraps CUDAGraph.capture_begin/capture_end to set/
clear the flag (set_capture_busy); every background CUDA touch site
wait_out_of_capture() before proceeding.  Un-wrapped processes see the
flag permanently False, so behaviour is bit-for-bit unchanged there.
"""
from __future__ import annotations

import threading
import time

_busy = threading.Event()   # set == a capture is in progress
_lock = threading.Lock()
_n = 0                      # capture nesting depth
_generation = 0             # completed-capture generation (V20 W1-1)


def capture_begin() -> None:
    global _n
    with _lock:
        _n += 1
        _busy.set()


def capture_end() -> None:
    global _n
    with _lock:
        _n = max(_n - 1, 0)
        if _n == 0:
            _busy.clear()


def capture_generation() -> int:
    with _lock:
        return _generation


def note_recapture(graph_id: int) -> None:
    """V20 W1-1 (B-2 stale-bind fix): the harness wrap calls this right
    after a CUDAGraph's capture_end -- the graph OBJECT survived a
    (re-)capture (vLLM lazy re-capture reuses it, id() unchanged), so any
    binding snapshotted from an EARLIER capture of this id holds dead
    CUgraphNode handles.  Bump the generation counter and drop the
    service state for exactly this gid (a blanket generation compare
    would force needless rebinds of every other tier's still-valid
    binding).  NOTE: _lock is NEVER held across the graph_service call
    (lock-order discipline: graph_service._LOCK -> capture_guard._lock
    only, never the reverse nesting).

    Un-wrapped processes never call this -- behaviour bit-for-bit
    unchanged there."""
    global _generation
    with _lock:
        _generation += 1
    try:
        from triton.pact.runtime.graph_service import get_service
        get_service().drop_stale_bindings(graph_id)
    except Exception:
        pass


def is_capture_busy() -> bool:
    return _busy.is_set()


def wait_out_of_capture(timeout_s: float = 600.0) -> bool:
    """Block until no capture is in progress (or timeout).  Returns True
    if the wait succeeded, False on timeout (caller proceeds anyway and
    records the race -- never deadlocks the serving path).

    600s: vLLM's mid-run multi-tier lazy capture legitimately takes
    tens of seconds; a 30s first cut timed out in the GuideLLM sweep
    rerun and proceeded INTO the capture window (the very failure this
    guard exists to prevent).  The flag clears in capture_end's finally
    even on a failed capture, so the long wait cannot deadlock.

    V18 C-3/BR-18 FIX: Event.wait() means "wait until SET" -- with the
    busy flag already set (mid-capture) it returned True IMMEDIATELY,
    so every guard point sailed straight through the very window it
    existed to avoid (the C-0 fix worked via the hold state machine +
    capture-window consumer guard, not this wait).  Poll until CLEARED
    instead; 5ms cadence is far below any capture window."""
    if not _busy.is_set():
        return True
    deadline = time.monotonic() + timeout_s
    while _busy.is_set():
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.005)
    return True


def note_race(where: str) -> None:
    """Called when a wait timed out: surfaces once per site per process
    through the service error channel if available."""
    try:
        from triton.pact.runtime.graph_service import get_service
        get_service()._err("capture_guard",
                           RuntimeError(f"proceeded while capturing: {where}"))
    except Exception:
        pass
