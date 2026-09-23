"""CUPTI Profiling/Range API probe.  Never fabricates occupancy numbers.

L3 of the collection ladder.  2026-09-21 live diagnosis (V15 Phase P-a)
corrects the historical record: on this WSL2 host the failure was
``cuptiProfilerInitialize rc=999 (CUPTI_ERROR_UNKNOWN)`` whose ROOT
CAUSE is a missing active CUDA context at probe time — NOT the old
"SM86 v4 rc=38 Metric API retirement" note this file used to carry.
With a context created first (``torch.cuda.init()``), the 13.0 library
(2025.3.1) returns available=True.  This module only talks to
cuptiProfiler*; any failure returns reason=unavailable.

 Mixing hazard (measured): loading the 12.2 libcupti in the same
 process as torch cu130 segfaults — do NOT point PACT_CUPTI_LIB at a
 12.x tree; _find_libcupti's CUDA_HOME preference already picks 13.0.
"""
from __future__ import annotations

import ctypes
import glob
import os
from typing import Callable, Dict, Optional


class _InitParams(ctypes.Structure):
    _fields_ = [("structSize", ctypes.c_size_t),
                ("pPriv", ctypes.c_void_p)]


class _CounterAvailParams(ctypes.Structure):
    # Field order MUST mirror CUpti_Profiler_GetCounterAvailability_Params
    # (cupti_profiler_target.h): structSize, pPriv, ctx,
    # counterAvailabilityImageSize, pCounterAvailabilityImage -- size
    # BEFORE image.  V16-T0 (2026-09-23): this struct had the last two
    # fields swapped; both layouts are 40 bytes so CUPTI accepted the
    # structSize, but shot-1 wrote the real size at offset 24 while we
    # read offset 32 -> image_size was ALWAYS 0.  The V15 P-b verdict
    # "counter path unreachable on WSL2" was this bug, not the platform
    # (a standalone C++ two-shot on the same host returns 9184 bytes).
    _fields_ = [("structSize", ctypes.c_size_t),
                ("pPriv", ctypes.c_void_p),
                ("ctx", ctypes.c_void_p),
                ("counterAvailabilityImageSize", ctypes.c_size_t),
                ("pCounterAvailabilityImage", ctypes.c_void_p)]


def _ensure_cuda_context():
    """rc=999 root cause fix: cuptiProfilerInitialize needs an ACTIVE
    CUDA context on WSL2.  In the compiler service and any vLLM-mounted
    process the context already exists and this is a no-op; standalone
    callers get a lazy init (init() alone does not bind the context to
    the thread — one runtime op does).  Failures degrade to the old
    behaviour (probe returns unavailable) — never raises past the
    caller."""
    try:
        import torch  # noqa: import at use; the pact stack always has it
        if torch.cuda.is_available():
            torch.cuda.init()
            _ = torch.zeros(1, device="cuda")
            return True
    except Exception:
        pass
    return False


def _find_libcupti() -> Optional[str]:
    env = os.environ.get("PACT_CUPTI_LIB")
    if env and os.path.exists(env):
        return env
    cands = []
    cuda = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or "/usr/local/cuda"
    cands += glob.glob(os.path.join(cuda, "lib64", "libcupti.so*"))
    try:
        from triton.knobs import proton
        d = getattr(proton, "cupti_lib_dir", None)
        val = d() if callable(d) else getattr(d, "val", None) if d is not None else None
        if val:
            cands += glob.glob(os.path.join(str(val), "libcupti.so*"))
    except Exception:
        pass
    # Prefer the highest-versioned real .so (not the unversioned symlink last)
    files = [p for p in cands if os.path.isfile(p)]
    if not files:
        return None
    files.sort()
    return files[-1]


def probe_cupti(lib_path: Optional[str] = None) -> Dict:
    """Availability only.  Does not return a numeric occupancy."""
    path = lib_path or _find_libcupti()
    if not path:
        return {"available": False, "reason": "libcupti not found"}
    try:
        cupti = ctypes.CDLL(path)
    except OSError as e:
        return {"available": False, "reason": f"dlopen failed: {e}", "lib": path}
    init = getattr(cupti, "cuptiProfilerInitialize", None)
    avail = getattr(cupti, "cuptiProfilerGetCounterAvailability", None)
    if init is None or avail is None:
        return {"available": False,
                "reason": "Profiling API symbols missing",
                "lib": path}
    # P-a fix: rc=999 was "no active CUDA context" on WSL2.  A serving
    # process already has one; a standalone probe lazily inits it here.
    had_context = _ensure_cuda_context()
    params = _InitParams(ctypes.sizeof(_InitParams), None)
    init.restype = ctypes.c_int
    rc = init(ctypes.byref(params))
    if rc != 0:
        return {"available": False,
                "reason": f"cuptiProfilerInitialize rc={rc}"
                          f" (cuda_context_preinited={had_context})",
                "lib": path}
    # Two-shot availability query.  ctx=NULL is accepted on some toolchains
    # as "current context"; a non-zero rc is still just unavailable.
    cap = _CounterAvailParams(ctypes.sizeof(_CounterAvailParams), None, None,
                              0, None)
    avail.restype = ctypes.c_int
    rc = avail(ctypes.byref(cap))
    return {
        "available": rc == 0,
        "reason": "ok" if rc == 0 else f"GetCounterAvailability rc={rc}",
        "lib": path,
        "image_size": int(cap.counterAvailabilityImageSize),
    }


# ---- V15 P-b: counter-availability spike (two-shot + real context) ----

def _current_context():
    """CUcontext of the calling thread via libcuda (no cuda-python)."""
    try:
        cu = ctypes.CDLL("libcuda.so.1")
        ctx = ctypes.c_void_p()
        cu.cuCtxGetCurrent.restype = ctypes.c_int
        rc = cu.cuCtxGetCurrent(ctypes.byref(ctx))
        return ctx.value if rc == 0 and ctx.value else None
    except Exception:
        return None


def counter_availability_image(lib_path: Optional[str] = None) -> Dict:
    """P-b spike step 1: fetch the counter-availability image (two-shot
    per the CUPTI docs: NULL buffer first to learn the size, then a real
    buffer), with the REAL current context (ctx=NULL gave rc=3 on this
    stack).  V16-T0 fixed the swapped size/image field order that made
    shot-1 report image_size=0 on a healthy stack.  Returns the image
    bytes + honest rc trail; never raises.
    """
    path = lib_path or _find_libcupti()
    if not path:
        return {"ok": False, "reason": "libcupti not found"}
    try:
        cupti = ctypes.CDLL(path)
    except OSError as e:
        return {"ok": False, "reason": f"dlopen failed: {e}"}
    init = getattr(cupti, "cuptiProfilerInitialize", None)
    avail = getattr(cupti, "cuptiProfilerGetCounterAvailability", None)
    if init is None or avail is None:
        return {"ok": False, "reason": "symbols missing"}
    _ensure_cuda_context()
    ip = _InitParams(ctypes.sizeof(_InitParams), None)
    init.restype = ctypes.c_int
    rc0 = init(ctypes.byref(ip))
    out = {"lib": path, "init_rc": rc0}
    if rc0 != 0:
        out.update(ok=False, reason=f"initialize rc={rc0}")
        return out
    ctx = _current_context()
    out["ctx"] = hex(ctx) if ctx else None
    # shot 1: NULL image -> size (V16-T0: size field precedes the image
    # pointer in the real struct; the old swapped layout read a stale 0)
    cap = _CounterAvailParams(ctypes.sizeof(_CounterAvailParams), None,
                              ctypes.c_void_p(ctx) if ctx else None,
                              0, None)
    avail.restype = ctypes.c_int
    rc1 = avail(ctypes.byref(cap))
    size = int(cap.counterAvailabilityImageSize or 0)
    out["shot1_rc"], out["image_size"] = rc1, size
    if rc1 != 0 or size == 0:
        out.update(ok=False,
                   reason=f"shot1 rc={rc1} size={size} (ctx_NULL rc was 3)")
        return out
    buf = ctypes.create_string_buffer(size)
    cap2 = _CounterAvailParams(ctypes.sizeof(_CounterAvailParams), None,
                               ctypes.c_void_p(ctx) if ctx else None,
                               size, ctypes.cast(buf, ctypes.c_void_p))
    rc2 = avail(ctypes.byref(cap2))
    out["shot2_rc"] = rc2
    if rc2 != 0:
        out.update(ok=False, reason=f"shot2 rc={rc2}")
        return out
    out.update(ok=True, reason="ok",
               image_b64=__import__("base64").b64encode(
                   buf.raw[:size]).decode())
    return out


def profiler_symbols(lib_path: Optional[str] = None) -> Dict:
    """P-b spike step 0: which cuptiProfiler* entry points the 13.0
    library actually exports (drives what a session can use)."""
    path = lib_path or _find_libcupti()
    if not path:
        return {"ok": False}
    import subprocess
    r = subprocess.run(["nm", "-D", path], capture_output=True, text=True)
    syms = sorted({l.split()[-1] for l in r.stdout.splitlines()
                   if "cuptiProfiler" in l})
    return {"ok": bool(syms), "lib": path, "symbols": syms}


_PROBE_CACHE: Dict[Optional[str], Dict] = {}


def collect_or_unavailable(launch_fn: Optional[Callable] = None,
                           lib_path: Optional[str] = None) -> Dict:
    """Online path: never invents occupancy/stall numbers.

    launch_fn is reserved for a future Range-Profiler session around a
    replica launch.  Until that path is validated on this SM, we only
    return the availability probe.  V13 Phase1: the probe result is
    cached per lib_path — cuptiProfilerInitialize costs ~270ms of
    loader/init work, so every RPC used to pay it (the true identity of
    the "GIL spike" window_max; thread-stack-caught in the 4000-forward
    run).  P-a (V15): with the CUDA-context guard the probe returns
    available=True on the 13.0 library; the four numeric fields stay
    null until counter collection is implemented — the explicit reason
    strings below now win over the probe's own "ok" (the **probe
    spread used to overwrite them; order fixed).
    """
    probe = _PROBE_CACHE.get(lib_path)
    if probe is None:
        probe = probe_cupti(lib_path)
        _PROBE_CACHE[lib_path] = probe
    if not probe.get("available"):
        return {**probe,
                "active_warp_ratio_permille": None,
                "stall_memory_permille": None,
                "sm_efficiency_permille": None,
                "l2_hit_permille": None,
                "source": "unavailable"}
    if launch_fn is None or os.environ.get("PACT_CUPTI_PROFILING", "0") != "1":
        return {**probe,
                "active_warp_ratio_permille": None,
                "stall_memory_permille": None,
                "sm_efficiency_permille": None,
                "l2_hit_permille": None,
                "source": "probe_only",
                "reason": "Profiling API present; range collection not "
                          "enabled (set PACT_CUPTI_PROFILING=1 and pass "
                          "launch_fn)"}
    # Honest: range collection is not wired to counters yet.  Do not fake.
    try:
        launch_fn()
    except Exception as e:
        return {**probe,
                "active_warp_ratio_permille": None,
                "stall_memory_permille": None,
                "sm_efficiency_permille": None,
                "l2_hit_permille": None,
                "source": "unavailable",
                "reason": f"launch_fn raised: {e}"}
    return {**probe,
            "active_warp_ratio_permille": None,
            "stall_memory_permille": None,
            "sm_efficiency_permille": None,
            "l2_hit_permille": None,
            "source": "unavailable",
            "reason": "Profiling API available but counter config/read "
                      "not implemented; refusing to fabricate numbers"}
