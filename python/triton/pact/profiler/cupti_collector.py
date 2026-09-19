"""CUPTI Profiling/Range API probe.  Never fabricates occupancy numbers.

L3 of the collection ladder.  On SM86 the deprecated Metric API is known
unavailable (v4 rc=38).  This module only talks to cuptiProfiler*; any
failure returns reason=unavailable.
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
    _fields_ = [("structSize", ctypes.c_size_t),
                ("pPriv", ctypes.c_void_p),
                ("ctx", ctypes.c_void_p),
                ("pCounterAvailabilityImage", ctypes.c_void_p),
                ("counterAvailabilityImageSize", ctypes.c_size_t)]


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
    params = _InitParams(ctypes.sizeof(_InitParams), None)
    init.restype = ctypes.c_int
    rc = init(ctypes.byref(params))
    if rc != 0:
        return {"available": False, "reason": f"cuptiProfilerInitialize rc={rc}",
                "lib": path}
    # Two-shot availability query.  ctx=NULL is accepted on some toolchains
    # as "current context"; a non-zero rc is still just unavailable.
    cap = _CounterAvailParams(ctypes.sizeof(_CounterAvailParams), None, None,
                              None, 0)
    avail.restype = ctypes.c_int
    rc = avail(ctypes.byref(cap))
    return {
        "available": rc == 0,
        "reason": "ok" if rc == 0 else f"GetCounterAvailability rc={rc}",
        "lib": path,
        "image_size": int(cap.counterAvailabilityImageSize),
    }


_PROBE_CACHE: Dict[Optional[str], Dict] = {}


def collect_or_unavailable(launch_fn: Optional[Callable] = None,
                           lib_path: Optional[str] = None) -> Dict:
    """Online path: never invents occupancy/stall numbers.

    launch_fn is reserved for a future Range-Profiler session around a
    replica launch.  Until that path is validated on this SM, we only
    return the availability probe.  V13 Phase1: the probe result is
    cached per lib_path — cuptiProfilerInitialize costs ~270ms of
    loader/init work and SM86 is deterministically unavailable, so every
    RPC used to pay it (the true identity of the "GIL spike" window_max;
    thread-stack-caught in the 4000-forward run).  The returned dict is
    value-identical to the uncached behaviour.
    """
    probe = _PROBE_CACHE.get(lib_path)
    if probe is None:
        probe = probe_cupti(lib_path)
        _PROBE_CACHE[lib_path] = probe
    if not probe.get("available"):
        return {
            "active_warp_ratio_permille": None,
            "stall_memory_permille": None,
            "sm_efficiency_permille": None,
            "l2_hit_permille": None,
            "source": "unavailable",
            **probe,
        }
    if launch_fn is None or os.environ.get("PACT_CUPTI_PROFILING", "0") != "1":
        return {
            "active_warp_ratio_permille": None,
            "stall_memory_permille": None,
            "sm_efficiency_permille": None,
            "l2_hit_permille": None,
            "source": "probe_only",
            "reason": "Profiling API present; range collection not enabled "
                      "(set PACT_CUPTI_PROFILING=1 and pass launch_fn)",
            **probe,
        }
    # Honest: range collection is not wired to counters yet.  Do not fake.
    try:
        launch_fn()
    except Exception as e:
        return {
            "active_warp_ratio_permille": None,
            "stall_memory_permille": None,
            "sm_efficiency_permille": None,
            "l2_hit_permille": None,
            "source": "unavailable",
            "reason": f"launch_fn raised: {e}",
            **probe,
        }
    return {
        "active_warp_ratio_permille": None,
        "stall_memory_permille": None,
        "sm_efficiency_permille": None,
        "l2_hit_permille": None,
        "source": "unavailable",
        "reason": "Profiling API available but counter config/read not "
                  "implemented; refusing to fabricate numbers",
        **probe,
    }
