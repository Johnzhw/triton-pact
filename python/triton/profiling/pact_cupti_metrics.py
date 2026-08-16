"""Best-effort CUPTI hardware active-warp probe (PGO branch).

The deprecated CUPTI Metric API (cupti_metrics.h) is documented as unsupported
for SM >= 7.5.  This module first checks whether that API can resolve the
metric name; on the local RTX 3080 (SM86) it returns rc=38 and the caller
records `active_warp_ratio_unavailable` instead of fabricating a number.

When PACT_CUPTI_PROFILING=1, an additional legacy CUPTI Profiling API probe is
attempted:
  1. cuptiProfilerInitialize
  2. cuptiProfilerGetCounterAvailability (two-call buffer query)
  3. metric-id resolution through the deprecated metric API
If all three succeed, the module still returns `unavailable` because a real
per-launch counter read must be wired into an actual kernel launch; this probe
only proves that the detection/initialization path is usable on a future
host (H100/A100) where metric resolution works.  No number is ever invented.
"""
import ctypes
import os
from pathlib import Path
from typing import Dict, Optional

from triton import knobs

_METRIC = "sm__warps_active.avg.pct_of_peak_sustained_active"


def _find_libcupti() -> Optional[Path]:
    candidates = []
    for key in ("TRITON_CUPTI_LIB_PATH",):
        value = os.environ.get(key)
        if value:
            candidates.append(Path(value))
    candidates.append(Path(str(knobs.proton.cupti_lib_dir)) / "libcupti.so")
    for c in candidates:
        if c.is_file():
            return c
    return None


def _profiling_api_enabled() -> bool:
    return os.environ.get("PACT_CUPTI_PROFILING") == "1"


class _InitParams(ctypes.Structure):
    _fields_ = [("structSize", ctypes.c_size_t), ("pPriv", ctypes.c_void_p)]


class _CounterAvailabilityParams(ctypes.Structure):
    _fields_ = [
        ("structSize", ctypes.c_size_t),
        ("pPriv", ctypes.c_void_p),
        ("ctx", ctypes.c_void_p),
        ("counterAvailabilityImageSize", ctypes.c_size_t),
        ("pCounterAvailabilityImage", ctypes.POINTER(ctypes.c_uint8)),
    ]


def _resolve_metric_id(cupti, device: int):
    """Returns (rc, metric_id)."""
    metric_id = ctypes.c_uint32(0)
    fn = cupti.cuptiMetricGetIdFromName
    fn.argtypes = [ctypes.c_int, ctypes.c_char_p,
                   ctypes.POINTER(ctypes.c_uint32)]
    fn.restype = ctypes.c_int
    rc = fn(device, _METRIC.encode(), ctypes.byref(metric_id))
    return rc, metric_id.value


def _probe_profiling_api(cupti, device: int) -> Dict[str, object]:
    """Legacy CUPTI Profiling API detection path (never measures a number).

    Any failure is reported as `active_warp_ratio_unavailable + reason`.
    """
    try:
        init = cupti.cuptiProfilerInitialize
        deinit = cupti.cuptiProfilerDeInitialize
        get_avail = cupti.cuptiProfilerGetCounterAvailability
    except AttributeError as e:
        return {"active_warp_ratio_unavailable":
                f"CUPTI profiling API symbols missing: {e}"}

    init.argtypes = [ctypes.POINTER(_InitParams)]
    init.restype = ctypes.c_int
    deinit.argtypes = [ctypes.POINTER(_InitParams)]
    deinit.restype = ctypes.c_int
    get_avail.argtypes = [ctypes.POINTER(_CounterAvailabilityParams)]
    get_avail.restype = ctypes.c_int

    init_params = _InitParams(ctypes.sizeof(_InitParams), None)
    initialized = False
    try:
        rc = init(ctypes.byref(init_params))
        if rc != 0:
            return {"active_warp_ratio_unavailable":
                    f"cuptiProfilerInitialize rc={rc}"}
        initialized = True

        size_params = _CounterAvailabilityParams(
            ctypes.sizeof(_CounterAvailabilityParams), None, None,
            ctypes.c_size_t(0), None)
        rc = get_avail(ctypes.byref(size_params))
        if rc != 0:
            return {"active_warp_ratio_unavailable":
                    f"cuptiProfilerGetCounterAvailability rc={rc}"}
        if size_params.counterAvailabilityImageSize == 0:
            return {"active_warp_ratio_unavailable":
                    "cuptiProfilerGetCounterAvailability returned image size 0"}

        buf = (ctypes.c_uint8 *
               size_params.counterAvailabilityImageSize)()
        data_params = _CounterAvailabilityParams(
            ctypes.sizeof(_CounterAvailabilityParams), None, None,
            ctypes.c_size_t(size_params.counterAvailabilityImageSize),
            ctypes.cast(buf, ctypes.POINTER(ctypes.c_uint8)))
        rc = get_avail(ctypes.byref(data_params))
        if rc != 0:
            return {"active_warp_ratio_unavailable":
                    f"cuptiProfilerGetCounterAvailability(image) rc={rc}"}

        rc, _ = _resolve_metric_id(cupti, device)
        if rc != 0:
            return {"active_warp_ratio_unavailable":
                    f"profiling API initialized but cuptiMetricGetIdFromName "
                    f"rc={rc}"}
        # Detection layers all pass.  A numeric result still requires a real
        # per-launch counter read (event-group enable / read around a kernel),
        # which is not wired into this build's no-kernel probe.
        return {"active_warp_ratio_unavailable":
                "CUPTI profiling API and metric detection succeeded; per-launch "
                "counter collection not wired into probe (H100/A100 validation "
                "pending)"}
    except OSError as e:
        return {"active_warp_ratio_unavailable":
                f"CUPTI profiling API probe failed: {e}"}
    finally:
        if initialized:
            deinit_params = _InitParams(ctypes.sizeof(_InitParams), None)
            try:
                deinit(ctypes.byref(deinit_params))
            except OSError:
                pass


def probe_active_warp_permille(device: int = 0) -> Dict[str, object]:
    """Return {'active_warp_ratio_permille': int, 'source': 'cupti'} on a real
    metric reading, or an honest unavailable record.

    Default path keeps the v3 deprecated-API behavior byte-for-byte.
    PACT_CUPTI_PROFILING=1 additionally probes the legacy CUPTI Profiling API
    initialization/counter-availability path.
    """
    lib = _find_libcupti()
    if lib is None:
        return {"active_warp_ratio_unavailable":
                "libcupti.so not found under knobs.proton.cupti_lib_dir"}
    try:
        cupti = ctypes.CDLL(str(lib))
        rc, _ = _resolve_metric_id(cupti, device)
        if rc != 0:
            reason = (f"cuptiMetricGetIdFromName rc={rc}; deprecated metric API "
                      f"unsupported on SM>=7.5 (CUDA toolchain on this host)")
            if _profiling_api_enabled():
                profiler = _probe_profiling_api(cupti, device)
                profiler["active_warp_ratio_unavailable"] = (
                    f"{profiler['active_warp_ratio_unavailable']} "
                    f"(after {reason})")
                return profiler
            return {"active_warp_ratio_unavailable": reason}
        # Metric id resolution alone is not a measurement.  A full profiling-API
        # collection path is required before this module may return a number.
        return {"active_warp_ratio_unavailable":
                "metric resolved but CUPTI profiling collection is not "
                "implemented in this build"}
    except OSError as e:
        return {"active_warp_ratio_unavailable": f"libcupti load failed: {e}"}
