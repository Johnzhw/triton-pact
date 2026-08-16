"""Best-effort CUPTI hardware active-warp probe (PGO branch).

The deprecated CUPTI Metric API (cupti_metrics.h) is documented as unsupported
for SM >= 7.5.  This module therefore first checks whether that API is usable;
on the local RTX 3080 (SM86) it is expected to be unavailable, and the caller
records `active_warp_ratio_unavailable` instead of fabricating a number.
Implementing the replacement CUPTI profiling API (cupti_profiler_host.h /
cupti_range_profiler.h) is the next step when a machine/toolkit with a working
metric collection path is available.
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


def probe_active_warp_permille(device: int = 0) -> Dict[str, object]:
    """Return {'active_warp_ratio_permille': int, 'source': 'cupti'} on a real
    metric reading, or an honest unavailable record."""
    lib = _find_libcupti()
    if lib is None:
        return {"active_warp_ratio_unavailable":
                "libcupti.so not found under knobs.proton.cupti_lib_dir"}
    try:
        cupti = ctypes.CDLL(str(lib))
        metric_id = ctypes.c_uint32(0)
        # CUptiResult cuptiMetricGetIdFromName(CUdevice, const char*, CUpti_MetricID*)
        fn = cupti.cuptiMetricGetIdFromName
        fn.argtypes = [ctypes.c_int, ctypes.c_char_p,
                       ctypes.POINTER(ctypes.c_uint32)]
        fn.restype = ctypes.c_int
        rc = fn(device, _METRIC.encode(), ctypes.byref(metric_id))
        if rc != 0:
            return {"active_warp_ratio_unavailable":
                    f"cuptiMetricGetIdFromName rc={rc}; deprecated metric API "
                    f"unsupported on SM>=7.5 (CUDA toolchain on this host)"}
        # Metric id resolution alone is not a measurement.  A full profiling-API
        # collection path is required before this module may return a number.
        return {"active_warp_ratio_unavailable":
                "metric resolved but CUPTI profiling collection is not "
                "implemented in this build"}
    except OSError as e:
        return {"active_warp_ratio_unavailable": f"libcupti load failed: {e}"}
