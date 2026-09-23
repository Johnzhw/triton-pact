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


# ---- V16-T0: host-eval counter session (the P-c step the v15 verdict
# left "not implemented").  Struct layouts below mirror the vendored
# headers byte for byte; CUPTI validates structSize, so any layout drift
# surfaces as an rc in the returned trail instead of silent garbage.

class _HostInitParams(ctypes.Structure):
    # CUpti_Profiler_Host_Initialize_Params (cupti_profiler_host.h)
    _fields_ = [("structSize", ctypes.c_size_t),
                ("pPriv", ctypes.c_void_p),
                ("profilerType", ctypes.c_int),        # CUPTI_PROFILER_TYPE_RANGE_PROFILER = 0
                ("pChipName", ctypes.c_char_p),
                ("pCounterAvailabilityImage", ctypes.POINTER(ctypes.c_uint8)),
                ("pHostObject", ctypes.c_void_p)]      # [out]


class _HostDeinitParams(ctypes.Structure):
    _fields_ = [("structSize", ctypes.c_size_t),
                ("pPriv", ctypes.c_void_p),
                ("pHostObject", ctypes.c_void_p)]


class _HostAddMetricsParams(ctypes.Structure):
    _fields_ = [("structSize", ctypes.c_size_t),
                ("pPriv", ctypes.c_void_p),
                ("pHostObject", ctypes.c_void_p),
                ("ppMetricNames", ctypes.POINTER(ctypes.c_char_p)),
                ("numMetrics", ctypes.c_size_t)]


class _HostConfigImageSizeParams(ctypes.Structure):
    _fields_ = [("structSize", ctypes.c_size_t),
                ("pPriv", ctypes.c_void_p),
                ("pHostObject", ctypes.c_void_p),
                ("configImageSize", ctypes.c_size_t)]  # [out]


class _HostConfigImageParams(ctypes.Structure):
    _fields_ = [("structSize", ctypes.c_size_t),
                ("pPriv", ctypes.c_void_p),
                ("pHostObject", ctypes.c_void_p),
                ("configImageSize", ctypes.c_size_t),
                ("pConfigImage", ctypes.POINTER(ctypes.c_uint8))]  # [out]


class _HostEvaluateParams(ctypes.Structure):
    _fields_ = [("structSize", ctypes.c_size_t),
                ("pPriv", ctypes.c_void_p),
                ("pHostObject", ctypes.c_void_p),
                ("pCounterDataImage", ctypes.POINTER(ctypes.c_uint8)),
                ("counterDataImageSize", ctypes.c_size_t),
                ("rangeIndex", ctypes.c_size_t),
                ("ppMetricNames", ctypes.POINTER(ctypes.c_char_p)),
                ("numMetrics", ctypes.c_size_t),
                ("pMetricValues", ctypes.POINTER(ctypes.c_double))]  # [out]


class _CDIOptions(ctypes.Structure):
    # CUpti_Profiler_CounterDataImageOptions (cupti_profiler_target.h)
    _fields_ = [("structSize", ctypes.c_size_t),
                ("pPriv", ctypes.c_void_p),
                ("pCounterDataPrefix", ctypes.POINTER(ctypes.c_uint8)),
                ("counterDataPrefixSize", ctypes.c_size_t),
                ("maxNumRanges", ctypes.c_uint32),
                ("maxNumRangeTreeNodes", ctypes.c_uint32),
                ("maxRangeNameLength", ctypes.c_uint32)]


class _CDICalcSizeParams(ctypes.Structure):
    _fields_ = [("structSize", ctypes.c_size_t),
                ("pPriv", ctypes.c_void_p),
                ("sizeofCounterDataImageOptions", ctypes.c_size_t),
                ("pOptions", ctypes.POINTER(_CDIOptions)),
                ("counterDataImageSize", ctypes.c_size_t)]  # [out]


class _CDIInitParams(ctypes.Structure):
    _fields_ = [("structSize", ctypes.c_size_t),
                ("pPriv", ctypes.c_void_p),
                ("sizeofCounterDataImageOptions", ctypes.c_size_t),
                ("pOptions", ctypes.POINTER(_CDIOptions)),
                ("counterDataImageSize", ctypes.c_size_t),
                ("pCounterDataImage", ctypes.POINTER(ctypes.c_uint8))]


class _CDIScratchSizeParams(ctypes.Structure):
    _fields_ = [("structSize", ctypes.c_size_t),
                ("pPriv", ctypes.c_void_p),
                ("counterDataImageSize", ctypes.c_size_t),
                ("pCounterDataImage", ctypes.POINTER(ctypes.c_uint8)),
                ("counterDataScratchBufferSize", ctypes.c_size_t)]  # [out]


class _CDIScratchInitParams(ctypes.Structure):
    _fields_ = [("structSize", ctypes.c_size_t),
                ("pPriv", ctypes.c_void_p),
                ("counterDataImageSize", ctypes.c_size_t),
                ("pCounterDataImage", ctypes.POINTER(ctypes.c_uint8)),
                ("counterDataScratchBufferSize", ctypes.c_size_t),
                ("pCounterDataScratchBuffer", ctypes.POINTER(ctypes.c_uint8))]


class _BeginSessionParams(ctypes.Structure):
    _fields_ = [("structSize", ctypes.c_size_t),
                ("pPriv", ctypes.c_void_p),
                ("ctx", ctypes.c_void_p),
                ("counterDataImageSize", ctypes.c_size_t),
                ("pCounterDataImage", ctypes.POINTER(ctypes.c_uint8)),
                ("counterDataScratchBufferSize", ctypes.c_size_t),
                ("pCounterDataScratchBuffer", ctypes.POINTER(ctypes.c_uint8)),
                ("bDumpCounterDataInFile", ctypes.c_uint8),
                ("pCounterDataFilePath", ctypes.c_char_p),
                ("range", ctypes.c_int),               # CUPTI_AutoRange = 1
                ("replayMode", ctypes.c_int),          # CUPTI_KernelReplay = 1
                ("maxRangesPerPass", ctypes.c_size_t),
                ("maxLaunchesPerPass", ctypes.c_size_t)]


class _SetConfigParams(ctypes.Structure):
    _fields_ = [("structSize", ctypes.c_size_t),
                ("pPriv", ctypes.c_void_p),
                ("ctx", ctypes.c_void_p),
                ("pConfig", ctypes.POINTER(ctypes.c_uint8)),
                ("configSize", ctypes.c_size_t),
                ("minNestingLevel", ctypes.c_uint16),
                ("numNestingLevels", ctypes.c_uint16),
                ("passIndex", ctypes.c_size_t),
                ("targetNestingLevel", ctypes.c_uint16)]


class _CtxParams(ctypes.Structure):
    """Shared shape of Enable/DisableProfiling, UnsetConfig, EndSession."""
    _fields_ = [("structSize", ctypes.c_size_t),
                ("pPriv", ctypes.c_void_p),
                ("ctx", ctypes.c_void_p)]


class _FlushParams(ctypes.Structure):
    # CUpti_Profiler_FlushCounterData_Params; without this call the
    # KernelReplay passes never get decoded into the counter data image
    # and every metric evaluates to NaN
    _fields_ = [("structSize", ctypes.c_size_t),
                ("pPriv", ctypes.c_void_p),
                ("ctx", ctypes.c_void_p),
                ("numRangesDropped", ctypes.c_size_t),  # [out]
                ("numTraceBytesDropped", ctypes.c_size_t)]  # [out]


class _PushRangeParams(ctypes.Structure):
    _fields_ = [("structSize", ctypes.c_size_t),
                ("pPriv", ctypes.c_void_p),
                ("ctx", ctypes.c_void_p),
                ("pRangeName", ctypes.c_char_p),
                ("rangeNameLength", ctypes.c_size_t)]


class _EndPassParams(ctypes.Structure):
    _fields_ = [("structSize", ctypes.c_size_t),
                ("pPriv", ctypes.c_void_p),
                ("ctx", ctypes.c_void_p),
                ("targetNestingLevel", ctypes.c_uint16),  # [out]
                ("passIndex", ctypes.c_size_t),           # [out]
                ("allPassesSubmitted", ctypes.c_uint8)]   # [out]


_EndPassParams_STRUCT_SIZE = 41  # offsetof(allPassesSubmitted)=40 + 1


class _IsPassCollectedParams(ctypes.Structure):
    _fields_ = [("structSize", ctypes.c_size_t),
                ("pPriv", ctypes.c_void_p),
                ("ctx", ctypes.c_void_p),
                ("numRangesDropped", ctypes.c_size_t),    # [out]
                ("numTraceBytesDropped", ctypes.c_size_t),  # [out]
                ("onePassCollected", ctypes.c_uint8),     # [out]
                ("allPassesCollected", ctypes.c_uint8)]   # [out]


_IsPassCollectedParams_STRUCT_SIZE = 42  # offsetof(allPassesCollected)=41 + 1


# CUPTI's STRUCT_SIZE macro is offsetof(last field) + sizeof(last
# field), which can be SMALLER than sizeof(struct) when the last field
# is narrower than a pointer.  Two of our structs hit that; pass the
# macro value, NOT ctypes.sizeof, or CUPTI answers INVALID_PARAMETER:
_CDIOptions_STRUCT_SIZE = 44       # offsetof(maxRangeNameLength)=40 + 4
_SetConfigParams_STRUCT_SIZE = 58  # offsetof(targetNestingLevel)=56 + 2


# PACT decision channels -> CUPTI Profiling-API metric names.  The three
# pct metrics map *10 straight to permille; the stall ratio is a
# warps-per-issue-active quantity, NOT a 0-1 fraction -- callers get the
# raw value under *_raw and the permille conversion is left to the
# decider-side calibration (T6), never silently rescaled here.
COUNTER_METRICS = {
    "active_warp_ratio": "sm__warps_active.avg.pct_of_peak_sustained_active",
    "stall_memory": "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "sm_efficiency": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "l2_hit": "lts__t_sector_hit_rate.pct",
}

_CHIP_BY_CC = {(8, 0): b"GA100", (8, 6): b"GA10C", (8, 7): b"GA10C",
               (8, 9): b"AD10C", (9, 0): b"GH100"}

_AVAIL_IMAGE_CACHE: Dict[str, bytes] = {}


# ---- NVPW host SDK (libnvperf_host.so, shipped in CUDA lib64): the
# counter-data PREFIX that CUPTI's CounterDataImageOptions demands is
# generated by NVPW_CounterDataBuilder; NULL prefix -> INVALID_PARAM.
# Structs mirror nvperf_host.h / nvperf_cuda_host.h; NVPA_STRUCT_SIZE
# follows the same offsetof(last)+sizeof(last) rule as CUPTI.

class _NvpwInitHostParams(ctypes.Structure):
    _fields_ = [("structSize", ctypes.c_size_t),
                ("pPriv", ctypes.c_void_p)]


class _NvpwEvalScratchSizeParams(ctypes.Structure):
    _fields_ = [("structSize", ctypes.c_size_t),
                ("pPriv", ctypes.c_void_p),
                ("pChipName", ctypes.c_char_p),
                ("pCounterAvailabilityImage", ctypes.POINTER(ctypes.c_uint8)),
                ("scratchBufferSize", ctypes.c_size_t)]  # [out]


class _NvpwEvalInitParams(ctypes.Structure):
    _fields_ = [("structSize", ctypes.c_size_t),
                ("pPriv", ctypes.c_void_p),
                ("pScratchBuffer", ctypes.POINTER(ctypes.c_uint8)),
                ("scratchBufferSize", ctypes.c_size_t),
                ("pChipName", ctypes.c_char_p),
                ("pCounterAvailabilityImage", ctypes.POINTER(ctypes.c_uint8)),
                ("pCounterDataImage", ctypes.POINTER(ctypes.c_uint8)),
                ("counterDataImageSize", ctypes.c_size_t),
                ("pMetricsEvaluator", ctypes.c_void_p)]  # [out]


class _NvpwMetricEvalRequest(ctypes.Structure):
    # NVPW_MetricEvalRequest has NO structSize/pPriv prefix
    _fields_ = [("metricIndex", ctypes.c_size_t),
                ("metricType", ctypes.c_uint8),
                ("rollupOp", ctypes.c_uint8),
                ("submetric", ctypes.c_uint16)]


_NVPW_MetricEvalRequest_STRUCT_SIZE = 12  # offsetof(submetric)+2


class _NvpwConvertParams(ctypes.Structure):
    _fields_ = [("structSize", ctypes.c_size_t),
                ("pPriv", ctypes.c_void_p),
                ("pMetricsEvaluator", ctypes.c_void_p),
                ("pMetricName", ctypes.c_char_p),
                ("pMetricEvalRequest", ctypes.POINTER(_NvpwMetricEvalRequest)),
                ("metricEvalRequestStructSize", ctypes.c_size_t)]


class _NvpwRawDepsParams(ctypes.Structure):
    _fields_ = [("structSize", ctypes.c_size_t),
                ("pPriv", ctypes.c_void_p),
                ("pMetricsEvaluator", ctypes.c_void_p),
                ("pMetricEvalRequests", ctypes.POINTER(_NvpwMetricEvalRequest)),
                ("numMetricEvalRequests", ctypes.c_size_t),
                ("metricEvalRequestStructSize", ctypes.c_size_t),
                ("metricEvalRequestStrideSize", ctypes.c_size_t),
                ("ppRawDependencies", ctypes.POINTER(ctypes.c_char_p)),
                ("numRawDependencies", ctypes.c_size_t),
                ("ppOptionalRawDependencies", ctypes.POINTER(ctypes.c_char_p)),
                ("numOptionalRawDependencies", ctypes.c_size_t)]


class _NvpwCdbCreateParams(ctypes.Structure):
    _fields_ = [("structSize", ctypes.c_size_t),
                ("pPriv", ctypes.c_void_p),
                ("pChipName", ctypes.c_char_p),
                ("pCounterAvailabilityImage", ctypes.POINTER(ctypes.c_uint8)),
                ("pCounterDataBuilder", ctypes.c_void_p)]  # [out]


class _NvpwCdbDestroyParams(ctypes.Structure):
    # builder sits at offset 16 here (no chip/image fields) -- do NOT
    # reuse the create layout for destroy
    _fields_ = [("structSize", ctypes.c_size_t),
                ("pPriv", ctypes.c_void_p),
                ("pCounterDataBuilder", ctypes.c_void_p)]


class _NvpwRawCounterRequest(ctypes.Structure):
    # no structSize/pPriv prefix either; NVPA_Bool is uint8
    _fields_ = [("pPriv", ctypes.c_void_p),
                ("pRawCounterName", ctypes.c_char_p),
                ("domain", ctypes.c_uint32),
                ("keepInstances", ctypes.c_uint8)]


_NVPW_RawCounterRequest_STRUCT_SIZE = 21  # offsetof(keepInstances)+1


class _NvpwAddRawCountersParams(ctypes.Structure):
    _fields_ = [("structSize", ctypes.c_size_t),
                ("pPriv", ctypes.c_void_p),
                ("pCounterDataBuilder", ctypes.c_void_p),
                ("rawCounterRequestStructSize", ctypes.c_size_t),
                ("numRawCounterRequests", ctypes.c_size_t),
                ("pRawCounterRequests", ctypes.POINTER(_NvpwRawCounterRequest))]


class _NvpwGetPrefixParams(ctypes.Structure):
    _fields_ = [("structSize", ctypes.c_size_t),
                ("pPriv", ctypes.c_void_p),
                ("pCounterDataBuilder", ctypes.c_void_p),
                ("bytesAllocated", ctypes.c_size_t),
                ("pBuffer", ctypes.POINTER(ctypes.c_uint8)),
                ("bytesCopied", ctypes.c_size_t)]  # [out]


def _find_libnvperf_host() -> Optional[str]:
    env = os.environ.get("PACT_NVPERF_LIB")
    if env and os.path.exists(env):
        return env
    cuda = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or "/usr/local/cuda"
    cand = os.path.join(cuda, "lib64", "libnvperf_host.so")
    return cand if os.path.isfile(cand) else None


_PREFIX_CACHE: Dict = {}


def _nvperf_prefix(avail_ptr, metric_names, trail) -> Optional[bytes]:
    """CounterDataPrefix via the NVPW host SDK: evaluator converts the
    metric names to raw counter dependencies, a CounterDataBuilder turns
    those into the prefix blob CUPTI wants.  Cached per (lib, metrics).
    """
    host = _find_libnvperf_host()
    if not host:
        trail["nvperf"] = "libnvperf_host not found"
        return None
    key = (host, tuple(metric_names))
    if key in _PREFIX_CACHE:
        return _PREFIX_CACHE[key]
    try:
        nv = ctypes.CDLL(host)
    except OSError as e:
        trail["nvperf"] = f"dlopen failed: {e}"
        return None
    for sym in ("NVPW_InitializeHost",
                "NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize",
                "NVPW_CUDA_MetricsEvaluator_Initialize",
                "NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest",
                "NVPW_MetricsEvaluator_GetMetricRawDependencies",
                "NVPW_CUDA_CounterDataBuilder_Create",
                "NVPW_CounterDataBuilder_AddRawCounters",
                "NVPW_CounterDataBuilder_GetCounterDataPrefix",
                "NVPW_CounterDataBuilder_Destroy",
                "NVPW_MetricsEvaluator_Destroy"):
        if not hasattr(nv, sym):
            trail["nvperf"] = f"symbol missing: {sym}"
            return None
        getattr(nv, sym).restype = ctypes.c_int

    hp = _NvpwInitHostParams(ctypes.sizeof(_NvpwInitHostParams), None)
    trail["nvpw_init_rc"] = nv.NVPW_InitializeHost(ctypes.byref(hp))
    if trail["nvpw_init_rc"] != 0:
        return None

    ssp = _NvpwEvalScratchSizeParams(
        ctypes.sizeof(_NvpwEvalScratchSizeParams), None, None, avail_ptr, 0)
    trail["eval_scratch_rc"] = nv.NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize(
        ctypes.byref(ssp))
    if trail["eval_scratch_rc"] != 0 or ssp.scratchBufferSize == 0:
        return None
    scratch = (ctypes.c_uint8 * ssp.scratchBufferSize)()
    eip = _NvpwEvalInitParams(
        ctypes.sizeof(_NvpwEvalInitParams), None,
        ctypes.cast(scratch, ctypes.POINTER(ctypes.c_uint8)),
        ssp.scratchBufferSize, None, avail_ptr, None, 0, None)
    trail["eval_init_rc"] = nv.NVPW_CUDA_MetricsEvaluator_Initialize(
        ctypes.byref(eip))
    if trail["eval_init_rc"] != 0:
        return None
    ev = eip.pMetricsEvaluator

    reqs = (_NvpwMetricEvalRequest * len(metric_names))()
    for i, name in enumerate(metric_names):
        cp = _NvpwConvertParams(
            ctypes.sizeof(_NvpwConvertParams), None, ev, name.encode(),
            ctypes.pointer(reqs[i]), _NVPW_MetricEvalRequest_STRUCT_SIZE)
        rc = nv.NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest(
            ctypes.byref(cp))
        trail.setdefault("convert_rcs", {})[name] = int(rc)
        if rc != 0:
            return None

    # raw dependencies: shot 1 counts, shot 2 fills the name array
    rdp = _NvpwRawDepsParams(ctypes.sizeof(_NvpwRawDepsParams), None, ev,
                             reqs, len(metric_names),
                             _NVPW_MetricEvalRequest_STRUCT_SIZE,
                             ctypes.sizeof(_NvpwMetricEvalRequest),
                             None, 0, None, 0)
    trail["rawdeps_count_rc"] = nv.NVPW_MetricsEvaluator_GetMetricRawDependencies(
        ctypes.byref(rdp))
    if trail["rawdeps_count_rc"] != 0 or rdp.numRawDependencies == 0:
        return None
    dep_arr = (ctypes.c_char_p * rdp.numRawDependencies)()
    rdp.ppRawDependencies = dep_arr
    rdp.numRawDependencies = len(dep_arr)
    trail["rawdeps_fill_rc"] = nv.NVPW_MetricsEvaluator_GetMetricRawDependencies(
        ctypes.byref(rdp))
    if trail["rawdeps_fill_rc"] != 0:
        return None
    counter_names = [dep_arr[i] for i in range(rdp.numRawDependencies)
                     if dep_arr[i] is not None]
    trail["num_raw_counters"] = len(counter_names)

    bp = _NvpwCdbCreateParams(ctypes.sizeof(_NvpwCdbCreateParams), None,
                              None, avail_ptr, None)
    trail["cdb_create_rc"] = nv.NVPW_CUDA_CounterDataBuilder_Create(
        ctypes.byref(bp))
    if trail["cdb_create_rc"] != 0:
        return None
    builder = bp.pCounterDataBuilder

    rcqs = (_NvpwRawCounterRequest * len(counter_names))()
    for i, cn in enumerate(counter_names):
        rcqs[i] = _NvpwRawCounterRequest(None, cn, 0, 0)
    acp = _NvpwAddRawCountersParams(
        ctypes.sizeof(_NvpwAddRawCountersParams), None, builder,
        _NVPW_RawCounterRequest_STRUCT_SIZE, len(counter_names), rcqs)
    trail["cdb_add_rc"] = nv.NVPW_CounterDataBuilder_AddRawCounters(
        ctypes.byref(acp))
    if trail["cdb_add_rc"] != 0:
        nv.NVPW_CounterDataBuilder_Destroy(ctypes.byref(
            _NvpwCdbDestroyParams(ctypes.sizeof(_NvpwCdbDestroyParams), None,
                                  builder)))
        return None

    # prefix: shot 1 size, shot 2 bytes (align 8 from malloc'd buffer)
    gp = _NvpwGetPrefixParams(ctypes.sizeof(_NvpwGetPrefixParams), None,
                              builder, 0, None, 0)
    trail["prefix_size_rc"] = nv.NVPW_CounterDataBuilder_GetCounterDataPrefix(
        ctypes.byref(gp))
    if trail["prefix_size_rc"] != 0 or gp.bytesCopied == 0:
        return None
    pbuf = (ctypes.c_uint8 * gp.bytesCopied)()
    gp.bytesAllocated = gp.bytesCopied
    gp.pBuffer = ctypes.cast(pbuf, ctypes.POINTER(ctypes.c_uint8))
    gp.bytesCopied = 0
    trail["prefix_fill_rc"] = nv.NVPW_CounterDataBuilder_GetCounterDataPrefix(
        ctypes.byref(gp))
    if trail["prefix_fill_rc"] != 0 or gp.bytesCopied == 0:
        return None
    prefix = bytes(pbuf[:gp.bytesCopied])
    trail["prefix_bytes"] = gp.bytesCopied
    _PREFIX_CACHE[key] = prefix
    # keep evaluator alive? destroy is safe: dep strings were copied into
    # counter_names via c_char_p already-materialised python bytes
    return prefix


def _availability_bytes(path: str, cupti) -> Optional[bytes]:
    cached = _AVAIL_IMAGE_CACHE.get(path)
    if cached is not None:
        return cached
    img = counter_availability_image(path)
    if not img.get("ok"):
        return None
    import base64
    raw = base64.b64decode(img["image_b64"])
    _AVAIL_IMAGE_CACHE[path] = raw
    return raw


def counter_session(launch_fn: Callable,
                    metric_names=None,
                    lib_path: Optional[str] = None,
                    range_name: Optional[str] = None) -> Dict:
    """One host-eval counter session around ``launch_fn``.

    ``range_name=None`` profiles in CUPTI_AutoRange mode (one implicit
    range per kernel launch); passing a name switches to
    CUPTI_UserRange + Push/PopRange, which explicitly brackets
    ``launch_fn`` -- the mode the CUPTI samples use and the one that
    reliably captures torch/cublas launches.

    Chain: ProfilerInitialize -> CounterAvailability (cached image) ->
    HostInitialize -> HostConfigAddMetrics -> GetConfigImage ->
    NVPW CounterDataPrefix -> CounterDataImage alloc/init + scratch ->
    BeginSession -> SetConfig -> EnableProfiling [-> PushRange] ->
    launch [-> PopRange] -> Disable -> Flush -> Unset -> EndSession ->
    HostEvaluateToGpuValues(first non-empty range) -> HostDeinitialize.

    Every rc lands in the returned trail; nothing is ever fabricated and
    nothing raises past the caller.  Cost note: the ~270ms
    cuptiProfilerInitialize loader-lock window and the ms-scale session
    setup mean this belongs in a trigger-event measurement window, never
    on the serving hot path (the PACT_CUPTI_PROFILING gate enforces that).
    """
    path = lib_path or _find_libcupti()
    if not path:
        return {"ok": False, "reason": "libcupti not found"}
    cupti = ctypes.CDLL(path)
    for sym in ("cuptiProfilerInitialize", "cuptiProfilerHostInitialize",
                "cuptiProfilerHostConfigAddMetrics",
                "cuptiProfilerHostGetConfigImageSize",
                "cuptiProfilerHostGetConfigImage",
                "cuptiProfilerHostEvaluateToGpuValues",
                "cuptiProfilerHostDeinitialize",
                "cuptiProfilerCounterDataImageCalculateSize",
                "cuptiProfilerCounterDataImageInitialize",
                "cuptiProfilerCounterDataImageCalculateScratchBufferSize",
                "cuptiProfilerCounterDataImageInitializeScratchBuffer",
                "cuptiProfilerBeginSession", "cuptiProfilerSetConfig",
                "cuptiProfilerUnsetConfig", "cuptiProfilerEndSession",
                "cuptiProfilerFlushCounterData",
                "cuptiProfilerPushRange", "cuptiProfilerPopRange",
                "cuptiProfilerHostGetRangeName",
                "cuptiProfilerBeginPass", "cuptiProfilerEndPass",
                "cuptiProfilerIsPassCollected",
                "cuptiProfilerEnableProfiling",
                "cuptiProfilerDisableProfiling"):
        if not hasattr(cupti, sym):
            return {"ok": False, "reason": f"symbol missing: {sym}"}
        getattr(cupti, sym).restype = ctypes.c_int
    trail: Dict[str, int] = {}
    names = list(metric_names or COUNTER_METRICS.values())

    # context FIRST (P-a lesson): initialize without an active context
    # returns rc=999 on WSL2
    _ensure_cuda_context()
    ip = _InitParams(ctypes.sizeof(_InitParams), None)
    trail["init_rc"] = cupti.cuptiProfilerInitialize(ctypes.byref(ip))
    if trail["init_rc"] != 0:
        return {"ok": False, "reason": "initialize", "trail": trail}

    ctx = _current_context()
    avail = _availability_bytes(path, cupti)
    if avail is None:
        return {"ok": False, "reason": "counter availability image", "trail": trail}
    avail_arr = (ctypes.c_uint8 * len(avail)).from_buffer_copy(avail)
    avail_ptr = ctypes.cast(avail_arr, ctypes.POINTER(ctypes.c_uint8))

    chip = None
    try:
        import torch
        chip = _CHIP_BY_CC.get(torch.cuda.get_device_capability())
    except Exception:
        pass

    def host_init():
        hp = _HostInitParams(ctypes.sizeof(_HostInitParams), None, 0,
                             chip, avail_ptr, None)
        rc = cupti.cuptiProfilerHostInitialize(ctypes.byref(hp))
        return rc, hp

    rc, hp = host_init()
    trail["host_init_rc"] = rc
    if rc != 0 and chip is not None:
        # some stacks refuse the chip name but accept the image alone
        chip = None
        rc, hp = host_init()
        trail["host_init_nochip_rc"] = rc
    if rc != 0:
        return {"ok": False, "reason": "host initialize", "trail": trail,
                "chip_used": bool(chip)}
    host_obj = hp.pHostObject

    # metric support: try all at once; on rejection probe one-by-one on
    # fresh host objects and keep the supported subset
    def add_metrics(names_):
        arr = (ctypes.c_char_p * len(names_))(*[n.encode() for n in names_])
        ap = _HostAddMetricsParams(ctypes.sizeof(_HostAddMetricsParams),
                                   None, host_obj, arr, len(names_))
        return cupti.cuptiProfilerHostConfigAddMetrics(ctypes.byref(ap))

    trail["add_all_rc"] = add_metrics(names)
    if trail["add_all_rc"] != 0:
        supported, probe_rcs = [], {}
        for n in names:
            rc2, hp2 = host_init()
            if rc2 != 0:
                break
            arr = (ctypes.c_char_p * 1)(n.encode())
            ap = _HostAddMetricsParams(
                ctypes.sizeof(_HostAddMetricsParams), None, hp2.pHostObject,
                arr, 1)
            probe_rcs[n] = cupti.cuptiProfilerHostConfigAddMetrics(
                ctypes.byref(ap))
            dp = _HostDeinitParams(ctypes.sizeof(_HostDeinitParams), None,
                                   hp2.pHostObject)
            cupti.cuptiProfilerHostDeinitialize(ctypes.byref(dp))
            if probe_rcs[n] == 0:
                supported.append(n)
        trail["metric_probe_rcs"] = {k: int(v) for k, v in probe_rcs.items()}
        if not supported:
            dp = _HostDeinitParams(ctypes.sizeof(_HostDeinitParams), None,
                                   host_obj)
            cupti.cuptiProfilerHostDeinitialize(ctypes.byref(dp))
            return {"ok": False, "reason": "no supported metrics", "trail": trail}
        # rebuild the live host object with only the supported subset
        dp = _HostDeinitParams(ctypes.sizeof(_HostDeinitParams), None, host_obj)
        cupti.cuptiProfilerHostDeinitialize(ctypes.byref(dp))
        rc3, hp3 = host_init()
        if rc3 != 0:
            return {"ok": False, "reason": "host re-init for subset", "trail": trail}
        host_obj = hp3.pHostObject
        rc4 = add_metrics(supported)
        if rc4 != 0:
            return {"ok": False, "reason": f"add subset rc={rc4}", "trail": trail}
        names = supported

    sp = _HostConfigImageSizeParams(ctypes.sizeof(_HostConfigImageSizeParams),
                                    None, host_obj, 0)
    trail["config_size_rc"] = cupti.cuptiProfilerHostGetConfigImageSize(
        ctypes.byref(sp))
    if trail["config_size_rc"] != 0 or sp.configImageSize == 0:
        return {"ok": False, "reason": "config image size", "trail": trail}
    cfg_buf = (ctypes.c_uint8 * sp.configImageSize)()
    gp = _HostConfigImageParams(ctypes.sizeof(_HostConfigImageParams), None,
                                host_obj, sp.configImageSize,
                                ctypes.cast(cfg_buf, ctypes.POINTER(ctypes.c_uint8)))
    trail["config_image_rc"] = cupti.cuptiProfilerHostGetConfigImage(
        ctypes.byref(gp))
    if trail["config_image_rc"] != 0:
        return {"ok": False, "reason": "config image", "trail": trail}

    # the prefix NVPW builds from the SAME metric set the config uses
    prefix = _nvperf_prefix(avail_ptr, names, trail)
    if prefix is None:
        return {"ok": False, "reason": "counter data prefix (NVPW)", "trail": trail}
    prefix_arr = (ctypes.c_uint8 * len(prefix)).from_buffer_copy(prefix)
    prefix_ptr = ctypes.cast(prefix_arr, ctypes.POINTER(ctypes.c_uint8))

    # mirror the userrange_profiling sample exactly: with a range name
    # we run UserRange+UserReplay (all four capacity values 1, enable/
    # disable INSIDE the pass loop bracketing Push/Pop); without one we
    # fall back to AutoRange (Push/Pop is illegal there)
    use_user = bool(range_name)
    opts = _CDIOptions(_CDIOptions_STRUCT_SIZE, None, prefix_ptr,
                       len(prefix), 1, 1, 64)
    cp = _CDICalcSizeParams(ctypes.sizeof(_CDICalcSizeParams), None,
                            _CDIOptions_STRUCT_SIZE,
                            ctypes.pointer(opts), 0)
    trail["cdi_size_rc"] = cupti.cuptiProfilerCounterDataImageCalculateSize(
        ctypes.byref(cp))
    if trail["cdi_size_rc"] != 0 or cp.counterDataImageSize == 0:
        return {"ok": False, "reason": "counter data image size", "trail": trail}
    cdi = (ctypes.c_uint8 * cp.counterDataImageSize)()
    cdi_ptr = ctypes.cast(cdi, ctypes.POINTER(ctypes.c_uint8))
    cip = _CDIInitParams(ctypes.sizeof(_CDIInitParams), None,
                         _CDIOptions_STRUCT_SIZE, ctypes.pointer(opts),
                         cp.counterDataImageSize, cdi_ptr)
    trail["cdi_init_rc"] = cupti.cuptiProfilerCounterDataImageInitialize(
        ctypes.byref(cip))
    if trail["cdi_init_rc"] != 0:
        return {"ok": False, "reason": "counter data image init", "trail": trail}
    szp = _CDIScratchSizeParams(ctypes.sizeof(_CDIScratchSizeParams), None,
                                cp.counterDataImageSize, cdi_ptr, 0)
    trail["scratch_size_rc"] = cupti.cuptiProfilerCounterDataImageCalculateScratchBufferSize(
        ctypes.byref(szp))
    if trail["scratch_size_rc"] != 0:
        return {"ok": False, "reason": "scratch size", "trail": trail}
    scratch = (ctypes.c_uint8 * szp.counterDataScratchBufferSize)()
    scratch_ptr = ctypes.cast(scratch, ctypes.POINTER(ctypes.c_uint8))
    sip = _CDIScratchInitParams(ctypes.sizeof(_CDIScratchInitParams), None,
                                cp.counterDataImageSize, cdi_ptr,
                                szp.counterDataScratchBufferSize, scratch_ptr)
    trail["scratch_init_rc"] = cupti.cuptiProfilerCounterDataImageInitializeScratchBuffer(
        ctypes.byref(sip))
    if trail["scratch_init_rc"] != 0:
        return {"ok": False, "reason": "scratch init", "trail": trail}

    def fail(reason):
        dp = _HostDeinitParams(ctypes.sizeof(_HostDeinitParams), None, host_obj)
        cupti.cuptiProfilerHostDeinitialize(ctypes.byref(dp))
        return {"ok": False, "reason": reason, "trail": trail}

    # the userrange_profiling sample passes ctx=NULL everywhere (NULL =
    # current context); passing the explicit primary ctx made UserRange
    # BeginSession return INVALID_PARAMETER on this stack
    _ctx_arg = None
    # CUpti_ProfilerReplayMode: ApplicationReplay=1, KernelReplay=2,
    # UserReplay=3 (NOT 1/2 -- misreading this made every earlier combo
    # an illegal pairing: UserRange+KernelReplay -> begin rc=1)
    bsp = _BeginSessionParams(
        ctypes.sizeof(_BeginSessionParams), None,
        _ctx_arg, cp.counterDataImageSize,
        cdi_ptr, szp.counterDataScratchBufferSize, scratch_ptr,
        0, None, 2 if use_user else 1, 3 if use_user else 2, 1, 1)
    trail["begin_session_rc"] = cupti.cuptiProfilerBeginSession(ctypes.byref(bsp))
    if trail["begin_session_rc"] != 0:
        return fail("begin session")
    scp = _SetConfigParams(_SetConfigParams_STRUCT_SIZE, None,
                           _ctx_arg,
                           ctypes.cast(cfg_buf, ctypes.POINTER(ctypes.c_uint8)),
                           sp.configImageSize, 1, 1, 0, 0)
    trail["set_config_rc"] = cupti.cuptiProfilerSetConfig(ctypes.byref(scp))
    if trail["set_config_rc"] != 0:
        cupti.cuptiProfilerEndSession(ctypes.byref(
            _CtxParams(ctypes.sizeof(_CtxParams), None,
                       _ctx_arg)))
        return fail("set config")
    try:
        # UserReplay multipass, sample-shaped: enable/disable bracket
        # Push/Pop INSIDE each pass; we re-issue launch_fn ourselves
        # every pass until CUPTI reports all passes submitted
        passes = 0
        while passes < 16:
            bpc = _CtxParams(ctypes.sizeof(_CtxParams), None,
                             _ctx_arg)
            trail["begin_pass_rc"] = cupti.cuptiProfilerBeginPass(
                ctypes.byref(bpc))
            if trail["begin_pass_rc"] != 0:
                break
            enp = _CtxParams(ctypes.sizeof(_CtxParams), None,
                             _ctx_arg)
            trail["enable_rc"] = cupti.cuptiProfilerEnableProfiling(
                ctypes.byref(enp))
            if trail["enable_rc"] != 0:
                break
            if use_user:
                prp = _PushRangeParams(
                    ctypes.sizeof(_PushRangeParams), None,
                    _ctx_arg,
                    range_name.encode(), 0)
                trail["push_range_rc"] = cupti.cuptiProfilerPushRange(
                    ctypes.byref(prp))
                if trail["push_range_rc"] != 0:
                    break
            launch_fn()
            passes += 1
            try:
                import torch
                torch.cuda.synchronize()
            except Exception:
                pass
            if use_user:
                trail["pop_range_rc"] = cupti.cuptiProfilerPopRange(ctypes.byref(
                    _CtxParams(ctypes.sizeof(_CtxParams), None,
                               _ctx_arg)))
            dsp = _CtxParams(ctypes.sizeof(_CtxParams), None,
                             _ctx_arg)
            trail["disable_rc"] = cupti.cuptiProfilerDisableProfiling(
                ctypes.byref(dsp))
            epp = _EndPassParams(_EndPassParams_STRUCT_SIZE, None,
                                 _ctx_arg,
                                 0, 0, 0)
            rc_ep = cupti.cuptiProfilerEndPass(ctypes.byref(epp))
            trail["end_pass_rc"] = rc_ep
            trail["passes"] = passes
            if rc_ep != 0:
                break
            if epp.allPassesSubmitted:
                break
    except Exception as e:
        cupti.cuptiProfilerDisableProfiling(ctypes.byref(
            _CtxParams(ctypes.sizeof(_CtxParams), None,
                       _ctx_arg)))
        cupti.cuptiProfilerEndSession(ctypes.byref(
            _CtxParams(ctypes.sizeof(_CtxParams), None,
                       _ctx_arg)))
        return fail(f"launch_fn raised: {e}")
    dp2 = _CtxParams(ctypes.sizeof(_CtxParams), None,
                     _ctx_arg)
    trail["disable_rc"] = cupti.cuptiProfilerDisableProfiling(ctypes.byref(dp2))
    fp = _FlushParams(ctypes.sizeof(_FlushParams), None,
                      _ctx_arg, 0, 0)
    trail["flush_rc"] = cupti.cuptiProfilerFlushCounterData(ctypes.byref(fp))
    trail["ranges_dropped"] = int(fp.numRangesDropped)
    cupti.cuptiProfilerUnsetConfig(ctypes.byref(
        _CtxParams(ctypes.sizeof(_CtxParams), None,
                   _ctx_arg)))
    trail["end_session_rc"] = cupti.cuptiProfilerEndSession(ctypes.byref(
        _CtxParams(ctypes.sizeof(_CtxParams), None,
                   _ctx_arg)))

    values = (ctypes.c_double * len(names))()
    name_arr = (ctypes.c_char_p * len(names))(*[n.encode() for n in names])
    metrics = {}

    class _RangeNameParams(ctypes.Structure):
        _fields_ = [("structSize", ctypes.c_size_t),
                    ("pPriv", ctypes.c_void_p),
                    ("pCounterDataImage", ctypes.POINTER(ctypes.c_uint8)),
                    ("counterDataImageSize", ctypes.c_size_t),
                    ("rangeIndex", ctypes.c_size_t),
                    ("delimiter", ctypes.c_char_p),
                    ("pRangeName", ctypes.c_void_p)]  # [out] CUPTI-allocated

    for ri in range(3):  # diagnostics: what landed in the CDI image
        rnp = _RangeNameParams(ctypes.sizeof(_RangeNameParams), None,
                               cdi_ptr, cp.counterDataImageSize, ri,
                               b"/", None)
        rc = cupti.cuptiProfilerHostGetRangeName(ctypes.byref(rnp))
        nm = None
        if rc == 0 and rnp.pRangeName:
            nm = ctypes.cast(rnp.pRangeName, ctypes.c_char_p).value
            nm = nm.decode(errors="replace") if nm else None
        trail.setdefault("range_names", {})[str(ri)] = [int(rc), nm]

    trail["evaluate_rc"] = -1
    for range_idx in range(4):  # first non-empty AutoRange wins
        evp = _HostEvaluateParams(
            ctypes.sizeof(_HostEvaluateParams), None, host_obj, cdi_ptr,
            cp.counterDataImageSize, range_idx, name_arr, len(names),
            ctypes.cast(values, ctypes.POINTER(ctypes.c_double)))
        rc = cupti.cuptiProfilerHostEvaluateToGpuValues(ctypes.byref(evp))
        vals = {n: float(values[i]) for i, n in enumerate(names)}
        if rc == 0 and any(v == v for v in vals.values()):
            trail["evaluate_rc"], metrics = 0, vals
            trail["range_index_used"] = range_idx
            break
        if rc != 0:
            trail["evaluate_rc"] = rc
            break
    dp3 = _HostDeinitParams(ctypes.sizeof(_HostDeinitParams), None, host_obj)
    cupti.cuptiProfilerHostDeinitialize(ctypes.byref(dp3))
    if trail["evaluate_rc"] != 0:
        return {"ok": False, "reason": "evaluate", "trail": trail,
                "metric_names": names}
    return {"ok": True, "metric_names": names, "metrics": metrics,
            "trail": trail, "chip_used": bool(chip),
            "counter_data_image_size": cp.counterDataImageSize}



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
    # V16-T0: real counter session (host-eval chain above).  Still
    # strictly gated: only runs inside a trigger-event measurement
    # window, never on the serving hot path.  UserRange+UserReplay is
    # the userrange_profiling-sample shape (KernelReplay re-reads the
    # captured launch params after the fact and corrupts the context
    # with triton's staging buffers).
    try:
        sess = counter_session(launch_fn, lib_path=lib_path,
                               range_name="pact_measure")
    except Exception as e:
        return {**probe,
                "active_warp_ratio_permille": None,
                "stall_memory_permille": None,
                "sm_efficiency_permille": None,
                "l2_hit_permille": None,
                "source": "unavailable",
                "reason": f"session raised: {e}"}
    if not sess.get("ok"):
        return {**probe,
                "active_warp_ratio_permille": None,
                "stall_memory_permille": None,
                "sm_efficiency_permille": None,
                "l2_hit_permille": None,
                "source": "unavailable",
                "reason": f"counter session failed: {sess.get('reason')}",
                "trail": sess.get("trail")}
    name2chan = {v: k for k, v in COUNTER_METRICS.items()}
    m = sess["metrics"]

    def _permille(chan, scale=10.0):
        for name, v in m.items():
            if name2chan.get(name) == chan and v is not None:
                if v != v:  # NaN: metric not derivable for this range
                    return None
                return int(round(v * scale))
        return None

    return {**probe,
            "active_warp_ratio_permille": _permille("active_warp_ratio"),
            "stall_memory_permille": _permille("stall_memory"),
            "sm_efficiency_permille": _permille("sm_efficiency"),
            "l2_hit_permille": _permille("l2_hit"),
            "source": "cupti_counters",
            "metric_names": sess.get("metric_names"),
            "raw_metrics": {k: round(v, 4) for k, v in m.items()},
            "stall_metric_semantics": ("stall_memory_permille is the "
                                       "per-issue-active warps ratio x10, "
                                       "not a 0-1 fraction; final scaling "
                                       "belongs to the decider (T6)")}
