"""PACT-AOBO: third tree (V11-6b) — in-process asynchronous kernel switching.

Branch ``pact-aobo-unified`` (from ``pact-pgo-unified`` HEAD).  Only this
package is added on top of the pgo tree:

    aobo-vs-unified diff = python/triton/pact/** + python/triton/pact_aobo/**
    aobo-vs-pgo     diff = python/triton/pact_aobo/** only

Motivation (PLAN_CCFB_V11 V11-6b + E2E_INTEGRATION_AND_AOBO.md §2.4): the
two-process PGO shape (CompilerService thread + socket + shm) is a standing
source of races and switching cost.  Following AOBO (TACO'25) the engine is
restructured as *resident + inline*:

  resident_pool  variants stay compiled & loaded (cuModule-resident);
                the AOBO "预载文件入内存" counterpart
  inline_decider decision runs inside the process (learned/table reuse),
                no socket round-trip at all
  async_switch  decode NEVER blocks on a compile: the current slot serves
                every forward; a background thread compiles the decided
                variant; installation is an atomic slot exchange at the
                next launch boundary (AOBO 调用点重编码 analogue)
  graph_recode  CUDA-Graph counterpart: swap the kernel function of an
                instantiated graph node via cudaGraphExecKernelNodeSetParams
                (ctypes spike + helpers)

Nothing here changes the ``triton.pact`` packages; the pgo dynamic path is
untouched (V11-0-fixed) and remains the reference for comparison.
"""
from triton.pact_aobo.resident_pool import ResidentPool
from triton.pact_aobo.async_switch import AsyncKernelSwitch
from triton.pact_aobo.inline_decider import decide_inline

__all__ = ["ResidentPool", "AsyncKernelSwitch", "decide_inline"]
