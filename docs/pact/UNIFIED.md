# PACT Unified Architecture (dynamic / PGO form)

`pact-pgo-unified` is the v5 dynamic hot-swap tree. It contains the same
static passes as `pact-unified` plus `python/triton/pact/**` (socket + shm +
CUPTI probe + one-variant compiler + dual-slot G4 swap).

Rollback: `pact-pgo-unified-v4` is the previous Proton/GatedController PGO
tree and is **not** an ancestor of this branch. Checkout that tag to restore
the v4 PGO runtime.

## Measurement contract: one PACT preset per process (v7)

Triton caches compiled kernels **per process** keyed on
`compute_cache_key(kernel_cache, specialization, options)`
(`python/triton/runtime/jit.py`) — that key contains **no `PACT_*` environment
variable**. Only the *disk* key does (`include/triton/Tools/Sys/GetEnv.h`,
`CACHE_INVALIDATING_ENV_VARS`). Consequences:

- compiling the same kernel specialization twice in one process with different
  PACT presets silently reuses the **first** kernel;
- `rm -rf ~/.triton/cache` does **not** fix it — the stale kernel is in memory;
- benchmark/QA harnesses must therefore run one preset per process, and read
  artifacts back from a cache directory that only that process wrote.

`pact_paper/suite/harness/worker.py` + `run_one.py` implement this: each cell
gets a fresh process and a fresh `TRITON_CACHE_DIR`. Any harness that measures
`PACT_ENABLE=0` and `PACT_ENABLE=1` inside one process is measuring vanilla
against itself.

## How architecture selection works
Same as the static tree: `PACT_SM_VERSION` / `PACT_AMD_ARCH` at JIT time,
LLVM targets at build time, `SMDetector` resource tables.

## Dynamic path
1. Inference process launches the active CompiledKernel (slot 0).
2. On (B,S) bucket change it sends `profile_and_compile` over a Unix socket.
3. Compiler service: replica CUDA-event probe (L2) + CUPTI Profiling API
   (L3, honest unavailable on this SM86 host) + family table + one JIT.
4. Candidate cubin+metadata is written to shm; inference process compiles
   the same env (cache hit if `TRITON_CACHE_DIR` is shared) and G4-swaps.
5. `PACT_HW_HINTS_JSON` injects `pact.hw.*` module attrs into P6/P11.
6. `PACT_OVERRIDE_WARPS/STAGES/V` pins a family choice.

Family table (`PACT_FAMILY_TABLE`, default
`pact_paper/eval/offline/family_table.json`) is **disabled** until the
hold-out accuracy gate passes; lookup then returns `theory`.

## Validation status (v5)
- lit: 14/14 (shared with static tree).
- Python unit: 10/10 in `pact_paper/eval/unit`.
- Dual-process demo: S=256→4096 swaps, launch loop not blocked.
- CUPTI: `cuptiProfilerInitialize rc=999` → unavailable, no fabricated
  occupancy (`results/cupti_probe_v5.json`).
  【V16-T0 更正 2026-09-23】rc=999 根因=无 CUDA context（P-a 已修）；
  v5"SM86 v4 rc=38 Metric API 退役"与 v15 P-b"WSL2 计数器不可达
  （image size=0）"均为误归因——size=0 是 `_CounterAvailParams` 末两
  字段（image/size）写反所致（两种布局同 40 字节，rc=0 不报错，读错
  偏移恒 0）。修复后同机 availability image=9184B，且 host-eval 会话
  四指标真值可达（`suite/results/v16/cupti_probe_v16.json`、
  `cupti_counters_v16.json`）。
- NVIDIA SM80/86/89/90 + P11 4→2, AMD gfx942 hsaco, gfx936
  ConvertWarpPipeline: `results/verification_dynamic_v5.json`.
