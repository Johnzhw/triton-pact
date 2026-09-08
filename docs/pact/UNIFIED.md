# PACT Unified Architecture (dynamic / PGO form)

`pact-pgo-unified` is the v5 dynamic hot-swap tree. It contains the same
static passes as `pact-unified` plus `python/triton/pact/**` (socket + shm +
CUPTI probe + one-variant compiler + dual-slot G4 swap).

Rollback: `pact-pgo-unified-v4` is the previous Proton/GatedController PGO
tree and is **not** an ancestor of this branch. Checkout that tag to restore
the v4 PGO runtime.

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
- NVIDIA SM80/86/89/90 + P11 4→2, AMD gfx942 hsaco, gfx936
  ConvertWarpPipeline: `results/verification_dynamic_v5.json`.
