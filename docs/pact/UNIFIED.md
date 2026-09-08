# PACT Unified Architecture (non-PGO form)

`pact-unified` is the non-PGO single-codebase form of PACT v5: the same tree
supports NVIDIA, generic AMD, and Hygon gfx936 targets. The dynamic/PGO form
lives in `pact-pgo-unified` (`pact-pgo-unified-v5` tag). v4 tags remain as
rollback points (`pact-unified-v4` / `pact-pgo-unified-v4`).

**This tree contains no PGO code**: there are no `pact.pgo.*` attribute
readers/writers, no `PACT_PGO_HINTS_JSON`, and no PGO runtime modules or
Proton PACT hook. P6/P11 use only IR-static inputs and the theory model
below, so the build is fully independent of runtime-collected hardware
parameters.

## How architecture selection works
1. Triton selects the backend compiler by active driver at runtime: CUDA uses
   `third_party/nvidia/backend/compiler.py`, HIP uses
   `third_party/amd/backend/compiler.py`.
2. Each backend propagates the driver-reported target:
   - NVIDIA: `PACT_SM_VERSION=<capability>` (e.g. 86).
   - AMD/Hygon: `PACT_AMD_ARCH=<gfx string>` (e.g. gfx942, gfx936).
3. At build time, `CMakeLists.txt` derives `TRITON_CODEGEN_BACKENDS` from the
   LLVM targets the host LLVM was built with (`NVPTX→nvidia`,
   `AMDGPU→amd`); `setup.py` installs the matching Python backends.
4. `SMDetector::detect()` reads the propagated variables at JIT time, selects
   `TargetBackend::NVIDIA/AMD/Unknown`, and fills the architecture resource
   table. `numCUs` and `hasTMA` are reserved interfaces: intentionally unread
   until official per-SKU capacity data / a TMA policy land.

## Theory-as-code layering
- L1: the page-internal F₂ `LinearLayout` (`PageLocalAnalysis.cpp`) and the
  candidate-register `LinearLayout` (M2 block of `CoalesceUtils.cpp`) compute
  `mem_contig` / `reg_contig` / `V` at compile time. No theorem value is
  hard-coded.
- L2: `SMDetector::estimateOccupancy` computes the capacity equations
  (`min{SMEM, regs, warps, threads}`); `PactDecision.cpp` turns them into
  decisions with a computed one-CTA discretization error bound.
- The unknown register count is an explicit named model-input assumption
  (`PactDecisionConstants::kUnknownRegsPerThread`), not a theorem constant.
- P3 is architecture-neutral; AMD P6 keeps the native default stages until a
  CDNA-specific model is validated. P6 publishes `pact.native_num_stages`
  so downstream consumers (the PGO trigger in the other tree) compare the
  chosen stage count against the same baseline P6 used.

## v5 additions (static tree)
- Optional measured stall/SM-efficiency coefficients on SelectStagesInput
  (default 0 keeps v4 ranking).  P6/P11 read optional `pact.hw.*` attrs.
- `PACT_OVERRIDE_WARPS/STAGES/V` pins, cache-keyed.  Lit 14/14.
- P11 keeps `stagesPerBlock=3` (SM80 16x64 data gate).  Publishes
  `pact.p11.stages_assumption`.

## Validation status (v5)
- lit: 14/14.
- zero-dynamic gate PASS.
- NVIDIA/AMD matrix: `pact_paper/results/verification_dynamic_v5.json`.
