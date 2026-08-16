# PACT Unified Architecture (PGO form)

`pact-pgo-unified` (and its non-PGO counterpart `pact-unified`) is the final
single-codebase form of PACT v4: the same tree supports NVIDIA, generic AMD,
and Hygon gfx936 targets. The earlier `pact-amd`, `pact-hygon`,
`pact-pgo-amd`, `pact-pgo-hygon` branches are development snapshots; their
code is fully contained here.

This tree contains the PGO layer: runtime-collected hardware facts
(`measured_iterations`, `regs_per_thread`, and — only after a successful
hardware probe — `active_warp_ratio_permille`) are injected as `pact.pgo.*`
module attributes and enter the *same* theory decision functions as the
non-PGO tree. The non-PGO tree contains no PGO code and keeps the
theory-only path.

## How architecture selection works
1. Triton selects the backend compiler by active driver at runtime: CUDA uses
   `third_party/nvidia/backend/compiler.py`, HIP uses
   `third_party/amd/backend/compiler.py`.
2. Each backend propagates the driver-reported target:
   - NVIDIA: `PACT_SM_VERSION=<capability>` (e.g. 86).
   - AMD/Hygon: `PACT_AMD_ARCH=<gfx string>` (e.g. gfx942, gfx936).
3. At build time, `CMakeLists.txt` derives `TRITON_CODEGEN_BACKENDS` from the
   LLVM targets the host LLVM was built with; `setup.py` installs the
   matching Python backends.
4. `SMDetector::detect()` reads the propagated variables at JIT time, selects
   `TargetBackend::NVIDIA/AMD/Unknown`, and fills the architecture resource
   table. `numCUs` and `hasTMA` are reserved interfaces.

## Theory-as-code layering
- L1: the page-internal F₂ `LinearLayout` (`PageLocalAnalysis.cpp`) and the
  candidate-register `LinearLayout` (M2 block of `CoalesceUtils.cpp`) compute
  `mem_contig` / `reg_contig` / `V` at compile time.
- L2: `SMDetector::estimateOccupancy` computes the capacity equations;
  `PactDecision.cpp` turns them into decisions with a computed one-CTA
  discretization error bound.
- L3 (this tree only): measured facts are inputs to the same L1/L2 functions.
  P6 consumes `measured_iterations`; P11 consumes `regs_per_thread` and
  `active_warp_ratio_permille`. Missing facts fall back to the theory-only
  inputs. P6 publishes `pact.native_num_stages`; `PactPgoTrigger` reads it so
  the trigger compares the chosen stage count against the same native
  baseline P6 used (no false stage opportunity on Hopper/Blackwell).

## PGO fact chain
`pact_profile_collector` (Proton instrumentation; `n_regs` producer) →
`facts_to_hints` → `PACT_PGO_HINTS_JSON` (content-hashed path) → module
attrs injected by NVIDIA **and** AMD `make_ttir` → `AutoNumStages` /
`AutoNumWarps` → `PactPgoTrigger` → metadata → `PactPgoGatedController`
(G1 context / G2 trigger / G3 measured gain with **total** collect+compile
amortization cost / G4 rollback). G3 measures theory, 2-warp, and (S4,
evidence-gated) stage-down candidates. CUPTI probing remains best-effort:
`PACT_CUPTI_PROFILING=1` runs the legacy profiling-API detection path and
still reports unavailable on SM86; no number is fabricated.

## Validation status (v4)
- lit: 15/15 (non-PGO 12 + 3 PGO trigger tests).
- PGO chain grep: producer → inject → consumer present for every
  `pact.pgo.*` fact.
- NVIDIA SM80/86/89/90 + P11 SM80 assertion:
  `pact_paper/results/verification_nvidia_smoke_v4.json`.
- PGO full staircase: `pact_paper/results/pgo_gated_v4.json` — every shape
  triggers on `contig`; with corrected total-cost amortization no context
  passes the swap gate and no rollback occurs.
- AMD gfx942: logic smoke + `hip:gfx942` → `hsaco`
  (`verification_amd_logic_pgo_v4.json`, `verification_pgo_amd_target_v4.json`).
- Hygon gfx936: logic smoke passes; `hip:gfx936` reaches the AMD LLIR
  pipeline and fails in native `ConvertWarpPipeline`; runtime validation is
  deferred to DTK/HIP hardware.
- AMD PGO hint injection: compile-time verified for gfx942/gfx936; AMD P6
  keeps the native default stages until a CDNA-specific model is validated on
  real hardware.
