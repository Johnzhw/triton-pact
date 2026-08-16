# PACT Unified Architecture (PGO form)

`pact-pgo-unified` (and its non-PGO counterpart `pact-unified`) is the final
single-codebase form: the same tree supports NVIDIA, generic AMD, and Hygon
gfx936 targets. The earlier `pact-amd`, `pact-hygon`, `pact-pgo-amd`,
`pact-pgo-hygon` branches are development snapshots; their code is fully
contained here.

This tree contains the PGO layer: runtime-collected hardware facts
(`measured_iterations`, `regs_per_thread`, `active_warp_ratio_permille`) are
injected as `pact.pgo.*` module attributes and enter the *same* theory
decision functions as the non-PGO tree. The non-PGO tree contains no PGO code
and keeps the theory-only path.

## How architecture selection works
1. Triton selects the backend compiler by active driver at runtime: CUDA uses
   `third_party/nvidia/backend/compiler.py`, HIP uses
   `third_party/amd/backend/compiler.py`.
2. Each backend propagates the driver-reported target:
   - NVIDIA: `PACT_SM_VERSION=<capability>` (e.g. 86).
   - AMD/Hygon: `PACT_AMD_ARCH=<gfx string>` (e.g. gfx942, gfx936).
3. At build time, `CMakeLists.txt` derives `TRITON_CODEGEN_BACKENDS` from the
   LLVM targets the host LLVM was built with (`NVPTX→nvidia`,
   `AMDGPU→amd`); `setup.py` installs the matching Python backends. An LLVM
   configured with `LLVM_TARGETS_TO_BUILD=host+<GPU target>` therefore builds
   only the matching codegen backend(s) on the target machine.
4. `SMDetector::detect()` reads the propagated variables at JIT time, selects
   `TargetBackend::NVIDIA/AMD/Unknown`, and fills the architecture resource
   table (`resourcesFor(sm)` for NVIDIA, `resourcesForAMD(gfx)` for AMD).
   `numCUs` and `hasTMA` are reserved interfaces: intentionally unread until
   official per-SKU capacity data / a TMA policy land.

## Theory-as-code layering
- L1: the page-internal F₂ `LinearLayout` (`PageLocalAnalysis.cpp`) and the
  candidate-register `LinearLayout` (M2 block of `CoalesceUtils.cpp`) compute
  `mem_contig` / `reg_contig` / `V` at compile time. No theorem value is
  hard-coded.
- L2: `SMDetector::estimateOccupancy` computes the capacity equations
  (`min{SMEM, regs, warps, threads}`); `PactDecision.cpp` turns them into
  decisions with a computed one-CTA discretization error bound.
- L3 (this tree only): measured facts are inputs to the same L1/L2 functions.
  P6 consumes `measured_iterations`; P11 consumes `regs_per_thread` and
  `active_warp_ratio_permille`. The candidate compiler explicitly enables
  P11 (`pact_kernel_swapper.compile_candidate`); P6 is ON. Missing facts fall
  back to the theory-only inputs and the failure is recorded in the profile DB.

## PGO fact chain
`pact_profile_collector` (Proton instrumentation; `n_regs` producer) →
`facts_to_hints` → `PACT_PGO_HINTS_JSON` → module attrs injected by the
NVIDIA **and** AMD `make_ttir` → `AutoNumStages` / `AutoNumWarps` →
observable compile difference. The fallback collector (CUPTI/ROCTracer) only
yields `latency_us`; it has no downstream action and is documented as such.

## Validation status
- NVIDIA SM86: PACT smoke correctness diff 5.8e-5; lit
  `pact-page-local-analysis`, `pact-exact-vectorization`,
  `pact-auto-num-warps`, `pact-auto-num-stages` all pass.
- NVIDIA SM80/86/89/90: SMDetector branch assertions + P3 contiguity=64
  recorded in pact_paper (`verification_nvidia_smoke.json`).
- PGO full chain smoke (S=256): facts include measured iterations, active
  warp ratio, and `regs_per_thread`; hints are injected and consumed;
  hot-swap demo runs (`results/hotswap_demo.json`).
- AMD gfx942: `PACT_AMD_ARCH=gfx942` logic smoke passes and a true
  `hip:gfx942` compile reaches `hsaco` on this machine (no HIP runtime).
- Hygon gfx936: logic smoke passes; true `hip:gfx936` codegen reaches the AMD
  LLIR pipeline and fails in native `ConvertWarpPipeline` because this LLVM
  build does not know gfx936 (`ISAFamily::Unknown`). Runtime validation is
  deferred to DTK/HIP hardware.
- AMD PGO hint injection: compile-time verified for gfx942/gfx936
  (`verification_pgo_amd_target.json`); AMD P6 keeps the native default
  stages until a CDNA-specific model is validated on real hardware.
