# PACT Unified Architecture

`pact-unified` is the non-PGO single-codebase form: the same tree supports NVIDIA, generic AMD, and Hygon
gfx936 targets. The earlier `pact-amd` and `pact-hygon` branches are development snapshots; their non-PGO code is fully contained here. The PGO-enabled final form is `pact-pgo-unified`.

## How architecture selection works
1. Triton selects the backend compiler by active driver: CUDA uses
   `third_party/nvidia/backend/compiler.py`, HIP uses
   `third_party/amd/backend/compiler.py`.
2. Each backend propagates the driver-reported target:
   - NVIDIA: `PACT_SM_VERSION=<capability>` (e.g. 86).
   - AMD/Hygon: `PACT_AMD_ARCH=<gfx string>` (e.g. gfx942, gfx936).
3. `SMDetector::detect()` reads those variables at JIT time, selects
   `TargetBackend::NVIDIA/AMD/Unknown`, and fills the architecture resource
   table (`resourcesFor(sm)` for NVIDIA, `resourcesForAMD(gfx)` for AMD).
4. PACT C++ passes are registered in both backend pipelines and read
   `SMDetector::getBackend()` / `getResources()` for any architecture-specific
   policy. P3 is architecture-neutral; P6/P11 keep conservative defaults until
   a target-specific model is validated.

## Validation status
- NVIDIA SM86: PACT smoke correctness diff 5.8e-5.
- NVIDIA SM80/86/89/90: SMDetector branch assertions recorded in pact_paper.
- gfx936 logic smoke: `PACT_AMD_ARCH=gfx936` selects AMD(gfx936), P3 exact
  contiguity=64, P6 keeps native default, correctness diff 5.8e-5.
- True gfx936 codegen: reaches AMD LLIR pipeline; native ConvertWarpPipeline
  requires the DTK/HIP runtime environment. Runtime validation is deferred to
  real BW150 hardware.
