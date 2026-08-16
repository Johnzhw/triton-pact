//===- PactSMDetect.cpp - PACT SM Architecture Detection -------------------===//
//
// Unified target-architecture detection for all PACT passes.
//
// Detection model (M0): the target is a *runtime (JIT-time) input*, not a
// compile-time -D.  The Python frontend (compiler.py) propagates the driver-
// reported capability at JIT time:
//   - NVIDIA: PACT_SM_VERSION = <numeric SM version> (e.g. 86)
//   - AMD:    PACT_AMD_ARCH   = <gfx arch string>  (e.g. gfx942)
//
// No silent fallback to a specific SKU (the old hardcoded RTX-3080 fallback is
// removed).  If neither variable is set, we log a warning and use a conservative
// "unknown" resource table so downstream passes behave safely.
//
//===----------------------------------------------------------------------===//

#include "triton/Support/PactSMDetect.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <sstream>

namespace mlir::triton::pact {

int SMDetector::version = -1;
TargetBackend SMDetector::backend = TargetBackend::Unknown;
std::string SMDetector::gfxArch;
SMResources SMDetector::resources = {};
bool SMDetector::initialized = false;

namespace {

// Conservative resource table used when the target is unknown.  All
// downstream heuristics must stay safe under this (minimal SMEM/occupancy,
// no TMA, no aggressive cp.async).
SMResources conservativeResources(int version) {
  return {
      /*smVersion*/ version,
      /*smemPerSM*/ 96 * 1024,
      /*maxRegsPerSM*/ 65536,
      /*maxWarpsPerSM*/ 32,
      /*maxThreadsPerSM*/ 1024,
      /*waveSize*/ 32,
      /*numCUs*/ 0,
      /*effectiveSmemPerBlock*/ 16 * 1024,
      /*pipelineThreshold*/ 256,
      /*occupancyCliffStages*/ 2,
      /*hasTMA*/ false,
      /*cpAsyncLatency*/ 40,
      /*optimalNumStages*/ 2,
      /*asyncCopyMinWidth*/ 16};
}

// M0b: exact SM-version-keyed resource table, separating architecture family
// (80/90/100) from SKU (A100 vs RTX 3080, H100 vs RTX 40 Ada).
SMResources resourcesFor(int sm) {
  if (sm >= 100) {
    // Blackwell (B100/B200)
    return {
        /*smVersion*/ sm,
        /*smemPerSM*/ 228 * 1024,
        /*maxRegsPerSM*/ 65536,
        /*maxWarpsPerSM*/ 64,
        /*maxThreadsPerSM*/ 2048,
        /*waveSize*/ 32,
        /*numCUs*/ 0,
        /*effectiveSmemPerBlock*/ 48 * 1024,
        /*pipelineThreshold*/ 64,
        /*occupancyCliffStages*/ 5,
        /*hasTMA*/ true,
        /*cpAsyncLatency*/ 18,
        /*optimalNumStages*/ 5,
        /*asyncCopyMinWidth*/ 8};
  }
  if (sm >= 90) {
    // Hopper (H100/H200)
    return {
        /*smVersion*/ sm,
        /*smemPerSM*/ 228 * 1024,
        /*maxRegsPerSM*/ 65536,
        /*maxWarpsPerSM*/ 64,
        /*maxThreadsPerSM*/ 2048,
        /*waveSize*/ 32,
        /*numCUs*/ 0,
        /*effectiveSmemPerBlock*/ 48 * 1024,
        /*pipelineThreshold*/ 64,
        /*occupancyCliffStages*/ 5,
        /*hasTMA*/ true,
        /*cpAsyncLatency*/ 20,
        /*optimalNumStages*/ 5,
        /*asyncCopyMinWidth*/ 8};
  }
  if (sm >= 89) {
    // Ada (RTX 40 series): 100KB SMEM, 64 warps, 1536 threads, no TMA
    return {
        /*smVersion*/ sm,
        /*smemPerSM*/ 100 * 1024,
        /*maxRegsPerSM*/ 65536,
        /*maxWarpsPerSM*/ 64,
        /*maxThreadsPerSM*/ 1536,
        /*waveSize*/ 32,
        /*numCUs*/ 0,
        /*effectiveSmemPerBlock*/ 24 * 1024,
        /*pipelineThreshold*/ 128,
        /*occupancyCliffStages*/ 3,
        /*hasTMA*/ false,
        /*cpAsyncLatency*/ 30,
        /*optimalNumStages*/ 3,
        /*asyncCopyMinWidth*/ 16};
  }
  if (sm >= 86) {
    // Ampere GA10x (RTX 3080/3090): 99KB SMEM, 48 warps, 1536 threads.
    // Key constraint: num_stages 3→4 drops occupancy 1024→768 threads.
    return {
        /*smVersion*/ sm,
        /*smemPerSM*/ 99 * 1024,
        /*maxRegsPerSM*/ 65536,
        /*maxWarpsPerSM*/ 48,
        /*maxThreadsPerSM*/ 1536,
        /*waveSize*/ 32,
        /*numCUs*/ 0,
        /*effectiveSmemPerBlock*/ 24 * 1024,
        /*pipelineThreshold*/ 128,
        /*occupancyCliffStages*/ 3,
        /*hasTMA*/ false,
        /*cpAsyncLatency*/ 30,
        /*optimalNumStages*/ 3,
        /*asyncCopyMinWidth*/ 16};
  }
  if (sm >= 80) {
    // Ampere GA100 (A100): 164KB SMEM, 64 warps, 2048 threads.
    // NOTE: distinct from GA10x (RTX 3080) — previously conflated.
    return {
        /*smVersion*/ sm,
        /*smemPerSM*/ 164 * 1024,
        /*maxRegsPerSM*/ 65536,
        /*maxWarpsPerSM*/ 64,
        /*maxThreadsPerSM*/ 2048,
        /*waveSize*/ 32,
        /*numCUs*/ 0,
        /*effectiveSmemPerBlock*/ 48 * 1024,
        /*pipelineThreshold*/ 128,
        /*occupancyCliffStages*/ 4,
        /*hasTMA*/ false,
        /*cpAsyncLatency*/ 30,
        /*optimalNumStages*/ 4,
        /*asyncCopyMinWidth*/ 16};
  }
  // Volta/Turing (SM 70-79) — conservative fallback
  return conservativeResources(sm);
}

// AMD capacity hints.  Until official per-SKU numbers are available these
// values are conservative: wave64/32, 64KB LDS per CU, 16 waves per CU
// accounted, no TMA.  numCUs=0 means device capacity is not modeled yet.
SMResources resourcesForAMD(const std::string &arch) {
  auto amd = [](int waveSize, int64_t ldsPerCU) {
    SMResources r = conservativeResources(/*version=*/0);
    r.waveSize = waveSize;
    r.smemPerSM = (int)ldsPerCU;
    r.maxWarpsPerSM = 16;      // conservative waves-per-CU accounting
    r.maxThreadsPerSM = 16 * waveSize;
    r.numCUs = 0;              // capacity unknown; occupancy stays conservative
    r.hasTMA = false;
    r.cpAsyncLatency = 40;
    r.asyncCopyMinWidth = 16;
    return r;
  };
  if (arch == "gfx942")
    return amd(/*waveSize=*/64, /*ldsPerCU=*/64 * 1024);
  if (arch == "gfx950" || arch == "gfx1250")
    return amd(/*waveSize=*/32, /*ldsPerCU=*/64 * 1024);
  if (arch == "gfx936") {
    // Hygon DCU BW150 / DTK 25.04.04 (gfx936-class).  Official per-SKU
    // capacity parameters (CU count, LDS size, VGPR/AGPR layout, matrix-core
    // shapes) must be filled from Hygon DTK 25.04.04 documents and the real
    // device properties; until then use the conservative wave64 table so the
    // compiler never over-commits occupancy.
    return amd(/*waveSize=*/64, /*ldsPerCU=*/64 * 1024);
  }

  llvm::errs() << "[PACT SMDetect] WARNING: AMD target '" << arch
               << "' has no resource table — using conservative fallback.\n";
  return amd(/*waveSize=*/64, /*ldsPerCU=*/64 * 1024);
}

} // anonymous namespace

int SMDetector::detect() {
  if (version >= 0)
    return version;

  // AMD takes precedence: PACT_AMD_ARCH is a gfx string.
  if (const char *amdEnv = std::getenv("PACT_AMD_ARCH")) {
    backend = TargetBackend::AMD;
    gfxArch = std::string(amdEnv);
    // Store a non-negative sentinel so detect() is idempotent; the actual
    // architecture is carried by gfxArch + backend, not the numeric version.
    version = 0;
    resources = resourcesForAMD(gfxArch);
    initialized = true;
    return version;
  }

  if (const char *env = std::getenv("PACT_SM_VERSION")) {
    backend = TargetBackend::NVIDIA;
    gfxArch.clear();
    version = std::atoi(env);

    // Validate: clamp to a plausible range; anything out of range becomes
    // "unknown" rather than silently assuming a specific SKU.
    if (version < 70 || version > 120) {
      llvm::errs() << "[PACT SMDetect] WARNING: unexpected SM version "
                   << version << " — using conservative fallback.\n";
      version = 0;
      backend = TargetBackend::Unknown;
      resources = conservativeResources(version);
    } else {
      resources = resourcesFor(version);
    }
    initialized = true;
    return version;
  }

  // Neither PACT_SM_VERSION nor PACT_AMD_ARCH is set.  No silent fallback.
  llvm::errs() << "[PACT SMDetect] WARNING: PACT_SM_VERSION/PACT_AMD_ARCH not "
               << "set — PACT passes should run after compiler.py propagates "
               << "capability. Using conservative fallback.\n";
  backend = TargetBackend::Unknown;
  gfxArch.clear();
  version = 0;
  resources = conservativeResources(version);
  initialized = true;
  return version;
}

TargetBackend SMDetector::getBackend() {
  detect();
  return backend;
}

bool SMDetector::isAmpere() {
  int sm = detect();
  return backend == TargetBackend::NVIDIA && sm >= 80 && sm < 90;
}

bool SMDetector::isHopper() {
  return backend == TargetBackend::NVIDIA && detect() >= 90;
}

bool SMDetector::isVolta() {
  int sm = detect();
  return backend == TargetBackend::NVIDIA && sm >= 70 && sm < 80;
}

const SMResources &SMDetector::getResources() {
  detect();
  return resources;
}

PipelineBudget SMDetector::computePipelineBudget(
    int64_t tileBytes, int estIterations, int pageSize,
    int tileTokens, int defaultStages) {

  detect();
  PipelineBudget budget;
  const auto &sm = resources;

  // Fixed SMEM overhead: block table + scalars (~4KB)
  int64_t fixedSMEM = 4 * 1024;
  budget.smemBudget = sm.effectiveSmemPerBlock - fixedSMEM;

  // Max stages by SMEM
  if (tileBytes > 0)
    budget.maxStagesBySMEM =
        std::max(1, (int)(budget.smemBudget / tileBytes));
  else
    budget.maxStagesBySMEM = sm.optimalNumStages;

  // Max stages by occupancy (critical for Ampere!)
  if (backend == TargetBackend::NVIDIA && sm.smVersion < 90) {
    // Ampere occupancy model:
    //   num_stages=2: SMEM ~16KB → 6 blocks/SM → 1536 threads
    //   num_stages=3: SMEM ~24KB → 4 blocks/SM → 1024 threads
    //   num_stages=4: SMEM ~32KB → 3 blocks/SM → 768 threads ← CLIFF!
    budget.maxStagesByOccupancy = std::max(
        2, std::min(sm.occupancyCliffStages,
                    (int)(sm.effectiveSmemPerBlock /
                          std::max(tileBytes, (int64_t)1))));
  } else {
    // Hopper/Blackwell/Unknown: larger SMEM, occupancy constraint is looser.
    budget.maxStagesByOccupancy = 8;
  }

  // Max stages by iterations: a modulo-scheduled pipeline cannot usefully
  // have more stages than loop iterations.  This bound is computed from the
  // actual trip count, not a fixed divisor.
  if (estIterations > 0)
    budget.maxStagesByIters = std::max(1, estIterations);
  else
    budget.maxStagesByIters = std::max(1, sm.optimalNumStages);

  // Combined recommendation: take the minimum of all constraints
  budget.recommendedStages = std::min(
      {budget.maxStagesBySMEM, budget.maxStagesByOccupancy,
       budget.maxStagesByIters, sm.optimalNumStages + 1});
  budget.recommendedStages = std::max(2, budget.recommendedStages);

  return budget;
}

double SMDetector::estimateOccupancy(int numStages, int64_t smemPerBlock,
                                     int regsPerThread, int numWarps) {
  detect();
  const auto &sm = resources;

  // Guard against num_warps=0 (callers should always pass ≥1).
  numWarps = std::max(numWarps, 1);

  // numStages is folded into smemPerBlock by callers (tileBytes * numStages).
  // It is retained in the signature for API compatibility and debug logging.
  (void)numStages;

  int waveSize = sm.waveSize > 0 ? sm.waveSize : 32;
  int threadsPerBlock = numWarps * waveSize;
  int64_t smemTotal = smemPerBlock + 4 * 1024; // +fixed overhead

  // Blocks limited by SMEM
  int blocksBySMEM = sm.smemPerSM / std::max(smemTotal, (int64_t)1);
  // Blocks limited by registers.  regsPerThread <= 0 means "unknown": the
  // register constraint is intentionally ignored (model input convention,
  // see PactDecisionConstants::kUnboundedRegsPerThread).
  int blocksByRegs = std::numeric_limits<int>::max();
  if (regsPerThread > 0) {
    int regsPerWarp = regsPerThread * waveSize;
    int regsPerBlock = regsPerWarp * numWarps;
    blocksByRegs = sm.maxRegsPerSM / std::max(regsPerBlock, 1);
  }
  // Blocks limited by warps
  int blocksByWarps = sm.maxWarpsPerSM / numWarps;
  // Blocks limited by threads
  int blocksByThreads = sm.maxThreadsPerSM / std::max(threadsPerBlock, 1);

  int activeBlocks = std::min(
      {blocksBySMEM, blocksByRegs, blocksByWarps, blocksByThreads});
  int activeWarps = activeBlocks * numWarps;

  return (double)activeWarps / sm.maxWarpsPerSM;
}

std::string SMDetector::getGPUName() {
  detect();
  if (backend == TargetBackend::AMD)
    return "AMD (" + gfxArch + ")";
  if (backend == TargetBackend::Unknown)
    return "Unknown (SM" + std::to_string(version) + ")";
  if (version >= 100)
    return "Blackwell (SM" + std::to_string(version) + ")";
  if (version >= 90)
    return "Hopper (SM" + std::to_string(version) + ")";
  if (version >= 89)
    return "Ada (SM" + std::to_string(version) + ")";
  if (version >= 86)
    return "Ampere GA10x (SM" + std::to_string(version) + ")";
  if (version >= 80)
    return "Ampere GA100 (SM" + std::to_string(version) + ")";
  if (version >= 70)
    return "Volta/Turing (SM" + std::to_string(version) + ")";
  return "Unknown (SM" + std::to_string(version) + ")";
}

std::string PipelineBudget::toJSON() const {
  std::ostringstream oss;
  oss << "{"
      << "\"smem_budget\":" << smemBudget << ","
      << "\"max_stages_by_smem\":" << maxStagesBySMEM << ","
      << "\"max_stages_by_occupancy\":" << maxStagesByOccupancy << ","
      << "\"max_stages_by_iters\":" << maxStagesByIters << ","
      << "\"recommended_stages\":" << recommendedStages << "}";
  return oss.str();
}

} // namespace mlir::triton::pact
