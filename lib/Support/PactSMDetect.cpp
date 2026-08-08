//===- PactSMDetect.cpp - PACT SM Architecture Detection -------------------===//
//
// Unified SM architecture detection for all PACT passes.
// Reads PACT_SM_VERSION from environment, falling back to default 86 (Ampere).
//
//===----------------------------------------------------------------------===//

#include "triton/Support/PactSMDetect.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <sstream>

namespace mlir::triton::pact {

int SMDetector::version = -1;
SMResources SMDetector::resources = {};
bool SMDetector::initialized = false;

int SMDetector::detect() {
  if (version >= 0)
    return version;

  // Primary: read from environment variable (set by Python compiler)
  const char *env = std::getenv("PACT_SM_VERSION");
  if (env) {
    version = std::atoi(env);
  } else {
    // Try CUDA_VISIBLE_DEVICES + heuristic — fallback to Ampere 86
    version = 86;
  }

  // Validate
  if (version < 70 || version > 120) {
    llvm::errs() << "[PACT SMDetect] WARNING: unexpected SM version " << version
                 << ", clamping to 86 (Ampere)\n";
    version = 86;
  }

  // Populate resource limits based on SM version
  if (version >= 100) {
    // Blackwell (B100/B200)
    resources = {
        /*smVersion*/ version,
        /*smemPerSM*/ 228 * 1024,
        /*maxRegsPerSM*/ 65536,
        /*maxWarpsPerSM*/ 64,
        /*maxThreadsPerSM*/ 2048,
        /*effectiveSmemPerBlock*/ 48 * 1024,
        /*pipelineThreshold*/ 64,
        /*occupancyCliffStages*/ 5,
        /*hasTMA*/ true,
        /*cpAsyncLatency*/ 18,
        /*optimalNumStages*/ 5,
        /*asyncCopyMinWidth*/ 8};
  } else if (version >= 90) {
    // Hopper (H100, H200)
    resources = {
        /*smVersion*/ version,
        /*smemPerSM*/ 228 * 1024,
        /*maxRegsPerSM*/ 65536,
        /*maxWarpsPerSM*/ 64,
        /*maxThreadsPerSM*/ 2048,
        /*effectiveSmemPerBlock*/ 48 * 1024,
        /*pipelineThreshold*/ 64,
        /*occupancyCliffStages*/ 5,
        /*hasTMA*/ true,
        /*cpAsyncLatency*/ 20,
        /*optimalNumStages*/ 5,
        /*asyncCopyMinWidth*/ 8};
  } else if (version >= 80) {
    // Ampere (A100, RTX 3080/3090)
    // RTX 3080 specifics: 48 warps/SM, 99KB SMEM (100KB configurable)
    // Key constraint: num_stages 3→4 drops occupancy from 1024→768 threads
    resources = {
        /*smVersion*/ version,
        /*smemPerSM*/ 99 * 1024,
        /*maxRegsPerSM*/ 65536,
        /*maxWarpsPerSM*/ 48,
        /*maxThreadsPerSM*/ 1536,
        /*effectiveSmemPerBlock*/ 24 * 1024, // to maintain 50% occupancy
        /*pipelineThreshold*/ 128,           // higher threshold on Ampere
        /*occupancyCliffStages*/ 3,          // 3→4: occupancy drops sharply
        /*hasTMA*/ false,
        /*cpAsyncLatency*/ 30,
        /*optimalNumStages*/ 3,
        /*asyncCopyMinWidth*/ 16};           // need wider vectors for cp.async
  } else {
    // Volta/Turing (SM 70-79) — conservative fallback
    resources = {
        /*smVersion*/ version,
        /*smemPerSM*/ 96 * 1024,
        /*maxRegsPerSM*/ 65536,
        /*maxWarpsPerSM*/ 32,
        /*maxThreadsPerSM*/ 1024,
        /*effectiveSmemPerBlock*/ 16 * 1024,
        /*pipelineThreshold*/ 256,
        /*occupancyCliffStages*/ 2,
        /*hasTMA*/ false,
        /*cpAsyncLatency*/ 40,
        /*optimalNumStages*/ 2,
        /*asyncCopyMinWidth*/ 16};
  }

  initialized = true;
  return version;
}

bool SMDetector::isAmpere() {
  int sm = detect();
  return sm >= 80 && sm < 90;
}

bool SMDetector::isHopper() {
  return detect() >= 90;
}

bool SMDetector::isVolta() {
  int sm = detect();
  return sm >= 70 && sm < 80;
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
  if (sm.smVersion < 90) {
    // Ampere occupancy model:
    //   num_stages=2: SMEM ~16KB → 6 blocks/SM → 1536 threads
    //   num_stages=3: SMEM ~24KB → 4 blocks/SM → 1024 threads
    //   num_stages=4: SMEM ~32KB → 3 blocks/SM → 768 threads ← CLIFF!
    budget.maxStagesByOccupancy = std::max(
        2, std::min(sm.occupancyCliffStages,
                    (int)(sm.effectiveSmemPerBlock /
                          std::max(tileBytes, (int64_t)1))));
  } else {
    // Hopper: larger SMEM, occupancy constraint is looser
    budget.maxStagesByOccupancy = 8;
  }

  // Max stages by iterations
  if (estIterations > 0)
    budget.maxStagesByIters = std::max(2, estIterations / 4);
  else
    budget.maxStagesByIters = 6;

  // Combined recommendation: take the minimum of all constraints
  budget.recommendedStages = std::min(
      {budget.maxStagesBySMEM, budget.maxStagesByOccupancy,
       budget.maxStagesByIters, sm.optimalNumStages + 1});
  budget.recommendedStages = std::max(2, budget.recommendedStages);

  return budget;
}

double SMDetector::estimateOccupancy(int numStages, int64_t smemPerBlock,
                                     int regsPerThread) {
  detect();
  const auto &sm = resources;

  int threadsPerBlock = 128; // 4 warps = 128 threads
  int64_t smemTotal = smemPerBlock + 4 * 1024; // +fixed overhead

  // Blocks limited by SMEM
  int blocksBySMEM = sm.smemPerSM / std::max(smemTotal, (int64_t)1);
  // Blocks limited by registers
  int regsPerWarp = regsPerThread * 32;
  int regsPerBlock = regsPerWarp * 4;
  int blocksByRegs = sm.maxRegsPerSM / std::max(regsPerBlock, 1);
  // Blocks limited by warps
  int blocksByWarps = sm.maxWarpsPerSM / 4;
  // Blocks limited by threads
  int blocksByThreads = sm.maxThreadsPerSM / std::max(threadsPerBlock, 1);

  int activeBlocks = std::min(
      {blocksBySMEM, blocksByRegs, blocksByWarps, blocksByThreads});
  int activeWarps = activeBlocks * 4;

  return (double)activeWarps / sm.maxWarpsPerSM;
}

std::string SMDetector::getGPUName() {
  detect();
  if (version >= 100)
    return "Blackwell (SM" + std::to_string(version) + ")";
  if (version >= 90)
    return "Hopper (SM" + std::to_string(version) + ")";
  if (version >= 80)
    return "Ampere (SM" + std::to_string(version) + ")";
  if (version >= 70)
    return "Volta (SM" + std::to_string(version) + ")";
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
