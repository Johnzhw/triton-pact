//===- PactSMDetect.h - PACT SM Architecture Detection -------------*- C++ -*-===//
//
// Unified SM architecture detection for all PACT passes.
// Provides hardware resource limits, occupancy estimation, and pipeline budget
// computation tailored for page-attention kernels.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_SUPPORT_PACTSMDETECT_H
#define TRITON_SUPPORT_PACTSMDETECT_H

#include <cstdint>
#include <string>

namespace mlir::triton::pact {

// ============================================================================
// SMResources: hardware resource limits + PACT-specific derived parameters
// ============================================================================
struct SMResources {
  // === Raw hardware parameters ===
  int smVersion;          // 86 (Ampere), 90 (Hopper)
  int smemPerSM;          // bytes: 99KB (Ampere RTX 3080), 228KB (Hopper)
  int maxRegsPerSM;       // 65536 (both)
  int maxWarpsPerSM;      // 48 (RTX 3080), 64 (A100), 64 (H100)
  int maxThreadsPerSM;    // 1536 (Ampere), 2048 (Hopper)

  // === PACT derived parameters ===
  int effectiveSmemPerBlock; // usable SMEM considering block table overhead
  int pipelineThreshold;     // minimum tile bytes × iterations to trigger pipeline
  int occupancyCliffStages;  // num_stages threshold where occupancy drops sharply
  bool hasTMA;               // Hopper TMA hardware support
  int cpAsyncLatency;        // cp.async latency in cycles: ~30 (Ampere), ~20 (Hopper)

  // === Triton Pipeline characteristics ===
  int optimalNumStages;      // default optimal stages: 2-3 (Ampere), 4-5 (Hopper)
  int asyncCopyMinWidth;     // minimum vector width (bytes) to trigger cp.async
};

// ============================================================================
// PipelineBudget: page-attention pipeline resource budget model
// ============================================================================
struct PipelineBudget {
  int64_t smemBudget;        // SMEM available for pipeline stages (bytes)
  int maxStagesBySMEM;       // max stages constrained by SMEM
  int maxStagesByOccupancy;  // max stages to maintain target occupancy
  int maxStagesByIters;      // max stages constrained by iteration count
  int recommendedStages;     // combined recommendation

  std::string toJSON() const;
};

// ============================================================================
// SMDetector: singleton SM detection
// ============================================================================
class SMDetector {
  static int version;
  static SMResources resources;
  static bool initialized;

public:
  // Detect SM version and populate resources.
  // Reads PACT_SM_VERSION env var, or defaults to 86 (Ampere).
  static int detect();

  // Convenience queries
  static bool isAmpere();     // SM 80-89
  static bool isHopper();     // SM >= 90
  static bool isVolta();      // SM 70-79

  // Get resource limits
  static const SMResources &getResources();

  // PACT-specific: compute pipeline budget for a page-attention tile
  static PipelineBudget computePipelineBudget(
      int64_t tileBytes, int estIterations,
      int pageSize, int tileTokens,
      int defaultStages);

  // Estimate occupancy given num_stages and per-block resource usage
  static double estimateOccupancy(int numStages, int64_t smemPerBlock,
                                  int regsPerThread);

  // Get human-readable GPU name for logging
  static std::string getGPUName();
};

} // namespace mlir::triton::pact

#endif // TRITON_SUPPORT_PACTSMDETECT_H
