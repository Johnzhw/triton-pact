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
// TargetBackend: target hardware backend abstraction (M0d — AMD migration hook)
//
// NVIDIA uses a numeric SM version (70-120); AMD uses a gfx arch string
// (gfx942/gfx950/gfx1250).  This enum lets SMDetector distinguish backends
// without hardcoding NVIDIA-only semantics.  AMD resource tables are filled in
// at migration time (see PactSMDetect.cpp resourcesForAMD).
// ============================================================================
enum class TargetBackend { NVIDIA, AMD, Unknown };

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
  int waveSize;           // 32 (NVIDIA), 64 (AMD CDNA3/gfx9-class)
  // Reserved interface: device compute-unit count (AMD capacity hint).
  // Intentionally unread by the current heuristics until official per-SKU
  // capacity data (CU count, LDS, VGPR/AGPR, matrix cores) is available.
  int numCUs;
  // L2 cache capacity in bytes (S1 L2-residency input for P6).  Family
  // minimums follow the CUDA C Programming Guide's per-compute-capability
  // convention (SM86 = GA10x family value).  0 means "unknown" (conservative
  // table, AMD pending) — downstream residency logic must treat that as
  // "hot path" and leave the decision unchanged.
  int64_t l2Bytes;

  // === PACT derived parameters ===
  int effectiveSmemPerBlock; // usable SMEM considering block table overhead
  int pipelineThreshold;     // minimum tile bytes × iterations to trigger pipeline
  int occupancyCliffStages;  // num_stages threshold where occupancy drops sharply
  // Reserved interface: TMA hardware support flag.
  // Intentionally unread by the current heuristics until a TMA-specific
  // PACT policy is designed; do not remove.
  bool hasTMA;
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
  static TargetBackend backend;
  static std::string gfxArch;   // AMD gfx arch string (e.g. "gfx942"); empty for NVIDIA
  static SMResources resources;
  static bool initialized;

public:
  // Detect SM version and populate resources.
  // Reads PACT_SM_VERSION (NVIDIA) or PACT_AMD_ARCH (AMD gfx string).
  // If neither is set, logs a warning and falls back to a conservative
  // unknown backend (no silent RTX-3080 assumption).
  static int detect();

  // Which target backend was detected (NVIDIA / AMD / Unknown).
  static TargetBackend getBackend();

  // Convenience queries
  static bool isAmpere();     // SM 80-89 (NVIDIA)
  static bool isHopper();     // SM >= 90 (NVIDIA)
  static bool isVolta();      // SM 70-79 (NVIDIA)

  // Get resource limits
  static const SMResources &getResources();

  // PACT-specific: compute pipeline budget for a page-attention tile
  static PipelineBudget computePipelineBudget(
      int64_t tileBytes, int estIterations,
      int pageSize, int tileTokens,
      int defaultStages);

  // Estimate occupancy given num_stages, per-block SMEM usage, registers per
  // thread, and num_warps per CTA.  num_warps defaults to 4 for backward
  // compatibility; callers that select num_warps (e.g. P11) should pass it.
  static double estimateOccupancy(int numStages, int64_t smemPerBlock,
                                  int regsPerThread, int numWarps = 4);

  // Get human-readable GPU name for logging
  static std::string getGPUName();
};

} // namespace mlir::triton::pact

#endif // TRITON_SUPPORT_PACTSMDETECT_H
