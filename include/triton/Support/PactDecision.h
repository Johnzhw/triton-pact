//===- PactDecision.h - PACT theory-as-code decision core -----------*- C++ -*-===//
//
// PACT decision functions shared by P6 (num_stages) and P11 (num_warps).
//
// Layering of theory in PACT (see docs/pact/UNIFIED.md and the paper appendix):
//   L1. F2 LinearLayout theorems (page-bounded vectorization width V):
//       implemented in PageLocalAnalysis.cpp / CoalesceUtils.cpp — every
//       theorem quantity is computed at compile time by the LinearLayout code,
//       never hard-coded.
//   L2. Hardware-capacity equations (occupancy model): implemented in
//       PactSMDetect::estimateOccupancy as min{SMEM, regs, warps, threads}.
//       This file turns those equations into decisions by *computing* the
//       candidates, their predicted occupancy, and the model's discretization
//       error bound.  No hand-calibrated magic thresholds.
//   L3. Measured correction (PGO branch only): measured iterations /
//       registers / active-warp ratio are inputs to the same L2 equations and
//       to the measured model-error bound below; Profile-Safety Lemma covers
//       them.
//
// kUnknownRegsPerThread is an explicit *model input assumption* used when the
// register count is unknown (non-PGO tree, or a failed PGO measurement).  It
// is not a theorem constant and lives here in one named place.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_SUPPORT_PACTDECISION_H
#define TRITON_SUPPORT_PACTDECISION_H

#include <cstdint>
#include <optional>

namespace mlir::triton::pact {

namespace PactDecisionConstants {
// Explicit model input for unknown per-thread register usage.
inline constexpr int64_t kUnknownRegsPerThread = 64;
// Triton default num_warps; also the tie-break baseline.
inline constexpr int kDefaultNumWarps = 4;
// Register constraint is ignored by estimateOccupancy when <= 0.
inline constexpr int64_t kUnboundedRegsPerThread = 0;
} // namespace PactDecisionConstants

struct SelectWarpsResult {
  int numWarps = PactDecisionConstants::kDefaultNumWarps;
  double baselineOccupancy = 0.0; // occ(num_warps=4)
  double chosenOccupancy = 0.0;   // occ(selected candidate)
  double requiredGain = 0.0;      // computed model-error bound that any switch
                                  // must exceed (never a hard-coded ratio)
  bool switched = false;
};

// num_warps decision.
//
// Theory: for each legal num_warps w in {2, 1, 8} compute the L2 capacity
// occupancy occ(w).  The capacity equations round resource ratios down, so
// their discretization error is bounded by one CTA: granularity =
// w / maxWarpsPerSM.  Switch from the default (4) only when
//     occ(w) - occ(4) > requiredGain,
// where requiredGain = max(granularity, measured model error) in the PGO tree
// (measuredActiveWarpRatio) and requiredGain = granularity in the theory-only
// tree.  With no PGO input this reduces to a pure function of the IR tile and
// the architecture resource table.
//
// measuredActiveWarpRatio is expected in [0, 1]; out-of-range values are
// clamped to the valid domain before use.
SelectWarpsResult selectNumWarps(
    int64_t tileBytes, int64_t regsPerThread,
    std::optional<double> measuredActiveWarpRatio = std::nullopt,
    int stagesPerBlock = 3);

struct SelectStagesInput {
  int64_t tileBytes = 0;    // exact swizzled SMEM footprint of one stage
  int64_t estIterations = 0;
  int defaultStages = 3;
  int maxStages = 4;        // user/knob upper bound (PACT_MAX_PIPELINE_STAGES)
  int numWarps = PactDecisionConstants::kDefaultNumWarps;
  // Register count used by the L2 occupancy scan.  Defaults to the named
  // unknown-register assumption; a measured value (PGO / dynamic tree) is
  // substituted by the caller.  Unused optional metrics leave this path
  // bit-identical to the theory-only scan.
  int64_t regsPerThread = PactDecisionConstants::kUnknownRegsPerThread;
  // Optional measured counters in permille [0, 1000].  Absent -> ignored.
  std::optional<int> stallMemoryPermille;
  std::optional<int> smEfficiencyPermille;
  // Coefficients supplied by the caller (offline family table).  Zero keeps
  // the occupancy ranking unchanged.  C++ does not hard-code 0.30 / 0.50.
  double stallPenaltyPerExtraStage = 0.0;
  double smEffBonusPerExtraStage = 0.0;
  // S1 L2-residency input: the KV working set the loop streams (K+V bytes of
  // one sequence, scaled by the kv_heads hint when provided).  0 (default)
  // means "unknown" and leaves the selection bit-identical to the pre-S1
  // occupancy scan.  A positive value not exceeding the detected L2 capacity
  // (hot path) also leaves the scan unchanged — v12/v13 ncu evidence shows
  // the PACT gains live on the L2-hit path, so the current choice is already
  // optimal there.  Only a working set strictly larger than L2 (streaming
  // from DRAM) biases the Ampere scan toward deeper pipelines within the
  // computed feasible set.
  int64_t kvWorkingSetBytes = 0;
};

struct SelectStagesResult {
  int numStages = 2;
  double occupancy = 0.0;      // occupancy of the selected stage count
  double bestOccupancy = 0.0;  // best occupancy over the feasible set
  int feasibleMin = 2;
  int feasibleMax = 2;
  int smemBound = 2;
  int iterBound = 2;
  bool defaultKept = true;
  // S1 diagnostics (informational; the l2Cold fields mirror the inputs that
  // were actually used so logs can audit the residency decision).
  int64_t kvWorkingSetBytes = 0;
  int64_t l2Bytes = 0;
  bool l2Cold = false;         // working set strictly exceeded L2
  bool l2Deepened = false;     // cold path adopted a deeper pipeline
};

// num_stages decision.
//
// Feasible set (computed, not hard-coded):
//   s in [2, min(user max, smemBudget/tileBytes, estIterations)]
// Pre-Hopper NVIDIA (SM < 90, occupancy-first — includes Ada, whose
//   l2Bytes table entry can classify cold shapes): choose s maximizing
//   the L2 occupancy; ties prefer the stage count closest to
//   defaultStages (then the smaller one).  Short sequences therefore
//   converge to low stages through the equations instead of an `<=16` rule.
//   S1 cold path (kvWorkingSetBytes > l2Bytes > 0, SM < 90 only): the loop
//   streams K/V from DRAM, so among feasible s in [3,5] take the deepest one
//   whose score stays within the computed one-CTA discretization granularity
//   of the scan's best — deeper cp.async pipelines hide DRAM latency, and any
//   depth that the capacity equations say costs more than the model-error
//   bound is vetoed.
// Hopper+ (SM >= 90, pipeline-first):
//   choose the largest s whose occupancy is within the computed discretization
//   granularity (numWarps/maxWarpsPerSM) of the best occupancy.
// AMD / Unknown backend: keep defaultStages (CDNA capacity data pending).
SelectStagesResult selectNumStages(const SelectStagesInput &input);

} // namespace mlir::triton::pact

#endif // TRITON_SUPPORT_PACTDECISION_H
