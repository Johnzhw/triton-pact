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
};

// num_stages decision.
//
// Feasible set (computed, not hard-coded):
//   s in [2, min(user max, smemBudget/tileBytes, estIterations)]
// Ampere (SM 80-89, occupancy-first):
//   choose s maximizing the L2 occupancy; ties prefer the stage count closest
//   to defaultStages (then the smaller one).  Short sequences therefore
//   converge to low stages through the equations instead of an `<=16` rule.
// Hopper (SM >= 90, pipeline-first):
//   choose the largest s whose occupancy is within the computed discretization
//   granularity (numWarps/maxWarpsPerSM) of the best occupancy.
// AMD / Unknown backend: keep defaultStages (CDNA capacity data pending).
SelectStagesResult selectNumStages(const SelectStagesInput &input);

} // namespace mlir::triton::pact

#endif // TRITON_SUPPORT_PACTDECISION_H
