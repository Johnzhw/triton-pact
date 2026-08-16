//===- PactDecision.cpp - PACT theory-as-code decision core ----------------===//
//
// See PactDecision.h for the L1/L2/L3 layering.  This file contains no
// hard-coded gain ratios: the required gain is the capacity equations'
// discretization error bound (one CTA), optionally tightened by a measured
// model error in the PGO tree.
//
//===----------------------------------------------------------------------===//

#include "triton/Support/PactDecision.h"
#include "triton/Support/PactSMDetect.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace mlir::triton::pact {

namespace {

double occupancyFor(int64_t tileBytes, int64_t regsPerThread, int numStages,
                    int numWarps) {
  // numStages is folded into the per-block SMEM by the caller, matching
  // SMDetector::estimateOccupancy's documented contract.
  return SMDetector::estimateOccupancy(numStages, tileBytes, regsPerThread,
                                       numWarps);
}

double modelGranularity(int numWarps) {
  const SMResources &sm = SMDetector::getResources();
  int maxWarps = std::max(sm.maxWarpsPerSM, 1);
  return static_cast<double>(numWarps) / static_cast<double>(maxWarps);
}

} // namespace

SelectWarpsResult selectNumWarps(int64_t tileBytes, int64_t regsPerThread,
                                 std::optional<double> measuredActiveWarpRatio,
                                 int stagesPerBlock) {
  SelectWarpsResult result;
  const int baseline = PactDecisionConstants::kDefaultNumWarps;
  result.baselineOccupancy = occupancyFor(tileBytes, regsPerThread,
                                          stagesPerBlock, baseline);
  result.chosenOccupancy = result.baselineOccupancy;
  result.numWarps = baseline;

  // Discretization error of the capacity equations: they round resource
  // ratios down, so the true occupancy may differ by at most one CTA.
  result.requiredGain = modelGranularity(baseline);

  // PGO correction (L3): the measured active-warp ratio at the baseline gives
  // a direct estimate of the model error.  Use the larger of the computed
  // granularity and the measured error so an over-optimistic measurement can
  // never make the decision *less* conservative than the equations alone.
  if (measuredActiveWarpRatio.has_value()) {
    double ratio = std::clamp(*measuredActiveWarpRatio, 0.0, 1.0);
    double measuredError = std::max(0.0, result.baselineOccupancy - ratio);
    result.requiredGain = std::max(result.requiredGain, measuredError);
  }

  double bestGain = -std::numeric_limits<double>::infinity();
  // Order {2, 1, 8}: on exact ties prefer num_warps=2 (the closest useful
  // low-warp candidate), then 1, then 8.
  for (int candidate : {2, 1, 8}) {
    if (candidate == baseline)
      continue;
    double occ = occupancyFor(tileBytes, regsPerThread, stagesPerBlock,
                              candidate);
    double gain = occ - result.baselineOccupancy;
    if (gain > result.requiredGain && gain > bestGain) {
      bestGain = gain;
      result.numWarps = candidate;
      result.chosenOccupancy = occ;
      result.switched = true;
    }
  }
  return result;
}

SelectStagesResult selectNumStages(const SelectStagesInput &input) {
  SelectStagesResult result;
  result.numStages = input.defaultStages;
  result.defaultKept = true;

  SMDetector::detect();
  TargetBackend backend = SMDetector::getBackend();
  const SMResources &sm = SMDetector::getResources();

  // AMD and Unknown keep the native default until per-SKU capacity data and a
  // validated CDNA-specific model are available.
  if (backend != TargetBackend::NVIDIA)
    return result;

  PipelineBudget budget = SMDetector::computePipelineBudget(
      input.tileBytes, static_cast<int>(input.estIterations),
      /*pageSize=*/16, /*tileTokens=*/16, input.defaultStages);

  result.smemBound = budget.maxStagesBySMEM;
  result.iterBound = budget.maxStagesByIters;
  result.feasibleMax =
      std::min({input.maxStages, result.smemBound, result.iterBound});
  if (result.feasibleMax < 2)
    return result;
  result.feasibleMin = 2;

  // Scan the computed feasible set with the L2 capacity equations.
  double bestOcc = -std::numeric_limits<double>::infinity();
  int bestStages = input.defaultStages;
  for (int s = 2; s <= result.feasibleMax; ++s) {
    double occ = occupancyFor(input.tileBytes * s,
                              PactDecisionConstants::kUnknownRegsPerThread, s,
                              input.numWarps);
    constexpr double kEps = 1e-9;
    if (occ > bestOcc + kEps) {
      bestOcc = occ;
      bestStages = s;
    } else if (std::abs(occ - bestOcc) <= kEps) {
      // Occupancy tie: prefer the candidate closest to the native default.
      int oldDist = std::abs(bestStages - input.defaultStages);
      int newDist = std::abs(s - input.defaultStages);
      if (newDist < oldDist || (newDist == oldDist && s < bestStages))
        bestStages = s;
    }
  }
  result.bestOccupancy = bestOcc;
  result.occupancy = bestOcc;

  int chosen = bestStages;
  if (sm.smVersion >= 90) {
    // Hopper pipeline-first policy: among the feasible set take the largest
    // stage count whose occupancy stays within the computed one-CTA
    // discretization granularity of the best occupancy.
    double tolerance = modelGranularity(input.numWarps);
    for (int s = result.feasibleMax; s >= 2; --s) {
      double occ = occupancyFor(input.tileBytes * s,
                                PactDecisionConstants::kUnknownRegsPerThread,
                                s, input.numWarps);
      if (occ >= bestOcc - tolerance) {
        chosen = s;
        result.occupancy = occ;
        break;
      }
    }
  } else {
    result.occupancy = occupancyFor(
        input.tileBytes * bestStages,
        PactDecisionConstants::kUnknownRegsPerThread, bestStages,
        input.numWarps);
    chosen = bestStages;
  }

  result.numStages = chosen;
  result.defaultKept = (chosen == input.defaultStages);
  return result;
}

} // namespace mlir::triton::pact
