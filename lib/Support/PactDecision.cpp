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

double occupancyFor(int64_t smemPerBlock, int64_t regsPerThread, int numStages,
                    int numWarps) {
  // smemPerBlock already includes numStages (callers fold it), matching
  // SMDetector::estimateOccupancy's documented contract.
  return SMDetector::estimateOccupancy(numStages, smemPerBlock, regsPerThread,
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
  result.baselineOccupancy = occupancyFor(tileBytes * stagesPerBlock,
                                          regsPerThread, stagesPerBlock,
                                          baseline);
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
  constexpr double kTieEps = 1e-9;
  // Strictly-better candidates must beat the computed one-CTA error bound.
  // A separate S3a tie rule only switches on an exact occupancy tie, where
  // lowering the warp count cannot cost model occupancy; the order {2, 1, 8}
  // therefore prefers 2 warps on ties.  No negative-tolerance band is used.
  for (int candidate : {2, 1, 8}) {
    if (candidate == baseline)
      continue;
    double occ = occupancyFor(tileBytes * stagesPerBlock, regsPerThread,
                              stagesPerBlock, candidate);
    double gain = occ - result.baselineOccupancy;
    bool exactTie = std::abs(gain) <= kTieEps;
    if ((gain > result.requiredGain ||
         (exactTie && candidate < baseline && !result.switched)) &&
        gain > bestGain) {
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

  const int64_t regsPerThread = input.regsPerThread > 0
                                    ? input.regsPerThread
                                    : PactDecisionConstants::kUnknownRegsPerThread;
  auto clamp01 = [](double x) { return std::clamp(x, 0.0, 1.0); };
  const double stallFrac =
      input.stallMemoryPermille.has_value()
          ? clamp01(static_cast<double>(*input.stallMemoryPermille) / 1000.0)
          : 0.0;
  const double smEffFrac =
      input.smEfficiencyPermille.has_value()
          ? clamp01(static_cast<double>(*input.smEfficiencyPermille) / 1000.0)
          : 0.0;

  // S1 L2-residency classification.  l2Bytes == 0 (unknown backend/AMD) or
  // kvWorkingSetBytes == 0 (no static estimate) both mean "hot": the scan
  // below then runs exactly as it did before S1 — this is the
  // default-path-unchanged guarantee, not a silent heuristic.
  result.kvWorkingSetBytes = input.kvWorkingSetBytes;
  result.l2Bytes = sm.l2Bytes;
  // V18 T8c: the weights share L2 with the KV stream during a decode
  // step -- the verdict sees the combined pressure.  weightBytes == 0
  // (no hint) leaves the arithmetic bit-identical to the S1 form.
  int64_t residentCompetitor = input.kvWorkingSetBytes + input.weightBytes;
  result.l2Cold = input.kvWorkingSetBytes > 0 && sm.l2Bytes > 0 &&
                  residentCompetitor > sm.l2Bytes;

  auto scoreFor = [&](int s) {
    double occ = occupancyFor(input.tileBytes * s, regsPerThread, s,
                              input.numWarps);
    int extra = std::max(0, s - input.defaultStages);
    // Caller-supplied linear terms; both coefficients default to 0 so the
    // ranking is identical to the occupancy-only scan when no table is loaded.
    occ -= input.stallPenaltyPerExtraStage * stallFrac *
           static_cast<double>(extra);
    occ += input.smEffBonusPerExtraStage * (1.0 - smEffFrac) *
           static_cast<double>(extra);
    return occ;
  };

  // Scan the computed feasible set with the L2 capacity equations.
  double bestOcc = -std::numeric_limits<double>::infinity();
  int bestStages = input.defaultStages;
  for (int s = 2; s <= result.feasibleMax; ++s) {
    double occ = scoreFor(s);
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
      double occ = scoreFor(s);
      if (occ >= bestOcc - tolerance) {
        chosen = s;
        result.occupancy = occ;
        break;
      }
    }
  } else {
    result.occupancy = scoreFor(bestStages);
    chosen = bestStages;
    // S1 cold path: the K/V working set exceeds L2, so the loop streams from
    // DRAM.  Within the computed feasible set prefer a deeper pipeline (up to
    // 5) — deeper cp.async staging hides DRAM latency.  The acceptance gate
    // mirrors the Hopper pipeline-first policy: a depth is adopted when its
    // score stays within the capacity equations' one-CTA discretization
    // granularity of the scan's best (the computed model-error bound, not a
    // hand-tuned ratio); anything worse is vetoed.
    if (result.l2Cold) {
      double tolerance = modelGranularity(input.numWarps);
      int deepMax = std::min(result.feasibleMax, 5);
      for (int s = deepMax; s >= 3; --s) {
        double occ = scoreFor(s);
        if (occ >= bestOcc - tolerance) {
          chosen = s;
          result.occupancy = occ;
          result.l2Deepened = (s != bestStages);
          break;
        }
      }
    }
  }

  result.numStages = chosen;
  result.defaultKept = (chosen == input.defaultStages);
  return result;
}

} // namespace mlir::triton::pact
