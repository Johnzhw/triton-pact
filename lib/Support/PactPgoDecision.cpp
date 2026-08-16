//===- PactPgoDecision.cpp - PACT PGO trigger theory ----------------------===//
//
// See PactPgoDecision.h.  Theory-as-code: every trigger bit is computed from
// P1/P2/P3/P6/P11 facts and no threshold constant is introduced.
//
//===----------------------------------------------------------------------===//

#include "triton/Support/PactPgoDecision.h"

namespace mlir::triton::pact {

PgoTriggerResult selectPgoTrigger(const PgoTriggerInput &input) {
  PgoTriggerResult result;
  result.trigger = false;
  result.reason = "none";
  result.opportunityBits = 0;

  if (input.estIterations < 2)
    return result;

  bool contigOpportunity =
      input.pageContig > input.baselineContig && input.baselineContig > 0;
  bool stageOpportunity = input.chosenStages != input.defaultStages;
  bool warpOpportunity = input.chosenWarps != input.numWarps;

  std::string reason;
  if (contigOpportunity) {
    result.opportunityBits |= 1;
    reason += "contig";
  }
  if (stageOpportunity) {
    result.opportunityBits |= 2;
    if (!reason.empty())
      reason += "|";
    reason += "stage";
  }
  if (warpOpportunity) {
    result.opportunityBits |= 4;
    if (!reason.empty())
      reason += "|";
    reason += "warp";
  }

  result.trigger = result.opportunityBits != 0;
  result.reason = reason.empty() ? "none" : reason;
  return result;
}

} // namespace mlir::triton::pact
