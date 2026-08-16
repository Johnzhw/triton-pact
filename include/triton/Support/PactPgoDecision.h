//===- PactPgoDecision.h - PACT PGO trigger theory ----------------- C++ -*-===//
//
// PGO-branch-only decision core.  The trigger decides *whether* it is worth
// running the expensive profile collection + recompilation; it never decides
// the final swap (that is the measured gain gate in Python).
//
// All trigger opportunities are computed from existing theory/IR facts:
//   contig: page-bounded contiguity > AxisInfo's pre-override baseline
//   stage : selectNumStages chose a count different from the native default
//   warp  : selectNumWarps switched away from the default 4 warps
// There are no hand-coded gain constants.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_SUPPORT_PACTPGODECISION_H
#define TRITON_SUPPORT_PACTPGODECISION_H

#include <cstdint>
#include <string>

namespace mlir::triton::pact {

struct PgoTriggerInput {
  int64_t pageContig = 1;     // pact.pagelocal.dim_contiguity[head_dim]
  int64_t baselineContig = 1; // pact.axisinfo.baseline_contiguity (module max)
  int64_t estIterations = 0;  // static trip count or measured_iterations
  int defaultStages = 3;      // native/arch default
  int chosenStages = 3;       // selectNumStages output
  int numWarps = 4;           // default warp count
  int chosenWarps = 4;        // selectNumWarps output
};

struct PgoTriggerResult {
  bool trigger = false;
  std::string reason;   // "none" or "|"-joined opportunity names
  int opportunityBits = 0; // bit0 contig, bit1 stage, bit2 warp
};

PgoTriggerResult selectPgoTrigger(const PgoTriggerInput &input);

} // namespace mlir::triton::pact

#endif // TRITON_SUPPORT_PACTPGODECISION_H
