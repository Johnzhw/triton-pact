//===- PactPgoTrigger.cpp - PACT PGO trigger gate -------------------------===//
//
// PGO branch only.  Runs after P6 in make_ttgir and writes
// pact.pgo.trigger / trigger_reason / opportunity_bits onto the module.  The
// trigger is theory-as-code: contiguity recovery, stage change and warp change
// are each read from IR facts produced by P2/P3/P6/P11.  No gain constants.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Support/PactPgoDecision.h"
#include "triton/Support/PactSMDetect.h"

#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdlib>
#include <string>

#define DEBUG_TYPE "pact-pgo-trigger"

namespace mlir::triton::gpu {

#define GEN_PASS_DEF_PACTPGOTRIGGER
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

namespace {

static bool isEnabled() {
  const char *env = std::getenv("PACT_ENABLE_PGO_TRIGGER");
  return !env || std::string(env) != "0";
}

// Static trip count, or -1 when the bounds are runtime values.
static int64_t estimateTripCount(scf::ForOp forOp) {
  auto upperConst =
      forOp.getUpperBound().getDefiningOp<arith::ConstantOp>();
  auto lowerConst =
      forOp.getLowerBound().getDefiningOp<arith::ConstantOp>();
  if (!upperConst || !lowerConst)
    return -1;
  int64_t upperVal =
      mlir::cast<mlir::IntegerAttr>(upperConst.getValue()).getInt();
  int64_t lowerVal =
      mlir::cast<mlir::IntegerAttr>(lowerConst.getValue()).getInt();
  int64_t step = 1;
  if (auto stepOp = forOp.getStep().getDefiningOp<arith::ConstantOp>())
    step = mlir::cast<mlir::IntegerAttr>(stepOp.getValue()).getInt();
  if (step <= 0)
    return -1;
  return (upperVal - lowerVal) / step;
}

struct PACTPgoTriggerPass
    : public impl::PACTPgoTriggerBase<PACTPgoTriggerPass> {

  void runOnOperation() override {
    if (!isEnabled())
      return;

    ModuleOp mod = getOperation();
    pact::PgoTriggerInput input;

    int64_t pgoIterations = -1;
    if (auto attr = mod->getAttrOfType<IntegerAttr>(
            "pact.pgo.measured_iterations"))
      pgoIterations = attr.getInt();
    if (pgoIterations > 0)
      input.estIterations = pgoIterations;
    else
      mod.walk([&](scf::ForOp forOp) {
        input.estIterations =
            std::max(input.estIterations, estimateTripCount(forOp));
      });

    // P3 baseline fact (max over paged loads, published by AxisInfo).
    if (auto attr = mod->getAttrOfType<IntegerAttr>(
            "pact.axisinfo.baseline_contiguity"))
      input.baselineContig = attr.getInt();

    // P2 page-bounded contiguity for the head_dim of any paged load.
    mod.walk([&](triton::LoadOp loadOp) {
      if (!loadOp->hasAttr("pact.paged_load"))
        return WalkResult::advance();
      int64_t headDimIdx = 1;
      if (auto hd = loadOp->getAttrOfType<IntegerAttr>("pact.head_dim_idx"))
        headDimIdx = hd.getInt();
      if (auto contig = loadOp->getAttrOfType<DenseI64ArrayAttr>(
              "pact.pagelocal.dim_contiguity"))
        if (headDimIdx >= 0 && headDimIdx < (int)contig.size())
          input.pageContig =
              std::max(input.pageContig, contig[headDimIdx]);
      return WalkResult::advance();
    });

    // N3: compare chosenStages against the same native baseline P6 used.
    // P6 publishes pact.native_num_stages; only when P6 did not run fall back
    // to the same architecture default AutoNumStages would have computed.
    // The module-level tt.num_stages attribute is never written by the
    // pipeline, but is honored as a legacy override for external producers.
    int defaultStages = 3;
    auto smResources = pact::SMDetector::getResources();
    if (smResources.smVersion >= 90)
      defaultStages = smResources.optimalNumStages;
    if (auto attr = mod->getAttrOfType<IntegerAttr>("pact.native_num_stages"))
      defaultStages = attr.getInt();
    else if (auto attr = mod->getAttrOfType<IntegerAttr>("tt.num_stages"))
      defaultStages = attr.getInt();
    input.defaultStages = defaultStages;
    input.chosenStages = input.defaultStages;
    if (auto attr = mod->getAttrOfType<IntegerAttr>("pact.optimal_num_stages"))
      input.chosenStages = attr.getInt();

    input.numWarps = 4;
    if (auto attr = mod->getAttrOfType<IntegerAttr>("ttg.num-warps"))
      input.numWarps = attr.getInt();
    input.chosenWarps = input.numWarps;
    if (auto attr = mod->getAttrOfType<IntegerAttr>("pact.optimal_num_warps"))
      input.chosenWarps = attr.getInt();

    auto decision = pact::selectPgoTrigger(input);

    auto i32 = IntegerType::get(&getContext(), 32);
    mod->setAttr("pact.pgo.trigger",
                 IntegerAttr::get(i32, decision.trigger ? 1 : 0));
    mod->setAttr("pact.pgo.trigger_reason",
                 StringAttr::get(&getContext(), decision.reason));
    mod->setAttr("pact.pgo.opportunity_bits",
                 IntegerAttr::get(i32, decision.opportunityBits));

    llvm::errs() << "[PACT PGO trigger] trigger="
                 << (decision.trigger ? "true" : "false")
                 << " reason=" << decision.reason
                 << " bits=" << decision.opportunityBits
                 << " (pageContig=" << input.pageContig
                 << ", baselineContig=" << input.baselineContig
                 << ", iters=" << input.estIterations
                 << ", stages=" << input.defaultStages << "->"
                 << input.chosenStages
                 << ", warps=" << input.numWarps << "->"
                 << input.chosenWarps << ")\n";
  }
};

} // namespace

} // namespace mlir::triton::gpu
