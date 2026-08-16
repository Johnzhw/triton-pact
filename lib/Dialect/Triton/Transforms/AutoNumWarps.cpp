//===- AutoNumWarps.cpp - PACT P11 num_warps Selection --------------------===//
//
// P11 (TTIR): recommends a num_warps value for paged attention kernels and
// writes pact.optimal_num_warps on the module; compiler.py reads it before
// make_ttgir to override opt.num_warps.
//
// Inputs (theory-only path, no profile-collected hardware parameters):
//   1. paged-load tile geometry from IR attributes
//   2. the SMDetector hardware-capacity occupancy model (L2 equations)
//
// Without a resolvable signal the recommendation stays at the Triton default
// (4).
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "triton/Support/PactDecision.h"
#include "triton/Support/PactSMDetect.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <string>
#include <algorithm>

namespace mlir::triton {

#define GEN_PASS_DEF_PACTAUTONUMWARPS
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

namespace {

static bool isEnabled() {
  const char *env = std::getenv("PACT_ENABLE_AUTO_NUM_WARPS");
  return env && std::string(env) != "0";
}

struct PACTAutoNumWarpsPass
    : public impl::PACTAutoNumWarpsBase<PACTAutoNumWarpsPass> {

  void runOnOperation() override {
    if (!isEnabled()) {
      llvm::errs() << "[PACT P11] AutoNumWarps: disabled\n";
      return;
    }

    ModuleOp mod = getOperation();
    int numPagedLoads = 0;
    int64_t maxTileBytes = 0;

    mod.walk([&](Operation *op) {
      if (!op->hasAttr("pact.paged_load"))
        return WalkResult::advance();
      numPagedLoads++;
      auto resultTy = dyn_cast<RankedTensorType>(op->getResultTypes()[0]);
      if (!resultTy)
        return WalkResult::advance();

      int64_t headDim = 64;
      if (auto attr =
              op->getAttrOfType<IntegerAttr>("pact.head_dim_size"))
        headDim = attr.getInt();
      int64_t tile = 1;
      if (resultTy.getShape().size() >= 2)
        tile = resultTy.getShape()[0];
      int64_t elemBytes =
          std::max((int64_t)1,
                   (int64_t)(resultTy.getElementTypeBitWidth() / 8));
      maxTileBytes = std::max(maxTileBytes, tile * headDim * elemBytes);
      return WalkResult::advance();
    });

    int optimalWarps = pact::PactDecisionConstants::kDefaultNumWarps;
    // The canonical paged-attention tile has exactly two annotated K/V loads.
    // Requiring >=4 loads silently disabled P11 for the primary target shape.
    if (numPagedLoads >= 1 && maxTileBytes > 0) {
      // Theory-only path: the L2 capacity equations compute occ(w) for every
      // legal warp count and the required gain is the equations' own
      // discretization granularity.  The register count is unknown here, so
      // it is passed as the explicit named model-input assumption.
      auto decision = pact::selectNumWarps(
          maxTileBytes, pact::PactDecisionConstants::kUnknownRegsPerThread);
      optimalWarps = decision.numWarps;
      llvm::errs() << "[PACT P11] selectNumWarps: " << optimalWarps
                   << " (occ4=" << decision.baselineOccupancy
                   << ", occChosen=" << decision.chosenOccupancy
                   << ", requiredGain=" << decision.requiredGain
                   << ", switched=" << (decision.switched ? "yes" : "no")
                   << ")\n";
    }

    mod->setAttr("pact.optimal_num_warps",
                 IntegerAttr::get(IntegerType::get(&getContext(), 32),
                                  optimalWarps));

    llvm::errs() << "[PACT P11] num_warps recommendation: " << optimalWarps
                 << " (pagedLoads=" << numPagedLoads
                 << ", maxTileB=" << maxTileBytes << ")\n";
  }
};

} // anonymous namespace
} // namespace mlir::triton
