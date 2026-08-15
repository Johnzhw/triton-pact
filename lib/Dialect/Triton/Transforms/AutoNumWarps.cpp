//===- AutoNumWarps.cpp - PACT P11 num_warps Selection --------------------===//
//
// P11 (TTIR): recommends a num_warps value for paged attention kernels and
// writes pact.optimal_num_warps on the module; compiler.py reads it before
// make_ttgir to override opt.num_warps.
//
// Inputs (in priority order):
//   1. pact.pgo.regs_per_thread / pact.pgo.active_warp_ratio (PGO branch)
//   2. conservative SMDetector occupancy estimates (static theory-only path)
//
// Without either signal the recommendation stays at the Triton default (4).
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
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

    // Optional PGO-provided facts (injected only by the PGO branch).
    int64_t pgoRegsPerThread = -1;
    if (auto attr =
            mod->getAttrOfType<IntegerAttr>("pact.pgo.regs_per_thread"))
      pgoRegsPerThread = attr.getInt();
    double pgoActiveWarpRatio = -1.0;
    if (auto attr =
            mod->getAttrOfType<FloatAttr>("pact.pgo.active_warp_ratio"))
      pgoActiveWarpRatio = attr.getValueAsDouble();

    int optimalWarps = 4;
    if (numPagedLoads >= 4 && maxTileBytes > 0) {
      int64_t regsPerThread = pgoRegsPerThread > 0 ? pgoRegsPerThread : 64;
      double occ4 =
          pact::SMDetector::estimateOccupancy(/*numStages=*/3, maxTileBytes,
                                              regsPerThread, /*numWarps=*/4);
      double occ2 =
          pact::SMDetector::estimateOccupancy(/*numStages=*/3, maxTileBytes,
                                              regsPerThread, /*numWarps=*/2);

      // Static theory: switch only for a large occupancy gain.  PGO makes the
      // threshold more willing when the measured warp occupancy is actually
      // low, and refuses the switch when occupancy is already healthy.
      double gainThreshold = 1.30;
      if (pgoActiveWarpRatio >= 0.0) {
        if (pgoActiveWarpRatio < 0.5)
          gainThreshold = 1.15;
        else if (pgoActiveWarpRatio >= 0.75)
          gainThreshold = 2.0; // effectively keep warps=4
      }

      if (occ2 > occ4 * gainThreshold)
        optimalWarps = 2;
    }

    mod->setAttr("pact.optimal_num_warps",
                 IntegerAttr::get(IntegerType::get(&getContext(), 32),
                                  optimalWarps));

    llvm::errs() << "[PACT P11] num_warps recommendation: " << optimalWarps
                 << " (pagedLoads=" << numPagedLoads
                 << ", maxTileB=" << maxTileBytes
                 << ", pgoRegs=" << pgoRegsPerThread
                 << ", pgoActiveWarp=" << pgoActiveWarpRatio << ")\n";
  }
};

} // anonymous namespace
} // namespace mlir::triton
