//===- AutoNumWarps.cpp - PACT P11: Auto num_warps Selection ------------===//
//
// P11 (TTIR): Analyzes kernel structure (tile sizes, MMA presence)
//             and writes pact.optimal_num_warps module attribute.
//             compiler.py reads this before make_ttgir to set opt.num_warps.
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
    bool hasDot = false;

    // Analyze kernel structure
    mod.walk([&](Operation *op) {
      if (op->hasAttr("pact.paged_load")) {
        numPagedLoads++;
        if (auto attr = op->getAttrOfType<IntegerAttr>("pact.head_dim_size")) {
          int64_t hd = attr.getInt();
          // Estimate tile bytes from head_dim and tensor shape
          auto resultTy = dyn_cast<RankedTensorType>(op->getResultTypes()[0]);
          if (resultTy && resultTy.getShape().size() >= 2) {
            int64_t tile = resultTy.getShape()[0];
            maxTileBytes = std::max(maxTileBytes, tile * hd * 2); // f16=2B
          }
        }
      }
      if (op->getName().getStringRef().contains("dot"))
        hasDot = true;
    });

    // M5c: occupancy-aware warp selection via SMDetector (was a hardcoded
    // `numPagedLoads >= 4 → 2 warps` heuristic).  Use estimateOccupancy with the
    // numWarps parameter (M0c) to compare warps=2 vs warps=4.  Only drop to 2
    // warps when occupancy improves substantially AND there is warp contention
    // (many paged loads), so register/warp pressure is the binding constraint.
    int optimalWarps = 4;
    if (numPagedLoads >= 4 && maxTileBytes > 0) {
      double occ4 =
          pact::SMDetector::estimateOccupancy(/*numStages=*/3, maxTileBytes,
                                              /*regsPerThread=*/64, /*numWarps=*/4);
      double occ2 =
          pact::SMDetector::estimateOccupancy(/*numStages=*/3, maxTileBytes,
                                              /*regsPerThread=*/64, /*numWarps=*/2);
      if (occ2 > occ4 * 1.3) {
        optimalWarps = 2; // occupancy gain >30% → worth reducing warps
      }
    }

    // Write recommendation to module attribute
    mod->setAttr("pact.optimal_num_warps",
                 IntegerAttr::get(IntegerType::get(&getContext(), 32),
                                  optimalWarps));

    llvm::errs() << "[PACT P11] num_warps recommendation: " << optimalWarps
                 << " (pagedLoads=" << numPagedLoads
                 << ", maxTileB=" << maxTileBytes
                 << ", hasDot=" << hasDot << ")\n";
  }
};

} // anonymous namespace
} // namespace mlir::triton
