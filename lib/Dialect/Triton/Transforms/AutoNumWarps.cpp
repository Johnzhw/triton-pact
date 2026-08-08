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

    // Conservative heuristic: keep 4 warps unless strong reason
    int optimalWarps = 4;
    if (numPagedLoads >= 4) {
      // Many paged loads → reduce warp contention
      optimalWarps = 2;
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
