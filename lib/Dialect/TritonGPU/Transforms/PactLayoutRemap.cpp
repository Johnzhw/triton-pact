//===- PactLayoutRemap.cpp - PACT Layout Remap Pass -----------------------===//
//
//   1. V load: replaces per-element mask with splat-1
//   2. K load: marks with pact.layout_remapped for nBytes=2 pairing
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace mlir {
namespace triton {
namespace gpu {

#define GEN_PASS_DEF_TRITONGPUPACTLAYOUTREMAP
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

namespace {

struct PactLayoutRemapPass
    : public impl::TritonGPUPactLayoutRemapBase<PactLayoutRemapPass> {

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    int numKMarked = 0, numVMaskFixed = 0;

    mod.walk([&](triton::LoadOp loadOp) {
      if (!loadOp->hasAttr("pact.paged_load"))
        return WalkResult::advance();

      auto rTy = dyn_cast<RankedTensorType>(loadOp.getResult().getType());
      if (!rTy || rTy.getShape().size() < 2)
        return WalkResult::advance();

      auto enc = dyn_cast<BlockedEncodingAttr>(rTy.getEncoding());
      if (!enc) return WalkResult::advance();

      auto ctx = &getContext();
      auto b = ImplicitLocOpBuilder(loadOp.getLoc(), ctx);
      b.setInsertionPoint(loadOp);

      // V load mask fix
      Value mask = loadOp.getMask();
      if (mask) {
        bool allOnes = false;
        if (auto c = mask.getDefiningOp<arith::ConstantOp>())
          if (auto da = dyn_cast<DenseIntElementsAttr>(c.getValue()))
            if (da.isSplat() && da.getSplatValue<APInt>().isOne())
              allOnes = true;
        if (!allOnes) {
          auto mTy = cast<RankedTensorType>(mask.getType());
          auto one = DenseIntElementsAttr::get(mTy, APInt(1, 1));
          loadOp.getMaskMutable().assign(
              b.create<arith::ConstantOp>(mTy, one).getResult());
          numVMaskFixed++;
        }
      }

      // K load: mark for nBytes=2 pairing in lowering
      auto sz = enc.getSizePerThread();
      auto order = enc.getOrder();
      if (!order.empty()) {
        unsigned bytes = sz[order[0]] * (rTy.getElementTypeBitWidth() / 8);
        if (bytes < 4) {
          loadOp->setAttr("pact.layout_remapped", UnitAttr::get(ctx));
          numKMarked++;
        }
      }

      return WalkResult::advance();
    });

    if (numKMarked > 0 || numVMaskFixed > 0)
      llvm::errs() << "[PACT PactLayoutRemap] Marked " << numKMarked
                   << " K load(s), fixed " << numVMaskFixed << " V mask(s)\n";
  }
};

} // namespace
} // namespace gpu
} // namespace triton
} // namespace mlir
