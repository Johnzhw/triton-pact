//===- PactLayoutRemap.cpp - PACT Layout Remap Pass -----------------------===//
//
// Fixes paged K/V loads for cp.async compatibility:
//   1. V load: replace per-element mask with splat-1
//   2. K load: mark for contiguity override + nBytes=2 padding
//
// nBytes=2 strategy: emitCpAsync pads to cpSize=4, srcSize=2.
// Elements at odd offsets are 2-byte aligned → handled in lowering
// by pairing: if addr is 4-byte aligned, use cp.async; else emit
// individual ld.global + st.shared.
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

    // First pass: fix V load masks (modify in-place)
    mod.walk([&](triton::LoadOp loadOp) {
      if (!loadOp->hasAttr("pact.paged_load"))
        return WalkResult::advance();
      Value mask = loadOp.getMask();
      if (!mask) return WalkResult::advance();

      bool isAllOnes = false;
      if (auto constOp = mask.getDefiningOp<arith::ConstantOp>())
        if (auto da = dyn_cast<DenseIntElementsAttr>(constOp.getValue()))
          if (da.isSplat() && da.getSplatValue<APInt>().isOne())
            isAllOnes = true;
      if (isAllOnes) return WalkResult::advance();

      ImplicitLocOpBuilder b(loadOp.getLoc(), &getContext());
      b.setInsertionPoint(loadOp);
      auto maskTy = cast<RankedTensorType>(mask.getType());
      auto oneAttr = DenseIntElementsAttr::get(maskTy, APInt(1, 1));
      loadOp.getMaskMutable().assign(
          b.create<arith::ConstantOp>(maskTy, oneAttr).getResult());
      numVMaskFixed++;
      return WalkResult::advance();
    });

    // Second pass: mark K loads needing contiguity override
    mod.walk([&](triton::LoadOp loadOp) {
      if (!loadOp->hasAttr("pact.paged_load"))
        return WalkResult::advance();
      auto rTy = dyn_cast<RankedTensorType>(loadOp.getResult().getType());
      if (!rTy || rTy.getShape().size() < 2) return WalkResult::advance();
      auto enc = dyn_cast<BlockedEncodingAttr>(rTy.getEncoding());
      if (!enc) return WalkResult::advance();

      auto sz = enc.getSizePerThread();
      auto order = enc.getOrder();
      if (order.empty()) return WalkResult::advance();
      unsigned bytes = sz[order[0]] * (rTy.getElementTypeBitWidth() / 8);
      if (bytes < 4) {
        loadOp->setAttr("pact.layout_remapped", UnitAttr::get(&getContext()));
        numKMarked++;
      }
      return WalkResult::advance();
    });

    if (numKMarked > 0 || numVMaskFixed > 0)
      llvm::errs() << "[PACT PactLayoutRemap] Marked " << numKMarked
                   << " K load(s), fixed " << numVMaskFixed << " V mask(s)\n";
  }
};

} // anonymous namespace
} // namespace gpu
} // namespace triton
} // namespace mlir
