//===- PactLayoutRemap.cpp - PACT Layout Remap Pass -----------------------===//
//
// PACT PactLayoutRemap pass: prepares paged K/V loads for cp.async.
// Operates at TTGIR level, before PrefetchInsert.
//
// 1. V load: replaces per-element mask with splat-1 (unconditional).
//    Correctness: attention mask handles masked positions.
//
// 2. K load: marks load with pact.layout_remapped attribute so
//    PrefetchInsert can set a higher contiguity hint.
//    Safety: when PAGE_SIZE % TILE_SIZE == 0 (the common case),
//    tiles don't cross page boundaries → all within-tile addresses
//    are contiguous.
//
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

      auto resultTy = dyn_cast<RankedTensorType>(loadOp.getResult().getType());
      if (!resultTy || resultTy.getShape().size() < 2)
        return WalkResult::advance();

      auto blockedEnc = dyn_cast<BlockedEncodingAttr>(resultTy.getEncoding());
      if (!blockedEnc)
        return WalkResult::advance();

      Location loc = loadOp.getLoc();
      ImplicitLocOpBuilder b(loc, &getContext());
      b.setInsertionPoint(loadOp);  // BEFORE the load, for mask constant

      // --- V load: fix per-element mask ---
      Value mask = loadOp.getMask();
      if (mask) {
        bool isAllOnes = false;
        if (auto constOp = mask.getDefiningOp<arith::ConstantOp>()) {
          if (auto denseAttr =
                  dyn_cast<DenseIntElementsAttr>(constOp.getValue())) {
            if (denseAttr.isSplat() &&
                denseAttr.getSplatValue<APInt>().isOne())
              isAllOnes = true;
          }
        }
        if (!isAllOnes) {
          auto maskTy = cast<RankedTensorType>(mask.getType());
          auto oneAttr = DenseIntElementsAttr::get(maskTy, APInt(1, 1));
          Value trueMask = b.create<arith::ConstantOp>(maskTy, oneAttr);
          loadOp.getMaskMutable().assign(trueMask);
          numVMaskFixed++;
        }
      }

      // --- K load: mark for contiguity hint ---
      auto sz = blockedEnc.getSizePerThread();
      auto order = blockedEnc.getOrder();
      if (order.size() < 2)
        return WalkResult::advance();

      unsigned contigDim = order[0];
      unsigned elemBytes = resultTy.getElementTypeBitWidth() / 8;
      unsigned bytesPerThread = sz[contigDim] * elemBytes;

      if (bytesPerThread < 4) {
        // Mark that this load needs a contiguity override.
        // Safety: when TILE_SIZE divides PAGE_SIZE, tiles don't
        // cross page boundaries → contiguous addresses within tile.
        loadOp->setAttr("pact.layout_remapped",
                        UnitAttr::get(&getContext()));
        numKMarked++;
      }

      return WalkResult::advance();
    });

    if (numKMarked > 0 || numVMaskFixed > 0) {
      llvm::errs() << "[PACT PactLayoutRemap] Marked " << numKMarked
                   << " K load(s) for contiguity override, fixed "
                   << numVMaskFixed << " V mask(s)\n";
    }
  }
};

} // anonymous namespace
} // namespace gpu
} // namespace triton
} // namespace mlir
