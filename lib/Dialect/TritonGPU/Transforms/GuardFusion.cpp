//===- GuardFusion.cpp - PACT Mask/Page-Boundary Guard Fusion -------------===//
//
// P8: Fuses seq boundary mask and page boundary mask into a single
//     comparison (min(seq_remaining, page_remaining) < tile_size).
//     Only activates when page_boundary_safe=false.
//
//     Default OFF.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <string>

#define DEBUG_TYPE "pact-guard-fusion"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::triton::gpu {

#define GEN_PASS_DEF_PACTGUARDFUSION
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

namespace {

static bool isEnabled() {
  const char *env = std::getenv("PACT_ENABLE_GUARD_FUSION");
  if (!env) return false; // Default OFF
  return std::string(env) == "1";
}

// Match: AND of two CMP ops → fuse into min-based single CMP
struct FuseGuardPattern : public OpRewritePattern<triton::LoadOp> {
  using OpRewritePattern<triton::LoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::LoadOp loadOp,
                                 PatternRewriter &rewriter) const override {
    if (!loadOp->hasAttr("pact.paged_load"))
      return failure();

    // Skip if statically safe (no page boundary mask needed)
    if (loadOp->hasAttr("pact.page_boundary_safe"))
      return failure();

    Value mask = loadOp.getMask();
    if (!mask)
      return failure();

    // Check if mask is AND of two compares
    auto andOp = mask.getDefiningOp<arith::AndIOp>();
    if (!andOp)
      return failure();

    Value lhs = andOp.getLhs();
    Value rhs = andOp.getRhs();

    auto cmp1 = lhs.getDefiningOp<arith::CmpIOp>();
    auto cmp2 = rhs.getDefiningOp<arith::CmpIOp>();
    if (!cmp1 || !cmp2)
      return failure();

    // Identify which is seq mask and which is page mask
    // Heuristic: page mask's RHS is a constant (page_size)
    arith::CmpIOp seqCmp = cmp1, pageCmp = cmp2;
    auto pageConst = pageCmp.getRhs().getDefiningOp<arith::ConstantOp>();
    auto seqConst = seqCmp.getRhs().getDefiningOp<arith::ConstantOp>();

    if (!pageConst && seqConst) {
      std::swap(seqCmp, pageCmp);
    }

    pageConst = pageCmp.getRhs().getDefiningOp<arith::ConstantOp>();
    if (!pageConst)
      return failure();

    // Build fused mask: min(seq_remaining, page_remaining) < tile_size
    auto loc = loadOp.getLoc();
    Value seqLen = seqCmp.getRhs();
    Value tokenIdx = seqCmp.getLhs();
    Value seqRemaining = arith::SubIOp::create(rewriter, loc, seqLen, tokenIdx);

    Value pageSizeV = pageCmp.getRhs();
    Value pageOffset = pageCmp.getLhs();
    Value pageRemaining = arith::SubIOp::create(rewriter, 
        loc, pageSizeV, pageOffset);

    Value bound = arith::MinSIOp::create(rewriter, 
        loc, seqRemaining, pageRemaining);

    int64_t tileSize = 16;
    if (auto attr = loadOp->getAttrOfType<mlir::IntegerAttr>(
            "pact.tile_tokens"))
      tileSize = attr.getInt();
    Value tileSizeV = arith::ConstantOp::create(rewriter, loc,
        rewriter.getI64IntegerAttr(tileSize));
    Value effectiveMask = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::slt, tileSizeV, bound);

    auto newLoad = triton::LoadOp::create(
        rewriter, loc, loadOp.getResult().getType(),
        loadOp.getPtr(), effectiveMask, loadOp.getOther(),
        loadOp.getCache(), loadOp.getEvict(), loadOp.getIsVolatile());

    for (auto namedAttr : loadOp->getAttrs()) {
      newLoad->setAttr(namedAttr.getName(), namedAttr.getValue());
    }

    rewriter.replaceOp(loadOp, newLoad.getResult());
    LDBG("PACT P8: fused guard for load");
    return success();
  }
};

struct PACTGuardFusionPass
    : public impl::PACTGuardFusionBase<PACTGuardFusionPass> {

  void runOnOperation() override {
    if (!isEnabled())
      return;

    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<FuseGuardPattern>(ctx);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // anonymous namespace
} // namespace mlir::triton::gpu
