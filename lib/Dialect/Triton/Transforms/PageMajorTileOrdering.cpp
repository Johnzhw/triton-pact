//===- PageMajorTileOrdering.cpp - PACT Page-Major Tile Reordering --------===//
//
// P7: Transforms token-major scf.for loops into double-nested page-major
//     loops (page loop + tile-in-page loop) to improve L2 cache locality
//     when page_size > tile_size.
//
//     Default OFF — changes FP16 accumulation order of softmax.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/IRMapping.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <string>

#define DEBUG_TYPE "pact-page-major"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::triton {

#define GEN_PASS_DEF_PACTPAGEMAJORTILE
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

namespace {

static bool isEnabled() {
  const char *env = std::getenv("PACT_ENABLE_PAGE_MAJOR_TILE");
  if (!env) return false; // Default OFF
  return std::string(env) == "1";
}

struct PageMajorTileOrderingPass
    : public impl::PACTPageMajorTileBase<PageMajorTileOrderingPass> {

  void runOnOperation() override {
    if (!isEnabled())
      return;

    ModuleOp mod = getOperation();
    int numTransformed = 0;

    mod.walk([&](scf::ForOp forOp) {
      // Get page_size
      int64_t pageSize = 0;
      auto *parent = forOp->getParentOp();
      while (parent) {
        if (auto attr = parent->getAttrOfType<mlir::IntegerAttr>(
                "pact.page_size")) {
          pageSize = attr.getInt();
          break;
        }
        parent = parent->getParentOp();
      }
      if (pageSize <= 0)
        return WalkResult::advance();

      // Get tile_tokens
      int64_t tileTokens = 0;
      forOp.walk([&](triton::LoadOp loadOp) {
        if (loadOp->hasAttr("pact.paged_load") && tileTokens == 0) {
          if (auto attr = loadOp->getAttrOfType<mlir::IntegerAttr>(
                  "pact.tile_tokens")) {
            tileTokens = attr.getInt();
          }
        }
      });
      if (tileTokens <= 0)
        tileTokens = 16;

      // Only beneficial when page_size >= 2 * tile_size
      if (pageSize < tileTokens * 2)
        return WalkResult::advance();

      int64_t tilesPerPage = pageSize / tileTokens;
      if (tilesPerPage < 2)
        return WalkResult::advance();

      // Get upper bound
      auto upperConst =
          forOp.getUpperBound().getDefiningOp<arith::ConstantOp>();
      if (!upperConst)
        return WalkResult::advance();
      int64_t numTiles =
          mlir::cast<mlir::IntegerAttr>(upperConst.getValue()).getInt();

      // Skip if not evenly divisible (simplified implementation)
      if (numTiles % tilesPerPage != 0)
        return WalkResult::advance();

      int64_t numPages = numTiles / tilesPerPage;

      LDBG("PACT P7: Transforming loop: " << numTiles << " tiles, "
           << tilesPerPage << " tiles/page, " << numPages << " pages");

      // Transform: scf.for j=0..N → scf.for p=0..P { scf.for t=0..T }
      OpBuilder builder(forOp);
      auto loc = forOp.getLoc();
      auto ctx = builder.getContext();
      auto i64Ty = mlir::IntegerType::get(ctx, 64);

      Value c0 = arith::ConstantOp::create(
          builder, loc, i64Ty, mlir::IntegerAttr::get(i64Ty, 0));
      Value c1 = arith::ConstantOp::create(
          builder, loc, i64Ty, mlir::IntegerAttr::get(i64Ty, 1));
      Value cP = arith::ConstantOp::create(
          builder, loc, i64Ty, mlir::IntegerAttr::get(i64Ty, numPages));
      Value cT = arith::ConstantOp::create(
          builder, loc, i64Ty, mlir::IntegerAttr::get(i64Ty, tilesPerPage));
      Value cTMul = arith::ConstantOp::create(
          builder, loc, i64Ty, mlir::IntegerAttr::get(i64Ty, tilesPerPage));

      auto pageLoop = scf::ForOp::create(
          builder, loc, c0, cP, c1, forOp.getInitArgs());

      builder.setInsertionPointToStart(pageLoop.getBody());
      Value p = pageLoop.getInductionVar();
      Value pBase = arith::MulIOp::create(builder, loc, p, cTMul);

      auto tileLoop = scf::ForOp::create(
          builder, loc, c0, cT, c1, pageLoop.getRegionIterArgs());

      builder.setInsertionPointToStart(tileLoop.getBody());
      Value t = tileLoop.getInductionVar();
      Value j = arith::AddIOp::create(builder, loc, pBase, t);

      // Clone original loop body, replacing IV with j
      mlir::IRMapping mapping;
      mapping.map(forOp.getInductionVar(), j);
      for (auto &op : forOp.getBody()->without_terminator()) {
        builder.clone(op, mapping);
      }

      auto oldYield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
      SmallVector<mlir::Value> tileYieldOperands;
      for (auto operand : oldYield.getOperands()) {
        tileYieldOperands.push_back(mapping.lookup(operand));
      }
      scf::YieldOp::create(builder, loc, tileYieldOperands);

      builder.setInsertionPointAfter(tileLoop);
      scf::YieldOp::create(builder, loc, tileLoop.getResults());

      forOp.replaceAllUsesWith(pageLoop.getResults());
      forOp.erase();

      numTransformed++;
      return WalkResult::advance();
    });

    if (numTransformed > 0) {
      llvm::errs() << "[PACT P7] Page-major reordering: transformed "
                   << numTransformed << " loop(s)\n";
    }
  }
};

} // anonymous namespace
} // namespace mlir::triton
