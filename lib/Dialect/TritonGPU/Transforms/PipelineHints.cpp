//===- PipelineHints.cpp - PACT Pipeline Hints ----------------------------===//
//
// P4: Computes 13 pipeline hint values from PACT semantic attributes
//     and attaches them to tt.load ops for consumption by Coalesce,
//     Pipeline, and LLVM lowering passes.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <string>
#include <algorithm>

#define DEBUG_TYPE "pact-hints"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::triton::gpu {

#define GEN_PASS_DEF_PACTPIPELINEHINTS
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

namespace {

static bool isEnabled() {
  const char *env = std::getenv("PACT_ENABLE_PIPELINE_HINTS");
  return !env || std::string(env) != "0";
}

// Estimate loop trip count from scf.for bounds
static int64_t estimateTripCount(scf::ForOp forOp) {
  auto upperConst =
      forOp.getUpperBound().getDefiningOp<arith::ConstantOp>();
  auto lowerConst =
      forOp.getLowerBound().getDefiningOp<arith::ConstantOp>();

  if (upperConst && lowerConst) {
    int64_t upperVal = mlir::cast<mlir::IntegerAttr>(upperConst.getValue()).getInt();
    int64_t lowerVal = mlir::cast<mlir::IntegerAttr>(lowerConst.getValue()).getInt();
    int64_t step = 1;
    if (auto stepOp = forOp.getStep().getDefiningOp<arith::ConstantOp>()) {
      step = mlir::cast<mlir::IntegerAttr>(stepOp.getValue()).getInt();
    }
    if (step > 0)
      return (upperVal - lowerVal) / step;
  }
  return 128; // conservative default
}

// Get total elements in a ranked tensor
static int64_t getTotalElements(RankedTensorType ty) {
  int64_t total = 1;
  for (auto dim : ty.getShape())
    total *= dim;
  return total;
}

struct PACTPipelineHintsPass
    : public impl::PACTPipelineHintsBase<PACTPipelineHintsPass> {

  void runOnOperation() override {
    if (!isEnabled())
      return;

    ModuleOp mod = getOperation();
    int numHintsAttached = 0;

    mod.walk([&](scf::ForOp forOp) {
      // Collect paged loads within this loop
      SmallVector<triton::LoadOp> pagedLoads;
      forOp.walk([&](triton::LoadOp loadOp) {
        if (loadOp->hasAttr("pact.paged_load"))
          pagedLoads.push_back(loadOp);
      });

      if (pagedLoads.empty())
        return WalkResult::advance();

      // Get page_size
      int64_t pageSize = 16;
      if (auto attr = forOp->getParentOp()->getAttrOfType<mlir::IntegerAttr>(
              "pact.page_size")) {
        pageSize = attr.getInt();
      }
      if (pageSize <= 0 && !pagedLoads.empty()) {
        if (auto attr = pagedLoads[0]->getAttrOfType<mlir::IntegerAttr>(
                "pact.page_size")) {
          pageSize = attr.getInt();
        }
      }

      int64_t estIterations = estimateTripCount(forOp);

      for (auto loadOp : pagedLoads) {
        auto resultTy = cast<RankedTensorType>(loadOp.getResult().getType());
        int64_t totalElements = getTotalElements(resultTy);
        int64_t totalBytes = totalElements * 2; // f16

        int64_t tileTokens = 16;
        if (auto attr = loadOp->getAttrOfType<mlir::IntegerAttr>(
                "pact.tile_tokens"))
          tileTokens = attr.getInt();

        bool boundarySafe = false;
        if (auto attr = loadOp->getAttrOfType<mlir::BoolAttr>(
                "pact.page_boundary_safe"))
          boundarySafe = attr.getValue();

        int64_t headSize = 64;
        if (auto attr = loadOp->getAttrOfType<mlir::IntegerAttr>(
                "pact.head_dim_size"))
          headSize = attr.getInt();

        int64_t pageLocalContiguity = 1;
        if (auto attr = loadOp->getAttrOfType<mlir::IntegerAttr>(
                "pact.pagelocal.contiguity"))
          pageLocalContiguity = attr.getInt();

        int64_t safeVecWidth = 1;
        if (auto attr = loadOp->getAttrOfType<mlir::IntegerAttr>(
                "pact.pagelocal.safe_vector_width"))
          safeVecWidth = attr.getInt();

        bool requiresGuard = false;
        if (auto attr = loadOp->getAttrOfType<mlir::BoolAttr>(
                "pact.pagelocal.requires_guard"))
          requiresGuard = attr.getValue();

        // === Compute 13 hint values ===

        // Hint 1: pageSize
        int64_t hint_pageSize = pageSize;

        // Hint 2: headDimAlignment
        int64_t hint_headDimAlignment = std::min(headSize, int64_t(8));

        // Hint 3: safeAsyncCopyWidth (bytes)
        int64_t hint_safeAsyncCopyWidth = 4;
        if (boundarySafe && pageLocalContiguity >= 8)
          hint_safeAsyncCopyWidth = 16; // 8×f16 = 128-bit
        else if (pageLocalContiguity >= 4)
          hint_safeAsyncCopyWidth = 8;  // 4×f16 = 64-bit
        else if (pageLocalContiguity >= 2)
          hint_safeAsyncCopyWidth = 4;  // 2×f16 = 32-bit

        // Hint 4: pageLocalContiguity
        int64_t hint_pageLocalContiguity = pageLocalContiguity;

        // Hint 5: isPageCrossing
        bool hint_isPageCrossing = !boundarySafe;

        // Hint 6: fullPageTile
        bool hint_fullPageTile = (tileTokens == pageSize);

        // Hint 7: estimatedIterations
        int64_t hint_estimatedIterations = estIterations;

        // Hint 8: tileBytes
        int64_t hint_tileBytes = totalBytes;

        // Hint 9: preferAsync
        bool hint_preferAsync =
            (totalBytes >= 128) && (estIterations >= 16) &&
            (estIterations >= 32 || totalBytes >= 256);

        // Hint 10: suggestedNumStages (page-aware heuristic)
        int tilesPerPage = (pageSize > 0) ? pageSize / tileTokens : 1;
        int hint_suggestedNumStages;
        if (estIterations < 16) {
          hint_suggestedNumStages = 2;
        } else if (tilesPerPage >= 4) {
          hint_suggestedNumStages = 2; // high L2 locality
        } else if (tilesPerPage <= 1) {
          hint_suggestedNumStages = 3; // low locality
        } else {
          hint_suggestedNumStages = (totalBytes > 256) ? 3 : 2;
        }

        // Hint 11: suggestedPrefetchDistance
        int hint_suggestedPrefetchDistance =
            (tilesPerPage >= 4) ? 1 : 2;

        // Hint 12: vectorWidth (for getVectorSize)
        int64_t hint_vectorWidth = std::min(safeVecWidth, int64_t(8));

        // Hint 13: divisibilityBoost (bytes)
        int hint_divisibilityBoost = boundarySafe ? 16 : 4;

        // === Attach hints to loadOp ===
        auto ctx = &getContext();
        auto i64Ty = IntegerType::get(ctx, 64);
        loadOp->setAttr("pact.hint.page_size",
            mlir::IntegerAttr::get(i64Ty, hint_pageSize));
        loadOp->setAttr("pact.hint.head_dim_alignment",
            mlir::IntegerAttr::get(i64Ty, hint_headDimAlignment));
        loadOp->setAttr("pact.hint.safe_async_copy_width",
            mlir::IntegerAttr::get(i64Ty, hint_safeAsyncCopyWidth));
        loadOp->setAttr("pact.hint.page_local_contiguity",
            mlir::IntegerAttr::get(i64Ty, hint_pageLocalContiguity));
        loadOp->setAttr("pact.hint.is_page_crossing",
            mlir::BoolAttr::get(ctx, hint_isPageCrossing));
        loadOp->setAttr("pact.hint.full_page_tile",
            mlir::BoolAttr::get(ctx, hint_fullPageTile));
        loadOp->setAttr("pact.hint.estimated_iterations",
            mlir::IntegerAttr::get(i64Ty, hint_estimatedIterations));
        loadOp->setAttr("pact.hint.tile_bytes",
            mlir::IntegerAttr::get(i64Ty, hint_tileBytes));
        loadOp->setAttr("pact.hint.tile_tokens",
            mlir::IntegerAttr::get(i64Ty, tileTokens));
        loadOp->setAttr("pact.hint.prefer_async",
            mlir::BoolAttr::get(ctx, hint_preferAsync));
        loadOp->setAttr("pact.hint.suggested_num_stages",
            mlir::IntegerAttr::get(i64Ty, hint_suggestedNumStages));
        loadOp->setAttr("pact.hint.suggested_prefetch_distance",
            mlir::IntegerAttr::get(i64Ty, hint_suggestedPrefetchDistance));
        loadOp->setAttr("pact.hint.vector_width",
            mlir::IntegerAttr::get(i64Ty, hint_vectorWidth));
        loadOp->setAttr("pact.hint.divisibility_boost",
            mlir::IntegerAttr::get(i64Ty, hint_divisibilityBoost));

        numHintsAttached++;
        LDBG("PACT P4: hints for load: bytes=" << totalBytes
             << ", safeCpWidth=" << hint_safeAsyncCopyWidth
             << ", preferAsync=" << hint_preferAsync
             << ", numStages=" << hint_suggestedNumStages);
      }

      return WalkResult::advance();
    });

    if (numHintsAttached > 0) {
      llvm::errs() << "[PACT P4] Pipeline Hints: attached to "
                   << numHintsAttached << " paged load(s)\n";
    }
  }
};

} // anonymous namespace
} // namespace mlir::triton::gpu
