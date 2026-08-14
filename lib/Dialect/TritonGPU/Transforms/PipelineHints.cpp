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
  // Bug5 fix: runtime-bound loops (num_tiles = cdiv(seq_len, TILE_SIZE)) cannot
  // be statically resolved.  The old hardcoded 128 always classified these as
  // "long sequence", defeating P6's short/medium/long heuristic.  Return a
  // "medium" default so P6 keeps the default num_stages instead of
  // over-committing to a pipeline that a short decode sequence cannot amortize.
  return 64; // medium-sequence default (was 128 = always "long")
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

      // Bug2 fix: P1 writes pact.page_size on the load op (and module), NOT on
      // the scf.for's parent (the function op).  Read it from the first paged
      // load first (always correct), then fall back to the module.
      int64_t pageSize = 16;
      if (!pagedLoads.empty()) {
        if (auto attr = pagedLoads[0]->getAttrOfType<mlir::IntegerAttr>(
                "pact.page_size")) {
          pageSize = attr.getInt();
        }
      }
      if (pageSize <= 0) {
        if (auto attr = forOp->getParentOp()->getAttrOfType<mlir::IntegerAttr>(
                "pact.page_size")) {
          pageSize = attr.getInt();
        }
      }

      int64_t estIterations = estimateTripCount(forOp);

      for (auto loadOp : pagedLoads) {
        auto resultTy = cast<RankedTensorType>(loadOp.getResult().getType());
        int64_t totalElements = getTotalElements(resultTy);
        // M9c: derive element width from the tensor type instead of hardcoding
        // the f16 width (2 bytes) — otherwise f32 (4B) paged loads would
        // under-estimate SMEM by 2×.
        int64_t totalBytes =
            totalElements * (resultTy.getElementTypeBitWidth() / 8);

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

        // === Compute 8 hint values (simplified from 14 — each has a consumer) ===
        //
        // Group A: Page semantics (4 hints — P5, P6 consumers)
        // Group B: Pipeline enhancement (4 hints — Pipeline pass consumers)

        // Hint 1: pageSize (P5, P6 consumer)
        int64_t hint_pageSize = pageSize;

        // Hint 2: tileTokens (P5, P6 consumer)
        int64_t hint_tileTokens = tileTokens;

        // Hint 3: tileBytes (P5, P6, P10 consumer)
        int64_t hint_tileBytes = totalBytes;

        // Hint 4: estimatedIterations (P5, P6 consumer)
        int64_t hint_estimatedIterations = estIterations;

        // Hint 5: pageLocalContiguity (P3, P9 consumer)
        int64_t hint_pageLocalContiguity = pageLocalContiguity;

        // Hint 6: isPageCrossing (P3, P7 consumer)
        bool hint_isPageCrossing = !boundarySafe;

        // Hint 7: preferAsync (Pipeline pass — LowerLoops consumer)
        // Paged attention heuristic: prefer async copy for large tiles with
        // sufficient iterations and multi-tile pages (page locality).
        int tilesPerPage = (pageSize > 0 && tileTokens > 0)
                              ? pageSize / tileTokens : 1;
        bool hint_preferAsync =
            (totalBytes >= 128) && (estIterations >= 16) &&
            (tilesPerPage >= 2 || totalBytes >= 512);

        // Hint 8: suggestedNumStages (Pipeline pass — ScheduleLoops consumer)
        // Simple page-aware heuristic; P6 AutoNumStages is the primary decision maker.
        int hint_suggestedNumStages;
        if (tilesPerPage >= 4) {
          hint_suggestedNumStages = 2; // high L2 locality → fewer stages
        } else if (tilesPerPage <= 1) {
          hint_suggestedNumStages = 4; // low locality → more stages
        } else {
          hint_suggestedNumStages = 3; // medium
        }
        if (estIterations < 16)
          hint_suggestedNumStages = std::min(hint_suggestedNumStages, 2);

        // === Removed 6 redundant hints (v2 simplification) ===
        // - safe_async_copy_width: via AxisInfo contiguity
        // - head_dim_alignment: via AxisInfo divisibility
        // - full_page_tile: derivable from page_size/tile_tokens
        // - suggested_prefetch_distance: no consumer
        // - vector_width: via AxisInfo contiguity
        // - divisibility_boost: via AxisInfo divisibility

        // === Attach 8 hints to loadOp ===
        auto ctx = &getContext();
        auto i64Ty = IntegerType::get(ctx, 64);
        loadOp->setAttr("pact.hint.page_size",
            mlir::IntegerAttr::get(i64Ty, hint_pageSize));
        loadOp->setAttr("pact.hint.tile_tokens",
            mlir::IntegerAttr::get(i64Ty, hint_tileTokens));
        loadOp->setAttr("pact.hint.tile_bytes",
            mlir::IntegerAttr::get(i64Ty, hint_tileBytes));
        loadOp->setAttr("pact.hint.estimated_iterations",
            mlir::IntegerAttr::get(i64Ty, hint_estimatedIterations));
        loadOp->setAttr("pact.hint.page_local_contiguity",
            mlir::IntegerAttr::get(i64Ty, hint_pageLocalContiguity));
        loadOp->setAttr("pact.hint.is_page_crossing",
            mlir::BoolAttr::get(ctx, hint_isPageCrossing));
        loadOp->setAttr("pact.hint.prefer_async",
            mlir::BoolAttr::get(ctx, hint_preferAsync));
        loadOp->setAttr("pact.hint.suggested_num_stages",
            mlir::IntegerAttr::get(i64Ty, hint_suggestedNumStages));

        numHintsAttached++;
        LDBG("PACT P4: hints for load: bytes=" << totalBytes
             << ", preferAsync=" << hint_preferAsync
             << ", numStages=" << hint_suggestedNumStages
             << ", tilesPerPage=" << tilesPerPage);
      }

      return WalkResult::advance();
    });

    if (numHintsAttached > 0) {
      llvm::errs() << "[PACT P4] Pipeline Hints: attached to "
                   << numHintsAttached << " paged load(s)\n";
    }

    // P10: Module-level resource hints — use data already collected above.
    const char *rhEnv = std::getenv("PACT_ENABLE_RESOURCE_HINTS");
    if (rhEnv && std::string(rhEnv) == "1" && numHintsAttached > 0) {
      int maxStages = 0, estIters = 0, totalRegs = 0;
      int64_t totalBytes = 0;
      bool hasAsync = false;

      // Re-walk only the loads we already found (cheap, module already walked)
      mod.walk([&](triton::LoadOp loadOp) {
        if (!loadOp->hasAttr("pact.hint.tile_bytes"))
          return WalkResult::advance();
        totalBytes += loadOp->getAttrOfType<mlir::IntegerAttr>(
            "pact.hint.tile_bytes").getInt();
        if (auto a = loadOp->getAttrOfType<mlir::IntegerAttr>(
                "pact.hint.suggested_num_stages"))
          maxStages = std::max(maxStages, (int)a.getInt());
        if (auto a = loadOp->getAttrOfType<mlir::IntegerAttr>(
                "pact.hint.estimated_iterations"))
          estIters = std::max(estIters, (int)a.getInt());
        if (auto a = loadOp->getAttrOfType<mlir::IntegerAttr>(
                "pact.hint.safe_async_copy_width"))
          if (a.getInt() >= 4) hasAsync = true;
        return WalkResult::advance();
      });

      int64_t smem = totalBytes * std::max(2, maxStages);
      int numWarps = 4;
      if (auto a = mod->getAttrOfType<mlir::IntegerAttr>("ttg.num-warps"))
        numWarps = a.getInt();
      int rpt = std::max(1, totalRegs / std::max(1, numWarps * 32));
      int mwr = std::max(1, std::min(65536/(rpt*32), 48));
      double occ = (double)mwr / 48.0;
      bool memB = ((double)totalBytes/numHintsAttached >= 256) || (estIters < 64);

      std::string json = "{\"cp_async\":" + std::string(hasAsync?"true":"false") +
        ",\"num_paged_loads\":" + std::to_string(numHintsAttached) +
        ",\"num_stages\":" + std::to_string(maxStages) +
        ",\"est_smem_bytes\":" + std::to_string(smem) +
        ",\"est_regs_per_thread\":" + std::to_string(rpt) +
        ",\"est_occupancy\":" + std::to_string(occ).substr(0,4) +
        ",\"est_iterations\":" + std::to_string(estIters) +
        ",\"est_tile_bytes\":" + std::to_string(totalBytes) +
        ",\"memory_bound\":" + std::string(memB?"true":"false") +
        ",\"fill_drain_cost_iter\":" + std::to_string(maxStages*2) + "}";
      mod->setAttr("pact.resource_hints", mlir::StringAttr::get(&getContext(), json));
      llvm::errs() << "[PACT P4+P10] Resource Hints: " << json << "\n";
    }
  }
};

} // anonymous namespace
} // namespace mlir::triton::gpu
