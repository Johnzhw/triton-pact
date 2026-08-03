//===- PageLocalAnalysis.cpp - PACT Page-Local Memory Analysis ------------===//
//
// P2: Derives page-local memory properties from PACT semantic attributes.
//     Does NOT modify AxisInfo — only outputs verifiable analysis results
//     as IR attributes for downstream consumers (P3, P4, P8).
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <string>
#include <algorithm>

#define DEBUG_TYPE "pact-page-local"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::triton {

#define GEN_PASS_DEF_PACTPAGELOCALANALYSIS
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

namespace {

// Check if PageLocalAnalysis is enabled via env var.
// Defaults to ON. Set PACT_ENABLE_PAGE_LOCAL_ANALYSIS=0 to disable.
static bool isEnabled() {
  const char *env = std::getenv("PACT_ENABLE_PAGE_LOCAL_ANALYSIS");
  return !env || std::string(env) != "0";
}

struct PageLocalAnalysisPass
    : public impl::PACTPageLocalAnalysisBase<PageLocalAnalysisPass> {

  void runOnOperation() override {
    if (!isEnabled())
      return;

    ModuleOp mod = getOperation();
    int numAnalyzed = 0;

    mod.walk([&](triton::LoadOp loadOp) {
      // Only analyze loads annotated by PageTransform
      if (!loadOp->hasAttr("pact.paged_load"))
        return WalkResult::advance();

      // === Step 1: read P1 semantic attributes ===
      int64_t pageSize = 16; // default
      if (auto attr = loadOp->getAttrOfType<IntegerAttr>("pact.page_size"))
        pageSize = attr.getInt();

      int64_t tileTokens = 1;
      if (auto attr = loadOp->getAttrOfType<IntegerAttr>("pact.tile_tokens"))
        tileTokens = attr.getInt();

      int64_t headDimIdx = 1;
      if (auto attr = loadOp->getAttrOfType<IntegerAttr>("pact.head_dim_idx"))
        headDimIdx = attr.getInt();

      int64_t headSize = 64;
      if (auto attr = loadOp->getAttrOfType<IntegerAttr>("pact.head_dim_size"))
        headSize = attr.getInt();

      // === Step 2: determine static safety ===
      bool staticallySafe = false;
      if (auto attr = loadOp->getAttrOfType<BoolAttr>(
              "pact.page_boundary_safe")) {
        staticallySafe = attr.getValue();
      }

      // === Step 3: compute pageLocalContiguity ===
      auto resultTy = cast<RankedTensorType>(loadOp.getResult().getType());
      int64_t headDimElements = 1;
      if (headDimIdx < (int64_t)resultTy.getShape().size())
        headDimElements = resultTy.getShape()[headDimIdx];

      int64_t pageLocalContiguity = 1;

      if (staticallySafe) {
        // Static proof: tile does not cross page boundary
        // All head_dim elements within a tile are physically contiguous
        pageLocalContiguity = headSize;
      } else {
        // Conservative: head_dim elements are contiguous within a page,
        // but cannot guarantee full tile coverage
        pageLocalContiguity = std::min(headSize, headDimElements);
        // Further constrain: if pageSize < tileTokens, page boundary may
        // cut within the tile
        if (pageSize > 0 && pageSize < tileTokens) {
          pageLocalContiguity = std::min(pageLocalContiguity, pageSize);
        }
      }

      // === Step 4: compute max safe vector width ===
      // NVIDIA: max 128-bit per load = 8 × f16 elements
      int elementBitWidth = 16; // f16
      int maxVecElements = 128 / elementBitWidth; // = 8
      int64_t maxSafeVectorWidth = std::min((int64_t)maxVecElements,
                                             pageLocalContiguity);

      // === Step 5: determine if runtime guard is needed ===
      bool requiresRuntimeGuard = !staticallySafe && pageLocalContiguity > 1;

      // === Step 6: estimate bytes until page boundary ===
      int64_t bytesUntilPageBoundary = 0;
      if (staticallySafe) {
        bytesUntilPageBoundary = pageSize * elementBitWidth / 8;
      } else {
        // Dynamic: conservative estimate
        bytesUntilPageBoundary = pageSize * elementBitWidth / 8;
      }

      // === Step 7: output analysis as IR attributes ===
      auto ctx = &getContext();
      auto i64Ty = IntegerType::get(ctx, 64);
      loadOp->setAttr("pact.pagelocal.contiguity",
          IntegerAttr::get(i64Ty, pageLocalContiguity));
      loadOp->setAttr("pact.pagelocal.safe_vector_width",
          IntegerAttr::get(i64Ty, maxSafeVectorWidth));
      loadOp->setAttr("pact.pagelocal.statically_safe",
          BoolAttr::get(ctx, staticallySafe));
      loadOp->setAttr("pact.pagelocal.requires_guard",
          BoolAttr::get(ctx, requiresRuntimeGuard));
      loadOp->setAttr("pact.pagelocal.bytes_until_boundary",
          IntegerAttr::get(i64Ty, bytesUntilPageBoundary));

      numAnalyzed++;
      LDBG("PACT P2: load analyzed: contiguity=" << pageLocalContiguity
           << ", vecWidth=" << maxSafeVectorWidth
           << ", staticSafe=" << staticallySafe
           << ", needsGuard=" << requiresRuntimeGuard);

      return WalkResult::advance();
    });

    if (numAnalyzed > 0) {
      llvm::errs() << "[PACT P2] PageLocalAnalysis: analyzed "
                   << numAnalyzed << " paged load(s)\n";
    }
  }
};

} // anonymous namespace
} // namespace mlir::triton
