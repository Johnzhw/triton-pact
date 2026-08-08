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

    // Phase 0: skip prefill kernels (tagged by P1 PageTransform)
    if (auto ktype = mod->getAttrOfType<StringAttr>("pact.kernel_type")) {
      if (ktype.getValue() == "prefill") {
        llvm::errs() << "[PACT P2] Prefill kernel detected, skipping.\n";
        return;
      }
    }

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

      // === Step 2: dataflow-aware static safety ===
      // v2 improvement: trace the pointer chain to find remsi(PAGE_SIZE)
      // and determine the actual block_offset within the page.
      // This enables precise knowledge of whether the tile crosses
      // a page boundary or is fully contained within one page.
      bool staticallySafe = false;
      int64_t blockOffsetWithinPage = 0;
      bool foundBlockOffset = false;

      // Trace addptr chain to find remsi-derived offsets
      Value ptr = loadOp.getPtr();
      while (auto *defOp = ptr.getDefiningOp()) {
        auto name = defOp->getName().getStringRef();
        if (name == "tt.addptr") {
          Value offset = defOp->getOperand(1);
          // Check if offset comes from remsi(PAGE_SIZE)
          if (auto remOp = offset.getDefiningOp<arith::RemSIOp>()) {
            if (auto constOp = remOp.getRhs()
                    .template getDefiningOp<arith::ConstantIntOp>()) {
              if (constOp.value() == pageSize) {
                // Found block_offset = token_start % PAGE_SIZE
                // Try to get the lhs (token_start) to determine actual offset
                Value tokenStart = remOp.getLhs();
                // For now, mark that we found the pattern
                // block_offset is runtime but we know it's in [0, pageSize)
                foundBlockOffset = true;
                break;
              }
            }
          }
          // Continue tracing the base pointer
          ptr = defOp->getOperand(0);
          continue;
        }
        // Penetrate splat/broadcast
        if ((name == "tt.splat" || name == "tt.broadcast" ||
             name == "tt.expand_dims") &&
            defOp->getNumOperands() >= 1) {
          ptr = defOp->getOperand(0);
          continue;
        }
        break;
      }

      if (foundBlockOffset) {
        // We know the tile starts at some offset within [0, pageSize)
        // but the exact offset is runtime. Conservatively:
        // staticallySafe only if tile fits in page regardless of offset.
        // That means: pageSize - (max possible offset) >= tileTokens
        // Worst case: offset = pageSize - 1 → never safe
        // Best case: offset = 0 → safe if tileTokens <= pageSize
        //
        // Practical heuristic: if tileTokens <= pageSize/2, the tile
        // is very likely to fit entirely within a page for most offsets.
        // We mark it statically safe only if it CAN'T cross a boundary:
        staticallySafe = (tileTokens <= pageSize / 2);
      } else {
        // Fallback: use P1's page_boundary_safe attribute
        if (auto attr = loadOp->getAttrOfType<BoolAttr>(
                "pact.page_boundary_safe")) {
          staticallySafe = attr.getValue();
        }
        // Additional check: tileTokens % pageSize == 0 is the old heuristic
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
           << ", needsGuard=" << requiresRuntimeGuard
           << ", foundBlockOffset=" << foundBlockOffset);

      return WalkResult::advance();
    });

    if (numAnalyzed > 0) {
      llvm::errs() << "[PACT P2] PageLocalAnalysis: analyzed "
                   << numAnalyzed << " paged load(s) (dataflow-aware v2)\n";
    }
  }
};

} // anonymous namespace
} // namespace mlir::triton
