//===- PageLocalAnalysis.cpp - PACT Page-Local Memory Analysis ------------===//
//
// P2: Derives page-local memory properties from PACT semantic attributes.
//     Does NOT modify AxisInfo — only outputs verifiable analysis results
//     as IR attributes for downstream consumers (P3).
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
#include "triton/Tools/LinearLayout.h"

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

static bool isPowerOf2(int64_t n) { return n > 0 && (n & (n - 1)) == 0; }

// M9: construct the *page-internal memory layout* as an F₂-linear layout.
//
// A paged KV cache page stores [pageSize tokens × headDim elements] contiguously
// in memory:  offset(token_in_page, head_idx) = token_in_page * headDim + head_idx.
// This is a *memory* layout (address computation), NOT a register layout — it
// does not depend on the TTGIR register encoding, so it is constructible at the
// TTIR layer.  In F₂ terms:
//   - head_idx basis i  → offset = i          (stride 1: contiguous)
//   - token_in_page basis j → offset = j*headDim (stride headDim: not contiguous)
// getNumConsecutiveInOut() then returns the exact page-bounded memory contiguity
// along head_idx (= headDim), independent of whether the tile crosses a page
// boundary (which only concerns the token dimension).
//
// This replaces PACT's semantic heuristic for pageLocalContiguity with an exact
// F₂-linear computation.  headDim/pageSize must be powers of two (LinearLayout
// requirement); callers guard with isPowerOf2 and fall back otherwise.
static LinearLayout buildPagedMemoryLayout(int64_t headDim, int64_t pageSize,
                                           MLIRContext *ctx) {
  auto headIdx = StringAttr::get(ctx, "head_idx");
  auto tokenInPage = StringAttr::get(ctx, "token_in_page");
  auto offset = StringAttr::get(ctx, "offset");

  std::vector<std::vector<int32_t>> headBases;
  for (int32_t i = 1; i < headDim; i *= 2)
    headBases.push_back({i}); // stride 1

  std::vector<std::vector<int32_t>> tokenBases;
  for (int32_t j = 1; j < pageSize; j *= 2)
    tokenBases.push_back({(int32_t)(j * headDim)}); // stride headDim

  std::vector<std::pair<StringAttr, std::vector<std::vector<int32_t>>>> bases = {
      {headIdx, headBases}, {tokenInPage, tokenBases}};
  return LinearLayout(bases, {offset});
}

// Bug1 fix: deep-trace an offset value back to remsi(PAGE_SIZE), penetrating
// the intermediate arithmetic/type-cast/broadcast ops that sit between the
// addptr offset and the remsi in real paged-attention IR.
//   block_offset = remsi(token_start, PAGE_SIZE)
//                 → extsi → addi/muli → splat/broadcast/expand_dims → addptr
// The old code only inspected the *direct* offset operand of addptr, so it
// never found the remsi and always fell back to P1's page_boundary_safe.
static bool traceToRemSIOp(Value val, int64_t pageSize, int maxDepth,
                           Value divisorArg = Value()) {
  if (maxDepth <= 0)
    return false;

  auto *defOp = val.getDefiningOp();
  if (!defOp)
    return false;

  // Direct hit: remsi(_, PAGE_SIZE)
  if (auto remOp = dyn_cast<arith::RemSIOp>(defOp)) {
    if (auto constOp =
            remOp.getRhs().template getDefiningOp<arith::ConstantIntOp>()) {
      if (constOp.value() == pageSize)
        return true;
    }
    // V20 W6'a family A: remsi by the RUNTIME divisor argument (its
    // identity was recorded by P1 as pact.page_divisor_arg)
    if (divisorArg) {
      Value rhs = remOp.getRhs();
      for (int i = 0; i < 6 && rhs.getDefiningOp(); i++) {
        auto *rdef = rhs.getDefiningOp();
        auto rname = rdef->getName().getStringRef();
        if (rname == "arith.extsi" || rname == "arith.extui" ||
            rname == "arith.index_cast" || rname == "tt.splat")
          rhs = rdef->getOperand(0);
        else
          break;
      }
      if (rhs == divisorArg)
        return true;
    }
  }

  auto name = defOp->getName().getStringRef();

  // Penetrate integer casts: extsi/extui/trunci → operand 0
  if (name == "arith.extsi" || name == "arith.extui" ||
      name == "arith.trunci") {
    if (defOp->getNumOperands() >= 1)
      return traceToRemSIOp(defOp->getOperand(0), pageSize, maxDepth - 1);
    return false;
  }

  // Penetrate addi/muli on the non-constant side
  if ((name == "arith.addi" || name == "arith.muli") &&
      defOp->getNumOperands() == 2) {
    bool lhsConst =
        defOp->getOperand(0).template getDefiningOp<arith::ConstantOp>() !=
        nullptr;
    bool rhsConst =
        defOp->getOperand(1).template getDefiningOp<arith::ConstantOp>() !=
        nullptr;
    if (lhsConst && !rhsConst)
      return traceToRemSIOp(defOp->getOperand(1), pageSize, maxDepth - 1);
    if (!lhsConst && rhsConst)
      return traceToRemSIOp(defOp->getOperand(0), pageSize, maxDepth - 1);
    return false;
  }

  // Penetrate splat/broadcast/expand_dims → operand 0
  if (name == "tt.splat" || name == "tt.broadcast" ||
      name == "tt.expand_dims") {
    if (defOp->getNumOperands() >= 1)
      return traceToRemSIOp(defOp->getOperand(0), pageSize, maxDepth - 1);
    return false;
  }

  return false;
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

      int64_t headSize = 64;
      if (auto attr = loadOp->getAttrOfType<IntegerAttr>("pact.head_dim_size"))
        headSize = attr.getInt();

      // V20 W6'a family A: runtime divisor identity (P1 recorded the
      // argument index when the page size is a runtime scalar)
      Value divisorArg;
      bool runtimePage = false;
      if (auto attr = loadOp->getAttrOfType<IntegerAttr>(
              "pact.page_divisor_arg")) {
        auto func = loadOp->getParentOfType<triton::FuncOp>();
        if (func && attr.getInt() >= 0 &&
            attr.getInt() < func.getNumArguments()) {
          divisorArg = func.getArgument(attr.getInt());
          runtimePage = true;
        }
      }

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
          // Bug1 fix: check the direct remsi(PAGE_SIZE) operand first, then
          // deep-trace through extsi/addi/muli/splat/broadcast/expand_dims.
          bool directRem = false;
          if (auto remOp = offset.getDefiningOp<arith::RemSIOp>()) {
            if (auto constOp = remOp.getRhs()
                    .template getDefiningOp<arith::ConstantIntOp>()) {
              if (constOp.value() == pageSize) {
                directRem = true;
              }
            }
            if (!directRem && divisorArg) {
              Value rhs = remOp.getRhs();
              for (int i = 0; i < 6 && rhs.getDefiningOp(); i++) {
                auto *rdef = rhs.getDefiningOp();
                auto rname = rdef->getName().getStringRef();
                if (rname == "arith.extsi" || rname == "arith.extui" ||
                    rname == "arith.index_cast" || rname == "tt.splat")
                  rhs = rdef->getOperand(0);
                else
                  break;
              }
              if (rhs == divisorArg)
                directRem = true;
            }
          }
          if (directRem ||
              traceToRemSIOp(offset, pageSize, /*maxDepth=*/6,
                             divisorArg)) {
            // block_offset = token_start % PAGE_SIZE ∈ [0, pageSize)
            foundBlockOffset = true;
            break;
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
        // Tile starts at token_start = tile_idx * tileTokens, so block_offset =
        // token_start % pageSize is tile-aligned.  The tile is guaranteed not to
        // cross a page boundary iff it is page-aligned:
        //   pageSize % tileTokens == 0  → block_offset ∈ {0, TILE, 2·TILE, …},
        //     tile ends exactly at the page boundary.
        //   tileTokens % pageSize == 0  → block_offset ≡ 0 (tile ≡ page).
        // V20 W6'a family A: with a RUNTIME page size the numeric
        // comparison is impossible — stay conservative (mask path).
        staticallySafe = !runtimePage &&
            (pageSize % tileTokens == 0 || tileTokens % pageSize == 0);
      } else {
        // Fallback: use P1's page_boundary_safe attribute
        if (auto attr = loadOp->getAttrOfType<BoolAttr>(
                "pact.page_boundary_safe")) {
          staticallySafe = attr.getValue();
        }
        // Additional check: tileTokens % pageSize == 0 is the old heuristic
      }

      // === Step 3: compute pageLocalContiguity (M9: exact via Paged Linear Layout) ===
      // The head_idx dimension within a page is always stride-1 (contiguous),
      // so its memory contiguity is exactly headSize — independent of whether
      // the tile crosses a page boundary (which only concerns the *token*
      // dimension).  The old code capped pageLocalContiguity at pageSize when
      // pageSize < tileTokens, wrongly applying a token-dimension bound to the
      // head_dim contiguity.
      //
      // M9: compute this exactly from the page-internal memory layout expressed
      // as an F₂-linear layout, instead of the semantic staticallySafe branch.
      int64_t pageLocalContiguity = headSize; // fallback (non-power-of-2)
      int64_t tokenContiguity = 1;            // token is page-strided (stride headDim)
      if (isPowerOf2(headSize) && isPowerOf2(pageSize)) {
        LinearLayout pagedMem =
            buildPagedMemoryLayout(headSize, pageSize, &getContext());
        // head_idx basis is stride 1 → contiguous (headDim).
        pageLocalContiguity = pagedMem.getNumConsecutiveInOut();
        // token_in_page basis is stride headDim → NOT contiguous.  Transpose so
        // token is the most-minor input dim and getNumConsecutiveInOut() returns
        // the exact token-dimension contiguity (= 1).  This is the principled
        // F₂ evidence that only head_idx should be vectorization-overridden.
        auto headIdx = StringAttr::get(&getContext(), "head_idx");
        auto tokenInPage = StringAttr::get(&getContext(), "token_in_page");
        tokenContiguity = pagedMem.transposeIns({tokenInPage, headIdx})
                              .getNumConsecutiveInOut();
      }

      // === Step 4: output analysis as IR attributes (schema v2) ===
      auto ctx = &getContext();
      SmallVector<int64_t, 2> dimContiguity{tokenContiguity,
                                            pageLocalContiguity};
      loadOp->setAttr("pact.pagelocal.dim_contiguity",
          DenseI64ArrayAttr::get(ctx, dimContiguity));
      loadOp->setAttr("pact.pagelocal.statically_safe",
          BoolAttr::get(ctx, staticallySafe));

      numAnalyzed++;
      LDBG("PACT P2: load analyzed: dimContiguity=[" << tokenContiguity << ","
           << pageLocalContiguity << "]"
           << ", staticSafe=" << staticallySafe
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
