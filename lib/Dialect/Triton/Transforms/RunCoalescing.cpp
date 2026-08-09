//===- RunCoalescing.cpp - PACT P9 Importance-Guided Run Coalescing --------===//
//
// P9 v2: Groups sparse token selections into page-aware contiguous runs,
//     guided by token importance (hot/warm/cold) based on position within page.
//     Hot tokens → keep fine-grained loads (precise access)
//     Cold tokens → aggressive coalescing (batch load, fewer instructions)
//
//     Default OFF — requires external pact.token_mask attribute to trigger.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <string>
#include <algorithm>

#define DEBUG_TYPE "pact-run-coalesce"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::triton {

#define GEN_PASS_DEF_PACTRUNCOALESCING
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

namespace {

static bool isEnabled() {
  const char *env = std::getenv("PACT_ENABLE_RUN_COALESCE");
  if (!env) return false; // Default OFF
  return std::string(env) == "1";
}

// ═══════════════════════════════════════════════════════════════
// Token importance: guides coalescing aggressiveness
//
// Based on position-in-page heuristic:
//   Hot  — Page start token (block table lookup is "fresh")
//   Warm — Page middle tokens
//   Cold — Page end tokens (batch loading candidates)
//
// Future: integrate attention score feedback from Phase2 Proton profiler.
// ═══════════════════════════════════════════════════════════════
enum class TokenImportance {
  Hot,   // Keep independent load — precise access
  Warm,  // Moderate coalescing (2-4 tokens)
  Cold   // Aggressive coalescing (4-8 tokens)
};

static TokenImportance getTokenImportance(int64_t offsetInPage,
                                          int64_t pageSize) {
  if (pageSize <= 0) return TokenImportance::Warm;

  // First 25% of page: Hot (block table lookup most recent)
  if (offsetInPage < pageSize / 4)
    return TokenImportance::Hot;

  // Middle 50% of page: Warm
  if (offsetInPage < pageSize * 3 / 4)
    return TokenImportance::Warm;

  // Last 25% of page: Cold (batch loading candidates)
  return TokenImportance::Cold;
}

// Max merge size based on importance level
static int getMaxMergeSize(TokenImportance importance) {
  switch (importance) {
  case TokenImportance::Hot:  return 1;  // No merging
  case TokenImportance::Warm: return 2;  // Moderate
  case TokenImportance::Cold: return 4;  // Aggressive
  }
  return 1;
}

struct TokenRun {
  int64_t pageId;
  int64_t startOffset;
  int64_t runLength;
  bool isFullPage;
  TokenImportance importance; // ← v2: dominant importance of this run

  enum LoadStrategy { FullPageAsync, ContiguousRunAsync, GatherFallback };

  LoadStrategy getStrategy(int64_t minRunForAsync = 4) const {
    if (isFullPage) return FullPageAsync;
    if (runLength >= minRunForAsync) return ContiguousRunAsync;
    return GatherFallback;
  }

  const char *importanceStr() const {
    switch (importance) {
    case TokenImportance::Hot:  return "hot";
    case TokenImportance::Warm: return "warm";
    case TokenImportance::Cold: return "cold";
    }
    return "?";
  }
};

// ═══════════════════════════════════════════════════════════════
// v2: Importance-guided run coalescing
//
// Instead of simply grouping contiguous tokens, this algorithm
// considers token importance when deciding merge boundaries.
//
// Hot tokens: keep as individual runs (precise access for
//   recently-looked-up block table entries)
// Warm tokens: moderate merging (2-4 tokens/run)
// Cold tokens: aggressive merging (4-8 tokens/run)
// ═══════════════════════════════════════════════════════════════
static SmallVector<TokenRun> coalesceTokensToRuns(
    const SmallVector<int64_t> &selectedTokens, int64_t pageSize) {
  SmallVector<TokenRun> runs;
  if (selectedTokens.empty()) return runs;

  auto sorted = selectedTokens;
  std::sort(sorted.begin(), sorted.end());

  TokenRun current;
  current.pageId = sorted[0] / pageSize;
  current.startOffset = sorted[0] % pageSize;
  current.runLength = 1;
  current.isFullPage = false;
  current.importance = getTokenImportance(current.startOffset, pageSize);

  for (size_t i = 1; i < sorted.size(); i++) {
    int64_t pageId = sorted[i] / pageSize;
    int64_t offset = sorted[i] % pageSize;
    TokenImportance imp = getTokenImportance(offset, pageSize);

    bool samePage = (pageId == current.pageId);
    bool contiguous = (offset == current.startOffset + current.runLength);
    int maxMerge = getMaxMergeSize(current.importance);

    // Keep merging if: same page, contiguous, and within importance cap
    if (samePage && contiguous && current.runLength < maxMerge) {
      current.runLength++;
      // Upgrade importance to the more conservative level
      if (imp == TokenImportance::Hot)
        current.importance = TokenImportance::Hot;
      else if (imp == TokenImportance::Warm &&
               current.importance == TokenImportance::Cold)
        current.importance = TokenImportance::Warm;
    } else {
      current.isFullPage = (current.runLength == pageSize);
      runs.push_back(current);
      current = {pageId, offset, 1, false, imp};
    }
  }
  current.isFullPage = (current.runLength == pageSize);
  runs.push_back(current);
  return runs;
}

struct PACTRunCoalescingPass
    : public impl::PACTRunCoalescingBase<PACTRunCoalescingPass> {

  void runOnOperation() override {
    if (!isEnabled()) return;

    ModuleOp mod = getOperation();

    // Phase 0: early exit if no token_mask attribute anywhere
    bool hasTokenMask = false;
    mod.walk([&](Operation *op) {
      if (op->hasAttr("pact.token_mask")) {
        hasTokenMask = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (!hasTokenMask) {
      llvm::errs() << "[PACT P9] No token_mask found, pass skipped (v2).\n";
      return;
    }

    int numCoalesced = 0;
    int hotRuns = 0, warmRuns = 0, coldRuns = 0;

    mod.walk([&](triton::LoadOp loadOp) {
      if (!loadOp->hasAttr("pact.paged_load"))
        return WalkResult::advance();
      if (!loadOp->hasAttr("pact.token_mask"))
        return WalkResult::advance();

      int64_t pageSize = 16;
      if (auto attr = loadOp->getAttrOfType<mlir::IntegerAttr>(
              "pact.page_size"))
        pageSize = attr.getInt();

      // Read token_mask: which tokens are selected (sparse pattern)
      auto maskAttr = loadOp->getAttrOfType<mlir::DenseIntElementsAttr>(
          "pact.token_mask");
      if (!maskAttr) return WalkResult::advance();

      SmallVector<int64_t> selectedTokens;
      for (auto val : maskAttr.getValues<int64_t>()) {
        if (val >= 0) selectedTokens.push_back(val);
      }

      if (selectedTokens.empty()) return WalkResult::advance();

      // v2: importance-guided coalescing
      auto runs = coalesceTokensToRuns(selectedTokens, pageSize);

      int fullPages = 0, contiguousRuns = 0, gathers = 0;
      for (auto &run : runs) {
        switch (run.getStrategy()) {
        case TokenRun::FullPageAsync: fullPages++; break;
        case TokenRun::ContiguousRunAsync: contiguousRuns++; break;
        case TokenRun::GatherFallback: gathers++; break;
        }
        // Count by importance
        switch (run.importance) {
        case TokenImportance::Hot:  hotRuns++; break;
        case TokenImportance::Warm: warmRuns++; break;
        case TokenImportance::Cold: coldRuns++; break;
        }
      }

      // Attach coalescing metadata
      auto ctx = &getContext();
      auto i64Ty = mlir::IntegerType::get(ctx, 64);
      loadOp->setAttr("pact.run_coalesce.full_pages",
          mlir::IntegerAttr::get(i64Ty, fullPages));
      loadOp->setAttr("pact.run_coalesce.contiguous_runs",
          mlir::IntegerAttr::get(i64Ty, contiguousRuns));
      loadOp->setAttr("pact.run_coalesce.gather_fallbacks",
          mlir::IntegerAttr::get(i64Ty, gathers));
      loadOp->setAttr("pact.run_coalesce.total_runs",
          mlir::IntegerAttr::get(i64Ty, (int64_t)runs.size()));
      loadOp->setAttr("pact.run_coalesce.hot_runs",
          mlir::IntegerAttr::get(i64Ty, hotRuns));
      loadOp->setAttr("pact.run_coalesce.warm_runs",
          mlir::IntegerAttr::get(i64Ty, warmRuns));
      loadOp->setAttr("pact.run_coalesce.cold_runs",
          mlir::IntegerAttr::get(i64Ty, coldRuns));

      numCoalesced++;
      return WalkResult::advance();
    });

    if (numCoalesced > 0) {
      llvm::errs() << "[PACT P9] Importance-guided coalescing (v2): "
                   << numCoalesced << " sparse load(s)"
                   << " | hot=" << hotRuns
                   << " warm=" << warmRuns
                   << " cold=" << coldRuns << "\n";
    }
  }
};

} // anonymous namespace
} // namespace mlir::triton
