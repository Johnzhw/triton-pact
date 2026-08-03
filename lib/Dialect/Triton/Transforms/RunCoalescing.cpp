//===- RunCoalescing.cpp - PACT Contiguous-Run Coalescing -----------------===//
//
// P9: Groups sparse token selections into page-aware contiguous runs.
//     Runs get wide cp.async; isolated tokens use gather fallback.
//     Default OFF — requires external pact.token_mask attribute.
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

struct TokenRun {
  int64_t pageId;
  int64_t startOffset;
  int64_t runLength;
  bool isFullPage;
  enum LoadStrategy { FullPageAsync, ContiguousRunAsync, GatherFallback };
  LoadStrategy getStrategy(int64_t minRunForAsync = 4) const {
    if (isFullPage) return FullPageAsync;
    if (runLength >= minRunForAsync) return ContiguousRunAsync;
    return GatherFallback;
  }
};

// Core algorithm: sort+group sparse tokens into page-aware runs
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

  for (size_t i = 1; i < sorted.size(); i++) {
    int64_t pageId = sorted[i] / pageSize;
    int64_t offset = sorted[i] % pageSize;
    if (pageId == current.pageId &&
        offset == current.startOffset + current.runLength) {
      current.runLength++;
    } else {
      current.isFullPage = (current.runLength == pageSize);
      runs.push_back(current);
      current = {pageId, offset, 1, false};
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
    int numCoalesced = 0;

    mod.walk([&](triton::LoadOp loadOp) {
      if (!loadOp->hasAttr("pact.paged_load"))
        return WalkResult::advance();
      if (!loadOp->hasAttr("pact.token_mask"))
        return WalkResult::advance();

      int64_t pageSize = 16;
      if (auto attr = loadOp->getAttrOfType<mlir::IntegerAttr>(
              "pact.page_size"))
        pageSize = attr.getInt();

      auto maskAttr = loadOp->getAttrOfType<mlir::DenseIntElementsAttr>(
          "pact.token_mask");
      if (!maskAttr) return WalkResult::advance();

      SmallVector<int64_t> selectedTokens;
      for (auto val : maskAttr.getValues<int64_t>()) {
        if (val >= 0) selectedTokens.push_back(val);
      }

      if (selectedTokens.empty()) return WalkResult::advance();

      auto runs = coalesceTokensToRuns(selectedTokens, pageSize);

      int fullPages = 0, contiguousRuns = 0, gathers = 0;
      for (auto &run : runs) {
        switch (run.getStrategy()) {
        case TokenRun::FullPageAsync: fullPages++; break;
        case TokenRun::ContiguousRunAsync: contiguousRuns++; break;
        case TokenRun::GatherFallback: gathers++; break;
        }
      }

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

      LDBG("PACT P9: " << selectedTokens.size() << " tokens -> "
           << runs.size() << " runs (fp=" << fullPages
           << ", cr=" << contiguousRuns << ", gf=" << gathers << ")");
      numCoalesced++;
      return WalkResult::advance();
    });

    if (numCoalesced > 0) {
      llvm::errs() << "[PACT P9] Run coalescing: processed "
                   << numCoalesced << " sparse load(s)\n";
    }
  }
};

} // anonymous namespace
} // namespace mlir::triton
