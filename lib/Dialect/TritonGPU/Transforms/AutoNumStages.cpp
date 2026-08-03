//===- AutoNumStages.cpp - PACT P6 Auto num_stages Selection --------------===//
//
// P6: Sets ttg.num_stages attribute on paged scf.for loops based on
//     page-aware heuristics (page size, tile size, SMEM budget).
//     Runs BEFORE AssignLatencies/Schedule/Pipeline so the pipeline
//     pass picks up the overridden value.
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

#define DEBUG_TYPE "pact-auto-stage"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::triton::gpu {

#define GEN_PASS_DEF_PACTAUTONUMSTAGES
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

namespace {

static bool isEnabled() {
  const char *env = std::getenv("PACT_ENABLE_AUTO_NUM_STAGES");
  return !env || std::string(env) != "0";
}

static int getMaxPipelineStages() {
  const char *env = std::getenv("PACT_MAX_PIPELINE_STAGES");
  if (env) {
    int val = std::atoi(env);
    return std::max(2, std::min(val, 6));
  }
  return 4;
}

// Calculate the optimal number of pipeline stages for a given page/tile config.
static int computeOptimalNumStages(int64_t tileBytes, int estIterations,
                                   int pageSize, int tileTokens,
                                   int defaultStages) {
  int minStages = 2;
  int maxStages = getMaxPipelineStages();

  // SMEM constraint
  int maxStagesBySMEM = 1;
  if (tileBytes > 0)
    maxStagesBySMEM = std::max(1, 102400 / (int)tileBytes);
  int upperBound = std::min(maxStages, maxStagesBySMEM);

  // Iteration constraint
  if (estIterations > 0)
    upperBound = std::min(upperBound,
                          std::max(2, std::min(maxStages, estIterations / 4)));

  upperBound = std::max(minStages, upperBound);
  if (upperBound <= minStages)
    return minStages;

  // Page-aware heuristic
  int tilesPerPage = (tileTokens > 0) ? pageSize / tileTokens : 1;

  if (tilesPerPage >= 4)
    return std::max(minStages, upperBound - 1); // high L2 locality
  if (tilesPerPage <= 1)
    return upperBound; // low locality
  return std::max(minStages, upperBound - 1); // medium
}

struct PACTAutoNumStagesPass
    : public impl::PACTAutoNumStagesBase<PACTAutoNumStagesPass> {

  void runOnOperation() override {
    if (!isEnabled())
      return;

    ModuleOp mod = getOperation();

    // Get default num_stages from existing attribute or environment
    int defaultStages = 3;
    if (auto attr = mod->getAttrOfType<mlir::IntegerAttr>(
            "ttg.num-stages"))
      defaultStages = attr.getInt();

    mod.walk([&](scf::ForOp forOp) {
      // Check if this loop has paged loads (via P4 hints)
      int64_t tileBytes = 0;
      int estIterations = 128;
      int pageSize = 16;
      int tileTokens = 16;
      bool hasPagedLoad = false;

      forOp.walk([&](Operation *op) {
        if (auto hint = op->getAttrOfType<mlir::IntegerAttr>(
                "pact.hint.tile_bytes")) {
          hasPagedLoad = true;
          tileBytes = std::max(tileBytes, hint.getInt());
        }
        if (auto hint = op->getAttrOfType<mlir::IntegerAttr>(
                "pact.hint.estimated_iterations"))
          estIterations = hint.getInt();
        if (auto attr = op->getAttrOfType<mlir::IntegerAttr>(
                "pact.hint.page_size"))
          pageSize = attr.getInt();
        if (auto attr = op->getAttrOfType<mlir::IntegerAttr>(
                "pact.hint.tile_tokens"))
          tileTokens = attr.getInt();
      });

      if (!hasPagedLoad)
        return WalkResult::advance();

      int optimal = computeOptimalNumStages(tileBytes, estIterations,
                                            pageSize, tileTokens,
                                            defaultStages);

      // Set as loop attribute so pipeline pass picks it up
      forOp->setAttr("ttg.num_stages",
                     mlir::IntegerAttr::get(
                         mlir::IntegerType::get(&getContext(), 32), optimal));

      llvm::errs() << "[PACT P6] num_stages: default=" << defaultStages
                   << " -> optimal=" << optimal
                   << " (tileB=" << tileBytes
                   << ", iters=" << estIterations
                   << ", pageSize=" << pageSize
                   << ", tilesPerPage="
                   << (tileTokens > 0 ? pageSize / tileTokens : 0) << ")\n";
      return WalkResult::advance();
    });
  }
};

} // anonymous namespace
} // namespace mlir::triton::gpu
