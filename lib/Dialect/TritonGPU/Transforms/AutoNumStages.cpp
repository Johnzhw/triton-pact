//===- AutoNumStages.cpp - PACT P6 Occupancy-Aware num_stages --------------===//
//
// P6 v3: Occupancy-aware + page-locality joint optimization for num_stages.
//
// Core insight: num_stages has a fundamental tradeoff:
//   num_stages ↑ → more preload → better latency hiding → POSITIVE
//   num_stages ↑ → SMEM ↑ → occupancy ↓ → warps ↓ → bar.sync wait ↑ → NEGATIVE
//
// Ampere strategy: Occupancy-First — don't increase num_stages unless
//   the latency hiding benefit clearly outweighs the occupancy cost.
//
// Hopper strategy: Pipeline-First — larger SMEM enables more aggressive
//   num_stages, and TMA handles 2D copies efficiently.
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
#include "triton/Support/PactSMDetect.h"

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
    return std::max(2, std::min(val, 8));
  }
  return 4;
}

// ═══════════════════════════════════════════════════════════════
// P6 v3: Occupancy-Aware + Page-Locality Joint Optimization
// ═══════════════════════════════════════════════════════════════

static int computeOptimalNumStages(int64_t tileBytes, int estIterations,
                                   int pageSize, int tileTokens,
                                   int defaultStages) {
  auto sm = pact::SMDetector::getResources();
  auto budget = pact::SMDetector::computePipelineBudget(
      tileBytes, estIterations, pageSize, tileTokens, defaultStages);

  int tilesPerPage = (tileTokens > 0 && tileTokens <= pageSize)
                        ? pageSize / tileTokens : 1;

  // ═══════════════════════════════════════════════════════════
  // Ampere (SM 80-89): Occupancy-First Heuristic
  //
  // Key constraint: num_stages 3→4 drops occupancy sharply
  // (1024→768 threads/SM on RTX 3080).
  //
  // Strategy: default to keeping defaultStages. Only increase
  // when there's clear evidence that latency hiding benefit
  // outweighs the occupancy cost.
  // ═══════════════════════════════════════════════════════════
  if (sm.smVersion < 90) {
    // === Short sequence (≤16 iters): minimize stages, maximize occupancy ===
    if (estIterations <= 16) {
      llvm::errs() << "[PACT P6 v3] " << pact::SMDetector::getGPUName()
                   << ": short seq (" << estIterations
                   << " iters) → num_stages=2 (max occupancy)\n";
      return 2;
    }

    // === Medium sequence (17-64 iters): keep default ===
    if (estIterations <= 64) {
      llvm::errs() << "[PACT P6 v3] " << pact::SMDetector::getGPUName()
                   << ": medium seq (" << estIterations
                   << " iters) → num_stages=" << defaultStages
                   << " (keep default)\n";
      return defaultStages;
    }

    // === Long sequence (65+ iters): evaluate pipeline benefit vs occupancy cost ===
    // Only increase stages if ALL conditions met:
    //   1. Large tile (≥2KB) — worth the SMEM cost
    //   2. Low page locality (tilesPerPage ≤ 2) — need pipeline to hide latency
    //   3. Occupancy loss < 20%
    bool largeTile = tileBytes >= 2048;
    bool lowLocality = tilesPerPage <= 2;
    double occCurrent = pact::SMDetector::estimateOccupancy(
        defaultStages, tileBytes * defaultStages, 96);
    double occNext = pact::SMDetector::estimateOccupancy(
        defaultStages + 1, tileBytes * (defaultStages + 1), 96);
    bool occAcceptable =
        occCurrent > 0 && (occCurrent - occNext) / occCurrent < 0.20;

    if (largeTile && lowLocality && occAcceptable && estIterations >= 128) {
      int optimal = std::min(defaultStages + 1, sm.optimalNumStages);
      llvm::errs() << "[PACT P6 v3] " << pact::SMDetector::getGPUName()
                   << ": long seq + large tile + low locality"
                   << " → num_stages=" << optimal
                   << " (occ: " << (int)(occCurrent * 100) << "% → "
                   << (int)(occNext * 100) << "%)\n";
      return optimal;
    }

    // For high page locality (tilesPerPage ≥ 4): fewer stages suffice
    if (tilesPerPage >= 4) {
      llvm::errs() << "[PACT P6 v3] " << pact::SMDetector::getGPUName()
                   << ": high page locality (" << tilesPerPage
                   << " tiles/page) → num_stages=2 (L2 cache friendly)\n";
      return 2;
    }

    llvm::errs() << "[PACT P6 v3] " << pact::SMDetector::getGPUName()
                 << ": keeping num_stages=" << defaultStages
                 << " (tile=" << tileBytes << "B, iters=" << estIterations
                 << ", TPP=" << tilesPerPage
                 << ", occLoss="
                 << (occCurrent > 0 ? (int)((occCurrent - occNext) / occCurrent * 100)
                                   : 0) << "%)\n";
    return defaultStages;
  }

  // ═══════════════════════════════════════════════════════════
  // Hopper (SM 90+): Pipeline-First Heuristic
  //
  // Larger SMEM (228KB) and TMA hardware enable more aggressive
  // pipeline depths. Use computePipelineBudget for bounds.
  // ═══════════════════════════════════════════════════════════
  int maxStages = getMaxPipelineStages();
  int candidate = std::min({sm.optimalNumStages + 2,
                             budget.maxStagesBySMEM,
                             budget.maxStagesByIters,
                             budget.maxStagesByOccupancy,
                             maxStages});

  // Page-locality adjustment (Hopper: TMA 2D copies benefit from page locality)
  if (tilesPerPage >= 4) {
    // High L2 locality → TMA 2D copy is efficient → fewer stages needed
    candidate = std::min(candidate, sm.optimalNumStages - 1);
  } else if (tilesPerPage <= 1) {
    // Low locality → need more stages to hide latency
    candidate = std::min(candidate, sm.optimalNumStages + 2);
  }

  candidate = std::clamp(candidate, 2, 7);

  llvm::errs() << "[PACT P6 v3] " << pact::SMDetector::getGPUName()
               << ": num_stages=" << candidate
               << " (tile=" << tileBytes << "B, iters=" << estIterations
               << ", TPP=" << tilesPerPage
               << ", smemBudget=" << budget.smemBudget / 1024 << "KB)\n";
  return candidate;
}

struct PACTAutoNumStagesPass
    : public impl::PACTAutoNumStagesBase<PACTAutoNumStagesPass> {

  void runOnOperation() override {
    if (!isEnabled())
      return;

    ModuleOp mod = getOperation();

    // Skip prefill kernels (tagged by P1 PageTransform)
    if (auto ktype = mod->getAttrOfType<StringAttr>("pact.kernel_type")) {
      if (ktype.getValue() == "prefill") {
        llvm::errs() << "[PACT P6] Prefill kernel detected, skipping.\n";
        return;
      }
    }

    // Architecture-aware default: 2-3 for Ampere, 5 for Hopper
    int defaultStages = 3;
    auto sm = pact::SMDetector::getResources();
    if (sm.smVersion >= 90)
      defaultStages = sm.optimalNumStages; // 5 on Hopper

    // Check for existing attribute
    if (auto attr = mod->getAttrOfType<mlir::IntegerAttr>("ttg.num-stages"))
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

      // Only set attribute if different from default
      if (optimal == defaultStages) {
        llvm::errs() << "[PACT P6] Keeping default num_stages="
                     << defaultStages << " (optimal=" << optimal
                     << ", no change needed)\n";
        return WalkResult::advance();
      }

      // Set as loop attribute so pipeline pass picks it up.
      auto stagesAttr = mlir::IntegerAttr::get(
          mlir::IntegerType::get(&getContext(), 32), optimal);
      forOp->setAttr("tt.num_stages", stagesAttr);
      forOp->setAttr("ttg.num_stages", stagesAttr);

      // Also write to module attribute for compiler.py fallback
      mod->setAttr("pact.optimal_num_stages",
                   mlir::IntegerAttr::get(
                       mlir::IntegerType::get(&getContext(), 32), optimal));

      llvm::errs() << "[PACT P6 v3] " << pact::SMDetector::getGPUName()
                   << ": num_stages " << defaultStages << " → " << optimal
                   << " | tile=" << tileBytes << "B"
                   << " | iters=" << estIterations
                   << " | TPP="
                   << (tileTokens > 0 && tileTokens <= pageSize
                           ? pageSize / tileTokens : 1)
                   << " | SMEM=" << (tileBytes * optimal) / 1024 << "KB/block\n";

      return WalkResult::advance();
    });
  }
};

} // anonymous namespace
} // namespace mlir::triton::gpu
