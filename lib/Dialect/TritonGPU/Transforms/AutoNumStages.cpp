//===- AutoNumStages.cpp - PACT P6 Architecture-Aware num_stages -----------===//
//
// P6: architecture-aware num_stages decision for paged attention loops.
//
// Inputs (in priority order):
//   1. pact.pgo.measured_iterations  (injected by this branch's compiler.py;
//      without it the static theory-only path is used)
//   2. statically-resolvable scf.for bounds
//   3. no estimate -> keep the native/default num_stages (no override)
//
// The SMEM footprint used by the occupancy model is computed exactly from the
// shared encoding that Triton's pipeline pass would use, including swizzle
// padding, via LinearLayout's "offset" input dimension.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Support/PactDecision.h"
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

static bool hasExplicitMaxPipelineStages() {
  return std::getenv("PACT_MAX_PIPELINE_STAGES") != nullptr;
}

static int getMaxPipelineStages() {
  const char *env = std::getenv("PACT_MAX_PIPELINE_STAGES");
  if (env) {
    int val = std::atoi(env);
    return std::max(2, std::min(val, 8));
  }
  return 4;
}

static int64_t getTotalElements(RankedTensorType ty) {
  int64_t total = 1;
  for (auto dim : ty.getShape())
    total *= dim;
  return total;
}

// Static trip count, or -1 when the bounds are runtime values.
static int64_t estimateTripCount(scf::ForOp forOp) {
  auto upperConst =
      forOp.getUpperBound().getDefiningOp<arith::ConstantOp>();
  auto lowerConst =
      forOp.getLowerBound().getDefiningOp<arith::ConstantOp>();
  if (!upperConst || !lowerConst)
    return -1;

  int64_t upperVal =
      mlir::cast<mlir::IntegerAttr>(upperConst.getValue()).getInt();
  int64_t lowerVal =
      mlir::cast<mlir::IntegerAttr>(lowerConst.getValue()).getInt();
  int64_t step = 1;
  if (auto stepOp = forOp.getStep().getDefiningOp<arith::ConstantOp>())
    step = mlir::cast<mlir::IntegerAttr>(stepOp.getValue()).getInt();
  if (step <= 0)
    return -1;
  return (upperVal - lowerVal) / step;
}

static int computeOptimalNumStages(int64_t tileBytes, int64_t estIterations,
                                   bool haveIterationEstimate, int pageSize,
                                   int tileTokens, int defaultStages,
                                   int numWarps) {
  // Without a static or measured trip count PACT must not override the native
  // pipeline decision.
  if (!haveIterationEstimate) {
    llvm::errs() << "[PACT P6] " << pact::SMDetector::getGPUName()
                 << ": no iteration estimate — keeping num_stages="
                 << defaultStages << "\n";
    return defaultStages;
  }

  // Theory-as-code: the feasible stage set and each candidate's occupancy are
  // computed from the capacity equations and the LinearLayout-exact SMEM
  // footprint.  The measured trip count (PGO) and the static bound are both
  // just inputs to this same computation.
  pact::SelectStagesInput input;
  input.tileBytes = tileBytes;
  input.estIterations = estIterations;
  input.defaultStages = defaultStages;
  // B5 fix: an *explicit* PACT_MAX_PIPELINE_STAGES stays a hard user cap.
  // When the knob is not set, the default value (4) must not silently clamp a
  // higher architecture default such as Hopper's 5.
  input.maxStages = getMaxPipelineStages();
  if (!hasExplicitMaxPipelineStages())
    input.maxStages = std::max(input.maxStages, defaultStages);
  input.numWarps = numWarps;

  auto decision = pact::selectNumStages(input);

  int tilesPerPage = (tileTokens > 0 && tileTokens <= pageSize)
                         ? pageSize / tileTokens : 1;
  llvm::errs() << "[PACT P6] " << pact::SMDetector::getGPUName()
               << ": selectNumStages=" << decision.numStages
               << " (feasible=[" << decision.feasibleMin << ","
               << decision.feasibleMax << "] smemBound=" << decision.smemBound
               << " iterBound=" << decision.iterBound
               << ", bestOcc=" << decision.bestOccupancy
               << ", occ=" << decision.occupancy
               << ", tile=" << tileBytes << "B, iters=" << estIterations
               << ", TPP=" << tilesPerPage << " [informational])\n";
  return decision.numStages;
}

struct PACTAutoNumStagesPass
    : public impl::PACTAutoNumStagesBase<PACTAutoNumStagesPass> {

  void runOnOperation() override {
    if (!isEnabled())
      return;

    ModuleOp mod = getOperation();

    // Pure prefill (no paged access pattern) was skipped by P1; keep P6
    // consistent with that skip decision.
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

    if (auto attr = mod->getAttrOfType<mlir::IntegerAttr>("tt.num_stages"))
      defaultStages = attr.getInt();

    // N3: publish the native/default baseline that P6 itself compared
    // against, so the PGO trigger pass uses the same reference instead of
    // guessing from a module attribute that the pipeline never writes.
    mod->setAttr("pact.native_num_stages",
                 mlir::IntegerAttr::get(
                     mlir::IntegerType::get(&getContext(), 32), defaultStages));

    // Optional PGO-provided measured trip count (injected by the PGO branch).
    // It enters the same selectNumStages computation as a static bound would.
    int64_t pgoIterations = -1;
    if (auto attr = mod->getAttrOfType<mlir::IntegerAttr>(
            "pact.pgo.measured_iterations"))
      pgoIterations = attr.getInt();

    int numWarps = pact::PactDecisionConstants::kDefaultNumWarps;
    if (auto attr = mod->getAttrOfType<mlir::IntegerAttr>("ttg.num-warps"))
      numWarps = attr.getInt();

    mod.walk([&](scf::ForOp forOp) {
      int64_t tileBytes = 0;
      int64_t exactSMEMBytes = 0;
      int pageSize = 16;
      int tileTokens = 16;
      bool hasPagedLoad = false;
      int64_t estIterations = estimateTripCount(forOp);
      bool haveIterationEstimate = estIterations > 0;
      if (pgoIterations > 0) {
        estIterations = pgoIterations;
        haveIterationEstimate = true;
      }

      forOp.walk([&](triton::LoadOp loadOp) {
        if (!loadOp->hasAttr("pact.paged_load"))
          return WalkResult::advance();

        hasPagedLoad = true;
        auto ty = cast<RankedTensorType>(loadOp.getResult().getType());
        int64_t elemBytes = std::max((int64_t)1, (int64_t)(ty.getElementTypeBitWidth() / 8));
        int64_t totalBytes = getTotalElements(ty) * elemBytes;
        tileBytes = std::max(tileBytes, totalBytes);

        if (auto attr =
                loadOp->getAttrOfType<mlir::IntegerAttr>("pact.page_size"))
          pageSize = attr.getInt();
        if (auto attr =
                loadOp->getAttrOfType<mlir::IntegerAttr>("pact.tile_tokens"))
          tileTokens = attr.getInt();

        // M9c: exact shared-memory footprint, including swizzle padding. The
        // physical footprint is the size of the shared layout's "offset" input
        // dimension — NOT getTotalInDimSize(), which would multiply by the CGA
        // block dimension for num_ctas > 1.
        if (ty.getEncoding()) {
          auto sharedEnc = mlir::triton::getSharedEncoding(loadOp);
          auto ll = triton::gpu::toLinearLayout(ty.getShape(), sharedEnc);
          auto offsetName = StringAttr::get(&getContext(), "offset");
          int64_t footprint = 0;
          if (ll.hasInDim(offsetName))
            footprint = ll.getInDimSize(offsetName) * elemBytes;
          exactSMEMBytes = std::max(exactSMEMBytes, footprint);
        }
        return WalkResult::advance();
      });

      if (!hasPagedLoad)
        return WalkResult::advance();

      if (exactSMEMBytes > 0 && exactSMEMBytes != tileBytes) {
        llvm::errs() << "[PACT P6 M9c] SMEM footprint: raw tileBytes="
                     << tileBytes << "B → exact swizzled=" << exactSMEMBytes
                     << "B (padding=" << (exactSMEMBytes - tileBytes)
                     << "B)\n";
        tileBytes = exactSMEMBytes;
      }

      int optimal = computeOptimalNumStages(
          tileBytes, estIterations, haveIterationEstimate, pageSize,
          tileTokens, defaultStages, numWarps);

      // Always publish the computed decision on the module so the PGO branch
      // can read the theory-selected stage count back from metadata even when
      // it equals the native default.
      mod->setAttr("pact.optimal_num_stages",
                   mlir::IntegerAttr::get(
                       mlir::IntegerType::get(&getContext(), 32), optimal));

      if (optimal == defaultStages) {
        llvm::errs() << "[PACT P6] Keeping default num_stages="
                     << defaultStages << " (optimal=" << optimal
                     << ", no change needed)\n";
        return WalkResult::advance();
      }

      // Set as loop attribute so the pipeline pass picks it up.
      auto stagesAttr = mlir::IntegerAttr::get(
          mlir::IntegerType::get(&getContext(), 32), optimal);
      forOp->setAttr("tt.num_stages", stagesAttr);

      llvm::errs() << "[PACT P6] " << pact::SMDetector::getGPUName()
                   << ": num_stages " << defaultStages << " → " << optimal
                   << " | tile=" << tileBytes << "B"
                   << " | iters=" << estIterations << " | TPP="
                   << (tileTokens > 0 && tileTokens <= pageSize
                           ? pageSize / tileTokens : 1)
                   << " | SMEM=" << (tileBytes * optimal) / 1024
                   << "KB/block\n";

      return WalkResult::advance();
    });
  }
};

} // anonymous namespace
} // namespace mlir::triton::gpu
