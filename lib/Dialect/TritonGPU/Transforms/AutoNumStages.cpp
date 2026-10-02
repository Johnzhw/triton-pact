//===- AutoNumStages.cpp - PACT P6 Architecture-Aware num_stages -----------===//
//
// P6: architecture-aware num_stages decision for paged attention loops.
//
// Inputs (theory-only path, no profile-collected hardware parameters):
//   1. statically-resolvable scf.for bounds
//   2. no estimate -> keep the native/default num_stages (no override)
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
                                   int numWarps, pact::SelectStagesInput extra) {
  // Without a statically-resolvable trip count PACT must not override the
  // native *stage-count* decision (no profile input exists in this tree).
  // V9-A1: an explicit tt.num_stages loop attribute is still required to
  // unlock pipelining for dot-free loops — AssignLatencies only pipelines
  // loads of such loops when pipelineWithoutDot is set by the attribute, so
  // a paged KV loop without tl.dot is otherwise never scheduled (0 cp.async
  // on the naive kernel while the hand-optimised vLLM kernel pipelines via
  // its MMA path).  The conservative floor is 2 stages: the minimal
  // fill/drain prologue, measured neutral on the shortest shapes and
  // >=1.3x on long sequences (suite/results/v9/a0_probe.json).
  if (!haveIterationEstimate) {
    llvm::errs() << "[PACT P6] " << pact::SMDetector::getGPUName()
                 << ": no static iteration estimate — keeping num_stages="
                 << defaultStages << "\n";
    return defaultStages;
  }

  // Theory-as-code: the feasible stage set and the occupancy of each stage
  // count are computed from the capacity equations (SMDetector) and the
  // LinearLayout-exact SMEM footprint; the discretization tolerance is
  // derived from the equations themselves.  No hard-coded stage thresholds.
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
  input.regsPerThread = extra.regsPerThread;
  input.stallMemoryPermille = extra.stallMemoryPermille;
  input.smEfficiencyPermille = extra.smEfficiencyPermille;
  input.stallPenaltyPerExtraStage = extra.stallPenaltyPerExtraStage;
  input.smEffBonusPerExtraStage = extra.smEffBonusPerExtraStage;
  input.kvWorkingSetBytes = extra.kvWorkingSetBytes;

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
               << ", TPP=" << tilesPerPage
               << ", kv=" << decision.kvWorkingSetBytes
               << "B, l2=" << decision.l2Bytes << "B"
               << (decision.l2Cold
                       ? (decision.l2Deepened ? ", l2cold=deep" : ", l2cold=kept")
                       : "")
               << " [informational])\n";
  return decision.numStages;
}

struct PACTAutoNumStagesPass
    : public impl::PACTAutoNumStagesBase<PACTAutoNumStagesPass> {

  void runOnOperation() override {
    if (!isEnabled())
      return;

    ModuleOp mod = getOperation();

    // Pure prefill (no paged access pattern) was skipped by P1; keep P6
    // consistent with that skip decision.  V20 W3' bisect tightens the
    // domain further: on NON-paged kernels (tritonbench inductor/
    // elementwise family) the global stage choice misfired -- the five
    // regressing cells (0.641/0.856/0.894/0.905/0.941) all return to
    // band with P6 off (suite/results/v20/tb_pair/bisect_nop6.log).
    // The first gate cut (prefill/unknown) fixed 3/5; the two remaining
    // inductor fusions classify prefill_paged FALSE-POSITIVELY (a divsi
    // + tile signal that is not real paging), so the firing domain is
    // narrowed to the families with MEASURED P6 upside: decode_paged
    // (V9 >=1.3x long-S micro evidence) and paged_rt_or_gather
    // (W6'c/W7' target ops).  prefill_paged keeps the native default
    // until it earns its own pairing evidence.
    if (auto ktype = mod->getAttrOfType<StringAttr>("pact.kernel_type")) {
      auto kv = ktype.getValue();
      if (kv != "decode_paged" && kv != "paged_rt_or_gather") {
        llvm::errs() << "[PACT P6] Out-of-domain kernel (" << kv
                     << ") detected, skipping (V20 W3' domain gate).\n";
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

    int numWarps = pact::PactDecisionConstants::kDefaultNumWarps;
    if (auto attr = mod->getAttrOfType<mlir::IntegerAttr>("ttg.num-warps"))
      numWarps = attr.getInt();

    pact::SelectStagesInput hw;
    hw.regsPerThread = pact::PactDecisionConstants::kUnknownRegsPerThread;
    if (auto attr = mod->getAttrOfType<mlir::IntegerAttr>("pact.hw.regs_per_thread"))
      if (attr.getInt() > 0)
        hw.regsPerThread = attr.getInt();
    if (auto attr = mod->getAttrOfType<mlir::IntegerAttr>(
            "pact.hw.stall_memory_permille"))
      hw.stallMemoryPermille = (int)attr.getInt();
    if (auto attr = mod->getAttrOfType<mlir::IntegerAttr>(
            "pact.hw.sm_efficiency_permille"))
      hw.smEfficiencyPermille = (int)attr.getInt();
    // V16-T6: activate the measured-counter coefficients.  v15 wired the
    // permilles into the hint attrs but left both coefficients at the
    // 0.0 default ("supplied by the caller" -- no caller ever did), so
    // the runtime truths never reached the stage scoring.  v1
    // calibration, aligned with the decider's counter_adjust rule:
    // a stall-heavy, throughput-starved kernel benefits from deeper
    // pipelining -> the bonus term (which multiplies (1 - smEffFrac))
    // scales with the measured stall fraction.  Absent hints keep both
    // coefficients at zero, so fixtures without hints (lit, dump_ir)
    // stay bit-identical.
    if (hw.stallMemoryPermille && hw.stallMemoryPermille.value() > 0)
      hw.smEffBonusPerExtraStage =
          0.05 * hw.stallMemoryPermille.value() / 1000.0;
    int64_t hwIters = -1;
    if (auto attr = mod->getAttrOfType<mlir::IntegerAttr>(
            "pact.hw.measured_iterations"))
      hwIters = attr.getInt();
    // S1: optional KV-heads hint.  The paged loads are per-KV-head tiles
    // (heads are split across the grid), so the IR alone cannot see how many
    // heads stream through L2; the PGO hint scales the working-set estimate.
    // Default 1 keeps the pre-S1 shapes classified exactly as before.
    int64_t kvHeads = 1;
    if (auto attr = mod->getAttrOfType<mlir::IntegerAttr>("pact.hw.kv_heads"))
      if (attr.getInt() > 0)
        kvHeads = attr.getInt();
    // V18 T8c: per-layer weight bytes (qkv + gate_up) competing with the
    // KV stream for L2.  0 (absent hint) keeps the verdict bit-identical.
    if (auto attr = mod->getAttrOfType<mlir::IntegerAttr>(
            "pact.hw.weight_bytes"))
      if (attr.getInt() > 0)
        hw.weightBytes = attr.getInt();

    mod.walk([&](scf::ForOp forOp) {
      int64_t tileBytes = 0;
      int64_t exactSMEMBytes = 0;
      int pageSize = 16;
      int tileTokens = 16;
      bool hasPagedLoad = false;
      int64_t headDim = 0;   // S1: pact.head_dim_size annotation
      int64_t kvElemBytes = 0; // S1: widest paged-load element seen
      int64_t estIterations = estimateTripCount(forOp);
      bool haveIterationEstimate = estIterations > 0;
      // V9-A1: dot-free loops need an explicit tt.num_stages attribute to be
      // scheduled at all (pipelineWithoutDot); loops that feed a dot are
      // pipelined natively and must keep the historical no-touch behaviour.
      // V10-P0 (B1 cleanup): interface-based probe with early interrupt
      // instead of a full string-matching walk. The interface set is a
      // superset of "tt.dot" (e.g. dot_scaled); those feed the MMA pipeline
      // natively too, so they belong on the hasDot side — the decode target
      // family contains none of them and lit pins the existing behaviour.
      bool hasDot = false;
      forOp.walk([&](Operation *op) {
        if (isa<mlir::triton::DotOpInterface>(op)) {
          hasDot = true;
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
      if (hwIters > 0) {
        estIterations = hwIters;
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
        if (auto attr = loadOp->getAttrOfType<mlir::IntegerAttr>(
                "pact.head_dim_size"))
          headDim = std::max(headDim, attr.getInt());
        kvElemBytes = std::max(kvElemBytes, elemBytes);

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

      // S1 L2-residency working set: the K+V bytes one sequence streams
      // through this loop.  S = iterations × tokens-per-iteration; the ×2 is
      // K+V; the kv_heads hint scales the per-head tiles to the sequence's
      // full stream.  0 (no static trip count or no head-dim annotation)
      // classifies as "hot" and leaves the decision unchanged.
      hw.kvWorkingSetBytes =
          haveIterationEstimate && headDim > 0 && kvElemBytes > 0
              ? estIterations * tileTokens * headDim * kvElemBytes * 2 * kvHeads
              : 0;

      int optimal = computeOptimalNumStages(
          tileBytes, estIterations, haveIterationEstimate, pageSize,
          tileTokens, defaultStages, numWarps, hw);
      if (!haveIterationEstimate && !hasDot) {
        // V9-A1 floor: dot-free paged loop, no trip-count estimate. Without
        // the attribute AssignLatencies never schedules it (0 cp.async); 2
        // stages is the minimal prologue — neutral on the shortest shapes,
        // >=1.3x on long ones (suite/results/v9/a0_probe.json).
        optimal = std::min(optimal, 2);
      }
      if (const char *env = std::getenv("PACT_OVERRIDE_STAGES")) {
        int pinned = std::atoi(env);
        if (pinned >= 2 && pinned <= 8 && pinned != optimal) {
          llvm::errs() << "[PACT P6] OVERRIDE_STAGES " << optimal << " -> "
                       << pinned << "\n";
          optimal = pinned;
        } else if (pinned >= 2 && pinned <= 8) {
          optimal = pinned;
        }
      }

      // Always publish the computed decision on the module so the PGO branch
      // can read the theory-selected stage count back from metadata even when
      // it equals the native default.
      mod->setAttr("pact.optimal_num_stages",
                   mlir::IntegerAttr::get(
                       mlir::IntegerType::get(&getContext(), 32), optimal));

      if (optimal == defaultStages) {
        if (!hasDot) {
          // V9-A1: even when the theory keeps the native value, dot-free
          // loops need the explicit attribute to be scheduled at all.
          llvm::errs() << "[PACT P6] Keeping default num_stages="
                       << defaultStages << " (optimal=" << optimal
                       << ", writing loop attr to enable dot-free pipelining)\n";
          auto confirmAttr = mlir::IntegerAttr::get(
              mlir::IntegerType::get(&getContext(), 32), optimal);
          forOp->setAttr("tt.num_stages", confirmAttr);
        } else {
          llvm::errs() << "[PACT P6] Keeping default num_stages="
                       << defaultStages << " (optimal=" << optimal
                       << ", no change needed; dot loop scheduled natively)\n";
        }
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
