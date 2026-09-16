//===- AutoNumWarps.cpp - PACT P11 num_warps Selection --------------------===//
//
// P11 (TTIR): recommends a num_warps value for paged attention kernels and
// writes pact.optimal_num_warps on the module; compiler.py reads it before
// make_ttgir to override opt.num_warps.
//
// Inputs (theory-only path, no profile-collected hardware parameters):
//   1. paged-load tile geometry from IR attributes
//   2. the SMDetector hardware-capacity occupancy model (L2 equations)
//
// Without a resolvable signal the recommendation stays at the Triton default
// (4).
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "triton/Support/PactDecision.h"
#include "triton/Support/PactSMDetect.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <optional>
#include <string>
#include <algorithm>

namespace mlir::triton {

#define GEN_PASS_DEF_PACTAUTONUMWARPS
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

namespace {

static bool isEnabled() {
  const char *env = std::getenv("PACT_ENABLE_AUTO_NUM_WARPS");
  return env && std::string(env) != "0";
}

struct PACTAutoNumWarpsPass
    : public impl::PACTAutoNumWarpsBase<PACTAutoNumWarpsPass> {

  void runOnOperation() override {
    if (!isEnabled()) {
      llvm::errs() << "[PACT P11] AutoNumWarps: disabled\n";
      return;
    }

    ModuleOp mod = getOperation();
    int numPagedLoads = 0;
    int64_t maxTileBytes = 0;

    mod.walk([&](Operation *op) {
      if (!op->hasAttr("pact.paged_load"))
        return WalkResult::advance();
      numPagedLoads++;
      auto resultTy = dyn_cast<RankedTensorType>(op->getResultTypes()[0]);
      if (!resultTy)
        return WalkResult::advance();

      int64_t headDim = 64;
      if (auto attr =
              op->getAttrOfType<IntegerAttr>("pact.head_dim_size"))
        headDim = attr.getInt();
      int64_t tile = 1;
      if (resultTy.getShape().size() >= 2)
        tile = resultTy.getShape()[0];
      int64_t elemBytes =
          std::max((int64_t)1,
                   (int64_t)(resultTy.getElementTypeBitWidth() / 8));
      maxTileBytes = std::max(maxTileBytes, tile * headDim * elemBytes);
      return WalkResult::advance();
    });

    int optimalWarps = pact::PactDecisionConstants::kDefaultNumWarps;
    // Data gate (Phase 1, 2026-09-08): substituting SMDetector::optimalNumStages
    // for the historical stagesPerBlock=3 changes the SM80 16x64 f16 choice
    // from 2 warps (stages=3, exact occupancy tie) back to 4 warps (stages=4).
    // That would break the SM80 P11 4→2 assertion, so the architecture default
    // is NOT used.  An *explicit* PACT_MAX_PIPELINE_STAGES still overrides.
    //
    // V10-P0 (BEH-1): stagesPerBlock is a *model input assumption* — it folds
    // into the SMEM term of the occupancy equations (tileBytes *
    // stagesPerBlock), it is not a fact about the loop.  P6 runs later (TTGIR)
    // and may ultimately write 2 (SM86 dot-free floor) or 5 (Hopper default);
    // P11 cannot read that in a single compilation, so 3 remains the default
    // (calibrated with the SM80 tie).  An *explicit* PACT_OVERRIDE_STAGES
    // aligns the assumption with the value P6 will pin (same 2..8 domain as
    // P6) and wins over PACT_MAX_PIPELINE_STAGES: a pin is more specific than
    // a cap.  Unset knobs keep the v9 behaviour bit-for-bit.
    int stagesPerBlock = 3;
    if (const char *env = std::getenv("PACT_MAX_PIPELINE_STAGES")) {
      int val = std::atoi(env);
      stagesPerBlock = std::max(2, std::min(val, 8));
    }
    if (const char *env = std::getenv("PACT_OVERRIDE_STAGES")) {
      int val = std::atoi(env);
      if (val >= 2 && val <= 8)
        stagesPerBlock = val;
    }
    // V11-3 (B2'): with NO explicit knob, align the assumption with the
    // value P6 will actually pin for this kernel family instead of the
    // historical constant 3.  For the decode_paged family P6 writes the
    // dot-free floor 2 on Ampere (V9-A1: AssignLatencies only pipelines
    // such loops when P6 sets tt.num_stages; the conservative floor is 2),
    // so assuming 3 there overstated the SMEM term and skewed the warp
    // decision inside the D128 flip window (tile ∈ (3462, 5193] B:
    // assumption 3 → 8 warps, OVERRIDE=2 → 4 warps — BEH-1b evidence).
    // prefill_paged keeps 3 (P6 keeps the native default there).
    // PACT_P11_LEGACY_ASSUMPTION=1 restores the constant-3 v9/v10
    // behaviour bit-for-bit (rollback switch, part of the red line).
    if (!std::getenv("PACT_MAX_PIPELINE_STAGES") &&
        !std::getenv("PACT_OVERRIDE_STAGES") &&
        !std::getenv("PACT_P11_LEGACY_ASSUMPTION")) {
      if (auto attr = mod->getAttrOfType<StringAttr>("pact.kernel_type")) {
        if (attr.getValue() == "decode_paged")
          stagesPerBlock = 2;
      }
    }
    // The canonical paged-attention tile has exactly two annotated K/V loads.
    // Requiring >=4 loads silently disabled P11 for the primary target shape.
    int64_t regsPerThread = pact::PactDecisionConstants::kUnknownRegsPerThread;
    if (auto attr = mod->getAttrOfType<IntegerAttr>("pact.hw.regs_per_thread"))
      if (attr.getInt() > 0)
        regsPerThread = attr.getInt();
    std::optional<double> measuredOcc;
    if (auto attr = mod->getAttrOfType<IntegerAttr>(
            "pact.hw.active_warp_ratio_permille"))
      measuredOcc = attr.getInt() / 1000.0;

    if (numPagedLoads >= 1 && maxTileBytes > 0) {
      // Optional pact.hw.* attrs (injected by the dynamic tree) enter the
      // same L2 equations; absent attrs keep the theory-only path.
      auto decision = pact::selectNumWarps(
          maxTileBytes, regsPerThread, measuredOcc, stagesPerBlock);
      optimalWarps = decision.numWarps;
      llvm::errs() << "[PACT P11] selectNumWarps: " << optimalWarps
                   << " (occ4=" << decision.baselineOccupancy
                   << ", occChosen=" << decision.chosenOccupancy
                   << ", requiredGain=" << decision.requiredGain
                   << ", switched=" << (decision.switched ? "yes" : "no")
                   << ")\n";
    }

    auto i32 = IntegerType::get(&getContext(), 32);
    mod->setAttr("pact.optimal_num_warps",
                 IntegerAttr::get(i32, optimalWarps));
    mod->setAttr("pact.p11.stages_assumption",
                 IntegerAttr::get(i32, stagesPerBlock));

    llvm::errs() << "[PACT P11] num_warps recommendation: " << optimalWarps
                 << " (pagedLoads=" << numPagedLoads
                 << ", maxTileB=" << maxTileBytes
                 << ", stagesAssumption=" << stagesPerBlock << ")\n";
  }
};

} // anonymous namespace
} // namespace mlir::triton
