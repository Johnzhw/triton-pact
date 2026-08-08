//===- DotPromotion.cpp - PACT P12 TTIR: Mark sum(mul) for MMA promotion --===//
//
// P12 (TTIR Phase): Marks tt.reduce ops that match sum(mul(q,k)) pattern
//                   with pact.dot_promotable = true for downstream TTGIR
//                   consumer to perform the actual tt.dot promotion.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"

#include "llvm/Support/raw_ostream.h"
#include <cstdlib>
#include <string>

namespace mlir::triton {

#define GEN_PASS_DEF_PACTDOTPROMOTION
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

namespace {

static bool isEnabled() {
  const char *env = std::getenv("PACT_ENABLE_DOT_PROMOTION");
  return !env || std::string(env) != "0";
}

// Trace through broadcast(expand_dims(x)) → x (1D source)
static Value get1DSource(Value val) {
  if (auto broadcastOp = val.getDefiningOp<BroadcastOp>())
    val = broadcastOp.getSrc();
  if (auto expandOp = val.getDefiningOp<ExpandDimsOp>()) {
    if (expandOp.getAxis() == 0) {
      auto src = expandOp.getSrc();
      if (isa<RankedTensorType>(src.getType()) &&
          cast<RankedTensorType>(src.getType()).getShape().size() == 1)
        return src;
    }
  }
  auto ty = dyn_cast<RankedTensorType>(val.getType());
  if (ty && ty.getShape().size() == 1)
    return val;
  return nullptr;
}

struct PACTDotPromotionPass
    : public impl::PACTDotPromotionBase<PACTDotPromotionPass> {

  void runOnOperation() override {
    if (!isEnabled())
      return;

    ModuleOp mod = getOperation();
    int marked = 0;

    mod.walk([&](ReduceOp reduceOp) {
      // Must be sum reduction on axis=1
      if (reduceOp.getAxis() != 1)
        return WalkResult::advance();

      Value reduceInput = reduceOp.getOperand(0);
      auto inputTy = dyn_cast<RankedTensorType>(reduceInput.getType());
      if (!inputTy || inputTy.getShape().size() != 2)
        return WalkResult::advance();

      // Find mul
      auto mulOp = reduceInput.getDefiningOp<arith::MulFOp>();
      if (!mulOp)
        return WalkResult::advance();

      Value lhs = mulOp.getLhs();
      Value rhs = mulOp.getRhs();
      auto lhsTy = dyn_cast<RankedTensorType>(lhs.getType());
      auto rhsTy = dyn_cast<RankedTensorType>(rhs.getType());
      if (!lhsTy || !rhsTy || lhsTy.getShape().size() != 2 || rhsTy.getShape().size() != 2)
        return WalkResult::advance();

      int64_t D1 = lhsTy.getShape()[1];
      int64_t D2 = rhsTy.getShape()[1];
      if (D1 != D2)
        return WalkResult::advance();

      int64_t TILE = lhsTy.getShape()[0];
      int64_t D = D1;

      // K dimension must be MMA-friendly
      if (D != 16 && D != 32 && D != 64 && D != 128)
        return WalkResult::advance();

      // Element type check: f32 or f16
      auto elemTy = lhsTy.getElementType();
      if (!elemTy.isF32() && !elemTy.isF16() && !elemTy.isBF16())
        return WalkResult::advance();

      // Find 1D Q source
      Value qSrc = get1DSource(lhs);
      Value k2D = nullptr;
      if (qSrc) {
        k2D = rhs;
      } else {
        qSrc = get1DSource(rhs);
        if (qSrc) k2D = lhs;
      }
      if (!k2D || !qSrc)
        return WalkResult::advance();

      auto qSrcTy = dyn_cast<RankedTensorType>(qSrc.getType());
      if (!qSrcTy || qSrcTy.getShape().size() != 1 || qSrcTy.getShape()[0] != D)
        return WalkResult::advance();

      // Mark the reduce op for TTGIR-level promotion
      reduceOp->setAttr("pact.dot_promotable",
                        BoolAttr::get(&getContext(), true));

      llvm::errs() << "[PACT P12] DotPromotion: marked reduce["
                   << TILE << "," << D << "] for MMA promotion\n";
      marked++;
      return WalkResult::advance();
    });

    if (marked > 0)
      llvm::errs() << "[PACT P12] DotPromotion: marked " << marked
                   << " reduce(s)\n";
  }
};

} // anonymous namespace
} // namespace mlir::triton
