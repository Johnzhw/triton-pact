//===- PatternSpecialize.cpp - PACT Pattern Specialize Pass ---------------===//
//
// PACT PatternSpecialize pass: detects MQA/GQA patterns from IR structure
// via def-use chain analysis (get_program_id(y) → constant multiplier →
// Q head index computation).  Falls back to pact.mqa/pact.gqa function
// attributes if def-use analysis finds nothing.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::triton {

#define GEN_PASS_DEF_TRITONPATTERNSPECIALIZE
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Helper: extract a constant integer value from an MLIR attribute, supporting
// both scalar IntegerAttr and splat DenseElementsAttr.
//===----------------------------------------------------------------------===//
static std::optional<int64_t> extractConstantInt(Attribute attr) {
  if (auto intAttr = dyn_cast<IntegerAttr>(attr))
    return intAttr.getInt();
  if (auto denseAttr = dyn_cast<DenseIntElementsAttr>(attr)) {
    if (denseAttr.isSplat())
      return denseAttr.getSplatValue<APInt>().getSExtValue();
  }
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// Helper: check if value V is transitively used by any op in targetOps set.
// maxDepth limits recursion.
//===----------------------------------------------------------------------===//
static bool usedByAny(Value val, const DenseSet<StringRef> &targetOps,
                       int maxDepth = 12) {
  if (maxDepth <= 0)
    return false;

  for (auto *user : val.getUsers()) {
    if (targetOps.contains(user->getName().getStringRef()))
      return true;

    for (auto result : user->getResults()) {
      if (result == val)
        continue;
      if (usedByAny(result, targetOps, maxDepth - 1))
        return true;
    }
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Helper: walk through splat, broadcast, expand_dims, extsi/index_cast to
// find the "semantic" source of a value (e.g., the scalar before splat).
//===----------------------------------------------------------------------===//
static Value skipReshapeOps(Value val, int maxDepth = 6) {
  if (maxDepth <= 0)
    return val;

  auto *defOp = val.getDefiningOp();
  if (!defOp)
    return val;

  auto name = defOp->getName().getStringRef();
  if (name == "tt.splat" || name == "tt.broadcast" ||
      name == "tt.expand_dims" || name == "arith.extsi" ||
      name == "arith.extui" || name == "arith.index_cast" ||
      name == "arith.sitofp") {
    return skipReshapeOps(defOp->getOperand(0), maxDepth - 1);
  }

  return val;
}

//===----------------------------------------------------------------------===//
// Detect MQA/GQA by analyzing get_program_id(y) def-use chain.
//
// Pattern:
//   %kv_head = tt.get_program_id y
//   %base = arith.muli %kv_head, %const_K
//   %offset = arith.remsi %offs_m_range, %const_K2
//   %q_head = arith.addi %base_reshaped, %offset
//
// If const_K == const_K2, then const_K = GQA group size.
// const_K == 1 → MQA/MHA (no KV head sharing beyond 1:1)
// const_K > 1  → GQA with group_size = const_K
//===----------------------------------------------------------------------===//
static int64_t detectGQAFromProgramId(Operation *funcOp,
                                       MLIRContext *ctx) {
  // Step 1: find all get_program_id y ops
  SmallVector<Operation *> progIdYOps;
  funcOp->walk([&](Operation *op) {
    if (op->getName().getStringRef() == "tt.get_program_id") {
      // axis can be IntegerAttr (0=x, 1=y, 2=z) or StringAttr ("x"/"y"/"z")
      bool isY = false;
      if (auto axisStr = op->getAttrOfType<StringAttr>("axis")) {
        isY = (axisStr.getValue() == "y");
      } else if (auto axisInt = op->getAttrOfType<IntegerAttr>("axis")) {
        isY = (axisInt.getInt() == 1);
      }
      if (isY)
        progIdYOps.push_back(op);
    }
  });

  if (progIdYOps.empty())
    return 0;

  // Step 2: for each get_program_id y, find arith.muli(kv_head, const)
  for (auto *pidOp : progIdYOps) {
    Value kvHead = pidOp->getResult(0);

    for (auto *user : kvHead.getUsers()) {
      if (user->getName().getStringRef() != "arith.muli")
        continue;

      // Check if the other operand (non-kvHead) is a constant
      Value other = (user->getOperand(0) == kvHead)
                    ? user->getOperand(1) : user->getOperand(0);
      auto constOp = other.getDefiningOp<arith::ConstantOp>();
      if (!constOp)
        continue;

      auto constVal = extractConstantInt(constOp.getValue());
      if (!constVal || *constVal <= 0)
        continue;

      int64_t groupSize = *constVal;

      // Step 3: verify muli result is used in Q head index computation.
      // Look for: addi(muli_result_reshaped, remsi(offs_m_range, groupSize))
      Value muliResult = user->getResult(0);

      // The muli result should be reshaped (splat/broadcast/expand_dims)
      // and then used in an arith.addi
      bool foundAddi = false;
      for (auto *muliUser : muliResult.getUsers()) {
        Value reshaped = muliResult;
        // Allow reshape ops between muli and addi
        if (muliUser->getName().getStringRef() == "tt.splat" ||
            muliUser->getName().getStringRef() == "tt.broadcast" ||
            muliUser->getName().getStringRef() == "tt.expand_dims" ||
            muliUser->getName().getStringRef() == "arith.extsi") {
          reshaped = muliUser->getResult(0);
        }

        for (auto *reshapedUser : reshaped.getUsers()) {
          if (reshapedUser->getName().getStringRef() != "arith.addi")
            continue;

          // Check if the OTHER addi operand is arith.remsi by groupSize
          Value addiOther = (reshapedUser->getOperand(0) == reshaped)
                            ? reshapedUser->getOperand(1)
                            : reshapedUser->getOperand(0);

          // Walk through reshape ops on the other side
          Value addiSrc = skipReshapeOps(addiOther);

          auto *remOp = addiSrc.getDefiningOp();
          if (!remOp)
            continue;
          if (remOp->getName().getStringRef() != "arith.remsi")
            continue;

          // Check if remsi divisor == groupSize
          auto remConstOp =
              remOp->getOperand(1).getDefiningOp<arith::ConstantOp>();
          if (!remConstOp)
            continue;
          auto remVal = extractConstantInt(remConstOp.getValue());
          if (!remVal || *remVal != groupSize)
            continue;

          foundAddi = true;
          break;
        }
        if (foundAddi)
          break;
      }

      if (foundAddi) {
        return groupSize;
      }
    }
  }

  return 0; // not found via def-use
}

//===----------------------------------------------------------------------===//
// PatternSpecialize Pass
//===----------------------------------------------------------------------===//
struct PatternSpecializePass
    : public impl::TritonPatternSpecializeBase<PatternSpecializePass> {

  void runOnOperation() override {
    ModuleOp mod = getOperation();

    int numPagedOps = 0;
    int numDetected = 0;
    mod.walk([&](Operation *op) {
      // Only process paged functions
      if (!op->hasAttr("pact.paged"))
        return WalkResult::advance();
      numPagedOps++;

      // Try def-use chain detection first
      int64_t groupSize = detectGQAFromProgramId(op, &getContext());

      // Fall back to attribute-based detection
      if (groupSize == 0) {
        bool isMQA = op->hasAttr("pact.mqa");
        bool isGQA = op->hasAttr("pact.gqa");
        if (!isMQA && !isGQA)
          return WalkResult::advance();

        if (isGQA) {
          auto gqaAttr = op->getAttrOfType<IntegerAttr>("pact.gqa_group_size");
          groupSize = gqaAttr ? gqaAttr.getInt() : 0;
        } else if (isMQA) {
          groupSize = 1;
        }
      }

      if (groupSize <= 0)
        return WalkResult::advance();

      llvm::errs() << "[PACT PatternSpecialize] ";

      if (groupSize == 1) {
        // MQA: num_kv_heads == 1, all Q heads share one KV head
        op->setAttr("pact.mqa_specialized", UnitAttr::get(&getContext()));
        numDetected++;
        llvm::errs() << "MQA specialized (kv_head=0 for all Q heads)\n";
      } else {
        // GQA: num_query_heads / num_kv_heads = groupSize
        op->setAttr("pact.gqa_specialized", UnitAttr::get(&getContext()));
        op->setAttr("pact.gqa_group_size",
                    IntegerAttr::get(IntegerType::get(&getContext(), 64),
                                     groupSize));
        op->setAttr("pact.kv_head_mapping",
                    StringAttr::get(&getContext(), "q_head / group_size"));
        numDetected++;
        llvm::errs() << "GQA specialized (group_size=" << groupSize
                     << ", kv_head = q_head / " << groupSize << ")\n";
      }

      return WalkResult::advance();
    });

    if (numPagedOps > 0) {
      llvm::errs() << "[PACT PatternSpecialize] Processed " << numPagedOps
                   << " paged op(s), " << numDetected << " GQA/MQA detected\n";
    }
  }
};

} // anonymous namespace
} // namespace mlir::triton
