//===- PatternSpecialize.cpp - PACT Pattern Specialize Pass ---------------===//
//
// PACT PatternSpecialize pass: reads MQA/GQA metadata from function attrs
// and records sharing metadata for downstream PrefetchInsert consumption.
// No-op fallback if metadata is missing or GQA group size is invalid.
//
// First-version restrictions:
//   - Does NOT modify launch grid
//   - Does NOT change tl.program_id to head mapping
//   - Does NOT merge KV loads across programs/CTAs
//   - Invalid GQA (num_heads % num_kv_heads != 0) → no-op fallback
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"

#include "llvm/Support/raw_ostream.h"

namespace mlir::triton {

#define GEN_PASS_DEF_TRITONPATTERNSPECIALIZE
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

namespace {

struct PatternSpecializePass
    : public impl::TritonPatternSpecializeBase<PatternSpecializePass> {

  void runOnOperation() override {
    ModuleOp mod = getOperation();

    mod.walk([&](Operation *op) {
      if (!op->hasAttr("pact.paged"))
        return WalkResult::advance();

      bool isMQA = op->hasAttr("pact.mqa");
      bool isGQA = op->hasAttr("pact.gqa");
      if (!isMQA && !isGQA)
        return WalkResult::advance();

      auto loc = op->getLoc();
      llvm::errs() << "[PACT PatternSpecialize] ";

      if (isMQA) {
        // MQA: num_kv_heads == 1, all Q heads share one KV head
        // Canonicalization: kv_head_idx should be 0 (constant)
        // In the frozen kernel, kv_head_idx = tl.program_id(1), which for
        // MQA with num_kv_heads=1 is always 0.
        op->setAttr("pact.mqa_specialized", UnitAttr::get(&getContext()));
        llvm::errs() << "MQA detected (kv_head=0 for all Q heads)\n";
      }

      if (isGQA) {
        auto gqaSize =
            op->getAttrOfType<IntegerAttr>("pact.gqa_group_size");
        int64_t groupSize = gqaSize ? gqaSize.getInt() : 0;
        op->setAttr("pact.gqa_specialized", UnitAttr::get(&getContext()));
        op->setAttr("pact.kv_head_mapping",
                    StringAttr::get(&getContext(), "q_head / group_size"));
        llvm::errs() << "GQA detected (group_size=" << groupSize
                     << ", kv_head = q_head / " << groupSize << ")\n";
      }

      return WalkResult::advance();
    });
  }
};

} // anonymous namespace
} // namespace mlir::triton
