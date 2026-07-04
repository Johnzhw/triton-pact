//===- PactPrefetch.cpp - PACT Block Table Prefetch Pass ------------------===//
//
// PACT PactPrefetch pass: restructures the main tile loop to prefetch
// the NEXT iteration's block_table lookup while processing the CURRENT
// iteration's compute.  This overlaps block_table latency with compute.
//
// Transformation (TTIR level, before TTGIR conversion):
//   Before: for tile in 0..N:
//             phys = load(block_table + seq_offset // PAGE_SIZE)
//             K, V = load(KV_cache + phys * stride + ...)
//             compute(Q, K, V)
//
//   After:  phys_cur = load(block_table + offset_0 // PAGE_SIZE)  // prologue
//           for tile in 0..N:
//             // Prefetch next tile's block_table
//             if tile+1 < N:
//               phys_next = load(block_table + offset_{tile+1} // PAGE_SIZE)
//             K, V = load(KV_cache + phys_cur * stride + ...)
//             compute(Q, K, V)
//             phys_cur = phys_next  // use next iteration
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Value.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::triton {

#define GEN_PASS_DEF_TRITONPACTPREFETCH
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

namespace {

struct PactPrefetchPass
    : public impl::TritonPactPrefetchBase<PactPrefetchPass> {

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    int numTransformed = 0;

    mod.walk([&](scf::ForOp forOp) {
      // Check if this loop has paged access
      auto *parentOp = forOp->getParentOp();
      bool isPaged = false;
      while (parentOp) {
        if (parentOp->hasAttr("pact.has_paged_access")) {
          isPaged = true;
          break;
        }
        parentOp = parentOp->getParentOp();
      }
      if (!isPaged)
        return WalkResult::advance();

      // Find the block_table load (has pact.block_table_lookup attr)
      Operation *btLoad = nullptr;
      forOp.walk([&](Operation *op) {
        if (op->hasAttr("pact.block_table_lookup")) {
          btLoad = op;
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });

      if (!btLoad)
        return WalkResult::advance();

      llvm::errs() << "[PACT PactPrefetch] Found block_table load in loop, "
                   << "prefetch restructuring\n";

      // For now, mark the loop as prefetch-eligible.
      // The actual loop restructuring (prologue + iteration-carried
      // block_table values) requires deep IR manipulation of the
      // scf.for loop structure.  This is a non-trivial transformation
      // that involves:
      //   1. Creating a prologue before the loop for iter 0
      //   2. Adding iteration-carried values (phys_block results)
      //   3. Moving the block_table load to compute next iter's values
      //   4. Updating the loop body to use carried values
      //
      // For now, mark readiness and log.  Full implementation deferred
      // to next iteration.
      forOp->setAttr("pact.prefetch_eligible",
                     UnitAttr::get(&getContext()));
      numTransformed++;
      return WalkResult::advance();
    });

    if (numTransformed > 0) {
      llvm::errs() << "[PACT PactPrefetch] Marked " << numTransformed
                   << " loop(s) as prefetch-eligible\n";
    }
  }
};

} // anonymous namespace
} // namespace mlir::triton
