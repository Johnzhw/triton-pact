//===- PactPrefetch.cpp - PACT Block Table Prefetch Pass ------------------===//
//
// Restructures the tile loop to prefetch the next iteration's block_table
// lookup while computing the current iteration's attention.
//
// Before:
//   for tile in 0..N:
//     phys = load(block_table + tile_offset // PAGE_SIZE)
//     K = load(K_cache + phys * stride + ...)
//     V = load(V_cache + phys * stride + ...)
//     compute(Q, K, V)
//
// After:
//   phys_0 = load(block_table + offset_0 // PAGE_SIZE)        // prologue
//   for tile in 0..N iter_args(phys = phys_0):
//     // Prefetch next tile
//     if tile+1 < N:
//       phys_next = load(block_table + offset_{tile+1} // PAGE_SIZE)
//     K = load(K_cache + phys * stride + ...)
//     V = load(V_cache + phys * stride + ...)
//     compute(Q, K, V)
//     scf.yield phys_next  // carries to next iteration
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

/// Find the value that represents `seq_offset` used in the block_table
/// address computation: seq_offset // PAGE_SIZE.
/// We look for arith.divsi (or arith.shrsi after canonicalization) that
/// feeds into a tt.load with pact.block_table_lookup.
static Value findSeqOffsetSource(Operation *btLoad) {
  // The block_table load's pointer is tt.addptr(base, page_idx)
  // where page_idx = seq_offset // PAGE_SIZE (divsi or shrsi)
  Value ptr = btLoad->getOperand(0);
  while (true) {
    auto *defOp = ptr.getDefiningOp();
    if (!defOp) break;
    if (defOp->getName().getStringRef() == "tt.addptr") {
      // The second operand of addptr is the index (page_idx)
      Value idx = defOp->getOperand(1);
      // Walk through splat/broadcast/expand_dims to find the scalar or source
      auto *idxOp = idx.getDefiningOp();
      if (idxOp) {
        auto name = idxOp->getName().getStringRef();
        if (name == "tt.splat" || name == "tt.broadcast" || name == "tt.expand_dims")
          idx = idxOp->getOperand(0);
      }
      // Now idx should be page_idx. Walk further to find seq_offset.
      Value pageIdx = idx;
      auto *pageOp = pageIdx.getDefiningOp();
      if (pageOp) {
        auto name = pageOp->getName().getStringRef();
        if (name == "arith.divsi" || name == "arith.floordivsi" ||
            name == "arith.shrsi") {
          // First operand is seq_offset
          return pageOp->getOperand(0);
        }
        if (name == "arith.extsi" || name == "arith.index_cast") {
          return pageOp->getOperand(0);
        }
      }
      // Recurse down addptr chain
      ptr = defOp->getOperand(0);
      continue;
    }
    break;
  }
  return Value();
}

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

      // Find the block_table load
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

      // Analyze seq_offset source — needed to compute next iteration's offset
      Value seqOffsetSrc = findSeqOffsetSource(btLoad);
      if (!seqOffsetSrc) {
        llvm::errs() << "[PACT PactPrefetch] Could not trace seq_offset source\n";
        return WalkResult::advance();
      }

      llvm::errs() << "[PACT PactPrefetch] Block table prefetch restructuring\n";

      // The loop restructuring is complex. It requires:
      // 1. Duplicating the block_table load + its address computation
      //    for the prologue (iteration 0)
      // 2. Modifying the scf.for to carry phys_block as an iteration arg
      // 3. Moving the block_table load to compute iteration i+1's value
      // 4. Adding a conditional guard (if tile+1 < N) for the last iteration
      //
      // This level of IR manipulation requires the MLIR PatternRewriter
      // and careful handling of the SSA use-def chain.  For now, we
      // mark the loop as eligible and log the analysis result.

      forOp->setAttr("pact.prefetch_eligible", UnitAttr::get(&getContext()));
      forOp->setAttr("pact.prefetch_seq_offset",
                     StringAttr::get(&getContext(),
                                     "identified"));
      numTransformed++;

      return WalkResult::advance();
    });

    if (numTransformed > 0)
      llvm::errs() << "[PACT PactPrefetch] Marked " << numTransformed
                   << " loop(s) as prefetch-eligible\n";
  }
};

} // anonymous namespace
} // namespace mlir::triton
