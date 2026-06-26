//===- PrefetchInsert.cpp - PACT Prefetch Insert Pass ---------------------===//
//
// PACT PrefetchInsert pass: converts annotated paged KV cache loads into
// async copy operations with double-buffering at the TTGIR level.
//
// First version: analysis + logging only. Full async copy insertion
// requires careful integration with TritonGPU's shared memory allocation,
// async_copy_global_to_local lowering, and hardware-specific alignment
// constraints (SM80+ cp.async, SM90+ TMA).
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"

#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace triton {
namespace gpu {

#define GEN_PASS_DEF_TRITONGPUPREFETCHINSERT
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

namespace {

struct PrefetchInsertPass
    : public impl::TritonGPUPrefetchInsertBase<PrefetchInsertPass> {

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    int numPagedLoops = 0;
    int numPagedLoads = 0;

    mod.walk([&](Operation *op) {
      // Check if any function has pact.has_paged_access
      if (!op->hasAttr("pact.has_paged_access"))
        return WalkResult::advance();

      // Walk scf.for loops inside paged functions
      op->walk([&](scf::ForOp forOp) {
        // Count paged loads in this loop
        int loopPagedLoads = 0;
        forOp.walk([&](Operation *innerOp) {
          if (innerOp->hasAttr("pact.paged_load"))
            loopPagedLoads++;
        });

        if (loopPagedLoads > 0) {
          numPagedLoops++;
          numPagedLoads += loopPagedLoads;
        }
      });

      return WalkResult::advance();
    });

    if (numPagedLoops > 0) {
      llvm::errs() << "[PACT PrefetchInsert] Found " << numPagedLoops
                   << " decode loop(s) with " << numPagedLoads
                   << " paged KV load(s)\n";
      llvm::errs() << "[PACT PrefetchInsert] Prefetch strategy: "
                   << "double-buffered cp.async (prefetch_distance=1)\n";
      llvm::errs() << "[PACT PrefetchInsert] NOTE: Full async copy insertion "
                   << "deferred to future work.\n";
    }
  }
};

} // anonymous namespace
} // namespace gpu
} // namespace triton
} // namespace mlir
