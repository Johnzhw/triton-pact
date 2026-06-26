//===- PrefetchInsert.cpp - PACT Prefetch Insert Pass ---------------------===//
//
// PACT PrefetchInsert pass: Identifies paged KV loads in decode loops and
// prepares them for async copy conversion (cp.async on SM80+).
//
// Current status: Loop/load analysis + strategy identification (working).
// Full async copy insertion requires API migration to OpTy::create pattern.
//
// Transformation plan (for future):
//   For each pact.paged_load inside scf.for:
//   1. Replace tt.load + local_alloc(init) with:
//      local_alloc(empty) → async_copy_global_to_local → commit → wait →
//      local_load
//   2. Extend to double-buffering (prefetch_distance=1):
//      - 2×MemDesc buffers per K/V load
//      - Prologue: async_copy iter 0 into buf[0]
//      - Loop: wait buf[i], compute, async_copy iter j+1 into buf[1-i]
//      - Epilogue: wait final buffer
//
// Key APIs needed (ImplicitLocOpBuilder versions):
//   LocalAllocOp::create(builder, memDescType)
//   AsyncCopyGlobalToLocalOp::create(builder, src, dst, mask, other,
//       cacheAttr, evictAttr, isVolatile, contiguity)
//   AsyncCommitGroupOp::create(builder)  // then add operand
//   AsyncWaitOp::create(builder, token, num)
//   LocalLoadOp::create(builder, retType, srcMemDesc, token)
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"

#include "llvm/ADT/SmallVector.h"
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

    mod.walk([&](scf::ForOp forOp) {
      // Only process loops inside paged functions
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

      // Count and characterize paged loads in this loop
      int loopPagedLoads = 0;
      int kLoads = 0, vLoads = 0;
      forOp.walk([&](Operation *innerOp) {
        if (innerOp->hasAttr("pact.paged_load")) {
          loopPagedLoads++;
          // Determine K vs V by shape: K is [64,16,f16], V is [16,64,f16]
          if (auto loadOp = dyn_cast<LoadOp>(innerOp)) {
            auto resultType = loadOp.getType();
            if (auto tensorType = dyn_cast<RankedTensorType>(resultType)) {
              auto shape = tensorType.getShape();
              if (shape.size() == 2) {
                if (shape[0] > shape[1])
                  kLoads++;
                else
                  vLoads++;
              }
            }
          }
        }
      });

      if (loopPagedLoads > 0) {
        numPagedLoops++;
        numPagedLoads += loopPagedLoads;

        // Report loop characteristics
        auto lb = forOp.getLowerBound();
        auto ub = forOp.getUpperBound();
        int numIterArgs = forOp.getInitArgs().size();
        llvm::errs() << "[PACT PrefetchInsert] Loop with " << loopPagedLoads
                     << " paged loads (" << kLoads << " K, " << vLoads
                     << " V), " << numIterArgs << " iter_args\n";
      }

      return WalkResult::advance();
    });

    if (numPagedLoops > 0) {
      llvm::errs() << "[PACT PrefetchInsert] Found " << numPagedLoops
                   << " decode loop(s) with " << numPagedLoads
                   << " paged KV load(s)\n";
      llvm::errs() << "[PACT PrefetchInsert] Prefetch strategy: "
                   << "double-buffered cp.async (prefetch_distance=1)\n";
      llvm::errs() << "[PACT PrefetchInsert] NOTE: Async copy insertion "
                   << "requires MLIR ::create API migration.\n";
      llvm::errs() << "[PACT PrefetchInsert] Target: "
                   << "local_alloc(empty) → async_copy → commit → wait → "
                   << "local_load per paged load.\n";
    }
  }
};

} // anonymous namespace
} // namespace gpu
} // namespace triton
} // namespace mlir
