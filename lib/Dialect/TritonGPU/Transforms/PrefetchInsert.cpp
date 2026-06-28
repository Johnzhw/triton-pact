//===- PrefetchInsert.cpp - PACT Prefetch Insert Pass ---------------------===//
//
// PACT PrefetchInsert pass: Converts paged KV loads in decode loops to
// async cp.async operations (SM80+). Supports single-buffer (v1).
//
// Transformation (single-buffer):
//   For each pact.paged_load inside scf.for:
//     Before: tt.load → ttg.local_alloc(init) → ttg.local_load(memdesc)
//     After:  ttg.local_alloc(empty) → async_copy_global_to_local
//             → async_commit_group → async_wait → ttg.local_load(buf, token)
//
// Current status:
//   - Async copy insertion: WORKING (verified in TTGIR dump)
//   - Blocked by cp.async alignment: sizePerThread=[1,1] = 2 bytes < 4 bytes
//     minimum. Needs either layout adjustment (4+ bytes/thread) or
//     contiguity analysis to batch elements.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace mlir {
namespace triton {
namespace gpu {

#define GEN_PASS_DEF_TRITONGPUPREFETCHINSERT
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

namespace {

/// Return true if the value is a constant zero tensor.
static bool isZeroConst(Value val) {
  if (!val)
    return false;
  if (auto constOp = val.getDefiningOp<arith::ConstantOp>()) {
    if (auto denseAttr = dyn_cast<DenseFPElementsAttr>(constOp.getValue())) {
      return denseAttr.isSplat() && denseAttr.getSplatValue<APFloat>().isZero();
    }
    if (auto intAttr = dyn_cast<DenseIntElementsAttr>(constOp.getValue())) {
      return intAttr.isSplat() && intAttr.getSplatValue<APInt>().isZero();
    }
  }
  return false;
}

/// Walk the use-def chain from the load result to find the LocalAllocOp.
static LocalAllocOp findLocalAllocThroughLayoutConversions(
    triton::LoadOp loadOp) {
  SmallVector<Value> worklist;
  worklist.push_back(loadOp.getResult());

  // BFS through convert_layout ops (limit depth to avoid infinite loops)
  for (int depth = 0; depth < 8 && !worklist.empty(); ++depth) {
    SmallVector<Value> next;
    for (Value val : worklist) {
      for (auto *user : val.getUsers()) {
        if (auto alloc = dyn_cast<LocalAllocOp>(user))
          return alloc;
        if (auto cvt = dyn_cast<ConvertLayoutOp>(user))
          next.push_back(cvt.getResult());
      }
    }
    worklist = std::move(next);
  }
  return nullptr;
}

struct PrefetchInsertPass
    : public impl::TritonGPUPrefetchInsertBase<PrefetchInsertPass> {

  /// Convert a synchronous paged load → local_alloc pattern into async.
  LogicalResult convertSingleBuffer(triton::LoadOp loadOp,
                                    LocalAllocOp allocOp,
                                    OpBuilder &builder) {
    Location loc = loadOp.getLoc();

    // Collect load parameters
    Value src = loadOp.getPtr();
    Value mask = loadOp.getMask();
    Value other = loadOp.getOther();
    triton::CacheModifier cache = loadOp.getCache();
    triton::EvictionPolicy evict = loadOp.getEvict();
    bool isVolatile = loadOp.getIsVolatile();
    uint32_t contiguity = 1; // conservative default

    // Get the memdesc type and make it mutable for empty alloc.
    auto oldMemDescType = cast<MemDescType>(allocOp.getResult().getType());
    auto mutableMemDescType = MemDescType::get(
        oldMemDescType.getShape(), oldMemDescType.getElementType(),
        oldMemDescType.getEncoding(), oldMemDescType.getMemorySpace(),
        /*mutableMemory=*/true);

    ImplicitLocOpBuilder b(loc, builder);

    // 1. Create empty local_alloc before the load
    b.setInsertionPoint(loadOp);
    Value newBuf = LocalAllocOp::create(b, mutableMemDescType).getResult();

    // 2. Async copy: global → local (LowerLoops.cpp pattern)
    Operation *copy = AsyncCopyGlobalToLocalOp::create(
        b, /*src=*/src, /*result=*/newBuf,
        /*mask=*/mask, /*other=*/other,
        /*cache=*/cache, /*evict=*/evict,
        /*isVolatile=*/isVolatile, /*contiguity=*/contiguity);

    // 3. Commit
    Operation *commit =
        AsyncCommitGroupOp::create(b, copy->getResult(0));

    // 4. Wait
    Operation *wait =
        AsyncWaitOp::create(b, commit->getResult(0), /*num=*/0);

    // Collect all uses of the old memdesc
    Value oldMemDesc = allocOp.getResult();
    SmallVector<OpOperand *> memDescUses;
    for (auto &use : oldMemDesc.getUses())
      memDescUses.push_back(&use);

    // 5. Replace old local_load uses with new ones using the async token
    for (OpOperand *use : memDescUses) {
      Operation *user = use->getOwner();

      if (auto oldLocalLoad = dyn_cast<LocalLoadOp>(user)) {
        b.setInsertionPoint(oldLocalLoad);
        Type resultType = oldLocalLoad.getResult().getType();

        if (!loadOp.getOther() || isZeroConst(loadOp.getOther())) {
          auto newLocalLoad = LocalLoadOp::create(
              b, resultType, newBuf, wait->getResult(0));
          oldLocalLoad.getResult().replaceAllUsesWith(
              newLocalLoad.getResult());
        } else {
          auto sharedLoad = LocalLoadOp::create(
              b, resultType, newBuf, wait->getResult(0));
          auto select = arith::SelectOp::create(
              b, resultType,
              loadOp.getMask(), sharedLoad.getResult(), other);
          oldLocalLoad.getResult().replaceAllUsesWith(
              select->getResult(0));
        }
        oldLocalLoad->erase();
      } else {
        user->setOperand(use->getOperandNumber(), newBuf);
      }
    }

    // Erase the old local_alloc and load
    allocOp.erase();
    loadOp.erase();

    return success();
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();

    struct PagedLoadInfo {
      triton::LoadOp loadOp;
      LocalAllocOp allocOp;
      scf::ForOp forOp;
    };
    SmallVector<PagedLoadInfo> pagedLoads;

    mod.walk([&](scf::ForOp forOp) {
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

      forOp.walk([&](triton::LoadOp loadOp) {
        if (!loadOp->hasAttr("pact.paged_load"))
          return WalkResult::advance();

        LocalAllocOp allocOp =
            findLocalAllocThroughLayoutConversions(loadOp);
        if (allocOp)
          pagedLoads.push_back({loadOp, allocOp, forOp});

        return WalkResult::advance();
      });

      return WalkResult::advance();
    });

    llvm::errs() << "[PACT PrefetchInsert] Found " << pagedLoads.size()
                 << " paged load(s) in paged loop(s)\n";

    int converted = 0;
    OpBuilder builder(mod.getContext());

    for (auto &info : pagedLoads) {
      if (succeeded(
              convertSingleBuffer(info.loadOp, info.allocOp, builder)))
        converted++;
    }

    if (converted > 0) {
      llvm::errs() << "[PACT PrefetchInsert] Converted " << converted
                   << " paged load(s) to async copy (single-buffer)\n";
      llvm::errs() << "[PACT PrefetchInsert] NOTE: cp.async alignment "
                   << "constraint may block LLVM lowering if "
                   << "bytes/thread < 4.\n";
    }
  }
};

} // anonymous namespace
} // namespace gpu
} // namespace triton
} // namespace mlir
