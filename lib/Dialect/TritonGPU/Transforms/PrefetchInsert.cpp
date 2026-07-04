//===- PrefetchInsert.cpp - PACT Prefetch Insert Pass ---------------------===//
//
// PACT PrefetchInsert pass: Converts eligible paged KV loads in decode
// loops to async cp.async (SM80+).  Loads that cannot use cp.async
// (insufficient layout contiguity or per-element masks) stay synchronous.
//
// cp.async requirements:
//   1. Layout: sizePerThread in the contiguous (order[0]) dimension ×
//      element size must be exactly 4, 8, or 16 bytes.
//   2. Mask: must be absent or a splat all-ones constant.  Per-element
//      masks (arising from boundary checks in attention kernels) force
//      a sync fallback because cp.async uses one mask bit per vector group.
//
// Transformation (single-buffer, when eligible):
//   Before: tt.load(ptr, mask, other) → local_alloc(init) → local_load
//   After:  local_alloc(empty) → async_copy_global_to_local(ptr, mask, other)
//           → commit → wait → local_load(buf, token)
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

/// Walk the use-def chain from the load result to find the LocalAllocOp.
static LocalAllocOp findLocalAllocThroughLayoutConversions(
    triton::LoadOp loadOp) {
  SmallVector<Value> worklist;
  worklist.push_back(loadOp.getResult());

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

/// Check whether cp.async can be used for this load.
///
/// (1) Layout: sizePerThread[order[0]] × sizeof(elem) ∈ {4, 8, 16}.
/// (2) Mask: absent or splat-1 constant (no per-element masking).
static bool canUseCpAsync(triton::LoadOp loadOp) {
  auto resultType = dyn_cast<RankedTensorType>(loadOp.getResult().getType());
  if (!resultType)
    return false;

  auto blockedEnc = dyn_cast<BlockedEncodingAttr>(resultType.getEncoding());
  if (!blockedEnc)
    return false;

  unsigned elemBytes = resultType.getElementTypeBitWidth() / 8;
  auto sz = blockedEnc.getSizePerThread();
  auto order = blockedEnc.getOrder();
  if (sz.size() < 2 || order.size() < 2)
    return false;

  unsigned contiguousDim = order[0];
  unsigned bytesPerVec = sz[contiguousDim] * elemBytes;
  // For layout-remapped paged loads, allow contiguity override via hint.
  // The LLVM lowering will use op.getContiguity() to override the vector size.
  bool layoutRemapped = loadOp->hasAttr("pact.layout_remapped");
  if (!layoutRemapped && bytesPerVec != 4 && bytesPerVec != 8 && bytesPerVec != 16)
    return false;
  if (layoutRemapped && bytesPerVec * 2 < 4)  // need at least 2 elements to make 4 bytes via contiguity hint
    return false;

  Value mask = loadOp.getMask();
  if (mask) {
    if (auto constOp = mask.getDefiningOp<arith::ConstantOp>()) {
      if (auto denseAttr =
              dyn_cast<DenseIntElementsAttr>(constOp.getValue())) {
        if (!denseAttr.isSplat() ||
            !denseAttr.getSplatValue<APInt>().isOne())
          return false;
      } else {
        return false;
      }
    } else {
      return false;
    }
  }

  return true;
}

struct PrefetchInsertPass
    : public impl::TritonGPUPrefetchInsertBase<PrefetchInsertPass> {

  LogicalResult convertSingleBuffer(triton::LoadOp loadOp,
                                    LocalAllocOp allocOp,
                                    OpBuilder &builder) {
    Location loc = loadOp.getLoc();

    Value src = loadOp.getPtr();
    Value mask = loadOp.getMask();
    Value other = loadOp.getOther();
    auto cache = loadOp.getCache();
    auto evict = loadOp.getEvict();
    bool isVolatile = loadOp.getIsVolatile();
    uint32_t contiguity = 1;

    // For paged loads, set a contiguity hint based on the encoding.
    // Within a tile, all tokens share the same physical page →
    // addresses are contiguous (consecutive elements differ by elemSize).
    // Safe when TILE_SIZE divides PAGE_SIZE (the common case).
    if (loadOp->hasAttr("pact.paged_load")) {
      auto resTy = dyn_cast<RankedTensorType>(loadOp.getResult().getType());
      if (auto blockedEnc =
              dyn_cast<BlockedEncodingAttr>(resTy.getEncoding())) {
        auto sz = blockedEnc.getSizePerThread();
        auto order = blockedEnc.getOrder();
        if (order.size() > 0) {
          unsigned layoutContig = sz[order[0]];
          contiguity = std::max(2u, layoutContig);
        }
      }
    }

    auto oldMemDescType = cast<MemDescType>(allocOp.getResult().getType());
    auto oldEnc = oldMemDescType.getEncoding();

    // For paged loads, use a non-swizzled (trivial) shared memory encoding.
    // Swizzled layouts produce register-to-offset mappings that can't be
    // vectorized (largestVectorisation returns elemsPerVec=1 → nBytes=2).
    // A trivial layout (perPhase=1, maxPhase=1) eliminates swizzling and
    // allows cp.async vectorization.
    Attribute newSharedEnc;
    if (loadOp->hasAttr("pact.paged_load")) {
      auto ctx = oldEnc.getContext();
      auto swizzledEnc = mlir::cast<SwizzledSharedEncodingAttr>(oldEnc);
      auto order = swizzledEnc.getOrder();
      auto cgaLayout = swizzledEnc.getCGALayout();
      newSharedEnc = SwizzledSharedEncodingAttr::get(
          ctx, /*vec=*/2, /*perPhase=*/1, /*maxPhase=*/1, order, cgaLayout);
    } else {
      newSharedEnc = oldEnc;
    }

    auto mutableMemDescType = MemDescType::get(
        oldMemDescType.getShape(), oldMemDescType.getElementType(),
        newSharedEnc, oldMemDescType.getMemorySpace(),
        /*mutableMemory=*/true);

    ImplicitLocOpBuilder b(loc, builder);

    // For splat-1 masks (all-ones), skip the mask operand — cp.async
    // can be unconditional and the mask just adds overhead in lowering.
    bool isAllOnes = false;
    if (mask) {
      if (auto constOp = mask.getDefiningOp<arith::ConstantOp>()) {
        if (auto denseAttr =
                dyn_cast<DenseIntElementsAttr>(constOp.getValue())) {
          if (denseAttr.isSplat() &&
              denseAttr.getSplatValue<APInt>().isOne())
            isAllOnes = true;
        }
      }
    }
    Value asyncMask = isAllOnes ? Value() : mask;

    b.setInsertionPoint(loadOp);
    Value newBuf = LocalAllocOp::create(b, mutableMemDescType).getResult();

    Operation *copy = AsyncCopyGlobalToLocalOp::create(
        b, src, newBuf, asyncMask, other,
        cache, evict, isVolatile, contiguity);

    // Propagate pact attributes to the async copy op for lowering hints
    if (loadOp->hasAttr("pact.paged_load"))
      copy->setAttr("pact.paged_load",
                    loadOp->getAttr("pact.paged_load"));
    if (loadOp->hasAttr("pact.layout_remapped"))
      copy->setAttr("pact.layout_remapped",
                    loadOp->getAttr("pact.layout_remapped"));

    Operation *commit =
        AsyncCommitGroupOp::create(b, copy->getResult(0));

    Operation *wait =
        AsyncWaitOp::create(b, commit->getResult(0), 0);

    Value oldMemDesc = allocOp.getResult();
    SmallVector<OpOperand *> memDescUses;
    for (auto &use : oldMemDesc.getUses())
      memDescUses.push_back(&use);

    for (OpOperand *use : memDescUses) {
      Operation *user = use->getOwner();
      if (auto oldLocalLoad = dyn_cast<LocalLoadOp>(user)) {
        b.setInsertionPoint(oldLocalLoad);
        Type resultType = oldLocalLoad.getResult().getType();
        auto newLocalLoad = LocalLoadOp::create(
            b, resultType, newBuf, wait->getResult(0));
        oldLocalLoad.getResult().replaceAllUsesWith(
            newLocalLoad.getResult());
        oldLocalLoad->erase();
      } else {
        user->setOperand(use->getOperandNumber(), newBuf);
      }
    }

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
      bool asyncEligible = false;
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
        if (!allocOp)
          return WalkResult::advance();

        PagedLoadInfo info;
        info.loadOp = loadOp;
        info.allocOp = allocOp;
        info.forOp = forOp;
        info.asyncEligible = canUseCpAsync(loadOp);
        pagedLoads.push_back(info);
        return WalkResult::advance();
      });
      return WalkResult::advance();
    });

    int asyncOk = 0, syncOk = 0;
    for (auto &i : pagedLoads) {
      if (i.asyncEligible) asyncOk++; else syncOk++;
    }
    llvm::errs() << "[PACT PrefetchInsert] Found " << pagedLoads.size()
                 << " paged load(s): " << asyncOk << " async-eligible, "
                 << syncOk << " sync-fallback\n";

    int converted = 0;
    OpBuilder builder(mod.getContext());
    for (auto &info : pagedLoads) {
      if (!info.asyncEligible) {
        llvm::errs() << "[PACT PrefetchInsert] Sync fallback — "
                     << "cp.async requires layout contiguity ∈ {4,8,16} bytes "
                     << "AND an all-ones (or absent) mask\n";
        continue;
      }
      if (succeeded(convertSingleBuffer(info.loadOp, info.allocOp, builder)))
        converted++;
    }

    if (converted > 0) {
      llvm::errs() << "[PACT PrefetchInsert] Converted " << converted
                   << " paged load(s) to async copy (single-buffer)\n";
    }
  }
};

} // anonymous namespace
} // namespace gpu
} // namespace triton
} // namespace mlir
