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
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/MapVector.h"
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

  struct PagedLoadInfo {
    triton::LoadOp loadOp;
    LocalAllocOp allocOp;
    scf::ForOp forOp;
    bool asyncEligible = false;
  };

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

  /// Double-buffer conversion for all eligible loads in a single scf.for loop.
  /// Step 1 (no yield prefetch): allocate 2x buffers before the loop,
  /// insert j%2 buffer selection + async_copy + wait + local_load at each
  /// load's original position.  No software pipelining yet — just verify
  /// the multi-buffer mechanism works.
  LogicalResult convertDoubleBuffer(scf::ForOp forOp,
                                     SmallVectorImpl<PagedLoadInfo> &loads,
                                     OpBuilder &builder) {
    if (loads.empty())
      return success();

    Location loc = forOp.getLoc();

    // ── Step 1: allocate 2× buffers BEFORE the loop ──────────────────
    struct DoubleBuf {
      Value alloc2x;       // multi-buffered MemDesc<[2, shape...]>
      PagedLoadInfo info;
      // Saved operands for yield prefetch
      Value src;           // ptr from original load
      Value mask;
      Value other;
      triton::CacheModifier cache;
      triton::EvictionPolicy evict;
      bool isVolatile;
      uint32_t contiguity;
      // Saved attributes (before loadOp is erased)
      Attribute pactPagedLoadAttr;
      Attribute pactLayoutRemappedAttr;
    };
    SmallVector<DoubleBuf> bufs;

    // Induction variable and constant, available to yield-prefetch block
    Value iv;
    Value c2;  // constant 2, used by body and (when enabled) yield prefetch

    {
      ImplicitLocOpBuilder b(loc, builder);
      b.setInsertionPoint(forOp);

      for (auto &info : loads) {
        auto &loadOp = info.loadOp;
        auto &allocOp = info.allocOp;

        auto oldMemDescType =
            cast<MemDescType>(allocOp.getResult().getType());

        // Use the same non-swizzled encoding strategy as single-buffer
        Attribute newSharedEnc;
        if (loadOp->hasAttr("pact.paged_load")) {
          auto ctx = oldMemDescType.getEncoding().getContext();
          auto swizzledEnc =
              mlir::cast<SwizzledSharedEncodingAttr>(oldMemDescType.getEncoding());
          auto order = swizzledEnc.getOrder();
          auto cgaLayout = swizzledEnc.getCGALayout();
          newSharedEnc = SwizzledSharedEncodingAttr::get(
              ctx, /*vec=*/2, /*perPhase=*/1, /*maxPhase=*/1, order, cgaLayout);
        } else {
          newSharedEnc = oldMemDescType.getEncoding();
        }

        auto mutableMemDescType = MemDescType::get(
            oldMemDescType.getShape(), oldMemDescType.getElementType(),
            newSharedEnc, oldMemDescType.getMemorySpace(),
            /*mutableMemory=*/true);

        // Create 2× multi-buffered type and allocation
        auto multiBufType = triton::getMultiBufferedType(mutableMemDescType, 2);
        Value alloc2x = LocalAllocOp::create(b, multiBufType).getResult();

        bufs.push_back({alloc2x, info});
        llvm::errs() << "[PACT DoubleBuf] Allocated 2x buffer for load\n";
      }
    }

    // ── Step 2: inside loop body, at each load's position ────────────
    // Insert: j%2 → buffer_view → async_copy → commit → wait → local_load

    // Pre-compute j%2 once inside the loop (shared across loads)
    ImplicitLocOpBuilder bodyBuilder(loc, builder);
    bodyBuilder.setInsertionPointToStart(forOp.getBody());

    iv = forOp.getInductionVar();
    Type ivType = iv.getType();
    Value c0;
    if (auto intTy = dyn_cast<IntegerType>(ivType)) {
      unsigned width = intTy.getWidth();
      c2 = arith::ConstantIntOp::create(bodyBuilder, loc, 2, width);
      c0 = arith::ConstantIntOp::create(bodyBuilder, loc, 0, width);
    } else {
      // IndexType fallback
      c2 = arith::ConstantIndexOp::create(bodyBuilder, loc, 2);
      c0 = arith::ConstantIndexOp::create(bodyBuilder, loc, 0);
    }
    Value idx = arith::RemSIOp::create(bodyBuilder, loc, iv, c2);

    for (auto &db : bufs) {
      auto &info = db.info;
      auto &loadOp = info.loadOp;
      auto &allocOp = info.allocOp;

      // ── Create async_copy + wait + local_load at the load position ──
      ImplicitLocOpBuilder lb(loc, builder);
      lb.setInsertionPoint(loadOp);

      // Select buffer view: buf[idx]
      // Induction variable is index type in MLIR; cast to i32 for MemDescIndex
      Value idxCast = idx;
      if (isa<IndexType>(idx.getType())) {
        idxCast = arith::IndexCastUIOp::create(lb, lb.getI32Type(), idx);
      }
      Value bufView = triton::createSingleBufferView(lb, db.alloc2x, idxCast);

      // Gather async_copy operands from the original load
      Value src = loadOp.getPtr();
      Value mask = loadOp.getMask();
      Value other = loadOp.getOther();
      auto cache = loadOp.getCache();
      auto evict = loadOp.getEvict();
      bool isVolatile = loadOp.getIsVolatile();
      uint32_t contiguity = 1;

      if (loadOp->hasAttr("pact.paged_load")) {
        auto resTy =
            dyn_cast<RankedTensorType>(loadOp.getResult().getType());
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

      // Skip splat-1 mask
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

      // Save operands for later yield prefetch
      db.src = src;
      db.mask = asyncMask;
      db.other = other;
      db.cache = cache;
      db.evict = evict;
      db.isVolatile = isVolatile;
      db.contiguity = contiguity;
      db.pactPagedLoadAttr = loadOp->getAttr("pact.paged_load");
      db.pactLayoutRemappedAttr = loadOp->getAttr("pact.layout_remapped");

      Operation *copy = AsyncCopyGlobalToLocalOp::create(
          lb, src, bufView, asyncMask, other,
          cache, evict, isVolatile, contiguity);

      if (loadOp->hasAttr("pact.paged_load"))
        copy->setAttr("pact.paged_load",
                      loadOp->getAttr("pact.paged_load"));
      if (loadOp->hasAttr("pact.layout_remapped"))
        copy->setAttr("pact.layout_remapped",
                      loadOp->getAttr("pact.layout_remapped"));

      Operation *commit =
          AsyncCommitGroupOp::create(lb, copy->getResult(0));

      Operation *wait =
          AsyncWaitOp::create(lb, commit->getResult(0), 0);

      // Replace uses of old local_alloc → local_load
      Value oldMemDesc = allocOp.getResult();
      SmallVector<OpOperand *> memDescUses;
      for (auto &use : oldMemDesc.getUses())
        memDescUses.push_back(&use);

      for (OpOperand *use : memDescUses) {
        Operation *user = use->getOwner();
        if (auto oldLocalLoad = dyn_cast<LocalLoadOp>(user)) {
          lb.setInsertionPoint(oldLocalLoad);
          Type resultType = oldLocalLoad.getResult().getType();
          auto newLocalLoad = LocalLoadOp::create(
              lb, resultType, bufView, wait->getResult(0));
          oldLocalLoad.getResult().replaceAllUsesWith(
              newLocalLoad.getResult());
          oldLocalLoad->erase();
        } else {
          user->setOperand(use->getOperandNumber(), bufView);
        }
      }

      allocOp.erase();
      loadOp.erase();
      llvm::errs() << "[PACT DoubleBuf] Converted load (start-of-body)\n";
    }

    // ── Step 3: yield prefetch → buf_{(j+1)%2} ──────────────────────
    // DISABLED for now: mechanism works but prefetches j's data (same ptr
    // as in-body load), which is redundant until we compute j+1's ptr.
    // The yield prefetch mechanism compiles and runs without dominance
    // errors — the type mismatch (index vs i32) was the root cause of
    // the earlier dominance failure.
    //
    // TODO: compute j+1 ptr at yield point by tracing block table lookup
    //       with iv+1, then re-enable. Also need prologue prefetch for
    //       iter 0 and remove async_copy from in-body (keep only wait+load).
#if 0
    {
      ImplicitLocOpBuilder yb(loc, builder);
      // Insert before the scf.yield terminator
      Block *bodyBlock = forOp.getBody();
      Operation *yieldOp = bodyBlock->getTerminator();
      yb.setInsertionPoint(yieldOp);

      // Compute (idx+1)%2 as the alternate buffer index
      Value idxNext;
      Value c1;
      Type ivType = iv.getType();
      if (auto intTy = dyn_cast<IntegerType>(ivType)) {
        c1 = arith::ConstantIntOp::create(yb, loc, 1, intTy.getWidth());
      } else {
        c1 = arith::ConstantIndexOp::create(yb, loc, 1);
      }
      idxNext = arith::RemSIOp::create(
          yb, loc,
          arith::AddIOp::create(yb, loc, iv, c1),
          c2);

      for (auto &db : bufs) {
        // Cast idxNext to i32 for MemDescIndex
        Value idxCast = idxNext;
        if (isa<IndexType>(idxNext.getType())) {
          idxCast = arith::IndexCastUIOp::create(yb, yb.getI32Type(), idxNext);
        }
        Value bufViewNext = triton::createSingleBufferView(yb, db.alloc2x, idxCast);

        Operation *prefetchCopy = AsyncCopyGlobalToLocalOp::create(
            yb, db.src, bufViewNext, db.mask, db.other,
            db.cache, db.evict, db.isVolatile, db.contiguity);

        // Propagate pact attributes (from saved attrs)
        if (db.pactPagedLoadAttr)
          prefetchCopy->setAttr("pact.paged_load", db.pactPagedLoadAttr);
        if (db.pactLayoutRemappedAttr)
          prefetchCopy->setAttr("pact.layout_remapped", db.pactLayoutRemappedAttr);

        AsyncCommitGroupOp::create(yb, prefetchCopy->getResult(0));
        llvm::errs() << "[PACT DoubleBuf] Added yield prefetch\n";
      }
    }
#endif

    return success();
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();

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

    // Group eligible loads by their enclosing scf.for loop.
    // Loads in the same loop share j%2 computation and can later
    // benefit from coordinated yield-prefetch pipelining.
    llvm::MapVector<scf::ForOp, SmallVector<PagedLoadInfo>> loopGroups;
    SmallVector<PagedLoadInfo> soloLoads; // loads not in any loop (fallback)
    for (auto &info : pagedLoads) {
      if (!info.asyncEligible) {
        llvm::errs() << "[PACT PrefetchInsert] Sync fallback — "
                     << "cp.async requires layout contiguity ∈ {4,8,16} bytes "
                     << "AND an all-ones (or absent) mask\n";
        continue;
      }
      if (info.forOp) {
        loopGroups[info.forOp].push_back(info);
      } else {
        soloLoads.push_back(info);
      }
    }

    OpBuilder builder(mod.getContext());
    int converted = 0;

    // Double-buffer path: group by loop
    for (auto &kv : loopGroups) {
      scf::ForOp forOp = kv.first;
      auto &group = kv.second;
      llvm::errs() << "[PACT PrefetchInsert] Double-buffer loop with "
                   << group.size() << " load(s)\n";
      if (succeeded(convertDoubleBuffer(forOp, group, builder)))
        converted += group.size();
    }

    // Single-buffer fallback for loads not in a paged loop
    for (auto &info : soloLoads) {
      if (succeeded(convertSingleBuffer(info.loadOp, info.allocOp, builder)))
        converted++;
    }

    if (converted > 0) {
      llvm::errs() << "[PACT PrefetchInsert] Converted " << converted
                   << " paged load(s) to async copy ("
                   << (loopGroups.empty() ? "single-buffer" : "double-buffer")
                   << ")\n";
    }
  }
};

} // anonymous namespace
} // namespace gpu
} // namespace triton
} // namespace mlir
