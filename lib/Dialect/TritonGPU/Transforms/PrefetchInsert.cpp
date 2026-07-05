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

//===----------------------------------------------------------------------===//
// Helper: clone a value's definition chain, replacing oldIV with newIV.
// Walks the use-def chain from `origVal` back through ops in `forBody`,
// collects them in topological order, then clones each one with the IV
// substitution applied.  Returns the cloned value corresponding to origVal,
// or a null Value if no cloning was needed (origVal doesn't depend on IV).
//===----------------------------------------------------------------------===//
static Value cloneChainWithIVReplacement(Value origVal, Value oldIV,
                                          Value newIV, Block *forBody,
                                          ImplicitLocOpBuilder &builder) {
  // If origVal already equals oldIV, just return newIV
  if (origVal == oldIV)
    return newIV;

  // Step 1: collect ops in the definition chain (BFS from origVal upward)
  SmallVector<Operation *> chainOps; // top-down (defs before uses)
  SmallVector<Value> worklist;
  SmallPtrSet<Operation *, 16> seen;
  DenseMap<Value, Value> cloneMap;

  worklist.push_back(origVal);
  cloneMap[oldIV] = newIV;

  while (!worklist.empty()) {
    Value val = worklist.pop_back_val();
    if (cloneMap.count(val))
      continue;

    // Block arguments that aren't the IV are left as-is
    auto blockArg = dyn_cast<BlockArgument>(val);
    if (blockArg) {
      cloneMap[val] = val; // pass through unchanged
      continue;
    }

    Operation *defOp = val.getDefiningOp();
    if (!defOp || seen.count(defOp))
      continue;

    // Only clone ops inside the loop body.
    // For ops outside the loop, map results to themselves (pass-through).
    if (!defOp->getBlock() || defOp->getBlock() != forBody) {
      for (Value res : defOp->getResults())
        cloneMap[res] = res;
      continue;
    }

    seen.insert(defOp);
    chainOps.push_back(defOp);

    // Visit operands
    for (Value operand : defOp->getOperands()) {
      if (!cloneMap.count(operand))
        worklist.push_back(operand);
    }
  }

  if (chainOps.empty()) {
    // origVal doesn't depend on anything inside the loop — pass through
    return origVal;
  }

  // Step 2: reverse to get topological order (defs before uses)
  std::reverse(chainOps.begin(), chainOps.end());

  llvm::errs() << "[PACT cloneChain] Cloning " << chainOps.size()
               << " ops with IV replacement\n";

  // Step 3: clone each op in topological order.
  // Use iterative approach: repeatedly clone ops whose operands are all mapped,
  // until all ops are cloned or no progress is made.
  SmallVector<Operation *> pending(chainOps.begin(), chainOps.end());
  while (!pending.empty()) {
    bool madeProgress = false;
    SmallVector<Operation *> nextPending;

    for (Operation *op : pending) {
      // Check if all operands are already mapped.
      // Add pass-through mappings for loop-invariant values on the fly.
      bool allMapped = true;
      for (Value operand : op->getOperands()) {
        if (cloneMap.count(operand))
          continue;
        // Loop-invariant: block arg (not IV) or defined outside forBody
        auto *defOp = operand.getDefiningOp();
        if (!defOp || defOp->getBlock() != forBody) {
          cloneMap[operand] = operand; // pass-through
          continue;
        }
        allMapped = false;
        break;
      }
      if (!allMapped) {
        nextPending.push_back(op);
        continue;
      }

      madeProgress = true;

      // Build new operands
      SmallVector<Value> newOperands;
      for (Value operand : op->getOperands())
        newOperands.push_back(cloneMap.lookup(operand));

      // Clone and insert
      Operation *cloned = op->clone();
      if (!cloned) {
        llvm::errs() << "  FAILED to clone " << op->getName() << "\n";
        return Value();
      }
      builder.insert(cloned);

      // Set the substituted operands
      for (unsigned i = 0; i < newOperands.size(); ++i)
        cloned->setOperand(i, newOperands[i]);

      // Map old results to new results
      for (auto [oldRes, newRes] :
           llvm::zip(op->getResults(), cloned->getResults()))
        cloneMap[oldRes] = newRes;
    }

    if (!madeProgress) {
      llvm::errs() << "[PACT cloneChain] ERROR: " << nextPending.size()
                   << " ops unclonable (dependency cycle or missing operands)\n";
      return Value();
    }
    pending = std::move(nextPending);
  }

  return cloneMap.lookup(origVal);

  return cloneMap.lookup(origVal);
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

  /// Double-buffer conversion with software pipelining.
  ///
  /// Pattern:
  ///   Prologue (before loop): async_copy ptr(j=0) → buf[0], commit
  ///
  ///   Loop body (iteration j):
  ///     1. j%2, wait(0) — get data prefetched by prev iteration or prologue
  ///     2. local_load buf[j%2] — use current iteration's data
  ///     3. Compute ptr for j+1 (clone ptr chain with IV→j+1)
  ///     4. async_copy ptr(j+1) → buf[(j+1)%2], commit — prefetch for next iter
  ///     5. ... computation runs concurrently with step 4 ...
  ///
  ///   The prologue is created by cloning the ptr definition chain with the
  ///   induction variable replaced by constant 0, before the scf.for.
  LogicalResult convertDoubleBuffer(scf::ForOp forOp,
                                     SmallVectorImpl<PagedLoadInfo> &loads,
                                     OpBuilder &builder) {
    if (loads.empty())
      return success();

    Location loc = forOp.getLoc();
    Block *bodyBlock = forOp.getBody();

    // Get induction variable — used for j%2 and ptr cloning
    Value iv = forOp.getInductionVar();
    Type ivType = iv.getType();

    // ── Constants (matching IV type) ──────────────────────────────────
    ImplicitLocOpBuilder constBuilder(loc, builder);
    constBuilder.setInsertionPoint(forOp);

    Value c0, c1, c2;
    if (auto intTy = dyn_cast<IntegerType>(ivType)) {
      unsigned w = intTy.getWidth();
      c0 = arith::ConstantIntOp::create(constBuilder, loc, 0, w);
      c1 = arith::ConstantIntOp::create(constBuilder, loc, 1, w);
      c2 = arith::ConstantIntOp::create(constBuilder, loc, 2, w);
    } else {
      c0 = arith::ConstantIndexOp::create(constBuilder, loc, 0);
      c1 = arith::ConstantIndexOp::create(constBuilder, loc, 1);
      c2 = arith::ConstantIndexOp::create(constBuilder, loc, 2);
    }

    // ── Step 1: allocate 2× buffers BEFORE the loop ───────────────────
    struct DoubleBuf {
      Value alloc2x;       // multi-buffered MemDesc<[2, shape...]>
      triton::LoadOp loadOp;
      LocalAllocOp allocOp;
      Value origSrc;       // ptr from original load
      Value mask;
      Value other;
      triton::CacheModifier cache;
      triton::EvictionPolicy evict;
      bool isVolatile;
      uint32_t contiguity;
      Attribute pactPagedLoadAttr;
      Attribute pactLayoutRemappedAttr;
    };
    SmallVector<DoubleBuf> bufs;

    {
      ImplicitLocOpBuilder b(loc, builder);
      b.setInsertionPoint(forOp);

      for (auto &info : loads) {
        auto &loadOp = info.loadOp;
        auto &allocOp = info.allocOp;

        auto oldMemDescType =
            cast<MemDescType>(allocOp.getResult().getType());

        Attribute newSharedEnc;
        if (loadOp->hasAttr("pact.paged_load")) {
          auto ctx = oldMemDescType.getEncoding().getContext();
          auto swizzledEnc = mlir::cast<SwizzledSharedEncodingAttr>(
              oldMemDescType.getEncoding());
          auto order = swizzledEnc.getOrder();
          auto cgaLayout = swizzledEnc.getCGALayout();
          newSharedEnc = SwizzledSharedEncodingAttr::get(
              ctx, /*vec=*/2, /*perPhase=*/1, /*maxPhase=*/1, order,
              cgaLayout);
        } else {
          newSharedEnc = oldMemDescType.getEncoding();
        }

        auto mutableMemDescType = MemDescType::get(
            oldMemDescType.getShape(), oldMemDescType.getElementType(),
            newSharedEnc, oldMemDescType.getMemorySpace(),
            /*mutableMemory=*/true);

        auto multiBufType =
            triton::getMultiBufferedType(mutableMemDescType, 2);
        Value alloc2x = LocalAllocOp::create(b, multiBufType).getResult();

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

        DoubleBuf db;
        db.alloc2x = alloc2x;
        db.loadOp = loadOp;
        db.allocOp = allocOp;
        db.origSrc = src;
        db.mask = asyncMask;
        db.other = other;
        db.cache = cache;
        db.evict = evict;
        db.isVolatile = isVolatile;
        db.contiguity = contiguity;
        db.pactPagedLoadAttr = loadOp->getAttr("pact.paged_load");
        db.pactLayoutRemappedAttr = loadOp->getAttr("pact.layout_remapped");
        bufs.push_back(db);

        llvm::errs() << "[PACT DoubleBuf] Allocated 2x buffer\n";
      }
    }

    // ── Step 2: Prologue — prefetch iter 0 data into buf[0] ──────────
    // Clone the ptr computation chain with IV replaced by constant 0,
    // inserted before the loop.
    SmallVector<Value> prologueCommits;
    {
      ImplicitLocOpBuilder pb(loc, builder);
      pb.setInsertionPoint(forOp);

      for (unsigned i = 0; i < bufs.size(); ++i) {
        auto &db = bufs[i];
        // Clone ptr chain: replace IV with constant 0
        Value ptr0 = cloneChainWithIVReplacement(
            db.origSrc, iv, c0, bodyBlock, pb);

        if (!ptr0) {
          llvm::errs() << "[PACT DoubleBuf] WARNING: prologue clone "
                          "returned null, skipping\n";
          continue;
        }

        // buf[0] = prologue data
        Value bufView0 = triton::createSingleBufferView(pb, db.alloc2x, 0);

        Operation *copy = AsyncCopyGlobalToLocalOp::create(
            pb, ptr0, bufView0, db.mask, db.other,
            db.cache, db.evict, db.isVolatile, db.contiguity);

        if (db.pactPagedLoadAttr)
          copy->setAttr("pact.paged_load", db.pactPagedLoadAttr);
        if (db.pactLayoutRemappedAttr)
          copy->setAttr("pact.layout_remapped", db.pactLayoutRemappedAttr);

        Operation *commit =
            AsyncCommitGroupOp::create(pb, copy->getResult(0));
        prologueCommits.push_back(commit->getResult(0));

        llvm::errs() << "[PACT DoubleBuf] Prologue prefetch to buf[0]\n";
      }
    }

    // ── Step 3: Body start — j%2 + wait(0) ──────────────────────────
    ImplicitLocOpBuilder bodyBuilder(loc, builder);
    bodyBuilder.setInsertionPointToStart(bodyBlock);

    Value idx = arith::RemSIOp::create(bodyBuilder, loc, iv, c2);

    // Cast idx to i32 for MemDescIndex (if needed)
    Value idxI32 = idx;
    if (isa<IndexType>(idx.getType()))
      idxI32 = arith::IndexCastUIOp::create(bodyBuilder, bodyBuilder.getI32Type(), idx);

    // wait(0) — drains prologue (iter 0) or previous iteration's prefetch.
    // wait 0 drains all pending groups; the token is just a dummy reference.
    Value waitToken;
    if (!prologueCommits.empty()) {
      waitToken = prologueCommits.front();
    }
    AsyncWaitOp::create(bodyBuilder, waitToken, 0);

    // ── Step 4: Compute j+1 and idxNext for yield prefetch ───────────
    Value jPlus1 = arith::AddIOp::create(bodyBuilder, loc, iv, c1);
    Value idxNext = arith::RemSIOp::create(bodyBuilder, loc, jPlus1, c2);
    Value idxNextI32 = idxNext;
    if (isa<IndexType>(idxNext.getType()))
      idxNextI32 = arith::IndexCastUIOp::create(bodyBuilder, bodyBuilder.getI32Type(), idxNext);

    // ── Step 5: For each load, local_load + async_copy next ──────────
    // We process loads at their original position:
    //   a) local_load from buf[j%2] (replaces old local_load)
    //   b) clone next ptr + async_copy into buf[(j+1)%2] + commit
    for (auto &db : bufs) {
      auto &loadOp = db.loadOp;
      auto &allocOp = db.allocOp;
      Location loadLoc = loadOp.getLoc();

      // Save the block position before the loadOp for later insertions
      Block *loadBlock = loadOp->getBlock();
      Block::iterator loadPos(loadOp);
      // loadPos points to loadOp; inserting at loadPos puts new ops BEFORE loadOp

      // -- 5a: Create bufView and replace local_load --
      // Use a temporary builder at the load position
      ImplicitLocOpBuilder lb(loadLoc, builder);
      lb.setInsertionPoint(loadBlock, loadPos);

      Value bufView = triton::createSingleBufferView(lb, db.alloc2x, idxI32);

      // Replace old local_alloc uses with local_load from bufView
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
              lb, resultType, bufView, /*token=*/Value());
          oldLocalLoad.getResult().replaceAllUsesWith(
              newLocalLoad.getResult());
          oldLocalLoad->erase();
        } else {
          user->setOperand(use->getOperandNumber(), bufView);
        }
      }

      // -- 5b: clone j+1 ptr and prefetch into alternate buffer --
      // Do this BEFORE erasing loadOp — the IP points to before loadOp
      lb.setInsertionPoint(loadBlock, loadPos);
      Value nextPtr = cloneChainWithIVReplacement(
          db.origSrc, iv, jPlus1, bodyBlock, lb);

      if (nextPtr) {
        lb.setInsertionPoint(loadBlock, loadPos);
        Value bufViewNext =
            triton::createSingleBufferView(lb, db.alloc2x, idxNextI32);

        Operation *prefetchCopy = AsyncCopyGlobalToLocalOp::create(
            lb, nextPtr, bufViewNext, db.mask, db.other,
            db.cache, db.evict, db.isVolatile, db.contiguity);

        if (db.pactPagedLoadAttr)
          prefetchCopy->setAttr("pact.paged_load", db.pactPagedLoadAttr);
        if (db.pactLayoutRemappedAttr)
          prefetchCopy->setAttr("pact.layout_remapped",
                                db.pactLayoutRemappedAttr);

        AsyncCommitGroupOp::create(lb, prefetchCopy->getResult(0));
        llvm::errs() << "[PACT DoubleBuf] Prefetch j+1 to alternate buf\n";
      } else {
        llvm::errs() << "[PACT DoubleBuf] WARNING: in-body clone "
                        "returned null, skipping prefetch\n";
      }

      // -- Clean up old allocOp and loadOp --
      allocOp.erase();
      loadOp.erase();
    }

    llvm::errs() << "[PACT DoubleBuf] Pipelined conversion complete\n";
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
