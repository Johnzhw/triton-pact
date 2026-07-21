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
#include "triton/Dialect/Triton/Transforms/LoopPeeling.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace mlir {
namespace triton {
namespace gpu {

#define GEN_PASS_DEF_TRITONGPUPREFETCHINSERT
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

namespace {

/// Find the block_table load in the ptr chain of a paged KV load.
/// Walks backward from the load's ptr through addptr, arith, and reshape
/// ops to find the tt.load with pact.block_table_lookup attribute.
static triton::LoadOp findBlockTableLoad(Value ptr, Value &tailOut) {
  // Walk from ptr backward through the SSA chain.
  SmallVector<Value> worklist;
  SmallPtrSet<Value, 16> visited;
  worklist.push_back(ptr);
  for (int depth = 0; depth < 32 && !worklist.empty(); ++depth) {
    SmallVector<Value> next;
    for (Value val : worklist) {
      if (!visited.insert(val).second)
        continue;
      auto *defOp = val.getDefiningOp();
      if (!defOp) {
        // Block argument — might be the IV or other loop arg
        if (auto blockArg = dyn_cast<BlockArgument>(val)) {
          // Don't walk into block arguments (IV, iter args)
          continue;
        }
        continue;
      }
      // If this is the block_table load, we found it
      if (auto loadOp = dyn_cast<triton::LoadOp>(defOp)) {
        if (loadOp->hasAttr("pact.block_table_lookup")) {
          // tailOut is the value just downstream from the block_table load.
          // Walk back from ptr to find the immediate user of the block_table
          // load result in the chain.
          tailOut = ptr; // default: use ptr as tail
          return loadOp;
        }
      }
      // Skip obvious non-chain ops (masks, scalars, etc.)
      if (isa<arith::ConstantOp>(defOp))
        continue;
      // Walk through intermediate ops: addptr, arith, reshape, etc.
      for (Value operand : defOp->getOperands())
        next.push_back(operand);
    }
    worklist = std::move(next);
  }
  return nullptr;
}

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
//
// When `stopMap` is provided, any value whose defining op is in stopMap
// is NOT cloned — the mapped replacement value is used instead.  This
// allows the caller to pre-compute expensive sub-chains (e.g., block_table
// loads) and share them across multiple clone chains.
//===----------------------------------------------------------------------===//
static Value cloneChainWithIVReplacement(Value origVal, Value oldIV,
                                          Value newIV, Block *forBody,
                                          ImplicitLocOpBuilder &builder,
                    const DenseMap<Operation *, Value> *stopMap = nullptr) {
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

    // Check stopMap: if this op should not be cloned, use the replacement.
    if (stopMap && stopMap->count(defOp)) {
      cloneMap[val] = stopMap->lookup(defOp);
      continue;
    }

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
    LocalAllocOp allocOp;       // may be null if no existing local_alloc
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

    // Determine SMEM encoding. If we have an allocOp, use its encoding
    // as the base. Otherwise, create a non-swizzled SMEM encoding from
    // the load's blocked encoding.
    Attribute newSharedEnc;
    MemDescType mutableMemDescType;
    Type loadResultType = loadOp.getResult().getType();

    if (allocOp) {
      auto oldMemDescType = cast<MemDescType>(allocOp.getResult().getType());
      auto oldEnc = oldMemDescType.getEncoding();

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

      mutableMemDescType = MemDescType::get(
          oldMemDescType.getShape(), oldMemDescType.getElementType(),
          newSharedEnc, oldMemDescType.getMemorySpace(),
          /*mutableMemory=*/true);
    } else {
      // No existing allocOp: create SMEM encoding from the load's layout.
      auto resTy = dyn_cast<RankedTensorType>(loadResultType);
      if (!resTy)
        return failure();

      auto blockedEnc =
          dyn_cast<BlockedEncodingAttr>(resTy.getEncoding());
      if (!blockedEnc) {
        llvm::errs() << "[PACT] No blocked encoding on load result\n";
        return failure();
      }

      auto ctx = blockedEnc.getContext();
      auto order = blockedEnc.getOrder();
      // Get CGA layout from the blocked encoding
      auto cgaLayout = triton::gpu::getCGALayout(blockedEnc);

      Attribute sharedMemorySpace =
          triton::gpu::SharedMemorySpaceAttr::get(ctx);
      newSharedEnc = SwizzledSharedEncodingAttr::get(
          ctx, /*vec=*/2, /*perPhase=*/1, /*maxPhase=*/1, order, cgaLayout);

      auto shape = resTy.getShape();
      auto elemType = resTy.getElementType();
      mutableMemDescType = MemDescType::get(
          SmallVector<int64_t>(shape.begin(), shape.end()),
          elemType, newSharedEnc, sharedMemorySpace,
          /*mutableMemory=*/true);
    }

    ImplicitLocOpBuilder b(loc, builder);

    // For splat-1 masks (all-ones), skip the mask operand
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

    // Propagate pact attributes
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

    if (allocOp) {
      // Replace uses of old MemDesc
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
    } else {
      // No existing allocOp: replace load result uses directly
      // with local_load from the new SMEM buffer.
      b.setInsertionPointAfter(wait);
      auto localLoad = LocalLoadOp::create(
          b, loadResultType, newBuf, wait->getResult(0));
      loadOp.getResult().replaceAllUsesWith(localLoad.getResult());
    }

    loadOp.erase();
    return success();
  }

  /// Double-buffer conversion for paged K/V loads in a scf.for loop.
  ///
  /// Two modes controlled by PACT_DOUBLEBUF_PIPELINE:
  ///   0 (default): 2× buf allocation + in-body async_copy (single-buffer style).
  ///      Allocates 2× multi-buffered local_alloc before the loop, uses j%2 to
  ///      select the buffer at the load position, and does async_copy+wait+load
  ///      inline.  No software pipelining — correct for all cases.
  ///
  ///   1 (experimental): Software-pipelined double-buffer.  Prologue async_copy
  ///      to buf[0], body wait+load+prefetch(j+1).  Code compiles and passes
  ///      MLIR verification (7/7 PASS at 96494986e), but has two known issues:
  ///      (a) j+1 OOB on last iteration — needs scf.if bounds guard
  ///      (b) cudaErrorMisalignedAddress in cp.async nBytes=2 paired path
  ///      (also affects simple mode with CUDA_LAUNCH_BLOCKING=1).
  ///      Disabled until both issues are resolved.
#define PACT_DOUBLEBUF_PIPELINE 0

  LogicalResult convertDoubleBuffer(scf::ForOp forOp,
                                     SmallVectorImpl<PagedLoadInfo> &loads,
                                     OpBuilder &builder) {
    if (loads.empty())
      return success();

    Location loc = forOp.getLoc();
    Block *bodyBlock = forOp.getBody();

    // Get induction variable
    Value iv = forOp.getInductionVar();
    Type ivType = iv.getType();

    // Simple mode: delegate each load to convertSingleBuffer.
    // No 2x buf, no MemDescIndex — avoids non-swizzled SMEM alignment
    // issues and the cp.async partial-tile OOB problem.
#if !PACT_DOUBLEBUF_PIPELINE
    for (auto &info : loads) {
      if (failed(convertSingleBuffer(info.loadOp, info.allocOp, builder)))
        llvm::errs() << "[PACT] convertSingleBuffer failed in loop\n";
    }
    return success();
#endif

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

        Attribute newSharedEnc;
        MemDescType mutableMemDescType;

        if (allocOp) {
          auto oldMemDescType =
              cast<MemDescType>(allocOp.getResult().getType());

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

          mutableMemDescType = MemDescType::get(
              oldMemDescType.getShape(), oldMemDescType.getElementType(),
              newSharedEnc, oldMemDescType.getMemorySpace(),
              /*mutableMemory=*/true);
        } else {
          // No existing allocOp: build SMEM encoding from load layout
          auto resTy =
              dyn_cast<RankedTensorType>(loadOp.getResult().getType());
          if (!resTy) {
            llvm::errs() << "[PACT DoubleBuf] ERROR: no ranked tensor type\n";
            continue;
          }

          auto blockedEnc =
              dyn_cast<BlockedEncodingAttr>(resTy.getEncoding());
          if (!blockedEnc) {
            llvm::errs() << "[PACT DoubleBuf] ERROR: no blocked encoding\n";
            continue;
          }

          auto ctx = blockedEnc.getContext();
          auto order = blockedEnc.getOrder();
          // Get CGA layout from the blocked encoding's own CGA layout
          auto cgaLayout = triton::gpu::getCGALayout(blockedEnc);
          Attribute sharedMemorySpace =
              triton::gpu::SharedMemorySpaceAttr::get(ctx);
          newSharedEnc = SwizzledSharedEncodingAttr::get(
              ctx, /*vec=*/2, /*perPhase=*/1, /*maxPhase=*/1, order,
              cgaLayout);

          auto shape = resTy.getShape();
          auto elemType = resTy.getElementType();
          mutableMemDescType = MemDescType::get(
              SmallVector<int64_t>(shape.begin(), shape.end()),
              elemType, newSharedEnc, sharedMemorySpace,
              /*mutableMemory=*/true);
        }

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

#if PACT_DOUBLEBUF_PIPELINE
    // ═══════════════════════════════════════════════════════════════════
    // PIPELINE MODE: prologue + wait + j+1 prefetch (experimental)
    // ═══════════════════════════════════════════════════════════════════

    // ── Step 2: Prologue — prefetch iter 0 data into buf[0] ──────────
    // Clone the ptr computation chain with IV replaced by constant 0,
    // inserted before the loop.  All async copies share one merged commit.
    SmallVector<Value> prologueCommits;
    {
      ImplicitLocOpBuilder pb(loc, builder);
      pb.setInsertionPoint(forOp);

      SmallVector<Value> prologueAsyncTokens;
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

        prologueAsyncTokens.push_back(copy->getResult(0));

        llvm::errs() << "[PACT DoubleBuf] Prologue prefetch to buf[0]\n";
      }

      // Merged commit for all prologue async copies
      if (!prologueAsyncTokens.empty()) {
        Operation *commit = AsyncCommitGroupOp::create(pb, prologueAsyncTokens);
        prologueCommits.push_back(commit->getResult(0));
        llvm::errs() << "[PACT DoubleBuf] Prologue merged commit for "
                     << prologueAsyncTokens.size() << " async copies\n";
      }
    }

    // ── Step 3: Body start — j%2 + wait(0) ──────────────────────────
    ImplicitLocOpBuilder bodyBuilder(loc, builder);
    bodyBuilder.setInsertionPointToStart(bodyBlock);

    Value idx = arith::RemSIOp::create(bodyBuilder, loc, iv, c2);

    // Cast idx to i32 for MemDescIndex (if needed).
    // MemDescIndex requires i32; IV may be i64 or index type.
    Value idxI32 = idx;
    Type idxType = idx.getType();
    if (isa<IndexType>(idxType)) {
      idxI32 = arith::IndexCastUIOp::create(bodyBuilder, bodyBuilder.getI32Type(), idx);
    } else if (auto intTy = dyn_cast<IntegerType>(idxType)) {
      if (intTy.getWidth() != 32) {
        idxI32 = arith::TruncIOp::create(bodyBuilder, bodyBuilder.getI32Type(), idx);
      }
    }

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
    Type idxNextType = idxNext.getType();
    if (isa<IndexType>(idxNextType)) {
      idxNextI32 = arith::IndexCastUIOp::create(bodyBuilder, bodyBuilder.getI32Type(), idxNext);
    } else if (auto intTy = dyn_cast<IntegerType>(idxNextType)) {
      if (intTy.getWidth() != 32) {
        idxNextI32 = arith::TruncIOp::create(bodyBuilder, bodyBuilder.getI32Type(), idxNext);
      }
    }

    // ── Pre-step: Pre-compute block_table lookup for iter j+1 ─────────
    // The K and V loads share the same phys_block pointer from the
    // block_table lookup.  Instead of cloning this lookup inside each
    // cloneChain call (2× per iteration), compute it once here and
    // share across both prefetches.
    //
    // IMPORTANT: insert at the position of the FIRST load, not at body
    // start.  jPlus1 was inserted at body start; inserting before it
    // would create a dominance violation.
    DenseMap<Operation *, Value> sharedBlockTableMap;
    SmallVector<Operation *> sharedBTOps;
    if (!bufs.empty()) {
      ImplicitLocOpBuilder sbtb(loc, builder);
      auto &firstLoad = bufs[0].loadOp;
      sbtb.setInsertionPoint(firstLoad);
      for (auto &db : bufs) {
        Value _tail;
        auto btLoad = findBlockTableLoad(db.origSrc, _tail);
        if (btLoad && !sharedBlockTableMap.count(btLoad)) {
          Value physNext = cloneChainWithIVReplacement(
              btLoad.getResult(), iv, jPlus1, bodyBlock, sbtb);
          if (physNext) {
            sharedBlockTableMap[btLoad] = physNext;
            sharedBTOps.push_back(btLoad);
            llvm::errs() << "[PACT DoubleBuf] Shared block_table"
                         << " prefetch for iter j+1\n";
          }
        }
      }
    }

    // ── Step 5: For each load, local_load + async_copy next ──────────
    // We process loads at their original position:
    //   a) local_load from buf[j%2] (replaces old local_load)
    //   b) clone next ptr + async_copy into buf[(j+1)%2]
    //   c) Merged commit for all async copies (reduces overhead)
    SmallVector<Value> bodyAsyncTokens;
    for (auto &db : bufs) {
      auto &loadOp = db.loadOp;
      auto &allocOp = db.allocOp;
      Location loadLoc = loadOp.getLoc();

      // Save the block position before the loadOp for later insertions
      Block *loadBlock = loadOp->getBlock();
      Block::iterator loadPos(loadOp);

      // -- 5a: Create bufView and replace local_load --
      ImplicitLocOpBuilder lb(loadLoc, builder);
      lb.setInsertionPoint(loadBlock, loadPos);

      Value bufView = triton::createSingleBufferView(lb, db.alloc2x, idxI32);

      Type loadResultType = loadOp.getResult().getType();

      if (allocOp) {
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
        allocOp.erase();
      } else {
        // No existing allocOp: replace load result uses directly
        lb.setInsertionPoint(loadBlock, loadPos);
        auto localLoad = LocalLoadOp::create(
            lb, loadResultType, bufView, /*token=*/Value());
        loadOp.getResult().replaceAllUsesWith(localLoad.getResult());
      }

      // -- 5b: clone j+1 ptr and prefetch into alternate buffer --
      // Pass sharedBlockTableMap so cloneChain reuses the pre-computed
      // phys_block instead of cloning the block_table load again.
      lb.setInsertionPoint(loadBlock, loadPos);
      const DenseMap<Operation *, Value> *stopMap =
          sharedBlockTableMap.empty() ? nullptr : &sharedBlockTableMap;
      Value nextPtr = cloneChainWithIVReplacement(
          db.origSrc, iv, jPlus1, bodyBlock, lb, stopMap);

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

        bodyAsyncTokens.push_back(prefetchCopy->getResult(0));
        llvm::errs() << "[PACT DoubleBuf] Prefetch j+1 to alternate buf\n";
      } else {
        llvm::errs() << "[PACT DoubleBuf] WARNING: in-body clone "
                        "returned null, skipping prefetch\n";
      }

      // -- Clean up old loadOp --
      // (allocOp was already erased in Step 5a if it existed)
      loadOp.erase();
    }
    // ── Step 5c: Merged commit for all prefetch async copies ─────────
    // Instead of one commit per async copy, batch all tokens into one
    // commit group.  Reduces instruction overhead without affecting
    // correctness (wait(0) drains all pending groups).
    if (!bodyAsyncTokens.empty()) {
      ImplicitLocOpBuilder cb(loc, builder);
      cb.setInsertionPointAfter(bodyAsyncTokens.back().getDefiningOp());
      AsyncCommitGroupOp::create(cb, bodyAsyncTokens);
      llvm::errs() << "[PACT DoubleBuf] Merged commit for "
                   << bodyAsyncTokens.size() << " async copies\n";
    }

    llvm::errs() << "[PACT DoubleBuf] Pipelined conversion complete\n";

    // ── Step 6: Peel the last iteration to avoid OOB prefetch ──────────
    // The pipeline body does wait+load+prefetch(j+1).  On the last
    // iteration (j=N-1), the prefetch for j+1=N would access OOB
    // addresses.  Instead of an scf.if guard on every iteration, we peel
    // the last iteration into an epilogue where the prefetch ops are
    // removed.  The peeled iteration runs outside the loop with only
    // wait+load (the data was prefetched in the penultimate loop iter).
    //
    // Edge cases handled by peelLoopEpilogue:
    //   - N=1: loop runs 0 iters, epilogue runs once (data from prologue)
    //   - N=2: loop runs 1 iter (prefetch for iter 1), epilogue consumes it
    {
      llvm::errs() << "[PACT DoubleBuf] Peeling last iteration"
                   << " to eliminate OOB prefetch\n";
      mlir::triton::peelLoopEpilogue(forOp);

      // After peeling, the epilogue is an scf.if after the forOp.
      // Its then-region contains a clone of the full loop body (including
      // the prefetch async_copy+commit).  Remove those prefetch ops.
      // The peelLoopEpilogue inserts computation ops (lastIV, cond) between
      // the forOp and the scf.if, so walk forward to find the ifOp.
      Operation *nextOp = forOp->getNextNode();
      scf::IfOp ifOp = nullptr;
      while (nextOp) {
        ifOp = dyn_cast<scf::IfOp>(nextOp);
        if (ifOp) break;
        nextOp = nextOp->getNextNode();
      }
      if (ifOp) {
        Block &epilogueBlock = ifOp.getThenRegion().front();
        SmallVector<Operation *> toErase;
        for (auto &op : epilogueBlock.without_terminator()) {
          if (isa<AsyncCopyGlobalToLocalOp>(op) ||
              isa<AsyncCommitGroupOp>(op)) {
            toErase.push_back(&op);
          }
        }
        // Erase in reverse order: commit uses async_copy result, so
        // erase commit first, then async_copy.
        for (auto *op : llvm::reverse(toErase)) {
          op->dropAllUses();
          op->erase();
        }
        llvm::errs() << "[PACT DoubleBuf] Epilogue: removed "
                     << toErase.size() << " prefetch op(s)\n";
      } else {
        llvm::errs() << "[PACT DoubleBuf] WARNING: expected scf.if"
                     << " epilogue after loop peeling\n";
      }
    }
#else  // PACT_DOUBLEBUF_PIPELINE == 0
    // ═══════════════════════════════════════════════════════════════════
    // SIMPLE MODE: 2x buf allocation + in-body async_copy (7/7 PASS)
    //
    // Same pattern as convertSingleBuffer, but uses the 2x multi-buffered
    // allocation and j%2 buffer selection.  No software pipelining —
    // async_copy + commit + wait + local_load all happen inline at the
    // original load position.  Equivalent PTX to single-buffer (9 cp.async).
    // ═══════════════════════════════════════════════════════════════════

    // Body start: j%2 computation (same constants already created above)
    ImplicitLocOpBuilder bodyBuilder(loc, builder);
    bodyBuilder.setInsertionPointToStart(bodyBlock);

    Value idx = arith::RemSIOp::create(bodyBuilder, loc, iv, c2);
    Value idxI32 = idx;
    if (isa<IndexType>(idx.getType()))
      idxI32 = arith::IndexCastUIOp::create(bodyBuilder,
                                             bodyBuilder.getI32Type(), idx);

    // Process each load: async_copy + wait + local_load inline
    for (auto &db : bufs) {
      auto &loadOp = db.loadOp;
      auto &allocOp = db.allocOp;

      ImplicitLocOpBuilder lb(loc, builder);
      lb.setInsertionPoint(loadOp);

      Value bufView = triton::createSingleBufferView(lb, db.alloc2x, idxI32);

      Operation *copy = AsyncCopyGlobalToLocalOp::create(
          lb, db.origSrc, bufView, db.mask, db.other,
          db.cache, db.evict, db.isVolatile, db.contiguity);

      if (db.pactPagedLoadAttr)
        copy->setAttr("pact.paged_load", db.pactPagedLoadAttr);
      if (db.pactLayoutRemappedAttr)
        copy->setAttr("pact.layout_remapped", db.pactLayoutRemappedAttr);

      Operation *commit =
          AsyncCommitGroupOp::create(lb, copy->getResult(0));
      Operation *wait =
          AsyncWaitOp::create(lb, commit->getResult(0), 0);

      // Replace old local_load uses
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
    }

    llvm::errs() << "[PACT DoubleBuf] Simple 2x buf conversion complete\n";
#endif  // PACT_DOUBLEBUF_PIPELINE

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
        if (!loadOp->hasAttr("pact.paged_load")) {
          return WalkResult::advance();
        }

        LocalAllocOp allocOp =
            findLocalAllocThroughLayoutConversions(loadOp);
        // allocOp may be null — we'll create our own SMEM

        if (!canUseCpAsync(loadOp)) {
          return WalkResult::advance();
        }

        PagedLoadInfo info;
        info.loadOp = loadOp;
        info.allocOp = allocOp;  // may be null
        info.forOp = forOp;
        info.asyncEligible = true;
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
