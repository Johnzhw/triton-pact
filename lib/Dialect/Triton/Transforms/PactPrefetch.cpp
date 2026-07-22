//===- PactPrefetch.cpp - PACT Block Table Prefetch Pass ------------------===//
//
// Inserts cache-warming prefetch loads at the end of the tile loop.
// For each iteration j, after computing attention, we issue a load for
// the block_table entry that iteration j+1 will need.  When iteration j+1
// starts and performs its own block_table load, the data is likely already
// in L1 cache, hiding global memory latency.
//
// This is a non-intrusive "cache warmup" strategy — the loop structure
// is unchanged; we only add prefetch loads before the scf.yield.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::triton {

#define GEN_PASS_DEF_TRITONPACTPREFETCH
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

namespace {

/// Lightweight analysis: extract loop-invariant values needed to compute
/// phys at an arbitrary IV value.
struct PrefetchInfo {
  Value tileSize;         // i32 constant (TILE_SIZE, typically 16)
  Value offsM;            // tensor<16xi32> (tt.make_range 0..15)
  Value pageSize;         // tensor<16xi32> (PAGE_SIZE constant)
  Value btScalarPtr;      // !tt.ptr<i32> — scalar block_table base (outside loop)
  Type btSplatPtrType;    // tensor<16x!tt.ptr<i32>>
  Type btLoadType;        // tensor<16xi32>
  Value numTiles;         // i32 upper bound
  bool valid = false;
};

static PrefetchInfo analyzeBlockTable(scf::ForOp forOp) {
  PrefetchInfo info;
  info.numTiles = forOp.getUpperBound();

  Operation *btLoad = nullptr;
  forOp.walk([&](Operation *op) {
    if (op->hasAttr("pact.block_table_lookup")) {
      btLoad = op;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (!btLoad)
    return info;

  info.btLoadType = btLoad->getResult(0).getType();

  auto loadOp = dyn_cast<triton::LoadOp>(btLoad);
  if (!loadOp)
    return info;
  Value btPtr = loadOp.getPtr();

  // bt_ptr = tt.addptr %btBaseSplat, %page_idx
  auto *addptrOp = btPtr.getDefiningOp();
  if (!addptrOp || addptrOp->getName().getStringRef() != "tt.addptr")
    return info;
  Value btBaseSplat = addptrOp->getOperand(0);
  info.btSplatPtrType = btBaseSplat.getType();
  Value pageIdx = addptrOp->getOperand(1);

  // page_idx = divsi %seq_tensor, %pageSize
  auto *divOp = pageIdx.getDefiningOp();
  if (!divOp ||
      (divOp->getName().getStringRef() != "arith.divsi" &&
       divOp->getName().getStringRef() != "arith.floordivsi" &&
       divOp->getName().getStringRef() != "arith.shrsi"))
    return info;
  Value seqTensor = divOp->getOperand(0);
  info.pageSize = divOp->getOperand(1);

  // seq_tensor = addi %seq_splat, %offs_m
  auto *addiOp = seqTensor.getDefiningOp();
  if (!addiOp || addiOp->getName().getStringRef() != "arith.addi")
    return info;
  Value a0 = addiOp->getOperand(0), a1 = addiOp->getOperand(1);
  auto *s0 = a0.getDefiningOp(), *s1 = a1.getDefiningOp();
  Value splatVal;
  if (s0 && s0->getName().getStringRef() == "tt.splat") {
    splatVal = a0;
    info.offsM = a1;
  } else if (s1 && s1->getName().getStringRef() == "tt.splat") {
    splatVal = a1;
    info.offsM = a0;
  } else {
    return info;
  }

  // seq_scalar = arith.muli %iv, %tileSize
  Value seqScalar = splatVal.getDefiningOp()->getOperand(0);
  auto *muliOp = seqScalar.getDefiningOp();
  if (!muliOp || muliOp->getName().getStringRef() != "arith.muli")
    return info;
  info.tileSize = (muliOp->getOperand(0) == forOp.getInductionVar())
                      ? muliOp->getOperand(1)
                      : muliOp->getOperand(0);

  // btBaseSplat = tt.splat %scalarPtr
  auto *splatBase = btBaseSplat.getDefiningOp();
  if (!splatBase || splatBase->getName().getStringRef() != "tt.splat")
    return info;
  info.btScalarPtr = splatBase->getOperand(0);

  info.valid = true;
  return info;
}

struct PactPrefetchPass
    : public impl::TritonPactPrefetchBase<PactPrefetchPass> {

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    int numTransformed = 0;

    SmallVector<scf::ForOp> worklist;
    mod.walk([&](scf::ForOp forOp) {
      auto *p = forOp->getParentOp();
      while (p) {
        if (p->hasAttr("pact.has_paged_access")) {
          worklist.push_back(forOp);
          break;
        }
        p = p->getParentOp();
      }
    });

    for (auto forOp : worklist) {
      auto info = analyzeBlockTable(forOp);
      if (!info.valid) {
        forOp->setAttr("pact.prefetch_eligible",
                       UnitAttr::get(&getContext()));
        continue;
      }

      insertPrefetch(forOp, info);
      numTransformed++;
    }

    if (numTransformed > 0)
      llvm::errs() << "[PACT PactPrefetch] Prefetch inserted in "
                   << numTransformed << " loop(s)\n";
  }

private:
  /// Insert a prefetch load at the end of the loop body (before yield).
  void insertPrefetch(scf::ForOp forOp, const PrefetchInfo &info) {
    Location loc = forOp.getLoc();
    // Insert before the terminator (yield) of the for body.
    // Use the iterator right before the terminator.
    Block *body = forOp.getBody();
    auto termIt = body->getTerminator()->getIterator();
    OpBuilder builder(body, termIt);

    Value iv = forOp.getInductionVar();

    // j_next = iv + 1
    Value c1 = arith::ConstantIntOp::create(builder, loc, 1, 32);
    Value jNext = arith::AddIOp::create(builder, loc, iv, c1);

    // Build the prefetch chain directly (no scf.if — the OOB load on
    // the last iteration is harmless since its result is unused)
    Value snScalar =
        arith::MulIOp::create(builder, loc, jNext, info.tileSize);
    Value snSplat =
        triton::SplatOp::create(builder, loc, info.btLoadType, snScalar);
    Value sn =
        arith::AddIOp::create(builder, loc, snSplat, info.offsM);
    Value pn =
        arith::DivSIOp::create(builder, loc, sn, info.pageSize);
    Value bts =
        triton::SplatOp::create(builder, loc, info.btSplatPtrType,
                                 info.btScalarPtr);
    Value ptrN =
        triton::AddPtrOp::create(builder, loc, info.btSplatPtrType, bts, pn);

    // Issue the prefetch load — result unused, cache warming only
    triton::LoadOp::create(builder, loc, info.btLoadType, ptrN,
                            Value(), Value(),
                            triton::CacheModifier::CA,
                            triton::EvictionPolicy::NORMAL, false);

  }
};

} // anonymous namespace
} // namespace mlir::triton
