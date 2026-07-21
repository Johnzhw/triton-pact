//===- AddressStrengthReduce.cpp - PACT Address Strength Reduction -------===//
//
// PACT Address Strength Reduction pass: Simplifies per-tile address
// computation in paged attention tile loops by recognizing algebraic
// identities when PAGE_SIZE and TILE_SIZE are compile-time constants.
//
// Key optimization (PAGE_SIZE == TILE_SIZE):
//   Inside:  page_off = (j * TILE_SIZE + offs_t) % PAGE_SIZE
//   Simplifies to: offs_t  (all elements of offs_t are < PAGE_SIZE)
//
// This eliminates the page-offset computation chain (remsi → expand →
// extsi → mul → broadcast) from the loop body, reducing per-iteration
// address ops.
//
// For PAGE_SIZE > TILE_SIZE (future):
//   page_off_next = page_off_prev + TILE_SIZE
//   if page_off_next >= PAGE_SIZE: reload block_table, reset page_off
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Visitors.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace mlir {
namespace triton {
namespace gpu {

#define GEN_PASS_DEF_TRITONGPUADDRESSSTRENGTHREDUCE
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Helper: get compile-time integer from Value (through splat/broadcast).
//===----------------------------------------------------------------------===//
static FailureOr<int64_t> getConstantInt(Value val) {
  // Direct scalar constant
  if (auto constOp = val.getDefiningOp<arith::ConstantOp>()) {
    if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue()))
      return intAttr.getInt();
    // Dense tensor constant (e.g., dense<16>)
    if (auto denseAttr = dyn_cast<DenseIntElementsAttr>(constOp.getValue())) {
      if (denseAttr.isSplat())
        return denseAttr.getSplatValue<APInt>().getSExtValue();
    }
  }
  // Through splat
  if (auto splatOp = val.getDefiningOp<triton::SplatOp>()) {
    return getConstantInt(splatOp.getSrc());
  }
  return failure();
}

//===----------------------------------------------------------------------===//
// Helper: check if a value is (or derives from) a tt.make_range.
//===----------------------------------------------------------------------===//
static triton::MakeRangeOp getMakeRangeSource(Value val) {
  if (auto mrOp = val.getDefiningOp<triton::MakeRangeOp>())
    return mrOp;
  // Look through extsi, expand_dims, splat, broadcast
  if (auto defOp = val.getDefiningOp()) {
    auto name = defOp->getName().getStringRef();
    if (name == "arith.extsi" || name == "arith.extui" ||
        name == "tt.expand_dims" || name == "tt.broadcast") {
      return getMakeRangeSource(defOp->getOperand(0));
    }
  }
  return nullptr;
}

//===----------------------------------------------------------------------===//
// Helper: get PAGE_SIZE from the module attribute.
//===----------------------------------------------------------------------===//
static FailureOr<int64_t> getPageSize(ModuleOp mod) {
  if (auto attr = mod->getAttrOfType<IntegerAttr>("pact.page_size"))
    return attr.getInt();
  return failure();
}

//===----------------------------------------------------------------------===//
// Helper: detect TILE_SIZE by looking for arith.muli(j, constant).
//===----------------------------------------------------------------------===//
static FailureOr<int64_t> detectTileSize(scf::ForOp forOp) {
  Value iv = forOp.getInductionVar();
  for (auto *user : iv.getUsers()) {
    if (auto mulOp = dyn_cast<arith::MulIOp>(user)) {
      auto tileSize = getConstantInt(mulOp.getRhs());
      if (succeeded(tileSize))
        return *tileSize;
      tileSize = getConstantInt(mulOp.getLhs());
      if (succeeded(tileSize))
        return *tileSize;
    }
  }
  return failure();
}

//===----------------------------------------------------------------------===//
// Core optimization: simplify arith.remsi(seq_offset, PAGE_SIZE) → offs_t
// when TILE_SIZE == PAGE_SIZE.
//
// Pattern:
//   %seq_offset_scalar = arith.muli %j, %cTileSize   // j * TILE_SIZE
//   %seq_offset_tensor = tt.splat(%seq_offset_scalar) + offs_t  // + [0..15]
//   %page_off = arith.remsi %seq_offset_tensor, %cPAGE_SIZE
//
// When TILE_SIZE == PAGE_SIZE:
//   (j*PAGE_SIZE + offs_t) % PAGE_SIZE == offs_t
//
// We replace %page_off with offs_t (or remsi(offs_t, PAGE_SIZE) for safety).
//===----------------------------------------------------------------------===//
static int simplifyPageOffsetRemSI(scf::ForOp forOp, int64_t pageSize,
                                    int64_t tileSize) {
  if (pageSize != tileSize)
    return 0; // algebraic simplification only when sizes match

  Block *body = forOp.getBody();
  int simplified = 0;

  // Find all arith.remsi ops in the loop body
  SmallVector<arith::RemSIOp> remsiOps;
  forOp.walk([&](arith::RemSIOp op) {
    if (op->getBlock() == body)
      remsiOps.push_back(op);
  });

  for (arith::RemSIOp remsiOp : remsiOps) {
    Value rhs = remsiOp.getRhs();
    auto rhsConst = getConstantInt(rhs);
    if (failed(rhsConst) || *rhsConst != pageSize)
      continue;

    // Now check the LHS: should be seq_offset_tensor = addi(splat(muli(j, TILE)), offs_t)
    Value lhs = remsiOp.getLhs();
    auto addiOp = lhs.getDefiningOp<arith::AddIOp>();
    if (!addiOp)
      continue;

    // Find the offs_t operand of the addi
    Value offsOperand;
    bool hasIVDep = false;
    for (Value operand : addiOp.getOperands()) {
      // Check if this operand traces to a make_range (offs_t)
      if (getMakeRangeSource(operand)) {
        offsOperand = operand;
        continue;
      }
      // Check if this operand traces to muli(j, TILE_SIZE)
      Value src = operand;
      if (auto splatOp = src.getDefiningOp<triton::SplatOp>())
        src = splatOp.getSrc();
      if (auto mulOp = src.getDefiningOp<arith::MulIOp>()) {
        if (mulOp.getLhs() == forOp.getInductionVar() ||
            mulOp.getRhs() == forOp.getInductionVar()) {
          hasIVDep = true;
          continue;
        }
      }
      if (src == forOp.getInductionVar()) {
        hasIVDep = true;
        continue;
      }
    }

    if (!offsOperand || !hasIVDep)
      continue;

    // ── Found the pattern! ──────────────────────────────────────────
    // offsOperand is the loop-invariant offs_t.
    // remsi(offsOperand, pageSize) == offsOperand since all values < pageSize.
    // Replace remsiOp result with offsOperand.

    llvm::errs() << "[PACT ASR] Simplifying page_offset remsi"
                 << " (page_size=" << pageSize
                 << ", tile_size=" << tileSize << ")\n";

    // We need the replacement to have the same type as the remsi result.
    // offsOperand might need a cast (extsi, index_cast, etc.)
    Value replacement = offsOperand;
    Type resultType = remsiOp.getResult().getType();

    // If offsOperand type doesn't match, add cast
    if (replacement.getType() != resultType) {
      OpBuilder builder(remsiOp);
      // Check if we need extsi
      if (isa<IntegerType>(resultType) && isa<IntegerType>(replacement.getType())) {
        auto fromInt = cast<IntegerType>(replacement.getType());
        auto toInt = cast<IntegerType>(resultType);
        if (fromInt.getWidth() < toInt.getWidth()) {
          replacement = arith::ExtSIOp::create(
              builder, remsiOp.getLoc(), resultType, replacement);
        } else if (fromInt.getWidth() > toInt.getWidth()) {
          replacement = arith::TruncIOp::create(
              builder, remsiOp.getLoc(), resultType, replacement);
        }
      }
    }

    if (replacement.getType() != resultType) {
      llvm::errs() << "[PACT ASR] Type mismatch after cast, skipping\n";
      continue;
    }

    remsiOp.getResult().replaceAllUsesWith(replacement);
    simplified++;
  }

  // ── Cleanup: erase dead remsi ops ─────────────────────────────────
  for (arith::RemSIOp remsiOp : remsiOps) {
    if (remsiOp.getResult().use_empty())
      remsiOp.erase();
  }

  return simplified;
}

//===----------------------------------------------------------------------===//
// Simplify arith.divsi(seq_offset, PAGE_SIZE) → j when TILE_SIZE == PAGE_SIZE
//
// Pattern:
//   %page_idx_tensor = arith.divsi %seq_offset_tensor, %cPAGE_SIZE
//   where seq_offset_tensor = splat(j*TILE_SIZE) + offs_t
//
// When TILE_SIZE == PAGE_SIZE:
//   (j*PAGE_SIZE + offs_t) / PAGE_SIZE == j + offs_t/PAGE_SIZE == j
//
// All elements in the tensor are equal to j, making the divsi redundant.
//===----------------------------------------------------------------------===//
static int simplifyPageIndexDivSI(scf::ForOp forOp, int64_t pageSize,
                                   int64_t tileSize) {
  if (pageSize != tileSize)
    return 0;

  Block *body = forOp.getBody();
  Value iv = forOp.getInductionVar();
  int simplified = 0;

  SmallVector<arith::DivSIOp> divsiOps;
  forOp.walk([&](arith::DivSIOp op) {
    if (op->getBlock() == body)
      divsiOps.push_back(op);
  });

  for (arith::DivSIOp divsiOp : divsiOps) {
    auto rhsConst = getConstantInt(divsiOp.getRhs());
    if (failed(rhsConst) || *rhsConst != pageSize)
      continue;

    // LHS should be addi(splat(j*TILE), offs_t)
    Value lhs = divsiOp.getLhs();
    auto addiOp = lhs.getDefiningOp<arith::AddIOp>();
    if (!addiOp)
      continue;

    Value offsOperand;
    bool hasIVDep = false;
    for (Value operand : addiOp.getOperands()) {
      if (getMakeRangeSource(operand)) {
        offsOperand = operand;
        continue;
      }
      Value src = operand;
      if (auto splatOp = src.getDefiningOp<triton::SplatOp>())
        src = splatOp.getSrc();
      if (auto mulOp = src.getDefiningOp<arith::MulIOp>()) {
        if (mulOp.getLhs() == iv || mulOp.getRhs() == iv) {
          hasIVDep = true;
          continue;
        }
      }
    }

    if (!offsOperand || !hasIVDep)
      continue;

    // ── Pattern matched ─────────────────────────────────────────────
    // divsi(j*PAGE_SIZE + offs_t, PAGE_SIZE) == j
    // Create a splat of j to replace the divsi result.
    llvm::errs() << "[PACT ASR] Simplifying page_index divsi → splat(j)\n";

    OpBuilder builder(divsiOp);
    Type resultType = divsiOp.getResult().getType();
    auto tensorType = dyn_cast<RankedTensorType>(resultType);
    if (!tensorType) continue;

    // Create splat(j)
    Value splatJV = triton::SplatOp::create(builder, divsiOp.getLoc(), resultType, iv);

    // Check type compatibility
    Type ivType = iv.getType();
    Type elemType = tensorType.getElementType();
    if (ivType != elemType) {
      if (isa<IntegerType>(ivType) && isa<IntegerType>(elemType)) {
        auto fromInt = cast<IntegerType>(ivType);
        auto toInt = cast<IntegerType>(elemType);
        Value castJV;
        if (fromInt.getWidth() < toInt.getWidth()) {
          castJV = arith::ExtSIOp::create(builder, divsiOp.getLoc(), elemType, iv);
        } else {
          castJV = arith::TruncIOp::create(builder, divsiOp.getLoc(), elemType, iv);
        }
        splatJV = triton::SplatOp::create(builder, divsiOp.getLoc(), resultType, castJV);
      }
    }

    divsiOp.getResult().replaceAllUsesWith(splatJV);
    simplified++;
  }

  for (arith::DivSIOp divsiOp : divsiOps) {
    if (divsiOp.getResult().use_empty())
      divsiOp.erase();
  }

  return simplified;
}

//===----------------------------------------------------------------------===//
// Pass definition
//===----------------------------------------------------------------------===//
struct AddressStrengthReducePass
    : public impl::TritonGPUAddressStrengthReduceBase<AddressStrengthReducePass> {

  void runOnOperation() override {
    ModuleOp mod = getOperation();

    if (!mod->hasAttr("pact.paged")) {
      llvm::errs() << "[PACT ASR] Module is not paged, skipping\n";
      return;
    }

    auto pageSizeResult = getPageSize(mod);
    if (failed(pageSizeResult)) {
      llvm::errs() << "[PACT ASR] PAGE_SIZE is not a compile-time constant,"
                   << " skipping\n";
      return;
    }
    int64_t pageSize = *pageSizeResult;

    int totalSimplified = 0;
    int loopsProcessed = 0;

    mod.walk([&](scf::ForOp forOp) {
      if (!forOp->hasAttr("pact.prefetch_eligible"))
        return WalkResult::advance();

      auto tileSizeResult = detectTileSize(forOp);
      if (failed(tileSizeResult)) {
        llvm::errs() << "[PACT ASR] Could not detect TILE_SIZE, skipping\n";
        return WalkResult::advance();
      }
      int64_t tileSize = *tileSizeResult;

      if (tileSize > pageSize) {
        llvm::errs() << "[PACT ASR] TILE_SIZE=" << tileSize
                     << " > PAGE_SIZE=" << pageSize
                     << ", skipping\n";
        return WalkResult::advance();
      }

      int remSimplified = simplifyPageOffsetRemSI(forOp, pageSize, tileSize);
      // NOTE: divsi simplification is disabled — replacing divsi with
      // splat(j) introduces encoding mismatches that degrade performance.
      // int divSimplified = simplifyPageIndexDivSI(forOp, pageSize, tileSize);
      int simplified = remSimplified; // + divSimplified;

      if (simplified > 0) {
        totalSimplified += simplified;
        loopsProcessed++;
        llvm::errs() << "[PACT ASR] Simplified " << simplified
                     << " arithmetic ops (page_size=" << pageSize
                     << ", tile_size=" << tileSize << ")\n";
      } else {
        llvm::errs() << "[PACT ASR] Pattern not matched for this loop"
                     << " (page_size=" << pageSize
                     << ", tile_size=" << tileSize << ")\n";
      }

      return WalkResult::advance();
    });

    if (loopsProcessed > 0) {
      llvm::errs() << "[PACT ASR] Total simplified: " << totalSimplified
                   << " ops across " << loopsProcessed << " loop(s)\n";
    } else {
      llvm::errs() << "[PACT ASR] No tile loops matched"
                   << " (fallback: AxisInfo only, no crash)\n";
    }
  }
};

} // anonymous namespace
} // namespace gpu
} // namespace triton
} // namespace mlir
