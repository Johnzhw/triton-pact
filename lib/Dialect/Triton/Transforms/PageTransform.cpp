//===- PageTransform.cpp - PACT Page Transform Pass -----------------------===//
//
// PACT PageTransform pass: recognizes paged KV cache access patterns in TTIR.
// Stage A: semantic recognition — adds pact.* attributes to matching loads.
// Stage B: safe canonicalization — div→shift, mod→and, conservative hoisting.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::triton {

#define GEN_PASS_DEF_TRITONPAGETRANSFORM
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Helper: trace through tt.addptr chain to find the base pointer and collect
// all offset operands.
//===----------------------------------------------------------------------===//
static Value traceToBasePointer(Value ptr,
                                 SmallVectorImpl<Value> &offsetChain) {
  while (true) {
    auto *defOp = ptr.getDefiningOp();
    if (!defOp)
      break;
    // tt.addptr(base, offset) → base is the ptr, offset is the index
    if (defOp->getName().getStringRef() == "tt.addptr") {
      offsetChain.push_back(defOp->getOperand(1)); // the offset/idx
      ptr = defOp->getOperand(0);                   // the base pointer
      continue;
    }
    // arith ops (extsi, etc.) — skip through
    if (defOp->getNumResults() == 1 && defOp->getNumOperands() == 1) {
      auto name = defOp->getName().getStringRef();
      if (name.starts_with("arith.ext") || name == "arith.index_cast") {
        ptr = defOp->getOperand(0);
        continue;
      }
    }
    break;
  }
  return ptr;
}

//===----------------------------------------------------------------------===//
// Helper: check if a value depends on a specific defining op
//===----------------------------------------------------------------------===//
static bool dependsOn(Value val, Operation *target, int maxDepth = 8) {
  if (maxDepth <= 0)
    return false;
  auto *defOp = val.getDefiningOp();
  if (!defOp)
    return false;
  if (defOp == target)
    return true;
  for (auto operand : defOp->getOperands()) {
    if (operand == val) // skip self
      continue;
    if (dependsOn(operand, target, maxDepth - 1))
      return true;
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Helper: extract a constant integer value from an MLIR attribute, supporting
// both scalar IntegerAttr and splat DenseElementsAttr.
//===----------------------------------------------------------------------===//
static std::optional<int64_t> extractConstantInt(Attribute attr) {
  if (auto intAttr = dyn_cast<IntegerAttr>(attr))
    return intAttr.getInt();
  if (auto denseAttr = dyn_cast<DenseIntElementsAttr>(attr)) {
    if (denseAttr.isSplat())
      return denseAttr.getSplatValue<APInt>().getSExtValue();
  }
  if (auto denseFPAttr = dyn_cast<DenseFPElementsAttr>(attr)) {
    if (denseFPAttr.isSplat())
      return static_cast<int64_t>(denseFPAttr.getSplatValue<APFloat>()
                                      .convertToDouble());
  }
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// Helper: check if a value involves arith.divsi or arith.remsi by a constant
//===----------------------------------------------------------------------===//
static bool hasDivOrRemByConst(Value val, int64_t divisor, bool checkDiv,
                                bool checkRem, int maxDepth = 8) {
  if (maxDepth <= 0)
    return false;
  auto *defOp = val.getDefiningOp();
  if (!defOp)
    return false;
  auto name = defOp->getName().getStringRef();

  if (checkDiv && (name == "arith.divsi" || name == "arith.floordivsi")) {
    if (auto constOp =
            defOp->getOperand(1).getDefiningOp<arith::ConstantOp>()) {
      auto val = extractConstantInt(constOp.getValue());
      if (val && *val == divisor)
        return true;
    }
  }

  if (checkRem && name == "arith.remsi") {
    if (auto constOp =
            defOp->getOperand(1).getDefiningOp<arith::ConstantOp>()) {
      auto val = extractConstantInt(constOp.getValue());
      if (val && *val == divisor)
        return true;
    }
  }

  // Recurse into operands (skip constants)
  for (auto operand : defOp->getOperands()) {
    if (operand.getDefiningOp<arith::ConstantOp>())
      continue;
    if (hasDivOrRemByConst(operand, divisor, checkDiv, checkRem, maxDepth - 1))
      return true;
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Stage A: Semantic Recognition
//===----------------------------------------------------------------------===//
struct PageTransformPass
    : public impl::TritonPageTransformBase<PageTransformPass> {

  void runOnOperation() override {
    ModuleOp mod = getOperation();

    // Check module-level and function-level pact.paged attributes
    int64_t pageSize = 0;
    bool hasPagedFunc = false;

    // Check module op itself
    if (mod->hasAttr("pact.paged")) {
      hasPagedFunc = true;
      if (auto ps = mod->getAttrOfType<IntegerAttr>("pact.page_size"))
        pageSize = ps.getInt();
    }

    // Check nested operations (function ops)
    mod.walk([&](Operation *op) {
      if (op->hasAttr("pact.paged")) {
        hasPagedFunc = true;
        if (auto ps = op->getAttrOfType<IntegerAttr>("pact.page_size"))
          pageSize = ps.getInt();
      }
    });
    if (!hasPagedFunc)
      return; // No-op fallback

    if (pageSize <= 0)
      return; // Invalid page size

    // Counters
    int numBlockTableLoads = 0;
    int numKVLoadsAnnotated = 0;

    // Walk all scf.for loops in paged functions
    mod.walk([&](scf::ForOp forOp) {
      // Check parent function for pact.paged (try both module and func attrs)
      auto *parentOp = forOp->getParentOp();
      bool isPaged = mod->hasAttr("pact.paged");
      while (parentOp && !isPaged) {
        isPaged = parentOp->hasAttr("pact.paged");
        parentOp = parentOp->getParentOp();
      }
      if (!isPaged)
        return WalkResult::advance();

      // Phase 1: Identify block table loads inside this loop
      // A block table load is a tt.load whose pointer traces to
      // block_tables_ptr (by name) and whose index involves arith.divsi
      // by pageSize.
      SmallVector<Operation *> blockTableLoads;
      int numLoadOps = 0, numTtLoadOps = 0, numWithDiv = 0, numWithOffsetChain = 0;
      int numPhase2Loads = 0, numPhase2DependsBT = 0, numPhase2HasRem = 0;
      forOp.walk([&](Operation *op) {
        numLoadOps++;
        if (op->getName().getStringRef() != "tt.load")
          return WalkResult::advance();

        numTtLoadOps++;
        auto loadOp = op;
        Value ptr = loadOp->getOperand(0);

        // Trace pointer chain to find base
        SmallVector<Value, 8> offsetChain;
        Value base = traceToBasePointer(ptr, offsetChain);
        numWithOffsetChain += (offsetChain.size() > 0) ? 1 : 0;

        // The paged access is identified by the presence of divsi by
        // PAGE_SIZE in the offset chain, not by the base pointer name.
        // The base pointer name heuristic is unreliable in optimized IR.
        // Fallback: check if any offset involves divsi by pageSize
        bool hasDivByPageSize = false;
        for (auto off : offsetChain) {
          if (hasDivOrRemByConst(off, pageSize, /*checkDiv=*/true,
                                 /*checkRem=*/false)) {
            hasDivByPageSize = true;
            break;
          }
        }

        if (hasDivByPageSize) {
          numWithDiv++;
          // This is likely a block table lookup
          loadOp->setAttr("pact.block_table_lookup",
                          UnitAttr::get(&getContext()));
          blockTableLoads.push_back(loadOp);
          numBlockTableLoads++;
        }

        return WalkResult::advance();
      });

      // If no block table loads found, skip prefetch annotation
      if (blockTableLoads.empty())
        return WalkResult::advance();

      // Mark the function as having paged access
      auto *pagedFuncOp = forOp->getParentOp();
      pagedFuncOp->setAttr("pact.has_paged_access",
                      UnitAttr::get(&getContext()));

      // Phase 2: Identify K/V loads.
      // Heuristic: inside the same scf.for loop as the block table lookup,
      // any tt.load (not already annotated as block_table_lookup) whose
      // offset chain involves arith.remsi by pageSize is a paged KV load.
      forOp.walk([&](Operation *op) {
        numPhase2Loads++;
        if (op->getName().getStringRef() != "tt.load")
          return WalkResult::advance();
        // Skip block table loads themselves
        if (op->hasAttr("pact.block_table_lookup"))
          return WalkResult::advance();

        auto loadOp = op;
        Value ptr = loadOp->getOperand(0);

        // Trace pointer chain to find offsets involving remsi by pageSize
        SmallVector<Value, 8> offsetChain;
        Value base = traceToBasePointer(ptr, offsetChain);
        bool hasRemByPageSize = false;
        for (auto off : offsetChain) {
          if (hasDivOrRemByConst(off, pageSize, /*checkDiv=*/false,
                                 /*checkRem=*/true)) {
            hasRemByPageSize = true;
            break;
          }
        }

        if (!hasRemByPageSize)
          return WalkResult::advance();

        numPhase2HasRem++;

        // Annotate as paged KV load
        loadOp->setAttr("pact.paged_load", UnitAttr::get(&getContext()));
        numKVLoadsAnnotated++;

        return WalkResult::advance();
      });

      return WalkResult::advance();
    });

    // Summary (only log when matches found)
    if (numBlockTableLoads > 0) {
      llvm::errs() << "[PACT PageTransform] Recognized " << numBlockTableLoads
                   << " block_table lookup(s) and " << numKVLoadsAnnotated
                   << " paged KV load(s) (pageSize=" << pageSize << ")\n";
    }
  }
};

} // anonymous namespace
} // namespace mlir::triton
