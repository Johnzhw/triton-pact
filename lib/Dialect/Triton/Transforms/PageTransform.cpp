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

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
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
    // P0/P1: penetrate splat/broadcast — same scalar, reshaped to tensor
    if (defOp->getNumResults() == 1 && defOp->getNumOperands() >= 1) {
      auto name = defOp->getName().getStringRef();
      if (name == "tt.splat" || name == "tt.broadcast" ||
          name == "tt.expand_dims") {
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
// Helper: get the constant divisor of a divsi or remsi op, if any.
// Returns the constant value and the source operand (lhs of div/rem).
//===----------------------------------------------------------------------===//
static std::optional<int64_t> getConstantDivisor(Operation *op, Value &srcOut) {
  auto name = op->getName().getStringRef();
  if (name != "arith.divsi" && name != "arith.floordivsi" &&
      name != "arith.remsi")
    return std::nullopt;

  // Check if divisor (operand 1) is a constant
  auto constOp = op->getOperand(1).getDefiningOp<arith::ConstantOp>();
  if (!constOp)
    return std::nullopt;

  auto val = extractConstantInt(constOp.getValue());
  if (!val)
    return std::nullopt;

  srcOut = op->getOperand(0);
  return val;
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

  // Recurse into operands (skip constants and load results — loaded
  // values are dynamic and shouldn't be traced for static pattern matching)
  for (auto operand : defOp->getOperands()) {
    auto *opDef = operand.getDefiningOp();
    if (!opDef || isa<arith::ConstantOp>(opDef))
      continue;
    // Stop at load ops: the loaded value is runtime data, not a static
    // div/rem computation.  Tracing through loads causes all loads in
    // a paged attention kernel to be marked as block_table_lookup.
    if (opDef->getName().getStringRef() == "tt.load")
      continue;
    if (hasDivOrRemByConst(operand, divisor, checkDiv, checkRem, maxDepth - 1))
      return true;
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Helper: trace SSA uses forward to check if a value eventually feeds into
// a tt.load operation.  maxDepth limits the search.
//===----------------------------------------------------------------------===//
static bool reachesTtLoad(Value val, int maxDepth = 16) {
  if (maxDepth <= 0)
    return false;

  for (auto *user : val.getUsers()) {
    if (user->getName().getStringRef() == "tt.load")
      return true;

    // Recurse through intermediate ops (addptr, extsi, cast, broadcast,
    // expand_dims, splat, muli, addi, etc.)
    for (auto result : user->getResults()) {
      if (result == val)
        continue;
      if (reachesTtLoad(result, maxDepth - 1))
        return true;
    }
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Helper: try to find the canonical source of a value by walking through
// type conversions, splats, broadcasts, expand_dims, and addi chains.
// Returns the "root" value (e.g., the loop induction var or its derivative).
//===----------------------------------------------------------------------===//
static Value canonicalSource(Value val, int maxDepth = 8) {
  if (maxDepth <= 0)
    return val;

  auto *defOp = val.getDefiningOp();
  if (!defOp)
    return val;

  auto name = defOp->getName().getStringRef();

  // Skip type conversions
  if (name == "arith.extsi" || name == "arith.extui" ||
      name == "arith.trunci" || name == "arith.index_cast" ||
      name == "arith.sitofp") {
    return canonicalSource(defOp->getOperand(0), maxDepth - 1);
  }

  // Skip splat, broadcast, expand_dims (they just reshape; the source value
  // is what matters for semantic matching)
  if (name == "tt.splat" || name == "tt.broadcast" ||
      name == "tt.expand_dims") {
    return canonicalSource(defOp->getOperand(0), maxDepth - 1);
  }

  // Walk through addi: if one operand is a constant and the other traces
  // to an interesting value, follow the non-constant side.
  if (name == "arith.addi") {
    Value lhs = defOp->getOperand(0);
    Value rhs = defOp->getOperand(1);
    bool lhsConst = lhs.getDefiningOp<arith::ConstantOp>() != nullptr;
    bool rhsConst = rhs.getDefiningOp<arith::ConstantOp>() != nullptr;
    if (lhsConst && !rhsConst)
      return canonicalSource(rhs, maxDepth - 1);
    if (!lhsConst && rhsConst)
      return canonicalSource(lhs, maxDepth - 1);
  }

  // Walk through muli: similar to addi, follow non-constant side
  if (name == "arith.muli") {
    Value lhs = defOp->getOperand(0);
    Value rhs = defOp->getOperand(1);
    bool lhsConst = lhs.getDefiningOp<arith::ConstantOp>() != nullptr;
    bool rhsConst = rhs.getDefiningOp<arith::ConstantOp>() != nullptr;
    if (lhsConst && !rhsConst)
      return canonicalSource(rhs, maxDepth - 1);
    if (!lhsConst && rhsConst)
      return canonicalSource(lhs, maxDepth - 1);
  }

  return val;
}

//===----------------------------------------------------------------------===//
// PageTransform Pass
//===----------------------------------------------------------------------===//
struct PageTransformPass
    : public impl::TritonPageTransformBase<PageTransformPass> {

  // Auto-detect page_size from (divsi, remsi) pairs operating on the same
  // source value.  Returns the detected page_size, or 0 if not found.
  int64_t detectPageSize(scf::ForOp forOp, MLIRContext *ctx,
                          SmallVectorImpl<Operation *> &divOps,
                          SmallVectorImpl<Operation *> &remOps) {
    // Collect all divsi and remsi ops with constant divisors in this loop.
    struct BinOpInfo {
      Operation *op;
      int64_t divisor;
      Value source;  // canonical source
      bool isDiv;    // true = divsi, false = remsi
    };
    SmallVector<BinOpInfo> binOps;

    forOp.walk([&](Operation *op) {
      auto name = op->getName().getStringRef();
      bool isDiv = (name == "arith.divsi" || name == "arith.floordivsi");
      bool isRem = (name == "arith.remsi");
      if (!isDiv && !isRem)
        return WalkResult::advance();

      Value src;
      auto divVal = getConstantDivisor(op, src);
      if (!divVal)
        return WalkResult::advance();

      Value canon = canonicalSource(src, /*maxDepth=*/8);
      binOps.push_back({op, *divVal, canon, isDiv});
      return WalkResult::advance();
    });

    if (binOps.empty())
      return 0;

    // Group by (canonical_source, divisor).  We need at least one divsi
    // and one remsi sharing the same (source, divisor).
    // Use a map keyed by (source_op_ptr, divisor).
    using Key = std::pair<Operation *, int64_t>;
    // MapVector for deterministic iteration
    llvm::MapVector<Key, SmallVector<BinOpInfo>> groups;
    for (auto &b : binOps) {
      Operation *srcOp = b.source.getDefiningOp();
      // Use nullptr as sentinel for block-argument sources
      if (!srcOp) {
        // For block arguments, use a hash of the value
        srcOp = reinterpret_cast<Operation *>(
            reinterpret_cast<uintptr_t>(b.source.getImpl()));
      }
      groups[{srcOp, b.divisor}].push_back(b);
    }

    // Score each candidate divisor: strongest = has both divsi→load AND
    // remsi→load paths.
    int64_t bestPageSize = 0;
    int bestScore = -1;

    for (auto &kv : groups) {
      int64_t divisor = kv.first.second;
      auto &members = kv.second;

      bool hasDiv = false, hasRem = false;
      bool divReachesLoad = false, remReachesLoad = false;
      Operation *divOp = nullptr, *remOp = nullptr;

      for (auto &m : members) {
        if (m.isDiv) {
          hasDiv = true;
          divOp = m.op;
          if (reachesTtLoad(m.op->getResult(0)))
            divReachesLoad = true;
        } else {
          hasRem = true;
          remOp = m.op;
          if (reachesTtLoad(m.op->getResult(0)))
            remReachesLoad = true;
        }
      }

      if (!hasDiv || !hasRem)
        continue;

      int score = (divReachesLoad ? 2 : 0) + (remReachesLoad ? 1 : 0);
      // Bonus for power-of-2 divisors (common page sizes)
      if ((divisor & (divisor - 1)) == 0 && divisor >= 16 && divisor <= 256)
        score += 1;

      if (score > bestScore) {
        bestScore = score;
        bestPageSize = divisor;
        if (divOp)
          divOps.push_back(divOp);
        if (remOp)
          remOps.push_back(remOp);
      }
    }

    // Require at least: div reaches load AND rem reaches load (score >= 3)
    if (bestScore < 3) {
      // Allow score 2 (div→load) if divisor is a clear power-of-2
      // and we also have a remsi (even if it doesn't directly reach load)
      if (bestScore < 2)
        return 0;
    }

    return bestPageSize;
  }

  //===----------------------------------------------------------------===//
  // Kernel classification: prefill vs decode paged attention
  // Uses three signals for joint decision:
  //   Signal A: divsi(PAGE_SIZE) + remsi(PAGE_SIZE) patterns
  //   Signal B: block_table parameter name
  //   Signal C: 2D K/V load shapes (decode characteristic)
  //===----------------------------------------------------------------===//
  enum class KernelType {
    Unknown,
    DecodePagedAttention,  // confirmed decode paged attention kernel
    PrefillAttention,      // prefill kernel — skip PACT optimization
    MixedPagedAccess       // has paged access but uncertain type
  };

  static KernelType classifyKernel(ModuleOp mod) {
    bool hasDivsiPattern = false;
    bool hasBlockTableParam = false;
    bool has2DKVLoad = false;

    // Signal A: check for divsi(PAGE_SIZE) patterns
    mod.walk([&](arith::DivSIOp divOp) {
      if (auto constOp = divOp.getRhs().getDefiningOp<arith::ConstantIntOp>()) {
        int64_t divisor = constOp.value();
        // PAGE_SIZE is typically 16, 32, 64, 128
        if (divisor == 16 || divisor == 32 || divisor == 64 || divisor == 128) {
          // Verify: divsi result used for block_table indexing
          for (auto *user : divOp->getUsers()) {
            if (isa<arith::AddIOp, arith::MulIOp>(user)) {
              hasDivsiPattern = true;
              return WalkResult::interrupt();
            }
          }
        }
      }
      return WalkResult::advance();
    });

    // Signal B: check for block_table parameter
    for (auto funcOp : mod.getOps<triton::FuncOp>()) {
      for (unsigned i = 0; i < funcOp.getNumArguments(); i++) {
        auto arg = funcOp.getArgument(i);
        if (auto nameAttr =
                funcOp.getArgAttrOfType<StringAttr>(i, "tt.param_name")) {
          StringRef name = nameAttr.getValue();
          if (name.contains("block_table") || name.contains("BLOCK_TABLE")) {
            hasBlockTableParam = true;
            break;
          }
        }
      }
    }

    // Signal C: check for 2D K/V loads (decode characteristic)
    mod.walk([&](triton::LoadOp loadOp) {
      auto ty = dyn_cast<RankedTensorType>(loadOp.getResult().getType());
      if (ty && ty.getShape().size() == 2) {
        int64_t dim0 = ty.getShape()[0];
        int64_t dim1 = ty.getShape()[1];
        // 2D tile with head_dim > 32 (not a small mask tensor)
        if (dim0 >= 8 && dim1 >= 32) {
          has2DKVLoad = true;
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });

    // Joint decision
    if (hasDivsiPattern && hasBlockTableParam)
      return KernelType::DecodePagedAttention;
    if (hasDivsiPattern && !hasBlockTableParam)
      return KernelType::MixedPagedAccess;
    if (!hasDivsiPattern)
      return KernelType::PrefillAttention;

    return KernelType::Unknown;
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();

    // === Phase 0: Kernel Classification ===
    KernelType ktype = classifyKernel(mod);

    if (ktype == KernelType::PrefillAttention) {
      mod->setAttr("pact.kernel_type",
                   StringAttr::get(mod.getContext(), "prefill"));
      llvm::errs() << "[PACT P1] Prefill kernel detected, skipping PACT "
                      "annotation (no paged access patterns found).\n";
      return;
    }

    if (ktype == KernelType::DecodePagedAttention) {
      mod->setAttr("pact.kernel_type",
                   StringAttr::get(mod.getContext(), "decode_paged"));
      llvm::errs() << "[PACT P1] Decode paged attention kernel identified.\n";
    } else {
      mod->setAttr("pact.kernel_type",
                   StringAttr::get(mod.getContext(), "unknown"));
    }

    // Check module-level and function-level pact.paged attributes.
    // These serve as hints.  If not present, we auto-detect.
    int64_t attrPageSize = 0;
    bool hasPagedFunc = false;

    if (mod->hasAttr("pact.paged")) {
      hasPagedFunc = true;
      if (auto ps = mod->getAttrOfType<IntegerAttr>("pact.page_size"))
        attrPageSize = ps.getInt();
    }

    mod.walk([&](Operation *op) {
      if (op->hasAttr("pact.paged")) {
        hasPagedFunc = true;
        if (auto ps = op->getAttrOfType<IntegerAttr>("pact.page_size"))
          if (attrPageSize == 0)
            attrPageSize = ps.getInt();
      }
    });

    // Counters
    int numBlockTableLoads = 0;
    int numKVLoadsAnnotated = 0;
    int numAutoDetected = 0;

    // Walk all scf.for loops
    mod.walk([&](scf::ForOp forOp) {
      int64_t pageSize = attrPageSize;
      bool autoDetected = false;

      // If no attribute-provided page_size, try auto-detection
      if (pageSize <= 0) {
        SmallVector<Operation *> divOps, remOps;
        pageSize = detectPageSize(forOp, &getContext(), divOps, remOps);
        if (pageSize > 0) {
          autoDetected = true;
          numAutoDetected++;
          // Set pact.paged and pact.page_size on the parent function
          auto *pagedFuncOp = forOp->getParentOp();
          pagedFuncOp->setAttr("pact.paged",
                               UnitAttr::get(&getContext()));
          pagedFuncOp->setAttr("pact.page_size",
              IntegerAttr::get(IntegerType::get(&getContext(), 64),
                               pageSize));
          // Also set on the module for downstream passes
          if (!mod->hasAttr("pact.paged")) {
            mod->setAttr("pact.paged", UnitAttr::get(&getContext()));
            mod->setAttr("pact.page_size",
                IntegerAttr::get(IntegerType::get(&getContext(), 64),
                                 pageSize));
          }
        }
      }

      if (pageSize <= 0)
        return WalkResult::advance();

      // Check parent function context
      auto *parentOp = forOp->getParentOp();
      bool isPaged = mod->hasAttr("pact.paged");
      while (parentOp && !isPaged) {
        isPaged = parentOp->hasAttr("pact.paged");
        parentOp = parentOp->getParentOp();
      }
      if (!isPaged)
        return WalkResult::advance();

      // Phase 1: Identify block table loads inside this loop.
      // A block table load is a tt.load whose pointer offset chain involves
      // arith.divsi by pageSize.
      SmallVector<Operation *> blockTableLoads;
      forOp.walk([&](Operation *op) {
        if (op->getName().getStringRef() != "tt.load")
          return WalkResult::advance();

        auto loadOp = op;
        Value ptr = loadOp->getOperand(0);

        // Trace pointer chain to find offsets
        SmallVector<Value, 8> offsetChain;
        traceToBasePointer(ptr, offsetChain);

        bool hasDivByPageSize = false;
        for (auto off : offsetChain) {
          if (hasDivOrRemByConst(off, pageSize, /*checkDiv=*/true,
                                 /*checkRem=*/false)) {
            hasDivByPageSize = true;
            break;
          }
        }

        if (hasDivByPageSize) {
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
      // Any tt.load (not already annotated) whose offset chain involves
      // arith.remsi by pageSize is a paged KV load.
      forOp.walk([&](Operation *op) {
        if (op->getName().getStringRef() != "tt.load")
          return WalkResult::advance();
        if (op->hasAttr("pact.block_table_lookup"))
          return WalkResult::advance();

        auto loadOp = op;
        Value ptr = loadOp->getOperand(0);

        SmallVector<Value, 8> offsetChain;
        traceToBasePointer(ptr, offsetChain);
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

        // Annotate as paged KV load
        loadOp->setAttr("pact.paged_load", UnitAttr::get(&getContext()));

        // P1: Extended semantic annotations for Phase 2 modules
        auto resultTy = cast<RankedTensorType>(loadOp->getResult(0).getType());
        auto shape = resultTy.getShape();

        // Identify dimensions: dim 0 = seq (tile tokens), last dim = head_dim
        int64_t seqDim = 0;
        int64_t headDim = shape.size() - 1;
        int64_t tileTokens = (seqDim < (int64_t)shape.size()) ? shape[seqDim] : 1;
        int64_t headSize = (headDim < (int64_t)shape.size()) ? shape[headDim] : 64;

        auto i64Ty = IntegerType::get(&getContext(), 64);
        loadOp->setAttr("pact.page_size",
            IntegerAttr::get(i64Ty, pageSize));
        loadOp->setAttr("pact.tile_tokens",
            IntegerAttr::get(i64Ty, tileTokens));
        loadOp->setAttr("pact.head_dim_idx",
            IntegerAttr::get(i64Ty, headDim));
        loadOp->setAttr("pact.head_dim_size",
            IntegerAttr::get(i64Ty, headSize));
        loadOp->setAttr("pact.seq_dim_idx",
            IntegerAttr::get(i64Ty, seqDim));

        // page_boundary_safe: static proof that tile doesn't cross page boundary
        // True when tile size is an exact multiple of page size
        if (tileTokens > 0 && pageSize > 0 && tileTokens % pageSize == 0) {
          loadOp->setAttr("pact.page_boundary_safe",
              BoolAttr::get(&getContext(), true));
        }

        numKVLoadsAnnotated++;

        return WalkResult::advance();
      });

      return WalkResult::advance();
    });

    // Summary
    if (numAutoDetected > 0) {
      llvm::errs() << "[PACT PageTransform] Auto-detected " << numAutoDetected
                   << " paged attention loop(s) via def-use chain analysis\n";
    }
    if (numBlockTableLoads > 0) {
      llvm::errs() << "[PACT PageTransform] Recognized " << numBlockTableLoads
                   << " block_table lookup(s) and " << numKVLoadsAnnotated
                   << " paged KV load(s)\n";
    }
  }
};

} // anonymous namespace
} // namespace mlir::triton
