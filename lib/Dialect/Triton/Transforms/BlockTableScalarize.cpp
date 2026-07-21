//===- BlockTableScalarize.cpp - PACT Block Table Scalarization ----------===//
//
// Converts uniform block_table tensor gather loads to scalar load + splat.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::triton {

#define GEN_PASS_DEF_TRITONBLOCKTABLESCALARIZE
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

namespace {

static std::optional<int64_t> getConstInt(Value val) {
  if (auto constOp = val.getDefiningOp<arith::ConstantOp>()) {
    if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue()))
      return intAttr.getInt();
    if (auto denseAttr = dyn_cast<DenseIntElementsAttr>(constOp.getValue()))
      if (denseAttr.isSplat())
        return denseAttr.getSplatValue<APInt>().getSExtValue();
  }
  return std::nullopt;
}

/// Walk through splat, ext, cast to find canonical source.
static Value skipConversions(Value val) {
  while (true) {
    auto *defOp = val.getDefiningOp();
    if (!defOp) break;
    auto name = defOp->getName().getStringRef();
    if (name == "tt.splat" || name == "tt.broadcast" ||
        name == "tt.expand_dims" || name == "arith.extsi" ||
        name == "arith.extui" || name == "arith.trunci" ||
        name == "arith.index_cast") {
      val = defOp->getOperand(0);
      continue;
    }
    break;
  }
  return val;
}

struct BlockTableScalarizePass
    : public impl::TritonBlockTableScalarizeBase<BlockTableScalarizePass> {

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    if (!mod->hasAttr("pact.paged")) return;

    auto pageSizeAttr = mod->getAttrOfType<IntegerAttr>("pact.page_size");
    if (!pageSizeAttr) return;
    int64_t pageSize = pageSizeAttr.getInt();

    int numScalarized = 0, numSkipped = 0;
    MLIRContext *ctx = &getContext();

    SmallVector<scf::ForOp> loops;
    mod.walk([&](scf::ForOp forOp) {
      auto *p = forOp->getParentOp();
      while (p) {
        if (p->hasAttr("pact.has_paged_access")) {
          loops.push_back(forOp); break;
        }
        p = p->getParentOp();
      }
    });

    for (scf::ForOp forOp : loops) {
      Value iv = forOp.getInductionVar();
      Location loc = forOp.getLoc();

      SmallVector<triton::LoadOp> btLoads;
      forOp.walk([&](triton::LoadOp op) {
        if (op->hasAttr("pact.block_table_lookup"))
          btLoads.push_back(op);
      });
      if (btLoads.empty()) continue;

      for (triton::LoadOp btLoad : btLoads) {
        // ── Analyze: load(addptr(splat(base), divsi(splat(j*TILE)+offs, PAGE))) ─
        Value ptr = btLoad.getPtr();
        auto addPtrOp = ptr.getDefiningOp<triton::AddPtrOp>();
        if (!addPtrOp) { numSkipped++; continue; }

        Value pageIdx = addPtrOp.getOperand(1);
        Value pageIdxSrc = skipConversions(pageIdx);
        auto divOp = pageIdxSrc.getDefiningOp<arith::DivSIOp>();
        if (!divOp) { numSkipped++; continue; }

        auto divisor = getConstInt(divOp.getRhs());
        if (!divisor || *divisor != pageSize) { numSkipped++; continue; }

        // LHS = addi(splat(j*TILE), offs_t)
        Value lhs = skipConversions(divOp.getLhs());
        auto addiOp = lhs.getDefiningOp<arith::AddIOp>();
        if (!addiOp) { numSkipped++; continue; }

        // Find j*TILE part and offs_t part
        Value jMulPart, offsPart;
        int64_t tileSize = 0;
        for (Value operand : addiOp->getOperands()) {
          Value src = skipConversions(operand);
          if (auto mulOp = src.getDefiningOp<arith::MulIOp>()) {
            if (mulOp.getLhs() == iv || mulOp.getRhs() == iv) {
              auto ts = getConstInt(mulOp.getRhs());
              if (!ts) ts = getConstInt(mulOp.getLhs());
              if (ts && *ts > 0) {
                tileSize = *ts;
                jMulPart = operand;
                continue;
              }
            }
          }
          if (src.getDefiningOp<triton::MakeRangeOp>())
            offsPart = operand;
        }
        if (tileSize == 0) { numSkipped++; continue; }

        // Skip if TILE > PAGE (non-uniform)
        if (tileSize > pageSize) { numSkipped++; continue; }

        llvm::errs() << "[PACT BTScalarize] Scalarizing:"
                     << " page=" << pageSize << " tile=" << tileSize << "\n";

        // ── Rewrite ──────────────────────────────────────────────────
        ImplicitLocOpBuilder b(loc, OpBuilder(btLoad));
        b.setInsertionPoint(btLoad);

        // Get base pointer from the splat
        Value splatBase = addPtrOp.getOperand(0);
        Value btBase = splatBase.getDefiningOp<triton::SplatOp>()->getOperand(0);

        // Scalar index: j * TILE / PAGE
        Type ivTy = iv.getType();
        auto ivBits = cast<IntegerType>(ivTy).getWidth();
        Value cTile = arith::ConstantIntOp::create(b, loc, tileSize, ivBits);
        Value cPage = arith::ConstantIntOp::create(b, loc, pageSize, ivBits);
        Value scalarMul = arith::MulIOp::create(b, loc, iv, cTile);
        Value scalarIdx = arith::DivSIOp::create(b, loc, scalarMul, cPage);

        // Scalar pointer: addptr(btBase, idx)
        Value scalarIdxI64 = scalarIdx;
        if (ivBits < 64)
          scalarIdxI64 = arith::ExtSIOp::create(b, loc, b.getI64Type(), scalarIdx);
        Value scalarPtr = triton::AddPtrOp::create(b, loc,
            btBase.getType(), btBase, scalarIdxI64);

        // Scalar load
        auto scalarLoad = triton::LoadOp::create(
            b, loc, b.getI32Type(), scalarPtr,
            /*mask=*/Value(), /*other=*/Value(),
            triton::CacheModifier::NONE, triton::EvictionPolicy::NORMAL,
            /*isVolatile=*/false);
        scalarLoad->setAttr("pact.block_table_lookup_scalar", UnitAttr::get(ctx));
        scalarLoad->setAttr("pact.block_table_lookup", UnitAttr::get(ctx));

        // Splat to original tensor type
        Value splatVal = triton::SplatOp::create(b, loc,
            btLoad.getResult().getType(),
            scalarLoad.getResult());
        splatVal.getDefiningOp()->setAttr("pact.block_table_lookup",
                                          UnitAttr::get(ctx));

        // Replace and erase
        btLoad.getResult().replaceAllUsesWith(splatVal);
        btLoad.erase();
        numScalarized++;
      }
    }

    if (numScalarized > 0)
      llvm::errs() << "[PACT BTScalarize] Scalarized " << numScalarized
                   << " block_table load(s) (skipped " << numSkipped << ")\n";
  }
};

} // anonymous namespace
} // namespace mlir::triton
