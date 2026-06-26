//===- PageTransform.cpp - PACT Page Transform Pass -----------------------===//
//
// PACT PageTransform pass: recognizes paged KV cache access patterns in TTIR.
// Stage A: semantic recognition — adds pact.* attributes to matching loads.
// Stage B: safe canonicalization — div→shift, mod→and, conservative hoisting.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"

#include "llvm/Support/raw_ostream.h"

namespace mlir::triton {

#define GEN_PASS_DEF_TRITONPAGETRANSFORM
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

namespace {

struct PageTransformPass
    : public impl::TritonPageTransformBase<PageTransformPass> {

  void runOnOperation() override {
    // Stage A: Semantic recognition for paged KV cache access patterns.
    // Reads pact.* attributes set by the JIT frontend annotation.
    ModuleOp mod = getOperation();

    // Check if any function is marked as paged attention
    bool hasPagedFunc = false;
    int64_t pageSize = 0;
    mod.walk([&](Operation *op) {
      if (op->hasAttr("pact.paged")) {
        hasPagedFunc = true;
        if (auto ps = op->getAttrOfType<IntegerAttr>("pact.page_size"))
          pageSize = ps.getInt();
      }
    });
    if (!hasPagedFunc)
      return;  // No-op: no paged functions to transform

    // TODO: Implement Stage A pattern matching
    // 1. Walk tt.load ops inside scf.for loops
    // 2. Trace pointer chains to find block_table-driven paged access
    // 3. Annotate matching loads with pact.paged_load / pact.block_table_lookup
  }
};

} // anonymous namespace
} // namespace mlir::triton
