//===- PrefetchInsert.cpp - PACT Prefetch Insert Pass ---------------------===//
//
// PACT PrefetchInsert pass: converts annotated paged KV cache loads into
// async copy operations with double-buffering at the TTGIR level.
// Uses existing ttg.async_copy_global_to_local / async_commit_group /
// async_wait infrastructure. No-op fallback on missing annotations or
// unsupported hardware.
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"

namespace mlir {
namespace triton {
namespace gpu {

#define GEN_PASS_DEF_TRITONGPUPREFETCHINSERT
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

namespace {

struct PrefetchInsertPass
    : public impl::TritonGPUPrefetchInsertBase<PrefetchInsertPass> {

  void runOnOperation() override {
    // TODO: Implement paged KV cache async prefetch with double buffering.
  }
};

} // anonymous namespace
} // namespace gpu
} // namespace triton
} // namespace mlir
