//===- ResourceHints.cpp - PACT NanoFlow Resource Hints -------------------===//
//
// P10: Aggregates PACT pipeline hints into kernel-level resource estimates
//      (SMEM, regs, occupancy, memory/compute-bound) as a module attribute.
//      Default OFF.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <string>
#include <algorithm>

#define DEBUG_TYPE "pact-resource"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::triton::gpu {

#define GEN_PASS_DEF_PACTRESOURCEHINTS
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

namespace {

static bool isEnabled() {
  const char *env = std::getenv("PACT_ENABLE_RESOURCE_HINTS");
  if (!env) return false; // Default OFF
  return std::string(env) == "1";
}

struct PACTResourceHintsPass
    : public impl::PACTResourceHintsBase<PACTResourceHintsPass> {

  void runOnOperation() override {
    if (!isEnabled()) return;

    ModuleOp mod = getOperation();
    auto ctx = &getContext();

    int64_t totalSMEM = 0;
    int maxStages = 0;
    int totalRegs = 0;
    int64_t totalTileBytes = 0;
    bool hasAsyncCopy = false;
    int numPagedLoads = 0;
    int estimatedIterations = 0;

    mod.walk([&](triton::LoadOp loadOp) {
      if (!loadOp->hasAttr("pact.paged_load"))
        return WalkResult::advance();

      numPagedLoads++;

      if (auto attr = loadOp->getAttrOfType<mlir::IntegerAttr>(
              "pact.hint.tile_bytes")) {
        totalTileBytes += attr.getInt();
      }
      if (auto attr = loadOp->getAttrOfType<mlir::IntegerAttr>(
              "pact.hint.suggested_num_stages")) {
        maxStages = std::max(maxStages, (int)attr.getInt());
      }
      if (auto attr = loadOp->getAttrOfType<mlir::IntegerAttr>(
              "pact.hint.estimated_iterations")) {
        estimatedIterations = std::max(estimatedIterations,
                                        (int)attr.getInt());
      }

      int64_t elements = 1;
      auto resultTy = cast<RankedTensorType>(loadOp.getResult().getType());
      for (auto dim : resultTy.getShape()) elements *= dim;
      totalRegs += elements / 4; // rough: ~4 f16 elements/reg

      if (auto attr = loadOp->getAttrOfType<mlir::IntegerAttr>(
              "pact.hint.safe_async_copy_width")) {
        if (attr.getInt() >= 4) hasAsyncCopy = true;
      }

      return WalkResult::advance();
    });

    if (numPagedLoads == 0) return;

    totalSMEM = totalTileBytes * std::max(2, maxStages);

    int numWarps = 4;
    if (auto attr = mod->getAttrOfType<mlir::IntegerAttr>("ttg.num-warps"))
      numWarps = attr.getInt();
    int numThreads = numWarps * 32;
    int regsPerThread = totalRegs / std::max(1, numThreads);
    int maxWarpsByRegs = 65536 / (regsPerThread * 32);
    maxWarpsByRegs = std::max(1, std::min(maxWarpsByRegs, 48));
    double occupancy = (double)maxWarpsByRegs / 48.0;

    double bytesPerIter = numPagedLoads > 0
        ? (double)totalTileBytes / numPagedLoads : 0;
    bool isMemoryBound = (bytesPerIter >= 256) || (estimatedIterations < 64);

    std::string hintJson = "{";
    hintJson += "\"cp_async\": " + std::string(hasAsyncCopy ? "true" : "false") + ", ";
    hintJson += "\"num_paged_loads\": " + std::to_string(numPagedLoads) + ", ";
    hintJson += "\"num_stages\": " + std::to_string(maxStages) + ", ";
    hintJson += "\"est_smem_bytes\": " + std::to_string(totalSMEM) + ", ";
    hintJson += "\"est_regs_per_thread\": " + std::to_string(regsPerThread) + ", ";
    hintJson += "\"est_occupancy\": " + std::to_string(occupancy).substr(0, 4) + ", ";
    hintJson += "\"est_iterations\": " + std::to_string(estimatedIterations) + ", ";
    hintJson += "\"est_tile_bytes\": " + std::to_string(totalTileBytes) + ", ";
    hintJson += "\"memory_bound\": " + std::string(isMemoryBound ? "true" : "false") + ", ";
    hintJson += "\"fill_drain_cost_iter\": " + std::to_string(maxStages * 2);
    hintJson += "}";

    mod->setAttr("pact.resource_hints", mlir::StringAttr::get(ctx, hintJson));

    llvm::errs() << "[PACT P10] Resource Hints: " << hintJson << "\n";
  }
};

} // anonymous namespace
} // namespace mlir::triton::gpu
