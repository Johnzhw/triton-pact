

#include "triton/Dialect/TritonGPU/Transforms/CoalesceUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/LLVM.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/LinearLayout.h"
#include "triton/Tools/StrUtil.h"
#include "llvm/Support/Debug.h"

#include <cstdlib>

#define DEBUG_TYPE "tritongpu-coalesce"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::triton::gpu {
BlockedEncodingAttr
buildCoalescedEncoding(ModuleAxisInfoAnalysis &axisInfoAnalysis, Operation *op,
                       int numWarps, int threadsPerWarp,
                       triton::gpu::CGAEncodingAttr cgaLayout,
                       SmallVector<int64_t> shapePerCTA) {
  Value ptr = getMemAccessPtr(op);
  auto refTensorType = cast<RankedTensorType>(ptr.getType());

  LDBG("Considering op: " << *op);
  LLVM_DEBUG({
    DBGS() << "axis info of pointer: ";
    axisInfoAnalysis.getAxisInfo(ptr)->print(llvm::dbgs());
    llvm::dbgs() << "\n";
  });

  auto contiguity = axisInfoAnalysis.getAxisInfo(ptr)->getContiguity();
  SmallVector<unsigned> order = getOrderFromContiguity(contiguity);
  LDBG("order=[" << triton::join(order, ", ") << "]");

  auto matchesShape = [&refTensorType](const Value &val) {
    auto rttType = dyn_cast<RankedTensorType>(val.getType());
    return rttType && rttType.getShape() == refTensorType.getShape();
  };

  // The desired divisibility is the maximum divisibility among all dependent
  // pointers which have the same shape and order as `ptr`.
  llvm::SmallSetVector<Operation *, 32> memAccessesSameOrder;
  memAccessesSameOrder.insert(op);
  if (ptr.getDefiningOp()) {
    for (Operation *use : mlir::getSlice(op)) {
      Value val = getMemAccessPtr(use);
      if (!val || !matchesShape(val) || memAccessesSameOrder.contains(use))
        continue;
      auto currOrder = getOrderFromContiguity(
          axisInfoAnalysis.getAxisInfo(val)->getContiguity());
      if (order == currOrder) {
        LDBG("multi-root-slice: insert to memAccessesSameOrder " << *use);
        memAccessesSameOrder.insert(use);
      }
    }
  }

  LDBG("shapePerCTA=[" << triton::join(shapePerCTA, ", ") << "]");

  int numElems = product<int64_t>(shapePerCTA);
  int numThreads = numWarps * threadsPerWarp;

  unsigned perThread =
      getNumElementsPerThread(op, order, axisInfoAnalysis, shapePerCTA);
  LDBG("perThread for op: " << perThread);

  for (Operation *opSameOrder : memAccessesSameOrder) {
    if (opSameOrder == op)
      continue;
    unsigned currPerThread = getNumElementsPerThread(
        opSameOrder, order, axisInfoAnalysis, shapePerCTA);
    LDBG("perThread for opSameOrder: " << currPerThread);
    perThread = std::max(perThread, currPerThread);
  }

  perThread = std::min<int>(perThread, std::max(numElems / numThreads, 1));
  LDBG("perThread: " << perThread);

  // PACT M2/B1: for paged loads compute the exact sound vectorization width
  //   V = min(mem_contig, reg_contig)
  // mem_contig comes from P2's F₂ page-internal memory layout; reg_contig comes
  // from the register layout LinearLayout with head_dim made the most-minor
  // output dimension.  V is clamped to the 128-bit hardware maximum and the
  // per-CTA element budget.
  if (auto loadOp = dyn_cast<triton::LoadOp>(op)) {
    if (loadOp->hasAttr("pact.paged_load")) {
      int64_t headDimIdx = 1;
      if (auto hd = loadOp->getAttrOfType<IntegerAttr>(
              "pact.head_dim_idx"))
        headDimIdx = hd.getInt();

      int64_t memContig = 1;
      if (auto attr = loadOp->getAttrOfType<DenseI64ArrayAttr>(
              "pact.pagelocal.dim_contiguity")) {
        if (headDimIdx >= 0 && headDimIdx < (int)attr.size())
          memContig = attr[headDimIdx];
      }

      if (memContig > 1) {
        unsigned elemNumBits = getElementBitWidth(refTensorType);
        int64_t memCap =
            std::min(memContig, (int64_t)(128 / elemNumBits));
        memCap = std::min(memCap, std::max<int64_t>(numElems / numThreads, 1));

        // Build the *candidate* blocked layout that Coalesce is about to emit
        // and compute its register-side contiguity with head_dim made the
        // most-minor output dimension.  Using the pre-Coalesce encoding is
        // wrong: its sizePerThread is the input layout's, not the vectorized
        // candidate's.
        SmallVector<unsigned> candidateSizePerThread(refTensorType.getRank(),
                                                     1);
        candidateSizePerThread[order[0]] = (unsigned)memCap;
        auto candidateEnc = BlockedEncodingAttr::get(
            op->getContext(), refTensorType.getShape(), candidateSizePerThread,
            order, numWarps, threadsPerWarp, cgaLayout);
        auto regLL = triton::gpu::toLinearLayout(refTensorType.getShape(),
                                                 candidateEnc);
        auto outNames = llvm::to_vector(regLL.getOutDimNames());
        int64_t regContig = 1;
        if (headDimIdx >= 0 && headDimIdx < (int)outNames.size()) {
          SmallVector<StringAttr> headFirstOrder;
          headFirstOrder.push_back(outNames[headDimIdx]);
          for (int i = 0; i < (int)outNames.size(); ++i)
            if (i != headDimIdx)
              headFirstOrder.push_back(outNames[i]);
          auto headFirst = regLL.transposeOuts(headFirstOrder).flattenOuts();
          regContig = headFirst.getNumConsecutiveInOut();
        }

        int64_t exactV = std::min(memCap, regContig);
        // M2 restores a lower bound: the exact page-bounded vector width.
        // It must never lower the width below what AxisInfo already proves
        // for this load.  On vLLM kernel_unified_attention the K/V loads are
        // natively coalesced to 128-bit and cp.async-pipelined; replacing
        // perThread with exactV=1 there demoted them to scalar loads and
        // cost 12 cp.async conversions (v7 regression #1).
        perThread = (unsigned)std::max<int64_t>(
            std::max<unsigned>(perThread, 1), exactV);
        if (const char *env = std::getenv("PACT_OVERRIDE_V")) {
          int pinned = std::atoi(env);
          if (pinned >= 1) {
            perThread = (unsigned)pinned;
            LDBG("PACT OVERRIDE_V: perThread -> " << perThread);
          }
        }
        LDBG("PACT M2 exact V: perThread -> " << perThread
             << " (memContig=" << memContig
             << ", memCap=" << memCap
             << ", regContig=" << regContig << ")");
      }
    }
  }

  if (!dyn_cast<triton::LoadOp>(op)) {
    // For ops that can result in a global memory write, we should enforce
    // that each thread handles at most 128 bits, which is the widest
    // available vectorized store op; otherwise, the store will have "gaps"
    // in the memory write at the warp level, resulting in worse performance.
    // For loads, we can expect that the gaps won't matter due to the L1
    // cache.
    perThread = std::min<int>(
        perThread,
        getNumElementsPerThread(op, order, axisInfoAnalysis, shapePerCTA));
  }
  SmallVector<unsigned> sizePerThread(refTensorType.getRank(), 1);
  sizePerThread[order[0]] = perThread;
  return BlockedEncodingAttr::get(op->getContext(), refTensorType.getShape(),
                                  sizePerThread, order, numWarps,
                                  threadsPerWarp, cgaLayout);
}
} // namespace mlir::triton::gpu
