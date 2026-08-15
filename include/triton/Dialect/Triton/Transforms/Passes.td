#ifndef TRITON_PASSES
#define TRITON_PASSES

include "mlir/Pass/PassBase.td"

def TritonCombineOps : Pass</*cli-arg*/"triton-combine", /*Op*/"mlir::ModuleOp"> {
  let summary = "combine ops";
  let description = [{
    This pass aims to optimize the five following patterns:
    - `dot(a, b, 0) + c => dot(a, b, c)`

    - `addptr(addptr(ptr, idx0), idx1) => addptr(ptr, AddI(idx0, idx1))`

    - `select(cond, load(ptrs, broadcast(cond), ???), other) =>
         load(ptrs, broadcast(cond), other)`

    - `broadcast(constant) => reshaped_constant`
    - `torch.sum(x[:,:,None].expand(-1,-1,n) * y[None,:,:].expand(m,-1,-1),1)
       => dot(x,y,splat(0))`
  }];

  let dependentDialects = ["mlir::arith::ArithDialect"];
}

def TritonReorderBroadcast : Pass</*cli-arg*/"triton-reorder-broadcast", /*Op*/"mlir::ModuleOp"> {
  let summary = "Moves broadcast and splat after elementwise operations";
  let description = [{
    The purpose of this pass is to transform:
      - `elementwise(broadcast(a)) => broadcast(elementwise(a))`
      - `elementwise(splat(a), splat(b), ...) => splat(elementwise(a, b, ...))`
    In the event of a match, the broadcast (or splat) operation is delayed
    and performed after the ElementWise operation.
  }];

  let dependentDialects = ["mlir::triton::TritonDialect"];
}

def TritonRewriteTensorDescriptorToPointer : Pass</*cli-arg*/"triton-rewrite-tensor-descriptor-to-pointer", /*Op*/"mlir::ModuleOp"> {
  let summary = "Rewrite load/stores of tensor descriptors into pointer load/stores";
  let description = [{
    This pass rewrites all load/store semantics initiated by a `tt.make_tensor_descriptor` into pointer semantics. After
    this pass, `tt.make_tensor_descriptor`  will disappear, and it generates logics to compute the pointer/mask/other
    for each load/store.
  }];

  let dependentDialects = ["mlir::triton::TritonDialect"];
}

def TritonLoopUnroll : Pass</*cli-arg*/"triton-loop-unroll", /*Op*/"mlir::ModuleOp"> {
  let summary = "Loop unroller";
  let description = [{
    The pass unrolls a scf loop with tt.loop_unroll_factor attribute. The attribute specialises how many iterations
    the loop should be unrolled.
  }];

  let dependentDialects = ["mlir::triton::TritonDialect"];
}

def TritonLoopInvariantCodeMotion : Pass</*cli-arg*/"triton-licm", /*Op*/"mlir::ModuleOp"> {
  let summary = "MLIR's LICM plus hoist load ops out of loops with masks.";
  let description = [{
    This pass uses MLIR's LICM pass as base. Additionally, it hoists load ops
    out of loops that consists of pure/read-only ops. For scf.for loops, it
    generates a trip-count check. For scf.while loops, it clones the condition
    from the before body.
  }];

  let dependentDialects = ["mlir::triton::TritonDialect"];
}

def TritonLoopAwareCSE : Pass<"triton-loop-aware-cse", "mlir::ModuleOp"> {
  let summary = "CSE within loop bodies";

  let description = [{
    The `triton-loop-aware-cse` pass performs recursive common subexpression
    elimination within loop bodies. Unlike regular CSE, which is a single-pass
    greedy algorithm, this pass can recursively eliminate loop iteration
    arguments and subcomputations that always have the same value.
  }];
}

def TritonPageTransform : Pass<"triton-page-transform", "mlir::ModuleOp"> {
  let summary = "Recognize and canonicalize paged KV cache access patterns";
  let description = [{
    Detects block-table-driven paged attention patterns where loads access KV
    caches through block_tables. Annotates matching loads with pact.* semantic
    attributes and classifies the kernel as decode_paged or prefill_paged
    (unknown falls back to conservative handling).
  }];
  let dependentDialects = ["mlir::triton::TritonDialect", "mlir::arith::ArithDialect"];
}

def PACTPageLocalAnalysis : Pass<"triton-pact-page-local-analysis", "mlir::ModuleOp"> {
  let summary = "Analyze page-local memory contiguity for paged KV loads";
  let description = [{
    Reads PACT semantic attributes (pact.page_size, pact.tile_tokens, etc.)
    from tt.load ops annotated with pact.paged_load, and computes verifiable
    page-local memory properties via an F₂-linear page-internal memory layout:
    per-dimension memory contiguity and page-boundary safety.

    Outputs pact.pagelocal.dim_contiguity, pact.pagelocal.safe_vector_width
    (transitional), and pact.pagelocal.statically_safe. Does not modify IR
    structure or AxisInfo — purely an analysis pass.
  }];
  let dependentDialects = ["mlir::triton::TritonDialect", "mlir::arith::ArithDialect"];
}

def PACTAutoNumWarps : Pass<"triton-pact-auto-num-warps", "mlir::ModuleOp"> {
  let summary = "Recommend optimal num_warps based on kernel structure analysis";
  let description = [{
    Analyzes kernel structure (number of paged loads, tile sizes, MMA presence)
    and writes pact.optimal_num_warps module attribute.  compiler.py reads this
    before make_ttgir to adjust opt.num_warps.  Default OFF.
  }];
  let dependentDialects = ["mlir::triton::TritonDialect"];
}

#endif
