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
    caches through block_tables (tt.load from block_tables_ptr indexed by
    seq_offset // BLOCK_SIZE). Annotates matching loads with pact.* attributes
    and performs safe canonicalization (div→shift, mod→and for power-of-2
    PAGE_SIZE, conservative block_table load hoisting).
  }];
  let dependentDialects = ["mlir::triton::TritonDialect", "mlir::arith::ArithDialect"];
}

def TritonPactPrefetch : Pass<"triton-pact-prefetch", "mlir::ModuleOp"> {
  let summary = "Software block-table prefetch restructuring for paged attention";
  let description = [{
    Restructures the main tile loop to prefetch the next iteration's
    block_table lookup while computing the current iteration's attention.
    Operates at TTIR level, before TTGIR conversion.
  }];
  let dependentDialects = ["mlir::triton::TritonDialect", "mlir::arith::ArithDialect"];
}

def TritonBlockTableScalarize : Pass<"triton-block-table-scalarize", "mlir::ModuleOp"> {
  let summary = "Scalarize uniform block_table tensor loads to scalar+splat";
  let description = [{
    Converts block_table tensor gather loads (tt.load with
    pact.block_table_lookup, loading tensor<16xi32> where all 16 lanes
    carry the same value) into a scalar load + splat.  This eliminates
    redundant lanes and simplifies downstream optimizations.

    Only applied when uniformity can be proven: PAGE_SIZE and TILE_SIZE
    are compile-time constants, TILE_SIZE <= PAGE_SIZE, and the page index
    computation is divsi(j*TILE + offs_t, PAGE) which yields identical
    values for all lanes when TILE divides PAGE.
  }];
  let dependentDialects = ["mlir::triton::TritonDialect", "mlir::arith::ArithDialect"];
}

def TritonPatternSpecialize : Pass<"triton-pattern-specialize", "mlir::ModuleOp"> {
  let summary = "Specialize MQA/GQA head mapping for paged attention";
  let description = [{
    Reads pact.mqa / pact.gqa metadata from function attributes and performs
    conservative KV head mapping canonicalization. For MQA, canonicalizes
    kv_head_idx to 0. For GQA, ensures kv_head_idx = q_head_idx / group_size.
    Records sharing metadata for downstream PrefetchInsert. No-op fallback
    if metadata is missing or GQA group size is invalid.
  }];
  let dependentDialects = ["mlir::triton::TritonDialect", "mlir::arith::ArithDialect"];
}

#endif
