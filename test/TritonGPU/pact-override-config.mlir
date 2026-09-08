// RUN: env PACT_OVERRIDE_V=4 triton-opt %s -tritongpu-coalesce | FileCheck %s
//
// PACT_OVERRIDE_V pins sizePerThread on a paged load after M2 computes V.
// Without the pin this IR emits [1,8] (see pact-exact-vectorization.mlir).

#blocked0 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>

// CHECK-DAG: sizePerThread = [1, 4]

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:86", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @paged_load_override_v
  tt.func @paged_load_override_v(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32},
                                 %arg1: tensor<16x64xi32, #blocked0>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<16x64xf16, #blocked0>
    %base = tt.splat %arg0 : !tt.ptr<f16> -> tensor<16x64x!tt.ptr<f16>, #blocked0>
    %ptr = tt.addptr %base, %arg1 : tensor<16x64x!tt.ptr<f16>, #blocked0>, tensor<16x64xi32, #blocked0>
    // CHECK: tt.load {{.*}} : tensor<16x64x!tt.ptr<f16>, #{{.*}}>
    %v = tt.load %ptr {pact.head_dim_idx = 1 : i64, pact.page_size = 64 : i64,
                       pact.paged_load, pact.pagelocal.dim_contiguity = array<i64: 1, 64>,
                       pact.pagelocal.statically_safe = true,
                       pact.tile_tokens = 16 : i64} : tensor<16x64x!tt.ptr<f16>, #blocked0>
    tt.return
  }
}
