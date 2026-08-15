// RUN: triton-opt %s -tritongpu-coalesce | FileCheck %s
//
// PACT B1: for a paged load with exact memory contiguity [token=1, head=64],
// Coalesce must emit a blocked layout with sizePerThread=[1,8] (V = min(64,
// register contiguity, 128-bit hardware cap) = 8 for f16).

#blocked0 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>

// CHECK-DAG: #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:86", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @paged_load_exact_v
  tt.func @paged_load_exact_v(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32},
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
