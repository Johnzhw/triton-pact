// RUN: env PACT_SM_VERSION=86 PACT_ENABLE_AUTO_NUM_WARPS=1 triton-opt %s -triton-pact-auto-num-warps | FileCheck %s
//
// Regression for the old `numPagedLoads >= 4` gate: the canonical decode tile
// has exactly two annotated K/V loads and must still enter the P11 decision.
// With tile 16x64 f16 (2048B) and 3 stages on SM86, occ(4)=0.667 beats
// occ(2)=0.375 by more than the one-CTA bound, so the decision stays at 4.

module {
  tt.func @two_paged_loads(%p0: !tt.ptr<f16>, %p1: !tt.ptr<f16>) {
    %range16 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %range64 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %e16 = tt.expand_dims %range16 {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %e64 = tt.expand_dims %range64 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %b16 = tt.broadcast %e16 : tensor<16x1xi32> -> tensor<16x64xi32>
    %b64 = tt.broadcast %e64 : tensor<1x64xi32> -> tensor<16x64xi32>
    %cst = arith.constant dense<0.000000e+00> : tensor<16x64xf16>

    %s0 = tt.splat %p0 : !tt.ptr<f16> -> tensor<16x64x!tt.ptr<f16>>
    %ptr0 = tt.addptr %s0, %b16 : tensor<16x64x!tt.ptr<f16>>, tensor<16x64xi32>
    %v0 = tt.load %ptr0 {pact.head_dim_size = 64 : i64, pact.page_size = 64 : i64,
                         pact.paged_load, pact.tile_tokens = 16 : i64}
          : tensor<16x64x!tt.ptr<f16>>

    %s1 = tt.splat %p1 : !tt.ptr<f16> -> tensor<16x64x!tt.ptr<f16>>
    %ptr1 = tt.addptr %s1, %b16 : tensor<16x64x!tt.ptr<f16>>, tensor<16x64xi32>
    %v1 = tt.load %ptr1 {pact.head_dim_size = 64 : i64, pact.page_size = 64 : i64,
                         pact.paged_load, pact.tile_tokens = 16 : i64}
          : tensor<16x64x!tt.ptr<f16>>
    tt.return
  }
}

// CHECK: module attributes {pact.optimal_num_warps = 4 : i32
// CHECK-SAME: pact.p11.stages_assumption = 3 : i32
