// RUN: env PACT_SM_VERSION=86 PACT_ENABLE_AUTO_NUM_WARPS=1 triton-opt %s -triton-pact-auto-num-warps | FileCheck %s
//
// PACT P11 theory-only decision core: for four paged f16 loads of tile
// 16x64 (2048B), the L2 capacity equations on SM86 give occ(4) == occ(2)
// (both 8 blocks/SM by the register bound), and the computed one-CTA
// discretization granularity requires occ(2) - occ(4) > 4/48 to switch.
// The theory-as-code decision therefore keeps num_warps = 4.

module {
  tt.func @four_paged_loads(%p0: !tt.ptr<f16>, %p1: !tt.ptr<f16>,
                            %p2: !tt.ptr<f16>, %p3: !tt.ptr<f16>) {
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

    %s2 = tt.splat %p2 : !tt.ptr<f16> -> tensor<16x64x!tt.ptr<f16>>
    %ptr2 = tt.addptr %s2, %b16 : tensor<16x64x!tt.ptr<f16>>, tensor<16x64xi32>
    %v2 = tt.load %ptr2 {pact.head_dim_size = 64 : i64, pact.page_size = 64 : i64,
                         pact.paged_load, pact.tile_tokens = 16 : i64}
          : tensor<16x64x!tt.ptr<f16>>

    %s3 = tt.splat %p3 : !tt.ptr<f16> -> tensor<16x64x!tt.ptr<f16>>
    %ptr3 = tt.addptr %s3, %b16 : tensor<16x64x!tt.ptr<f16>>, tensor<16x64xi32>
    %v3 = tt.load %ptr3 {pact.head_dim_size = 64 : i64, pact.page_size = 64 : i64,
                         pact.paged_load, pact.tile_tokens = 16 : i64}
          : tensor<16x64x!tt.ptr<f16>>
    tt.return
  }
}

// CHECK: module attributes {pact.optimal_num_warps = 4 : i32}
