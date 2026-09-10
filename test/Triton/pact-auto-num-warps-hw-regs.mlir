// RUN: env PACT_SM_VERSION=86 PACT_ENABLE_AUTO_NUM_WARPS=1 triton-opt %s -triton-pact-auto-num-warps | FileCheck %s
//
// P11 with measured regs_per_thread=128 on SM86 16x64 f16 / stages=3:
// both 4-warp and 2-warp are register-limited to the same occupancy, so the
// S3a exact-tie rule selects 2 warps.  Does not change the default (64-reg)
// theory path in pact-auto-num-warps.mlir.

module attributes {pact.hw.regs_per_thread = 128 : i32} {
  tt.func @two_paged_loads(%p0: !tt.ptr<f16>, %p1: !tt.ptr<f16>) {
    %range16 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %range64 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %e16 = tt.expand_dims %range16 {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %e64 = tt.expand_dims %range64 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %b16 = tt.broadcast %e16 : tensor<16x1xi32> -> tensor<16x64xi32>

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

// CHECK: module attributes
// CHECK-DAG: pact.optimal_num_warps = 2 : i32
// CHECK-DAG: pact.hw.regs_per_thread = 128 : i32
