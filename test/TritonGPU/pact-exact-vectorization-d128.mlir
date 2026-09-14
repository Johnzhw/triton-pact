// RUN: triton-opt %s -tritongpu-coalesce | FileCheck %s
//
// M2 on a head_dim=128 paged tile.
//
// `pact-exact-vectorization.mlir` pins the 16x64 shape, where Coalesce emits
// `sizePerThread=[1,8], threadsPerWarp=[4,8]`.  The 16x128 shape does *not*
// follow: with 4 warps over 16x128 there are not enough threads to give every
// lane 8 contiguous elements on the head axis, so the register layout comes out
// `[1,8]` over `threadsPerWarp=[2,16]`.
//
// v7 note (superseded by the v8 fix): with that layout the real kernel still
// emitted scalar 32-bit loads and this was read as a "wide-load gap" of M2.
// v8 diagnosis (suite/results/v8/d128_diag/) showed the layout was never the
// blocker: the old P3 divisibility boost bound elemSize*2 (=4B for f16)
// gcd-capped the pointer divisibility at 4 bytes on *every* shape, and the
// single 128-bit load in the v7 d64 dump was the dense Q load, not a paged
// K/V load.  The v8 fix raises the rhs bound to min(16, head_dim_size *
// elemSize) in AxisInfo; paged K/V loads now emit ld.global.v4.b32 for both
// D=64 and D=128 (suite/results/ir_dump_v8fix/), while the Coalesce layout
// pinned below is unchanged.
//
// This test pins the layout M2 produces today so a future change to the
// regContig/memCap computation cannot silently move it.

#blocked0 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>

// CHECK-DAG: #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:86", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @paged_load_exact_v_d128
  tt.func @paged_load_exact_v_d128(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32},
                                   %arg1: tensor<16x128xi32, #blocked0>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<16x128xf16, #blocked0>
    %base = tt.splat %arg0 : !tt.ptr<f16> -> tensor<16x128x!tt.ptr<f16>, #blocked0>
    %ptr = tt.addptr %base, %arg1 : tensor<16x128x!tt.ptr<f16>, #blocked0>, tensor<16x128xi32, #blocked0>
    // CHECK: tt.load {{.*}} : tensor<16x128x!tt.ptr<f16>, #{{.*}}>
    %v = tt.load %ptr {pact.head_dim_idx = 1 : i64, pact.head_dim_size = 128 : i64,
                       pact.paged_load, pact.pagelocal.dim_contiguity = array<i64: 1, 128>,
                       pact.pagelocal.statically_safe = true,
                       pact.page_size = 64 : i64,
                       pact.tile_tokens = 16 : i64} : tensor<16x128x!tt.ptr<f16>, #blocked0>
    tt.return
  }
}
