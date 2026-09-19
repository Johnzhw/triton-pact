// RUN: env PACT_SM_VERSION=86 PACT_ENABLE_AUTO_NUM_STAGES=1 PACT_MAX_PIPELINE_STAGES=5 triton-opt %s -tritongpu-pact-auto-num-stages | FileCheck %s
//
// PACT P6 S1 L2-residency, cold variant.  Identical occupancy structure to
// pact-auto-num-stages-l2-hot.mlir (register-bound tie at 8 blocks,
// occ=0.667), but trip count 2048 × tile_tokens 16 gives S=32768 and the
// K+V working set 32768×64×2B×2 = 8MB > the SM86 L2 capacity (4MB).  The
// loop therefore streams from DRAM and the S1 cold path deepens the
// pipeline under the Hopper-style acceptance gate: a depth is adopted when
// its occupancy stays within the one-CTA discretization granularity
// (4/48 ≈ 0.083) of the scan's best.  With a 16x64 f16 tile, s=5 needs
// 5×2048B+4KB = 14KB SMEM → 7 blocks → occ 0.583 = 0.667 − 4/48, exactly
// the computed model-error bound, so s=5 is adopted under the explicit
// PACT_MAX_PIPELINE_STAGES=5 cap (measured 2.17x vs vanilla at this shape
// where the pre-S1 tie rule kept 3; deeper drops beyond one CTA are vetoed).

#blocked0 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32,
                   ttg.target = "cuda:86", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @paged_loop_l2_cold(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    %c0 = arith.constant 0 : index
    %c2048 = arith.constant 2048 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %c2048 step %c1 {
      %range16 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
      %range64 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
      %e16 = tt.expand_dims %range16 {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
      %e64 = tt.expand_dims %range64 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
      %b16 = tt.broadcast %e16 : tensor<16x1xi32> -> tensor<16x64xi32>
      %b64 = tt.broadcast %e64 : tensor<1x64xi32> -> tensor<16x64xi32>
      %base = tt.splat %arg0 : !tt.ptr<f16> -> tensor<16x64x!tt.ptr<f16>, #blocked0>
      %ptr = tt.addptr %base, %b16 : tensor<16x64x!tt.ptr<f16>, #blocked0>, tensor<16x64xi32>
      %v = tt.load %ptr {pact.head_dim_size = 64 : i64, pact.page_size = 64 : i64,
                         pact.paged_load, pact.tile_tokens = 16 : i64}
           : tensor<16x64x!tt.ptr<f16>, #blocked0>
    }
    tt.return
  }
}

// CHECK: module attributes {pact.native_num_stages = 3 : i32
// CHECK-SAME: pact.optimal_num_stages = 5 : i32
// CHECK: scf.for
// CHECK: tt.num_stages = 5 : i32
