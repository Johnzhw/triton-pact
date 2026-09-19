// RUN: env PACT_SM_VERSION=86 PACT_ENABLE_AUTO_NUM_STAGES=1 triton-opt %s -tritongpu-pact-auto-num-stages | FileCheck %s
//
// PACT P6 S1 L2-residency, hot variant.  Trip count 1024 × tile_tokens 16
// gives S=16384; head_dim 64 f16 makes the K+V working set
// 16384×64×2B×2 = 4MB, exactly the SM86 L2 capacity (4MB).  The residency
// classification is "hot" (working set <= L2), so the decision must be
// identical to the pre-S1 occupancy scan: with a 16x64 f16 tile the
// register bound (8 blocks, occ=0.667) dominates every feasible s∈[2,4],
// the occupancy ties, and the tie rule keeps the stage count closest to
// the native default (3).  This pins both the default-path-unchanged
// guarantee for L2-resident shapes and the inclusive <= boundary.

#blocked0 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32,
                   ttg.target = "cuda:86", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @paged_loop_l2_hot(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    %c0 = arith.constant 0 : index
    %c1024 = arith.constant 1024 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %c1024 step %c1 {
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
// CHECK-SAME: pact.optimal_num_stages = 3 : i32
// CHECK: scf.for
// CHECK: tt.num_stages = 3 : i32
