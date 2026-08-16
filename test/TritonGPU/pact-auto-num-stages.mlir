// RUN: env PACT_SM_VERSION=86 PACT_ENABLE_AUTO_NUM_STAGES=1 triton-opt %s -tritongpu-pact-auto-num-stages | FileCheck %s
//
// PACT P6 theory-as-code: with a static trip count of 16 and an exact tile of
// 16x128 f16 (4096B), the feasible stage set is [2, min(maxStages=4,
// smemBound, iterBound=16)].  The SM86 capacity equations compute
// occ(2)=0.667, occ(3)=0.5, occ(4)=0.333, so the occupancy argmax selects
// num_stages=2 — computed, not taken from a hard-coded `<=16 -> 2` branch.

#blocked0 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32,
                   ttg.target = "cuda:86", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @paged_loop(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %c16 step %c1 {
      %range16 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
      %range128 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
      %e16 = tt.expand_dims %range16 {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
      %e128 = tt.expand_dims %range128 {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
      %b16 = tt.broadcast %e16 : tensor<16x1xi32> -> tensor<16x128xi32>
      %b128 = tt.broadcast %e128 : tensor<1x128xi32> -> tensor<16x128xi32>
      %base = tt.splat %arg0 : !tt.ptr<f16> -> tensor<16x128x!tt.ptr<f16>, #blocked0>
      %ptr = tt.addptr %base, %b16 : tensor<16x128x!tt.ptr<f16>, #blocked0>, tensor<16x128xi32>
      %v = tt.load %ptr {pact.head_dim_size = 128 : i64, pact.page_size = 64 : i64,
                         pact.paged_load, pact.tile_tokens = 16 : i64}
           : tensor<16x128x!tt.ptr<f16>, #blocked0>
    }
    tt.return
  }
}

// CHECK: module attributes {pact.optimal_num_stages = 2 : i32
// CHECK: scf.for
// CHECK: tt.num_stages = 2 : i32
