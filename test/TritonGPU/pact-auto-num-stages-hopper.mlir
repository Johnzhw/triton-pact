// RUN: env PACT_SM_VERSION=90 PACT_ENABLE_AUTO_NUM_STAGES=1 triton-opt %s -tritongpu-pact-auto-num-stages | FileCheck %s
//
// B5 regression: with no explicit PACT_MAX_PIPELINE_STAGES the default knob
// value 4 must not clamp Hopper's architecture default of 5 stages.  Tile
// 16x64 f16 (2048B), 16 iterations: smemBound=22 and iterBound=16, so the
// feasible set is [2, 5].  All candidates are register-bound at occ=0.5; the
// tie-break prefers defaultStages=5 and the Hopper policy takes the largest
// stage within the one-CTA tolerance.
//
// An explicit PACT_MAX_PIPELINE_STAGES=4 still acts as a hard cap (second RUN).

#blocked0 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32,
                   ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @paged_loop(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %c16 step %c1 {
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

// CHECK: module attributes {pact.native_num_stages = 5 : i32
// CHECK-SAME: pact.optimal_num_stages = 5 : i32
// When optimal == default, the loop attribute is intentionally left untouched.
// CHECK-NOT: tt.num_stages =
