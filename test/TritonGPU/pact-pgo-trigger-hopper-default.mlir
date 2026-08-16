// RUN: env PACT_SM_VERSION=90 PACT_ENABLE_PGO_TRIGGER=1 triton-opt %s -tritongpu-pact-pgo-trigger | FileCheck %s
//
// N3 regression: on Hopper the native default is 5 stages.  P6 publishes
// pact.native_num_stages = 5 and chooses optimal = 5 (no change), so the
// trigger must NOT report a stage opportunity.  The v3 code read a module
// tt.num_stages attribute that the pipeline never writes and would have
// compared 5 against a hard-coded 3.

#blocked0 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {"pact.axisinfo.baseline_contiguity" = 64 : i64,
                   "pact.native_num_stages" = 5 : i32,
                   "pact.optimal_num_stages" = 5 : i32,
                   "pact.optimal_num_warps" = 4 : i32,
                   "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32,
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
      %v = tt.load %ptr {pact.head_dim_idx = 1 : i64, pact.page_size = 64 : i64,
                         pact.paged_load, pact.tile_tokens = 16 : i64,
                         pact.pagelocal.dim_contiguity = array<i64: 1, 64>}
           : tensor<16x64x!tt.ptr<f16>, #blocked0>
    }
    tt.return
  }
}

// CHECK: module attributes {pact.axisinfo.baseline_contiguity = 64 : i64
// CHECK-SAME: pact.native_num_stages = 5 : i32
// CHECK-SAME: pact.optimal_num_stages = 5 : i32
// CHECK-SAME: pact.optimal_num_warps = 4 : i32
// CHECK-SAME: pact.pgo.opportunity_bits = 0 : i32
// CHECK-SAME: pact.pgo.trigger = 0 : i32
// CHECK-SAME: pact.pgo.trigger_reason = "none"
