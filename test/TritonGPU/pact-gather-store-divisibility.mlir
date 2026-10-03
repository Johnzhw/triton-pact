// RUN: triton-opt %s -tritongpu-coalesce | FileCheck %s
//
// V21-0c E-2 (0c-3): the family-B STORE-side divisibility proof.  The
// store ptr chain is base + row*STRIDE (STRIDE a runtime arg with
// tt.divisibility 16) + inner contiguous range: vanilla AxisInfo
// gcd-caps the final ptr divisibility at the range's 1 (the V8-1
// load-side shape, store edition), so Coalesce leaves the store at
// sizePerThread=[1,1].  The pact.gather_store annotation + stride_div
// (written by P1 only under PACT_GATHER_CONTIG=1) restores the 16B
// proof on the inner addptr whose user is the store — dim defaults to
// the LAST dim (stores carry no head_dim annotation; the store pointer
// has no shape information).

#blocked0 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>

// CHECK-DAG: #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:86", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @gather_store_e2
  tt.func @gather_store_e2(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32},
                           %row: tensor<16xi64> {tt.divisibility = 16 : i32},
                           %stride: i64 {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<16x64xf16, #blocked0>
    %strideT = tt.splat %stride : i64 -> tensor<16xi64>
    %rowoff = arith.muli %row, %strideT : tensor<16xi64>
    %e = tt.expand_dims %rowoff {axis = 1 : i32} : tensor<16xi64> -> tensor<16x1xi64>
    %rowT = tt.broadcast %e : tensor<16x1xi64> -> tensor<16x64xi64>
    %base = tt.splat %arg0 : !tt.ptr<f16> -> tensor<16x64x!tt.ptr<f16>, #blocked0>
    %rowbase = tt.addptr %base, %rowT : tensor<16x64x!tt.ptr<f16>, #blocked0>, tensor<16x64xi64>
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %range64 = tt.expand_dims %range {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %rangeT = tt.broadcast %range64 : tensor<1x64xi32> -> tensor<16x64xi32>
    %rangeE = arith.extsi %rangeT : tensor<16x64xi32> to tensor<16x64xi64>
    %ptrs = tt.addptr %rowbase, %rangeE : tensor<16x64x!tt.ptr<f16>, #blocked0>, tensor<16x64xi64>
    // CHECK: tt.store {{.*}} : tensor<16x64x!tt.ptr<f16>, #{{.*}}>
    tt.store %ptrs, %cst {pact.gather_store, pact.stride_div = 16 : i64} : tensor<16x64x!tt.ptr<f16>, #blocked0>
    tt.return
  }
}
