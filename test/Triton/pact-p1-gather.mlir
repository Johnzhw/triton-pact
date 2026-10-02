// RUN: env PACT_ENABLE=1 PACT_GATHER_CONTIG=1 triton-opt %s -triton-page-transform | FileCheck %s
//
// V20 W6'a family B: gather-row addressing — the addptr offset chain
// contains muli(row, %stride) where %stride is a runtime i64 argument
// with tt.divisibility = 16.  The frozen path sees no divsi at all, so
// family B is the only recognition: the tile load is annotated
// gather_load and the stride's divisibility is recorded for the
// AxisInfo alignment proof (wide vector loads on runtime strides).

module {
  tt.func @gather_row_load(%a: !tt.ptr<f16> {tt.divisibility = 16 : i32},
                           %idx: !tt.ptr<i32> {tt.divisibility = 16 : i32},
                           %stride: i64 {tt.divisibility = 16 : i32}) {
    %c0 = arith.constant 0 : i32
    %row = tt.load %idx : !tt.ptr<i32>
    %row64 = arith.extsi %row : i32 to i64
    %rowoff = arith.muli %row64, %stride : i64
    %base = tt.addptr %a, %rowoff : !tt.ptr<f16>, i64
    %baseT = tt.splat %base : !tt.ptr<f16> -> tensor<64x!tt.ptr<f16>>
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %range64 = arith.extsi %range : tensor<64xi32> to tensor<64xi64>
    %ptrs = tt.addptr %baseT, %range64 : tensor<64x!tt.ptr<f16>>, tensor<64xi64>
    %tile = tt.load %ptrs : tensor<64x!tt.ptr<f16>>
    tt.return
  }
}

// CHECK: pact.kernel_type = "paged_rt_or_gather"
// CHECK: {pact.gather_load, pact.head_dim_idx = 0 : i64, pact.head_dim_size = 64 : i64, pact.stride_arg = 2 : i64, pact.stride_div = 16 : i64}
