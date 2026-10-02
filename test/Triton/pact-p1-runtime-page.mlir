// RUN: env PACT_ENABLE=1 PACT_RUNTIME_PAGE=1 triton-opt %s -triton-page-transform | FileCheck %s
//
// V20 W6'a family A: paging by a RUNTIME divisor — divsi/remsi whose
// divisor is the scalar argument %page (tt.divisibility = 16), reaching
// the pattern as splat(extsi(arg)).  No constant divsi exists, so the
// frozen path classifies this kernel prefill and no-ops; family A
// recognizes the structure: the block-table load (div) is marked
// block_table_lookup, the KV tile load (rem) becomes pact.paged_load
// with the divisor's argument index recorded for P2 identity matching.

module {
  tt.func @runtime_page_copy(%src: !tt.ptr<f16> {tt.divisibility = 16 : i32},
                             %bt: !tt.ptr<i32> {tt.divisibility = 16 : i32},
                             %page: i32 {tt.divisibility = 16 : i32}) {
    %c0 = arith.constant 0 : i32
    %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %tok64 = arith.extsi %range : tensor<16xi32> to tensor<16xi64>
    %page64 = arith.extsi %page : i32 to i64
    %pageT = tt.splat %page64 : i64 -> tensor<16xi64>
    %divT = arith.divsi %tok64, %pageT : tensor<16xi64>
    %remT = arith.remsi %tok64, %pageT : tensor<16xi64>
    %div32 = arith.trunci %divT : tensor<16xi64> to tensor<16xi32>
    %btT = tt.splat %bt : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
    %bt_ptrs = tt.addptr %btT, %div32 : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
    %pages = tt.load %bt_ptrs : tensor<16x!tt.ptr<i32>>
    %pages64 = arith.extsi %pages : tensor<16xi32> to tensor<16xi64>
    %pageOff = arith.muli %pages64, %pageT : tensor<16xi64>
    %offs = arith.addi %pageOff, %remT : tensor<16xi64>
    %srcT = tt.splat %src : !tt.ptr<f16> -> tensor<16x!tt.ptr<f16>>
    %src_ptrs = tt.addptr %srcT, %offs : tensor<16x!tt.ptr<f16>>, tensor<16xi64>
    %tile = tt.load %src_ptrs : tensor<16x!tt.ptr<f16>>
    tt.return
  }
}

// CHECK: pact.kernel_type = "paged_rt_or_gather"
// CHECK: tt.load %{{[0-9]+}} {pact.block_table_lookup}
// CHECK: {pact.head_dim_idx = 0 : i64, pact.head_dim_size = 16 : i64, pact.page_boundary_safe = false, pact.page_divisor_arg = 2 : i64, pact.paged_load
