// RUN: env PACT_ENABLE=1 triton-opt %s -triton-page-transform | FileCheck %s
//
// S2 de-naming regression: the scalar Q pointer base is named `qbuf_ptr`, which
// matches none of the legacy q/q_ptr/query/q_* name rules.  The structural
// signal (pointer BlockArgument -> scalar addptr -> splat -> rank-1 load of
// the same element type) must still classify this kernel as decode_paged.

module {
  tt.func @value_decode_unnamed(%qbuf_ptr: !tt.ptr<f16> loc("qbuf_ptr"),
                                %bt_ptr: !tt.ptr<i32> loc("block_table_ptr"),
                                %kv_ptr: !tt.ptr<f16> loc("k_cache_ptr")) {
    %pid = tt.get_program_id x : i32
    %c8 = arith.constant 8 : i32
    %c16 = arith.constant 16 : i32
    %c64 = arith.constant 64 : i32

    // Scalar query path with a non-q-named base pointer.
    %q_off = arith.muli %pid, %c8 : i32
    %q_scalar = tt.addptr %qbuf_ptr, %q_off : !tt.ptr<f16>, i32
    %q_splat = tt.splat %q_scalar : !tt.ptr<f16> -> tensor<64x!tt.ptr<f16>>
    %range64 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %q_ptr_t = tt.addptr %q_splat, %range64 : tensor<64x!tt.ptr<f16>>, tensor<64xi32>
    %q = tt.load %q_ptr_t : tensor<64x!tt.ptr<f16>>

    // Ambiguous block-table chain: page_idx directly feeds addptr.
    %token_start = arith.muli %pid, %c16 : i32
    %page_idx = arith.divsi %token_start, %c64 : i32
    %bt_off = arith.muli %pid, %c8 : i32
    %bt_base = tt.addptr %bt_ptr, %bt_off : !tt.ptr<i32>, i32
    %bt_ptr2 = tt.addptr %bt_base, %page_idx : !tt.ptr<i32>, i32
    %block_num = tt.load %bt_ptr2 : !tt.ptr<i32>
    tt.return
  }
}

// CHECK: module attributes {pact.kernel_type = "decode_paged"
