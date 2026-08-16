// RUN: env PACT_ENABLE=1 triton-opt %s -triton-page-transform | FileCheck %s
//
// B1 v2: value-tensor decode.  page_idx feeds tt.addptr directly, but the
// scalar Q pointer is derived from program_id and the query argument is
// q-named, so the classifier must emit decode_paged (not prefill_paged).

module {
  tt.func @value_decode(%q_ptr: !tt.ptr<f16> loc("q_ptr"),
                        %bt_ptr: !tt.ptr<i32> loc("block_table_ptr"),
                        %kv_ptr: !tt.ptr<f16> loc("k_cache_ptr")) {
    %pid = tt.get_program_id x : i32
    %c8 = arith.constant 8 : i32
    %c16 = arith.constant 16 : i32
    %c64 = arith.constant 64 : i32

    // Scalar query path: pid -> muli -> scalar q addptr -> rank-1 load.
    %q_off = arith.muli %pid, %c8 : i32
    %q_scalar = tt.addptr %q_ptr, %q_off : !tt.ptr<f16>, i32
    %q_splat = tt.splat %q_scalar : !tt.ptr<f16> -> tensor<64x!tt.ptr<f16>>
    %range64 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %q_ptr_t = tt.addptr %q_splat, %range64 : tensor<64x!tt.ptr<f16>>, tensor<64xi32>
    %q = tt.load %q_ptr_t : tensor<64x!tt.ptr<f16>>

    // Ambiguous block-table chain: page_idx directly feeds addptr whose base
    // already carries the per-token stride.
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
