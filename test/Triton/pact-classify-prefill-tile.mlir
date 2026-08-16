// RUN: env PACT_ENABLE=1 triton-opt %s -triton-page-transform | FileCheck %s
//
// B1 v2 companion: chunked prefill shares the ambiguous block-table chain but
// derives a tile Q offset from program_id.  The classifier must stay
// conservative and emit prefill_paged.

module {
  tt.func @chunked_prefill(%q_ptr: !tt.ptr<f16> loc("q_ptr"),
                           %bt_ptr: !tt.ptr<i32> loc("block_table_ptr"),
                           %kv_ptr: !tt.ptr<f16> loc("k_cache_ptr")) {
    %pid = tt.get_program_id x : i32
    %c4 = arith.constant 4 : i32
    %c16 = arith.constant 16 : i32
    %c64 = arith.constant 64 : i32

    // Tile query path: pid -> muli -> splat into tensor tile offset.
    %token_start = arith.muli %pid, %c16 : i32
    %q_off_splat = tt.splat %token_start : i32 -> tensor<16xi32>
    %range16 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %q_off = arith.addi %q_off_splat, %range16 : tensor<16xi32>
    %q_off_2d = tt.expand_dims %q_off {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %q_off_2d_b = tt.broadcast %q_off_2d : tensor<16x1xi32> -> tensor<16x64xi32>
    %q_base = tt.splat %q_ptr : !tt.ptr<f16> -> tensor<16x64x!tt.ptr<f16>>
    %q_ptr_t = tt.addptr %q_base, %q_off_2d_b : tensor<16x64x!tt.ptr<f16>>, tensor<16x64xi32>
    %q = tt.load %q_ptr_t : tensor<16x64x!tt.ptr<f16>>

    // Ambiguous block-table chain (same as value-tensor decode).
    %page_idx = arith.divsi %token_start, %c64 : i32
    %bt_off = arith.muli %pid, %c4 : i32
    %bt_base = tt.addptr %bt_ptr, %bt_off : !tt.ptr<i32>, i32
    %bt_ptr2 = tt.addptr %bt_base, %page_idx : !tt.ptr<i32>, i32
    %block_num = tt.load %bt_ptr2 : !tt.ptr<i32>
    tt.return
  }
}

// CHECK: module attributes {pact.kernel_type = "prefill_paged"
