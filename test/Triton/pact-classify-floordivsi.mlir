// RUN: env PACT_ENABLE=1 triton-opt %s -triton-page-transform | FileCheck %s
//
// N6 regression: classifyKernel must accept arith.floordivsi exactly like
// detectPageSize does.  page_idx is merged into the block-table index through
// arith.addi (pointer-tensor decode style), so this kernel would have been
// mislabeled `prefill` (PACT skipped entirely) by the v3 divsi-only walk.

module {
  tt.func @floordiv_decode(%bt_ptr: !tt.ptr<i32> loc("block_table_ptr"),
                           %kv_ptr: !tt.ptr<f16> loc("k_cache_ptr")) {
    %pid = tt.get_program_id x : i32
    %c8 = arith.constant 8 : i32
    %c16 = arith.constant 16 : i32
    %c64 = arith.constant 64 : i32

    %token_start = arith.muli %pid, %c16 : i32
    %page_idx = arith.floordivsi %token_start, %c64 : i32
    %bt_off = arith.muli %pid, %c8 : i32
    %bt_idx = arith.addi %bt_off, %page_idx : i32
    %bt_ptr2 = tt.addptr %bt_ptr, %bt_idx : !tt.ptr<i32>, i32
    %block_num = tt.load %bt_ptr2 : !tt.ptr<i32>
    tt.return
  }
}

// CHECK: module attributes {pact.kernel_type = "decode_paged"
