// RUN: env PACT_ENABLE=1 triton-opt %s -triton-page-transform | FileCheck %s
//
// V18 波7-B regression anchor for the vLLM unified_attention signature
// (B0 probe, results/v18/p1_prod_gap/): the production kernel names its
// paged operands block_tables_ptr (PLURAL) / seq_lens_ptr and reaches
// the classifier through Signal B only -- its token indexing has no
// divsi(PAGE_SIZE) pattern.  The substring match ("block_table" in
// "block_tables_ptr") must keep hitting so the kernel stays classified
// prefill_paged (annotated downstream), never demoted to "prefill"
// (skip).  Locking the CURRENT behavior: if a rename or a stricter
// matcher ever drops this, the production-kernel P1/P2 chain goes dark
// silently.

module {
  tt.func @unified_attn_min(%q_ptr: !tt.ptr<f16> loc("q_ptr"),
                            %bt_ptr: !tt.ptr<i32> loc("block_tables_ptr"),
                            %sl_ptr: !tt.ptr<i32> loc("seq_lens_ptr"),
                            %kv_ptr: !tt.ptr<f16> loc("k_cache_ptr")) {
    %pid = tt.get_program_id x : i32
    %c1 = arith.constant 1 : i32
    %c16 = arith.constant 16 : i32
    %c64 = arith.constant 64 : i32

    // seq_lens load (runtime trip count -- no static divsi anywhere,
    // exactly the production kernel's decode loop shape)
    %sl_idx = arith.muli %pid, %c1 : i32
    %sl_addr = tt.addptr %sl_ptr, %sl_idx : !tt.ptr<i32>, i32
    %seq_len = tt.load %sl_addr : !tt.ptr<i32>

    // block-table indirect addressing WITHOUT divsi: the block number
    // is loaded from the table directly (Signal B territory only)
    %bt_off = arith.muli %pid, %c16 : i32
    %bt_base = tt.addptr %bt_ptr, %bt_off : !tt.ptr<i32>, i32
    %block_num = tt.load %bt_base : !tt.ptr<i32>
    %kv_off = arith.muli %block_num, %c64 : i32
    %kv_off_splat = tt.splat %kv_off : i32 -> tensor<64xi32>
    %kv_base = tt.splat %kv_ptr : !tt.ptr<f16> -> tensor<64x!tt.ptr<f16>>
    %kv_ptr_t = tt.addptr %kv_base, %kv_off_splat : tensor<64x!tt.ptr<f16>>, tensor<64xi32>
    %kv = tt.load %kv_ptr_t : tensor<64x!tt.ptr<f16>>
    tt.return
  }
}

// CHECK: module attributes {pact.kernel_type = "prefill_paged"
