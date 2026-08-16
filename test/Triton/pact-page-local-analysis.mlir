// RUN: triton-opt %s -triton-pact-page-local-analysis | FileCheck %s
//
// PACT P2 (M9a/M9b, Proposition 1 and 2): the page-internal F₂ layout
// must compute mem_contig = head_dim and token_contig = 1 exactly for any
// power-of-two page size / head dimension.

module {
  tt.func @paged_loads(%p: !tt.ptr<f16>) {
    %range16 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %range64 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %e16 = tt.expand_dims %range16 {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %e64 = tt.expand_dims %range64 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %b16 = tt.broadcast %e16 : tensor<16x1xi32> -> tensor<16x64xi32>
    %b64 = tt.broadcast %e64 : tensor<1x64xi32> -> tensor<16x64xi32>
    %base = tt.splat %p : !tt.ptr<f16> -> tensor<16x64x!tt.ptr<f16>>
    %ptr = tt.addptr %base, %b16 : tensor<16x64x!tt.ptr<f16>>, tensor<16x64xi32>
    %cst = arith.constant dense<0.000000e+00> : tensor<16x64xf16>
    // CHECK: tt.load {{.*}} {pact.head_dim_idx = 1 : i64, pact.head_dim_size = 64 : i64,
    // CHECK-SAME: pact.pagelocal.dim_contiguity = array<i64: 1, 64>,
    // CHECK-SAME: pact.pagelocal.statically_safe = true
    %v = tt.load %ptr {pact.head_dim_idx = 1 : i64, pact.head_dim_size = 64 : i64,
                       pact.page_boundary_safe = true, pact.page_size = 16 : i64,
                       pact.paged_load, pact.tile_tokens = 16 : i64}
             : tensor<16x64x!tt.ptr<f16>>
    // When P1's fallback flag says the tile is not provably page-aligned and
    // P2 cannot trace a remsi(PAGE_SIZE) offset, statically_safe must be
    // propagated as false.  The dim_contiguity output stays identical (the
    // ConservativeOverride path uses the same exact page-bounded width and
    // relies on the load mask for out-of-page elements).
    // CHECK: tt.load {{.*}} {pact.head_dim_idx = 1 : i64, pact.head_dim_size = 64 : i64,
    // CHECK-SAME: pact.pagelocal.dim_contiguity = array<i64: 1, 64>,
    // CHECK-SAME: pact.pagelocal.statically_safe = false
    %w = tt.load %ptr {pact.head_dim_idx = 1 : i64, pact.head_dim_size = 64 : i64,
                       pact.page_boundary_safe = false, pact.page_size = 16 : i64,
                       pact.paged_load, pact.tile_tokens = 16 : i64}
             : tensor<16x64x!tt.ptr<f16>>
    tt.return
  }
}
