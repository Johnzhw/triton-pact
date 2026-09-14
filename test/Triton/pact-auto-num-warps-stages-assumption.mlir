// RUN: env PACT_SM_VERSION=86 PACT_ENABLE_AUTO_NUM_WARPS=1 triton-opt %s -triton-pact-auto-num-warps | FileCheck %s
// RUN: env PACT_SM_VERSION=86 PACT_ENABLE_AUTO_NUM_WARPS=1 PACT_MAX_PIPELINE_STAGES=4 triton-opt %s -triton-pact-auto-num-warps | FileCheck %s --check-prefix=PINNED
// RUN: env PACT_SM_VERSION=86 PACT_ENABLE_AUTO_NUM_WARPS=1 PACT_OVERRIDE_STAGES=2 triton-opt %s -triton-pact-auto-num-warps | FileCheck %s --check-prefix=OVR2
//
// P11's stage assumption is a *model input*, not only a P6 cap.
//
// `PACT_MAX_PIPELINE_STAGES` is documented as P6's stage ceiling with default 4.
// AutoNumWarps used to read it with `std::getenv` and inherit that 4 whenever the
// knob was unset, even though the theory path was calibrated with
// stagesPerBlock=3 (which is what makes SM80 pick 2 warps on a 16x64 f16 tile).
// The default is therefore 3, and only an *explicit* value may move it.
//
// Both runs must also keep the P11 decision itself unchanged: the warp choice on
// SM86/16x64 is 4 either way, so this file guards the assumption, not the choice.
//
// V10-P0 (BEH-1b): an explicit PACT_OVERRIDE_STAGES is what P6 pins the loop to,
// so the assumption aligns with it (pin beats cap; 2..8 domain, same as P6).
// The 16x64 warp choice stays 4 under assumption 2 as well — the alignment only
// matters inside the D128 flip window (see pact_paper suite/results/v10/arch_review.md §1.4).

module {
  tt.func @two_paged_loads(%p0: !tt.ptr<f16>, %p1: !tt.ptr<f16>) {
    %range16 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %range64 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %e16 = tt.expand_dims %range16 {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %e64 = tt.expand_dims %range64 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %b16 = tt.broadcast %e16 : tensor<16x1xi32> -> tensor<16x64xi32>

    %s0 = tt.splat %p0 : !tt.ptr<f16> -> tensor<16x64x!tt.ptr<f16>>
    %ptr0 = tt.addptr %s0, %b16 : tensor<16x64x!tt.ptr<f16>>, tensor<16x64xi32>
    %v0 = tt.load %ptr0 {pact.head_dim_size = 64 : i64, pact.page_size = 64 : i64,
                         pact.paged_load, pact.tile_tokens = 16 : i64}
          : tensor<16x64x!tt.ptr<f16>>

    %s1 = tt.splat %p1 : !tt.ptr<f16> -> tensor<16x64x!tt.ptr<f16>>
    %ptr1 = tt.addptr %s1, %b16 : tensor<16x64x!tt.ptr<f16>>, tensor<16x64xi32>
    %v1 = tt.load %ptr1 {pact.head_dim_size = 64 : i64, pact.page_size = 64 : i64,
                         pact.paged_load, pact.tile_tokens = 16 : i64}
          : tensor<16x64x!tt.ptr<f16>>
    tt.return
  }
}

// Default (knob unset): the historical stagesPerBlock=3, never the P6 default 4.
// CHECK: module attributes {pact.optimal_num_warps = 4 : i32
// CHECK-SAME: pact.p11.stages_assumption = 3 : i32

// An explicit knob value is honoured and changes only the assumption.
// PINNED: module attributes {pact.optimal_num_warps = 4 : i32
// PINNED-SAME: pact.p11.stages_assumption = 4 : i32

// An explicit PACT_OVERRIDE_STAGES aligns the assumption with P6's pin
// (and beats PACT_MAX_PIPELINE_STAGES when both are set); on this 16x64
// tile the warp choice itself is unchanged under assumption 2.
// OVR2: module attributes {pact.optimal_num_warps = 4 : i32
// OVR2-SAME: pact.p11.stages_assumption = 2 : i32
