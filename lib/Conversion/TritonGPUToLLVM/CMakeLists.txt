set(TRITON_GPU_TO_LLVM_TARGET_LIBS "")
if(TRITON_BUILD_CODEGEN_NVIDIA)
  list(APPEND TRITON_GPU_TO_LLVM_TARGET_LIBS MLIRGPUToNVVMTransforms)
endif()
if(TRITON_BUILD_CODEGEN_AMD)
  list(APPEND TRITON_GPU_TO_LLVM_TARGET_LIBS MLIRGPUToROCDLTransforms)
endif()

add_triton_library(TritonGPUToLLVM
    DotOpToLLVM/FMA.cpp
    DotOpToLLVM/FMADotUtility.cpp
    AllocateSharedMemory.cpp
    AllocateSharedMemoryUtility.cpp
    AllocateWarpGroups.cpp
    AssertOpToLLVM.cpp
    CanonicalizeLLVMIR.cpp
    ControlFlowOpToLLVM.cpp
    ConvertLayoutOpToLLVM.cpp
    ElementwiseOpToLLVM.cpp
    FuncOpToLLVM.cpp
    GatherOpToLLVM.cpp
    GlobalScratchMemoryAllocation.cpp
    HistogramOpToLLVM.cpp
    MakeRangeOpToLLVM.cpp
    MemoryOpToLLVM.cpp
    PrintOpToLLVM.cpp
    ReduceOpToLLVM.cpp
    ScanOpToLLVM.cpp
    SPMDOpToLLVM.cpp
    TypeConverter.cpp
    Utility.cpp
    ViewOpToLLVM.cpp
    WarpSpecializeUtility.cpp

    DEPENDS
    TritonGPUConversionPassIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRGPUDialect
    ${TRITON_GPU_TO_LLVM_TARGET_LIBS}
    MLIRGPUTransforms
    TritonAnalysis
    TritonIR
    TritonGPUIR
    TritonGPUTransforms
    TritonNvidiaGPUTransforms
)
