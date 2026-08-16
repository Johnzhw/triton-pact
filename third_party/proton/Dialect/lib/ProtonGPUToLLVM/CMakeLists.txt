add_triton_library(ProtonGPUToLLVM
    AllocateProtonSharedMemory.cpp
    PatternProtonGPUOpToLLVM.cpp
    Utility.cpp

    DEPENDS
    ProtonGPUConversionPassIncGen

    LINK_LIBS PUBLIC
    ProtonIR
    ProtonGPUIR
    ProtonAnalysis
)

if(TRITON_BUILD_CODEGEN_NVIDIA)
  add_subdirectory(ProtonNvidiaGPUToLLVM)
endif()
if(TRITON_BUILD_CODEGEN_AMD)
  add_subdirectory(ProtonAMDGPUToLLVM)
endif()
