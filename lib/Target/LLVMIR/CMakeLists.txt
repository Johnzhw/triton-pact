set(TRITON_LLVMIR_TARGET_LIBS "")
if(TRITON_BUILD_CODEGEN_NVIDIA)
  list(APPEND TRITON_LLVMIR_TARGET_LIBS MLIRNVVMToLLVMIRTranslation)
endif()
if(TRITON_BUILD_CODEGEN_AMD)
  list(APPEND TRITON_LLVMIR_TARGET_LIBS MLIRROCDLToLLVMIRTranslation)
endif()

add_triton_library(TritonLLVMIR
        LLVMDIScope.cpp
        LLVMDILocalVariable.cpp
        LLVMIRBreakPhiStruct.cpp
        LLVMDIUtils.cpp

        DEPENDS
        LLVMIRIncGen

        LINK_LIBS
        ${CMAKE_DL_LIBS}
        PUBLIC
        MLIRArithToLLVM
        MLIRBuiltinToLLVMIRTranslation
        MLIRIndexToLLVM
        MLIRIR
        MLIRLLVMDialect
        MLIRNVVMToLLVM
        MLIRLLVMToLLVMIRTranslation
        ${TRITON_LLVMIR_TARGET_LIBS}
        MLIRSCFToControlFlow
        MLIRSupport
        MLIRTargetLLVMIRExport
        TritonGPUToLLVM
        )

set_source_files_properties(
        LLVMIRTranslation.cpp
        PROPERTIES
        COMPILE_FLAGS "-D__BUILD_DIR__=\\\"${CMAKE_BINARY_DIR}\\\"")
