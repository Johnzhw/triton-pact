add_triton_library(TritonSupport
  PactDecision.cpp
  PactSMDetect.cpp

  LINK_LIBS PUBLIC
  MLIRSupport
)
