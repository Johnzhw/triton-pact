add_triton_library(TritonSupport
  PactDecision.cpp
  PactPgoDecision.cpp
  PactSMDetect.cpp

  LINK_LIBS PUBLIC
  MLIRSupport
)
