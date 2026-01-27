#pragma once

// DERO Miner - GPU libraries header (stripped for DERO-only)

#ifdef DIRTYBIRD_HIP
#include <dirtybird_hip/hello.hpp>
#include <dirtybird_hip/core/test_hiprtc_isolation.hpp>
#endif

inline int GPUTest() {
  #ifdef DIRTYBIRD_HIP
    // Run HIPRTC isolation test FIRST, before any other GPU work
    test_hiprtc_isolation();
  #endif
  return 0;
}