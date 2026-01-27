if (WITH_HIP)
  include(cmake/embed_hip_sources.cmake)

  add_definitions(/DDERO_HIP)
  message(STATUS "Building with HIP GPU support")

  if(NOT DEFINED HIP_PLATFORM)
    set(HIP_PLATFORM "amd")
  endif()

  if (HIP_PLATFORM MATCHES "nvidia" OR HIP_PLATFORM MATCHES "nvcc")
    # set(DIRTYBIRD_RDC "-rdc=false")
  else()
    # set(DIRTYBIRD_RDC "-fno-gpu-rdc")
  endif()

  set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} ${DIRTYBIRD_RDC}")
  unset(DIRTYBIRD_RDC CACHE)

  # Global HIP source list for the whole project
  # (this will be visible in the top-level CMake)
  list(APPEND DERO_HIP_SOURCES
    "${PROJECT_SOURCE_DIR}/src/dirtybird_hip/core/main_hip.cpp"
    "${PROJECT_SOURCE_DIR}/src/dirtybird_hip/core/test_hiprtc_isolation.cpp"
    "${PROJECT_SOURCE_DIR}/src/dirtybird_hip/core/gpu_rtc_precompile.cpp"
    "${PROJECT_SOURCE_DIR}/src/dirtybird_hip/hello-world.hip"
    "${PROJECT_SOURCE_DIR}/src/dirtybird_hip/core/devInfo.hip"
    "${PROJECT_SOURCE_DIR}/src/core/hipkill.hip"
  )

  embed_hip_sources(
      OUTPUT_FILE "${PROJECT_BINARY_DIR}/generated/dirtybird_hip_common_embedded.hpp"
      SOURCES
          "${PROJECT_SOURCE_DIR}/src/dirtybird_hip/common/stdint-jit.hip.inc"
          "${PROJECT_SOURCE_DIR}/src/dirtybird_hip/common/uint128-compat.hip.inc"
      MANIFEST_NAME COMMON_HEADERS
  )

  # DERO-only: No additional GPU algorithms currently
  # Future: Add astrobwtv3 GPU implementation here

  if (HIP_PLATFORM MATCHES "nvidia")
    add_compile_definitions(__HIP_PLATFORM_NVIDIA__)
  else()
    add_compile_definitions(__HIP_PLATFORM_AMD__)
  endif()
else()
  remove_definitions(/DDIRTYBIRD_HIP)
endif()
