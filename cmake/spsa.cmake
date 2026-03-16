if(USE_ASTRO_SPSA)
  message(STATUS "=== SPSA Configuration ===")
  message(STATUS "USE_ASTRO_SPSA: ${USE_ASTRO_SPSA}")

  set(SPSA_LIB_VARIANT "auto" CACHE STRING "SPSA static library variant: auto|x86-64-v2|x86-64-v3|x86-64-v4|znver3|znver4|znver5")
  set_property(CACHE SPSA_LIB_VARIANT PROPERTY STRINGS auto x86-64-v2 x86-64-v3 x86-64-v4 znver3 znver4 znver5)

  if(WIN32)
    set(SPSA_OS_PREFIX "win")
  elseif(APPLE)
    set(SPSA_OS_PREFIX "macos")
  else()
    set(SPSA_OS_PREFIX "linux")
  endif()
  message(STATUS "SPSA_OS_PREFIX: ${SPSA_OS_PREFIX}")
  message(STATUS "TARGET_ARCH: ${TARGET_ARCH}")
  message(STATUS "CPU_ARCHTARGET: ${CPU_ARCHTARGET}")
  message(STATUS "SPSA_LIB_VARIANT: ${SPSA_LIB_VARIANT}")

  # NOTE: -flto removed here — already handled by main LTO block in CMakeLists.txt
  # (having it here bypassed DISABLE_LTO/USE_THIN_LTO controls)
  set(CMAKE_C_FLAGS_RELEASE   "${CMAKE_C_FLAGS_RELEASE} -DUSE_ASTRO_SPSA=ON")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DUSE_ASTRO_SPSA=ON")

  if(EXISTS ${PROJECT_SOURCE_DIR}/lib/astrospsa)
    set(SPSA_LIB_DIR ${PROJECT_SOURCE_DIR}/lib/astrospsa)
  else()
    include(FetchContent)
    ## Fetch the static library
    FetchContent_Declare(
        astrospsa
        GIT_REPOSITORY https://gitlab.com/Tritonn204/astro-spsa.git
        GIT_TAG        8938667bfa3253b52c622daf575fc11674d7067b
        # GIT_REPOSITORY https://gitlab.com/dirkerdero/astro-spsa-dirker.git
        # GIT_TAG        1a2acdd2ed187a9b565c33028dba4052a89f875b
    )
    FetchContent_MakeAvailable(astrospsa)
    set(SPSA_LIB_DIR ${astrospsa_SOURCE_DIR})
  endif()
  cmake_print_variables(SPSA_LIB_DIR)
  include_directories(${SPSA_LIB_DIR})

  # CMAKE_LIB_SUFFIX was set in CMakeLists.txt already
  set(_spsa_suffixes "")
  if(DEFINED CMAKE_LIB_SUFFIX AND NOT "${CMAKE_LIB_SUFFIX}" STREQUAL "")
    list(APPEND _spsa_suffixes "${CMAKE_LIB_SUFFIX}")
  endif()
  list(APPEND _spsa_suffixes "_clang_20" "_clang_19" "_clang_18" "_clang_17" "_clang_16" "")
  list(REMOVE_DUPLICATES _spsa_suffixes)

  set(SPSA_FULL_LIB_PATH "")
  string(TOLOWER "${SPSA_LIB_VARIANT}" _spsa_variant_lower)

  if(NOT "${_spsa_variant_lower}" STREQUAL "auto")
    set(_spsa_variant_candidates "")
    file(GLOB _spsa_variant_glob
      "${SPSA_LIB_DIR}/libastroSPSA_${SPSA_OS_PREFIX}_${TARGET_ARCH}_clang_*_${_spsa_variant_lower}.a"
      "${SPSA_LIB_DIR}/libastroSPSA_${SPSA_OS_PREFIX}_${TARGET_ARCH}_${_spsa_variant_lower}.a"
    )
    list(APPEND _spsa_variant_candidates ${_spsa_variant_glob})
    list(REMOVE_DUPLICATES _spsa_variant_candidates)
    list(LENGTH _spsa_variant_candidates _spsa_variant_count)
    if(_spsa_variant_count GREATER 0)
      list(SORT _spsa_variant_candidates COMPARE NATURAL ORDER DESCENDING)
      list(GET _spsa_variant_candidates 0 SPSA_FULL_LIB_PATH)
    endif()
  endif()

  if("${SPSA_FULL_LIB_PATH}" STREQUAL "" AND "${_spsa_variant_lower}" STREQUAL "auto")
    foreach(_sfx IN LISTS _spsa_suffixes)
      if(EXISTS ${SPSA_LIB_DIR}/libastroSPSA_${SPSA_OS_PREFIX}_${TARGET_ARCH}${_sfx}_${CPU_ARCHTARGET}.a)
        set(SPSA_FULL_LIB_PATH ${SPSA_LIB_DIR}/libastroSPSA_${SPSA_OS_PREFIX}_${TARGET_ARCH}${_sfx}_${CPU_ARCHTARGET}.a)
        break()
      elseif(EXISTS ${SPSA_LIB_DIR}/libastroSPSA_${SPSA_OS_PREFIX}_${TARGET_ARCH}${_sfx}.a)
        set(SPSA_FULL_LIB_PATH ${SPSA_LIB_DIR}/libastroSPSA_${SPSA_OS_PREFIX}_${TARGET_ARCH}${_sfx}.a)
        break()
      elseif(EXISTS ${SPSA_LIB_DIR}/libastroSPSA_${SPSA_OS_PREFIX}_${TARGET_ARCH}.a)
        set(SPSA_FULL_LIB_PATH ${SPSA_LIB_DIR}/libastroSPSA_${SPSA_OS_PREFIX}_${TARGET_ARCH}.a)
        break()
      elseif(EXISTS ${SPSA_LIB_DIR}/libastroSPSA_${SPSA_OS_PREFIX}.a)
        set(SPSA_FULL_LIB_PATH ${SPSA_LIB_DIR}/libastroSPSA_${SPSA_OS_PREFIX}.a)
        break()
      endif()
    endforeach()
  endif()

  if("${SPSA_FULL_LIB_PATH}" STREQUAL "" AND NOT "${_spsa_variant_lower}" STREQUAL "auto")
    message(STATUS "SPSA explicit variant search failed!")
    message(STATUS "Requested variant: ${SPSA_LIB_VARIANT}")
    message(STATUS "Searched in: ${SPSA_LIB_DIR}")
    file(GLOB SPSA_AVAILABLE_FILES "${SPSA_LIB_DIR}/libastroSPSA*.a")
    message(STATUS "Available SPSA libraries:")
    foreach(_lib IN LISTS SPSA_AVAILABLE_FILES)
      message(STATUS "  ${_lib}")
    endforeach()
    message(FATAL_ERROR "Requested SPSA_LIB_VARIANT='${SPSA_LIB_VARIANT}' was not found.")
  elseif("${SPSA_FULL_LIB_PATH}" STREQUAL "")
    message(STATUS "SPSA library search failed!")
    message(STATUS "Searched in: ${SPSA_LIB_DIR}")
    message(STATUS "Pattern: libastroSPSA_${SPSA_OS_PREFIX}_${TARGET_ARCH}*_${CPU_ARCHTARGET}.a")
    message(STATUS "Available suffixes tried: ${_spsa_suffixes}")
    # List available files for debugging
    file(GLOB SPSA_AVAILABLE_FILES "${SPSA_LIB_DIR}/libastroSPSA*.a")
    message(STATUS "Available SPSA libraries:")
    foreach(_lib IN LISTS SPSA_AVAILABLE_FILES)
      message(STATUS "  ${_lib}")
    endforeach()
    message(FATAL_ERROR "SPSA lib was not found: ${SPSA_LIB_DIR}")
  else()
    message(STATUS "=== SPSA Library Found ===")
    message(STATUS "SPSA_FULL_LIB_PATH: ${SPSA_FULL_LIB_PATH}")
  endif()
endif()
