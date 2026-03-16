# HIP support was added in CMake 3.21
cmake_minimum_required(VERSION 3.21)

#cmake_policy(SET CMP0156 NEW)

include(CMakePrintHelpers)
cmake_print_variables(CMAKE_HOST_SYSTEM_NAME)

# TODO: Rename WIN_CROSS to something more intelligent.  And maybe use it to specific the -march value?

# Specify C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

set(TARGET_ARCH "amd64")
set(CMAKE_BUILD_TYPE Release)
set(DIRTYBIRD_PREPROJECT_ANDROID OFF)
if ((DEFINED ANDROID_ABI AND NOT "${ANDROID_ABI}" STREQUAL "") OR
    (DEFINED CMAKE_TOOLCHAIN_FILE AND CMAKE_TOOLCHAIN_FILE MATCHES "android.toolchain.cmake"))
  set(DIRTYBIRD_PREPROJECT_ANDROID ON)
endif()

# =============================================================================
# PERFORMANCE OPTIMIZATION FLAGS
# =============================================================================
# Base optimization level
# THERMAL_MODE uses -O2 (cooler code) instead of -Ofast (hotter but faster per-hash)
option(THERMAL_MODE "Use -O2 for cooler sustained performance on thermally-limited CPUs" OFF)
option(ENABLE_LOOP_UNROLL "Enable -funroll-loops in release builds" ON)
if (THERMAL_MODE)
    message(STATUS "THERMAL MODE: Using -O2 for cooler sustained performance")
    set(CMAKE_C_FLAGS_RELEASE   "${CMAKE_C_FLAGS_RELEASE}   -O2 -DNDEBUG")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2 -Wuninitialized -DNDEBUG")
else()
    set(CMAKE_C_FLAGS_RELEASE   "${CMAKE_C_FLAGS_RELEASE}   -Ofast -DNDEBUG")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast -Wuninitialized -DNDEBUG")
endif()

# Loop unrolling for tight mining loops (RC4, SHA256, Salsa20)
if (ENABLE_LOOP_UNROLL)
    set(CMAKE_C_FLAGS_RELEASE   "${CMAKE_C_FLAGS_RELEASE}   -funroll-loops")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -funroll-loops")
endif()

# Omit frame pointer - frees up a register for hot code paths
set(CMAKE_C_FLAGS_RELEASE   "${CMAKE_C_FLAGS_RELEASE}   -fomit-frame-pointer")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fomit-frame-pointer")

# Aggressive inlining for crypto primitives
# Inlining: let LTO decide inline thresholds (LTO overrides compile-time settings)
# Removing aggressive inline threshold reduces .text size for better I-cache
option(AGGRESSIVE_INLINE "Enable aggressive function inlining" ON)
if (AGGRESSIVE_INLINE)
    set(CMAKE_C_FLAGS_RELEASE   "${CMAKE_C_FLAGS_RELEASE}   -finline-functions")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -finline-functions")
    # NOTE: Clang inline threshold is set after project() where CMAKE_CXX_COMPILER_ID is available
    # (see LTO Configuration section below)
endif()

# Use native CPU tuning for local builds (better cache/pipeline utilization)
option(TUNE_NATIVE "Enable -mtune=native for local builds" OFF)
if (TUNE_NATIVE AND NOT WITH_HIP AND NOT DIRTYBIRD_PREPROJECT_ANDROID)
    message(STATUS "Native CPU tuning enabled (-mtune=native)")
    set(CMAKE_C_FLAGS_RELEASE   "${CMAKE_C_FLAGS_RELEASE}   -mtune=native")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -mtune=native")
elseif (TUNE_NATIVE AND DIRTYBIRD_PREPROJECT_ANDROID)
    message(STATUS "Skipping -mtune=native for Android cross-build")
endif()

# Thin LTO for faster link times with similar performance
option(USE_THIN_LTO "Use Thin LTO instead of Full LTO (faster builds)" OFF)

# =============================================================================
# MAXIMUM PERFORMANCE PRESET
# =============================================================================
# For local builds where you want absolute maximum performance:
#   cmake -DMAX_PERFORMANCE=ON -DTUNE_NATIVE=ON -B build-max
#
# This enables all aggressive optimizations suitable for the local CPU
option(MAX_PERFORMANCE "Enable maximum performance optimizations" ON)
option(MAX_PERFORMANCE_DISABLE_AUTO_VECTORIZATION
       "When MAX_PERFORMANCE is ON, disable LLVM auto-vectorization and SLP vectorization"
       ON)
set(MAX_PERFORMANCE_PREFETCH_DISTANCE "256" CACHE STRING
    "Clang -mllvm prefetch-distance value used by MAX_PERFORMANCE; set to OFF to disable")
set(DIRTYBIRD_DEFERRED_CLANG_C_FLAGS_RELEASE "")
set(DIRTYBIRD_DEFERRED_CLANG_CXX_FLAGS_RELEASE "")
set(DIRTYBIRD_WINDOWS_CLANG_LINK_FLAGS "")
if (MAX_PERFORMANCE)
    message(STATUS "MAX_PERFORMANCE mode enabled - all optimizations active")
    set(TUNE_NATIVE ON CACHE BOOL "" FORCE)
    set(AGGRESSIVE_INLINE ON CACHE BOOL "" FORCE)
    set(WITH_AVX2 ON CACHE BOOL "" FORCE)
    # Prefetch distance tuning for mining workloads
    if (NOT MAX_PERFORMANCE_PREFETCH_DISTANCE STREQUAL "OFF")
        # project() has not run yet here, so compiler ID is not final.
        # Stash Clang-only flags and apply them later once the toolchain exists.
        set(DIRTYBIRD_DEFERRED_CLANG_C_FLAGS_RELEASE
            "${DIRTYBIRD_DEFERRED_CLANG_C_FLAGS_RELEASE} -mllvm -prefetch-distance=${MAX_PERFORMANCE_PREFETCH_DISTANCE}")
        set(DIRTYBIRD_DEFERRED_CLANG_CXX_FLAGS_RELEASE
            "${DIRTYBIRD_DEFERRED_CLANG_CXX_FLAGS_RELEASE} -mllvm -prefetch-distance=${MAX_PERFORMANCE_PREFETCH_DISTANCE}")
    endif()
    if (MAX_PERFORMANCE_DISABLE_AUTO_VECTORIZATION)
        # Disable auto-vectorization: lookup compute path uses scalar table lookups
        # AVX2 auto-vectorization has historically increased thermal throttling on
        # sustained workloads, so keep the old behavior as the default preset.
        set(CMAKE_C_FLAGS_RELEASE   "${CMAKE_C_FLAGS_RELEASE}   -fno-vectorize -fno-slp-vectorize")
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fno-vectorize -fno-slp-vectorize")
    endif()
endif()
# =============================================================================

option(USE_GCC        "Use GCC instead of Clang" OFF)
option(USE_ASTRO_SPSA "Use Astro Stamped Permutation Suffix Array library" ON)
option(USE_DLUNA_RADIX_SA "Use DeroLuna-style 24-bit prefix radix+collision SA" OFF)  # Experimental: faster in some benches but not reference-stable
option(USE_RADIX_SA   "Use DeroHE-style 2-pass radix sort SA (cache-friendly, lower thermal)" OFF)
option(USE_BUCKET_SA  "Use 2-byte prefix bucket sort SA (optimized for random bytes)" OFF)
option(USE_CUSTOM_SA  "Use custom 70KB SA-IS algorithm instead of divsufsort" OFF)  # Disabled: custom_sa has bugs and is 4x slower
option(USE_LIBSAIS   "Use libsais 2.10.4 for faster suffix array construction" ON)  # State-of-the-art SA library
option(DIRTYBIRD_SIMD_SALSA20 "Enable SIMD Salsa20 in the CPU hot path" ON)
option(DIRTYBIRD_CRYPTOGAMS_RC4_DUAL "Enable dual-state CRYPTOGAMS RC4 path" ON)
option(DIRTYBIRD_FAST_RC4 "Enable FastRc4 path" OFF)
option(WIN_CROSS      "Whether we are cross-compiling on Linux for Windows" OFF)
option(DISABLE_LTO    "Disable Link-Time Optimization" OFF)

# mimalloc: High-performance allocator (15-30% faster heap ops, better multi-thread scaling)
# Overrides operator new/delete globally, including for precompiled SPSA library
option(USE_MIMALLOC   "Use mimalloc allocator for faster heap operations" ON)

# OpenMP support for parallel suffix array construction (like TNN/Luna)
option(WITH_OPENMP    "Enable OpenMP for parallel SA construction in divsufsort" ON)

if (DIRTYBIRD_CRYPTOGAMS_RC4_DUAL AND DIRTYBIRD_FAST_RC4)
  message(FATAL_ERROR "DIRTYBIRD_CRYPTOGAMS_RC4_DUAL and DIRTYBIRD_FAST_RC4 cannot both be enabled.")
endif()

if (DIRTYBIRD_SIMD_SALSA20)
  add_compile_definitions(USE_SIMD_SALSA20=1)
endif()

if (DIRTYBIRD_CRYPTOGAMS_RC4_DUAL)
  add_compile_definitions(USE_CRYPTOGAMS_RC4_DUAL=1)
endif()

if (DIRTYBIRD_FAST_RC4)
  add_compile_definitions(USE_FAST_RC4=1)
endif()

# HIP_PLATFORM: "" (none), "amd", or "nvidia"
set(HIP_PLATFORM "${HIP_PLATFORM}" CACHE STRING "HIP platform: amd or nvidia")
set_property(CACHE HIP_PLATFORM PROPERTY STRINGS "" "amd" "nvidia")

# ----------------- Select C / C++ / HIP compilers -----------------
if (DIRTYBIRD_PREPROJECT_ANDROID)
  message(STATUS "Target system is Android (toolchain-driven build)")

elseif (WIN32)

  # ---------- Windows + HIP (MSVC / clang-cl toolchain) ----------
  if (WITH_HIP)
    # Resolve HIP_PATH from CMake var or environment
      if (NOT DEFINED HIP_PATH AND DEFINED ENV{HIP_PATH})
        set(HIP_PATH "$ENV{HIP_PATH}" CACHE PATH "HIP root" FORCE)
      endif()

      if (NOT DEFINED HIP_PATH)
        message(FATAL_ERROR "WITH_HIP=ON on Windows but HIP_PATH is not set (CMake or ENV).")
      endif()

      if (HIP_PLATFORM STREQUAL "")
        set(HIP_PLATFORM "amd" CACHE STRING "HIP platform" FORCE)
      endif()

      # HIP CMake modules + ROCm libs
      file(GLOB _hip_modules "${HIP_PATH}/lib/cmake/*")
      list(APPEND CMAKE_MODULE_PATH ${_hip_modules})
      link_directories("${HIP_PATH}/lib")

# ---------- Windows CPU-only ----------
  else()
    message(STATUS "Target system is Windows (CPU-only build)")

    if (USE_GCC)
      set(_DIRTYBIRD_WIN_C_NAMES gcc.exe gcc)
      set(_DIRTYBIRD_WIN_CXX_NAMES g++.exe g++)
    else()
      set(_DIRTYBIRD_WIN_C_NAMES clang.exe clang)
      set(_DIRTYBIRD_WIN_CXX_NAMES clang++.exe clang++)
    endif()

    # Resolve toolchain binaries from the active MSYS2 environment instead of
    # assuming a fixed C:/msys64 install root. GitHub-hosted runners install
    # MSYS2 under a temp directory, so hardcoded paths break release builds.
    if (CMAKE_C_COMPILER AND EXISTS "${CMAKE_C_COMPILER}")
      set(DIRTYBIRD_WIN_C_COMPILER "${CMAKE_C_COMPILER}")
    else()
      find_program(DIRTYBIRD_WIN_C_COMPILER NAMES ${_DIRTYBIRD_WIN_C_NAMES} REQUIRED)
    endif()

    if (CMAKE_CXX_COMPILER AND EXISTS "${CMAKE_CXX_COMPILER}")
      set(DIRTYBIRD_WIN_CXX_COMPILER "${CMAKE_CXX_COMPILER}")
    else()
      find_program(DIRTYBIRD_WIN_CXX_COMPILER NAMES ${_DIRTYBIRD_WIN_CXX_NAMES} REQUIRED)
    endif()

    if (CMAKE_ASM_COMPILER AND EXISTS "${CMAKE_ASM_COMPILER}")
      set(DIRTYBIRD_WIN_ASM_COMPILER "${CMAKE_ASM_COMPILER}")
    else()
      find_program(DIRTYBIRD_WIN_ASM_COMPILER NAMES ${_DIRTYBIRD_WIN_C_NAMES} REQUIRED)
    endif()

    set(CMAKE_C_COMPILER   "${DIRTYBIRD_WIN_C_COMPILER}"   CACHE FILEPATH "" FORCE)
    set(CMAKE_CXX_COMPILER "${DIRTYBIRD_WIN_CXX_COMPILER}" CACHE FILEPATH "" FORCE)
    set(CMAKE_ASM_COMPILER "${DIRTYBIRD_WIN_ASM_COMPILER}" CACHE FILEPATH "" FORCE)

    get_filename_component(DIRTYBIRD_MSYS2_BIN_DIR "${DIRTYBIRD_WIN_C_COMPILER}" DIRECTORY)
    get_filename_component(DIRTYBIRD_MSYS2_PREFIX "${DIRTYBIRD_MSYS2_BIN_DIR}/.." ABSOLUTE)
    if (EXISTS "${DIRTYBIRD_MSYS2_PREFIX}/include")
      set(DIRTYBIRD_MSYS2_INCLUDE_DIR "${DIRTYBIRD_MSYS2_PREFIX}/include")
      set(DIRTYBIRD_MSYS2_LIB_DIR "${DIRTYBIRD_MSYS2_PREFIX}/lib")
    endif()

    if (NOT USE_GCC)
      if (CMAKE_LINKER AND EXISTS "${CMAKE_LINKER}" AND
          CMAKE_LINKER MATCHES "([/\\\\]|^)ld\\.lld(\\.exe)?$")
        set(DIRTYBIRD_WIN_LINKER "${CMAKE_LINKER}")
      else()
        find_program(DIRTYBIRD_WIN_LINKER
          NAMES ld.lld.exe ld.lld
          HINTS "${DIRTYBIRD_MSYS2_BIN_DIR}"
          REQUIRED)
      endif()

      set(CMAKE_LINKER "${DIRTYBIRD_WIN_LINKER}" CACHE FILEPATH "" FORCE)
      set(DIRTYBIRD_WINDOWS_CLANG_LINK_FLAGS
          "-fuse-ld=lld --ld-path=${DIRTYBIRD_WIN_LINKER}")
      message(STATUS "Windows Clang linker: ${DIRTYBIRD_WIN_LINKER}")
    endif()
  endif()

# ----------------- Non-Windows platforms -----------------
else()
  if (USE_GCC)
    find_program(DIRTYBIRD_C_COMPILER NAMES gcc REQUIRED)
    find_program(DIRTYBIRD_CXX_COMPILER NAMES g++ REQUIRED)
    set(CMAKE_C_COMPILER   "${DIRTYBIRD_C_COMPILER}"   CACHE FILEPATH "" FORCE)
    set(CMAKE_CXX_COMPILER "${DIRTYBIRD_CXX_COMPILER}" CACHE FILEPATH "" FORCE)
    set(CMAKE_ASM_COMPILER "${DIRTYBIRD_C_COMPILER}"   CACHE FILEPATH "" FORCE)
  else()
    find_program(DIRTYBIRD_C_COMPILER NAMES clang REQUIRED)
    find_program(DIRTYBIRD_CXX_COMPILER NAMES clang++ REQUIRED)
    set(CMAKE_C_COMPILER   "${DIRTYBIRD_C_COMPILER}"   CACHE FILEPATH "" FORCE)
    set(CMAKE_CXX_COMPILER "${DIRTYBIRD_CXX_COMPILER}" CACHE FILEPATH "" FORCE)
    set(CMAKE_ASM_COMPILER "${DIRTYBIRD_C_COMPILER}"   CACHE FILEPATH "" FORCE)
  endif()
endif()

set(CMAKE_COLOR_DIAGNOSTICS ON)
# force C colored diagnostic output
set(CFLAGS "${CFLAGS} -fdiagnostics-color=always")
# force C++ colored diagnostic output
set(CXXFLAGS "${CXXFLAGS} -fdiagnostics-color=always")

# This looks weird, but we prefer the -D variable over the ENV var
string(TIMESTAMP VERSION_DEF "%Y.%m.%d.%H")
set(INTERNAL_DERO_VERSION ${VERSION_DEF})
if(DEFINED ENV{DERO_VERSION})
  set(INTERNAL_DERO_VERSION $ENV{DERO_VERSION})
endif()
if(DEFINED DERO_VERSION)
  set(INTERNAL_DERO_VERSION ${DERO_VERSION})
endif()

string(REGEX REPLACE "^[a-zA-Z]" ""  INTERNAL_DERO_VERSION ${INTERNAL_DERO_VERSION})

set(DIRTYBIRD_VERSION_DEFAULT "1.0.0-dev")
if(DEFINED DIRTYBIRD_VERSION)
  set(INTERNAL_DIRTYBIRD_VERSION "${DIRTYBIRD_VERSION}")
elseif(DEFINED ENV{DIRTYBIRD_VERSION})
  set(INTERNAL_DIRTYBIRD_VERSION "$ENV{DIRTYBIRD_VERSION}")
else()
  set(INTERNAL_DIRTYBIRD_VERSION "${DIRTYBIRD_VERSION_DEFAULT}")
endif()
string(REGEX REPLACE "^v" "" INTERNAL_DIRTYBIRD_VERSION "${INTERNAL_DIRTYBIRD_VERSION}")
set(DIRTYBIRD_VERSION "${INTERNAL_DIRTYBIRD_VERSION}" CACHE STRING "Public DIRTYBIRD version" FORCE)

# DERO Miner Options - AstroBWTv3 only
option(WITH_ASTROBWTV3      "Enable AstroBWTv3 (DERO) algorithm" ON)
# option(WITH_ARGON2          "Enable Argon2 algorithms family" ON)
# option(WITH_KAWPOW          "Enable KawPow algorithms family" ON)
# option(WITH_GHOSTRIDER      "Enable GhostRider algorithm" ON)
# option(WITH_EMBEDDED_CONFIG "Enable internal embedded JSON config" OFF)
# option(WITH_OPENCL          "Enable OpenCL backend" ON)
set(WITH_OPENCL_VERSION 200 CACHE STRING "Target OpenCL version")
set_property(CACHE WITH_OPENCL_VERSION PROPERTY STRINGS 120 200 210 220)
option(WITH_HIP             "Enable GPU support through the HIP SDK" OFF)
# option(WITH_CUDA            "Enable CUDA backend" ON)
# option(WITH_NVML            "Enable NVML (NVIDIA Management Library) support (only if CUDA backend enabled)" ON)
# option(WITH_ADL             "Enable ADL (AMD Display Library) or sysfs support (only if OpenCL backend enabled)" ON)
# option(WITH_STRICT_CACHE    "Enable strict checks for OpenCL cache" ON)
option(WITH_INTERLEAVE_DEBUG_LOG "Enable debug log for threads interleave" OFF)
option(WITH_PROFILING       "Enable profiling for developers" OFF)
option(WITH_SSE4_1          "Enable SSE 4.1" ON)
option(WITH_AVX2            "Enable AVX2" ON)
option(WITH_AVX512          "Enable AVX-512 (F, BW, VL, DQ, CD) - for Zen5/Ice Lake+" OFF)
# option(WITH_BENCHMARK       "Enable builtin RandomX benchmark and stress test" ON)
option(WITH_SECURE_JIT      "Enable secure access to JIT memory" OFF)
option(WITH_DMI             "Enable DMI/SMBIOS reader" ON)

# SA Instrumentation for collecting access pattern metrics (development/profiling)
option(ENABLE_SA_INSTRUMENTATION "Enable divsufsort access counting for SA analysis" OFF)

option(BUILD_STATIC         "Build static binary" ON)
option(ARM_V8               "Force ARMv8 (64 bit) architecture, use with caution if automatic detection fails, but you sure it may work" OFF)
option(ARM_V7               "Force ARMv7 (32 bit) architecture, use with caution if automatic detection fails, but you sure it may work" OFF)
set(DIRTYBIRD_LIBSAIS_DIR "${CMAKE_SOURCE_DIR}/extern/libsais" CACHE PATH
    "Bundled libsais source directory")
set(DIRTYBIRD_LIBSAIS_SOURCE "${DIRTYBIRD_LIBSAIS_DIR}/src/libsais.c")
set(DIRTYBIRD_LIBSAIS_INCLUDE_DIR "${DIRTYBIRD_LIBSAIS_DIR}/include")
# End of XMRig options

# LTO, section GC, and strip are configured AFTER project() since they need CMAKE_CXX_COMPILER_ID
# See "LTO Configuration" block below project()

if (WITH_HIP)
  set(USE_ASTRO_SPSA OFF)

  # ----- HIP on Windows: use clang-cl / MSVC-style toolchain -----
  if (WIN32)
    if (NOT CMAKE_GENERATOR MATCHES "Ninja")
      message(FATAL_ERROR
        "WITH_HIP=ON on Windows requires the Ninja generator.\n"
        "Current generator: '${CMAKE_GENERATOR}'.\n"
        "Run CMake like:\n"
        "  cmake -S . -B hip-build/win32 -G \"Ninja\" -DWITH_HIP=ON -DHIP_PATH=\"C:/Program Files/AMD/ROCm/6.x\""
      )
    endif()

    add_definitions(-DBOOST_ALL_NO_LIB)
    add_definitions(-D_WIN32_WINNT=0x0A00)
  endif()

  # ----- Common HIP CMake modules / library dirs (all platforms) -----
  if (DEFINED HIP_PATH)
    file(GLOB HIP_MODULES
      "${HIP_PATH}/lib/cmake/*"
    )
    list(APPEND CMAKE_MODULE_PATH ${HIP_MODULES})

    link_directories("${HIP_PATH}/lib")
  endif()

  set(HIP_LANG HIP)
endif()

if(APPLE)
  set(_LLVM_BIN "/opt/homebrew/opt/llvm@18/bin")

  find_program(_HB_LLVM_AR     NAMES llvm-ar     PATHS "${_LLVM_BIN}" NO_DEFAULT_PATH)
  find_program(_HB_LLVM_RANLIB NAMES llvm-ranlib PATHS "${_LLVM_BIN}" NO_DEFAULT_PATH)
  find_program(_HB_LLVM_NM     NAMES llvm-nm     PATHS "${_LLVM_BIN}" NO_DEFAULT_PATH)

  if(_HB_LLVM_AR AND _HB_LLVM_RANLIB)
    set(CMAKE_AR     "${_HB_LLVM_AR}"     CACHE FILEPATH "Archiver" FORCE)
    set(CMAKE_RANLIB "${_HB_LLVM_RANLIB}" CACHE FILEPATH "Ranlib"   FORCE)
  endif()

  if(_HB_LLVM_NM)
    set(CMAKE_NM "${_HB_LLVM_NM}" CACHE FILEPATH "NM" FORCE)
  endif()
endif()

project(
  dero-miner
  LANGUAGES C CXX ASM
  DESCRIPTION "High-Performance DERO Miner with SPSA, Incremental SA, and AVX-512 RC4"
)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  if (NOT DIRTYBIRD_DEFERRED_CLANG_C_FLAGS_RELEASE STREQUAL "")
    set(CMAKE_C_FLAGS_RELEASE
        "${CMAKE_C_FLAGS_RELEASE}${DIRTYBIRD_DEFERRED_CLANG_C_FLAGS_RELEASE}")
  endif()
  if (NOT DIRTYBIRD_DEFERRED_CLANG_CXX_FLAGS_RELEASE STREQUAL "")
    set(CMAKE_CXX_FLAGS_RELEASE
        "${CMAKE_CXX_FLAGS_RELEASE}${DIRTYBIRD_DEFERRED_CLANG_CXX_FLAGS_RELEASE}")
  endif()
  if (NOT MAX_PERFORMANCE_PREFETCH_DISTANCE STREQUAL "OFF")
    message(STATUS "Clang MAX_PERFORMANCE prefetch distance: ${MAX_PERFORMANCE_PREFETCH_DISTANCE}")
  endif()
endif()

# =============================================================================
# LTO Configuration (must be after project() so CMAKE_CXX_COMPILER_ID is set)
# =============================================================================
if (NOT WITH_HIP)
  if (NOT DISABLE_LTO)
    if (USE_THIN_LTO AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
      message(STATUS "Enabling Thin LTO for non-HIP build (faster link, similar perf)")
      set(CMAKE_C_FLAGS_RELEASE   "${CMAKE_C_FLAGS_RELEASE} -flto=thin")
      set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -flto=thin")
      set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -flto=thin")
    else()
      message(STATUS "Enabling Full LTO for non-HIP build")
      set(CMAKE_C_FLAGS_RELEASE   "${CMAKE_C_FLAGS_RELEASE} -flto")
      set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -flto")
    endif()

    # LTO backend optimization level: -O3 enables aggressive cross-module IPO
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--lto-O3")
    message(STATUS "LTO backend optimization level: O3")
  else()
    message(STATUS "LTO disabled by DISABLE_LTO option")
  endif()

  # Section-level dead code elimination (works with LTO for aggressive cleanup)
  set(CMAKE_C_FLAGS_RELEASE   "${CMAKE_C_FLAGS_RELEASE}   -ffunction-sections -fdata-sections")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -ffunction-sections -fdata-sections")

  # Strip symbols from release builds for smaller binary (better I-cache locality)
  if (NOT CMAKE_HOST_SYSTEM_NAME MATCHES "Darwin")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -s -Wl,--gc-sections")
    message(STATUS "Symbol stripping + section GC enabled for release build")
  endif()

  # Clang-specific: aggressive inline threshold (must be after project() for compiler ID)
  if (AGGRESSIVE_INLINE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(STATUS "Clang detected: applying -mllvm -inline-threshold=250")
    set(CMAKE_C_FLAGS_RELEASE   "${CMAKE_C_FLAGS_RELEASE}   -mllvm -inline-threshold=250")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -mllvm -inline-threshold=250")
    if (NOT DISABLE_LTO)
      set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-mllvm,-inline-threshold=250")
    endif()
  endif()

  # Keep the LTO backend aligned with the compile-time tuning target.
  # Without this, the final whole-program codegen can fall back to the generic
  # -march CPU and discard the local scheduling model from -mtune.
  if (NOT DISABLE_LTO AND CMAKE_CXX_COMPILER_ID MATCHES "Clang") 
    if (TUNE_NATIVE)
      set(DIRTYBIRD_LTO_MCPU "native")
    elseif (WIN32 AND TARGET_ARCH MATCHES "amd64" AND CPU_ARCHTARGET STREQUAL "x86-64-v3")
      set(DIRTYBIRD_LTO_MCPU "raptorlake")
    else()
      set(DIRTYBIRD_LTO_MCPU "")
    endif()

    if (NOT DIRTYBIRD_LTO_MCPU STREQUAL "")
      set(CMAKE_EXE_LINKER_FLAGS
          "${CMAKE_EXE_LINKER_FLAGS} -Wl,-plugin-opt=mcpu=${DIRTYBIRD_LTO_MCPU}")
      message(STATUS "Clang LTO backend CPU tuning: -plugin-opt=mcpu=${DIRTYBIRD_LTO_MCPU}")
    endif()
  endif()

  # CPU tuning for Raptor Lake (13th/14th gen Intel hybrid arch)
  if (NOT TUNE_NATIVE)
    set(CMAKE_C_FLAGS_RELEASE   "${CMAKE_C_FLAGS_RELEASE}   -mtune=raptorlake")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -mtune=raptorlake")
    message(STATUS "CPU tuning: -mtune=raptorlake (hybrid P+E core scheduling)")
  endif()
endif()

# OpenMP support for divsufsort parallelism (like TNN/Luna thread multiplication)
# Must be after project() so C/CXX languages are enabled
if (WITH_OPENMP AND NOT WITH_HIP)
  # CMake's OpenMP probe uses try_compile/try_link. On Windows/Clang our
  # full-LTO linker flags can make those tiny probe links fail even though the
  # real target links fine with libomp. Probe with a clean executable linker
  # flag set, then restore the production flags for the actual miner targets.
  set(_dirtybird_saved_exe_linker_flags "${CMAKE_EXE_LINKER_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "")
  find_package(OpenMP)
  set(CMAKE_EXE_LINKER_FLAGS "${_dirtybird_saved_exe_linker_flags}")
  unset(_dirtybird_saved_exe_linker_flags)
  if(OpenMP_FOUND)
    message(STATUS "OpenMP found - enabling parallel suffix array construction")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${OpenMP_CXX_FLAGS}")
    add_compile_definitions(DIRTYBIRD_OPENMP)
  else()
    message(STATUS "OpenMP not found - using single-threaded suffix array")
  endif()
endif()

if (WIN32 AND (MSVC OR (CMAKE_CXX_COMPILER_ID STREQUAL "Clang"
             AND WITH_HIP AND HIP_PLATFORM MATCHES "nvidia")))

  message(STATUS "MSVC WHY?... ${MSVC}, ${CMAKE_CXX_SIMULATE_ID}, ${HIP_PLATFORM}")

  add_compile_options(
    $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:/EHa>
    $<$<AND:$<NOT:$<CONFIG:Debug>>,$<COMPILE_LANGUAGE:CXX>>:/EHsc>
  )

  add_compile_options(
    $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:C>>:/MT>
    $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:/MT>
    $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:C>>:/MTd>
    $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:/MTd>
  )

  if (NOT WITH_HIP)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedStatic$<$<CONFIG:Debug>:Debug>")
  else()
    unset(CMAKE_MSVC_RUNTIME_LIBRARY CACHE)
    unset(CMAKE_MSVC_RUNTIME_LIBRARY)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "")
  endif()
endif()

set(PROJECT_VERSION ${INTERNAL_DERO_VERSION})

if(NOT USE_GCC)
  set(CMAKE_LIB_SUFFIX "")
  set(CMAKE_DO_OPAQUE "")

  if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    string(REGEX MATCH "^([0-9]+)" CXX_MAJOR_VERSION "${CMAKE_CXX_COMPILER_VERSION}")

    # Only try to (re)set if not already set by the Apple/Homebrew block
    if(NOT DEFINED CMAKE_AR OR CMAKE_AR STREQUAL "" OR CMAKE_AR MATCHES "XcodeDefault")
      # Prefer Homebrew location first, then fallback to PATH
      find_program(LLVM_AR_PROG
        NAMES llvm-ar-${CXX_MAJOR_VERSION} llvm-ar
        PATHS /opt/homebrew/opt/llvm@18/bin
      )
      if(LLVM_AR_PROG)
        set(CMAKE_AR "${LLVM_AR_PROG}" CACHE FILEPATH "Archiver" FORCE)
      endif()
    endif()

    if(NOT DEFINED CMAKE_RANLIB OR CMAKE_RANLIB STREQUAL "" OR CMAKE_RANLIB MATCHES "XcodeDefault")
      find_program(LLVM_RANLIB_PROG
        NAMES llvm-ranlib-${CXX_MAJOR_VERSION} llvm-ranlib
        PATHS /opt/homebrew/opt/llvm@18/bin
      )
      if(LLVM_RANLIB_PROG)
        set(CMAKE_RANLIB "${LLVM_RANLIB_PROG}" CACHE FILEPATH "Ranlib" FORCE)
      endif()
    endif()
  endif()

  set(CMAKE_C_FLAGS_RELEASE   "${CMAKE_C_FLAGS_RELEASE} -Wno-deprecated -Wno-nan-infinity-disabled ${CMAKE_DO_OPAQUE}")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wno-deprecated -Wno-nan-infinity-disabled ${CMAKE_DO_OPAQUE}")
else()
  set(CMAKE_C_FLAGS_RELEASE   "${CMAKE_C_FLAGS_RELEASE} -Wno-deprecated-declarations")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wno-deprecated-declarations")
endif()

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

include (CheckIncludeFile)
include (cmake/cpu.cmake)
include (cmake/os.cmake)
include (cmake/pgo.cmake)

# =============================================================================
# BOLT (Binary Optimization and Layout Tool) Support
# =============================================================================
# BOLT can provide additional 5-15% performance by optimizing code layout
# after linking based on runtime profiles. Requires LLVM BOLT tool.
#
# Usage:
#   1. Build normally with -fno-reorder-blocks-and-partition
#   2. Run with perf: perf record -e cycles:u -j any,u ./dirtybird-miner-cpu ...
#   3. Convert perf data: perf2bolt -p perf.data -o perf.fdata ./dirtybird-miner-cpu
#   4. Optimize: llvm-bolt ./dirtybird-miner-cpu -o ./dirtybird-miner-cpu-bolt \
#                -data=perf.fdata -reorder-blocks=ext-tsp -reorder-functions=hfsort
#
option(ENABLE_BOLT_COMPAT "Enable BOLT-compatible build (preserves relocs)" OFF)

if (ENABLE_BOLT_COMPAT AND CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    message(STATUS "BOLT compatibility enabled: preserving relocations for post-link optimization")
    # Preserve relocations for BOLT
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--emit-relocs")
    # Disable block reordering that interferes with BOLT
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -fno-reorder-blocks-and-partition")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fno-reorder-blocks-and-partition")
endif()

cmake_print_variables($ENV{GITLAB_CI})
cmake_print_variables(INTERNAL_DIRTYBIRD_VERSION)
cmake_print_variables(CMAKE_SYSTEM_PROCESSOR)

# This CMAKE variable isn't defined until 'project' is called.
set(DO_STATIC " ")

# Fully static Linux links break thread detection and Boost's CMake fallback in
# clean CI environments. Keep BUILD_STATIC for MinGW-style Windows packaging,
# but use normal system linkage elsewhere unless we are cross-compiling.
if (BUILD_STATIC)
  if (CMAKE_HOST_SYSTEM_NAME MATCHES "Darwin")
    message(STATUS "BUILD_STATIC requested on macOS; using dynamic system linkage.")
  elseif (WIN32 OR WIN_CROSS)
    if (NOT (MSVC
             OR (CMAKE_CXX_COMPILER_ID STREQUAL "Clang"
                 AND CMAKE_CXX_SIMULATE_ID STREQUAL "MSVC")))
      set(DO_STATIC " -static ")
    endif()
  else()
    message(STATUS "BUILD_STATIC requested on ${CMAKE_HOST_SYSTEM_NAME}; using dynamic system linkage for compatibility.")
  endif()
endif()

message(STATUS "DO_STATIC='${DO_STATIC}'")

if(DEFINED ENV{CPU_ARCHTARGET})
  set(CPU_ARCHTARGET "$ENV{CPU_ARCHTARGET}")
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
  set(WITH_HIP OFF)
  set(TARGET_ARCH "aarch64")
  if (NOT CPU_ARCHTARGET)
    set(CPU_ARCHTARGET "armv8-a+aes")
  endif()
  # AES needed for Xelisv2
  set(CMAKE_C_FLAGS_RELEASE   "${CMAKE_C_FLAGS_RELEASE} -flax-vector-conversions -march=${CPU_ARCHTARGET}")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -flax-vector-conversions -march=${CPU_ARCHTARGET}")
else()
  set(TARGET_ARCH "amd64")
  if (WITH_HIP)
    set(CPU_ARCHTARGET "x86-64")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -mtune=generic")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -mtune=generic")
  endif()
  if (NOT CPU_ARCHTARGET)
    set(CPU_ARCHTARGET "x86-64-v3")
  endif()
  set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -march=${CPU_ARCHTARGET} -m64")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=${CPU_ARCHTARGET} -m64")
endif()
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -D\"CPU_ARCHTARGET=${CPU_ARCHTARGET}\"")
#if(WIN_CROSS)
#  set(WIN_CROSS_OPTS "${WIN_CROSS_OPTS} --target=x86_64-pc-windows-msvc")
#  set(WIN_CROSS_OPTS "${WIN_CROSS_OPTS} \
#  -I/src/clang64/include \
#  -I/src/winsdk/crt/include \
#  -I/src/winsdk/sdk/Include/ucrt \
#  -I/src/winsdk/sdk/Include/shared \
#  -I/src/winsdk/sdk/Include/um")
#endif()
#set(CMAKE_LIB_SUFFIX "${CMAKE_LIB_SUFFIX}_${CPU_ARCHTARGET}")

if(TARGET_ARCH MATCHES "amd64")
  message(STATUS "Target architecture is amd64: enabling hardware crypto extensions")

  # SHA-NI for fast SHA256 (Intel Goldmont+/Ice Lake, AMD Zen+)
  add_compile_options("$<$<COMPILE_LANGUAGE:C>:-msha>")
  add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-msha>")

  # AES-NI for hardware AES acceleration
  add_compile_options("$<$<COMPILE_LANGUAGE:C>:-maes>")
  add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-maes>")

  # PCLMUL for carry-less multiplication (used in some crypto)
  add_compile_options("$<$<COMPILE_LANGUAGE:C>:-mpclmul>")
  add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-mpclmul>")

  # BMI2 for fast bit manipulation (pdep/pext)
  add_compile_options("$<$<COMPILE_LANGUAGE:C>:-mbmi2>")
  add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-mbmi2>")

  # AVX2 when enabled (x86-64-v3 includes this, but explicit for clarity)
  if (WITH_AVX2)
    add_compile_options("$<$<COMPILE_LANGUAGE:C>:-mavx2>")
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-mavx2>")
    message(STATUS "AVX2 explicitly enabled for SIMD crypto operations")
  endif()
endif()

set(CMAKE_C_FLAGS_RELEASE   "${CMAKE_C_FLAGS_RELEASE} ${DO_STATIC} -DDERO_VERSION=${INTERNAL_DERO_VERSION} -DDIRTYBIRD_VERSION_STR=\\\"${DIRTYBIRD_VERSION}\\\"") # ${WIN_CROSS_OPTS}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${DO_STATIC} -DDERO_VERSION=${INTERNAL_DERO_VERSION} -DDIRTYBIRD_VERSION_STR=\\\"${DIRTYBIRD_VERSION}\\\"") # ${WIN_CROSS_OPTS}")

# =============================================================================
# OPTIMIZATION SUMMARY
# =============================================================================
message(STATUS "========================================")
message(STATUS "OPTIMIZATION FLAGS SUMMARY")
message(STATUS "========================================")
message(STATUS "Target arch: ${CPU_ARCHTARGET}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
if(DISABLE_LTO)
    message(STATUS "LTO: DISABLED")
else()
    message(STATUS "LTO: ENABLED")
endif()
if (USE_THIN_LTO)
    message(STATUS "LTO Type: Thin LTO")
else()
    message(STATUS "LTO Type: Full LTO")
endif()
message(STATUS "PGO Generate: ${PGO_GENERATE}")
message(STATUS "PGO Use: ${PGO_USE}")
message(STATUS "BOLT Compatible: ${ENABLE_BOLT_COMPAT}")
message(STATUS "Aggressive Inline: ${AGGRESSIVE_INLINE}")
message(STATUS "Native Tune: ${TUNE_NATIVE}")
message(STATUS "AVX2: ${WITH_AVX2}")
message(STATUS "AVX512: ${WITH_AVX512}")
message(STATUS "========================================")
cmake_print_variables(CMAKE_C_FLAGS_RELEASE)
cmake_print_variables(CMAKE_CXX_FLAGS_RELEASE)

# if(WIN32)
# set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
# endif()

# if(UNIX)
# set(CMAKE_SYSTEM_NAME Linux)
# set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
# endif()

set (CMAKE_CXX_STANDARD 20)
set (CMAKE_STANDARD_REQUIRED TRUE)

set(OPENSSL_USE_STATIC_LIBS TRUE)

if(WIN32)
  message(STATUS "WIN32!")
  if(NOT WITH_HIP)
    # Prefer the active MSYS2 toolchain prefix discovered from the compiler.
    if (DIRTYBIRD_MSYS2_INCLUDE_DIR AND EXISTS "${DIRTYBIRD_MSYS2_INCLUDE_DIR}/openssl/ssl.h")
      set(OPENSSL_ROOT_DIR "${DIRTYBIRD_MSYS2_PREFIX}")
      set(SODIUM_ROOT_DIR "${DIRTYBIRD_MSYS2_PREFIX}")
      include_directories(
        "${DIRTYBIRD_MSYS2_INCLUDE_DIR}"
      )
      link_directories(
        "${DIRTYBIRD_MSYS2_LIB_DIR}"
      )
    # Try legacy local MSYS2 paths next, then fall back to clang64 paths
    elseif (EXISTS "C:/msys64/mingw64/include/openssl/ssl.h")
      set(OPENSSL_ROOT_DIR "C:/msys64/mingw64")
      set(SODIUM_ROOT_DIR "C:/msys64/mingw64")
      include_directories(
        "C:/msys64/mingw64/include"
      )
      link_directories(
        "C:/msys64/mingw64/lib"
      )
    else()
      set(OPENSSL_ROOT_DIR "C:/openssl/clang64")
      set(SODIUM_ROOT_DIR "C:/sodium/clang64")
      include_directories(
        "C:/mingw64/include"
        "C:/mingw64/x86_64-w64-mingw32/include"
        "C:/mingw64/include/c++"
      )
      link_directories(
        "C:/mingw64/lib"
        "C:/mingw64/x86_64-w64-mingw32/lib"
      )
    endif()
  else()
    if (HIP_PLATFORM MATCHES "nvidia")
      add_compile_definitions(BOOST_HAS_THREADS=1)
    endif()

    set(OPENSSL_ROOT_DIR "C:/openssl_msvc")
    add_library(OpenSSL::Crypto UNKNOWN IMPORTED)
    set_target_properties(OpenSSL::Crypto PROPERTIES IMPORTED_LOCATION "${OPENSSL_ROOT_DIR}/lib/libcrypto_static.lib")
  
    add_library(OpenSSL::SSL UNKNOWN IMPORTED)
    set_target_properties(OpenSSL::SSL PROPERTIES IMPORTED_LOCATION "${OPENSSL_ROOT_DIR}/lib/libssl_static.lib")
  endif()
  include_directories(
    ${OPENSSL_ROOT_DIR}/include
    ${SODIUM_ROOT_DIR}/include
  )
  link_directories(
    ${OPENSSL_ROOT_DIR}/lib
    ${SODIUM_ROOT_DIR}/lib
  )
endif()

# This stores downloaded dependencies so we do not need to re-download them
string(MAKE_C_IDENTIFIER "${CMAKE_CXX_COMPILER_ID}-${CMAKE_CXX_COMPILER_VERSION}" COMPILER_TAG)

if(UNIX)
  if(DEFINED ENV{XDG_CACHE_HOME})
    set(_cache_root "$ENV{XDG_CACHE_HOME}")
  else()
    set(_cache_root "$ENV{HOME}/.cache")
  endif()

  if (WITH_HIP)
    set(CPM_SOURCE_CACHE
        "${_cache_root}/Dirtybird-miner/CPM-hip-${CMAKE_SYSTEM_NAME}-${COMPILER_TAG}")
  else()
    set(CPM_SOURCE_CACHE
        "${_cache_root}/Dirtybird-miner/CPM-cpu-${CMAKE_SYSTEM_NAME}-${COMPILER_TAG}")
  endif()

else()
  if (WITH_HIP)
    set(CPM_SOURCE_CACHE
        "${PROJECT_SOURCE_DIR}/_deps/CPM-hip-${CMAKE_SYSTEM_NAME}-${COMPILER_TAG}")
  else()
    set(CPM_SOURCE_CACHE
        "${PROJECT_SOURCE_DIR}/_deps/CPM-cpu-${CMAKE_SYSTEM_NAME}-${COMPILER_TAG}")
  endif()
endif()
cmake_print_variables(CPM_SOURCE_CACHE)

set(CPM_DOWNLOAD_VERSION 0.40.8)

set(CPM_DOWNLOAD_LOCATION "${PROJECT_SOURCE_DIR}/_deps/CPM_${CPM_DOWNLOAD_VERSION}.cmake")

if(NOT (EXISTS ${CPM_DOWNLOAD_LOCATION}))
  message(STATUS "Downloading CPM.cmake to ${CPM_DOWNLOAD_LOCATION}")
  file(DOWNLOAD
       https://github.com/cpm-cmake/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake
       ${CPM_DOWNLOAD_LOCATION}
  )
endif()

cmake_print_variables(CPM_DOWNLOAD_LOCATION)
include(${CPM_DOWNLOAD_LOCATION})

#set(OPENSSL_USE_STATIC_LIBS TRUE)

if (WIN_CROSS)
  set(OPENSSL_ROOT "/src/clang64")
endif()

# For MSYS2 MinGW builds, help CMake find OpenSSL
if (WIN32 AND NOT MSVC AND NOT OPENSSL_ROOT_DIR)
  # Check the resolved MSYS2 toolchain prefix first.
  if (DIRTYBIRD_MSYS2_INCLUDE_DIR AND EXISTS "${DIRTYBIRD_MSYS2_INCLUDE_DIR}/openssl/ssl.h")
    set(OPENSSL_ROOT_DIR "${DIRTYBIRD_MSYS2_PREFIX}" CACHE PATH "OpenSSL root")
    set(OPENSSL_INCLUDE_DIR "${DIRTYBIRD_MSYS2_INCLUDE_DIR}" CACHE PATH "OpenSSL include")
    set(OPENSSL_CRYPTO_LIBRARY "${DIRTYBIRD_MSYS2_LIB_DIR}/libcrypto.dll.a" CACHE FILEPATH "OpenSSL crypto lib")
    set(OPENSSL_SSL_LIBRARY "${DIRTYBIRD_MSYS2_LIB_DIR}/libssl.dll.a" CACHE FILEPATH "OpenSSL ssl lib")
  elseif (EXISTS "C:/msys64/mingw64/include/openssl/ssl.h")
    set(OPENSSL_ROOT_DIR "C:/msys64/mingw64" CACHE PATH "OpenSSL root")
    set(OPENSSL_INCLUDE_DIR "C:/msys64/mingw64/include" CACHE PATH "OpenSSL include")
    set(OPENSSL_CRYPTO_LIBRARY "C:/msys64/mingw64/lib/libcrypto.dll.a" CACHE FILEPATH "OpenSSL crypto lib")
    set(OPENSSL_SSL_LIBRARY "C:/msys64/mingw64/lib/libssl.dll.a" CACHE FILEPATH "OpenSSL ssl lib")
  endif()
endif()

if (NOT WIN32 OR NOT WITH_HIP)
  find_package(OpenSSL REQUIRED)
  if(OPENSSL_INCLUDE_DIR)
    include_directories(${OPENSSL_INCLUDE_DIR})
  endif()
endif()

if (DIRTYBIRD_OS_ANDROID)
  set(THREAD_LIB "")
elseif (UNIX)
  set(THREAD_LIB pthread)
elseif(NOT WITH_HIP)
  set(THREAD_LIB winpthread)
endif()

if (WIN_CROSS)
  set(THREAD_LIB winpthread)
endif()
cmake_print_variables(THREAD_LIB)

if (WIN_CROSS)
  set(Boost_ROOT "${PROJECT_SOURCE_DIR}/export/boost-x64")
  set(COMMON_FLAGS "-DBOOST_HAS_THREADS=1 -Wno-nonportable-include-path -Wno-pragma-pack -Wno-ignored-attributes -Wno-ignored-pragma-intrinsic -Wno-comment -Wno-microsoft-anon-tag")
  set(CMAKE_C_FLAGS_RELEASE   "${CMAKE_C_FLAGS_RELEASE} ${COMMON_FLAGS}")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${COMMON_FLAGS}")
  #set(BOOST_HAS_THREADS "1")
  #set(BOOST_THREADAPI "win32")
endif()

if (WIN32 OR WIN_CROSS)
  set(Boost_USE_STATIC_LIBS ON)
  set(Boost_USE_STATIC_RUNTIME ON)
else()
  set(Boost_USE_STATIC_LIBS OFF)
  set(Boost_USE_STATIC_RUNTIME OFF)
endif()
set(Boost_USE_MULTITHREADED ON)
set(Boost_NO_WARN_NEW_VERSIONS ON)

# For MSYS2 MinGW builds, help CMake find Boost
if (WIN32 AND NOT MSVC AND NOT Boost_ROOT)
  if (DIRTYBIRD_MSYS2_INCLUDE_DIR AND EXISTS "${DIRTYBIRD_MSYS2_INCLUDE_DIR}/boost/version.hpp")
    set(BOOST_ROOT "${DIRTYBIRD_MSYS2_PREFIX}" CACHE PATH "Boost root")
    file(GLOB DIRTYBIRD_BOOST_CMAKE_DIRS "${DIRTYBIRD_MSYS2_LIB_DIR}/cmake/Boost-*")
    if (DIRTYBIRD_BOOST_CMAKE_DIRS)
      list(SORT DIRTYBIRD_BOOST_CMAKE_DIRS)
      list(GET DIRTYBIRD_BOOST_CMAKE_DIRS -1 DIRTYBIRD_BOOST_CMAKE_DIR)
      set(Boost_DIR "${DIRTYBIRD_BOOST_CMAKE_DIR}" CACHE PATH "Boost cmake")
    endif()
  elseif (EXISTS "C:/msys64/mingw64/include/boost/version.hpp")
    set(BOOST_ROOT "C:/msys64/mingw64" CACHE PATH "Boost root")
    set(Boost_DIR "C:/msys64/mingw64/lib/cmake/Boost-1.90.0" CACHE PATH "Boost cmake")
  endif()
endif()

if(APPLE)
  set(USE_CPM_BOOST ON CACHE BOOL "" FORCE)
endif()

if (NOT USE_CPM_BOOST)
find_package(Boost 1.80 COMPONENTS
  chrono
  context
  thread
  json
  atomic
  program_options
)
endif()

set(THREADS_PREFER_PTHREAD_FLAG ON)
# find_package(Threads REQUIRED)

if (WIN_CROSS AND NOT Boost_FOUND)
  message(FATAL_ERROR "Boost was not found: ${Boost_ROOT}")
endif()

if(Boost_FOUND AND NOT USE_CPM_BOOST)
  message("OS-provided Boost found")
  set(DIRTYBIRD_BOOST_INCLUDES ${Boost_INCLUDE_DIRS})
  include_directories(${DIRTYBIRD_BOOST_INCLUDES})
  set(DIRTYBIRD_BOOST_LIBDIRS ${Boost_LIBRARY_DIRS})

  message(STATUS "Boost library directories: ${DIRTYBIRD_BOOST_LIBDIRS}")

  link_directories(${DIRTYBIRD_BOOST_LIBDIRS})
  message(${Boost_VERSION})
else()
  message("Boost not found. Using CPM to add it")

  set(BOOST_TARBALL "${PROJECT_SOURCE_DIR}/_deps/boost-1.86.0-cmake.tar.xz")
  if(NOT EXISTS "${BOOST_TARBALL}")
    message(STATUS "Downloading Boost archive to ${BOOST_TARBALL}")
    file(DOWNLOAD
        https://github.com/boostorg/boost/releases/download/boost-1.86.0/boost-1.86.0-cmake.tar.xz
        "${BOOST_TARBALL}"
        EXPECTED_HASH SHA256=2c5ec5edcdff47ff55e27ed9560b0a0b94b07bd07ed9928b476150e16b0efc57)
    message(STATUS "Boost 1.86.0 has been downloaded successfully in the CPM cache directory!")
  else()
    message(STATUS "Using cached Boost archive at ${BOOST_TARBALL}")
  endif()

  # Only use MSVC compatibility for HIP builds on Windows
  # CPU builds use GNU/MinGW which is incompatible with -fms-compatibility
  if(WITH_HIP AND WIN32)
    set(BOOST_MS_COMPAT "CMAKE_CXX_FLAGS -fms-compatibility")
  else()
    set(BOOST_MS_COMPAT "")
  endif()

  CPMAddPackage(
    NAME Boost
    VERSION 1.86.0 # Versions less than 1.85.0 may need patches for installation targets.
    URL "${PROJECT_SOURCE_DIR}/_deps/boost-1.86.0-cmake.tar.xz"
    URL_HASH SHA256=2c5ec5edcdff47ff55e27ed9560b0a0b94b07bd07ed9928b476150e16b0efc57
    OPTIONS
      "BOOST_ENABLE_CMAKE ON"
      "BOOST_SKIP_INSTALL_RULES ON" # Set `OFF` for installation
      "BOOST_USE_STATIC_LIBS ON"
      "BOOST_USE_STATIC_RUNTIME ON"
      "BOOST_USE_MULTITHREADED ON"
      "BOOST_INSTALL_LAYOUT system"
      "BOOST_INCLUDE_LIBRARIES chrono\\\;context\\\;coroutine\\\;multiprecision\\\;json\\\;thread\\\;atomic\\\;program_options"
      "${BOOST_TOOLSET_OPTS}"  # Specify Clang toolset
      "${BOOST_MS_COMPAT}"
  )

  if(Boost_ADDED)
    file(GLOB files "${Boost_SOURCE_DIR}/libs/*/include")
    foreach(filename ${files})
      include_directories(${filename})
    endforeach()

    file(GLOB files "${Boost_SOURCE_DIR}/libs/*/*/include")
    foreach(filename ${files})
      include_directories(${filename})
    endforeach()

    file(GLOB_RECURSE BOOST_CMAKELISTS
      "${Boost_SOURCE_DIR}/CMakeLists.txt"
      "${Boost_SOURCE_DIR}/libs/*/CMakeLists.txt"
    )
  else()
    message(FATAL_ERROR "Boost was neither found nor added.")
  endif()
endif()

# This is universal regardless of how wheter we compiled or found Boost
#set(DIRTYBIRD_BOOST_LIBS Boost::chrono Boost::context Boost::coroutine Boost::thread Boost::json Boost::atomic Boost::program_options Boost::asio)
set(DIRTYBIRD_BOOST_LIBS Boost::atomic Boost::context Boost::chrono Boost::json Boost::program_options Boost::thread) #@Boost::beast Boost::bind Boost::json Boost::chrono Boost::thread Boost::atomic Boost::program_options )

set(WINSDK_DIR "${PROJECT_SOURCE_DIR}/wincross/winsdk")
if(WIN_CROSS)
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -target x86_64-pc-windows-msvc")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -target x86_64-pc-windows-msvc")
  set(CMAKE_SKIP_RPATH TRUE)
  #set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=ld.lld-${CXX_MAJOR_VERSION} -target x86_64-pc-windows-msvc") # -Wl,-machine:x64")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=lld-link-${CXX_MAJOR_VERSION} -v -target x86_64-pc-windows-msvc") # -Wl,-machine:x64")
  #link_directories("${WINSDK_DIR}/crt/lib/x64" "${WINSDK_DIR}/sdk/lib/ucrt/x64" "${WINSDK_DIR}/sdk/lib/um/x64")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L${WINSDK_DIR}/../clang64/lib")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L${WINSDK_DIR}/crt/lib/x64")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L${WINSDK_DIR}/sdk/lib/ucrt/x64")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L${WINSDK_DIR}/sdk/lib/um/x64")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L${DIRTYBIRD_BOOST_LIBDIRS}")
  #set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Xlinker /force")
  set(LINK_OPTIONS "${CMAKE_EXE_LINKER_FLAGS}")
elseif (ANDROID OR CMAKE_SYSTEM_NAME MATCHES "Android")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -v")
elseif (CMAKE_HOST_SYSTEM_NAME MATCHES "Darwin" OR USE_GCC)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lc++ -fuse-ld=ld")
elseif (CMAKE_HOST_SYSTEM_NAME MATCHES "Linux")
  if (NOT WITH_HIP OR HIP_PLATFORM MATCHES "amd")
    find_program(LLD_PROG NAMES ld.lld-${CXX_MAJOR_VERSION} ld.lld lld)
    if (LLD_PROG)
      get_filename_component(LLD_DIR "${LLD_PROG}" DIRECTORY)
      message(STATUS "Using lld at ${LLD_PROG}")
      set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -v -B${LLD_DIR} -fuse-ld=lld")
    else()
      message(STATUS "lld not found, falling back to default linker")
      set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -v")
    endif()
  else()
    # HIP + NVIDIA
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -v")
  endif()
elseif (CMAKE_HOST_SYSTEM_NAME MATCHES "Windows")
  if (NOT WITH_HIP AND NOT DIRTYBIRD_WINDOWS_CLANG_LINK_FLAGS STREQUAL "")
    set(CMAKE_EXE_LINKER_FLAGS
        "${CMAKE_EXE_LINKER_FLAGS} ${DIRTYBIRD_WINDOWS_CLANG_LINK_FLAGS}")
  endif()
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -v")
endif()
cmake_print_variables(CMAKE_EXE_LINKER_FLAGS)

# if(WITH_HIP)
# CPMAddPackage(
#   NAME sodium
#   GITHUB_REPOSITORY robinlinden/libsodium-cmake
#   GIT_TAG f568ff02f1bed155ea598c0e803ef3c9db2703d2
#   OPTIONS 
#     "CMAKE_C_FLAGS_RELEASE -msse3 -mssse3 -maes -mpclmul -mavx2" 
#     "CMAKE_CXX_FLAGS_RELEASE -msse3 -mssse3 -maes -mpclmul -mavx2"
# )
# endif()

# if (TARGET sodium)
#   # Add SSSE3 and PCLMUL flags for sodium target
#   target_compile_options(sodium PRIVATE -Ofast -mssse3 -mpclmul -maes -mavx2)
# endif()

include_directories(
  "${PROJECT_SOURCE_DIR}/include"
  "${PROJECT_SOURCE_DIR}/include/highwayhash"
  "${PROJECT_SOURCE_DIR}/src"
  "${PROJECT_SOURCE_DIR}/src/astrobwtv3/"
  # DERO Miner: Removed xelis-hash and spectrex
  "${PROJECT_SOURCE_DIR}/src/stratum/"
  "${PROJECT_SOURCE_DIR}/src/net/"
  "${PROJECT_SOURCE_DIR}/src/crypto/"
  "${PROJECT_SOURCE_DIR}/src/broadcast/"
  "${PROJECT_SOURCE_DIR}/lib"
  "${PROJECT_SOURCE_DIR}/extern"
)
if(WIN_CROSS)
  include_directories(
    "${WINSDK_DIR}/../clang64/include"
    "${WINSDK_DIR}/crt/include"
    "${WINSDK_DIR}/sdk/Include/ucrt"
    "${WINSDK_DIR}/sdk/Include/shared"
    "${WINSDK_DIR}/sdk/Include/um"
  )
endif()

file(GLOB_RECURSE SOURCES_CORE
  "${PROJECT_SOURCE_DIR}/include/libkeccak/*.c"
  "${PROJECT_SOURCE_DIR}/src/core/*.cpp"
  "${PROJECT_SOURCE_DIR}/src/core/*.c"
  "${PROJECT_SOURCE_DIR}/src/crypto/cshake/*.c"
  "${PROJECT_SOURCE_DIR}/src/crypto/skein/*.c"
  "${PROJECT_SOURCE_DIR}/src/crypto/sha3/*.c"
  "${PROJECT_SOURCE_DIR}/src/broadcast/*.cpp"
  "${PROJECT_SOURCE_DIR}/src/coins/dummy_miners.cpp"
  "${PROJECT_SOURCE_DIR}/src/util/*.c"
  "${PROJECT_SOURCE_DIR}/src/util/*.cpp"
)

# DERO Miner: Disabled salsa-simd and argon2 - they use GCC function multiversioning
# which requires ifunc, not supported on Windows. AstroBWTv3 uses header-only Salsa20.h instead.
# if (NOT WITH_HIP)
#   file(GLOB SOURCES_CORE_CRYPTO_SALSA
#        "${PROJECT_SOURCE_DIR}/src/crypto/salsa-simd/*.c")
#
#   file(GLOB SOURCES_CORE_ARGON2
#        "${PROJECT_SOURCE_DIR}/src/crypto/argon2/*.c"
#        "${PROJECT_SOURCE_DIR}/src/crypto/argon2/*.cpp")
#
#   list(APPEND SOURCES_CORE
#        ${SOURCES_CORE_CRYPTO_SALSA}
#        ${SOURCES_CORE_ARGON2})
# endif()

file(GLOB SOURCES_NET
  "${PROJECT_SOURCE_DIR}/src/net/*.cpp"
  "${PROJECT_SOURCE_DIR}/src/net/*.c"
  "${PROJECT_SOURCE_DIR}/src/net/dero/*.cpp"
  # DERO Miner: Removed spectre sources
)

# Radix SA: DeroHE-style 2-pass radix sort replaces SPSA
if(USE_DLUNA_RADIX_SA)
  set(CMAKE_C_FLAGS_RELEASE   "${CMAKE_C_FLAGS_RELEASE} -DUSE_DLUNA_RADIX_SA")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DUSE_DLUNA_RADIX_SA")
  message(STATUS "DLUNA RADIX SA: Using 24-bit prefix radix + collision sort")
endif()

# Radix SA: DeroHE-style 2-pass radix sort replaces SPSA
if(USE_RADIX_SA)
  set(CMAKE_C_FLAGS_RELEASE   "${CMAKE_C_FLAGS_RELEASE} -DUSE_RADIX_SA")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DUSE_RADIX_SA")
  message(STATUS "RADIX SA: Using DeroHE-style 2-pass radix sort")
endif()

# Bucket SA: 2-byte prefix counting sort optimized for random bytes
if(USE_BUCKET_SA)
  set(CMAKE_C_FLAGS_RELEASE   "${CMAKE_C_FLAGS_RELEASE} -DUSE_BUCKET_SA")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DUSE_BUCKET_SA")
  message(STATUS "BUCKET SA: Using 2-byte prefix bucket sort")
endif()

include(cmake/spsa.cmake)

if (NOT WITH_HIP)
  # DERO-only: Include AstroBWTv3 and supporting libraries
  include(cmake/astrobwtv3.cmake)
  include(cmake/blake2.cmake)
  include(cmake/chacha20.cmake)
else()
  # GPU build: Include HIP crypto support
  include(cmake/hip-crypto/hip-crypto.cmake)
endif()

# enable_language(HIP)
# set(CMAKE_HIP_COMPILER_WORKS 1)


if (WITH_BLAKE3)
  add_subdirectory(${PROJECT_SOURCE_DIR}/extern/BLAKE3/c)
  set(BLAKE3_LIBRARIES blake3)
endif()

# Remove aarch files from SOURCES_CORE
#if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64" AND NOT CMAKE_HOST_SYSTEM_NAME MATCHES "Darwin")
#  list(FILTER SOURCES_CORE EXCLUDE REGEX ".*/.*/*aarch64*")
#endif()

# WinRing0 for CPU feature detection (Windows only)
if (WIN32)
  include(cmake/winring0.cmake)
endif()

include(cmake/setup.cmake)

if (WITH_HIP)
  # STAGE 1: Create static library with all regular C++ code and dependencies using normal linking
  set(DERO_CORE_LIB "dirtybird-core-lib")
  add_library(${DERO_CORE_LIB} STATIC
    ${SOURCES_CORE}
    ${SOURCES_NET}
    ${SOURCES_CRYPTO}
    ${HEADERS_CRYPTO}
  )

  # Apply normal C++ linking to the static library (no HIP linking issues here)
  setup_target_libraries(${DERO_CORE_LIB})

  #
  # Windows + NVIDIA HIP: manual hipcc compilation of HIP sources
  #
  find_package(hip REQUIRED)

  # if (WIN32 AND HIP_PLATFORM MATCHES "nvidia")
  if (FALSE)
    message(STATUS "Configuring HIP (NVIDIA backend) on Windows using manual hipcc compilation")

    # Path to hipcc – can be overridden on the CMake command line
    if(WIN32)
      set(HIPCC_EXT ".exe")
    endif()

    if (NOT DEFINED HIPCC_EXECUTABLE)
      set(HIPCC_EXECUTABLE "$ENV{HIP_PATH}/bin/hipcc${HIPCC_EXT}")
    endif()

    if (NOT EXISTS "${HIPCC_EXECUTABLE}")
      message(FATAL_ERROR "HIPCC_EXECUTABLE not found: ${HIPCC_EXECUTABLE}")
    endif()

    # Output name for CUDA backend via HIP
    set(DERO_OUTPUT_NAME "dirtybird-miner-cuda")

    # Adjust architectures as desired

    set(DERO_HIP_OBJS "")

    set(HIPCC_INCLUDES_RSP
        "${CMAKE_CURRENT_BINARY_DIR}/hipcc_includes_$<CONFIG>.rsp")

    get_target_property(DIRTYBIRD_CORE_INCS ${DERO_CORE_LIB} INCLUDE_DIRECTORIES)

    set(HIPCC_INCLUDES_RSP "${CMAKE_CURRENT_BINARY_DIR}/hipcc_includes_$<CONFIG>.rsp")

    set(HIPCC_INCLUDES_CONTENT "")
    foreach(inc IN LISTS DIRTYBIRD_CORE_INCS)
      string(APPEND HIPCC_INCLUDES_CONTENT "-I\"${inc}\"\n")
    endforeach()

    file(GENERATE
      OUTPUT  "${HIPCC_INCLUDES_RSP}"
      CONTENT "${HIPCC_INCLUDES_CONTENT}"
    )

    foreach(src ${DERO_HIP_SOURCES})
      get_filename_component(src_name "${src}" NAME_WE)
      set(obj "${CMAKE_CURRENT_BINARY_DIR}/${src_name}.obj")

      set(HIPCC_ARCH_FLAGS
        "-gencode=arch=compute_90,code=sm_90"
      )

      add_custom_command(
        OUTPUT ${obj}
        COMMAND ${HIPCC_EXECUTABLE}
                -std=c++20
                -O3
                -x cu
                ${HIPCC_ARCH_FLAGS}
                -I"${HIP_PATH}/include"
                --options-file="${HIPCC_INCLUDES_RSP}"
                -c "${src}" -o "${obj}"
        DEPENDS "${src}"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        VERBATIM
      )

      list(APPEND DERO_HIP_OBJS "${obj}")
    endforeach()

    add_executable(${DERO_OUTPUT_NAME}
      ${DERO_HIP_OBJS}
    )

    if (NOT DEFINED ENV{CUDA_PATH})
      message(FATAL_ERROR
        "===============================================\n"
        " CUDA backend requires CUDA_PATH to be defined.\n"
        " On Windows this is normally set by the installer:\n"
        "   setx CUDA_PATH \"C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.X\"\n"
        "==============================================="
      )
    endif()

    # Link with core C++ library and HIP runtime
    target_link_libraries(${DERO_OUTPUT_NAME}
      PRIVATE
        ${DERO_CORE_LIB}
        "$ENV{CUDA_PATH}/lib/x64/cudart_static.lib"
        "$ENV{CUDA_PATH}/lib/x64/cuda.lib"
        cuda hiprtc nvrtc
    )

  #
  # All other HIP platforms: use proper HIP language integration
  #
  else()
    if (HIP_PLATFORM MATCHES "amd")
      set(CMAKE_HIP_ARCHITECTURES
        "gfx1100"
      )
      set(HIPRTC_LIB hiprtc)
      set(DERO_OUTPUT_NAME "dirtybird-miner-rocm")
      set(HIPRT_LIB hip::host)
    else()
      if(WIN32)
        if (NOT DEFINED ENV{CUDA_PATH})
          message(FATAL_ERROR
            "HIP_PLATFORM=nvidia requires CUDA_PATH environment variable "
            "pointing to the CUDA Toolkit root (e.g. C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1)."
          )
        endif()

        set(CMAKE_HIP_ARCHITECTURES "90")
        set(DERO_OUTPUT_NAME "dirtybird-miner-cuda")

        set(CUDA_LIB_DIR "$ENV{CUDA_PATH}/lib/x64")

        set(HIPRT_LIB
          "${CUDA_LIB_DIR}/cudart_static.lib"
          "${CUDA_LIB_DIR}/cudadevrt.lib"
          "${CUDA_LIB_DIR}/cuda.lib"
        )

        set(HIPRTC_LIB 
          "${CUDA_LIB_DIR}/nvrtc.lib" 
        )
      else()
        set(CMAKE_HIP_ARCHITECTURES "90")
        set(DERO_OUTPUT_NAME "dirtybird-miner-cuda")

        set(HIPRT_LIB  cudart_static cudadevrt cuda)
        set(HIPRTC_LIB nvrtc)
      endif()
    endif()

    set(HIP_LANG HIP)
    set(CMAKE_HIP_COMPILER_WORKS 1)

    enable_language(${HIP_LANG})

    add_executable(${DERO_OUTPUT_NAME}
      ${DERO_HIP_SOURCES}
    )

    if (WIN32 AND HIP_PLATFORM MATCHES "nvidia")
      set(CMAKE_HIP_LINK_EXECUTABLE
        "<CMAKE_LINKER> /machine:x64 /INCREMENTAL:NO <LINK_FLAGS> <OBJECTS> /OUT:<TARGET> <LINK_LIBRARIES>"
      )
    endif()

    # Collect HIP (and optionally CUDA) include dirs in one variable
    set(_tnn_hip_includes "")

    # HIP includes (needed on both AMD and NVIDIA HIP backends)
    if(DEFINED ENV{HIP_PATH})
      list(APPEND _tnn_hip_includes "$ENV{HIP_PATH}/include")
    else()
      message(WARNING "HIP_PATH is not set; HIP headers may not be found.")
    endif()

    # On NVIDIA backend, also add CUDA includes
    if(HIP_PLATFORM MATCHES "nvidia")
      if(DEFINED ENV{CUDA_PATH})
        list(APPEND _tnn_hip_includes "$ENV{CUDA_PATH}/include")
      else()
        message(FATAL_ERROR
          "HIP_PLATFORM=nvidia but CUDA_PATH is not set. "
          "CUDA headers are required for the NVIDIA HIP backend."
        )
      endif()
      set(HIPRTC_DLL_DIR "${CMAKE_SOURCE_DIR}/lib/nvrtc")
    else()
      set(HIPRTC_DLL_DIR "${CMAKE_SOURCE_DIR}/lib/hiprtc")
    endif()

    # Single include directive, works for both AMD and NVIDIA
    target_include_directories(${DERO_CORE_LIB}
      PRIVATE
        ${_tnn_hip_includes}
    )

    target_include_directories(${DERO_OUTPUT_NAME}
      PRIVATE
        ${_tnn_hip_includes}
    )

    if (WIN32)
      target_link_libraries(${DERO_OUTPUT_NAME}
        PRIVATE
          ${DERO_CORE_LIB}
          ${HIPRT_LIB}
          ${HIPRTC_LIB}
          user32.lib
      )
    else()
      target_link_libraries(${DERO_OUTPUT_NAME}
        PRIVATE
          ${DERO_CORE_LIB}
          ${HIPRT_LIB}
          ${HIPRTC_LIB}
      )
    endif()

    # Global HIP flags
    if (HIP_PLATFORM MATCHES "amd")
      # ROCm/clang backend
      set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -O3 -ffast-math")
    elseif (HIP_PLATFORM MATCHES "nvidia")
      # CUDA/nvcc backend
      set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -O3 --use_fast_math")
    else()
      set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -O3")
    endif()
  endif()

  if (WIN32)
    add_custom_command(
      TARGET ${DERO_OUTPUT_NAME} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${HIPRTC_DLL_DIR}"
        "$<TARGET_FILE_DIR:${DERO_OUTPUT_NAME}>"
    )
  endif()

else()
  # CPU-only build path
  set(DERO_OUTPUT_NAME "dirtybird-miner-cpu")
  add_executable(${DERO_OUTPUT_NAME}
    ${SOURCES_CORE}
    ${SOURCES_NET}
    ${SOURCES_CRYPTO}
    ${HEADERS_CRYPTO}
  )

  setup_target_libraries(${DERO_OUTPUT_NAME})
  if (USE_ASTRO_SPSA AND CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_definitions(${DERO_OUTPUT_NAME} PRIVATE DIRTYBIRD_SHA256_LINK_WRAP)
    target_link_options(${DERO_OUTPUT_NAME} PRIVATE
      "-Wl,--wrap=SHA256_Init"
      "-Wl,--wrap=SHA256_Update"
      "-Wl,--wrap=SHA256_Final"
    )
  endif()

  # mimalloc: faster allocator for all heap operations (new/delete/malloc/free)
  if (USE_MIMALLOC)
    # Prefer static library for fully static builds
    set(MIMALLOC_STATIC_LIB "")
    if (WIN32 AND DIRTYBIRD_MSYS2_LIB_DIR AND EXISTS "${DIRTYBIRD_MSYS2_LIB_DIR}/libmimalloc.a")
      set(MIMALLOC_STATIC_LIB "${DIRTYBIRD_MSYS2_LIB_DIR}/libmimalloc.a")
    elseif (WIN32 AND EXISTS "C:/msys64/mingw64/lib/libmimalloc.a")
      set(MIMALLOC_STATIC_LIB "C:/msys64/mingw64/lib/libmimalloc.a")
    else()
      find_library(MIMALLOC_STATIC_LIB mimalloc-static mimalloc PATHS "/usr/lib" "/usr/local/lib")
    endif()
    if (MIMALLOC_STATIC_LIB)
      message(STATUS "mimalloc (static) found: ${MIMALLOC_STATIC_LIB}")
      target_compile_definitions(${DERO_OUTPUT_NAME} PRIVATE USE_MIMALLOC)
      # Override operator new/delete globally via mimalloc_override.cpp
      # (SPSA library uses _Znwy/_ZdlPv which resolve to our replacements at link time)
      # Note: -include mimalloc-override.h not used - conflicts with MinGW headers
      target_link_libraries(${DERO_OUTPUT_NAME} ${MIMALLOC_STATIC_LIB})
    else()
      message(WARNING "mimalloc not found - using default allocator")
    endif()
  endif()

  # NOTE: Cold code -Os per-file overrides disabled - conflicts with Full LTO backend
  # optimization (all modules merge into single LTO unit, -Os can deoptimize hot code)
  # The PGO profile already handles code layout optimization.

  # Set output name
  set_target_properties(${DERO_OUTPUT_NAME} PROPERTIES OUTPUT_NAME ${DERO_OUTPUT_NAME})

  # UPX compression for smaller binary (like DeroLuna's 3.3MB)
  option(USE_UPX "Compress binary with UPX for smaller size" OFF)
  if (USE_UPX)
    if (DIRTYBIRD_MSYS2_BIN_DIR)
      find_program(UPX_EXECUTABLE upx HINTS "${DIRTYBIRD_MSYS2_BIN_DIR}")
    else()
      find_program(UPX_EXECUTABLE upx HINTS "C:/msys64/mingw64/bin")
    endif()
    if (UPX_EXECUTABLE)
      message(STATUS "UPX found: ${UPX_EXECUTABLE} - will compress binary after build")
      add_custom_command(TARGET ${DERO_OUTPUT_NAME} POST_BUILD
        COMMAND ${UPX_EXECUTABLE} --best --lzma "$<TARGET_FILE:${DERO_OUTPUT_NAME}>"
        COMMENT "Compressing ${DERO_OUTPUT_NAME} with UPX..."
      )
    else()
      message(WARNING "UPX not found - binary will not be compressed. Install with: pacman -S mingw-w64-x86_64-upx")
    endif()
  endif()
endif()

unset(HIP_LANG CACHE)

# =============================================================================
# SIMD Salsa20 Benchmark
# =============================================================================
option(BUILD_SALSA20_BENCH "Build Salsa20 SIMD benchmark" ON)

if (BUILD_SALSA20_BENCH AND NOT WITH_HIP)
    add_executable(bench-salsa20-simd
        src/bin/bench_salsa20_simd.cpp
        src/crypto/salsa20_simd.c
    )

    target_include_directories(bench-salsa20-simd PRIVATE
        ${CMAKE_SOURCE_DIR}/include
    )

    # Enable AVX2 for the benchmark
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(bench-salsa20-simd PRIVATE -mavx2)
    elseif (MSVC)
        target_compile_options(bench-salsa20-simd PRIVATE /arch:AVX2)
    endif()

    set_target_properties(bench-salsa20-simd PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )

    # Interleaved (Two Miners Per Thread) Benchmark
    # Note: Include astro sources directly since DERO_CORE_LIB is only for HIP builds
    # Exclude spsa_integrated.cpp since we stub SPSA for this standalone benchmark
    file(GLOB_RECURSE BENCH_ASTRO_SOURCES
        src/crypto/astrobwtv3/*.cpp
        src/crypto/astrobwtv3/*.c
    )
    # Add critical dependencies missing from GLOB
    list(APPEND BENCH_ASTRO_SOURCES
        src/crypto/sha256_override.cpp
        src/crypto/salsa20_simd.c
    )
    if (USE_LIBSAIS)
        list(APPEND BENCH_ASTRO_SOURCES "${DIRTYBIRD_LIBSAIS_SOURCE}")
    endif()
    # Remove SPSA-dependent files to avoid needing the library
    list(FILTER BENCH_ASTRO_SOURCES EXCLUDE REGEX ".*spsa_integrated\\.cpp$")

    add_executable(bench-interleaved
        src/bin/bench_interleaved.cpp
        ${BENCH_ASTRO_SOURCES}
    )
    target_include_directories(bench-interleaved PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/crypto
        ${DIRTYBIRD_BOOST_INCLUDES}
    )
    # Link Boost libraries (same as main target)
    target_link_libraries(bench-interleaved
        ${DIRTYBIRD_BOOST_LIBS}
        OpenSSL::Crypto
    )
    # Link OpenMP if found
    if(OpenMP_FOUND)
        target_link_libraries(bench-interleaved OpenMP::OpenMP_CXX)
    endif()
    # Windows-specific libraries
    if(WIN32)
        target_link_libraries(bench-interleaved mingw32 wsock32 ws2_32 kernel32)
    endif()
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(bench-interleaved PRIVATE -mavx2 -mstackrealign)
    endif()
    set_target_properties(bench-interleaved PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )

    # Vertical SIMD Benchmark
    add_executable(bench-vsimd
        src/bin/bench_vsimd.cpp
        ${BENCH_ASTRO_SOURCES}
    )
    target_include_directories(bench-vsimd PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/crypto
        ${DIRTYBIRD_BOOST_INCLUDES}
    )
    target_link_libraries(bench-vsimd
        ${DIRTYBIRD_BOOST_LIBS}
        OpenSSL::Crypto
    )
    if(OpenMP_FOUND)
        target_link_libraries(bench-vsimd OpenMP::OpenMP_CXX)
    endif()
    if(WIN32)
        target_link_libraries(bench-vsimd mingw32 wsock32 ws2_32 kernel32)
    endif()
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(bench-vsimd PRIVATE -mavx2 -mstackrealign)
    endif()
    set_target_properties(bench-vsimd PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )

    # Profile original phases (for comparison with lean)
    add_executable(profile-original-phases
        src/bin/profile_original_phases.cpp
        ${BENCH_ASTRO_SOURCES}
    )
    target_include_directories(profile-original-phases PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/crypto
        ${DIRTYBIRD_BOOST_INCLUDES}
    )
    target_link_libraries(profile-original-phases
        ${DIRTYBIRD_BOOST_LIBS}
        OpenSSL::Crypto
    )
    if(OpenMP_FOUND)
        target_link_libraries(profile-original-phases OpenMP::OpenMP_CXX)
    endif()
    if(WIN32)
        target_link_libraries(profile-original-phases mingw32 wsock32 ws2_32 kernel32)
    endif()
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(profile-original-phases PRIVATE -mavx2)
    endif()
    set_target_properties(profile-original-phases PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )

    file(GLOB_RECURSE REPLAY_ASTRO_SOURCES
        src/crypto/astrobwtv3/*.cpp
        src/crypto/astrobwtv3/*.c
    )
    list(FILTER REPLAY_ASTRO_SOURCES EXCLUDE REGEX ".*optimization_tests\\.cpp$")
    list(FILTER REPLAY_ASTRO_SOURCES EXCLUDE REGEX ".*sa_benchmark\\.cpp$")
    list(FILTER REPLAY_ASTRO_SOURCES EXCLUDE REGEX ".*sha256_scatter\\.cpp$")
    list(FILTER REPLAY_ASTRO_SOURCES EXCLUDE REGEX ".*astrotest\\.cpp$")
    list(FILTER REPLAY_ASTRO_SOURCES EXCLUDE REGEX ".*sa_instrumentation\\.c$")
    list(APPEND REPLAY_ASTRO_SOURCES
        src/crypto/sha256_override.cpp
        src/crypto/salsa20_simd.c
        src/core/build_info.cpp
        src/util/numa_optimizer.cpp
    )
    if (USE_LIBSAIS)
        list(APPEND REPLAY_ASTRO_SOURCES "${DIRTYBIRD_LIBSAIS_SOURCE}")
    endif()

    add_executable(bench-dero-replay
        src/bin/bench_dero_replay.cpp
        ${REPLAY_ASTRO_SOURCES}
    )
    target_include_directories(bench-dero-replay PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/crypto
        ${DIRTYBIRD_BOOST_INCLUDES}
    )
    setup_target_libraries(bench-dero-replay)
    if (USE_ASTRO_SPSA AND CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        target_compile_definitions(bench-dero-replay PRIVATE DIRTYBIRD_SHA256_LINK_WRAP)
        target_link_options(bench-dero-replay PRIVATE
            "-Wl,--wrap=SHA256_Init"
            "-Wl,--wrap=SHA256_Update"
            "-Wl,--wrap=SHA256_Final"
        )
    endif()
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(bench-dero-replay PRIVATE -mavx2)
    endif()
    set_target_properties(bench-dero-replay PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )

    # Test custom_sa_70kb without OpenMP (compares divsufsort, custom_sa_70kb, libsais)
    add_executable(test-custom-sa
        src/bin/test_custom_sa.cpp
        src/crypto/astrobwtv3/custom_sa_70kb.cpp
        src/crypto/astrobwtv3/divsufsort.c
        src/crypto/astrobwtv3/sssort.c
        src/crypto/astrobwtv3/trsort.c
        src/crypto/astrobwtv3/utils.c
        # libsais 2.10.4 for comparison
        ${DIRTYBIRD_LIBSAIS_SOURCE}
    )
    target_include_directories(test-custom-sa PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/crypto
        ${CMAKE_SOURCE_DIR}/src/crypto/astrobwtv3
        # libsais headers
        ${DIRTYBIRD_LIBSAIS_INCLUDE_DIR}
    )
    # Test with OpenMP undefined (like lean benchmark)
    target_compile_options(test-custom-sa PRIVATE -U_OPENMP -mavx2)
    target_compile_definitions(test-custom-sa PRIVATE USE_CUSTOM_SA USE_LIBSAIS)
    set_target_properties(test-custom-sa PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )

    # Debug test for custom SA
    add_executable(test-sa-debug
        src/bin/test_sa_debug.cpp
        src/crypto/astrobwtv3/custom_sa_70kb.cpp
        src/crypto/astrobwtv3/divsufsort.c
        src/crypto/astrobwtv3/sssort.c
        src/crypto/astrobwtv3/trsort.c
        src/crypto/astrobwtv3/utils.c
    )
    target_include_directories(test-sa-debug PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/crypto
        ${CMAKE_SOURCE_DIR}/src/crypto/astrobwtv3
    )
    target_compile_options(test-sa-debug PRIVATE -U_OPENMP -mavx2 -g -O0)
    target_compile_definitions(test-sa-debug PRIVATE SA_DEBUG)
    set_target_properties(test-sa-debug PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )

    # Small test for custom SA (without debug output)
    add_executable(test-sa-small
        src/bin/test_sa_small.cpp
        src/crypto/astrobwtv3/custom_sa_70kb.cpp
        src/crypto/astrobwtv3/divsufsort.c
        src/crypto/astrobwtv3/sssort.c
        src/crypto/astrobwtv3/trsort.c
        src/crypto/astrobwtv3/utils.c
    )
    target_include_directories(test-sa-small PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/crypto
        ${CMAKE_SOURCE_DIR}/src/crypto/astrobwtv3
    )
    target_compile_options(test-sa-small PRIVATE -U_OPENMP -mavx2 -g -O0)
    target_compile_definitions(test-sa-small PRIVATE SA_DEBUG)
    set_target_properties(test-sa-small PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )

    # Order debug test for custom SA
    add_executable(test-sa-order-debug
        src/bin/test_sa_order_debug.cpp
        src/crypto/astrobwtv3/custom_sa_70kb.cpp
        src/crypto/astrobwtv3/divsufsort.c
        src/crypto/astrobwtv3/sssort.c
        src/crypto/astrobwtv3/trsort.c
        src/crypto/astrobwtv3/utils.c
    )
    target_include_directories(test-sa-order-debug PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/crypto
        ${CMAKE_SOURCE_DIR}/src/crypto/astrobwtv3
    )
    target_compile_options(test-sa-order-debug PRIVATE -U_OPENMP -mavx2 -g -O0)
    set_target_properties(test-sa-order-debug PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )

    # SA trace test
    add_executable(test-sa-trace
        src/bin/test_sa_trace.cpp
        src/crypto/astrobwtv3/custom_sa_70kb.cpp
        src/crypto/astrobwtv3/divsufsort.c
        src/crypto/astrobwtv3/sssort.c
        src/crypto/astrobwtv3/trsort.c
        src/crypto/astrobwtv3/utils.c
    )
    target_include_directories(test-sa-trace PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/crypto
        ${CMAKE_SOURCE_DIR}/src/crypto/astrobwtv3
    )
    target_compile_options(test-sa-trace PRIVATE -U_OPENMP -mavx2 -g -O0)
    set_target_properties(test-sa-trace PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )

    # LMS sort test
    add_executable(test-lms-sort
        src/bin/test_lms_sort.cpp
        src/crypto/astrobwtv3/divsufsort.c
        src/crypto/astrobwtv3/sssort.c
        src/crypto/astrobwtv3/trsort.c
        src/crypto/astrobwtv3/utils.c
    )
    target_include_directories(test-lms-sort PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/crypto
        ${CMAKE_SOURCE_DIR}/src/crypto/astrobwtv3
    )
    target_compile_options(test-lms-sort PRIVATE -U_OPENMP -mavx2 -g -O0)
    set_target_properties(test-lms-sort PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )

    # Chain trace test
    add_executable(test-chain-trace
        src/bin/test_chain_trace.cpp
        src/crypto/astrobwtv3/divsufsort.c
        src/crypto/astrobwtv3/sssort.c
        src/crypto/astrobwtv3/trsort.c
        src/crypto/astrobwtv3/utils.c
    )
    target_include_directories(test-chain-trace PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/crypto
        ${CMAKE_SOURCE_DIR}/src/crypto/astrobwtv3
    )
    target_compile_options(test-chain-trace PRIVATE -U_OPENMP -mavx2 -g -O0)
    set_target_properties(test-chain-trace PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )

    # Random pattern debug test
    add_executable(test-random-debug
        src/bin/test_random_debug.cpp
        src/crypto/astrobwtv3/custom_sa_70kb.cpp
        src/crypto/astrobwtv3/divsufsort.c
        src/crypto/astrobwtv3/sssort.c
        src/crypto/astrobwtv3/trsort.c
        src/crypto/astrobwtv3/utils.c
    )
    target_include_directories(test-random-debug PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/crypto
        ${CMAKE_SOURCE_DIR}/src/crypto/astrobwtv3
    )
    target_compile_options(test-random-debug PRIVATE -U_OPENMP -mavx2 -g -O0)
    set_target_properties(test-random-debug PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )

    # Debug test for Salsa20
    add_executable(test-salsa20
        src/bin/test_salsa20.cpp
        src/crypto/salsa20_simd.c
    )
    target_include_directories(test-salsa20 PRIVATE ${CMAKE_SOURCE_DIR}/include)
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(test-salsa20 PRIVATE -mavx2)
    endif()
    set_target_properties(test-salsa20 PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )

    # RC4 correctness test (FastRc4 vs OpenSSL)
    add_executable(test-rc4-correctness
        src/bin/test_rc4_correctness.cpp
    )
    target_include_directories(test-rc4-correctness PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
    )
    target_link_libraries(test-rc4-correctness PRIVATE OpenSSL::Crypto)
    set_target_properties(test-rc4-correctness PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )

    # RC4 prep phase test
    add_executable(test-prep-phase
        src/bin/test_prep_phase.cpp
    )
    target_include_directories(test-prep-phase PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
    )
    target_link_libraries(test-prep-phase PRIVATE OpenSSL::Crypto)
    set_target_properties(test-prep-phase PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )

    # RC4 worker test (uses actual workerData struct)
    add_executable(test-worker-rc4
        src/bin/test_worker_rc4.cpp
    )
    target_include_directories(test-worker-rc4 PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/crypto/astrobwtv3
    )
    target_link_libraries(test-worker-rc4 PRIVATE OpenSSL::Crypto)
    set_target_properties(test-worker-rc4 PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )

    # RC4 miner simulation test
    add_executable(test-rc4-miner-sim
        src/bin/test_rc4_miner_sim.cpp
    )
    target_include_directories(test-rc4-miner-sim PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
    )
    target_link_libraries(test-rc4-miner-sim PRIVATE OpenSSL::Crypto)
    set_target_properties(test-rc4-miner-sim PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )

    # RC4 state persistence test
    add_executable(test-rc4-state
        src/bin/test_rc4_state.cpp
    )
    target_include_directories(test-rc4-state PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
    )
    target_link_libraries(test-rc4-state PRIVATE OpenSSL::Crypto)
    set_target_properties(test-rc4-state PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )

    # Worker initialization test
    add_executable(test-worker-init
        src/bin/test_worker_init.cpp
    )
    target_include_directories(test-worker-init PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
    )
    target_link_libraries(test-worker-init PRIVATE OpenSSL::Crypto)
    set_target_properties(test-worker-init PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )

    # Full hash flow test
    add_executable(test-hash-flow
        src/bin/test_hash_flow.cpp
    )
    target_include_directories(test-hash-flow PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
    )
    target_link_libraries(test-hash-flow PRIVATE OpenSSL::Crypto)
    set_target_properties(test-hash-flow PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )

    # Lean memory sizes test (minimal deps)
    add_executable(test-lean-sizes
        src/bin/test_lean_sizes.cpp
    )
    target_include_directories(test-lean-sizes PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/crypto
    )
    target_link_libraries(test-lean-sizes PRIVATE OpenSSL::Crypto)
    set_target_properties(test-lean-sizes PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )

    # Lean interleaved miner benchmark (full integration test)
    file(GLOB_RECURSE LEAN_ASTRO_SOURCES
        src/crypto/astrobwtv3/*.cpp
        src/crypto/astrobwtv3/*.c
    )
    list(APPEND LEAN_ASTRO_SOURCES
        src/crypto/salsa20_simd.c
    )
    # Remove SPSA-dependent files to avoid needing the library
    list(FILTER LEAN_ASTRO_SOURCES EXCLUDE REGEX ".*spsa_integrated\\.cpp$")

    add_executable(bench-lean-interleaved
        src/bin/bench_lean_interleaved.cpp
        ${LEAN_ASTRO_SOURCES}
    )
    target_include_directories(bench-lean-interleaved PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/crypto
        ${DIRTYBIRD_BOOST_INCLUDES}
    )
    target_link_libraries(bench-lean-interleaved
        ${DIRTYBIRD_BOOST_LIBS}
        OpenSSL::Crypto
    )
    # Disable OpenMP for lean benchmark - we do our own threading and
    # divsufsort's #pragma omp critical causes global lock contention.
    # Undefine _OPENMP to make divsufsort use the serial (thread-safe) path.
    target_compile_options(bench-lean-interleaved PRIVATE -U_OPENMP)
    # Still link OpenMP for any other potential uses
    if(OpenMP_FOUND)
        target_link_libraries(bench-lean-interleaved OpenMP::OpenMP_CXX)
    endif()
    if(WIN32)
        target_link_libraries(bench-lean-interleaved mingw32 wsock32 ws2_32 kernel32)
    endif()
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(bench-lean-interleaved PRIVATE -mavx2)
    endif()
    set_target_properties(bench-lean-interleaved PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )

    # Wolf comparison benchmark
    add_executable(bench-wolf-comparison
        src/bin/bench_wolf_comparison.cpp
        ${LEAN_ASTRO_SOURCES}
    )
    target_include_directories(bench-wolf-comparison PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/crypto
        ${DIRTYBIRD_BOOST_INCLUDES}
    )
    target_link_libraries(bench-wolf-comparison
        ${DIRTYBIRD_BOOST_LIBS}
        OpenSSL::Crypto
    )
    target_compile_options(bench-wolf-comparison PRIVATE -U_OPENMP)
    if(OpenMP_FOUND)
        target_link_libraries(bench-wolf-comparison OpenMP::OpenMP_CXX)
    endif()
    if(WIN32)
        target_link_libraries(bench-wolf-comparison mingw32 wsock32 ws2_32 kernel32)
    endif()
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(bench-wolf-comparison PRIVATE -mavx2)
    endif()
    set_target_properties(bench-wolf-comparison PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )

    # SA Comparison benchmark (libsais vs divsufsort)
    add_executable(bench-sa-comparison
        src/bin/bench_sa_comparison.cpp
        src/crypto/astrobwtv3/divsufsort.c
        src/crypto/astrobwtv3/sssort.c
        src/crypto/astrobwtv3/trsort.c
        src/crypto/astrobwtv3/utils.c
        ${DIRTYBIRD_LIBSAIS_SOURCE}
    )
    target_include_directories(bench-sa-comparison PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/crypto
        ${CMAKE_SOURCE_DIR}/src/crypto/astrobwtv3
        ${DIRTYBIRD_LIBSAIS_INCLUDE_DIR}
    )
    target_compile_options(bench-sa-comparison PRIVATE -U_OPENMP -mavx2 -O3)
    set_target_properties(bench-sa-comparison PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endif()
