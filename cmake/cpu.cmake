if (CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(DIRTYBIRD_64_BIT ON)
    add_definitions(-DDIRTYBIRD_64_BIT)
else()
    set(DIRTYBIRD_64_BIT OFF)
endif()

if (NOT CMAKE_SYSTEM_PROCESSOR)
    message(WARNING "CMAKE_SYSTEM_PROCESSOR not defined")
endif()

include(CheckCXXCompilerFlag)

if (CMAKE_CXX_COMPILER_ID MATCHES MSVC 
    OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang"
    OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(VAES_SUPPORTED ON)
else()
    CHECK_CXX_COMPILER_FLAG("-mavx2 -mvaes" VAES_SUPPORTED)
endif()

if (NOT VAES_SUPPORTED)
    set(WITH_VAES OFF)
endif()

if (DIRTYBIRD_64_BIT AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
    add_definitions(-DRAPIDJSON_SSE2)
else()
    set(WITH_SSE4_1 OFF)
    set(WITH_AVX2 OFF)
    set(WITH_VAES OFF)
endif()

if (ARM_V8)
    set(ARM_TARGET 8)
elseif (ARM_V7)
    set(ARM_TARGET 7)
endif()

if (NOT ARM_TARGET)
    if (CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|armv8-a)$")
        set(ARM_TARGET 8)
    elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "^(armv7|armv7f|armv7s|armv7k|armv7-a|armv7l|armv7ve)$")
        set(ARM_TARGET 7)
    endif()
endif()

if (ARM_TARGET AND ARM_TARGET GREATER 6)
    set(DIRTYBIRD_ARM ON)
    add_definitions(-DDIRTYBIRD_ARM=${ARM_TARGET})

    message(STATUS "Use ARM_TARGET=${ARM_TARGET} (${CMAKE_SYSTEM_PROCESSOR})")

    if (ARM_TARGET EQUAL 8)
        CHECK_CXX_COMPILER_FLAG(-march=armv8-a+crypto DIRTYBIRD_ARM_CRYPTO)

        if (DIRTYBIRD_ARM_CRYPTO)
            add_definitions(-DDIRTYBIRD_ARM_CRYPTO)
            set(ARM8_CXX_FLAGS "-march=armv8-a+crypto")
        else()
            set(ARM8_CXX_FLAGS "-march=armv8-a")
        endif()
    endif()
endif()

if (WITH_SSE4_1)
    add_definitions(-DDIRTYBIRD_FEATURE_SSE4_1)
endif()

if (WITH_AVX2)
    add_definitions(-DDIRTYBIRD_FEATURE_AVX2)
endif()

# AVX512 Support
# Requires: AVX512F, AVX512BW, AVX512VL, AVX512DQ, AVX512CD
# Zen5 (AMD) and Ice Lake+ (Intel) have native 512-bit execution
option(WITH_AVX512 "Enable AVX-512 support (F, BW, VL, DQ, CD)" OFF)

if (WITH_AVX512 AND DIRTYBIRD_64_BIT AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
    message(STATUS "AVX-512 support enabled")
    add_definitions(-DDIRTYBIRD_FEATURE_AVX512)

    if (MSVC)
        # MSVC: /arch:AVX512 enables F, VL, BW, DQ, CD extensions (VS 2019 16.4+)
        # Also defines: __AVX512F__, __AVX512CD__, __AVX512BW__, __AVX512DQ__, __AVX512VL__
        add_compile_options(/arch:AVX512)
        message(STATUS "MSVC AVX-512: /arch:AVX512")
    else()
        # GCC/Clang: explicit flags for each extension
        # Note: x86-64-v4 baseline includes AVX512 but we add explicit flags for clarity
        set(AVX512_FLAGS "-mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512cd")
        set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${AVX512_FLAGS}")
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${AVX512_FLAGS}")
        message(STATUS "GCC/Clang AVX-512: ${AVX512_FLAGS}")
    endif()
endif()
