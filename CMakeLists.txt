# DIRTYBIRD Miner -- CMake build
#
# Clean build:
#   mkdir build && cd build
#   cmake -G "MinGW Makefiles" -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ ..
#   cmake --build . -j$(nproc)
#
# PGO build (two-pass, local hashrate candidate):
#   mkdir build-pgo-gen && cd build-pgo-gen
#   cmake -DPGO_GENERATE=ON ..
#   cmake --build . --target dirtybird-pgo-train -j2
#   LLVM_PROFILE_FILE="pgo-%p.profraw" ./dirtybird-pgo-train -t 20 --seconds 60
#   llvm-profdata merge -o merged.profdata *.profraw
#   cd .. && mkdir build-pgo-use && cd build-pgo-use
#   cmake -DPGO_USE=ON -DPGO_PROFILE=../build-pgo-gen/merged.profdata ..
#   cmake --build . -j$(nproc)

cmake_minimum_required(VERSION 3.16)
project(dirtybird-miner-cpu LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 99)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ---- Version (drives the --version/banner string; release CI passes -DDIRTYBIRD_VERSION=<tag>) ----
set(DIRTYBIRD_VERSION_DEFAULT "1.0.17")
if(DEFINED DIRTYBIRD_VERSION)
    set(INTERNAL_DIRTYBIRD_VERSION "${DIRTYBIRD_VERSION}")
elseif(DEFINED ENV{DIRTYBIRD_VERSION})
    set(INTERNAL_DIRTYBIRD_VERSION "$ENV{DIRTYBIRD_VERSION}")
else()
    set(INTERNAL_DIRTYBIRD_VERSION "${DIRTYBIRD_VERSION_DEFAULT}")
endif()
string(REGEX REPLACE "^v" "" INTERNAL_DIRTYBIRD_VERSION "${INTERNAL_DIRTYBIRD_VERSION}")
message(STATUS "DIRTYBIRD version: ${INTERNAL_DIRTYBIRD_VERSION}")

# Release packaging expects executables under build/bin/.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# ---- Compiler detection ----
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(IS_CLANG TRUE)
else()
    set(IS_CLANG FALSE)
endif()

# ---- Base optimization flags ----
# -O2: proven winner over -O3 (code bloat hurts icache, confirmed dead end)
# -march=x86-64-v3: AVX2 baseline (Haswell+)
# -mtune=raptorlake: schedule for Raptor Lake (13th gen Intel)
# -flto: link-time optimization (+7.4% sustained hashrate, verified)
# -fno-vectorize -fno-slp-vectorize: auto-vectorization hurts thermal (dead end)
set(BASE_CXX_FLAGS "-O2 -DNDEBUG -march=x86-64-v3 -mtune=raptorlake -fomit-frame-pointer -mavx2 -fno-vectorize -fno-slp-vectorize -finline-functions")
set(BASE_C_FLAGS   "-O2 -DNDEBUG -march=x86-64-v3 -mtune=raptorlake -fomit-frame-pointer -mavx2")

# ---- LTO ----
option(DISABLE_LTO "Disable link-time optimization" OFF)
if(NOT DISABLE_LTO)
    set(BASE_CXX_FLAGS "${BASE_CXX_FLAGS} -flto")
    set(BASE_C_FLAGS   "${BASE_C_FLAGS} -flto")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -flto")
    if(IS_CLANG)
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=lld")
    endif()
endif()

# ---- PGO ----
option(PGO_GENERATE "Build with PGO instrumentation" OFF)
option(PGO_USE "Build with PGO optimization" OFF)
set(PGO_PROFILE "" CACHE FILEPATH "Path to merged.profdata for PGO use")
set(PGO_PROFILE_RESOLVED "")

if(PGO_GENERATE AND PGO_USE)
    message(FATAL_ERROR "PGO_GENERATE and PGO_USE are mutually exclusive")
endif()
if(PGO_USE)
    if(NOT PGO_PROFILE)
        message(FATAL_ERROR "PGO_USE requires -DPGO_PROFILE=/path/to/merged.profdata")
    endif()
    if(IS_ABSOLUTE "${PGO_PROFILE}")
        set(PGO_PROFILE_RESOLVED "${PGO_PROFILE}")
    elseif(EXISTS "${CMAKE_BINARY_DIR}/${PGO_PROFILE}")
        get_filename_component(PGO_PROFILE_RESOLVED
            "${CMAKE_BINARY_DIR}/${PGO_PROFILE}" ABSOLUTE)
    elseif(EXISTS "${CMAKE_SOURCE_DIR}/${PGO_PROFILE}")
        get_filename_component(PGO_PROFILE_RESOLVED
            "${CMAKE_SOURCE_DIR}/${PGO_PROFILE}" ABSOLUTE)
    else()
        set(PGO_PROFILE_RESOLVED "${PGO_PROFILE}")
    endif()
    if(NOT EXISTS "${PGO_PROFILE_RESOLVED}")
        message(FATAL_ERROR "PGO_PROFILE does not exist: ${PGO_PROFILE}")
    endif()
endif()

if(PGO_GENERATE)
    if(IS_CLANG)
        set(BASE_CXX_FLAGS "${BASE_CXX_FLAGS} -fprofile-generate -DDLUNA_PGO_GENERATE")
        set(BASE_C_FLAGS   "${BASE_C_FLAGS} -fprofile-generate")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fprofile-generate")
    else()
        set(BASE_CXX_FLAGS "${BASE_CXX_FLAGS} -fprofile-generate -DDLUNA_PGO_GENERATE")
        set(BASE_C_FLAGS   "${BASE_C_FLAGS} -fprofile-generate")
    endif()
elseif(PGO_USE)
    if(IS_CLANG)
        set(BASE_CXX_FLAGS "${BASE_CXX_FLAGS} -fprofile-use=${PGO_PROFILE_RESOLVED}")
        set(BASE_C_FLAGS   "${BASE_C_FLAGS} -fprofile-use=${PGO_PROFILE_RESOLVED}")
    else()
        set(BASE_CXX_FLAGS "${BASE_CXX_FLAGS} -fprofile-use=${PGO_PROFILE_RESOLVED} -fprofile-correction")
        set(BASE_C_FLAGS   "${BASE_C_FLAGS} -fprofile-use=${PGO_PROFILE_RESOLVED} -fprofile-correction")
    endif()
endif()

set(CMAKE_CXX_FLAGS_RELEASE "${BASE_CXX_FLAGS}")
set(CMAKE_C_FLAGS_RELEASE   "${BASE_C_FLAGS}")

# ---- SHA-NI wrap flags ----
# Redirect OpenSSL SHA256 calls to our SHA-NI hardware implementation.
# The nanosleep wrap is a legacy link-compat no-op; the miner uses
# Sleep()/dluna_sleep_ms (see include/dluna.h) for all timing.
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--wrap=SHA256_Init -Wl,--wrap=SHA256_Update -Wl,--wrap=SHA256_Final -Wl,--wrap=nanosleep")

add_definitions(-DUSE_ASTRO_SPSA=1)

# ---- OpenSSL ----
find_package(OpenSSL REQUIRED)
enable_testing()

function(dluna_copy_runtime_dlls target_name)
    if(WIN32)
        get_filename_component(DLUNA_RUNTIME_BIN_DIR "${CMAKE_CXX_COMPILER}" DIRECTORY)
        foreach(DLUNA_RUNTIME_DLL
            libcrypto-3-x64.dll
            libgcc_s_seh-1.dll
            libssl-3-x64.dll
            libstdc++-6.dll
            libwinpthread-1.dll
        )
            if(EXISTS "${DLUNA_RUNTIME_BIN_DIR}/${DLUNA_RUNTIME_DLL}")
                add_custom_command(TARGET ${target_name} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${DLUNA_RUNTIME_BIN_DIR}/${DLUNA_RUNTIME_DLL}"
                        "$<TARGET_FILE_DIR:${target_name}>/${DLUNA_RUNTIME_DLL}"
                    COMMENT "Copying ${DLUNA_RUNTIME_DLL} for ${target_name}"
                )
            endif()
        endforeach()
    endif()
endfunction()

# ---- Sources ----
# 2026-04-30 Phase B technique #1 reverted: gating USE_ASTRO_SPSA off
# measured Δ -0.31 KH/s sustained (within thermal noise, failed +1 KH/s
# gate per perf-study-2026-04-30/before-tech1-gate-spsa.log vs after-).
# The #if gates added to astroworker.h / astrobwt.cpp / main.cpp / miner.cpp
# are kept (harmless when USE_ASTRO_SPSA defined; useful for future ports).
# ---- libdluna_stages: stage code shared between mining exe and replay harness ----
add_library(dluna_stages STATIC
    src/astrobwt.cpp
    src/branch.cpp
    src/sha256_ni.cpp
    src/salsa20.cpp
    src/libsais.c
    src/spsa.cpp
    src/spsa_local.cpp
    src/sa_2byte_radix.cpp
    src/spsa_tritonn.cpp
    src/spsa_tritonn_helpers.cpp
    src/dluna_stages.cpp
    src/v114_stubs.cpp
)

target_include_directories(dluna_stages PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(dluna_stages PUBLIC OpenSSL::Crypto OpenSSL::SSL)

add_library(dluna_runtime STATIC
    src/runtime_tune.cpp
    src/console.cpp
)
target_include_directories(dluna_runtime PUBLIC ${CMAKE_SOURCE_DIR}/include)

# ---- Mining executable: thin wrapper around the lib ----
add_executable(dirtybird-miner-cpu
    src/main.cpp
    src/miner.cpp
    src/network.cpp
    src/difficulty.cpp
)

target_link_libraries(dirtybird-miner-cpu PRIVATE dluna_stages dluna_runtime)
target_compile_definitions(dirtybird-miner-cpu PRIVATE DIRTYBIRD_VERSION_STR="${INTERNAL_DIRTYBIRD_VERSION}")
dluna_copy_runtime_dlls(dirtybird-miner-cpu)

add_executable(dirtybird-pgo-train
    tools/pgo_trainer.cpp
    src/miner.cpp
    src/difficulty.cpp
)
target_include_directories(dirtybird-pgo-train PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(dirtybird-pgo-train PRIVATE dluna_stages dluna_runtime)
dluna_copy_runtime_dlls(dirtybird-pgo-train)

# 2026-04-29: phase-A profiling showed libsais is 90% of hash time. Bump it
# to -O3 for aggressive vectorization on its hot inner loops. Clone's general
# -O2 stays in place (icache-friendly for the rest of the binary).
set_source_files_properties(src/libsais.c PROPERTIES COMPILE_FLAGS "-O3")

# ---- SPSA: bundled in-tree clean-room implementation ----
# The SPSA stage is built from the in-tree sources above (no external
# library dependency), so source builds are fully self-contained.

# ---- Platform libs ----
if(WIN32)
    target_link_libraries(dirtybird-miner-cpu PRIVATE ws2_32)
endif()

add_subdirectory(tools/replay)

# ---- Strip release binary ----
if(CMAKE_BUILD_TYPE STREQUAL "Release" AND NOT PGO_GENERATE)
    add_custom_command(TARGET dirtybird-miner-cpu POST_BUILD
        COMMAND ${CMAKE_STRIP} $<TARGET_FILE:dirtybird-miner-cpu> || true
        COMMENT "Stripping debug symbols"
    )
endif()
