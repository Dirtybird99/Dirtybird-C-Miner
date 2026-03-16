# Profile-Guided Optimization (PGO) Support
#
# PGO can provide 10-20% performance improvement by:
# - Optimizing branch prediction based on actual runtime behavior
# - Improving code layout for I-cache efficiency
# - Better inlining decisions based on call frequency
#
# Usage:
#   1. Build with PGO_GENERATE=ON:
#      cmake -DPGO_GENERATE=ON -B build-pgo-gen
#      cmake --build build-pgo-gen --config Release
#
#   2. Run training workload (mine for 60-120 seconds):
#      ./build-pgo-gen/dirtybird-miner-cpu --dero --threads 12 --mine-time 120
#
#   3. Build optimized binary with PGO_USE=ON:
#      cmake -DPGO_USE=ON -DPGO_PROFILE_DIR=/path/to/profiles -B build-pgo-use
#      cmake --build build-pgo-use --config Release
#
# Compiler support:
#   - Clang: -fprofile-generate / -fprofile-use
#   - GCC: -fprofile-generate / -fprofile-use
#   - MSVC: /GENPROFILE / /USEPROFILE (requires different workflow)

option(PGO_GENERATE "Build instrumented binary for PGO training" OFF)
option(PGO_USE "Build optimized binary using PGO profile data" OFF)

# Default profile directory (can be overridden)
if (NOT PGO_PROFILE_DIR)
    set(PGO_PROFILE_DIR "${CMAKE_BINARY_DIR}/pgo-profiles" CACHE PATH "Directory for PGO profile data")
endif()

# Ensure PGO_GENERATE and PGO_USE are mutually exclusive
if (PGO_GENERATE AND PGO_USE)
    message(FATAL_ERROR "PGO_GENERATE and PGO_USE cannot both be enabled. Choose one.")
endif()

# PGO for Clang
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    if (PGO_GENERATE)
        message(STATUS "PGO: Enabling instrumentation (Clang -fprofile-generate)")
        message(STATUS "PGO: Profile data will be written to: ${PGO_PROFILE_DIR}")

        # Create profile directory
        file(MAKE_DIRECTORY "${PGO_PROFILE_DIR}")

        # Add instrumentation flags
        set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -fprofile-generate=${PGO_PROFILE_DIR}")
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fprofile-generate=${PGO_PROFILE_DIR}")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fprofile-generate=${PGO_PROFILE_DIR}")

        # Add definitions for runtime awareness
        add_definitions(-DPGO_TRAINING_BUILD)

    elseif (PGO_USE)
        # Check if profile directory exists
        if (NOT EXISTS "${PGO_PROFILE_DIR}")
            message(FATAL_ERROR "PGO_USE enabled but profile directory not found: ${PGO_PROFILE_DIR}\n"
                              "Run the instrumented binary first to generate profiles.")
        endif()

        # Check for profile data files
        file(GLOB PGO_PROFILES "${PGO_PROFILE_DIR}/*.profraw")
        if (NOT PGO_PROFILES)
            message(WARNING "No .profraw files found in ${PGO_PROFILE_DIR}. "
                           "Ensure you ran the instrumented binary before building with PGO_USE.")
        else()
            message(STATUS "PGO: Found ${CMAKE_MATCH_COUNT} profile files")
        endif()

        # Merge profiles if llvm-profdata is available
        find_program(LLVM_PROFDATA llvm-profdata)
        if (LLVM_PROFDATA)
            set(PGO_MERGED_PROFILE "${PGO_PROFILE_DIR}/merged.profdata")

            # Only merge if merged profile doesn't exist or is older than raw profiles
            if (NOT EXISTS "${PGO_MERGED_PROFILE}")
                message(STATUS "PGO: Merging profile data with llvm-profdata...")
                execute_process(
                    COMMAND ${LLVM_PROFDATA} merge -output=${PGO_MERGED_PROFILE} ${PGO_PROFILES}
                    RESULT_VARIABLE MERGE_RESULT
                )
                if (NOT MERGE_RESULT EQUAL 0)
                    message(WARNING "PGO: Failed to merge profiles. Using raw profiles.")
                    set(PGO_PROFILE_PATH "${PGO_PROFILE_DIR}")
                else()
                    set(PGO_PROFILE_PATH "${PGO_MERGED_PROFILE}")
                endif()
            else()
                set(PGO_PROFILE_PATH "${PGO_MERGED_PROFILE}")
            endif()
        else()
            message(STATUS "PGO: llvm-profdata not found via find_program, checking for merged.profdata...")
            if (EXISTS "${PGO_PROFILE_DIR}/merged.profdata")
                message(STATUS "PGO: Found pre-merged profile: ${PGO_PROFILE_DIR}/merged.profdata")
                set(PGO_PROFILE_PATH "${PGO_PROFILE_DIR}/merged.profdata")
            else()
                message(STATUS "PGO: No merged.profdata found, using raw profile directory")
                set(PGO_PROFILE_PATH "${PGO_PROFILE_DIR}")
            endif()
        endif()

        message(STATUS "PGO: Using profile data from: ${PGO_PROFILE_PATH}")

        # Add optimization flags
        set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -fprofile-use=${PGO_PROFILE_PATH}")
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fprofile-use=${PGO_PROFILE_PATH}")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fprofile-use=${PGO_PROFILE_PATH}")

        # Add definitions for runtime awareness
        add_definitions(-DPGO_OPTIMIZED_BUILD)
    endif()

# PGO for GCC
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if (PGO_GENERATE)
        message(STATUS "PGO: Enabling instrumentation (GCC -fprofile-generate)")
        message(STATUS "PGO: Profile data will be written to: ${PGO_PROFILE_DIR}")

        # Create profile directory
        file(MAKE_DIRECTORY "${PGO_PROFILE_DIR}")

        # Add instrumentation flags
        set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -fprofile-generate=${PGO_PROFILE_DIR} -fprofile-update=atomic")
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fprofile-generate=${PGO_PROFILE_DIR} -fprofile-update=atomic")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fprofile-generate=${PGO_PROFILE_DIR}")

        add_definitions(-DPGO_TRAINING_BUILD)

    elseif (PGO_USE)
        if (NOT EXISTS "${PGO_PROFILE_DIR}")
            message(FATAL_ERROR "PGO_USE enabled but profile directory not found: ${PGO_PROFILE_DIR}")
        endif()

        message(STATUS "PGO: Using profile data from: ${PGO_PROFILE_DIR}")

        # Add optimization flags
        set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -fprofile-use=${PGO_PROFILE_DIR} -fprofile-correction")
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fprofile-use=${PGO_PROFILE_DIR} -fprofile-correction")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fprofile-use=${PGO_PROFILE_DIR}")

        add_definitions(-DPGO_OPTIMIZED_BUILD)
    endif()

# PGO for MSVC (different workflow)
elseif (MSVC)
    if (PGO_GENERATE OR PGO_USE)
        message(STATUS "PGO: MSVC PGO requires different workflow:")
        message(STATUS "  1. Build Release: cmake --build . --config Release")
        message(STATUS "  2. Instrument: link /LTCG /GENPROFILE myapp.exe")
        message(STATUS "  3. Train: run myapp.exe with representative workload")
        message(STATUS "  4. Optimize: link /LTCG /USEPROFILE myapp.exe")
        message(STATUS "See: https://docs.microsoft.com/en-us/cpp/build/profile-guided-optimizations")

        if (PGO_GENERATE)
            # MSVC uses linker flags for PGO
            set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /LTCG:PGINSTRUMENT /GENPROFILE:PGD=${PGO_PROFILE_DIR}/miner.pgd")
            add_definitions(-DPGO_TRAINING_BUILD)
        elseif (PGO_USE)
            set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /LTCG:PGOPTIMIZE /USEPROFILE:PGD=${PGO_PROFILE_DIR}/miner.pgd")
            add_definitions(-DPGO_OPTIMIZED_BUILD)
        endif()
    endif()

else()
    if (PGO_GENERATE OR PGO_USE)
        message(WARNING "PGO: Unsupported compiler ${CMAKE_CXX_COMPILER_ID}. PGO disabled.")
    endif()
endif()

# Print PGO status summary
if (PGO_GENERATE)
    message(STATUS "========================================")
    message(STATUS "PGO BUILD MODE: INSTRUMENTATION")
    message(STATUS "Profile output: ${PGO_PROFILE_DIR}")
    message(STATUS "========================================")
    message(STATUS "After building, run the miner with representative workload:")
    message(STATUS "  ./dirtybird-miner-cpu --dero --threads <N> --mine-time 120")
    message(STATUS "Then rebuild with PGO_USE=ON")
elseif (PGO_USE)
    message(STATUS "========================================")
    message(STATUS "PGO BUILD MODE: OPTIMIZED")
    message(STATUS "Using profiles from: ${PGO_PROFILE_PATH}")
    message(STATUS "========================================")
endif()
