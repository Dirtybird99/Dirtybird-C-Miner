if (WITH_ASTROBWTV3)
    add_definitions(/DDIRTYBIRD_ASTROBWTV3)

    message(STATUS "Building with AstroBWTv3 support (DERO-only)")

    # SA backend selection
    if (USE_DLUNA_RADIX_SA)
        add_definitions(-DUSE_DLUNA_RADIX_SA)
        message(STATUS "Using DeroLuna-style 24-bit radix suffix array")
    elseif (USE_CUSTOM_SA)
        add_definitions(-DUSE_CUSTOM_SA)
        message(STATUS "Using custom 70KB SA-IS algorithm instead of divsufsort")
    elseif (USE_LIBSAIS)
        add_definitions(-DUSE_LIBSAIS)
        message(STATUS "Using libsais 2.10.4 for suffix array construction (state-of-the-art)")
    else()
        message(STATUS "Using divsufsort for suffix array construction")
    endif()

    # SA Instrumentation for access pattern analysis
    if (ENABLE_SA_INSTRUMENTATION)
        add_definitions(-DENABLE_SA_INSTRUMENTATION)
        message(STATUS "SA instrumentation ENABLED - collecting access pattern metrics")
    else()
        message(STATUS "SA instrumentation disabled (use -DENABLE_SA_INSTRUMENTATION=ON to enable)")
    endif()

    file(GLOB_RECURSE astroHeaders
      src/crypto/astrobwtv3/*.h
    )

    file(GLOB_RECURSE astroSources
      src/crypto/astrobwtv3/*.cpp
      src/crypto/astrobwtv3/*.c
      src/net/dero/*.cpp
    )

    # Exclude test/benchmark files from production build (saves ~1.5 MB)
    list(FILTER astroSources EXCLUDE REGEX ".*optimization_tests\\.cpp$")
    list(FILTER astroSources EXCLUDE REGEX ".*sa_benchmark\\.cpp$")
    list(FILTER astroSources EXCLUDE REGEX ".*sa_instrumentation\\.c$")

    list(APPEND astroSources
      src/coins/dero_worker_pool.cpp
      src/coins/mine_dero.cpp
      src/coins/mine_dero_batched.cpp
      src/coins/mine_dero_interleaved.cpp
      src/coins/mine_dero_lockfree.cpp
      src/crypto/sha256_override.cpp
      src/crypto/salsa20_simd.c
    )

    # libsais for faster suffix array construction
    if (USE_LIBSAIS)
        set(LIBSAIS_DIR "${CMAKE_SOURCE_DIR}/../libsais-2.10.4")
        if (EXISTS "${LIBSAIS_DIR}/src/libsais.c")
            list(APPEND astroSources "${LIBSAIS_DIR}/src/libsais.c")
            include_directories("${LIBSAIS_DIR}/include")
            message(STATUS "libsais source added: ${LIBSAIS_DIR}/src/libsais.c")
        else()
            message(FATAL_ERROR "libsais not found at ${LIBSAIS_DIR}. Please ensure dero-miner-main/libsais-2.10.4 exists.")
        endif()
    endif()

    list(APPEND HEADERS_CRYPTO
      ${astroHeaders}
    )

    list(APPEND SOURCES_CRYPTO
      ${astroSources}
    )

    # if (WITH_MSR AND NOT DIRTYBIRD_ARM AND CMAKE_SIZEOF_VOID_P EQUAL 8 AND (DIRTYBIRD_OS_WIN OR DIRTYBIRD_OS_LINUX))
    #     add_definitions(/DDIRTYBIRD_FEATURE_MSR)
    #     add_definitions(/DDIRTYBIRD_FIX_RYZEN)
    #     message("-- WITH_MSR=ON")

    #     if (DIRTYBIRD_OS_WIN)
    #         list(APPEND SOURCES_CRYPTO
    #             src/crypto/rx/RxFix_win.cpp
    #             src/hw/msr/Msr_win.cpp
    #             )
    #     elseif (DIRTYBIRD_OS_LINUX)
    #         list(APPEND SOURCES_CRYPTO
    #             src/crypto/rx/RxFix_linux.cpp
    #             src/hw/msr/Msr_linux.cpp
    #             )
    #     endif()

    #     list(APPEND HEADERS_CRYPTO
    #         src/crypto/rx/RxFix.h
    #         src/crypto/rx/RxMsr.h
    #         src/hw/msr/Msr.h
    #         src/hw/msr/MsrItem.h
    #         )

    #     list(APPEND SOURCES_CRYPTO
    #         src/crypto/rx/RxMsr.cpp
    #         src/hw/msr/Msr.cpp
    #         src/hw/msr/MsrItem.cpp
    #         )
    # else()
    #     remove_definitions(/DDIRTYBIRD_FEATURE_MSR)
    #     remove_definitions(/DDIRTYBIRD_FIX_RYZEN)
    #     message("-- WITH_MSR=OFF")
    # endif()

else()
    remove_definitions(/DDIRTYBIRD_ASTROBWTV3)
endif()
