if (WITH_ASTROBWTV3)
    add_definitions(/DDIRTYBIRD_ASTROBWTV3)

    message(STATUS "Building with AstroBWTv3 support (DERO-only)")

    # Custom SA-IS algorithm option (for 70KB inputs)
    if (USE_CUSTOM_SA)
        add_definitions(-DUSE_CUSTOM_SA)
        message(STATUS "Using custom 70KB SA-IS algorithm instead of divsufsort")
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

    list(APPEND astroSources
      src/coins/mine_dero.cpp
      src/coins/mine_dero_batched.cpp
      src/coins/mine_dero_interleaved.cpp
      src/coins/mine_dero_lockfree.cpp
      src/crypto/salsa20_simd.c
    )

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
