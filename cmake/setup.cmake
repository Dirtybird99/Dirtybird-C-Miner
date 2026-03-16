# Define a function to set properties and link libraries for a target
function(setup_target_libraries target_name)
  # Link Boost libraries
  target_link_libraries(${target_name} ${DIRTYBIRD_BOOST_LIBS})

  # Link OpenMP if found (for parallel divsufsort)
  if(OpenMP_FOUND)
    target_link_libraries(${target_name} OpenMP::OpenMP_CXX)
  endif()

  # Link Windows-specific libraries
  if(WIN32)
    if (NOT WITH_HIP)
      target_link_libraries(${target_name} mingw32)
    endif()
    target_link_libraries(${target_name} wsock32 ws2_32 kernel32 winmm)
    target_link_winring0(${target_name})
  endif()

  # Link libraries for non-Apple, non-Windows systems (likely Linux)
  if(NOT APPLE AND NOT WIN32 AND NOT WIN_CROSS AND NOT DIRTYBIRD_OS_ANDROID)
    target_link_libraries(${target_name} udns numa)
  endif()

  # Link AstroSPSA library if enabled
  if(USE_ASTRO_SPSA)
    cmake_print_variables(SPSA_FULL_LIB_PATH)
    target_link_libraries(${target_name} ${SPSA_FULL_LIB_PATH})
  endif()

  # if (WIN_CROSS)
  #   add_library(sodium STATIC IMPORTED)
  #   set(LibSodium_ROOT "${PROJECT_SOURCE_DIR}/wincross/clang64/lib")
  #   #target_link_libraries(${target_name} PRIVATE ${LibSodium_ROOT})
  #   #find_package(LibSodium REQUIRED)
  #   link_directories(LibSodium_ROOT)
  #   link_directories("${PROJECT_SOURCE_DIR}/export/boost-x64/lib")
  # endif()

  # Link threading libraries, OpenSSL, and other required libraries
  if (WIN_CROSS)
    #add_library(${THREAD_LIB} STATIC IMPORTED)
    #add_library(sodium STATIC IMPORTED)
    #target_link_libraries(${target_name} OpenSSL::SSL OpenSSL::Crypto)
    #target_link_libraries(${target_name} ${BLAKE3_LIBRARIES})
    #target_link_libraries(${target_name} ${THREAD_LIB} OpenSSL::SSL OpenSSL::Crypto sodium ${BLAKE3_LIBRARIES})
  else()
    #target_link_libraries(${target_name} ${THREAD_LIB} OpenSSL::SSL OpenSSL::Crypto sodium ${BLAKE3_LIBRARIES})
    target_link_libraries(${target_name} ${THREAD_LIB} OpenSSL::SSL OpenSSL::Crypto ${BLAKE3_LIBRARIES})
  endif()

  set_target_properties(${target_name} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  )

  if(WIN32 AND OpenMP_FOUND)
    set(DIRTYBIRD_RUNTIME_DLL_HINTS
      "C:/msys64/mingw64/bin"
      "$ENV{MSYSTEM_PREFIX}/bin"
    )

    if(NOT DEFINED DIRTYBIRD_LIBOMP_DLL)
      find_program(DIRTYBIRD_LIBOMP_DLL NAMES libomp.dll HINTS ${DIRTYBIRD_RUNTIME_DLL_HINTS})
    endif()
    if(NOT DEFINED DIRTYBIRD_LIBGCC_DLL)
      find_program(DIRTYBIRD_LIBGCC_DLL NAMES libgcc_s_seh-1.dll HINTS ${DIRTYBIRD_RUNTIME_DLL_HINTS})
    endif()
    if(NOT DEFINED DIRTYBIRD_WINPTHREAD_DLL)
      find_program(DIRTYBIRD_WINPTHREAD_DLL NAMES libwinpthread-1.dll HINTS ${DIRTYBIRD_RUNTIME_DLL_HINTS})
    endif()

    set(DIRTYBIRD_RUNTIME_DLLS "")
    foreach(runtime_dll
      "${DIRTYBIRD_LIBOMP_DLL}"
      "${DIRTYBIRD_LIBGCC_DLL}"
      "${DIRTYBIRD_WINPTHREAD_DLL}"
    )
      if(runtime_dll)
        list(APPEND DIRTYBIRD_RUNTIME_DLLS "${runtime_dll}")
      endif()
    endforeach()

    if(DIRTYBIRD_RUNTIME_DLLS)
      add_custom_command(TARGET ${target_name} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${DIRTYBIRD_RUNTIME_DLLS} $<TARGET_FILE_DIR:${target_name}>
        COMMENT "Copying MinGW/OpenMP runtimes"
      )
    endif()
  endif()
endfunction()
