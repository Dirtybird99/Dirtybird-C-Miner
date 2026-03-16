if(NOT WIN32)
  return()
endif()

set(WINRING0_ROOT_DIR "${PROJECT_SOURCE_DIR}/lib/winring0")
set(WINRING0_FOUND FALSE)
set(WINRING0_STATUS_MSG "")

if(CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(WINRING0_DLL "${WINRING0_ROOT_DIR}/WinRing0x64.dll")
  set(WINRING0_SYS "${WINRING0_ROOT_DIR}/WinRing0x64.sys")
  set(WINRING0_IMPLIB "${WINRING0_ROOT_DIR}/libWinRing0x64.a")
else()
  set(WINRING0_DLL "${WINRING0_ROOT_DIR}/WinRing0.dll")
  set(WINRING0_SYS "${WINRING0_ROOT_DIR}/WinRing0.sys")
  set(WINRING0_IMPLIB "${WINRING0_ROOT_DIR}/libWinRing0.a")
endif()

# DIRTYBIRD can build and run without a bundled WinRing0 drop-in. When these
# files are absent we keep MSR support runtime-optional instead of failing
# configure for clean source checkouts and CI runners.
if(NOT EXISTS "${WINRING0_ROOT_DIR}/OlsApi.h")
  set(WINRING0_STATUS_MSG "WinRing0 header not found at ${WINRING0_ROOT_DIR}/OlsApi.h; skipping bundled WinRing0 support.")
elseif(NOT EXISTS "${WINRING0_DLL}")
  set(WINRING0_STATUS_MSG "WinRing0 DLL not found at ${WINRING0_DLL}; skipping bundled WinRing0 support.")
elseif(NOT EXISTS "${WINRING0_SYS}")
  set(WINRING0_STATUS_MSG "WinRing0 driver not found at ${WINRING0_SYS}; skipping bundled WinRing0 support.")
else()
  set(WINRING0_FOUND TRUE)
endif()

function(target_link_winring0 target_name)
  if(NOT TARGET ${target_name})
    message(FATAL_ERROR "Invalid target: ${target_name}")
  endif()

  if(NOT WINRING0_FOUND)
    if(WINRING0_STATUS_MSG)
      message(STATUS "Skipping WinRing0 for ${target_name}: ${WINRING0_STATUS_MSG}")
    endif()
    return()
  endif()

  message(STATUS "Linking WinRing0 to ${target_name} using ${WINRING0_IMPLIB}")

  target_include_directories(${target_name} PRIVATE "${WINRING0_ROOT_DIR}")
  target_link_libraries(${target_name} ${WINRING0_IMPLIB})

  add_custom_command(TARGET ${target_name} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${WINRING0_DLL}" $<TARGET_FILE_DIR:${target_name}>
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${WINRING0_SYS}" $<TARGET_FILE_DIR:${target_name}>
    COMMENT "Copying WinRing0 runtime files"
  )

  target_compile_definitions(${target_name} PRIVATE DIRTYBIRD_HAS_WINRING0=1)
endfunction()

if(NOT WINRING0_FOUND)
  message(STATUS "${WINRING0_STATUS_MSG}")
  return()
endif()

# Create .def file for import library generation
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(WINRING0_DEF "${CMAKE_BINARY_DIR}/WinRing0x64.def")
else()
  set(WINRING0_DEF "${CMAKE_BINARY_DIR}/WinRing0.def")
endif()

# Write .def file with all the exports
file(WRITE ${WINRING0_DEF}
"EXPORTS
InitializeOls
DeinitializeOls
GetDllStatus
GetDllVersion
GetDriverVersion
GetDriverType
IsCpuid
IsMsr
IsTsc
Rdmsr
RdmsrTx
RdmsrPx
Wrmsr
WrmsrTx
WrmsrPx
Cpuid
CpuidTx
CpuidPx
Rdtsc
RdtscTx
RdtscPx
ReadIoPortByte
ReadIoPortWord
ReadIoPortDword
WriteIoPortByte
WriteIoPortWord
WriteIoPortDword
")

# Find dlltool (should be available with MinGW/clang)
find_program(DLLTOOL_EXECUTABLE NAMES 
  llvm-dlltool 
  dlltool
  x86_64-w64-mingw32-dlltool  # Common MinGW path
)

if(NOT DLLTOOL_EXECUTABLE)
  message(FATAL_ERROR "dlltool not found. Please install MinGW development tools.")
endif()

# Generate import library if it doesn't exist
if(NOT EXISTS ${WINRING0_IMPLIB})
  message(STATUS "Creating MinGW-compatible import library for WinRing0...")
  
  execute_process(
    COMMAND ${DLLTOOL_EXECUTABLE} 
      --def ${WINRING0_DEF}
      --dllname $<IF:$<EQUAL:${CMAKE_SIZEOF_VOID_P},8>,WinRing0x64.dll,WinRing0.dll>
      --output-lib ${WINRING0_IMPLIB}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    RESULT_VARIABLE DLLTOOL_RESULT
    OUTPUT_VARIABLE DLLTOOL_OUTPUT
    ERROR_VARIABLE DLLTOOL_ERROR
  )
  
  if(NOT DLLTOOL_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to create import library: ${DLLTOOL_ERROR}")
  endif()
  
  message(STATUS "Successfully created import library: ${WINRING0_IMPLIB}")
endif()

# set(WINRING0_FOUND TRUE)
