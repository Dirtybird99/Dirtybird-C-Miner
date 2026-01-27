@echo off
REM ============================================================================
REM DERO C++ Miner - Build and Test Script for Windows
REM ============================================================================
REM
REM This script builds the DERO C++ miner and runs integration tests.
REM
REM Prerequisites:
REM   - CMake 3.30+ (C:\Program Files\CMake\bin in PATH or use full path)
REM   - MSYS2 MinGW64 with Clang and required dependencies
REM   - Ninja build system
REM
REM Usage:
REM   build_and_test.bat [release|debug|profile] [--test] [--clean] [--help]
REM
REM Examples:
REM   build_and_test.bat                   Build release version
REM   build_and_test.bat debug             Build debug version
REM   build_and_test.bat release --test    Build release and run tests
REM   build_and_test.bat --clean           Clean build directory first
REM
REM ============================================================================

setlocal enabledelayedexpansion

REM Default values
set BUILD_TYPE=Release
set RUN_TESTS=0
set CLEAN_BUILD=0
set PARALLEL_JOBS=4

REM Script directory
set SCRIPT_DIR=%~dp0
set PROJECT_DIR=%SCRIPT_DIR%..
set BUILD_DIR=%PROJECT_DIR%\build

REM Parse arguments
:parse_args
if "%~1"=="" goto :done_args
if /i "%~1"=="release" (
    set BUILD_TYPE=Release
    shift
    goto :parse_args
)
if /i "%~1"=="debug" (
    set BUILD_TYPE=Debug
    shift
    goto :parse_args
)
if /i "%~1"=="profile" (
    set BUILD_TYPE=RelWithDebInfo
    shift
    goto :parse_args
)
if /i "%~1"=="--test" (
    set RUN_TESTS=1
    shift
    goto :parse_args
)
if /i "%~1"=="--clean" (
    set CLEAN_BUILD=1
    shift
    goto :parse_args
)
if /i "%~1"=="--help" (
    goto :show_help
)
if /i "%~1"=="-h" (
    goto :show_help
)
echo Unknown argument: %~1
goto :show_help

:done_args

REM Find CMake
set CMAKE_EXE=cmake
where cmake >nul 2>&1
if %errorlevel% neq 0 (
    if exist "C:\Program Files\CMake\bin\cmake.exe" (
        set CMAKE_EXE="C:\Program Files\CMake\bin\cmake.exe"
    ) else (
        echo ERROR: CMake not found. Please install CMake 3.30+ and add to PATH.
        exit /b 1
    )
)

echo ============================================================================
echo DERO C++ Miner Build Script
echo ============================================================================
echo Build Type: %BUILD_TYPE%
echo Build Dir:  %BUILD_DIR%
echo CMake:      %CMAKE_EXE%
echo ============================================================================

REM Clean if requested
if %CLEAN_BUILD%==1 (
    echo.
    echo Cleaning build directory...
    if exist "%BUILD_DIR%" (
        rmdir /s /q "%BUILD_DIR%"
    )
)

REM Create build directory
if not exist "%BUILD_DIR%" (
    mkdir "%BUILD_DIR%"
)

REM Configure with CMake
echo.
echo [1/3] Configuring with CMake...
cd /d "%PROJECT_DIR%"
%CMAKE_EXE% -B build -DCMAKE_BUILD_TYPE=%BUILD_TYPE% -G Ninja
if %errorlevel% neq 0 (
    echo ERROR: CMake configuration failed.
    exit /b 1
)

REM Build
echo.
echo [2/3] Building...
%CMAKE_EXE% --build build --parallel %PARALLEL_JOBS%
if %errorlevel% neq 0 (
    echo ERROR: Build failed.
    exit /b 1
)

echo.
echo ============================================================================
echo Build completed successfully!
echo Output: %BUILD_DIR%\bin\dirtybird-miner-cpu.exe
echo ============================================================================

REM Run tests if requested
if %RUN_TESTS%==1 (
    echo.
    echo [3/3] Running tests...
    call "%SCRIPT_DIR%run_tests.bat"
    if %errorlevel% neq 0 (
        echo WARNING: Some tests failed.
    )
)

exit /b 0

:show_help
echo.
echo Usage: build_and_test.bat [release^|debug^|profile] [options]
echo.
echo Build Types:
echo   release    Optimized build with LTO (default)
echo   debug      Debug build with symbols, no optimization
echo   profile    Release build with debug info for profiling
echo.
echo Options:
echo   --test     Run tests after building
echo   --clean    Clean build directory before building
echo   --help     Show this help message
echo.
echo Examples:
echo   build_and_test.bat                   Build release version
echo   build_and_test.bat debug             Build debug version
echo   build_and_test.bat release --test    Build release and run tests
echo   build_and_test.bat --clean debug     Clean build, then build debug
echo.
exit /b 0
