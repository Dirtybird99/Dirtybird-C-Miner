@echo off
REM ============================================================================
REM DERO C++ Miner - Profile Build Script for Windows
REM ============================================================================
REM
REM This script builds the DERO miner with debug information for profiling.
REM
REM The resulting binary has full optimizations but includes debug symbols,
REM making it suitable for use with profilers like:
REM   - Visual Studio Performance Profiler
REM   - Intel VTune
REM   - AMD uProf
REM   - perf (on Linux)
REM
REM Usage:
REM   build_profile.bat
REM
REM ============================================================================

setlocal enabledelayedexpansion

set SCRIPT_DIR=%~dp0
set PROJECT_DIR=%SCRIPT_DIR%..
set BUILD_DIR=%PROJECT_DIR%\build-profile

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
echo DERO C++ Miner - Profile Build
echo ============================================================================
echo Build Type: RelWithDebInfo (optimized with debug symbols)
echo Build Dir:  %BUILD_DIR%
echo ============================================================================

REM Configure with CMake preset
echo.
echo [1/2] Configuring with CMake preset 'profile'...
cd /d "%PROJECT_DIR%"
%CMAKE_EXE% --preset profile
if %errorlevel% neq 0 (
    echo ERROR: CMake configuration failed.
    exit /b 1
)

REM Build
echo.
echo [2/2] Building...
%CMAKE_EXE% --build build-profile --parallel 4
if %errorlevel% neq 0 (
    echo ERROR: Build failed.
    exit /b 1
)

echo.
echo ============================================================================
echo Profile build completed successfully!
echo.
echo Output: %BUILD_DIR%\bin\dirtybird-miner-cpu.exe
echo.
echo This binary is optimized but includes debug symbols for profiling.
echo You can now use profilers like:
echo   - Visual Studio: Analyze ^> Performance Profiler
echo   - Intel VTune: amplxe-cl -collect hotspots ./dirtybird-miner-cpu.exe
echo   - AMD uProf: AMDuProfCLI collect --config tbp ./dirtybird-miner-cpu.exe
echo ============================================================================

exit /b 0
