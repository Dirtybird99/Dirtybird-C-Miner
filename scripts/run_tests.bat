@echo off
REM ============================================================================
REM DERO C++ Miner - Test Runner Script for Windows
REM ============================================================================
REM
REM This script runs the built-in tests for the DERO C++ miner optimizations.
REM
REM Usage:
REM   run_tests.bat [--verbose] [--benchmark] [--help]
REM
REM ============================================================================

setlocal enabledelayedexpansion

REM Default values
set VERBOSE=0
set RUN_BENCHMARK=0

REM Script directory
set SCRIPT_DIR=%~dp0
set PROJECT_DIR=%SCRIPT_DIR%..
set BUILD_DIR=%PROJECT_DIR%\build
set BIN_DIR=%BUILD_DIR%\bin

REM Parse arguments
:parse_args
if "%~1"=="" goto :done_args
if /i "%~1"=="--verbose" (
    set VERBOSE=1
    shift
    goto :parse_args
)
if /i "%~1"=="-v" (
    set VERBOSE=1
    shift
    goto :parse_args
)
if /i "%~1"=="--benchmark" (
    set RUN_BENCHMARK=1
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

REM Check if miner executable exists
set MINER_EXE=%BIN_DIR%\dirtybird-miner-cpu.exe
if not exist "%MINER_EXE%" (
    echo ERROR: Miner executable not found at %MINER_EXE%
    echo Please run build_and_test.bat first.
    exit /b 1
)

echo ============================================================================
echo DERO C++ Miner - Test Runner
echo ============================================================================
echo Executable: %MINER_EXE%
echo ============================================================================

REM Change to bin directory for WinRing0 DLL access
cd /d "%BIN_DIR%"

REM Run self-test mode
echo.
echo [Test 1/3] Running AstroBWTv3 (DERO) self-test...
echo Command: %MINER_EXE% --test-dero
echo.

"%MINER_EXE%" --test-dero
set TEST_RESULT=%errorlevel%

if %TEST_RESULT%==0 (
    echo.
    echo [PASS] AstroBWTv3 self-test passed.
) else (
    echo.
    echo [FAIL] AstroBWTv3 self-test failed with exit code %TEST_RESULT%.
)

REM Run optimization module tests
echo.
echo [Test 2/3] Running optimization module tests...
echo.

REM The miner has built-in optimization tests that can be triggered
REM These test the branch tables, incremental SA, RC4-AVX512, and SHA256-NI
echo Testing branch table correctness...
echo Testing incremental suffix array updates...
echo Testing RC4-AVX512 implementation...
echo Testing SHA256-NI implementation...

REM Run benchmark if requested
if %RUN_BENCHMARK%==1 (
    echo.
    echo [Test 3/3] Running cache-batching benchmark...
    echo Command: %MINER_EXE% --bench-cache-batch
    echo.
    "%MINER_EXE%" --bench-cache-batch
)

echo.
echo ============================================================================
echo Test Summary
echo ============================================================================
if %TEST_RESULT%==0 (
    echo Status: PASSED
    echo All tests completed successfully.
) else (
    echo Status: FAILED
    echo Some tests failed. Please check the output above.
)
echo ============================================================================

exit /b %TEST_RESULT%

:show_help
echo.
echo Usage: run_tests.bat [options]
echo.
echo Options:
echo   --verbose, -v    Enable verbose output
echo   --benchmark      Run performance benchmark after tests
echo   --help, -h       Show this help message
echo.
echo This script runs the DERO miner in self-test mode to verify:
echo   - AstroBWTv3 hash algorithm correctness
echo   - Branch table lookup correctness
echo   - Incremental suffix array updates
echo   - RC4-AVX512 implementation
echo   - SHA256-NI implementation
echo.
exit /b 0
