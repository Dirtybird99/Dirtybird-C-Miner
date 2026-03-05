@echo off
REM SA Benchmark Build Script for Windows (MinGW/GCC)
REM
REM Usage:
REM   build_benchmark.bat           - Build with Release optimization
REM   build_benchmark.bat debug     - Build with debug symbols
REM   build_benchmark.bat clean     - Clean build artifacts

setlocal enabledelayedexpansion

set SRC_DIR=%~dp0
set BUILD_DIR=%SRC_DIR%benchmark_build

if "%1"=="clean" (
    echo Cleaning build artifacts...
    if exist "%BUILD_DIR%" rmdir /s /q "%BUILD_DIR%"
    if exist "%SRC_DIR%sa_benchmark.exe" del "%SRC_DIR%sa_benchmark.exe"
    echo Done.
    exit /b 0
)

REM Create build directory
if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"

REM Set compiler (prefer Clang if available)
where clang++ >nul 2>nul
if %errorlevel%==0 (
    set CXX=clang++
    set CC=clang
) else (
    set CXX=g++
    set CC=gcc
)

echo Using compiler: %CXX%

REM Set optimization flags
if "%1"=="debug" (
    set CXXFLAGS=-g -O0 -DDEBUG
    set CFLAGS=-g -O0 -DDEBUG
    echo Building in DEBUG mode...
) else (
    set CXXFLAGS=-O3 -DNDEBUG -march=native -mtune=native -fno-omit-frame-pointer
    set CFLAGS=-O3 -DNDEBUG -march=native -mtune=native
    echo Building in RELEASE mode...
)

REM Common flags
set CXXFLAGS=%CXXFLAGS% -std=c++20 -Wall -Wextra -DBUILD_SA_BENCHMARK -I"%SRC_DIR%" -I"%SRC_DIR%..\..\..\include"
set CFLAGS=%CFLAGS% -std=c11 -Wall -I"%SRC_DIR%"

echo.
echo Compiling C sources...

REM Compile C files
%CC% %CFLAGS% -c "%SRC_DIR%divsufsort.c" -o "%BUILD_DIR%\divsufsort.o"
if %errorlevel% neq 0 goto :error

%CC% %CFLAGS% -c "%SRC_DIR%sssort.c" -o "%BUILD_DIR%\sssort.o"
if %errorlevel% neq 0 goto :error

%CC% %CFLAGS% -c "%SRC_DIR%trsort.c" -o "%BUILD_DIR%\trsort.o"
if %errorlevel% neq 0 goto :error

%CC% %CFLAGS% -c "%SRC_DIR%utils.c" -o "%BUILD_DIR%\utils.o"
if %errorlevel% neq 0 goto :error

echo Compiling C++ sources...

REM Compile C++ files
%CXX% %CXXFLAGS% -c "%SRC_DIR%sa_incremental.cpp" -o "%BUILD_DIR%\sa_incremental.o"
if %errorlevel% neq 0 goto :error

%CXX% %CXXFLAGS% -c "%SRC_DIR%sa_benchmark.cpp" -o "%BUILD_DIR%\sa_benchmark.o"
if %errorlevel% neq 0 goto :error

echo Linking...

REM Link
%CXX% %CXXFLAGS% ^
    "%BUILD_DIR%\sa_benchmark.o" ^
    "%BUILD_DIR%\sa_incremental.o" ^
    "%BUILD_DIR%\divsufsort.o" ^
    "%BUILD_DIR%\sssort.o" ^
    "%BUILD_DIR%\trsort.o" ^
    "%BUILD_DIR%\utils.o" ^
    -o "%SRC_DIR%sa_benchmark.exe"
if %errorlevel% neq 0 goto :error

echo.
echo ========================================
echo Build successful!
echo Executable: %SRC_DIR%sa_benchmark.exe
echo ========================================
echo.
echo Usage examples:
echo   sa_benchmark.exe                    - Run all benchmarks
echo   sa_benchmark.exe -t                 - Run correctness tests
echo   sa_benchmark.exe -i                 - Run incremental benchmark
echo   sa_benchmark.exe -n 50000           - Run 50000 iterations
echo   sa_benchmark.exe -o results.csv     - Export to CSV
echo.

exit /b 0

:error
echo.
echo ========================================
echo BUILD FAILED!
echo ========================================
exit /b 1
