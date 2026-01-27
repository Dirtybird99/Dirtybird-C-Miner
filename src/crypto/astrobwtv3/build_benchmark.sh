#!/bin/bash
# SA Benchmark Build Script for Linux/macOS
#
# Usage:
#   ./build_benchmark.sh           - Build with Release optimization
#   ./build_benchmark.sh debug     - Build with debug symbols
#   ./build_benchmark.sh clean     - Clean build artifacts

set -e

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SRC_DIR}/benchmark_build"

if [ "$1" = "clean" ]; then
    echo "Cleaning build artifacts..."
    rm -rf "${BUILD_DIR}"
    rm -f "${SRC_DIR}/sa_benchmark"
    echo "Done."
    exit 0
fi

# Create build directory
mkdir -p "${BUILD_DIR}"

# Detect compiler
if command -v clang++ &> /dev/null; then
    CXX=clang++
    CC=clang
else
    CXX=g++
    CC=gcc
fi

echo "Using compiler: ${CXX}"

# Set optimization flags
if [ "$1" = "debug" ]; then
    CXXFLAGS="-g -O0 -DDEBUG"
    CFLAGS="-g -O0 -DDEBUG"
    echo "Building in DEBUG mode..."
else
    CXXFLAGS="-O3 -DNDEBUG -march=native -mtune=native -fno-omit-frame-pointer"
    CFLAGS="-O3 -DNDEBUG -march=native -mtune=native"
    echo "Building in RELEASE mode..."
fi

# Common flags
CXXFLAGS="${CXXFLAGS} -std=c++20 -Wall -Wextra -DBUILD_SA_BENCHMARK -I${SRC_DIR} -I${SRC_DIR}/../../../include"
CFLAGS="${CFLAGS} -std=c11 -Wall -I${SRC_DIR}"

# Platform-specific flags
if [ "$(uname)" = "Linux" ]; then
    LDFLAGS="-lpthread"
elif [ "$(uname)" = "Darwin" ]; then
    LDFLAGS=""
else
    LDFLAGS=""
fi

echo ""
echo "Compiling C sources..."

# Compile C files
${CC} ${CFLAGS} -c "${SRC_DIR}/divsufsort.c" -o "${BUILD_DIR}/divsufsort.o"
${CC} ${CFLAGS} -c "${SRC_DIR}/sssort.c" -o "${BUILD_DIR}/sssort.o"
${CC} ${CFLAGS} -c "${SRC_DIR}/trsort.c" -o "${BUILD_DIR}/trsort.o"
${CC} ${CFLAGS} -c "${SRC_DIR}/utils.c" -o "${BUILD_DIR}/utils.o"

echo "Compiling C++ sources..."

# Compile C++ files
${CXX} ${CXXFLAGS} -c "${SRC_DIR}/sa_incremental.cpp" -o "${BUILD_DIR}/sa_incremental.o"
${CXX} ${CXXFLAGS} -c "${SRC_DIR}/sa_benchmark.cpp" -o "${BUILD_DIR}/sa_benchmark.o"

echo "Linking..."

# Link
${CXX} ${CXXFLAGS} \
    "${BUILD_DIR}/sa_benchmark.o" \
    "${BUILD_DIR}/sa_incremental.o" \
    "${BUILD_DIR}/divsufsort.o" \
    "${BUILD_DIR}/sssort.o" \
    "${BUILD_DIR}/trsort.o" \
    "${BUILD_DIR}/utils.o" \
    ${LDFLAGS} \
    -o "${SRC_DIR}/sa_benchmark"

echo ""
echo "========================================"
echo "Build successful!"
echo "Executable: ${SRC_DIR}/sa_benchmark"
echo "========================================"
echo ""
echo "Usage examples:"
echo "  ./sa_benchmark                    - Run all benchmarks"
echo "  ./sa_benchmark -t                 - Run correctness tests"
echo "  ./sa_benchmark -i                 - Run incremental benchmark"
echo "  ./sa_benchmark -n 50000           - Run 50000 iterations"
echo "  ./sa_benchmark -o results.csv     - Export to CSV"
echo ""
