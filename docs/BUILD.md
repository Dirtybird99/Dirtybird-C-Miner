# DERO C++ Miner Build Guide

This document describes how to build the DERO C++ miner with optimized AstroBWTv3 implementation.

## Prerequisites

### Windows

1. **CMake 3.30+**
   - Download from: https://cmake.org/download/
   - Add to PATH or use full path: `C:\Program Files\CMake\bin\cmake.exe`

2. **MSYS2 MinGW64**
   - Download from: https://www.msys2.org/
   - Install to: `C:\msys64`
   - Open MSYS2 MinGW64 shell and install dependencies:
   ```bash
   pacman -Syu
   pacman -S mingw-w64-x86_64-clang mingw-w64-x86_64-lld mingw-w64-x86_64-ninja
   pacman -S mingw-w64-x86_64-boost mingw-w64-x86_64-openssl
   ```

3. **Ninja Build System** (installed via MSYS2 above)

### Linux

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install cmake ninja-build clang lld
sudo apt-get install libboost-all-dev libssl-dev libnuma-dev libudns-dev

# Fedora
sudo dnf install cmake ninja-build clang lld
sudo dnf install boost-devel openssl-devel numactl-devel udns-devel
```

### macOS

```bash
brew install cmake ninja llvm boost openssl@3
```

## Quick Start

### Using Build Scripts (Recommended for Windows)

```batch
cd dero-miner-cpp\scripts

REM Build release version
build_and_test.bat

REM Build and run tests
build_and_test.bat release --test

REM Build debug version
build_and_test.bat debug

REM Clean build
build_and_test.bat --clean release
```

### Using CMake Directly

```bash
cd dero-miner-cpp

# Configure (Release build)
cmake -B build -DCMAKE_BUILD_TYPE=Release -G Ninja

# Build
cmake --build build --parallel 4

# Output: build/bin/dirtybird-miner-cpu.exe (Windows) or dirtybird-miner-cpu (Linux)
```

## CMake Presets

The project includes CMake presets for common configurations. Use them with:

```bash
# Configure with preset
cmake --preset release

# Build with preset
cmake --build --preset release
```

### Available Presets

| Preset | Description | Build Dir |
|--------|-------------|-----------|
| `release` | Optimized build with LTO | `build/` |
| `debug` | Debug build with symbols | `build-debug/` |
| `profile` | Release with debug info for profiling | `build-profile/` |
| `no-lto` | Release without LTO (faster builds) | `build-nolto/` |
| `gcc` | Release using GCC instead of Clang | `build-gcc/` |
| `no-spsa` | Release without Astro-SPSA library | `build-nospsa/` |
| `benchmark` | Build with benchmark tools | `build-bench/` |

## CMake Options

### Main Options

| Option | Default | Description |
|--------|---------|-------------|
| `CMAKE_BUILD_TYPE` | Release | Build type (Release, Debug, RelWithDebInfo) |
| `WITH_ASTROBWTV3` | ON | Enable AstroBWTv3 algorithm |
| `USE_ASTRO_SPSA` | ON | Use Astro-SPSA suffix array library |
| `DISABLE_LTO` | OFF | Disable Link-Time Optimization |
| `WITH_PROFILING` | OFF | Enable profiling for developers |

### CPU Feature Options

| Option | Default | Description |
|--------|---------|-------------|
| `WITH_SSE4_1` | ON | Enable SSE 4.1 instructions |
| `WITH_AVX2` | ON | Enable AVX2 instructions |

### Compiler Options

| Option | Default | Description |
|--------|---------|-------------|
| `USE_GCC` | OFF | Use GCC instead of Clang |
| `BUILD_STATIC` | ON | Build static binary |

### GPU Options (HIP)

| Option | Default | Description |
|--------|---------|-------------|
| `WITH_HIP` | OFF | Enable GPU support through HIP SDK |
| `HIP_PLATFORM` | "" | HIP platform: "amd" or "nvidia" |

## Build Examples

### Release Build (Default)

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -G Ninja
cmake --build build --parallel
```

### Debug Build for Development

```bash
cmake -B build-debug -DCMAKE_BUILD_TYPE=Debug -DDISABLE_LTO=ON -G Ninja
cmake --build build-debug --parallel
```

### Profile Build for Performance Analysis

```bash
cmake -B build-profile -DCMAKE_BUILD_TYPE=RelWithDebInfo -DWITH_PROFILING=ON -G Ninja
cmake --build build-profile --parallel
```

### Build Without SPSA Library

```bash
cmake -B build -DUSE_ASTRO_SPSA=OFF -G Ninja
cmake --build build --parallel
```

### Build with GCC

```bash
cmake -B build-gcc -DUSE_GCC=ON -G Ninja
cmake --build build-gcc --parallel
```

### Build GPU Version (AMD ROCm)

```bash
cmake -B build-rocm -DWITH_HIP=ON -DHIP_PLATFORM=amd -G Ninja
cmake --build build-rocm --parallel
```

### Build GPU Version (NVIDIA CUDA via HIP)

```bash
cmake -B build-cuda -DWITH_HIP=ON -DHIP_PLATFORM=nvidia -G Ninja
cmake --build build-cuda --parallel
```

## Testing

### Run Self-Test

```bash
# After building
./build/bin/dirtybird-miner-cpu --test
```

### Run Benchmark

```bash
./build/bin/dirtybird-miner-cpu --benchmark 10
```

### Using Test Script (Windows)

```batch
cd scripts
run_tests.bat
run_tests.bat --benchmark
```

## Optimization Modules

The miner includes several optimization modules that can be tested independently:

1. **Branch Tables** (`branch_tables.cpp`)
   - Precomputed lookup tables for wolf branching operations
   - AVX2-optimized batch processing

2. **Incremental Suffix Array** (`sa_incremental.cpp`)
   - O(n) updates for single-byte changes
   - Avoids full O(n log n) rebuild

3. **RC4-AVX512** (`rc4_avx512.cpp`)
   - 16-way parallel RC4 using AVX-512
   - 8-way fallback for AVX2

4. **SHA256-NI** (`sha256_spsa.cpp`)
   - Hardware-accelerated SHA256 using Intel SHA-NI
   - Compressed state format for SPSA integration

5. **SPSA State** (`spsa_state.cpp`)
   - Stamped Permutation Suffix Array management
   - Optimized for repeated mining operations

## Troubleshooting

### CMake Configuration Fails

1. Ensure MSYS2 MinGW64 is installed correctly
2. Check that Clang is available: `clang --version`
3. Verify Boost is installed: check for `C:\msys64\mingw64\include\boost\version.hpp`

### Link Errors

1. Ensure all dependencies are installed
2. Try building without LTO: `-DDISABLE_LTO=ON`
3. Check for missing source files in cmake configuration

### Runtime Errors

1. Ensure `WinRing0x64.dll` and `WinRing0x64.sys` are in the same directory as the executable
2. Run with administrator privileges if WinRing0 fails to load
3. Check CPU feature support (AVX2, SHA-NI) matches build options

### Build Performance

- LTO significantly increases link time; use `--preset no-lto` for faster iterative builds
- Use parallel build: `cmake --build build --parallel 8`
- Consider incremental builds instead of full rebuilds

## Project Structure

```
dero-miner-cpp/
├── CMakeLists.txt           # Main CMake configuration
├── CMakePresets.json        # CMake presets for build configurations
├── cmake/                   # CMake modules
│   ├── astrobwtv3.cmake     # AstroBWTv3 sources
│   ├── cpu.cmake            # CPU feature detection
│   ├── os.cmake             # OS detection
│   ├── setup.cmake          # Target setup helpers
│   └── spsa.cmake           # SPSA library integration
├── scripts/                 # Build and test scripts
│   ├── build_and_test.bat   # Windows build script
│   └── run_tests.bat        # Windows test runner
├── src/
│   ├── core/                # Main miner logic
│   ├── crypto/astrobwtv3/   # AstroBWTv3 implementation
│   │   ├── astrobwtv3.cpp   # Main hash function
│   │   ├── branch_tables.cpp
│   │   ├── sa_incremental.cpp
│   │   ├── rc4_avx512.cpp
│   │   ├── sha256_spsa.cpp
│   │   └── optimization_tests.cpp
│   ├── coins/               # Coin-specific mining code
│   └── net/                 # Network communication
├── include/                 # Header files
├── lib/                     # External libraries
└── docs/                    # Documentation
```
