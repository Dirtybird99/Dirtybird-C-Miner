# DIRTYBIRD Miner

A high-performance DERO cryptocurrency miner based on TNN-miner, optimized with AVX2 SIMD instructions.

## Features

- **AVX2 wolfPermute**: SIMD-optimized branch processing (32 bytes at once)
- **SPSA Integration**: Stamped Permutation Suffix Array for efficient BWT computation
- **Huge Pages Support**: Reduced TLB misses for better memory performance
- **No Dev Fee**: 100% of hashrate goes to your wallet

## Performance

With 20 threads on Intel i7-13700HX:
- ~19-20 KH/s (competitive with other leading miners)

## Building

### Windows (MSYS2/MinGW64)

```bash
# Install dependencies
pacman -S mingw-w64-x86_64-clang mingw-w64-x86_64-cmake mingw-w64-x86_64-boost mingw-w64-x86_64-openssl mingw-w64-x86_64-openmp

# Build
mkdir build && cd build
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release
ninja
```

### Linux

```bash
# Install dependencies (Ubuntu/Debian)
sudo apt install git build-essential cmake clang libssl-dev libboost-all-dev libomp-dev

# Build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## Usage

```bash
./dirtybird-miner-cpu --daemon-address <node:port> --wallet <your-wallet> --threads <num>
```

### Options

| Option | Description |
|--------|-------------|
| `--daemon-address` | Node/pool URL (e.g., `pool.example.com:10100`) |
| `--wallet` | Your DERO wallet address |
| `--threads` | Number of mining threads |
| `--test-dero` | Run hash verification tests |

## Testing

```bash
./dirtybird-miner-cpu --test-dero
```

All 10 AstroBWTv3 tests should pass (1 intentional failure test).

## License

MIT License - See [LICENSE](LICENSE) for details.

## Credits

- Based on [TNN-miner](https://github.com/Tritonn204/tnn-miner) by Tritonn204
- AstroBWTv3 algorithm by DERO Project
- Wolf branch optimizations by @Wolf9466
