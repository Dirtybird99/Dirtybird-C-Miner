# DIRTYBIRD Miner

DIRTYBIRD Miner is a high-performance DERO CPU miner focused on AstroBWTv3 throughput, practical solo and pool usability, and straightforward release packaging.

The repository includes the required `libsais` source under `extern/libsais`, so standalone source builds do not depend on a sibling checkout.

## Prebuilt Releases

GitHub Releases publish ready-to-run CPU builds with TNN-style asset names:

- `dirtybird-miner-win64-vX.Y.Z.zip`
- `dirtybird-miner-amd64-vX.Y.Z.tar.gz`

Each package includes the miner binary, `config.json`, `config.json.example`, `README.md`, and `LICENSE`.

## Requirements

- 64-bit x86 CPU with AVX2 support
- Windows x64 or Linux amd64
- A DERO wallet address
- A DERO daemon or pool address

## Quick Start

1. Download the latest release asset for your platform.
2. Edit `config.json` and set `daemon-address`, `wallet`, and `threads`.
3. Start mining.

Windows:

```powershell
.\dirtybird-miner-cpu.exe --daemon-address pool.example.com:10100 --wallet YOUR_DERO_WALLET_ADDRESS --threads 20
```

Linux:

```bash
./dirtybird-miner-cpu --daemon-address pool.example.com:10100 --wallet YOUR_DERO_WALLET_ADDRESS --threads 20
```

Long options use double dashes, for example `--daemon-address`, `--wallet`, and `--threads`.

## Self-Test

Run the built-in DERO verification suite before mining:

```bash
./dirtybird-miner-cpu --test-dero
```

## Build From Source

### Windows (MSYS2 / MinGW64)

```bash
pacman -S --needed \
  mingw-w64-x86_64-clang \
  mingw-w64-x86_64-cmake \
  mingw-w64-x86_64-ninja \
  mingw-w64-x86_64-boost \
  mingw-w64-x86_64-openssl \
  mingw-w64-x86_64-openmp \
  mingw-w64-x86_64-mimalloc

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DDISABLE_LTO=ON
cmake --build build --parallel
```

### Linux

```bash
sudo apt-get update
sudo apt-get install -y \
  clang cmake ninja-build lld \
  libboost-all-dev libssl-dev libomp-dev \
  libnuma-dev libudns-dev libmimalloc-dev

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DDISABLE_LTO=ON
cmake --build build --parallel
```

The CPU binary is written to `build/bin/dirtybird-miner-cpu` on Linux and `build/bin/dirtybird-miner-cpu.exe` on Windows.

## Release Automation

The repository ships with GitHub Actions workflows for:

- CI builds and `--test-dero` validation on Windows and Linux
- Tagged releases with packaged binaries and checksums
- Offline 10-second replay benchmarks via `bench-dero-replay`
- Gitleaks and CodeQL security scanning

## License

MIT License. See [LICENSE](LICENSE).

## Credits

- Based on [TNN-miner](https://gitlab.com/Tritonn204/tnn-miner)
- AstroBWTv3 algorithm by the DERO Project
- Wolf branch optimizations by @Wolf9466
