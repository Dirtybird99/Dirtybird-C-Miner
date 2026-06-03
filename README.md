# DIRTYBIRD Miner

High-performance AstroBWTv3 CPU miner for DERO. Clean-room C++ (no Boost; pthreads + sockets),
descriptor-SA pipeline with optional profile-guided optimization (PGO).

**Status:** v1.0.17 · x86-64 with AVX2.

## Build

Toolchain: clang + lld, CMake, OpenSSL. On Windows use MSYS2 MINGW64.

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
cmake --build build --target dirtybird-miner-cpu
```

For the fastest binary, use the two-pass PGO flow documented in `build.sh` / `CMakeLists.txt`.

## Usage

```
dirtybird-miner-cpu -d <host:port> -w <wallet> -t <threads> [-p normal|max]
```

| flag | meaning |
|------|---------|
| `-d` | daemon address `host:port` |
| `-w` | DERO wallet address |
| `-t` | mining threads |
| `-p` | priority profile: `normal` (default, desktop-safe) or `max` (headless) |
| `-h`, `--help` / `-v`, `--version` | help / version |

## Correctness

At startup the miner computes `pow("a")` and verifies it equals
`54e2324ddacc3f0383501a9e5760f85d63e9bc6705e9124ca7aef89016ab81ea`.

## Performance

~20 KH/s sustained at 20 threads on an i7-13700HX. Measure over >=10 minutes for a
representative sustained figure.

## License

MIT - see [LICENSE](LICENSE).
