# DIRTYBIRD Miner

High-performance AstroBWTv3 CPU miner for DERO. Clean-room C++ (no Boost; pthreads + sockets),
descriptor-SA pipeline with optional profile-guided optimization (PGO).

**Status:** v1.0.24 · x86-64 (AVX2) and aarch64 (ARMv8).

## Build

Toolchain: clang + lld, CMake, OpenSSL. On Windows use MSYS2 MINGW64.

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
cmake --build build --target dirtybird-miner-cpu
```

For the fastest binary, use the two-pass PGO flow documented in `build.sh` / `CMakeLists.txt`.

## Usage

Configure via **`config.json`** (next to the exe) **or** CLI flags — your choice. Precedence is
**built-in defaults → `config.json` → CLI flags** (CLI wins). Editing `config.json` is enough; the
bundled `start.bat`/`start.sh` runs the miner with no flags so it picks up `config.json`.

```jsonc
// config.json
{
  "daemon-address": "host:port",
  "wallet": "dero1...",
  "threads": -1,          // -1 = auto (all hardware threads)
  "priority": "normal"    // normal (desktop-safe) | max (headless)
}
```

```
dirtybird-miner-cpu [-d <host:port>] [-w <wallet>] [-t <threads>] [-p normal|max]
```

| flag | meaning |
|------|---------|
| `-d` | daemon address `host:port` (overrides `daemon-address`) |
| `-w` | DERO wallet address (overrides `wallet`) |
| `-t` | mining threads (overrides `threads`) |
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
