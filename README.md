# DIRTYBIRD Miner

High-performance AstroBWTv3 CPU miner for DERO. Clean-room C++ (no Boost; pthreads + sockets),
descriptor-SA pipeline with optional profile-guided optimization (PGO).

**Status:** v1.0.25 · x86-64 (AVX2) and aarch64 (ARMv8).

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

## Android / Termux (mobile)

A download-only setup script handles everything: it fetches the pre-built aarch64
Android release, writes `config.json`, acquires a wake-lock (so Android Doze doesn't
pause the miner), and runs with auto-restart.

```bash
curl -sL https://raw.githubusercontent.com/Dirtybird99/Dirtybird-C-Miner/master/scripts/termux-setup.sh | bash
```

Or run it directly from a checkout:

```bash
bash scripts/termux-setup.sh              # install + run
bash scripts/termux-setup.sh --update     # upgrade to latest release
bash scripts/termux-setup.sh --reconfigure  # re-prompt for pool/wallet/threads
bash scripts/termux-setup.sh --uninstall  # remove installed files
```

Requirements: Termux (aarch64 / 64-bit ARM only). Install `termux-api` in Termux for
wake-lock support. Use `-p max` only when charging (headless) — `-p normal` is
thermal-safe for mobile.

## License

MIT - see [LICENSE](LICENSE).
