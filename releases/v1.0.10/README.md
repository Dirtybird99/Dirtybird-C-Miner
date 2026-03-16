# DIRTYBIRD Miner v1.0.10

Release artifacts staged in-repo, DeroLuna-style.

Files:
- `dirtybird-miner-v1.0.10_windows.zip`: Windows x64 package built from the current tree.
- `dirtybird-miner-v1.0.10_linux_hiveos_mmpos.tar.gz`: Linux x86_64 package staged with DeroLuna-style naming.
- `dirtybird-miner-v1.0.10_aarch64_android.tar.gz`: Android arm64 package built locally, with `libc++_shared.so` included.
- `SHA256SUMS.txt`: SHA-256 checksums for all staged archives.

Notes:
- This package targets 64-bit AVX2-capable CPUs.
- The Windows archive already includes the runtime files copied by the packaging script.
- The Android archive is a native `aarch64-linux-android` build from the current tree.
