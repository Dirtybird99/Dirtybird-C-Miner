#!/bin/bash

set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
    echo "Usage: $0 <version> [build-dir] [output-dir]" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VERSION="${1#v}"
BUILD_DIR="${2:-build}"
OUTPUT_DIR="${3:-dist}"

if [[ -z "$VERSION" ]]; then
    echo "Version cannot be empty." >&2
    exit 1
fi

ASSET_VERSION="v$VERSION"
BINARY_NAME="dirtybird-miner-cpu"
BUILD_ROOT="$REPO_ROOT/$BUILD_DIR"
BIN_DIR="$BUILD_ROOT/bin"
BINARY_PATH="$BIN_DIR/$BINARY_NAME"
STAGE_ROOT="$REPO_ROOT/$OUTPUT_DIR"
PACKAGE_NAME="dirtybird-miner-amd64-$ASSET_VERSION"
PACKAGE_DIR="$STAGE_ROOT/$PACKAGE_NAME"
ARCHIVE_PATH="$STAGE_ROOT/$PACKAGE_NAME.tar.gz"

echo "================================================"
echo "DIRTYBIRD Miner Linux Packaging"
echo "Version: $ASSET_VERSION"
echo "Build dir: $BUILD_ROOT"
echo "Output dir: $STAGE_ROOT"
echo "================================================"

if [[ ! -f "$BINARY_PATH" ]]; then
    echo "Binary not found: $BINARY_PATH" >&2
    exit 1
fi

mkdir -p "$STAGE_ROOT"
rm -rf "$PACKAGE_DIR"
mkdir -p "$PACKAGE_DIR"

cp "$BINARY_PATH" "$PACKAGE_DIR/"
chmod +x "$PACKAGE_DIR/$BINARY_NAME"
cp "$REPO_ROOT/README.md" "$PACKAGE_DIR/"
cp "$REPO_ROOT/LICENSE" "$PACKAGE_DIR/"
cp "$REPO_ROOT/config.json.example" "$PACKAGE_DIR/"
cp "$REPO_ROOT/config.json.example" "$PACKAGE_DIR/config.json"

cat > "$PACKAGE_DIR/start.sh" <<'EOF'
#!/bin/bash
set -e
cd "$(dirname "$0")"
./dirtybird-miner-cpu "$@"
EOF
chmod +x "$PACKAGE_DIR/start.sh"

cat > "$PACKAGE_DIR/QUICKSTART.txt" <<EOF
DIRTYBIRD Miner $ASSET_VERSION
==============================

Contents:
- dirtybird-miner-cpu
- config.json
- config.json.example
- README.md
- LICENSE

Quick start:
1. Edit config.json and set your daemon-address, wallet, and threads.
2. Run ./start.sh or launch the miner directly:
   ./dirtybird-miner-cpu --daemon-address pool.example.com:10100 --wallet YOUR_DERO_WALLET_ADDRESS --threads 20

Self-test:
  ./dirtybird-miner-cpu --test-dero

Notes:
- This build targets 64-bit AVX2-capable CPUs.
- Linux builds may still rely on standard system libraries provided by your distro.
EOF

rm -f "$ARCHIVE_PATH"
tar -czf "$ARCHIVE_PATH" -C "$STAGE_ROOT" "$PACKAGE_NAME"

echo "Created package: $ARCHIVE_PATH"
