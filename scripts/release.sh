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
LIB_DIR="$PACKAGE_DIR/lib"

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

if ! command -v ldd >/dev/null 2>&1; then
    echo "ldd is required for Linux packaging." >&2
    exit 1
fi

if ! command -v patchelf >/dev/null 2>&1; then
    echo "patchelf is required for Linux packaging." >&2
    exit 1
fi

if ! command -v readelf >/dev/null 2>&1; then
    echo "readelf is required for Linux packaging." >&2
    exit 1
fi

should_bundle_runtime() {
    local soname="$1"
    case "$soname" in
        linux-vdso.so.1|linux-gate.so.1|ld-linux*.so.*|libc.so.*|libm.so.*|libpthread.so.*|librt.so.*|libdl.so.*|libutil.so.*|libresolv.so.*|libnsl.so.*|libanl.so.*)
            return 1
            ;;
        *)
            return 0
            ;;
    esac
}

bundle_runtime_libs() {
    local binary_path="$1"
    local output_dir="$2"
    local line soname resolved_path actual_path
    local -a missing_libs=()

    while IFS= read -r line; do
        if [[ "$line" =~ ^[[:space:]]*([^[:space:]]+)[[:space:]]+\=\>[[:space:]]+not[[:space:]]+found$ ]]; then
            soname="${BASH_REMATCH[1]}"
            if should_bundle_runtime "$soname"; then
                missing_libs+=("$soname")
            fi
            continue
        fi

        if [[ "$line" =~ ^[[:space:]]*([^[:space:]]+)[[:space:]]+\=\>[[:space:]]+([^[:space:]]+)[[:space:]]+\(0x[0-9a-fA-F]+\)$ ]]; then
            soname="${BASH_REMATCH[1]}"
            actual_path="${BASH_REMATCH[2]}"
            if ! should_bundle_runtime "$soname"; then
                continue
            fi
            resolved_path="$(readlink -f "$actual_path")"
            install -m 0644 "$resolved_path" "$output_dir/$soname"
        fi
    done < <(ldd "$binary_path")

    if [[ ${#missing_libs[@]} -gt 0 ]]; then
        printf 'Missing runtime libraries:%s\n' " ${missing_libs[*]}" >&2
        exit 1
    fi
}

validate_runtime_bundle() {
    local binary_path="$1"
    local output_dir="$2"
    local soname actual_rpath

    while IFS= read -r soname; do
        if should_bundle_runtime "$soname" && [[ ! -f "$output_dir/$soname" ]]; then
            echo "Bundled dependency missing from package: $soname" >&2
            exit 1
        fi
    done < <(readelf -d "$binary_path" | sed -n 's/^.*Shared library: \[\(.*\)\]$/\1/p')

    actual_rpath="$(patchelf --print-rpath "$binary_path")"
    if [[ "$actual_rpath" != '$ORIGIN/lib' ]]; then
        echo "Unexpected runtime search path: $actual_rpath" >&2
        exit 1
    fi
}

mkdir -p "$STAGE_ROOT"
rm -rf "$PACKAGE_DIR"
mkdir -p "$PACKAGE_DIR"
mkdir -p "$LIB_DIR"

cp "$BINARY_PATH" "$PACKAGE_DIR/"
chmod +x "$PACKAGE_DIR/$BINARY_NAME"
bundle_runtime_libs "$PACKAGE_DIR/$BINARY_NAME" "$LIB_DIR"
patchelf --set-rpath '$ORIGIN/lib' "$PACKAGE_DIR/$BINARY_NAME"
validate_runtime_bundle "$PACKAGE_DIR/$BINARY_NAME" "$LIB_DIR"

cp "$REPO_ROOT/README.md" "$PACKAGE_DIR/"
cp "$REPO_ROOT/LICENSE" "$PACKAGE_DIR/"
cp "$REPO_ROOT/config.json.example" "$PACKAGE_DIR/"
cp "$REPO_ROOT/config.json.example" "$PACKAGE_DIR/config.json"

cat > "$PACKAGE_DIR/start.sh" <<'EOF'
#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export LD_LIBRARY_PATH="$SCRIPT_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$SCRIPT_DIR"
exec ./dirtybird-miner-cpu "$@"
EOF
chmod +x "$PACKAGE_DIR/start.sh"

cat > "$PACKAGE_DIR/QUICKSTART.txt" <<EOF
DIRTYBIRD Miner $ASSET_VERSION
==============================

Contents:
- dirtybird-miner-cpu
- lib/
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
- Non-glibc runtime libraries are bundled under ./lib for portability.
EOF

rm -f "$ARCHIVE_PATH"
tar -czf "$ARCHIVE_PATH" -C "$STAGE_ROOT" "$PACKAGE_NAME"

echo "Created package: $ARCHIVE_PATH"
