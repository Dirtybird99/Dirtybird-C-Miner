#!/bin/bash

# DERO Miner Release Packaging Script
# Creates distributable packages for Linux

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

VERSION="${1:-$(date +%Y.%m.%d)}"
TARGET="${2:-cpu}"

echo "================================================"
echo "DERO Miner Release Packaging"
echo "Version: $VERSION"
echo "Target: $TARGET"
echo "================================================"

# Determine build directory and output name based on target
case "$TARGET" in
    cpu)
        BUILD_DIR="$REPO_ROOT/build"
        BINARY_NAME="dero-miner-cpu"
        RELEASE_NAME="dero-miner-cpu-linux64"
        ;;
    rocm)
        BUILD_DIR="$REPO_ROOT/hip-build/linux/amd"
        BINARY_NAME="dero-miner-rocm"
        RELEASE_NAME="dero-miner-rocm-linux64"
        ;;
    cuda)
        BUILD_DIR="$REPO_ROOT/hip-build/linux/nvidia"
        BINARY_NAME="dero-miner-cuda"
        RELEASE_NAME="dero-miner-cuda-linux64"
        ;;
    *)
        echo "Unknown target: $TARGET. Use 'cpu', 'rocm', or 'cuda'."
        exit 1
        ;;
esac

BIN_DIR="$BUILD_DIR/bin"
BINARY_PATH="$BIN_DIR/$BINARY_NAME"

# Create releases directory
RELEASE_DIR="$REPO_ROOT/releases"
VERSION_DIR="$RELEASE_DIR/v$VERSION"
PACKAGE_DIR="$VERSION_DIR/$RELEASE_NAME"

mkdir -p "$PACKAGE_DIR"

echo ""
echo "Checking for binary at: $BINARY_PATH"

if [ ! -f "$BINARY_PATH" ]; then
    echo "Binary not found. Building first..."

    if [ "$TARGET" = "cpu" ]; then
        "$SCRIPT_DIR/build.sh" "$VERSION"
    else
        "$SCRIPT_DIR/build_all.sh" "$VERSION" "$TARGET"
    fi

    if [ ! -f "$BINARY_PATH" ]; then
        echo "Build failed - binary still not found."
        exit 1
    fi
fi

echo "Binary found: $BINARY_PATH"

# Copy binary
echo "Copying binary..."
cp "$BINARY_PATH" "$PACKAGE_DIR/"
chmod +x "$PACKAGE_DIR/$BINARY_NAME"

# Copy config template
if [ -f "$REPO_ROOT/config.json" ]; then
    echo "Copying config template..."
    cp "$REPO_ROOT/config.json" "$PACKAGE_DIR/"
fi

# Create README
cat > "$PACKAGE_DIR/README.txt" << EOF
DERO Miner v$VERSION
====================

High-performance DERO miner using AstroBWTv3 algorithm.

Quick Start:
1. Edit config.json and set your wallet address
2. Run ./$BINARY_NAME

Configuration Options:
- daemon-address: DERO node address (default: node.derofoundation.org:443)
- wallet: Your DERO wallet address
- threads: Number of CPU threads (-1 = auto-detect)
- period: Hash rate reporting interval in seconds

Command Line Options:
  --daemon-address <address>   DERO node address
  --wallet <address>           Wallet address
  --threads <n>                Number of threads
  --help                       Show all options

Example:
  ./$BINARY_NAME --daemon-address node.derofoundation.org:443 --wallet dero1...

For pool mining, use stratum address format:
  ./$BINARY_NAME --daemon-address stratum+tcp://pool.address:port --wallet dero1...

Build Information:
- Target: $TARGET
- Platform: Linux x64
EOF

# Create start script
cat > "$PACKAGE_DIR/start.sh" << EOF
#!/bin/bash
cd "\$(dirname "\$0")"
./$BINARY_NAME "\$@"
EOF
chmod +x "$PACKAGE_DIR/start.sh"

# Create tar.gz archive
TAR_PATH="$VERSION_DIR/$RELEASE_NAME-v$VERSION.tar.gz"
echo ""
echo "Creating archive: $TAR_PATH"

cd "$VERSION_DIR"
tar -czvf "$TAR_PATH" "$RELEASE_NAME"

echo ""
echo "================================================"
echo "Release package created successfully!"
echo "================================================"
echo "Directory: $PACKAGE_DIR"
echo "Archive: $TAR_PATH"
echo ""

# List package contents
echo "Package contents:"
ls -la "$PACKAGE_DIR"
