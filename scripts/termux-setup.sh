#!/usr/bin/env bash
#
# DIRTYBIRD Miner -- Termux (Android) setup & launcher.
#
# Download-only installer: fetches the pre-built aarch64 Android release from
# GitHub, writes config.json, acquires a wake-lock so Android Doze doesn't kill
# the miner, and runs it with auto-restart.
#
# Usage:
#   bash scripts/termux-setup.sh             # install (if needed) + run
#   bash scripts/termux-setup.sh --update     # force re-download latest release
#   bash scripts/termux-setup.sh --reconfigure  # re-prompt for pool/wallet/threads
#   bash scripts/termux-setup.sh --uninstall  # remove installed files
#   bash scripts/termux-setup.sh --help
#
set -euo pipefail

REPO="Dirtybird99/dirtybird-miner"
DEFAULT_POOL="community-pools.mysrv.cloud:10300"
DEFAULT_WALLET="dero1qyvuemd6z0uzsx5ufc99f0jhyzvvpysmrd2t3526ht7a9dfh7jve2qqt0vu5y"
INSTALL_DIR="$HOME/dirtybird-miner"
BINARY_NAME="dirtybird-miner-cpu"
VERSION_FILE=".installed_version"
ARCHIVE_PREFIX="dirtybird-miner-v"
ARCHIVE_SUFFIX="_aarch64_android.tar.gz"

# ── colours (safe for Termux) ──────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'

info()  { printf "${GREEN}[*]${NC} %s\n" "$*"; }
warn()  { printf "${YELLOW}[!]${NC} %s\n" "$*"; }
err()   { printf "${RED}[x]${NC} %s\n" "$*" >&2; }
note()  { printf "${CYAN}[i]${NC} %s\n" "$*"; }

# ── flags ─────────────────────────────────────────────────────────────────────
FORCE_UPDATE=false
RECONFIGURE=false
UNINSTALL=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --update)        FORCE_UPDATE=true; shift ;;
        --reconfigure)   RECONFIGURE=true; shift ;;
        --uninstall)     UNINSTALL=true; shift ;;
        -h|--help)
            exit 0
            ;;
        *) err "Unknown option: $1"; exit 2 ;;
    esac
done

# ── detect platform ───────────────────────────────────────────────────────────
IS_ANDROID=false
ARCH="$(uname -m)"
if [ "$(uname -o 2>/dev/null)" = "Android" ]; then
    IS_ANDROID=true
fi

if [ "$IS_ANDROID" = true ]; then
    if [ "$ARCH" != "aarch64" ]; then
        err "Android on $ARCH is not supported by this script."
        err "Only aarch64 (64-bit ARM) Android is supported."
        err "For other platforms, download the appropriate release manually:"
        err "  https://github.com/$REPO/releases"
        exit 1
    fi
else
    err "This script is for Android/Termux only."
    err "On Linux x86_64, use the amd64 release:"
    err "  https://github.com/$REPO/releases  (dirtybird-miner-amd64-v*.tar.gz)"
    exit 1
fi

# ── step 1: install deps ──────────────────────────────────────────────────────
info "Checking dependencies..."
need_install=()
for cmd in tar; do
    command -v "$cmd" &>/dev/null || need_install+=("$cmd")
done
# prefer curl, fall back to wget
if ! command -v curl &>/dev/null && ! command -v wget &>/dev/null; then
    need_install+=(curl)
fi
if ! command -v jq &>/dev/null; then
    need_install+=(jq)
fi

if [ "${#need_install[@]}" -gt 0 ]; then
    info "Installing: ${need_install[*]}"
    pkg update -y >/dev/null 2>&1 || true
    pkg install -y "${need_install[@]}" >/dev/null 2>&1 || {
        err "Failed to install: ${need_install[*]}"
        err "Run: pkg install -y ${need_install[*]}"
        exit 1
    }
fi
info "Dependencies OK."

# ── step 2: handle --uninstall ────────────────────────────────────────────────
if [ "$UNINSTALL" = true ]; then
    info "Removing $INSTALL_DIR ..."
    rm -rf "$INSTALL_DIR"
    info "Done. (Config and binaries removed.)"
    exit 0
fi

# ── step 3: get the binary ────────────────────────────────────────────────────
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

# HTTP fetcher for API calls (stdout): curl preferred, wget fallback
fetch() {
    if command -v curl &>/dev/null; then
        curl -fsSL "$1"
    else
        wget -qO- "$1"
    fi
}

LATEST_TAG=""
if [ "$FORCE_UPDATE" = false ] && [ -f "$VERSION_FILE" ]; then
    info "Already installed ($(cat "$VERSION_FILE")). Use --update to upgrade."
else
    info "Fetching latest release info..."
    LATEST_TAG="$(fetch "https://api.github.com/repos/$REPO/releases/latest" \
        | jq -r '.tag_name // empty' 2>/dev/null || true)"

    if [ -z "$LATEST_TAG" ]; then
        err "Could not determine latest release. Check network connection."
        exit 1
    fi
    info "Latest release: $LATEST_TAG"

    ARCHIVE="${ARCHIVE_PREFIX}${LATEST_TAG#v}_aarch64_android.tar.gz"
    DOWNLOAD_URL="https://github.com/$REPO/releases/download/${LATEST_TAG}/${ARCHIVE}"

    info "Downloading $ARCHIVE ..."
    if command -v curl &>/dev/null; then
        if ! curl -fsSL "$DOWNLOAD_URL" -o "$ARCHIVE"; then
            err "Download failed (curl)."
            exit 1
        fi
    elif command -v wget &>/dev/null; then
        if ! wget -q --show-progress -O "$ARCHIVE" "$DOWNLOAD_URL"; then
            err "Download failed (wget)."
            exit 1
        fi
    else
        err "Neither curl nor wget is available."
        exit 1
    fi

    info "Extracting..."
    tar xzf "$ARCHIVE"
    rm -f "$ARCHIVE"

    # move binary to install dir if it's nested in a subdirectory
    if [ ! -f "./$BINARY_NAME" ]; then
        NESTED="$(find . -maxdepth 2 -name "$BINARY_NAME" -type f | head -1)"
        if [ -n "$NESTED" ]; then
            mv "$NESTED" "./$BINARY_NAME"
            rm -rf "$(dirname "$NESTED")" 2>/dev/null || true
        else
            err "Extraction succeeded but $BINARY_NAME binary not found."
            exit 1
        fi
    fi
    chmod +x "./$BINARY_NAME"
    echo "$LATEST_TAG" > "$VERSION_FILE"
    info "Installed $LATEST_TAG."
fi

# ── step 4: prompt for daemon address ────────────────────────────────────────
if [ "$RECONFIGURE" = true ] || [ ! -f "config.json" ]; then
    printf "\n"
    printf "${CYAN}Daemon/pool address [scheme://]host:port${NC}\n"
    printf "  Press Enter to use: ${GREEN}%s${NC}\n" "$DEFAULT_POOL"
    read -rp "  Address: " INPUT_POOL </dev/tty
    POOL="${INPUT_POOL:-$DEFAULT_POOL}"

    # ── step 5: prompt for wallet address ────────────────────────────────────
    printf "\n"
    printf "${CYAN}DERO wallet address${NC}\n"
    printf "  Press Enter to use: ${GREEN}%s${NC}\n" "$DEFAULT_WALLET"
    read -rp "  Wallet: " INPUT_WALLET </dev/tty
    WALLET="${INPUT_WALLET:-$DEFAULT_WALLET}"

    # validate wallet format
    if ! printf '%s' "$WALLET" | grep -qE '^(dero1|deto1)[a-z0-9]+$'; then
        err "Invalid wallet address: $WALLET"
        err "Must start with 'dero1' or 'deto1' followed by lowercase alphanumerics."
        exit 1
    fi

    # ── step 6: detect threads (nproc - 1, minimum 1) ───────────────────────
    CORES="$(nproc 2>/dev/null || echo 4)"
    THREADS=$((CORES - 1))
    [ "$THREADS" -lt 1 ] && THREADS=1

    # ── step 7: write config.json ────────────────────────────────────────────
    cat > config.json <<EOF
{
  "daemon-address": "$POOL",
  "wallet": "$WALLET",
  "threads": $THREADS,
  "priority": "normal"
}
EOF
    info "Config written to $INSTALL_DIR/config.json"
else
    info "Using existing config.json (use --reconfigure to change)."
fi

# ── step 8: battery / thermal advisory ────────────────────────────────────────
if command -v termux-battery-status &>/dev/null; then
    BAT_PCT="$(termux-battery-status 2>/dev/null | jq -r '.percentage // empty' 2>/dev/null || true)"
    BAT_PLUGGED="$(termux-battery-status 2>/dev/null | jq -r '.plugged // empty' 2>/dev/null || true)"
    if [ -n "$BAT_PCT" ] && [ "$BAT_PCT" -lt 40 ] 2>/dev/null; then
        warn "Battery is ${BAT_PCT}%. Mining drains battery fast; consider charging."
    fi
    if [ "$BAT_PLUGGED" != "PLUGGED_TYPE_AC" ] && [ "$BAT_PLUGGED" != "PLUGGED_TYPE_USB" ] 2>/dev/null; then
        warn "Device is not charging. Thermal throttling may reduce hashrate."
    fi
fi

# ── step 9: acquire wake-lock ────────────────────────────────────────────────
WAKE_LOCK=false
if command -v termux-wake-lock &>/dev/null; then
    termux-wake-lock 2>/dev/null && WAKE_LOCK=true || true
    if [ "$WAKE_LOCK" = true ]; then
        info "Wake-lock acquired (Android Doze will not suspend the miner)."
    fi
else
    note "Install termux-api + 'pkg install termux-api' for wake-lock support."
    note "Without it, Android Doze may pause the miner in background."
fi

# ── step 10: run with auto-restart ────────────────────────────────────────────
printf "\n"
printf "  Pool:     ${GREEN}%s${NC}\n" "$(jq -r '.["daemon-address"]' config.json)"
printf "  Wallet:   ${GREEN}%s${NC}\n" "$(jq -r '.wallet' config.json)"
printf "  Threads:  ${GREEN}%s${NC}\n" "$(jq -r '.threads' config.json)"
printf "  Priority: ${GREEN}%s${NC}\n" "$(jq -r '.priority' config.json)"
printf "\n"
info "Starting miner... (Ctrl-C to stop)"
printf "\n"

release_lock() {
    if [ "$WAKE_LOCK" = true ]; then
        termux-wake-unlock 2>/dev/null || true
        info "Wake-lock released."
    fi
}
trap release_lock EXIT INT TERM

BACKOFF=5
MAX_BACKOFF=30
while true; do
    set +e
    ./"$BINARY_NAME"
    EXIT_CODE=$?
    set -e
    if [ "$EXIT_CODE" -eq 0 ]; then
        info "Miner exited cleanly."
        break
    fi
    warn "Miner exited with code $EXIT_CODE. Restarting in ${BACKOFF}s..."
    sleep "$BACKOFF"
    BACKOFF=$((BACKOFF * 2))
    [ "$BACKOFF" -gt "$MAX_BACKOFF" ] && BACKOFF="$MAX_BACKOFF"
done
