param(
    [Parameter(Mandatory = $true)]
    [string]$Version,
    [string]$BuildDir = "build",
    [string]$OutputDir = "dist"
)

$ErrorActionPreference = "Stop"

$ScriptRoot = $PSScriptRoot
$RepoRoot = (Resolve-Path (Join-Path $ScriptRoot "..")).Path

$Version = $Version.Trim()
if ($Version.StartsWith("v")) { $Version = $Version.Substring(1) }
if (-not $Version) { throw "Version cannot be empty." }

$AssetVersion = "v$Version"
$BinaryName = "dirtybird-miner-cpu.exe"
$BuildRoot = Join-Path $RepoRoot $BuildDir
$BinDir = Join-Path $BuildRoot "bin"
$BinaryPath = Join-Path $BinDir $BinaryName
$StageRoot = Join-Path $RepoRoot $OutputDir
$PackageName = "dirtybird-miner-win64-$AssetVersion"
$PackageDir = Join-Path $StageRoot $PackageName
$ArchivePath = Join-Path $StageRoot "$PackageName.zip"

Write-Host "================================================"
Write-Host "DIRTYBIRD Miner Windows Packaging"
Write-Host "Version: $AssetVersion"
Write-Host "================================================"

if (-not (Test-Path $BinaryPath)) { throw "Binary not found: $BinaryPath" }

New-Item -ItemType Directory -Path $StageRoot -Force | Out-Null
Remove-Item $PackageDir -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Path $PackageDir -Force | Out-Null

Copy-Item $BinaryPath -Destination $PackageDir -Force
Copy-Item (Join-Path $RepoRoot "README.md") -Destination $PackageDir -Force
Copy-Item (Join-Path $RepoRoot "LICENSE") -Destination $PackageDir -Force

# Bundle the MinGW/OpenSSL runtime DLLs the dynamic build needs (skip any not present).
$RuntimeFiles = @(
    "libgcc_s_seh-1.dll",
    "libwinpthread-1.dll",
    "libstdc++-6.dll",
    "libcrypto-3-x64.dll",
    "libssl-3-x64.dll"
)
foreach ($runtimeFile in $RuntimeFiles) {
    $runtimePath = Join-Path $BinDir $runtimeFile
    if (Test-Path $runtimePath) { Copy-Item $runtimePath -Destination $PackageDir -Force }
}

# Auto-restart launcher (edit the daemon/wallet, then double-click).
$StartScript = @"
@echo off
cd /d "%~dp0"
title DIRTYBIRD Miner $AssetVersion
:loop
.\dirtybird-miner-cpu.exe -d YOUR_NODE_IP:10100 -w YOUR_DERO_WALLET -t 20
timeout 3
goto loop
"@
$StartScript | Out-File -FilePath (Join-Path $PackageDir "start.bat") -Encoding ascii

$QuickStart = @"
DIRTYBIRD Miner $AssetVersion
=============================

Contents:
- dirtybird-miner-cpu.exe
- *.dll  (runtime; keep them next to the exe)
- start.bat
- README.md
- LICENSE

Quick start:
1. Edit start.bat: set -d <daemon host:port>, -w <your DERO wallet>, -t <threads>.
2. Double-click start.bat (it auto-restarts the miner), or run directly:
   dirtybird-miner-cpu.exe -d host:port -w YOUR_DERO_WALLET -t 20
3. -p max for headless/AFK (more hashrate, may stutter the desktop); -p normal (default) is desktop-safe.

Notes:
- 64-bit AVX2 CPUs. Keep the EXE and DLLs in the same folder.
- Startup prints a pow("a") self-test; it must say PASS.
"@
$QuickStart | Out-File -FilePath (Join-Path $PackageDir "QUICKSTART.txt") -Encoding ascii

if (Test-Path $ArchivePath) { Remove-Item $ArchivePath -Force }
Compress-Archive -Path $PackageDir -DestinationPath $ArchivePath -Force

Write-Host "Created package: $ArchivePath"
