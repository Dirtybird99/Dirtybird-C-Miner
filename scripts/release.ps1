param(
    [string]$Version = "",
    [string]$Target = "cpu"
)

# DERO Miner Release Packaging Script
# Creates distributable packages for Windows

$ScriptRoot = $PSScriptRoot
$RepoRoot = (Resolve-Path (Join-Path $ScriptRoot "..")).Path

# Auto-generate version if not provided
if (-not $Version) {
    $Version = Get-Date -Format "yyyy.MM.dd"
}

Write-Host "================================================"
Write-Host "DERO Miner Release Packaging"
Write-Host "Version: $Version"
Write-Host "Target: $Target"
Write-Host "================================================"

# Determine build directory and output name based on target
switch ($Target.ToLower()) {
    "cpu" {
        $BuildDir = Join-Path $RepoRoot "build"
        $BinaryName = "dero-miner-cpu.exe"
        $ReleaseName = "dero-miner-cpu-win64"
    }
    "rocm" {
        $BuildDir = Join-Path $RepoRoot "hip-build\win32\amd"
        $BinaryName = "dero-miner-rocm.exe"
        $ReleaseName = "dero-miner-rocm-win64"
    }
    "cuda" {
        $BuildDir = Join-Path $RepoRoot "hip-build\win32\nvidia"
        $BinaryName = "dero-miner-cuda.exe"
        $ReleaseName = "dero-miner-cuda-win64"
    }
    default {
        Write-Host "Unknown target: $Target. Use 'cpu', 'rocm', or 'cuda'."
        exit 1
    }
}

$BinDir = Join-Path $BuildDir "bin"
$BinaryPath = Join-Path $BinDir $BinaryName

# Create releases directory
$ReleaseDir = Join-Path $RepoRoot "releases"
$VersionDir = Join-Path $ReleaseDir "v$Version"
$PackageDir = Join-Path $VersionDir $ReleaseName

if (-not (Test-Path $PackageDir)) {
    New-Item -ItemType Directory -Path $PackageDir -Force | Out-Null
}

Write-Host ""
Write-Host "Checking for binary at: $BinaryPath"

if (-not (Test-Path $BinaryPath)) {
    Write-Host "Binary not found. Building first..."

    # Run the appropriate build
    if ($Target -eq "cpu") {
        Push-Location $RepoRoot
        & "$ScriptRoot\build.ps1" -version $Version
        Pop-Location
    } else {
        & "$ScriptRoot\build_all.ps1" -TnnVersion $Version -Target $Target
    }

    if (-not (Test-Path $BinaryPath)) {
        Write-Host "Build failed - binary still not found."
        exit 1
    }
}

Write-Host "Binary found: $BinaryPath"

# Copy binary
Write-Host "Copying binary..."
Copy-Item $BinaryPath -Destination $PackageDir -Force

# Copy config template
$ConfigTemplate = Join-Path $RepoRoot "config.json"
if (Test-Path $ConfigTemplate) {
    Write-Host "Copying config template..."
    Copy-Item $ConfigTemplate -Destination $PackageDir -Force
}

# Copy WinRing0 driver files (needed for CPU optimization features)
$WinRing0Files = @(
    "WinRing0x64.dll",
    "WinRing0x64.sys"
)

foreach ($file in $WinRing0Files) {
    $srcFile = Join-Path $BinDir $file
    if (Test-Path $srcFile) {
        Write-Host "Copying $file..."
        Copy-Item $srcFile -Destination $PackageDir -Force
    }
}

# Copy HIP runtime DLLs for GPU builds
if ($Target -eq "cuda") {
    $NvrtcDir = Join-Path $RepoRoot "lib\nvrtc"
    if (Test-Path $NvrtcDir) {
        Write-Host "Copying NVRTC runtime files..."
        Get-ChildItem $NvrtcDir -Filter "*.dll" | ForEach-Object {
            Copy-Item $_.FullName -Destination $PackageDir -Force
        }
    }
}

if ($Target -eq "rocm") {
    $HiprtcDir = Join-Path $RepoRoot "lib\hiprtc"
    if (Test-Path $HiprtcDir) {
        Write-Host "Copying HIP runtime files..."
        Get-ChildItem $HiprtcDir -Filter "*.dll" | ForEach-Object {
            Copy-Item $_.FullName -Destination $PackageDir -Force
        }
    }
}

# Create README
$ReadmeContent = @"
DERO Miner v$Version
====================

High-performance DERO miner using AstroBWTv3 algorithm.

Quick Start:
1. Edit config.json and set your wallet address
2. Run $BinaryName

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
  $BinaryName --daemon-address node.derofoundation.org:443 --wallet dero1...

For pool mining, use stratum address format:
  $BinaryName --daemon-address stratum+tcp://pool.address:port --wallet dero1...

Build Information:
- Target: $Target
- Platform: Windows x64
"@

$ReadmePath = Join-Path $PackageDir "README.txt"
$ReadmeContent | Out-File -FilePath $ReadmePath -Encoding UTF8

# Create zip archive
$ZipPath = Join-Path $VersionDir "$ReleaseName-v$Version.zip"
Write-Host ""
Write-Host "Creating archive: $ZipPath"

if (Test-Path $ZipPath) {
    Remove-Item $ZipPath -Force
}

Compress-Archive -Path "$PackageDir\*" -DestinationPath $ZipPath -Force

Write-Host ""
Write-Host "================================================"
Write-Host "Release package created successfully!"
Write-Host "================================================"
Write-Host "Directory: $PackageDir"
Write-Host "Archive: $ZipPath"
Write-Host ""

# List package contents
Write-Host "Package contents:"
Get-ChildItem $PackageDir | ForEach-Object {
    Write-Host "  - $($_.Name)"
}
