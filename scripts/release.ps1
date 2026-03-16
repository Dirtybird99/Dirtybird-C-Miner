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
if ($Version.StartsWith("v")) {
    $Version = $Version.Substring(1)
}
if (-not $Version) {
    throw "Version cannot be empty."
}

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
Write-Host "Build dir: $BuildRoot"
Write-Host "Output dir: $StageRoot"
Write-Host "================================================"

if (-not (Test-Path $BinaryPath)) {
    throw "Binary not found: $BinaryPath"
}

New-Item -ItemType Directory -Path $StageRoot -Force | Out-Null
Remove-Item $PackageDir -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Path $PackageDir -Force | Out-Null

Copy-Item $BinaryPath -Destination $PackageDir -Force
Copy-Item (Join-Path $RepoRoot "README.md") -Destination $PackageDir -Force
Copy-Item (Join-Path $RepoRoot "LICENSE") -Destination $PackageDir -Force
Copy-Item (Join-Path $RepoRoot "config.json.example") -Destination $PackageDir -Force
Copy-Item (Join-Path $RepoRoot "config.json.example") -Destination (Join-Path $PackageDir "config.json") -Force

$RuntimeFiles = @(
    "WinRing0x64.dll",
    "WinRing0x64.sys",
    "libomp.dll",
    "libgcc_s_seh-1.dll",
    "libwinpthread-1.dll",
    "libstdc++-6.dll",
    "libcrypto-3-x64.dll",
    "libssl-3-x64.dll"
)

foreach ($runtimeFile in $RuntimeFiles) {
    $runtimePath = Join-Path $BinDir $runtimeFile
    if (Test-Path $runtimePath) {
        Copy-Item $runtimePath -Destination $PackageDir -Force
    }
}

$StartScript = @"
@echo off
setlocal
cd /d "%~dp0"
dirtybird-miner-cpu.exe %*
"@
$StartScript | Out-File -FilePath (Join-Path $PackageDir "start.bat") -Encoding ascii

$QuickStart = @"
DIRTYBIRD Miner $AssetVersion
=============================

Contents:
- dirtybird-miner-cpu.exe
- config.json
- config.json.example
- README.md
- LICENSE

Quick start:
1. Edit config.json and set your daemon-address, wallet, and threads.
2. Double-click start.bat or run dirtybird-miner-cpu.exe manually.
3. Long options use double dashes, for example:
   dirtybird-miner-cpu.exe --daemon-address pool.example.com:10100 --wallet YOUR_DERO_WALLET_ADDRESS --threads 20

Self-test:
  dirtybird-miner-cpu.exe --test-dero

Notes:
- This build targets 64-bit AVX2-capable CPUs.
- Keep the EXE and any DLL files in the same folder.
"@
$QuickStart | Out-File -FilePath (Join-Path $PackageDir "QUICKSTART.txt") -Encoding ascii

if (Test-Path $ArchivePath) {
    Remove-Item $ArchivePath -Force
}
Compress-Archive -Path $PackageDir -DestinationPath $ArchivePath -Force

Write-Host "Created package: $ArchivePath"
