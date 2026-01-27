# scripts/collect_sa_metrics.ps1
# Run DeroLuna with SA instrumentation enabled and capture metrics to CSV
#
# Usage:
#   .\collect_sa_metrics.ps1 [-Duration 60] [-OutputFile "sa_metrics.csv"] [-Threads 1]
#
# Prerequisites:
#   - MSYS2 MinGW64 with Clang installed (C:\msys64\mingw64\bin)
#   - CMake 3.18+ in PATH
#   - Ninja build system (optional, but recommended)

param(
    [int]$Duration = 60,                    # Mining duration in seconds
    [string]$OutputFile = "sa_metrics.csv", # Output CSV file
    [string]$DaemonAddress = "",            # Optional: daemon address (default: offline test mode)
    [int]$Threads = 1                       # Use 1 thread for cleaner metrics
)

$ErrorActionPreference = "Stop"

# Get script and project directories
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectDir = Split-Path -Parent $scriptDir
$buildDir = Join-Path $projectDir "build-instrumented"

Write-Host "=== DeroLuna SA Metrics Collection ===" -ForegroundColor Cyan
Write-Host "Project Directory: $projectDir"
Write-Host "Build Directory:   $buildDir"
Write-Host "Duration:          $Duration seconds"
Write-Host "Output File:       $OutputFile"
Write-Host "Threads:           $Threads"
Write-Host ""

# Check for CMake
$cmake = Get-Command cmake -ErrorAction SilentlyContinue
if (-not $cmake) {
    Write-Error "CMake not found in PATH. Please install CMake 3.18+ and add to PATH."
    exit 1
}

# Build with instrumentation enabled
Write-Host "Building with ENABLE_SA_INSTRUMENTATION=ON..." -ForegroundColor Yellow

Push-Location $projectDir
try {
    # Configure with instrumentation
    $configureArgs = @(
        "-B", $buildDir,
        "-DENABLE_SA_INSTRUMENTATION=ON",
        "-DWITH_ASTROBWTV3=ON",
        "-DUSE_ASTRO_SPSA=OFF",  # Disable SPSA to ensure divsufsort is always called
        "-DCMAKE_BUILD_TYPE=Release"
    )

    # Check for Ninja
    $ninja = Get-Command ninja -ErrorAction SilentlyContinue
    if ($ninja) {
        $configureArgs += @("-G", "Ninja")
        Write-Host "Using Ninja build system"
    }

    Write-Host "Running: cmake $($configureArgs -join ' ')"
    & cmake @configureArgs
    if ($LASTEXITCODE -ne 0) {
        throw "CMake configure failed with exit code $LASTEXITCODE"
    }

    # Build
    Write-Host "`nBuilding project..." -ForegroundColor Yellow
    & cmake --build $buildDir --config Release
    if ($LASTEXITCODE -ne 0) {
        throw "CMake build failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}

# Find the built executable
$exeName = "dirtybird-miner-cpu.exe"
$exePath = Join-Path $buildDir $exeName
if (-not (Test-Path $exePath)) {
    $exePath = Join-Path $buildDir "Release" $exeName
    if (-not (Test-Path $exePath)) {
        Write-Error "Could not find built executable: $exeName"
        exit 1
    }
}

Write-Host "`nExecutable found: $exePath" -ForegroundColor Green

# Create CSV header
$csvPath = Join-Path $projectDir $OutputFile
"Iteration,InputSize,SAReads,SAWrites,TextReads,BucketA,BucketB,Loops,Phase0_cycles,Phase1_cycles,Phase2_cycles,Total_cycles" | Out-File $csvPath -Encoding UTF8

# Raw output files
$rawOutputFile = Join-Path $projectDir "sa_raw_output.txt"
$rawStderrFile = Join-Path $projectDir "sa_raw_stderr.txt"

# Build command arguments
$minerArgs = @("--threads", $Threads)
if ($DaemonAddress) {
    $minerArgs += @("--daemon-address", $DaemonAddress)
} else {
    Write-Host "`nNo daemon address specified. Running in test/benchmark mode." -ForegroundColor Yellow
    Write-Host "For production metrics, use: -DaemonAddress 'localhost:10100'" -ForegroundColor Yellow
    # Add benchmark flag if available, or just run briefly
    # For now, we'll just run and capture what we can
}

Write-Host "`nRunning miner for $Duration seconds..." -ForegroundColor Yellow
Write-Host "Command: $exePath $($minerArgs -join ' ')"

# Start the process
$processInfo = New-Object System.Diagnostics.ProcessStartInfo
$processInfo.FileName = $exePath
$processInfo.Arguments = $minerArgs -join ' '
$processInfo.UseShellExecute = $false
$processInfo.RedirectStandardOutput = $true
$processInfo.RedirectStandardError = $true
$processInfo.WorkingDirectory = $projectDir
$processInfo.CreateNoWindow = $true

$process = New-Object System.Diagnostics.Process
$process.StartInfo = $processInfo

# Create StringBuilder for output capture
$outputBuilder = New-Object System.Text.StringBuilder
$errorBuilder = New-Object System.Text.StringBuilder

# Register event handlers for async output reading
$outputEvent = Register-ObjectEvent -InputObject $process -EventName OutputDataReceived -Action {
    if ($EventArgs.Data) {
        $null = $Event.MessageData.Append($EventArgs.Data + "`n")
    }
} -MessageData $outputBuilder

$errorEvent = Register-ObjectEvent -InputObject $process -EventName ErrorDataReceived -Action {
    if ($EventArgs.Data) {
        $null = $Event.MessageData.Append($EventArgs.Data + "`n")
    }
} -MessageData $errorBuilder

try {
    $process.Start() | Out-Null
    $process.BeginOutputReadLine()
    $process.BeginErrorReadLine()

    # Wait for duration
    $startTime = Get-Date
    while (((Get-Date) - $startTime).TotalSeconds -lt $Duration) {
        if ($process.HasExited) {
            Write-Host "Process exited early with code: $($process.ExitCode)" -ForegroundColor Yellow
            break
        }
        Start-Sleep -Milliseconds 500
    }

    # Stop the process if still running
    if (-not $process.HasExited) {
        Write-Host "`nStopping miner..." -ForegroundColor Yellow
        $process.Kill()
        $process.WaitForExit(5000)
    }
}
finally {
    # Cleanup event registrations
    Unregister-Event -SourceIdentifier $outputEvent.Name -ErrorAction SilentlyContinue
    Unregister-Event -SourceIdentifier $errorEvent.Name -ErrorAction SilentlyContinue
    Remove-Job -Name $outputEvent.Name -ErrorAction SilentlyContinue
    Remove-Job -Name $errorEvent.Name -ErrorAction SilentlyContinue
}

# Get the captured output
$rawOutput = $outputBuilder.ToString()
$rawError = $errorBuilder.ToString()

# Save raw output
$rawOutput | Out-File $rawOutputFile -Encoding UTF8
$rawError | Out-File $rawStderrFile -Encoding UTF8

Write-Host "`nProcessing SA metrics..." -ForegroundColor Yellow

# Parse SA_METRICS lines from output
$pattern = 'SA_METRICS:(\d+),(\d+),(\d+),(\d+),(\d+),(\d+),(\d+),(\d+),(\d+),(\d+),(\d+),(\d+)'
$lineCount = 0

foreach ($line in ($rawOutput -split "`n")) {
    if ($line -match $pattern) {
        $lineCount++
        $csvLine = $line -replace 'SA_METRICS:', ''
        $csvLine | Out-File $csvPath -Append -Encoding UTF8
    }
}

Write-Host "`nCollected $lineCount SA iterations to $csvPath" -ForegroundColor Green

# Generate quick stats if we have data
if ($lineCount -gt 0) {
    Write-Host "`n=== Quick Statistics ===" -ForegroundColor Cyan

    try {
        $data = Import-Csv $csvPath

        $avgSize = ($data | Measure-Object -Property InputSize -Average).Average
        $minSize = ($data | Measure-Object -Property InputSize -Minimum).Minimum
        $maxSize = ($data | Measure-Object -Property InputSize -Maximum).Maximum

        $avgReads = ($data | Measure-Object -Property SAReads -Average).Average
        $avgWrites = ($data | Measure-Object -Property SAWrites -Average).Average
        $avgText = ($data | Measure-Object -Property TextReads -Average).Average
        $avgLoops = ($data | Measure-Object -Property Loops -Average).Average

        $avgP0 = ($data | Measure-Object -Property Phase0_cycles -Average).Average
        $avgP1 = ($data | Measure-Object -Property Phase1_cycles -Average).Average
        $avgP2 = ($data | Measure-Object -Property Phase2_cycles -Average).Average
        $avgTotal = ($data | Measure-Object -Property Total_cycles -Average).Average

        Write-Host ""
        Write-Host "Input Size:"
        Write-Host "  Average:     $([math]::Round($avgSize)) bytes"
        Write-Host "  Range:       $minSize - $maxSize bytes"
        Write-Host ""
        Write-Host "Memory Operations (average):"
        Write-Host "  SA Reads:    $([math]::Round($avgReads))"
        Write-Host "  SA Writes:   $([math]::Round($avgWrites))"
        Write-Host "  Text Reads:  $([math]::Round($avgText))"
        Write-Host "  Loop Iters:  $([math]::Round($avgLoops))"
        if ($avgWrites -gt 0) {
            Write-Host "  Read/Write:  $([math]::Round($avgReads / $avgWrites, 2))"
        }
        Write-Host ""
        Write-Host "Phase Timing (average cycles):"
        Write-Host "  Phase 0 (sort_typeBstar): $([math]::Round($avgP0))"
        Write-Host "  Phase 1 (B-type pass):    $([math]::Round($avgP1))"
        Write-Host "  Phase 2 (L-type pass):    $([math]::Round($avgP2))"
        Write-Host "  Total:                    $([math]::Round($avgTotal))"
        if ($avgTotal -gt 0) {
            Write-Host ""
            Write-Host "Phase Distribution:"
            Write-Host "  Phase 0: $([math]::Round(100 * $avgP0 / $avgTotal, 1))%"
            Write-Host "  Phase 1: $([math]::Round(100 * $avgP1 / $avgTotal, 1))%"
            Write-Host "  Phase 2: $([math]::Round(100 * $avgP2 / $avgTotal, 1))%"
        }
    }
    catch {
        Write-Host "Warning: Could not compute statistics: $_" -ForegroundColor Yellow
    }
}
else {
    Write-Host ""
    Write-Host "No SA_METRICS lines found in output." -ForegroundColor Yellow
    Write-Host "This could mean:"
    Write-Host "  1. The build didn't include SA instrumentation"
    Write-Host "  2. The miner didn't compute any hashes (no work from daemon)"
    Write-Host "  3. USE_ASTRO_SPSA was enabled (divsufsort bypassed)"
    Write-Host ""
    Write-Host "Check the raw output files for debugging:"
    Write-Host "  - $rawOutputFile"
    Write-Host "  - $rawStderrFile"
}

Write-Host "`nDone!" -ForegroundColor Green
