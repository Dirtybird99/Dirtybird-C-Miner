<#
.SYNOPSIS
    DirtyBird Black-Box Optimization Benchmark

.DESCRIPTION
    Runs optimization benchmarks using the Python scripts.
    Supports full sweeps, single configs, and analysis.

.PARAMETER Mode
    sweep    - Full optimization sweep (default)
    baseline - Baseline only
    quick    - Quick 30s comparison
    analyze  - Analyze existing results

.PARAMETER Duration
    Mining duration per run in seconds (default: 60)

.PARAMETER Runs
    Number of runs per config (default: 3)

.PARAMETER IncludeSweeps
    Include SA prefetch and OMP thread sweeps

.EXAMPLE
    .\run_optimization.ps1 -Mode sweep
    .\run_optimization.ps1 -Mode quick -Duration 30 -Runs 1
#>

param(
    [ValidateSet("sweep", "baseline", "spsa", "cache", "quick", "analyze")]
    [string]$Mode = "sweep",
    [int]$Duration = 60,
    [int]$Runs = 3,
    [int]$Threads = 20,
    [switch]$IncludeSweeps,
    [switch]$Verbose
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir
$ConfigFile = Join-Path $ProjectDir "configs\optimization_matrix.json"
$OutputDir = Join-Path $ProjectDir "benchmark_results"

# Ensure output directory exists
if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir | Out-Null
}

function Write-Banner {
    param([string]$Text)
    $border = "=" * 60
    Write-Host ""
    Write-Host $border -ForegroundColor Cyan
    Write-Host " $Text" -ForegroundColor Cyan
    Write-Host $border -ForegroundColor Cyan
    Write-Host ""
}

function Run-Benchmark {
    param(
        [string]$ConfigName,
        [int]$BenchDuration = $Duration,
        [int]$BenchRuns = $Runs
    )

    $pythonArgs = @(
        (Join-Path $ScriptDir "bench_optimize.py"),
        "--config", $ConfigName,
        "--duration", $BenchDuration,
        "--runs", $BenchRuns,
        "--threads", $Threads,
        "--output", $OutputDir
    )

    if ($Verbose) {
        $pythonArgs += "-v"
    }

    Write-Host "Running: python $pythonArgs" -ForegroundColor Yellow
    & python @pythonArgs
}

function Run-Sweep {
    $pythonArgs = @(
        (Join-Path $ScriptDir "bench_optimize.py"),
        "--sweep",
        "--matrix", $ConfigFile,
        "--duration", $Duration,
        "--runs", $Runs,
        "--threads", $Threads,
        "--output", $OutputDir
    )

    if ($IncludeSweeps) {
        $pythonArgs += "--include-sweeps"
    }

    if ($Verbose) {
        $pythonArgs += "-v"
    }

    Write-Host "Running: python $pythonArgs" -ForegroundColor Yellow
    & python @pythonArgs
}

function Run-Analysis {
    $pythonArgs = @(
        (Join-Path $ScriptDir "analyze_results.py"),
        $OutputDir,
        "--report"
    )

    & python @pythonArgs
}

# Main execution
Write-Banner "DirtyBird Optimization Benchmark"

Write-Host "Mode: $Mode"
Write-Host "Duration: $Duration seconds"
Write-Host "Runs: $Runs per config"
Write-Host "Threads: $Threads"
Write-Host "Output: $OutputDir"
Write-Host ""

switch ($Mode) {
    "sweep" {
        Write-Banner "Running Full Optimization Sweep"
        Run-Sweep
        Write-Banner "Analysis"
        Run-Analysis
    }
    "baseline" {
        Write-Banner "Running Baseline Benchmark"
        Run-Benchmark -ConfigName "baseline_no_spsa"
    }
    "spsa" {
        Write-Banner "Running SPSA Benchmark"
        Run-Benchmark -ConfigName "default_spsa"
    }
    "cache" {
        Write-Banner "Running Cache Batch Benchmark"
        Run-Benchmark -ConfigName "cache_batch"
    }
    "quick" {
        Write-Banner "Running Quick Comparison (30s each)"
        Run-Benchmark -ConfigName "baseline_no_spsa" -BenchDuration 30 -BenchRuns 1
        Run-Benchmark -ConfigName "default_spsa" -BenchDuration 30 -BenchRuns 1
        Write-Banner "Analysis"
        Run-Analysis
    }
    "analyze" {
        Write-Banner "Analyzing Results"
        Run-Analysis
    }
}

Write-Host ""
Write-Host "Done." -ForegroundColor Green
