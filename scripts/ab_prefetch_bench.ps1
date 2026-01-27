<#
.SYNOPSIS
    A/B Benchmark Harness for SA Text Prefetch Validation

.DESCRIPTION
    Runs statistically rigorous benchmarks to detect ~1% performance difference
    from the SA text prefetch optimization with high confidence.

    Expected gain: ~0.7-1.2%
    At 17.6 KH/s, 1% = ~176 H/s

    Runs alternating tests to minimize thermal/time-of-day bias.

.PARAMETER Iterations
    Number of iterations per variant (default: 10)

.PARAMETER Duration
    Mining duration per run in seconds (default: 30)

.PARAMETER Threads
    Number of mining threads (default: auto-detect)

.PARAMETER DaemonAddress
    DERO daemon address (default: 203.0.113.10)

.PARAMETER Port
    DERO daemon port (default: 10100)

.PARAMETER Wallet
    DERO wallet address for mining

.PARAMETER CooldownSeconds
    Cooldown between runs in seconds (default: 5)

.PARAMETER SkipBuild
    Skip building and use existing binaries

.EXAMPLE
    .\ab_prefetch_bench.ps1 -Iterations 15 -Duration 60
#>

param(
    [int]$Iterations = 10,
    [int]$Duration = 30,
    [int]$Threads = 0,  # 0 = auto-detect
    [string]$DaemonAddress = "203.0.113.10",
    [int]$Port = 10100,
    [string]$Wallet = "DERO_WALLET_PLACEHOLDER",
    [int]$CooldownSeconds = 5,
    [switch]$SkipBuild = $false,
    [switch]$DisablePartitionPrefetch = $false
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir

# MSYS2 paths for build tools
$MSYS2_BIN = "C:\msys64\mingw64\bin"
$env:Path = "$MSYS2_BIN;$env:Path"

# Build directories
$BuildWithPrefetch = Join-Path $ProjectDir "build"
$BuildWithoutPrefetch = Join-Path $ProjectDir "build-noprefetch"

# Auto-detect threads if not specified
if ($Threads -eq 0) {
    $Threads = (Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors
    if ($Threads -gt 20) { $Threads = 20 }  # Cap at 20 for safety
    Write-Host "Auto-detected $Threads threads"
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

function Build-Variant {
    param(
        [string]$BuildDir,
        [string]$Name,
        [hashtable]$ExtraFlags = @{}
    )

    Write-Banner "Building $Name variant"

    # Clean and create build directory
    if (Test-Path $BuildDir) {
        Write-Host "Cleaning existing build directory..."
        Remove-Item -Recurse -Force $BuildDir
    }
    New-Item -ItemType Directory -Path $BuildDir | Out-Null

    # Prepare CMake flags
    $cmakeFlags = @(
        "-G", "Ninja",
        "-DCMAKE_BUILD_TYPE=Release"
    )

    # Add extra C flags for disabling prefetch
    if ($ExtraFlags.Count -gt 0) {
        $cFlags = $ExtraFlags.Values -join " "
        $cmakeFlags += "-DCMAKE_C_FLAGS_RELEASE=$cFlags"
        $cmakeFlags += "-DCMAKE_CXX_FLAGS_RELEASE=$cFlags"
        Write-Host "Extra flags: $cFlags"
    }

    # Run CMake
    Write-Host "Running CMake..."
    Push-Location $BuildDir
    try {
        $cmakeArgs = @("-S", $ProjectDir, "-B", ".") + $cmakeFlags
        & cmake $cmakeArgs
        if ($LASTEXITCODE -ne 0) { throw "CMake configuration failed" }

        Write-Host "Building..."
        & ninja
        if ($LASTEXITCODE -ne 0) { throw "Build failed" }
    }
    finally {
        Pop-Location
    }

    Write-Host "Build complete: $Name" -ForegroundColor Green
}

function Get-MinerPath {
    param([string]$BuildDir)

    $paths = @(
        (Join-Path $BuildDir "bin\dirtybird-miner-cpu.exe"),
        (Join-Path $BuildDir "dirtybird-miner-cpu.exe")
    )

    foreach ($path in $paths) {
        if (Test-Path $path) { return $path }
    }

    throw "Miner executable not found in $BuildDir"
}

function Run-Benchmark {
    param(
        [string]$MinerPath,
        [string]$Name
    )

    Write-Host "Running $Name benchmark ($Duration seconds)..." -ForegroundColor Yellow

    $minerArgs = @(
        "--dero",
        "--daemon-address", $DaemonAddress,
        "--port", $Port,
        "--wallet", $Wallet,
        "--threads", $Threads,
        "--mine-time", $Duration
    )

    $startTime = Get-Date
    # Use Start-Process to avoid PowerShell treating stderr as errors
    $tempFile = [System.IO.Path]::GetTempFileName()
    $process = Start-Process -FilePath $MinerPath -ArgumentList $minerArgs -NoNewWindow -Wait -RedirectStandardOutput $tempFile -RedirectStandardError ([System.IO.Path]::GetTempFileName()) -PassThru
    $output = Get-Content $tempFile -Raw
    Remove-Item $tempFile -ErrorAction SilentlyContinue
    $endTime = Get-Date

    # Parse hashrate from output
    # Look for patterns like "17.82 KH/s" or "17820 H/s"
    $hashrate = 0

    # Try to find final/average hashrate
    if ($output -match "(?:final|average|total).*?(\d+\.?\d*)\s*KH/s") {
        $hashrate = [double]$Matches[1] * 1000
    }
    elseif ($output -match "(?:final|average|total).*?(\d+\.?\d*)\s*H/s") {
        $hashrate = [double]$Matches[1]
    }
    # Fallback: find any hashrate
    elseif ($output -match "(\d+\.?\d*)\s*KH/s") {
        $hashrate = [double]$Matches[1] * 1000
    }
    elseif ($output -match "(\d+)\s*H/s") {
        $hashrate = [double]$Matches[1]
    }

    if ($hashrate -eq 0) {
        Write-Host "WARNING: Could not parse hashrate from output" -ForegroundColor Red
        Write-Host "Output snippet: $($output.Substring(0, [Math]::Min(500, $output.Length)))"
    }

    return @{
        Hashrate = $hashrate
        Duration = ($endTime - $startTime).TotalSeconds
        Output = $output
    }
}

function Calculate-Statistics {
    param([double[]]$Values)

    if ($Values.Count -eq 0) { return $null }

    $n = $Values.Count
    $mean = ($Values | Measure-Object -Sum).Sum / $n

    $variance = 0
    foreach ($v in $Values) {
        $variance += [Math]::Pow($v - $mean, 2)
    }
    $variance /= ($n - 1)
    $stddev = [Math]::Sqrt($variance)

    return @{
        Mean = $mean
        StdDev = $stddev
        Variance = $variance
        Count = $n
        Min = ($Values | Measure-Object -Minimum).Minimum
        Max = ($Values | Measure-Object -Maximum).Maximum
    }
}

function Calculate-TTest {
    param(
        [hashtable]$Stats1,
        [hashtable]$Stats2
    )

    # Welch's t-test (unequal variances)
    $mean1 = $Stats1.Mean
    $mean2 = $Stats2.Mean
    $var1 = $Stats1.Variance
    $var2 = $Stats2.Variance
    $n1 = $Stats1.Count
    $n2 = $Stats2.Count

    # T-statistic
    $se = [Math]::Sqrt($var1/$n1 + $var2/$n2)
    if ($se -eq 0) { return @{ TStatistic = 0; PValue = 1; DegreesOfFreedom = 0 } }

    $t = ($mean1 - $mean2) / $se

    # Welch-Satterthwaite degrees of freedom
    $num = [Math]::Pow($var1/$n1 + $var2/$n2, 2)
    $denom = [Math]::Pow($var1/$n1, 2)/($n1-1) + [Math]::Pow($var2/$n2, 2)/($n2-1)
    $df = $num / $denom

    # Approximate p-value using t-distribution
    # For simplicity, we use a lookup table for common values
    $absT = [Math]::Abs($t)
    $pValue = Get-ApproximatePValue -T $absT -DF $df

    return @{
        TStatistic = $t
        PValue = $pValue
        DegreesOfFreedom = $df
    }
}

function Get-ApproximatePValue {
    param(
        [double]$T,
        [double]$DF
    )

    # Approximation using normal distribution for large df
    # For smaller df, we use conservative estimates

    if ($DF -gt 30) {
        # Use normal approximation
        # P(|Z| > t) ~ 2 * (1 - Phi(t))
        # Approximate using error function
        $z = $T / [Math]::Sqrt(2)
        $erf = 1 - [Math]::Exp(-$z * $z * (1.273239544735163 + 0.14001228868667 * $z * $z) / (1 + 0.14001228868667 * $z * $z))
        $pValue = 1 - $erf
    }
    else {
        # Conservative lookup for small df
        # Critical values for two-tailed test
        if ($T -gt 4.0) { $pValue = 0.001 }
        elseif ($T -gt 3.5) { $pValue = 0.002 }
        elseif ($T -gt 3.0) { $pValue = 0.005 }
        elseif ($T -gt 2.75) { $pValue = 0.01 }
        elseif ($T -gt 2.5) { $pValue = 0.02 }
        elseif ($T -gt 2.2) { $pValue = 0.05 }
        elseif ($T -gt 2.0) { $pValue = 0.1 }
        elseif ($T -gt 1.5) { $pValue = 0.2 }
        else { $pValue = 0.5 }
    }

    return $pValue
}

function Format-Hashrate {
    param([double]$Hashrate)

    if ($Hashrate -ge 1000) {
        return "{0:F2} KH/s" -f ($Hashrate / 1000)
    }
    return "{0:F0} H/s" -f $Hashrate
}

# ============================================================================
# Main Script
# ============================================================================

Write-Banner "A/B Prefetch Benchmark Harness"

Write-Host "Configuration:"
Write-Host "  Iterations:   $Iterations per variant"
Write-Host "  Duration:     $Duration seconds per run"
Write-Host "  Threads:      $Threads"
Write-Host "  Daemon:       ${DaemonAddress}:${Port}"
Write-Host "  Cooldown:     $CooldownSeconds seconds"
Write-Host ""

# Build both variants
if (-not $SkipBuild) {
    # Build WITH prefetch (default)
    Build-Variant -BuildDir $BuildWithPrefetch -Name "WITH prefetch"

    # Build WITHOUT prefetch
    $noPrefetchFlags = @{
        "SA_TEXT" = "-DDISABLE_SA_TEXT_PREFETCH"
    }
    if ($DisablePartitionPrefetch) {
        $noPrefetchFlags["PARTITION"] = "-DDISABLE_PARTITION_PREFETCH"
    }
    Build-Variant -BuildDir $BuildWithoutPrefetch -Name "WITHOUT prefetch" -ExtraFlags $noPrefetchFlags
}

# Get miner paths
$minerWithPrefetch = Get-MinerPath -BuildDir $BuildWithPrefetch
$minerWithoutPrefetch = Get-MinerPath -BuildDir $BuildWithoutPrefetch

Write-Host "Miner WITH prefetch:    $minerWithPrefetch"
Write-Host "Miner WITHOUT prefetch: $minerWithoutPrefetch"

# Results arrays
$resultsWithPrefetch = @()
$resultsWithoutPrefetch = @()

Write-Banner "Starting A/B Benchmark"
Write-Host "Total runs: $($Iterations * 2) (alternating)"
Write-Host ""

# Run alternating benchmarks
for ($i = 0; $i -lt $Iterations; $i++) {
    $runNum = $i + 1
    Write-Host ""
    Write-Host "=== Iteration $runNum of $Iterations ===" -ForegroundColor Magenta

    # Run WITH prefetch
    $result = Run-Benchmark -MinerPath $minerWithPrefetch -Name "WITH prefetch"
    $resultsWithPrefetch += $result.Hashrate
    Write-Host "  WITH prefetch: $(Format-Hashrate $result.Hashrate)" -ForegroundColor Green

    # Cooldown
    Write-Host "  Cooling down for $CooldownSeconds seconds..."
    Start-Sleep -Seconds $CooldownSeconds

    # Run WITHOUT prefetch
    $result = Run-Benchmark -MinerPath $minerWithoutPrefetch -Name "WITHOUT prefetch"
    $resultsWithoutPrefetch += $result.Hashrate
    Write-Host "  WITHOUT prefetch: $(Format-Hashrate $result.Hashrate)" -ForegroundColor Yellow

    # Cooldown (except on last iteration)
    if ($i -lt $Iterations - 1) {
        Write-Host "  Cooling down for $CooldownSeconds seconds..."
        Start-Sleep -Seconds $CooldownSeconds
    }
}

# Calculate statistics
Write-Banner "A/B Prefetch Benchmark Results"

$statsWithPrefetch = Calculate-Statistics -Values $resultsWithPrefetch
$statsWithoutPrefetch = Calculate-Statistics -Values $resultsWithoutPrefetch
$ttest = Calculate-TTest -Stats1 $statsWithPrefetch -Stats2 $statsWithoutPrefetch

Write-Host "WITH Text Prefetch:" -ForegroundColor Green
Write-Host ("  Mean:     {0}" -f (Format-Hashrate $statsWithPrefetch.Mean))
Write-Host ("  StdDev:   {0}" -f (Format-Hashrate $statsWithPrefetch.StdDev))
Write-Host ("  Min:      {0}" -f (Format-Hashrate $statsWithPrefetch.Min))
Write-Host ("  Max:      {0}" -f (Format-Hashrate $statsWithPrefetch.Max))
Write-Host ("  Runs:     {0}" -f $statsWithPrefetch.Count)
Write-Host ""

Write-Host "WITHOUT Text Prefetch:" -ForegroundColor Yellow
Write-Host ("  Mean:     {0}" -f (Format-Hashrate $statsWithoutPrefetch.Mean))
Write-Host ("  StdDev:   {0}" -f (Format-Hashrate $statsWithoutPrefetch.StdDev))
Write-Host ("  Min:      {0}" -f (Format-Hashrate $statsWithoutPrefetch.Min))
Write-Host ("  Max:      {0}" -f (Format-Hashrate $statsWithoutPrefetch.Max))
Write-Host ("  Runs:     {0}" -f $statsWithoutPrefetch.Count)
Write-Host ""

$difference = $statsWithPrefetch.Mean - $statsWithoutPrefetch.Mean
$percentDiff = if ($statsWithoutPrefetch.Mean -gt 0) { ($difference / $statsWithoutPrefetch.Mean) * 100 } else { 0 }

Write-Host "Difference:" -ForegroundColor Cyan
$sign = if ($difference -ge 0) { "+" } else { "" }
Write-Host ("  {0}{1} ({2}{3:F2}%)" -f $sign, (Format-Hashrate $difference), $sign, $percentDiff)
Write-Host ""

Write-Host "Statistical Analysis:" -ForegroundColor Cyan
Write-Host ("  T-statistic:       {0:F3}" -f $ttest.TStatistic)
Write-Host ("  p-value:           {0:F4}" -f $ttest.PValue)
Write-Host ("  Degrees of freedom: {0:F1}" -f $ttest.DegreesOfFreedom)
Write-Host ""

# Determine significance
if ($ttest.PValue -lt 0.01) {
    Write-Host "Result: SIGNIFICANT at p<0.01" -ForegroundColor Green
}
elseif ($ttest.PValue -lt 0.05) {
    Write-Host "Result: SIGNIFICANT at p<0.05" -ForegroundColor Green
}
elseif ($ttest.PValue -lt 0.10) {
    Write-Host "Result: MARGINALLY SIGNIFICANT at p<0.10" -ForegroundColor Yellow
}
else {
    Write-Host "Result: NOT SIGNIFICANT (p >= 0.10)" -ForegroundColor Red
}

Write-Host ""
if ($difference -gt 0) {
    Write-Host ("Conclusion: Text prefetch provides ~{0:F1}% improvement" -f $percentDiff) -ForegroundColor Green
}
elseif ($difference -lt 0) {
    Write-Host ("Conclusion: Text prefetch causes ~{0:F1}% regression" -f [Math]::Abs($percentDiff)) -ForegroundColor Red
}
else {
    Write-Host "Conclusion: No measurable difference" -ForegroundColor Yellow
}

# Output raw data
Write-Host ""
Write-Host "Raw Data:" -ForegroundColor DarkGray
Write-Host "  WITH prefetch:    $($resultsWithPrefetch -join ', ')" -ForegroundColor DarkGray
Write-Host "  WITHOUT prefetch: $($resultsWithoutPrefetch -join ', ')" -ForegroundColor DarkGray

# Save results to file
$resultsFile = Join-Path $ProjectDir "ab_benchmark_results.json"
$resultsData = @{
    Timestamp = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    Configuration = @{
        Iterations = $Iterations
        Duration = $Duration
        Threads = $Threads
        DaemonAddress = $DaemonAddress
        Port = $Port
    }
    WithPrefetch = @{
        Hashrates = $resultsWithPrefetch
        Mean = $statsWithPrefetch.Mean
        StdDev = $statsWithPrefetch.StdDev
    }
    WithoutPrefetch = @{
        Hashrates = $resultsWithoutPrefetch
        Mean = $statsWithoutPrefetch.Mean
        StdDev = $statsWithoutPrefetch.StdDev
    }
    Analysis = @{
        Difference = $difference
        PercentDifference = $percentDiff
        TStatistic = $ttest.TStatistic
        PValue = $ttest.PValue
        DegreesOfFreedom = $ttest.DegreesOfFreedom
    }
}

$resultsData | ConvertTo-Json -Depth 10 | Set-Content $resultsFile
Write-Host ""
Write-Host "Results saved to: $resultsFile" -ForegroundColor Cyan
