<#
.SYNOPSIS
    A/B Benchmark Harness for Custom SA-IS vs divsufsort

.DESCRIPTION
    Runs statistically rigorous benchmarks to detect performance difference
    between the custom 70KB SA-IS algorithm and divsufsort.

    Expected gain: ~1-3%
    At 17.6 KH/s, 1% = ~176 H/s

    Runs alternating tests to minimize thermal/time-of-day bias.

.PARAMETER Iterations
    Number of iterations per variant (default: 15)

.PARAMETER Duration
    Mining duration per run in seconds (default: 60)

.PARAMETER Threads
    Number of mining threads (default: auto-detect)

.PARAMETER DaemonAddress
    DERO daemon address (default: 203.0.113.10)

.PARAMETER Port
    DERO daemon port (default: 10100)

.PARAMETER Wallet
    DERO wallet address for mining

.PARAMETER CooldownSeconds
    Cooldown between runs in seconds (default: 10)

.PARAMETER SkipBuild
    Skip building and use existing binaries

.EXAMPLE
    .\ab_custom_sa_bench.ps1 -Iterations 15 -Duration 60
#>

param(
    [int]$Iterations = 15,
    [int]$Duration = 60,
    [int]$Threads = 0,  # 0 = auto-detect
    [string]$DaemonAddress = "203.0.113.10",
    [int]$Port = 10100,
    [string]$Wallet = "DERO_WALLET_PLACEHOLDER",
    [int]$CooldownSeconds = 10,
    [switch]$SkipBuild = $false
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir

# MSYS2 paths for build tools
$MSYS2_BIN = "C:\msys64\mingw64\bin"
$env:Path = "$MSYS2_BIN;$env:Path"

# Build directories
$BuildCustomSA = Join-Path $ProjectDir "build-customsa"
$BuildDivsufsort = Join-Path $ProjectDir "build"

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
        [bool]$UseCustomSA = $false
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

    # Add USE_CUSTOM_SA flag if enabled
    if ($UseCustomSA) {
        $cmakeFlags += "-DUSE_CUSTOM_SA=ON"
        Write-Host "Using custom 70KB SA-IS algorithm"
    } else {
        $cmakeFlags += "-DUSE_CUSTOM_SA=OFF"
        Write-Host "Using divsufsort (baseline)"
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

Write-Banner "A/B Custom SA Benchmark Harness"

Write-Host "Configuration:"
Write-Host "  Iterations:   $Iterations per variant"
Write-Host "  Duration:     $Duration seconds per run"
Write-Host "  Threads:      $Threads"
Write-Host "  Daemon:       ${DaemonAddress}:${Port}"
Write-Host "  Cooldown:     $CooldownSeconds seconds"
Write-Host ""

# Build both variants
if (-not $SkipBuild) {
    # Build WITH custom SA
    Build-Variant -BuildDir $BuildCustomSA -Name "Custom SA-IS" -UseCustomSA $true

    # Build with divsufsort (baseline)
    Build-Variant -BuildDir $BuildDivsufsort -Name "divsufsort (baseline)" -UseCustomSA $false
}

# Get miner paths
$minerCustomSA = Get-MinerPath -BuildDir $BuildCustomSA
$minerDivsufsort = Get-MinerPath -BuildDir $BuildDivsufsort

Write-Host "Miner with Custom SA:    $minerCustomSA"
Write-Host "Miner with divsufsort:   $minerDivsufsort"

# Results arrays
$resultsCustomSA = @()
$resultsDivsufsort = @()

Write-Banner "Starting A/B Benchmark"
Write-Host "Total runs: $($Iterations * 2) (alternating)"
Write-Host ""

# Run alternating benchmarks
for ($i = 0; $i -lt $Iterations; $i++) {
    $runNum = $i + 1
    Write-Host ""
    Write-Host "=== Iteration $runNum of $Iterations ===" -ForegroundColor Magenta

    # Run Custom SA first
    $result = Run-Benchmark -MinerPath $minerCustomSA -Name "Custom SA-IS"
    $resultsCustomSA += $result.Hashrate
    Write-Host "  Custom SA-IS: $(Format-Hashrate $result.Hashrate)" -ForegroundColor Green

    # Cooldown
    Write-Host "  Cooling down for $CooldownSeconds seconds..."
    Start-Sleep -Seconds $CooldownSeconds

    # Run divsufsort baseline
    $result = Run-Benchmark -MinerPath $minerDivsufsort -Name "divsufsort"
    $resultsDivsufsort += $result.Hashrate
    Write-Host "  divsufsort: $(Format-Hashrate $result.Hashrate)" -ForegroundColor Yellow

    # Cooldown (except on last iteration)
    if ($i -lt $Iterations - 1) {
        Write-Host "  Cooling down for $CooldownSeconds seconds..."
        Start-Sleep -Seconds $CooldownSeconds
    }
}

# Calculate statistics
Write-Banner "A/B Custom SA Benchmark Results"

$statsCustomSA = Calculate-Statistics -Values $resultsCustomSA
$statsDivsufsort = Calculate-Statistics -Values $resultsDivsufsort
$ttest = Calculate-TTest -Stats1 $statsCustomSA -Stats2 $statsDivsufsort

Write-Host "Custom SA-IS (70KB optimized):" -ForegroundColor Green
Write-Host ("  Mean:     {0}" -f (Format-Hashrate $statsCustomSA.Mean))
Write-Host ("  StdDev:   {0}" -f (Format-Hashrate $statsCustomSA.StdDev))
Write-Host ("  Min:      {0}" -f (Format-Hashrate $statsCustomSA.Min))
Write-Host ("  Max:      {0}" -f (Format-Hashrate $statsCustomSA.Max))
Write-Host ("  Runs:     {0}" -f $statsCustomSA.Count)
Write-Host ""

Write-Host "divsufsort (baseline):" -ForegroundColor Yellow
Write-Host ("  Mean:     {0}" -f (Format-Hashrate $statsDivsufsort.Mean))
Write-Host ("  StdDev:   {0}" -f (Format-Hashrate $statsDivsufsort.StdDev))
Write-Host ("  Min:      {0}" -f (Format-Hashrate $statsDivsufsort.Min))
Write-Host ("  Max:      {0}" -f (Format-Hashrate $statsDivsufsort.Max))
Write-Host ("  Runs:     {0}" -f $statsDivsufsort.Count)
Write-Host ""

$difference = $statsCustomSA.Mean - $statsDivsufsort.Mean
$percentDiff = if ($statsDivsufsort.Mean -gt 0) { ($difference / $statsDivsufsort.Mean) * 100 } else { 0 }

Write-Host "Difference:" -ForegroundColor Cyan
$sign = if ($difference -ge 0) { "+" } else { "" }
Write-Host ("  {0}{1} ({2}{3:F2}%)" -f $sign, (Format-Hashrate $difference), $sign, $percentDiff)
Write-Host ""

Write-Host "Statistical Analysis (Welch's t-test):" -ForegroundColor Cyan
Write-Host ("  T-statistic:        {0:F3}" -f $ttest.TStatistic)
Write-Host ("  p-value:            {0:F4}" -f $ttest.PValue)
Write-Host ("  Degrees of freedom: {0:F1}" -f $ttest.DegreesOfFreedom)
Write-Host ""

# Determine significance
if ($ttest.PValue -lt 0.01) {
    Write-Host "Result: HIGHLY SIGNIFICANT at p<0.01" -ForegroundColor Green
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
    Write-Host ("Conclusion: Custom SA-IS provides ~{0:F2}% improvement over divsufsort" -f $percentDiff) -ForegroundColor Green
}
elseif ($difference -lt 0) {
    Write-Host ("Conclusion: Custom SA-IS causes ~{0:F2}% regression vs divsufsort" -f [Math]::Abs($percentDiff)) -ForegroundColor Red
}
else {
    Write-Host "Conclusion: No measurable difference" -ForegroundColor Yellow
}

# Expected vs Actual comparison
Write-Host ""
Write-Host "Expected improvement: 1-3%" -ForegroundColor DarkGray
if ($percentDiff -gt 0 -and $percentDiff -lt 1) {
    Write-Host "Note: Improvement is below expected range" -ForegroundColor Yellow
} elseif ($percentDiff -ge 1 -and $percentDiff -le 3) {
    Write-Host "Note: Improvement is within expected range!" -ForegroundColor Green
} elseif ($percentDiff -gt 3) {
    Write-Host "Note: Improvement exceeds expected range!" -ForegroundColor Cyan
}

# Output raw data
Write-Host ""
Write-Host "Raw Data:" -ForegroundColor DarkGray
Write-Host "  Custom SA-IS: $($resultsCustomSA -join ', ')" -ForegroundColor DarkGray
Write-Host "  divsufsort:   $($resultsDivsufsort -join ', ')" -ForegroundColor DarkGray

# Save results to file
$resultsFile = Join-Path $ProjectDir "ab_custom_sa_results.json"
$resultsData = @{
    Timestamp = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    Configuration = @{
        Iterations = $Iterations
        Duration = $Duration
        Threads = $Threads
        DaemonAddress = $DaemonAddress
        Port = $Port
    }
    CustomSA = @{
        Hashrates = $resultsCustomSA
        Mean = $statsCustomSA.Mean
        StdDev = $statsCustomSA.StdDev
    }
    Divsufsort = @{
        Hashrates = $resultsDivsufsort
        Mean = $statsDivsufsort.Mean
        StdDev = $statsDivsufsort.StdDev
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
