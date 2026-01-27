#!/usr/bin/env python3
"""
DirtyBird Black-Box Optimization Benchmark

Runs statistically rigorous benchmarks to optimize DirtyBird miner hashrate.
Supports warmup exclusion, multiple runs per config, and statistical A/B testing.

Usage:
    python bench_optimize.py --config baseline --runs 3 --duration 60
    python bench_optimize.py --sweep --output results/
    python bench_optimize.py --compare results/baseline.json results/cache_batch.json
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional
import statistics
import math


@dataclass
class BenchResult:
    """Stores benchmark results for a single configuration."""
    config_name: str
    flags: list[str]
    samples: list[float] = field(default_factory=list)  # Post-warmup hashrates in KH/s
    mean: float = 0.0
    stddev: float = 0.0
    cv: float = 0.0  # Coefficient of variation (stddev/mean)
    min_hr: float = 0.0
    max_hr: float = 0.0
    p5: float = 0.0  # 5th percentile
    p95: float = 0.0  # 95th percentile
    stability_ratio: float = 0.0  # p5/p95
    run_count: int = 0
    duration: float = 0.0
    warmup: float = 0.0
    timestamp: str = ""

    def compute_stats(self):
        """Calculate statistics from samples."""
        if len(self.samples) < 2:
            return

        self.mean = statistics.mean(self.samples)
        self.stddev = statistics.stdev(self.samples)
        self.cv = (self.stddev / self.mean * 100) if self.mean > 0 else 0
        self.min_hr = min(self.samples)
        self.max_hr = max(self.samples)

        # Percentiles
        sorted_samples = sorted(self.samples)
        n = len(sorted_samples)
        self.p5 = sorted_samples[int(n * 0.05)]
        self.p95 = sorted_samples[int(n * 0.95)]
        self.stability_ratio = (self.p5 / self.p95) if self.p95 > 0 else 0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


def find_miner_executable(base_dir: Path) -> Path:
    """Find the DirtyBird miner executable."""
    candidates = [
        base_dir / "build-optimized" / "bin" / "dirtybird-miner-cpu.exe",
        base_dir / "build" / "bin" / "dirtybird-miner-cpu.exe",
        base_dir / "build" / "dirtybird-miner-cpu.exe",
        base_dir / "dirtybird-miner-cpu.exe",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(f"Could not find dirtybird-miner-cpu.exe in {base_dir}")


def parse_hashrate_samples(output: str, warmup_seconds: int = 15) -> list[float]:
    """
    Parse hashrate samples from miner output, excluding warmup period.

    Looks for patterns like:
    - "HASHRATE X.XX KH/s"
    - "HASHRATE X H/s"
    - "UPTIME 0d-0h-0m-Ns"
    """
    samples = []

    # Pattern for hashrate line with uptime (KH/s)
    # DIRTYBIRD-MINER 0.0.1 HASHRATE 6.05 KH/s | ACCEPTED 0 | ... | UPTIME 0d-0h-0m-19s
    pattern_khs = re.compile(
        r'HASHRATE\s+(\d+\.?\d*)\s*KH/s.*?UPTIME\s+(\d+)d-(\d+)h-(\d+)m-(\d+)s',
        re.IGNORECASE
    )

    # Pattern for hashrate line with uptime (H/s - used by cache-batch mode)
    pattern_hs = re.compile(
        r'HASHRATE\s+(\d+\.?\d*)\s*H/s.*?UPTIME\s+(\d+)d-(\d+)h-(\d+)m-(\d+)s',
        re.IGNORECASE
    )

    # Try KH/s first
    for match in pattern_khs.finditer(output):
        hashrate = float(match.group(1))
        days = int(match.group(2))
        hours = int(match.group(3))
        minutes = int(match.group(4))
        seconds = int(match.group(5))

        uptime = days * 86400 + hours * 3600 + minutes * 60 + seconds

        # Exclude warmup period
        if uptime >= warmup_seconds:
            samples.append(hashrate)

    # If no KH/s samples, try H/s
    if not samples:
        for match in pattern_hs.finditer(output):
            hashrate = float(match.group(1)) / 1000.0  # Convert H/s to KH/s
            days = int(match.group(2))
            hours = int(match.group(3))
            minutes = int(match.group(4))
            seconds = int(match.group(5))

            uptime = days * 86400 + hours * 3600 + minutes * 60 + seconds

            # Exclude warmup period
            if uptime >= warmup_seconds:
                samples.append(hashrate)

    # If no samples found with uptime, try simpler patterns
    if not samples:
        # Try KH/s
        simple_khs = re.compile(r'HASHRATE\s+(\d+\.?\d*)\s*KH/s', re.IGNORECASE)
        matches = simple_khs.findall(output)
        if matches:
            warmup_samples = warmup_seconds // 3
            samples = [float(m) for m in matches[warmup_samples:]]
        else:
            # Try H/s
            simple_hs = re.compile(r'HASHRATE\s+(\d+\.?\d*)\s*H/s', re.IGNORECASE)
            matches = simple_hs.findall(output)
            if matches:
                warmup_samples = warmup_seconds // 3
                samples = [float(m) / 1000.0 for m in matches[warmup_samples:]]

    return samples


def parse_final_hashrate(output: str) -> Optional[float]:
    """Parse the final hashrate from the summary line."""
    # Look for: "Dero: 4 threads @ 6.05 with 0 shares"
    pattern = re.compile(r'threads?\s*@\s*(\d+\.?\d*)', re.IGNORECASE)
    match = pattern.search(output)
    if match:
        return float(match.group(1))

    # Fallback: last HASHRATE line (KH/s)
    pattern_khs = re.compile(r'HASHRATE\s+(\d+\.?\d*)\s*KH/s', re.IGNORECASE)
    matches = pattern_khs.findall(output)
    if matches:
        return float(matches[-1])

    # Fallback: last HASHRATE line (H/s) - convert to KH/s
    pattern_hs = re.compile(r'HASHRATE\s+(\d+\.?\d*)\s*H/s', re.IGNORECASE)
    matches = pattern_hs.findall(output)
    if matches:
        return float(matches[-1]) / 1000.0

    return None


def run_single_benchmark(
    miner_path: Path,
    flags: list[str],
    duration: int,
    threads: int,
    daemon_address: str,
    port: int,
    wallet: str,
    verbose: bool = False
) -> tuple[list[float], str, float]:
    """
    Run a single benchmark and return hashrate samples.

    Returns:
        Tuple of (samples, output, final_hashrate)
    """
    cmd = [
        str(miner_path),
        "--dero",
        "--daemon-address", daemon_address,
        "--port", str(port),
        "--wallet", wallet,
        "--threads", str(threads),
        "--mine-time", str(duration),
        "--tune-warmup", "1",
        "--tune-duration", "1",
    ] + flags

    if verbose:
        print(f"  Running: {' '.join(cmd)}")

    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=duration + 60,  # Extra time for startup/shutdown
            cwd=miner_path.parent
        )
        output = result.stdout + result.stderr
    except subprocess.TimeoutExpired as e:
        # With text=True, stdout/stderr are already strings
        output = (e.stdout or "") + (e.stderr or "")

    elapsed = time.time() - start_time

    if verbose:
        print(f"  Completed in {elapsed:.1f}s")

    return output, elapsed


def run_benchmark(
    miner_path: Path,
    config_name: str,
    flags: list[str],
    runs: int = 3,
    duration: int = 60,
    warmup: int = 15,
    threads: int = 0,
    daemon_address: str = "203.0.113.10",
    port: int = 10100,
    wallet: str = "DERO_WALLET_PLACEHOLDER",
    cooldown: int = 5,
    verbose: bool = False
) -> BenchResult:
    """
    Run multiple benchmark iterations and aggregate results.
    """
    if threads == 0:
        threads = min(os.cpu_count() or 8, 20)  # Cap at 20

    result = BenchResult(
        config_name=config_name,
        flags=flags,
        run_count=runs,
        duration=duration,
        warmup=warmup,
        timestamp=datetime.now().isoformat()
    )

    all_samples = []

    print(f"\n{'='*60}")
    print(f"Benchmarking: {config_name}")
    print(f"Flags: {' '.join(flags) if flags else '(default)'}")
    print(f"Duration: {duration}s, Warmup: {warmup}s, Runs: {runs}")
    print(f"{'='*60}")

    for run in range(runs):
        print(f"\n  Run {run + 1}/{runs}...")

        output, elapsed = run_single_benchmark(
            miner_path=miner_path,
            flags=flags,
            duration=duration,
            threads=threads,
            daemon_address=daemon_address,
            port=port,
            wallet=wallet,
            verbose=verbose
        )

        # Parse samples from this run
        samples = parse_hashrate_samples(output, warmup)
        final_hr = parse_final_hashrate(output)

        if samples:
            all_samples.extend(samples)
            run_mean = statistics.mean(samples) if samples else 0
            print(f"    Samples: {len(samples)}, Mean: {run_mean:.2f} KH/s, Final: {final_hr:.2f} KH/s")
        else:
            print(f"    WARNING: No samples collected (final: {final_hr})")
            if final_hr:
                all_samples.append(final_hr)

        # Cooldown between runs
        if run < runs - 1:
            print(f"    Cooldown: {cooldown}s...")
            time.sleep(cooldown)

    result.samples = all_samples
    result.compute_stats()

    print(f"\n  Results for {config_name}:")
    print(f"    Mean:      {result.mean:.2f} KH/s")
    print(f"    StdDev:    {result.stddev:.2f} KH/s")
    print(f"    CV:        {result.cv:.2f}%")
    print(f"    Min:       {result.min_hr:.2f} KH/s")
    print(f"    Max:       {result.max_hr:.2f} KH/s")
    print(f"    P5/P95:    {result.p5:.2f} / {result.p95:.2f} KH/s")
    print(f"    Stability: {result.stability_ratio:.2%}")

    return result


def welch_t_test(samples1: list[float], samples2: list[float]) -> tuple[float, float, float]:
    """
    Perform Welch's t-test (unequal variance t-test).

    Returns:
        Tuple of (t_statistic, p_value, degrees_of_freedom)
    """
    n1, n2 = len(samples1), len(samples2)
    if n1 < 2 or n2 < 2:
        return 0, 1, 0

    mean1, mean2 = statistics.mean(samples1), statistics.mean(samples2)
    var1, var2 = statistics.variance(samples1), statistics.variance(samples2)

    # Standard error
    se = math.sqrt(var1/n1 + var2/n2)
    if se == 0:
        return 0, 1, 0

    # T-statistic
    t = (mean1 - mean2) / se

    # Welch-Satterthwaite degrees of freedom
    num = (var1/n1 + var2/n2) ** 2
    denom = (var1/n1)**2 / (n1-1) + (var2/n2)**2 / (n2-1)
    df = num / denom if denom > 0 else 1

    # Approximate p-value (using normal approximation for df > 30)
    abs_t = abs(t)
    if df > 30:
        # Normal approximation
        z = abs_t / math.sqrt(2)
        # Error function approximation
        erf = 1 - math.exp(-z * z * (1.273239544735163 + 0.14001228868667 * z * z) /
                          (1 + 0.14001228868667 * z * z))
        p_value = 1 - erf
    else:
        # Conservative lookup
        if abs_t > 4.0: p_value = 0.001
        elif abs_t > 3.5: p_value = 0.002
        elif abs_t > 3.0: p_value = 0.005
        elif abs_t > 2.75: p_value = 0.01
        elif abs_t > 2.5: p_value = 0.02
        elif abs_t > 2.2: p_value = 0.05
        elif abs_t > 2.0: p_value = 0.1
        elif abs_t > 1.5: p_value = 0.2
        else: p_value = 0.5

    return t, p_value, df


def compare_results(result_a: BenchResult, result_b: BenchResult) -> dict:
    """
    Compare two benchmark results with statistical testing.
    """
    t_stat, p_value, df = welch_t_test(result_a.samples, result_b.samples)

    # Effect size (Cohen's d)
    pooled_std = math.sqrt((result_a.stddev**2 + result_b.stddev**2) / 2)
    effect_size = (result_a.mean - result_b.mean) / pooled_std if pooled_std > 0 else 0

    # Percent improvement
    pct_improvement = ((result_a.mean - result_b.mean) / result_b.mean * 100) if result_b.mean > 0 else 0

    return {
        "config_a": result_a.config_name,
        "config_b": result_b.config_name,
        "mean_a": result_a.mean,
        "mean_b": result_b.mean,
        "mean_diff": result_a.mean - result_b.mean,
        "pct_improvement": pct_improvement,
        "t_statistic": t_stat,
        "p_value": p_value,
        "degrees_of_freedom": df,
        "effect_size": effect_size,
        "significant": p_value < 0.05,
        "practically_meaningful": abs(effect_size) > 0.2,
        "cv_a": result_a.cv,
        "cv_b": result_b.cv,
        "cv_improved": result_a.cv < result_b.cv,
    }


def save_result(result: BenchResult, output_dir: Path):
    """Save benchmark result to JSON and CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON output
    json_path = output_dir / f"{result.config_name}.json"
    with open(json_path, 'w') as f:
        json.dump(result.to_dict(), f, indent=2)
    print(f"  Saved: {json_path}")

    # CSV time-series output
    csv_path = output_dir / f"{result.config_name}_samples.csv"
    with open(csv_path, 'w') as f:
        f.write("sample_index,hashrate_khs\n")
        for i, sample in enumerate(result.samples):
            f.write(f"{i},{sample}\n")
    print(f"  Saved: {csv_path}")


def load_result(json_path: Path) -> BenchResult:
    """Load benchmark result from JSON."""
    with open(json_path) as f:
        data = json.load(f)

    result = BenchResult(
        config_name=data["config_name"],
        flags=data["flags"],
        samples=data["samples"],
    )
    result.mean = data["mean"]
    result.stddev = data["stddev"]
    result.cv = data["cv"]
    result.min_hr = data["min_hr"]
    result.max_hr = data["max_hr"]
    result.p5 = data["p5"]
    result.p95 = data["p95"]
    result.stability_ratio = data["stability_ratio"]
    result.run_count = data["run_count"]
    result.duration = data["duration"]
    result.warmup = data["warmup"]
    result.timestamp = data["timestamp"]

    return result


# Default configurations for optimization sweep
DEFAULT_CONFIGS = [
    ("baseline_no_spsa", ["--no-spsa"]),
    ("default_spsa", []),
    ("cache_batch", ["--cache-batch"]),
    ("sa_tune", ["--sa-tune"]),
    ("cache_batch_sa_tune", ["--cache-batch", "--sa-tune"]),
    ("no_lock", ["--no-lock"]),
    ("no_lock_cache_batch", ["--no-lock", "--cache-batch"]),
    ("no_lock_sa_tune", ["--no-lock", "--sa-tune"]),
    ("no_lock_combined", ["--no-lock", "--cache-batch", "--sa-tune"]),
]


def load_config_from_json(config_path: Path) -> dict:
    """Load test configuration from optimization_matrix.json."""
    with open(config_path) as f:
        return json.load(f)


def get_configs_from_matrix(matrix: dict, include_sweeps: bool = False) -> list[tuple[str, list[str]]]:
    """Extract configurations from the optimization matrix."""
    configs = []

    # Main configurations
    for cfg in matrix.get("configurations", []):
        configs.append((cfg["name"], cfg["flags"]))

    # Optionally include sweep configurations
    if include_sweeps:
        # SA prefetch sweep
        for cfg in matrix.get("sa_prefetch_sweep", {}).get("configurations", []):
            configs.append((cfg["name"], cfg["flags"]))

        # OMP thread sweep
        for cfg in matrix.get("omp_thread_sweep", {}).get("configurations", []):
            configs.append((cfg["name"], cfg["flags"]))

    return configs


def run_sweep(
    miner_path: Path,
    configs: list[tuple[str, list[str]]],
    output_dir: Path,
    **kwargs
) -> list[BenchResult]:
    """Run a full configuration sweep."""
    results = []

    print(f"\n{'#'*60}")
    print(f"# OPTIMIZATION SWEEP")
    print(f"# Configs: {len(configs)}")
    print(f"# Output: {output_dir}")
    print(f"{'#'*60}")

    for config_name, flags in configs:
        try:
            result = run_benchmark(
                miner_path=miner_path,
                config_name=config_name,
                flags=flags,
                **kwargs
            )
            results.append(result)
            save_result(result, output_dir)
        except Exception as e:
            print(f"ERROR: Failed to benchmark {config_name}: {e}")

    return results


def print_summary(results: list[BenchResult], baseline_name: str = "baseline_no_spsa"):
    """Print comparison summary."""
    print(f"\n{'='*80}")
    print("OPTIMIZATION SUMMARY")
    print(f"{'='*80}")

    # Sort by mean hashrate (descending)
    sorted_results = sorted(results, key=lambda r: r.mean, reverse=True)

    # Find baseline
    baseline = next((r for r in results if r.config_name == baseline_name), results[0])

    print(f"\n{'Config':<30} {'Mean':>10} {'StdDev':>8} {'CV':>8} {'vs Base':>10} {'p-value':>10}")
    print("-" * 80)

    for result in sorted_results:
        comparison = compare_results(result, baseline)
        pct_diff = comparison["pct_improvement"]
        p_value = comparison["p_value"]
        sig = "*" if comparison["significant"] else ""

        print(f"{result.config_name:<30} {result.mean:>9.2f} {result.stddev:>7.2f} {result.cv:>7.2f}% {pct_diff:>+9.2f}% {p_value:>9.4f}{sig}")

    # Best result
    best = sorted_results[0]
    print(f"\nBEST CONFIG: {best.config_name}")
    print(f"  Flags: {' '.join(best.flags) if best.flags else '(default)'}")
    print(f"  Mean: {best.mean:.2f} KH/s, CV: {best.cv:.2f}%")

    # Recommendations
    print(f"\nRECOMMENDATIONS:")
    for result in sorted_results[:3]:
        if result.cv < 5:
            print(f"  - {result.config_name}: Excellent stability (CV < 5%)")
        elif result.cv < 10:
            print(f"  - {result.config_name}: Good stability (CV < 10%)")


def main():
    parser = argparse.ArgumentParser(
        description="DirtyBird Black-Box Optimization Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run single config benchmark
  python bench_optimize.py --config baseline_no_spsa --runs 3 --duration 60

  # Run full optimization sweep
  python bench_optimize.py --sweep --output benchmark_results/

  # Compare two results
  python bench_optimize.py --compare results/a.json results/b.json
        """
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--config", help="Run single config (name from defaults or 'custom')")
    mode_group.add_argument("--sweep", action="store_true", help="Run full optimization sweep")
    mode_group.add_argument("--compare", nargs=2, metavar=("A", "B"), help="Compare two result files")

    # Benchmark parameters
    parser.add_argument("--runs", type=int, default=3, help="Number of runs per config (default: 3)")
    parser.add_argument("--duration", type=int, default=60, help="Mining duration in seconds (default: 60)")
    parser.add_argument("--warmup", type=int, default=15, help="Warmup exclusion in seconds (default: 15)")
    parser.add_argument("--threads", type=int, default=0, help="Mining threads (default: auto)")
    parser.add_argument("--cooldown", type=int, default=5, help="Cooldown between runs in seconds (default: 5)")

    # Connection
    parser.add_argument("--daemon", default="203.0.113.10", help="Daemon address")
    parser.add_argument("--port", type=int, default=10100, help="Daemon port")
    parser.add_argument("--wallet", default="DERO_WALLET_PLACEHOLDER",
                        help="Wallet address")

    # Output
    parser.add_argument("--output", type=Path, default=Path("benchmark_results"),
                        help="Output directory")
    parser.add_argument("--miner", type=Path, help="Path to miner executable")
    parser.add_argument("--flags", type=str, default="", help="Custom flags for --config custom (space-separated, e.g., '--no-lock --sa-tune')")
    parser.add_argument("--name", help="Custom name for results (with --config custom)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    # Configuration matrix
    parser.add_argument("--matrix", type=Path, help="Load configuration from JSON matrix file")
    parser.add_argument("--include-sweeps", action="store_true",
                        help="Include SA prefetch and OMP sweeps (with --matrix)")

    args = parser.parse_args()

    # Find miner
    base_dir = Path(__file__).parent.parent
    if args.miner:
        miner_path = args.miner
    else:
        try:
            miner_path = find_miner_executable(base_dir)
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            return 1

    print(f"Miner: {miner_path}")

    common_kwargs = {
        "runs": args.runs,
        "duration": args.duration,
        "warmup": args.warmup,
        "threads": args.threads,
        "daemon_address": args.daemon,
        "port": args.port,
        "wallet": args.wallet,
        "cooldown": args.cooldown,
        "verbose": args.verbose,
    }

    if args.compare:
        # Compare two result files
        result_a = load_result(Path(args.compare[0]))
        result_b = load_result(Path(args.compare[1]))
        comparison = compare_results(result_a, result_b)

        print(f"\nComparison: {result_a.config_name} vs {result_b.config_name}")
        print(f"  Mean: {result_a.mean:.2f} vs {result_b.mean:.2f} KH/s")
        print(f"  Difference: {comparison['mean_diff']:.2f} KH/s ({comparison['pct_improvement']:+.2f}%)")
        print(f"  T-statistic: {comparison['t_statistic']:.3f}")
        print(f"  P-value: {comparison['p_value']:.4f}")
        print(f"  Effect size: {comparison['effect_size']:.3f}")
        print(f"  Significant: {comparison['significant']}")
        print(f"  Practically meaningful: {comparison['practically_meaningful']}")

    elif args.sweep:
        # Load configs from matrix or use defaults
        if args.matrix and args.matrix.exists():
            matrix = load_config_from_json(args.matrix)
            configs = get_configs_from_matrix(matrix, args.include_sweeps)

            # Override connection settings from matrix if not specified on command line
            env = matrix.get("test_environment", {})
            if args.daemon == "203.0.113.10":  # default
                common_kwargs["daemon_address"] = env.get("daemon_address", args.daemon)
            if args.port == 10100:  # default
                common_kwargs["port"] = env.get("port", args.port)
            if args.threads == 0:  # default (auto)
                common_kwargs["threads"] = env.get("threads", args.threads)
            if args.wallet == "DERO_WALLET_PLACEHOLDER":
                common_kwargs["wallet"] = env.get("wallet", args.wallet)

            # Use benchmark settings from matrix
            bench = matrix.get("benchmark_settings", {})
            if args.runs == 3:  # default
                common_kwargs["runs"] = bench.get("runs_per_config", args.runs)
            if args.duration == 60:  # default
                common_kwargs["duration"] = bench.get("duration_seconds", args.duration)
            if args.warmup == 15:  # default
                common_kwargs["warmup"] = bench.get("warmup_seconds", args.warmup)
            if args.cooldown == 5:  # default
                common_kwargs["cooldown"] = bench.get("cooldown_seconds", args.cooldown)

            print(f"Loaded {len(configs)} configurations from {args.matrix}")
        else:
            configs = DEFAULT_CONFIGS

        # Run full sweep
        results = run_sweep(
            miner_path=miner_path,
            configs=configs,
            output_dir=args.output,
            **common_kwargs
        )
        print_summary(results)

    else:
        # Single config
        if args.config == "custom":
            flags = args.flags.split() if args.flags else []
            config_name = args.name if args.name else "custom"
        else:
            # Look up in defaults
            config_dict = dict(DEFAULT_CONFIGS)
            if args.config not in config_dict:
                print(f"ERROR: Unknown config '{args.config}'")
                print(f"Available: {', '.join(config_dict.keys())}")
                return 1
            flags = config_dict[args.config]
            config_name = args.config

        result = run_benchmark(
            miner_path=miner_path,
            config_name=config_name,
            flags=flags,
            **common_kwargs
        )
        save_result(result, args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
