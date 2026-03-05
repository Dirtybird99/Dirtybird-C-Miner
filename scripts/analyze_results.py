#!/usr/bin/env python3
"""
DirtyBird Benchmark Results Analyzer

Analyzes benchmark results and generates statistical reports.

Usage:
    python analyze_results.py benchmark_results/ --report
    python analyze_results.py benchmark_results/ --best
    python analyze_results.py benchmark_results/ --export results.csv
"""

import argparse
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class BenchResult:
    """Benchmark result loaded from JSON."""
    config_name: str
    flags: list[str]
    samples: list[float]
    mean: float
    stddev: float
    cv: float
    min_hr: float
    max_hr: float
    p5: float
    p95: float
    stability_ratio: float
    run_count: int
    duration: float
    warmup: float
    timestamp: str


def load_results(results_dir: Path) -> list[BenchResult]:
    """Load all benchmark results from a directory."""
    results = []

    for json_file in results_dir.glob("*.json"):
        # Skip sample files
        if "_samples" in json_file.name:
            continue

        try:
            with open(json_file) as f:
                data = json.load(f)

            result = BenchResult(
                config_name=data.get("config_name", json_file.stem),
                flags=data.get("flags", []),
                samples=data.get("samples", []),
                mean=data.get("mean", 0),
                stddev=data.get("stddev", 0),
                cv=data.get("cv", 0),
                min_hr=data.get("min_hr", 0),
                max_hr=data.get("max_hr", 0),
                p5=data.get("p5", 0),
                p95=data.get("p95", 0),
                stability_ratio=data.get("stability_ratio", 0),
                run_count=data.get("run_count", 0),
                duration=data.get("duration", 0),
                warmup=data.get("warmup", 0),
                timestamp=data.get("timestamp", ""),
            )
            results.append(result)
        except Exception as e:
            print(f"Warning: Failed to load {json_file}: {e}")

    return results


def welch_t_test(samples1: list[float], samples2: list[float]) -> tuple[float, float, float]:
    """
    Perform Welch's t-test.
    Returns: (t_statistic, p_value, degrees_of_freedom)
    """
    n1, n2 = len(samples1), len(samples2)
    if n1 < 2 or n2 < 2:
        return 0, 1, 0

    mean1, mean2 = statistics.mean(samples1), statistics.mean(samples2)
    var1, var2 = statistics.variance(samples1), statistics.variance(samples2)

    se = math.sqrt(var1/n1 + var2/n2)
    if se == 0:
        return 0, 1, 0

    t = (mean1 - mean2) / se

    # Welch-Satterthwaite degrees of freedom
    num = (var1/n1 + var2/n2) ** 2
    denom = (var1/n1)**2 / (n1-1) + (var2/n2)**2 / (n2-1)
    df = num / denom if denom > 0 else 1

    # Approximate p-value
    abs_t = abs(t)
    if df > 30:
        z = abs_t / math.sqrt(2)
        erf = 1 - math.exp(-z * z * (1.273239544735163 + 0.14001228868667 * z * z) /
                          (1 + 0.14001228868667 * z * z))
        p_value = 1 - erf
    else:
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


def cohen_d(samples1: list[float], samples2: list[float]) -> float:
    """Calculate Cohen's d effect size."""
    if len(samples1) < 2 or len(samples2) < 2:
        return 0

    mean1, mean2 = statistics.mean(samples1), statistics.mean(samples2)
    std1, std2 = statistics.stdev(samples1), statistics.stdev(samples2)

    pooled_std = math.sqrt((std1**2 + std2**2) / 2)
    return (mean1 - mean2) / pooled_std if pooled_std > 0 else 0


def confidence_interval(samples: list[float], confidence: float = 0.95) -> tuple[float, float]:
    """Calculate confidence interval for the mean."""
    if len(samples) < 2:
        mean = samples[0] if samples else 0
        return mean, mean

    n = len(samples)
    mean = statistics.mean(samples)
    se = statistics.stdev(samples) / math.sqrt(n)

    # Z-score for 95% confidence
    z = 1.96 if confidence == 0.95 else 2.576  # 99%

    return mean - z * se, mean + z * se


def print_report(results: list[BenchResult], baseline_name: Optional[str] = None):
    """Print detailed analysis report."""
    if not results:
        print("No results to analyze.")
        return

    # Sort by mean hashrate
    sorted_results = sorted(results, key=lambda r: r.mean, reverse=True)

    # Find baseline
    if baseline_name:
        baseline = next((r for r in results if r.config_name == baseline_name), None)
    else:
        # Use the one with "baseline" in name, or first result
        baseline = next((r for r in results if "baseline" in r.config_name.lower()), results[0])

    print("\n" + "=" * 90)
    print("DIRTYBIRD OPTIMIZATION ANALYSIS REPORT")
    print("=" * 90)

    # Summary table
    print(f"\n{'CONFIGURATION SUMMARY':^90}")
    print("-" * 90)
    print(f"{'Config':<25} {'Mean':>8} {'StdDev':>7} {'CV':>7} {'Min':>7} {'Max':>7} {'P5':>7} {'P95':>7}")
    print("-" * 90)

    for r in sorted_results:
        print(f"{r.config_name:<25} {r.mean:>7.2f} {r.stddev:>6.2f} {r.cv:>6.2f}% {r.min_hr:>6.2f} {r.max_hr:>6.2f} {r.p5:>6.2f} {r.p95:>6.2f}")

    # Statistical comparison vs baseline
    print(f"\n{'STATISTICAL COMPARISON vs ' + baseline.config_name:^90}")
    print("-" * 90)
    print(f"{'Config':<25} {'Diff':>8} {'%Diff':>8} {'t-stat':>8} {'p-value':>10} {'d':>6} {'Sig':>5}")
    print("-" * 90)

    for r in sorted_results:
        if r.config_name == baseline.config_name:
            print(f"{r.config_name:<25} {'--':>8} {'--':>8} {'--':>8} {'--':>10} {'--':>6} {'--':>5}")
            continue

        diff = r.mean - baseline.mean
        pct_diff = (diff / baseline.mean * 100) if baseline.mean > 0 else 0
        t, p, df = welch_t_test(r.samples, baseline.samples)
        d = cohen_d(r.samples, baseline.samples)
        sig = "*" if p < 0.05 else ""
        sig += "*" if p < 0.01 else ""

        print(f"{r.config_name:<25} {diff:>+7.2f} {pct_diff:>+7.2f}% {t:>8.2f} {p:>10.4f} {d:>6.2f} {sig:>5}")

    print("\n* p < 0.05, ** p < 0.01")

    # Best configurations
    print(f"\n{'TOP CONFIGURATIONS':^90}")
    print("-" * 90)

    for i, r in enumerate(sorted_results[:3], 1):
        ci_low, ci_high = confidence_interval(r.samples)
        print(f"\n{i}. {r.config_name}")
        print(f"   Flags: {' '.join(r.flags) if r.flags else '(default)'}")
        print(f"   Mean: {r.mean:.2f} KH/s (95% CI: {ci_low:.2f} - {ci_high:.2f})")
        print(f"   CV: {r.cv:.2f}%, Stability: {r.stability_ratio:.2%}")

    # Stability analysis
    print(f"\n{'STABILITY ANALYSIS':^90}")
    print("-" * 90)

    stable_configs = [r for r in sorted_results if r.cv < 5]
    moderate_configs = [r for r in sorted_results if 5 <= r.cv < 10]
    unstable_configs = [r for r in sorted_results if r.cv >= 10]

    print(f"\nExcellent (CV < 5%): {len(stable_configs)}")
    for r in stable_configs:
        print(f"  - {r.config_name}: {r.cv:.2f}%")

    print(f"\nGood (5% <= CV < 10%): {len(moderate_configs)}")
    for r in moderate_configs:
        print(f"  - {r.config_name}: {r.cv:.2f}%")

    print(f"\nPoor (CV >= 10%): {len(unstable_configs)}")
    for r in unstable_configs:
        print(f"  - {r.config_name}: {r.cv:.2f}%")

    # Recommendations
    print(f"\n{'RECOMMENDATIONS':^90}")
    print("-" * 90)

    best = sorted_results[0]
    most_stable = min(sorted_results, key=lambda r: r.cv)

    print(f"\nHighest throughput: {best.config_name}")
    print(f"  {best.mean:.2f} KH/s, CV: {best.cv:.2f}%")

    print(f"\nMost stable: {most_stable.config_name}")
    print(f"  {most_stable.mean:.2f} KH/s, CV: {most_stable.cv:.2f}%")

    # Best balanced (high hashrate + low CV)
    # Score = mean * (1 - cv/100)
    scored = [(r, r.mean * (1 - r.cv/100)) for r in sorted_results]
    best_balanced = max(scored, key=lambda x: x[1])[0]

    print(f"\nBest balanced: {best_balanced.config_name}")
    print(f"  {best_balanced.mean:.2f} KH/s, CV: {best_balanced.cv:.2f}%")

    print("\n" + "=" * 90)


def print_best(results: list[BenchResult]):
    """Print just the best configuration."""
    if not results:
        print("No results found.")
        return

    best = max(results, key=lambda r: r.mean)

    print(f"\nBest configuration: {best.config_name}")
    print(f"Flags: {' '.join(best.flags) if best.flags else '(default)'}")
    print(f"Mean hashrate: {best.mean:.2f} KH/s")
    print(f"CV: {best.cv:.2f}%")
    print(f"Range: {best.min_hr:.2f} - {best.max_hr:.2f} KH/s")


def export_csv(results: list[BenchResult], output_path: Path):
    """Export results to CSV."""
    with open(output_path, 'w') as f:
        # Header
        f.write("config_name,flags,mean,stddev,cv,min_hr,max_hr,p5,p95,stability_ratio,samples,timestamp\n")

        for r in results:
            flags = " ".join(r.flags).replace(",", ";")
            f.write(f"{r.config_name},{flags},{r.mean:.4f},{r.stddev:.4f},{r.cv:.4f},")
            f.write(f"{r.min_hr:.4f},{r.max_hr:.4f},{r.p5:.4f},{r.p95:.4f},{r.stability_ratio:.4f},")
            f.write(f"{len(r.samples)},{r.timestamp}\n")

    print(f"Exported to: {output_path}")


def export_best_config(results: list[BenchResult], output_path: Path):
    """Export best configuration to JSON for use in config.json."""
    if not results:
        return

    best = max(results, key=lambda r: r.mean)
    most_stable = min(results, key=lambda r: r.cv)

    config = {
        "optimization_results": {
            "best_throughput": {
                "config_name": best.config_name,
                "flags": best.flags,
                "mean_hashrate": best.mean,
                "cv_percent": best.cv
            },
            "most_stable": {
                "config_name": most_stable.config_name,
                "flags": most_stable.flags,
                "mean_hashrate": most_stable.mean,
                "cv_percent": most_stable.cv
            },
            "recommended_flags": best.flags if best.cv < 10 else most_stable.flags
        }
    }

    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Best config exported to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze DirtyBird benchmark results")
    parser.add_argument("results_dir", type=Path, help="Directory containing result JSON files")
    parser.add_argument("--report", action="store_true", help="Print detailed analysis report")
    parser.add_argument("--best", action="store_true", help="Print just the best configuration")
    parser.add_argument("--export", type=Path, metavar="FILE", help="Export results to CSV")
    parser.add_argument("--export-config", type=Path, metavar="FILE", help="Export best config to JSON")
    parser.add_argument("--baseline", help="Baseline config name for comparisons")

    args = parser.parse_args()

    if not args.results_dir.exists():
        print(f"Error: Directory not found: {args.results_dir}")
        return 1

    results = load_results(args.results_dir)

    if not results:
        print(f"No results found in {args.results_dir}")
        return 1

    print(f"Loaded {len(results)} benchmark results")

    if args.report:
        print_report(results, args.baseline)

    if args.best:
        print_best(results)

    if args.export:
        export_csv(results, args.export)

    if args.export_config:
        export_best_config(results, args.export_config)

    # Default: show summary
    if not (args.report or args.best or args.export or args.export_config):
        print_report(results, args.baseline)

    return 0


if __name__ == "__main__":
    sys.exit(main())
