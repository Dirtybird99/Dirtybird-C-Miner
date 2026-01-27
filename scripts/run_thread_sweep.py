#!/usr/bin/env python3
"""
Thread Count Sweep for Hybrid CPU (P-core vs E-core optimization)

Intel 13700HX: 8 P-cores (16 threads) + 8 E-cores (8 threads) = 24 total
Hypothesis: Mining on E-cores hurts more than it helps due to:
  - 40% lower IPC than P-cores
  - Shared L3 contention
  - Memory bandwidth saturation

Tests thread counts to find optimal P-core utilization.
"""

import subprocess
import sys
from pathlib import Path

# Thread counts to test:
# - 8: P-cores only (1 thread per core)
# - 12: P-cores with some SMT
# - 16: All P-core threads (no E-cores)
# - 20: Current setting (some E-cores)
# - 24: All threads (full hybrid)
THREAD_CONFIGS = [
    ("threads_8_pcore_only", ["--no-lock", "--sa-tune"], 8),
    ("threads_12_pcore_smt", ["--no-lock", "--sa-tune"], 12),
    ("threads_16_pcore_full", ["--no-lock", "--sa-tune"], 16),
    ("threads_20_current", ["--no-lock", "--sa-tune"], 20),
    ("threads_24_all", ["--no-lock", "--sa-tune"], 24),
]

def main():
    script_dir = Path(__file__).parent
    bench_script = script_dir / "bench_optimize.py"
    output_dir = script_dir.parent / "benchmark_results"

    print("=" * 70)
    print("HYBRID CPU THREAD OPTIMIZATION SWEEP")
    print("Intel 13700HX: 8P + 8E cores")
    print("=" * 70)
    print("\nHypothesis: E-cores may hurt mining performance")
    print("Testing: 8, 12, 16, 20, 24 threads\n")

    for config_name, flags, threads in THREAD_CONFIGS:
        print(f"\n{'='*60}")
        print(f"Testing: {config_name} ({threads} threads)")
        print(f"Flags: {' '.join(flags)}")
        print(f"{'='*60}\n")

        cmd = [
            sys.executable, str(bench_script),
            "--config", "custom",
            "--name", config_name,
            "--flags", " ".join(flags),
            "--runs", "3",
            "--duration", "60",
            "--warmup", "15",
            "--threads", str(threads),
            "--output", str(output_dir)
        ]

        result = subprocess.run(cmd, capture_output=False)

        if result.returncode != 0:
            print(f"WARNING: {config_name} failed")

    print("\n" + "=" * 70)
    print("THREAD SWEEP COMPLETE")
    print("=" * 70)
    print("\nRun: python analyze_results.py ../benchmark_results --report")
    print("\nLook for:")
    print("  - Highest mean hashrate")
    print("  - Lowest CV (most stable)")
    print("  - Best hashrate-per-thread efficiency")

if __name__ == "__main__":
    main()
