#!/usr/bin/env python3
"""
SA Prefetch Distance Sweep with --no-lock base
Tests various prefetch distance combinations to find optimal values.
"""

import subprocess
import sys
from pathlib import Path

# SA prefetch configurations to test (sa, text, bucket)
SA_CONFIGS = [
    ("no_lock_sa_8_8_4", ["--no-lock", "--sa-prefetch", "8,8,4"]),
    ("no_lock_sa_16_8_4", ["--no-lock", "--sa-prefetch", "16,8,4"]),
    ("no_lock_sa_16_16_8", ["--no-lock", "--sa-prefetch", "16,16,8"]),
    ("no_lock_sa_16_24_8", ["--no-lock", "--sa-prefetch", "16,24,8"]),
    ("no_lock_sa_24_16_8", ["--no-lock", "--sa-prefetch", "24,16,8"]),
    ("no_lock_sa_32_16_8", ["--no-lock", "--sa-prefetch", "32,16,8"]),
    ("no_lock_sa_32_24_16", ["--no-lock", "--sa-prefetch", "32,24,16"]),
    ("no_lock_sa_48_32_16", ["--no-lock", "--sa-prefetch", "48,32,16"]),
]

def main():
    script_dir = Path(__file__).parent
    bench_script = script_dir / "bench_optimize.py"
    output_dir = script_dir.parent / "benchmark_results"

    for config_name, flags in SA_CONFIGS:
        print(f"\n{'='*60}")
        print(f"Testing: {config_name}")
        print(f"Flags: {' '.join(flags)}")
        print(f"{'='*60}\n")

        cmd = [
            sys.executable, str(bench_script),
            "--config", "custom",
            "--flags"] + flags + [
            "--runs", "3",
            "--duration", "60",
            "--warmup", "15",
            "--threads", "20",
            "--output", str(output_dir)
        ]

        # Override config name by modifying the output
        result = subprocess.run(cmd, capture_output=False)

        if result.returncode != 0:
            print(f"WARNING: {config_name} failed")

    print("\n" + "="*60)
    print("SA PREFETCH SWEEP COMPLETE")
    print("="*60)
    print("\nRun: python analyze_results.py ../benchmark_results --report")

if __name__ == "__main__":
    main()
