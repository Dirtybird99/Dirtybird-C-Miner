#!/usr/bin/env python3
"""
Head-to-head benchmark: DirtyBird vs DeroLuna

Compares both miners under identical conditions to establish
performance gap and identify optimization targets.
"""

import subprocess
import re
import time
import json
import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional
import statistics

@dataclass
class MinerConfig:
    name: str
    exe_path: str
    build_cmd: List[str]
    hashrate_pattern: str
    dev_fee: float  # As decimal (0.10 = 10%)

# Miner configurations
MINERS = {
    "dirtybird": MinerConfig(
        name="DirtyBird",
        exe_path=r"[USERPROFILE]\Desktop\dero-miner-main\dero-miner-cpp\build-optimized\bin\dirtybird-miner-cpu.exe",
        build_cmd=[
            "--dero", "--no-lock", "--sa-tune",
            "--daemon-address", "{daemon}",
            "--port", "{port}",
            "--wallet", "{wallet}",
            "--threads", "{threads}",
            "--mine-time", "{duration}"
        ],
        hashrate_pattern=r'HASHRATE\s+(\d+\.?\d*)\s*KH/s',
        dev_fee=0.0
    ),
    "deroluna": MinerConfig(
        name="DeroLuna",
        exe_path=r"[USERPROFILE]\Desktop\dero-miner-main\releases\v1.14\deroluna-miner.exe",
        build_cmd=[
            "-d", "{daemon}:{port}",
            "-w", "{wallet}",
            "-t", "{threads}",
            "--no-cr"
        ],
        hashrate_pattern=r'@\s+(\d+\.?\d*)\s*KH/s',
        dev_fee=0.10
    )
}

def run_miner_benchmark(
    miner: MinerConfig,
    daemon: str,
    port: int,
    wallet: str,
    threads: int,
    duration: int,
    warmup: int
) -> dict:
    """Run a single miner benchmark and collect samples."""

    # Build command
    cmd = [miner.exe_path]
    for arg in miner.build_cmd:
        cmd.append(arg.format(
            daemon=daemon,
            port=port,
            wallet=wallet,
            threads=threads,
            duration=duration + warmup + 5  # Extra buffer
        ))

    print(f"  Running: {miner.name}")
    print(f"  Command: {' '.join(cmd[:6])}...")

    samples = []
    start_time = time.time()

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        pattern = re.compile(miner.hashrate_pattern, re.IGNORECASE)

        while True:
            line = process.stdout.readline()
            if not line:
                if process.poll() is not None:
                    break
                continue

            elapsed = time.time() - start_time

            # Parse hashrate
            match = pattern.search(line)
            if match:
                hr = float(match.group(1))

                # Skip warmup period
                if elapsed > warmup:
                    samples.append(hr)

                    # Print progress
                    if len(samples) % 10 == 0:
                        print(f"    {len(samples)} samples, current: {hr:.2f} KH/s")

                # Check if we have enough samples
                if elapsed > warmup + duration:
                    process.terminate()
                    break

        process.wait(timeout=5)

    except subprocess.TimeoutExpired:
        process.kill()
    except Exception as e:
        print(f"  Error: {e}")
        return None

    if len(samples) < 5:
        print(f"  WARNING: Only {len(samples)} samples collected")
        return None

    # Calculate statistics
    mean = statistics.mean(samples)
    stddev = statistics.stdev(samples) if len(samples) > 1 else 0
    cv = (stddev / mean * 100) if mean > 0 else 0

    result = {
        "miner": miner.name,
        "samples": samples,
        "mean": mean,
        "stddev": stddev,
        "cv": cv,
        "min": min(samples),
        "max": max(samples),
        "dev_fee": miner.dev_fee,
        "net_effective": mean * (1 - miner.dev_fee),
        "sample_count": len(samples)
    }

    print(f"  Results: {mean:.2f} KH/s (CV: {cv:.2f}%), Net: {result['net_effective']:.2f} KH/s")

    return result


def run_comparison(
    daemon: str = "203.0.113.10",
    port: int = 10100,
    wallet: str = "DERO_WALLET_PLACEHOLDER",
    threads: int = 24,
    duration: int = 60,
    warmup: int = 15,
    runs: int = 3,
    output_dir: str = None
):
    """Run head-to-head comparison of both miners."""

    print("=" * 70)
    print("MINER COMPARISON BENCHMARK")
    print("=" * 70)
    print(f"Daemon: {daemon}:{port}")
    print(f"Threads: {threads}")
    print(f"Duration: {duration}s (warmup: {warmup}s)")
    print(f"Runs per miner: {runs}")
    print("=" * 70)

    results = {name: [] for name in MINERS.keys()}

    for run in range(1, runs + 1):
        print(f"\n--- Run {run}/{runs} ---\n")

        for miner_name, miner_config in MINERS.items():
            result = run_miner_benchmark(
                miner_config,
                daemon, port, wallet, threads,
                duration, warmup
            )

            if result:
                results[miner_name].append(result)

            # Cooldown between miners
            print("  Cooldown: 10s...")
            time.sleep(10)

    # Aggregate results
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    summary = {}
    for miner_name, runs_data in results.items():
        if not runs_data:
            continue

        all_samples = []
        for r in runs_data:
            all_samples.extend(r["samples"])

        if not all_samples:
            continue

        mean = statistics.mean(all_samples)
        stddev = statistics.stdev(all_samples) if len(all_samples) > 1 else 0
        cv = (stddev / mean * 100) if mean > 0 else 0
        dev_fee = runs_data[0]["dev_fee"]

        summary[miner_name] = {
            "miner": runs_data[0]["miner"],
            "mean": mean,
            "stddev": stddev,
            "cv": cv,
            "min": min(all_samples),
            "max": max(all_samples),
            "dev_fee": dev_fee,
            "net_effective": mean * (1 - dev_fee),
            "total_samples": len(all_samples),
            "runs": len(runs_data)
        }

        print(f"\n{runs_data[0]['miner']}:")
        print(f"  Raw hashrate:  {mean:.2f} KH/s (±{stddev:.2f}, CV: {cv:.2f}%)")
        print(f"  Dev fee:       {dev_fee*100:.0f}%")
        print(f"  Net effective: {mean * (1 - dev_fee):.2f} KH/s")
        print(f"  Range:         {min(all_samples):.2f} - {max(all_samples):.2f} KH/s")

    # Comparison
    if "dirtybird" in summary and "deroluna" in summary:
        db = summary["dirtybird"]
        dl = summary["deroluna"]

        raw_diff = dl["mean"] - db["mean"]
        raw_pct = (raw_diff / db["mean"]) * 100

        net_diff = dl["net_effective"] - db["net_effective"]
        net_pct = (net_diff / db["net_effective"]) * 100

        print("\n" + "-" * 70)
        print("COMPARISON (DeroLuna vs DirtyBird):")
        print("-" * 70)
        print(f"  Raw hashrate gap:  {raw_diff:+.2f} KH/s ({raw_pct:+.1f}%)")
        print(f"  Net effective gap: {net_diff:+.2f} KH/s ({net_pct:+.1f}%)")
        print(f"  Stability (CV):    DirtyBird {db['cv']:.2f}% vs DeroLuna {dl['cv']:.2f}%")

        if net_diff > 0:
            print(f"\n  >> DeroLuna is {net_pct:.1f}% faster (after dev fee)")
        else:
            print(f"\n  >> DirtyBird is {-net_pct:.1f}% faster (no dev fee)")

    # Save results
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        output_file = output_path / "miner_comparison.json"
        with open(output_file, 'w') as f:
            json.dump({
                "config": {
                    "daemon": daemon,
                    "port": port,
                    "threads": threads,
                    "duration": duration,
                    "warmup": warmup,
                    "runs": runs
                },
                "results": summary,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
            }, f, indent=2)

        print(f"\nResults saved to: {output_file}")

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare DirtyBird vs DeroLuna miners")
    parser.add_argument("--daemon", default="203.0.113.10", help="Daemon address")
    parser.add_argument("--port", type=int, default=10100, help="Daemon port")
    parser.add_argument("--wallet", default="DERO_WALLET_PLACEHOLDER")
    parser.add_argument("--threads", type=int, default=24, help="Thread count")
    parser.add_argument("--duration", type=int, default=60, help="Test duration per run")
    parser.add_argument("--warmup", type=int, default=15, help="Warmup seconds")
    parser.add_argument("--runs", type=int, default=3, help="Runs per miner")
    parser.add_argument("--output", default="../benchmark_results", help="Output directory")

    args = parser.parse_args()

    run_comparison(
        daemon=args.daemon,
        port=args.port,
        wallet=args.wallet,
        threads=args.threads,
        duration=args.duration,
        warmup=args.warmup,
        runs=args.runs,
        output_dir=args.output
    )
