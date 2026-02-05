#!/usr/bin/env python3
"""
Unified Benchmark Runner

Runs Axiom benchmarks and comparison benchmarks, outputting JSON results.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any

# Add parent directory for imports
BENCHMARK_DIR = Path(__file__).parent.parent
BUILD_DIR = BENCHMARK_DIR.parent / "build"


@dataclass
class BenchmarkResult:
    """Single benchmark result."""
    name: str
    time_ms: float
    gflops: Optional[float] = None
    bandwidth_gbps: Optional[float] = None
    iterations: int = 1
    extra: Optional[Dict[str, Any]] = None


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results."""
    name: str
    timestamp: str
    platform: str
    results: List[BenchmarkResult]
    metadata: Optional[Dict[str, Any]] = None


def run_command(cmd: List[str], timeout: int = 300) -> tuple:
    """Run a command and return (success, stdout, stderr)."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.returncode == 0, result.stdout.strip(), result.stderr.strip()
    except subprocess.TimeoutExpired:
        return False, "", "timeout"
    except Exception as e:
        return False, "", str(e)


def get_platform_info() -> str:
    """Get platform information string."""
    import platform
    return f"{platform.system()} {platform.machine()}"


def run_axiom_benchmark(binary: str, args: List[str] = None) -> List[BenchmarkResult]:
    """Run an Axiom benchmark binary and parse results."""
    binary_path = BUILD_DIR / "benchmarks" / binary
    if not binary_path.exists():
        print(f"Warning: {binary_path} not found")
        return []

    cmd = [str(binary_path)]
    if args:
        cmd.extend(args)

    # Add JSON output
    cmd.extend(["--benchmark_format=json"])

    success, stdout, stderr = run_command(cmd)
    if not success:
        print(f"Error running {binary}: {stderr}")
        return []

    try:
        data = json.loads(stdout)
        results = []
        for bench in data.get("benchmarks", []):
            name = bench.get("name", "unknown")
            # Google Benchmark reports real_time in nanoseconds by default
            time_ns = bench.get("real_time", 0)
            time_unit = bench.get("time_unit", "ns")

            # Convert to milliseconds
            if time_unit == "ns":
                time_ms = time_ns / 1e6
            elif time_unit == "us":
                time_ms = time_ns / 1e3
            elif time_unit == "ms":
                time_ms = time_ns
            else:
                time_ms = time_ns

            extra = {}
            if "GFLOPS" in bench:
                extra["gflops"] = bench["GFLOPS"]
            if "Bandwidth" in bench:
                extra["bandwidth"] = bench["Bandwidth"]

            results.append(BenchmarkResult(
                name=name,
                time_ms=time_ms,
                gflops=extra.get("gflops"),
                iterations=bench.get("iterations", 1),
                extra=extra if extra else None
            ))
        return results
    except json.JSONDecodeError:
        print(f"Failed to parse JSON from {binary}")
        return []


def run_comparison_benchmarks(sizes: List[int], output_dir: Path) -> Dict[str, List[float]]:
    """Run comparison benchmarks across all libraries."""
    results = {
        "sizes": sizes,
        "axiom_cpu": [],
        "axiom_gpu": [],
        "eigen": [],
        "armadillo": [],
        "numpy": [],
        "pytorch_cpu": [],
        "pytorch_mps": [],
    }

    compare_dir = BENCHMARK_DIR / "compare"
    build_compare = BUILD_DIR / "benchmarks" / "compare"

    # Check for numpy baseline
    numpy_baseline = BENCHMARK_DIR / "tools" / "baselines" / "numpy_baseline.py"

    for size in sizes:
        print(f"  Size {size}x{size}...")

        # Axiom CPU
        axiom_cpu = build_compare / "matmul_axiom"
        if axiom_cpu.exists():
            success, stdout, _ = run_command([str(axiom_cpu), str(size)])
            try:
                results["axiom_cpu"].append(float(stdout) if success else None)
            except ValueError:
                results["axiom_cpu"].append(None)
        else:
            results["axiom_cpu"].append(None)

        # Axiom GPU
        axiom_gpu = build_compare / "matmul_axiom_gpu"
        if axiom_gpu.exists():
            success, stdout, _ = run_command([str(axiom_gpu), str(size)])
            try:
                results["axiom_gpu"].append(float(stdout) if success else None)
            except ValueError:
                results["axiom_gpu"].append(None)
        else:
            results["axiom_gpu"].append(None)

        # Eigen
        eigen_bin = build_compare / "matmul_eigen"
        if eigen_bin.exists():
            success, stdout, _ = run_command([str(eigen_bin), str(size)])
            try:
                results["eigen"].append(float(stdout) if success else None)
            except ValueError:
                results["eigen"].append(None)
        else:
            results["eigen"].append(None)

        # Armadillo
        arma_bin = build_compare / "matmul_armadillo"
        if arma_bin.exists():
            success, stdout, _ = run_command([str(arma_bin), str(size)])
            try:
                results["armadillo"].append(float(stdout) if success else None)
            except ValueError:
                results["armadillo"].append(None)
        else:
            results["armadillo"].append(None)

        # NumPy
        if numpy_baseline.exists():
            success, stdout, _ = run_command(
                ["python3", str(numpy_baseline), "numpy", str(size)]
            )
            try:
                results["numpy"].append(float(stdout) if success else None)
            except ValueError:
                results["numpy"].append(None)
        else:
            results["numpy"].append(None)

        # PyTorch CPU
        if numpy_baseline.exists():
            success, stdout, _ = run_command(
                ["python3", str(numpy_baseline), "pytorch", str(size)]
            )
            try:
                val = float(stdout) if success else None
                results["pytorch_cpu"].append(val if val and val > 0 else None)
            except ValueError:
                results["pytorch_cpu"].append(None)
        else:
            results["pytorch_cpu"].append(None)

        # PyTorch MPS
        if numpy_baseline.exists():
            success, stdout, _ = run_command(
                ["python3", str(numpy_baseline), "pytorch_mps", str(size)]
            )
            try:
                val = float(stdout) if success else None
                results["pytorch_mps"].append(val if val and val > 0 else None)
            except ValueError:
                results["pytorch_mps"].append(None)
        else:
            results["pytorch_mps"].append(None)

    return results


def run_suite(suite: str, output_dir: Path) -> BenchmarkSuite:
    """Run a benchmark suite."""
    import datetime

    results = []
    metadata = {}

    if suite == "gemm" or suite == "all":
        print("Running GEMM benchmarks...")
        gemm_results = run_axiom_benchmark("bench_gemm")
        results.extend(gemm_results)

    if suite == "simd" or suite == "all":
        print("Running SIMD kernel benchmarks...")
        simd_results = run_axiom_benchmark("bench_simd_kernels")
        results.extend(simd_results)

    if suite == "fusion" or suite == "all":
        print("Running fusion benchmarks...")
        fusion_results = run_axiom_benchmark("bench_fusion")
        results.extend(fusion_results)

    return BenchmarkSuite(
        name=suite,
        timestamp=datetime.datetime.now().isoformat(),
        platform=get_platform_info(),
        results=results,
        metadata=metadata
    )


def main():
    parser = argparse.ArgumentParser(
        description="Unified Axiom benchmark runner"
    )
    parser.add_argument(
        "--suite",
        choices=["gemm", "simd", "fusion", "all"],
        default="all",
        help="Benchmark suite to run"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run library comparison benchmarks"
    )
    parser.add_argument(
        "--sizes",
        type=str,
        default="32,64,128,256,512,1024,2048,4096",
        help="Matrix sizes for comparison (comma-separated)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(BENCHMARK_DIR / "results"),
        help="Output directory for results"
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.compare:
        sizes = [int(s.strip()) for s in args.sizes.split(",")]
        print(f"Running comparison benchmarks for sizes: {sizes}")
        results = run_comparison_benchmarks(sizes, output_dir)

        output_file = output_dir / "comparison_results.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {output_file}")
    else:
        suite = run_suite(args.suite, output_dir)

        output_file = output_dir / f"{args.suite}_results.json"
        with open(output_file, "w") as f:
            # Convert dataclasses to dicts
            suite_dict = asdict(suite)
            json.dump(suite_dict, f, indent=2, default=str)
        print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
