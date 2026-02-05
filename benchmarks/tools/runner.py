#!/usr/bin/env python3
"""
Unified Benchmark Runner

Runs Axiom benchmarks and comparison benchmarks across multiple categories,
outputting JSON results for plotting.

Categories:
- matmul: Matrix multiplication
- elementwise: Binary operations (add, sub, mul, div)
- unary: Unary operations (exp, log, sqrt, sin, cos, tanh, etc.)
- linalg: Linear algebra (svd, qr, solve, cholesky, eig, inv, det)
- fft: FFT operations (fft, ifft, fft2, ifft2, rfft, rfft2)
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
            time_ns = bench.get("real_time", 0)
            time_unit = bench.get("time_unit", "ns")

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


# ============================================================================
# Matmul Comparison Benchmarks
# ============================================================================

def run_matmul_comparison(sizes: List[int], output_dir: Path) -> Dict[str, Any]:
    """Run matmul comparison benchmarks across all libraries (CPU only)."""
    results = {
        "category": "matmul",
        "metric": "GFLOPS",
        "sizes": sizes,
        "axiom": [],
        "eigen": [],
        "armadillo": [],
        "numpy": [],
        "pytorch": [],
    }

    compare_dir = BUILD_DIR / "benchmarks" / "compare"
    baseline_script = BENCHMARK_DIR / "tools" / "baselines" / "numpy_baseline.py"

    for size in sizes:
        print(f"  Matmul {size}x{size}...")

        # Axiom CPU
        binary = compare_dir / "matmul_axiom"
        if binary.exists():
            success, stdout, _ = run_command([str(binary), str(size)])
            try:
                results["axiom"].append(float(stdout) if success else None)
            except ValueError:
                results["axiom"].append(None)
        else:
            results["axiom"].append(None)

        # Eigen
        binary = compare_dir / "matmul_eigen"
        if binary.exists():
            success, stdout, _ = run_command([str(binary), str(size)])
            try:
                results["eigen"].append(float(stdout) if success else None)
            except ValueError:
                results["eigen"].append(None)
        else:
            results["eigen"].append(None)

        # Armadillo
        binary = compare_dir / "matmul_armadillo"
        if binary.exists():
            success, stdout, _ = run_command([str(binary), str(size)])
            try:
                results["armadillo"].append(float(stdout) if success else None)
            except ValueError:
                results["armadillo"].append(None)
        else:
            results["armadillo"].append(None)

        # NumPy
        if baseline_script.exists():
            success, stdout, _ = run_command(
                [sys.executable, str(baseline_script), "numpy", str(size)]
            )
            try:
                results["numpy"].append(float(stdout) if success else None)
            except ValueError:
                results["numpy"].append(None)
        else:
            results["numpy"].append(None)

        # PyTorch CPU
        if baseline_script.exists():
            success, stdout, _ = run_command(
                [sys.executable, str(baseline_script), "pytorch", str(size)]
            )
            try:
                val = float(stdout) if success else None
                results["pytorch"].append(val if val and val > 0 else None)
            except ValueError:
                results["pytorch"].append(None)
        else:
            results["pytorch"].append(None)

    return results


# ============================================================================
# Element-wise Comparison Benchmarks
# ============================================================================

def run_elementwise_comparison(sizes: List[int], output_dir: Path) -> Dict[str, Any]:
    """Run element-wise binary op comparison benchmarks."""
    ops = ["add", "sub", "mul", "div"]
    results = {
        "category": "elementwise",
        "metric": "GB/s",
        "sizes": sizes,
        "ops": ops,
    }

    # Initialize result arrays for each library (CPU only)
    for lib in ["axiom", "eigen", "numpy", "pytorch"]:
        results[lib] = {op: [] for op in ops}

    compare_dir = BUILD_DIR / "benchmarks" / "compare"
    baseline_script = BENCHMARK_DIR / "tools" / "baselines" / "comprehensive_baseline.py"

    for size in sizes:
        print(f"  Elementwise {size}x{size}...")

        # Axiom
        binary = compare_dir / "elementwise_axiom"
        if binary.exists():
            success, stdout, _ = run_command([str(binary), "all", str(size)])
            if success:
                try:
                    data = json.loads(stdout)
                    for op in ops:
                        results["axiom"][op].append(data.get(op))
                except (json.JSONDecodeError, KeyError):
                    for op in ops:
                        results["axiom"][op].append(None)
            else:
                for op in ops:
                    results["axiom"][op].append(None)
        else:
            for op in ops:
                results["axiom"][op].append(None)

        # Eigen
        binary = compare_dir / "elementwise_eigen"
        if binary.exists():
            success, stdout, _ = run_command([str(binary), "all", str(size)])
            if success:
                try:
                    data = json.loads(stdout)
                    for op in ops:
                        results["eigen"][op].append(data.get(op))
                except (json.JSONDecodeError, KeyError):
                    for op in ops:
                        results["eigen"][op].append(None)
            else:
                for op in ops:
                    results["eigen"][op].append(None)
        else:
            for op in ops:
                results["eigen"][op].append(None)

        # Python baselines (CPU only)
        if baseline_script.exists():
            for lib, py_lib in [("numpy", "numpy"), ("pytorch", "pytorch")]:
                success, stdout, _ = run_command(
                    [sys.executable, str(baseline_script), "elementwise", py_lib, str(size)]
                )
                if success:
                    try:
                        data = json.loads(stdout)
                        for op in ops:
                            val = data.get(op)
                            results[lib][op].append(val if val and val > 0 else None)
                    except (json.JSONDecodeError, KeyError):
                        for op in ops:
                            results[lib][op].append(None)
                else:
                    for op in ops:
                        results[lib][op].append(None)
        else:
            for lib in ["numpy", "pytorch"]:
                for op in ops:
                    results[lib][op].append(None)

    return results


# ============================================================================
# Unary Operation Comparison Benchmarks
# ============================================================================

def run_unary_comparison(sizes: List[int], output_dir: Path) -> Dict[str, Any]:
    """Run unary operation comparison benchmarks."""
    ops = ["exp", "log", "sqrt", "sin", "cos", "tanh", "abs", "neg", "relu", "sigmoid"]
    results = {
        "category": "unary",
        "metric": "GB/s",
        "sizes": sizes,
        "ops": ops,
    }

    # CPU only
    for lib in ["axiom", "eigen", "numpy", "pytorch"]:
        results[lib] = {op: [] for op in ops}

    compare_dir = BUILD_DIR / "benchmarks" / "compare"
    baseline_script = BENCHMARK_DIR / "tools" / "baselines" / "comprehensive_baseline.py"

    for size in sizes:
        print(f"  Unary {size}x{size}...")

        # Axiom
        binary = compare_dir / "unary_axiom"
        if binary.exists():
            success, stdout, _ = run_command([str(binary), "all", str(size)])
            if success:
                try:
                    data = json.loads(stdout)
                    for op in ops:
                        results["axiom"][op].append(data.get(op))
                except (json.JSONDecodeError, KeyError):
                    for op in ops:
                        results["axiom"][op].append(None)
            else:
                for op in ops:
                    results["axiom"][op].append(None)
        else:
            for op in ops:
                results["axiom"][op].append(None)

        # Eigen
        binary = compare_dir / "unary_eigen"
        if binary.exists():
            success, stdout, _ = run_command([str(binary), "all", str(size)])
            if success:
                try:
                    data = json.loads(stdout)
                    for op in ops:
                        results["eigen"][op].append(data.get(op))
                except (json.JSONDecodeError, KeyError):
                    for op in ops:
                        results["eigen"][op].append(None)
            else:
                for op in ops:
                    results["eigen"][op].append(None)
        else:
            for op in ops:
                results["eigen"][op].append(None)

        # Python baselines (CPU only)
        if baseline_script.exists():
            for lib, py_lib in [("numpy", "numpy"), ("pytorch", "pytorch")]:
                success, stdout, _ = run_command(
                    [sys.executable, str(baseline_script), "unary", py_lib, str(size)]
                )
                if success:
                    try:
                        data = json.loads(stdout)
                        for op in ops:
                            val = data.get(op)
                            results[lib][op].append(val if val and val > 0 else None)
                    except (json.JSONDecodeError, KeyError):
                        for op in ops:
                            results[lib][op].append(None)
                else:
                    for op in ops:
                        results[lib][op].append(None)
        else:
            for lib in ["numpy", "pytorch"]:
                for op in ops:
                    results[lib][op].append(None)

    return results


# ============================================================================
# Linear Algebra Comparison Benchmarks
# ============================================================================

def run_linalg_comparison(sizes: List[int], output_dir: Path) -> Dict[str, Any]:
    """Run linear algebra comparison benchmarks."""
    ops = ["svd", "qr", "solve", "cholesky", "eig", "inv", "det"]
    results = {
        "category": "linalg",
        "metric": "time_ms",
        "sizes": sizes,
        "ops": ops,
    }

    # CPU only
    for lib in ["axiom", "eigen", "numpy", "pytorch"]:
        results[lib] = {op: [] for op in ops}

    compare_dir = BUILD_DIR / "benchmarks" / "compare"
    baseline_script = BENCHMARK_DIR / "tools" / "baselines" / "comprehensive_baseline.py"

    for size in sizes:
        print(f"  Linalg {size}x{size}...")

        # Axiom
        binary = compare_dir / "linalg_axiom"
        if binary.exists():
            success, stdout, _ = run_command([str(binary), "all", str(size)], timeout=600)
            if success:
                try:
                    data = json.loads(stdout)
                    for op in ops:
                        if op in data and isinstance(data[op], dict):
                            results["axiom"][op].append(data[op].get("time_ms"))
                        else:
                            results["axiom"][op].append(None)
                except (json.JSONDecodeError, KeyError):
                    for op in ops:
                        results["axiom"][op].append(None)
            else:
                for op in ops:
                    results["axiom"][op].append(None)
        else:
            for op in ops:
                results["axiom"][op].append(None)

        # Eigen
        binary = compare_dir / "linalg_eigen"
        if binary.exists():
            success, stdout, _ = run_command([str(binary), "all", str(size)], timeout=600)
            if success:
                try:
                    data = json.loads(stdout)
                    for op in ops:
                        if op in data and isinstance(data[op], dict):
                            results["eigen"][op].append(data[op].get("time_ms"))
                        else:
                            results["eigen"][op].append(None)
                except (json.JSONDecodeError, KeyError):
                    for op in ops:
                        results["eigen"][op].append(None)
            else:
                for op in ops:
                    results["eigen"][op].append(None)
        else:
            for op in ops:
                results["eigen"][op].append(None)

        # Python baselines (CPU only)
        if baseline_script.exists():
            for lib, py_lib in [("numpy", "numpy"), ("pytorch", "pytorch")]:
                success, stdout, _ = run_command(
                    [sys.executable, str(baseline_script), "linalg", py_lib, str(size)],
                    timeout=600
                )
                if success:
                    try:
                        data = json.loads(stdout)
                        for op in ops:
                            val = data.get(op)
                            results[lib][op].append(val if val and val > 0 else None)
                    except (json.JSONDecodeError, KeyError):
                        for op in ops:
                            results[lib][op].append(None)
                else:
                    for op in ops:
                        results[lib][op].append(None)
        else:
            for lib in ["numpy", "pytorch"]:
                for op in ops:
                    results[lib][op].append(None)

    return results


# ============================================================================
# FFT Comparison Benchmarks
# ============================================================================

def run_fft_comparison(sizes: List[int], output_dir: Path) -> Dict[str, Any]:
    """Run FFT comparison benchmarks."""
    ops = ["fft", "ifft", "rfft", "fft2", "ifft2", "rfft2"]
    results = {
        "category": "fft",
        "metric": "time_ms",
        "sizes": sizes,
        "ops": ops,
    }

    # CPU only
    for lib in ["axiom", "numpy", "pytorch"]:
        results[lib] = {op: [] for op in ops}

    compare_dir = BUILD_DIR / "benchmarks" / "compare"
    baseline_script = BENCHMARK_DIR / "tools" / "baselines" / "comprehensive_baseline.py"

    for size in sizes:
        print(f"  FFT {size}...")

        # Axiom
        binary = compare_dir / "fft_axiom"
        if binary.exists():
            success, stdout, _ = run_command([str(binary), "all", str(size)])
            if success:
                try:
                    data = json.loads(stdout)
                    for op in ops:
                        if op in data and isinstance(data[op], dict):
                            results["axiom"][op].append(data[op].get("time_ms"))
                        else:
                            results["axiom"][op].append(None)
                except (json.JSONDecodeError, KeyError):
                    for op in ops:
                        results["axiom"][op].append(None)
            else:
                for op in ops:
                    results["axiom"][op].append(None)
        else:
            for op in ops:
                results["axiom"][op].append(None)

        # Python baselines (CPU only)
        if baseline_script.exists():
            for lib, py_lib in [("numpy", "numpy"), ("pytorch", "pytorch")]:
                success, stdout, _ = run_command(
                    [sys.executable, str(baseline_script), "fft", py_lib, str(size)]
                )
                if success:
                    try:
                        data = json.loads(stdout)
                        for op in ops:
                            val = data.get(op)
                            results[lib][op].append(val if val and val > 0 else None)
                    except (json.JSONDecodeError, KeyError):
                        for op in ops:
                            results[lib][op].append(None)
                else:
                    for op in ops:
                        results[lib][op].append(None)
        else:
            for lib in ["numpy", "pytorch"]:
                for op in ops:
                    results[lib][op].append(None)

    return results


# ============================================================================
# Comprehensive Comparison (All Categories)
# ============================================================================

def run_comparison_benchmarks(sizes: List[int], output_dir: Path,
                              categories: List[str] = None) -> Dict[str, Any]:
    """Run comparison benchmarks across selected categories."""
    all_categories = ["matmul", "elementwise", "unary", "linalg", "fft"]
    if categories is None:
        categories = all_categories

    results = {
        "timestamp": __import__("datetime").datetime.now().isoformat(),
        "platform": get_platform_info(),
        "sizes": sizes,
        "categories": {},
    }

    # Different size ranges for different benchmark types
    # Linalg is slower, so use smaller sizes
    linalg_sizes = [s for s in sizes if s <= 512]
    fft_sizes = [s for s in sizes if s <= 2048]

    for cat in categories:
        print(f"\nRunning {cat} benchmarks...")
        if cat == "matmul":
            results["categories"]["matmul"] = run_matmul_comparison(sizes, output_dir)
        elif cat == "elementwise":
            results["categories"]["elementwise"] = run_elementwise_comparison(sizes, output_dir)
        elif cat == "unary":
            results["categories"]["unary"] = run_unary_comparison(sizes, output_dir)
        elif cat == "linalg":
            results["categories"]["linalg"] = run_linalg_comparison(linalg_sizes, output_dir)
        elif cat == "fft":
            results["categories"]["fft"] = run_fft_comparison(fft_sizes, output_dir)

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
        "--categories",
        type=str,
        default="matmul,elementwise,unary,linalg,fft",
        help="Comparison categories to run (comma-separated)"
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
        categories = [c.strip() for c in args.categories.split(",")]
        print(f"Running comparison benchmarks for categories: {categories}")
        print(f"Sizes: {sizes}")

        results = run_comparison_benchmarks(sizes, output_dir, categories)

        # Save comprehensive results
        output_file = output_dir / "comprehensive_comparison.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nComprehensive results saved to: {output_file}")

        # Also save individual category files for backward compatibility
        for cat_name, cat_data in results.get("categories", {}).items():
            cat_file = output_dir / f"{cat_name}_comparison.json"
            with open(cat_file, "w") as f:
                json.dump(cat_data, f, indent=2)
            print(f"Category results saved to: {cat_file}")

        # Legacy format for matmul (backward compatibility)
        if "matmul" in results.get("categories", {}):
            legacy_file = output_dir / "comparison_results.json"
            with open(legacy_file, "w") as f:
                json.dump(results["categories"]["matmul"], f, indent=2)
    else:
        suite = run_suite(args.suite, output_dir)

        output_file = output_dir / f"{args.suite}_results.json"
        with open(output_file, "w") as f:
            suite_dict = asdict(suite)
            json.dump(suite_dict, f, indent=2, default=str)
        print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
