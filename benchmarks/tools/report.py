#!/usr/bin/env python3
"""
Benchmark Report Generator

Generates markdown reports from benchmark results.

Usage (from benchmarks/ directory):
    python tools/report.py --output ../docs/BENCHMARKS.md
"""

import argparse
import json
import platform
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

BENCHMARK_DIR = Path(__file__).parent.parent


def format_gflops(val: Optional[float]) -> str:
    """Format GFLOPS value."""
    if val is None:
        return "N/A"
    return f"{val:.1f}"


def format_speedup(val: float) -> str:
    """Format speedup value."""
    if val >= 1.0:
        return f"**{val:.2f}x**"
    return f"{val:.2f}x"


def generate_comparison_table(results: Dict) -> str:
    """Generate markdown table for comparison results."""
    sizes = results.get("sizes", [])
    if not sizes:
        return "No comparison data available.\n"

    libs = ["axiom_cpu", "eigen", "armadillo", "numpy", "pytorch_cpu"]
    lib_names = {
        "axiom_cpu": "Axiom (CPU)",
        "eigen": "Eigen3",
        "armadillo": "Armadillo",
        "numpy": "NumPy",
        "pytorch_cpu": "PyTorch (CPU)",
    }

    # Header
    header = "| Size |"
    for lib in libs:
        header += f" {lib_names.get(lib, lib)} |"
    header += "\n"

    # Separator
    sep = "|---:|"
    for _ in libs:
        sep += "---:|"
    sep += "\n"

    # Data rows
    rows = ""
    for i, size in enumerate(sizes):
        row = f"| {size}×{size} |"
        for lib in libs:
            data = results.get(lib, [])
            val = data[i] if i < len(data) else None
            row += f" {format_gflops(val)} |"
        rows += row + "\n"

    return header + sep + rows


def generate_gpu_comparison_table(results: Dict) -> str:
    """Generate markdown table for GPU comparison."""
    sizes = results.get("sizes", [])
    if not sizes:
        return "No GPU comparison data available.\n"

    libs = ["axiom_gpu", "pytorch_mps"]
    lib_names = {
        "axiom_gpu": "Axiom (Metal)",
        "pytorch_mps": "PyTorch (MPS)",
    }

    # Check if we have GPU data
    has_gpu_data = any(
        results.get(lib) and any(x is not None for x in results.get(lib, []))
        for lib in libs
    )
    if not has_gpu_data:
        return "No GPU benchmark data available.\n"

    header = "| Size |"
    for lib in libs:
        header += f" {lib_names.get(lib, lib)} |"
    header += " Speedup |\n"

    sep = "|---:|"
    for _ in libs:
        sep += "---:|"
    sep += "---:|\n"

    rows = ""
    for i, size in enumerate(sizes):
        row = f"| {size}×{size} |"
        axiom_val = None
        pytorch_val = None
        for lib in libs:
            data = results.get(lib, [])
            val = data[i] if i < len(data) else None
            if lib == "axiom_gpu":
                axiom_val = val
            elif lib == "pytorch_mps":
                pytorch_val = val
            row += f" {format_gflops(val)} |"

        # Calculate speedup (Axiom / PyTorch)
        if axiom_val and pytorch_val and pytorch_val > 0:
            speedup = axiom_val / pytorch_val
            row += f" {format_speedup(speedup)} |"
        else:
            row += " N/A |"
        rows += row + "\n"

    return header + sep + rows


def generate_fusion_table(results: Dict) -> str:
    """Generate markdown table for fusion results."""
    if "results" not in results:
        return "No fusion data available.\n"

    header = "| Pattern | Lazy (ms) | Eager (ms) | Speedup |\n"
    sep = "|:---|---:|---:|---:|\n"

    rows = ""
    for r in results["results"]:
        name = r.get("name", "Unknown")
        lazy_ms = r.get("time_ms", 0)
        extra = r.get("extra", {})
        eager_ms = extra.get("eager_ms", 0)
        speedup = extra.get("speedup", 1.0)

        rows += f"| {name} | {lazy_ms:.3f} | {eager_ms:.3f} | {format_speedup(speedup)} |\n"

    return header + sep + rows


def generate_report(output_path: Path, results_dir: Path):
    """Generate the full benchmark report."""
    report = []

    # Header
    report.append("# Axiom Benchmark Results\n")
    report.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n")
    report.append(f"*Platform: {platform.system()} {platform.machine()}*\n")
    report.append("\n")

    # Table of contents
    report.append("## Contents\n")
    report.append("1. [Matrix Multiplication (CPU)](#matrix-multiplication-cpu)\n")
    report.append("2. [Matrix Multiplication (GPU)](#matrix-multiplication-gpu)\n")
    report.append("3. [Fusion Patterns](#fusion-patterns)\n")
    report.append("4. [SIMD Kernels](#simd-kernels)\n")
    report.append("\n---\n\n")

    # CPU Comparison
    report.append("## Matrix Multiplication (CPU)\n\n")
    report.append("Performance comparison for square matrix multiplication (GFLOPS, higher is better).\n\n")

    comparison_file = results_dir / "comparison_results.json"
    if comparison_file.exists():
        with open(comparison_file) as f:
            comparison_results = json.load(f)
        report.append(generate_comparison_table(comparison_results))

        # Plot reference
        report.append("\n![CPU Comparison](results/plots/matmul_comparison.png)\n")
    else:
        report.append("*Run `make benchmark-compare` to generate comparison data.*\n")

    report.append("\n---\n\n")

    # GPU Comparison
    report.append("## Matrix Multiplication (GPU)\n\n")
    report.append("GPU performance comparison on Apple Silicon (GFLOPS, higher is better).\n\n")

    if comparison_file.exists():
        report.append(generate_gpu_comparison_table(comparison_results))
        report.append("\n![Scaling](results/plots/matmul_scaling.png)\n")
    else:
        report.append("*Run `make benchmark-compare` to generate GPU comparison data.*\n")

    report.append("\n---\n\n")

    # Fusion Patterns
    report.append("## Fusion Patterns\n\n")
    report.append("Lazy evaluation with fusion vs eager mode execution.\n\n")

    fusion_file = results_dir / "fusion_results.json"
    if fusion_file.exists():
        with open(fusion_file) as f:
            fusion_results = json.load(f)
        report.append(generate_fusion_table(fusion_results))
        report.append("\n![Fusion Speedup](results/plots/fusion_speedup.png)\n")
    else:
        report.append("*Run `make benchmark-fusion` to generate fusion data.*\n")

    report.append("\n---\n\n")

    # SIMD Kernels
    report.append("## SIMD Kernels\n\n")
    report.append("Element-wise operation throughput using Highway SIMD.\n\n")

    simd_file = results_dir / "simd_results.json"
    if simd_file.exists():
        report.append("![SIMD Throughput](results/plots/simd_throughput.png)\n")
    else:
        report.append("*Run `make benchmark-simd` to generate SIMD data.*\n")

    report.append("\n---\n\n")

    # Environment
    report.append("## Test Environment\n\n")
    report.append("```\n")
    report.append(f"OS: {platform.system()} {platform.release()}\n")
    report.append(f"Architecture: {platform.machine()}\n")
    report.append(f"Python: {platform.python_version()}\n")
    report.append("```\n")

    # Write report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("".join(report))

    print(f"Report generated: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark report")
    parser.add_argument(
        "--output",
        type=str,
        default=str(BENCHMARK_DIR.parent / "docs" / "BENCHMARKS.md"),
        help="Output markdown file"
    )
    parser.add_argument(
        "--results",
        type=str,
        default=str(BENCHMARK_DIR / "results"),
        help="Results directory"
    )
    args = parser.parse_args()

    generate_report(Path(args.output), Path(args.results))


if __name__ == "__main__":
    main()
