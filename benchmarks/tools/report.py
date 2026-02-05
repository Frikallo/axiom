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
from typing import Dict, List, Optional, Any

BENCHMARK_DIR = Path(__file__).parent.parent


def format_value(val: Optional[float], precision: int = 1) -> str:
    """Format a numeric value."""
    if val is None:
        return "—"
    if val >= 1000:
        return f"{val:,.0f}"
    elif val >= 100:
        return f"{val:.0f}"
    elif val >= 10:
        return f"{val:.1f}"
    else:
        return f"{val:.{precision}f}"


def format_speedup(val: float) -> str:
    """Format speedup value."""
    if val >= 1.0:
        return f"**{val:.2f}x**"
    return f"{val:.2f}x"


def generate_matmul_table(results: Dict) -> str:
    """Generate markdown table for matmul comparison results."""
    sizes = results.get("sizes", [])
    if not sizes:
        return "No comparison data available.\n"

    libs = ["axiom", "eigen", "pytorch", "numpy", "armadillo"]
    lib_names = {
        "axiom": "Axiom",
        "eigen": "Eigen3",
        "pytorch": "PyTorch",
        "numpy": "NumPy",
        "armadillo": "Armadillo",
    }

    # Filter to libs that have data
    available_libs = [lib for lib in libs if lib in results and results[lib]]

    # Header
    header = "| Size |"
    for lib in available_libs:
        header += f" {lib_names.get(lib, lib)} |"
    header += "\n"

    # Separator
    sep = "|---:|"
    for _ in available_libs:
        sep += "---:|"
    sep += "\n"

    # Data rows
    rows = ""
    for i, size in enumerate(sizes):
        row = f"| {size}×{size} |"
        for lib in available_libs:
            data = results.get(lib, [])
            val = data[i] if i < len(data) else None
            row += f" {format_value(val)} |"
        rows += row + "\n"

    return header + sep + rows


def generate_ops_table(results: Dict, category: str) -> str:
    """Generate markdown table for ops comparison (elementwise, unary, linalg, fft)."""
    ops = results.get("ops", [])
    sizes = results.get("sizes", [])
    metric = results.get("metric", "")

    if not ops or not sizes:
        return f"No {category} data available.\n"

    libs = ["axiom", "eigen", "pytorch", "numpy"]
    lib_names = {
        "axiom": "Axiom",
        "eigen": "Eigen3",
        "pytorch": "PyTorch",
        "numpy": "NumPy",
    }

    # Filter to libs that have data
    available_libs = [lib for lib in libs if lib in results]

    # Use the largest size for the summary table
    size_idx = len(sizes) - 1
    size = sizes[size_idx]

    # Header
    header = f"| Operation |"
    for lib in available_libs:
        header += f" {lib_names.get(lib, lib)} |"
    header += "\n"

    # Separator
    sep = "|:---|"
    for _ in available_libs:
        sep += "---:|"
    sep += "\n"

    # Data rows
    rows = ""
    for op in ops:
        row = f"| {op} |"
        for lib in available_libs:
            lib_data = results.get(lib, {})
            if isinstance(lib_data, dict) and op in lib_data:
                op_data = lib_data[op]
                val = op_data[size_idx] if size_idx < len(op_data) else None
                row += f" {format_value(val, 2)} |"
            else:
                row += " — |"
        rows += row + "\n"

    return f"*Results at {size}×{size} ({metric})*\n\n" + header + sep + rows


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
    report.append("# Axiom Benchmark Results\n\n")
    report.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n\n")
    report.append(f"*Platform: {platform.system()} {platform.machine()}*\n\n")

    # Check for comprehensive comparison
    comprehensive_file = results_dir / "comprehensive_comparison.json"
    has_comprehensive = comprehensive_file.exists()

    if has_comprehensive:
        with open(comprehensive_file) as f:
            comprehensive = json.load(f)
        categories = comprehensive.get("categories", {})
    else:
        categories = {}

    # Table of contents
    report.append("## Contents\n\n")
    report.append("- [Summary](#summary)\n")
    report.append("- [Matrix Multiplication](#matrix-multiplication)\n")
    report.append("- [Element-wise Operations](#element-wise-operations)\n")
    report.append("- [Unary Operations](#unary-operations)\n")
    report.append("- [Linear Algebra](#linear-algebra)\n")
    report.append("- [FFT Operations](#fft-operations)\n")
    report.append("- [Fusion Patterns](#fusion-patterns)\n")
    report.append("- [Test Environment](#test-environment)\n")
    report.append("\n---\n\n")

    # Summary
    report.append("## Summary\n\n")
    report.append("Comprehensive performance comparison across tensor operations.\n\n")
    if has_comprehensive:
        report.append("![Comprehensive Summary](../benchmarks/results/plots/comprehensive_summary.png)\n\n")
    else:
        report.append("*Run `make benchmark-compare` to generate comparison data.*\n\n")
    report.append("\n---\n\n")

    # Matrix Multiplication
    report.append("## Matrix Multiplication\n\n")
    report.append("Performance comparison for square matrix multiplication (GFLOPS, higher is better).\n\n")

    if "matmul" in categories:
        report.append(generate_matmul_table(categories["matmul"]))
        report.append("\n### Performance Comparison\n\n")
        report.append("![Matmul Comparison](../benchmarks/results/plots/matmul_comparison.png)\n\n")
        report.append("### Scaling Analysis\n\n")
        report.append("![Matmul Scaling](../benchmarks/results/plots/matmul_scaling.png)\n\n")
    else:
        # Try legacy file
        legacy_file = results_dir / "comparison_results.json"
        if legacy_file.exists():
            with open(legacy_file) as f:
                matmul_results = json.load(f)
            report.append(generate_matmul_table(matmul_results))
            report.append("\n![Matmul Comparison](../benchmarks/results/plots/matmul_comparison.png)\n\n")
        else:
            report.append("*Run `make benchmark-compare` to generate matmul data.*\n\n")

    report.append("\n---\n\n")

    # Element-wise Operations
    report.append("## Element-wise Operations\n\n")
    report.append("Binary element-wise operations (add, sub, mul, div) measured in GB/s throughput.\n\n")

    if "elementwise" in categories:
        report.append(generate_ops_table(categories["elementwise"], "elementwise"))
        report.append("\n### Performance by Operation\n\n")
        report.append("![Elementwise Comparison](../benchmarks/results/plots/elementwise_comparison.png)\n\n")
        report.append("### Bar Chart Comparison\n\n")
        # Find the bar chart file
        bar_files = list(results_dir.glob("plots/elementwise_bar_*.png"))
        if bar_files:
            bar_file = bar_files[0].name
            report.append(f"![Elementwise Bar](../benchmarks/results/plots/{bar_file})\n\n")
    else:
        report.append("*Run `make benchmark-compare` to generate elementwise data.*\n\n")

    report.append("\n---\n\n")

    # Unary Operations
    report.append("## Unary Operations\n\n")
    report.append("Unary operations (exp, log, sqrt, sin, cos, tanh, abs, neg, relu, sigmoid) measured in GB/s.\n\n")

    if "unary" in categories:
        report.append(generate_ops_table(categories["unary"], "unary"))
        report.append("\n### Performance by Operation\n\n")
        report.append("![Unary Comparison](../benchmarks/results/plots/unary_comparison.png)\n\n")
        report.append("### Bar Chart Comparison\n\n")
        bar_files = list(results_dir.glob("plots/unary_bar_*.png"))
        if bar_files:
            bar_file = bar_files[0].name
            report.append(f"![Unary Bar](../benchmarks/results/plots/{bar_file})\n\n")
    else:
        report.append("*Run `make benchmark-compare` to generate unary data.*\n\n")

    report.append("\n---\n\n")

    # Linear Algebra
    report.append("## Linear Algebra\n\n")
    report.append("Linear algebra operations (SVD, QR, solve, Cholesky, eigendecomposition, inverse, determinant).\n")
    report.append("Measured in milliseconds (lower is better).\n\n")

    if "linalg" in categories:
        report.append(generate_ops_table(categories["linalg"], "linalg"))
        report.append("\n### Performance by Operation\n\n")
        report.append("![Linalg Comparison](../benchmarks/results/plots/linalg_comparison.png)\n\n")
        report.append("### Bar Chart Comparison\n\n")
        bar_files = list(results_dir.glob("plots/linalg_bar_*.png"))
        if bar_files:
            bar_file = bar_files[0].name
            report.append(f"![Linalg Bar](../benchmarks/results/plots/{bar_file})\n\n")
    else:
        report.append("*Run `make benchmark-compare` to generate linalg data.*\n\n")

    report.append("\n---\n\n")

    # FFT Operations
    report.append("## FFT Operations\n\n")
    report.append("Fast Fourier Transform operations (fft, ifft, rfft, fft2, ifft2, rfft2).\n")
    report.append("Measured in milliseconds (lower is better).\n\n")

    if "fft" in categories:
        report.append(generate_ops_table(categories["fft"], "fft"))
        report.append("\n### Performance by Operation\n\n")
        report.append("![FFT Comparison](../benchmarks/results/plots/fft_comparison.png)\n\n")
        report.append("### Bar Chart Comparison\n\n")
        bar_files = list(results_dir.glob("plots/fft_bar_*.png"))
        if bar_files:
            bar_file = bar_files[0].name
            report.append(f"![FFT Bar](../benchmarks/results/plots/{bar_file})\n\n")
    else:
        report.append("*Run `make benchmark-compare` to generate FFT data.*\n\n")

    report.append("\n---\n\n")

    # Fusion Patterns
    report.append("## Fusion Patterns\n\n")
    report.append("Lazy evaluation with operation fusion vs eager mode execution.\n\n")

    fusion_file = results_dir / "fusion_results.json"
    if fusion_file.exists():
        with open(fusion_file) as f:
            fusion_results = json.load(f)
        report.append(generate_fusion_table(fusion_results))
        fusion_plot = results_dir / "plots" / "fusion_speedup.png"
        if fusion_plot.exists():
            report.append("\n![Fusion Speedup](../benchmarks/results/plots/fusion_speedup.png)\n\n")
    else:
        report.append("*Run `make benchmark-fusion` to generate fusion data.*\n\n")

    report.append("\n---\n\n")

    # Environment
    report.append("## Test Environment\n\n")
    report.append("```\n")
    report.append(f"OS: {platform.system()} {platform.release()}\n")
    report.append(f"Architecture: {platform.machine()}\n")
    report.append(f"Python: {platform.python_version()}\n")
    if has_comprehensive:
        report.append(f"Timestamp: {comprehensive.get('timestamp', 'N/A')}\n")
    report.append("```\n\n")

    # Notes
    report.append("## Notes\n\n")
    report.append("- All benchmarks run on CPU\n")
    report.append("- Axiom uses Accelerate framework (BLAS) on macOS\n")
    report.append("- Higher GFLOPS/GB/s = better for throughput metrics\n")
    report.append("- Lower ms = better for time metrics\n")
    report.append("- Results may vary based on system load and thermal conditions\n")

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
