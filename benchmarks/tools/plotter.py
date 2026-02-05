#!/usr/bin/env python3
"""
Benchmark Plot Generator

Generates visualization plots from benchmark results.

Usage:
    python tools/plotter.py --input results/comparison_results.json
    python tools/plotter.py --all --output results/plots/
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend - must be before pyplot import
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

BENCHMARK_DIR = Path(__file__).parent.parent


def setup_style():
    """Configure matplotlib style."""
    if not HAS_MATPLOTLIB:
        return
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.facecolor'] = '#fafafa'
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3


def plot_matmul_comparison(results: Dict, output_dir: Path):
    """Generate matmul comparison bar chart."""
    setup_style()

    sizes = results.get("sizes", [])
    if not sizes:
        print("No sizes found in results")
        return

    fig, ax = plt.subplots(figsize=(12, 7))

    # Color scheme
    colors = {
        "axiom_cpu": "#E63946",      # Red - primary
        "axiom_gpu": "#9D0208",      # Dark red
        "eigen": "#457B9D",          # Steel blue
        "armadillo": "#2A9D8F",      # Teal
        "numpy": "#AAA",             # Gray
        "pytorch_cpu": "#F4A261",    # Orange
        "pytorch_mps": "#7B2CBF",    # Purple
    }

    labels = {
        "axiom_cpu": "Axiom (CPU)",
        "axiom_gpu": "Axiom (GPU)",
        "eigen": "Eigen3",
        "armadillo": "Armadillo",
        "numpy": "NumPy",
        "pytorch_cpu": "PyTorch (CPU)",
        "pytorch_mps": "PyTorch (MPS)",
    }

    markers = {
        "axiom_cpu": "o",
        "axiom_gpu": "s",
        "eigen": "^",
        "armadillo": "v",
        "numpy": "d",
        "pytorch_cpu": "<",
        "pytorch_mps": ">",
    }

    # Plot order (Axiom last to be on top)
    plot_order = ["numpy", "eigen", "armadillo", "pytorch_cpu", "axiom_cpu"]

    for lib in plot_order:
        data = results.get(lib, [])
        if data and any(x is not None for x in data):
            valid_sizes = []
            valid_data = []
            for s, d in zip(sizes, data):
                if d is not None:
                    valid_sizes.append(s)
                    valid_data.append(d)

            if valid_data:
                linewidth = 3 if lib.startswith("axiom") else 2
                ax.plot(
                    valid_sizes, valid_data,
                    label=labels.get(lib, lib),
                    color=colors.get(lib, "#666"),
                    marker=markers.get(lib, "o"),
                    markersize=7 if lib.startswith("axiom") else 5,
                    linewidth=linewidth,
                    alpha=0.95 if lib.startswith("axiom") else 0.8,
                    zorder=10 if lib.startswith("axiom") else 5
                )

    ax.set_xlabel("Matrix Size (N×N)", fontsize=13, fontweight='medium')
    ax.set_ylabel("Performance (GFLOPS)", fontsize=13, fontweight='medium')
    ax.set_title("Matrix Multiplication Performance Comparison\n(Float32)",
                 fontsize=14, fontweight='bold')

    ax.set_xscale('log', base=2)
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9, edgecolor='#ddd')

    ax.text(
        0.98, 0.02,
        "Higher is better",
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='bottom',
        horizontalalignment='right',
        alpha=0.6,
        style='italic'
    )

    ax.set_ylim(bottom=1)
    plt.tight_layout()

    # Save plots
    output_path = output_dir / "matmul_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Saved: {output_path}")

    svg_path = output_dir / "matmul_comparison.svg"
    plt.savefig(svg_path, format='svg', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Saved: {svg_path}")

    plt.close()


def plot_matmul_scaling(results: Dict, output_dir: Path):
    """Generate log-log scaling plot."""
    setup_style()

    sizes = results.get("sizes", [])
    if not sizes:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {
        "axiom_cpu": "#E63946",
        "axiom_gpu": "#9D0208",
        "pytorch_mps": "#7B2CBF",
    }

    labels = {
        "axiom_cpu": "Axiom (CPU)",
        "axiom_gpu": "Axiom (GPU)",
        "pytorch_mps": "PyTorch (MPS)",
    }

    for lib in ["axiom_cpu", "axiom_gpu", "pytorch_mps"]:
        data = results.get(lib, [])
        if data and any(x is not None for x in data):
            valid_sizes = []
            valid_data = []
            for s, d in zip(sizes, data):
                if d is not None:
                    valid_sizes.append(s)
                    valid_data.append(d)

            if valid_data:
                ax.loglog(
                    valid_sizes, valid_data,
                    'o-',
                    label=labels.get(lib, lib),
                    color=colors.get(lib, "#666"),
                    linewidth=2,
                    markersize=6,
                    markeredgecolor='white',
                    markeredgewidth=1
                )

    ax.set_xlabel("Matrix Size (N×N)", fontsize=12)
    ax.set_ylabel("GFLOPS", fontsize=12)
    ax.set_title("Performance Scaling (Log-Log)", fontsize=13, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()

    output_path = output_dir / "matmul_scaling.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()


def plot_fusion_speedup(results: Dict, output_dir: Path):
    """Generate fusion speedup bar chart."""
    setup_style()

    # Expected format from fusion benchmark
    if "results" not in results:
        print("No fusion results found")
        return

    patterns = []
    speedups = []

    for r in results["results"]:
        name = r.get("name", "")
        if r.get("extra") and "speedup" in r["extra"]:
            patterns.append(name)
            speedups.append(r["extra"]["speedup"])

    if not patterns:
        print("No fusion speedup data found")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ['#3B82F6' if s >= 1.0 else '#EF4444' for s in speedups]
    bars = ax.bar(range(len(patterns)), speedups, color=colors, alpha=0.85,
                  edgecolor='white', linewidth=0.5)

    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)

    ax.set_xlabel("Fusion Pattern", fontsize=12)
    ax.set_ylabel("Speedup (Lazy / Eager)", fontsize=12)
    ax.set_title("Fusion Pattern Speedup\n(>1 = Lazy Faster)", fontsize=13, fontweight='bold')

    ax.set_xticks(range(len(patterns)))
    ax.set_xticklabels(patterns, rotation=45, ha='right', fontsize=9)

    for bar, val in zip(bars, speedups):
        ax.annotate(f'{val:.2f}x',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    output_path = output_dir / "fusion_speedup.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()


def plot_simd_throughput(results: Dict, output_dir: Path):
    """Generate SIMD kernel throughput plot."""
    setup_style()

    if "results" not in results:
        print("No SIMD results found")
        return

    # Group by operation type
    ops = {}
    for r in results["results"]:
        name = r.get("name", "")
        # Parse Google Benchmark name format: BM_BinaryAdd_Float/1048576
        parts = name.split("/")
        if len(parts) >= 2:
            op_name = parts[0].replace("BM_", "")
            try:
                size = int(parts[1])
            except ValueError:
                continue

            if op_name not in ops:
                ops[op_name] = {"sizes": [], "throughput": []}

            ops[op_name]["sizes"].append(size)
            # items_per_second from Google Benchmark
            if r.get("extra") and "items_per_second" in r["extra"]:
                ops[op_name]["throughput"].append(r["extra"]["items_per_second"] / 1e9)

    if not ops:
        print("No SIMD throughput data found")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    for op_name, data in ops.items():
        if data["throughput"]:
            ax.plot(data["sizes"], data["throughput"], 'o-',
                    label=op_name, linewidth=2, markersize=5)

    ax.set_xlabel("Elements", fontsize=12)
    ax.set_ylabel("Throughput (G elements/s)", fontsize=12)
    ax.set_title("SIMD Kernel Throughput", fontsize=13, fontweight='bold')
    ax.set_xscale('log', base=2)
    ax.legend(loc='upper left', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = output_dir / "simd_throughput.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    if not HAS_MATPLOTLIB:
        print("Error: matplotlib is required for plot generation.")
        print("Install with: pip install matplotlib")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Generate benchmark plots")
    parser.add_argument(
        "--input",
        type=str,
        help="Input JSON results file"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate all plots from results directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(BENCHMARK_DIR / "results" / "plots"),
        help="Output directory for plots"
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.all:
        results_dir = BENCHMARK_DIR / "results"

        # Comparison results
        comparison_file = results_dir / "comparison_results.json"
        if comparison_file.exists():
            print("Generating comparison plots...")
            with open(comparison_file) as f:
                results = json.load(f)
            plot_matmul_comparison(results, output_dir)
            plot_matmul_scaling(results, output_dir)

        # Fusion results
        fusion_file = results_dir / "fusion_results.json"
        if fusion_file.exists():
            print("Generating fusion plots...")
            with open(fusion_file) as f:
                results = json.load(f)
            plot_fusion_speedup(results, output_dir)

        # SIMD results
        simd_file = results_dir / "simd_results.json"
        if simd_file.exists():
            print("Generating SIMD plots...")
            with open(simd_file) as f:
                results = json.load(f)
            plot_simd_throughput(results, output_dir)

    elif args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: {input_path} not found")
            return

        with open(input_path) as f:
            results = json.load(f)

        # Detect type and generate appropriate plot
        if "sizes" in results and "axiom_cpu" in results:
            plot_matmul_comparison(results, output_dir)
            plot_matmul_scaling(results, output_dir)
        elif "results" in results:
            # Check first result to determine type
            first = results["results"][0] if results["results"] else {}
            name = first.get("name", "")
            if "Fusion" in name or "Add" in name:
                plot_fusion_speedup(results, output_dir)
            else:
                plot_simd_throughput(results, output_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
