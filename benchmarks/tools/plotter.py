#!/usr/bin/env python3
"""
Benchmark Plot Generator

Generates visualization plots from benchmark results.

Usage:
    python tools/plotter.py --input results/comprehensive_comparison.json
    python tools/plotter.py --all --output results/plots/
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

BENCHMARK_DIR = Path(__file__).parent.parent

# Color scheme for libraries (CPU only)
COLORS = {
    "axiom": "#E63946",       # Red - primary
    "eigen": "#457B9D",       # Steel blue
    "armadillo": "#2A9D8F",   # Teal
    "numpy": "#6C757D",       # Gray
    "pytorch": "#F4A261",     # Orange
}

LABELS = {
    "axiom": "Axiom",
    "eigen": "Eigen3",
    "armadillo": "Armadillo",
    "numpy": "NumPy",
    "pytorch": "PyTorch",
}

MARKERS = {
    "axiom": "o",
    "eigen": "^",
    "armadillo": "v",
    "numpy": "d",
    "pytorch": "<",
}


def setup_style():
    """Configure matplotlib style."""
    if not HAS_MATPLOTLIB:
        return
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.facecolor'] = '#fafafa'
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3


def plot_library_comparison(results: Dict, output_dir: Path, title: str,
                            ylabel: str, filename: str, higher_is_better: bool = True):
    """Generic library comparison line plot."""
    setup_style()

    sizes = results.get("sizes", [])
    if not sizes:
        print(f"No sizes found in results for {filename}")
        return

    fig, ax = plt.subplots(figsize=(12, 7))

    # Determine which libraries have data (CPU only)
    plot_order = ["numpy", "eigen", "armadillo", "pytorch", "axiom"]
    libs_with_data = []
    for lib in plot_order:
        if lib in results:
            data = results[lib]
            if isinstance(data, list) and any(x is not None for x in data):
                libs_with_data.append(lib)

    for lib in libs_with_data:
        data = results[lib]
        valid_sizes = []
        valid_data = []
        for s, d in zip(sizes, data):
            if d is not None:
                valid_sizes.append(s)
                valid_data.append(d)

        if valid_data:
            is_axiom = lib.startswith("axiom")
            ax.plot(
                valid_sizes, valid_data,
                label=LABELS.get(lib, lib),
                color=COLORS.get(lib, "#666"),
                marker=MARKERS.get(lib, "o"),
                markersize=7 if is_axiom else 5,
                linewidth=3 if is_axiom else 2,
                alpha=0.95 if is_axiom else 0.8,
                zorder=10 if is_axiom else 5
            )

    ax.set_xlabel("Matrix Size (N×N)", fontsize=13, fontweight='medium')
    ax.set_ylabel(ylabel, fontsize=13, fontweight='medium')
    ax.set_title(title, fontsize=14, fontweight='bold')

    ax.set_xscale('log', base=2)
    ax.legend(loc='upper left' if higher_is_better else 'upper right',
              fontsize=10, framealpha=0.9, edgecolor='#ddd')

    better_text = "Higher is better" if higher_is_better else "Lower is better"
    ax.text(0.98, 0.02, better_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', horizontalalignment='right',
            alpha=0.6, style='italic')

    if higher_is_better:
        ax.set_ylim(bottom=0)

    plt.tight_layout()

    output_path = output_dir / f"{filename}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")

    svg_path = output_dir / f"{filename}.svg"
    plt.savefig(svg_path, format='svg', bbox_inches='tight', facecolor='white')
    print(f"Saved: {svg_path}")

    plt.close()


def plot_matmul_comparison(results: Dict, output_dir: Path):
    """Generate matmul comparison chart."""
    plot_library_comparison(
        results, output_dir,
        title="Matrix Multiplication Performance Comparison\n(Float32)",
        ylabel="Performance (GFLOPS)",
        filename="matmul_comparison",
        higher_is_better=True
    )


def plot_matmul_scaling(results: Dict, output_dir: Path):
    """Generate log-log scaling plot for matmul."""
    setup_style()

    sizes = results.get("sizes", [])
    if not sizes:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for lib in ["axiom", "eigen", "pytorch", "numpy"]:
        data = results.get(lib, [])
        if data and any(x is not None for x in data):
            valid_sizes = []
            valid_data = []
            for s, d in zip(sizes, data):
                if d is not None:
                    valid_sizes.append(s)
                    valid_data.append(d)

            if valid_data:
                is_axiom = lib == "axiom"
                ax.loglog(
                    valid_sizes, valid_data, 'o-',
                    label=LABELS.get(lib, lib),
                    color=COLORS.get(lib, "#666"),
                    linewidth=3 if is_axiom else 2,
                    markersize=7 if is_axiom else 5,
                    markeredgecolor='white', markeredgewidth=1,
                    zorder=10 if is_axiom else 5
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


def plot_ops_comparison(results: Dict, output_dir: Path, category: str):
    """Generate comparison plot for ops with multiple operations (elementwise, unary, etc.)"""
    setup_style()

    ops = results.get("ops", [])
    sizes = results.get("sizes", [])
    metric = results.get("metric", "GB/s")
    higher_is_better = metric != "time_ms"

    if not ops or not sizes:
        print(f"No data for {category}")
        return

    # Libraries to compare (CPU only)
    libs = ["axiom", "eigen", "numpy", "pytorch"]
    libs_with_data = [lib for lib in libs if lib in results]

    # Create subplot grid
    n_ops = len(ops)
    n_cols = min(4, n_ops)
    n_rows = (n_ops + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = [[axes]]
    elif n_rows == 1:
        axes = [axes]
    elif n_cols == 1:
        axes = [[ax] for ax in axes]

    for idx, op in enumerate(ops):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row][col]

        for lib in libs_with_data:
            lib_data = results.get(lib, {})
            if isinstance(lib_data, dict) and op in lib_data:
                op_data = lib_data[op]
                valid_sizes = []
                valid_data = []
                for s, d in zip(sizes, op_data):
                    if d is not None:
                        valid_sizes.append(s)
                        valid_data.append(d)

                if valid_data:
                    is_axiom = lib.startswith("axiom")
                    ax.plot(
                        valid_sizes, valid_data,
                        label=LABELS.get(lib, lib),
                        color=COLORS.get(lib, "#666"),
                        marker=MARKERS.get(lib, "o"),
                        markersize=5 if is_axiom else 4,
                        linewidth=2 if is_axiom else 1.5,
                        alpha=0.95 if is_axiom else 0.7
                    )

        ax.set_title(op.upper(), fontsize=11, fontweight='bold')
        ax.set_xscale('log', base=2)
        ax.grid(True, alpha=0.3)
        if higher_is_better:
            ax.set_ylim(bottom=0)

    # Hide empty subplots
    for idx in range(n_ops, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row][col].set_visible(False)

    # Common labels
    fig.text(0.5, 0.02, 'Matrix Size (N×N)', ha='center', fontsize=12)
    fig.text(0.02, 0.5, metric, va='center', rotation='vertical', fontsize=12)

    # Legend
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.98, 0.98),
               fontsize=9, framealpha=0.9)

    title_map = {
        "elementwise": "Element-wise Binary Operations",
        "unary": "Unary Operations",
        "linalg": "Linear Algebra Operations",
        "fft": "FFT Operations",
    }

    fig.suptitle(f"{title_map.get(category, category)} Performance Comparison",
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0.03, 0.03, 0.95, 0.95])

    output_path = output_dir / f"{category}_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")

    svg_path = output_dir / f"{category}_comparison.svg"
    plt.savefig(svg_path, format='svg', bbox_inches='tight', facecolor='white')
    print(f"Saved: {svg_path}")

    plt.close()


def plot_ops_bar_chart(results: Dict, output_dir: Path, category: str, size: int = None):
    """Generate bar chart comparing operations at a specific size."""
    setup_style()

    ops = results.get("ops", [])
    sizes = results.get("sizes", [])
    metric = results.get("metric", "GB/s")
    higher_is_better = metric != "time_ms"

    if not ops or not sizes:
        return

    # Use largest size or specified size
    if size is None:
        size = max(sizes)
    if size not in sizes:
        size = sizes[-1]
    size_idx = sizes.index(size)

    # Libraries to compare (CPU only)
    libs = ["axiom", "eigen", "numpy", "pytorch"]
    libs_with_data = [lib for lib in libs if lib in results]

    # Build data matrix
    import numpy as np
    data_matrix = []
    valid_ops = []

    for op in ops:
        op_data = []
        has_data = False
        for lib in libs_with_data:
            lib_data = results.get(lib, {})
            if isinstance(lib_data, dict) and op in lib_data:
                val = lib_data[op][size_idx] if size_idx < len(lib_data[op]) else None
                op_data.append(val if val is not None else 0)
                if val is not None:
                    has_data = True
            else:
                op_data.append(0)
        if has_data:
            data_matrix.append(op_data)
            valid_ops.append(op)

    if not valid_ops:
        return

    data_matrix = np.array(data_matrix)

    fig, ax = plt.subplots(figsize=(max(10, len(valid_ops) * 1.5), 6))

    x = np.arange(len(valid_ops))
    width = 0.8 / len(libs_with_data)

    for i, lib in enumerate(libs_with_data):
        offset = width * (i - len(libs_with_data) / 2 + 0.5)
        bars = ax.bar(x + offset, data_matrix[:, i], width,
                      label=LABELS.get(lib, lib),
                      color=COLORS.get(lib, "#666"),
                      alpha=0.85, edgecolor='white', linewidth=0.5)

    ax.set_xlabel("Operation", fontsize=12)
    ax.set_ylabel(metric, fontsize=12)

    title_map = {
        "elementwise": "Element-wise Operations",
        "unary": "Unary Operations",
        "linalg": "Linear Algebra",
        "fft": "FFT",
    }
    ax.set_title(f"{title_map.get(category, category)} at {size}×{size}\n({metric})",
                 fontsize=13, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([op.upper() for op in valid_ops], rotation=45, ha='right')
    ax.legend(fontsize=9, loc='upper right')

    better_text = "Higher is better" if higher_is_better else "Lower is better"
    ax.text(0.02, 0.98, better_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', alpha=0.6, style='italic')

    plt.tight_layout()

    output_path = output_dir / f"{category}_bar_{size}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()


def plot_comprehensive_summary(results: Dict, output_dir: Path):
    """Generate a comprehensive summary plot with all categories."""
    setup_style()

    categories = results.get("categories", {})
    if not categories:
        return

    # Create a multi-panel figure
    n_cats = len(categories)
    fig = plt.figure(figsize=(16, 4 * n_cats))
    gs = gridspec.GridSpec(n_cats, 2, width_ratios=[2, 1], hspace=0.3)

    for cat_idx, (cat_name, cat_data) in enumerate(categories.items()):
        ops = cat_data.get("ops", [])
        sizes = cat_data.get("sizes", [])
        metric = cat_data.get("metric", "")

        # Left panel: line plot for first op or matmul
        ax_line = fig.add_subplot(gs[cat_idx, 0])

        libs = ["axiom", "eigen", "numpy", "pytorch"]
        libs_with_data = [lib for lib in libs if lib in cat_data]

        if cat_name == "matmul":
            # Direct array data
            for lib in libs_with_data:
                data = cat_data.get(lib, [])
                if isinstance(data, list) and any(x is not None for x in data):
                    valid_sizes = [s for s, d in zip(sizes, data) if d is not None]
                    valid_data = [d for d in data if d is not None]
                    if valid_data:
                        is_axiom = lib.startswith("axiom")
                        ax_line.plot(valid_sizes, valid_data,
                                     label=LABELS.get(lib, lib),
                                     color=COLORS.get(lib, "#666"),
                                     marker=MARKERS.get(lib, "o"),
                                     linewidth=2 if is_axiom else 1.5,
                                     alpha=0.9)
        else:
            # Nested dict data - plot first op
            first_op = ops[0] if ops else None
            if first_op:
                for lib in libs_with_data:
                    lib_data = cat_data.get(lib, {})
                    if isinstance(lib_data, dict) and first_op in lib_data:
                        op_data = lib_data[first_op]
                        valid_sizes = [s for s, d in zip(sizes, op_data) if d is not None]
                        valid_data = [d for d in op_data if d is not None]
                        if valid_data:
                            is_axiom = lib.startswith("axiom")
                            ax_line.plot(valid_sizes, valid_data,
                                         label=LABELS.get(lib, lib),
                                         color=COLORS.get(lib, "#666"),
                                         marker=MARKERS.get(lib, "o"),
                                         linewidth=2 if is_axiom else 1.5,
                                         alpha=0.9)

        ax_line.set_xscale('log', base=2)
        ax_line.set_title(f"{cat_name.title()} ({metric})", fontsize=11, fontweight='bold')
        ax_line.legend(fontsize=8, loc='best')
        ax_line.grid(True, alpha=0.3)

        # Right panel: bar chart at largest size
        ax_bar = fig.add_subplot(gs[cat_idx, 1])

        if sizes:
            largest_size = max(sizes)
            size_idx = sizes.index(largest_size)

            if cat_name == "matmul":
                bar_data = []
                bar_labels = []
                for lib in libs_with_data:
                    data = cat_data.get(lib, [])
                    if isinstance(data, list) and size_idx < len(data) and data[size_idx] is not None:
                        bar_data.append(data[size_idx])
                        bar_labels.append(LABELS.get(lib, lib))
            else:
                bar_data = []
                bar_labels = []
                # Average across all ops
                for lib in libs_with_data:
                    lib_data = cat_data.get(lib, {})
                    if isinstance(lib_data, dict):
                        vals = []
                        for op in ops:
                            if op in lib_data and size_idx < len(lib_data[op]):
                                v = lib_data[op][size_idx]
                                if v is not None:
                                    vals.append(v)
                        if vals:
                            bar_data.append(sum(vals) / len(vals))
                            bar_labels.append(LABELS.get(lib, lib))

            if bar_data:
                colors = [COLORS.get(lab.lower().replace(" ", "_").replace("(", "").replace(")", ""), "#666")
                          for lab in bar_labels]
                bars = ax_bar.barh(range(len(bar_data)), bar_data, color=colors, alpha=0.85)
                ax_bar.set_yticks(range(len(bar_data)))
                ax_bar.set_yticklabels(bar_labels, fontsize=9)
                ax_bar.set_title(f"@ {largest_size}×{largest_size}", fontsize=10)

    plt.suptitle("Comprehensive Library Comparison", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_path = output_dir / "comprehensive_summary.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()


def plot_fusion_speedup(results: Dict, output_dir: Path):
    """Generate fusion speedup bar chart."""
    setup_style()

    if "results" not in results:
        return

    patterns = []
    speedups = []

    for r in results["results"]:
        name = r.get("name", "")
        if r.get("extra") and "speedup" in r["extra"]:
            patterns.append(name)
            speedups.append(r["extra"]["speedup"])

    if not patterns:
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
        return

    ops = {}
    for r in results["results"]:
        name = r.get("name", "")
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
            if r.get("extra") and "items_per_second" in r["extra"]:
                ops[op_name]["throughput"].append(r["extra"]["items_per_second"] / 1e9)

    if not ops:
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
    parser.add_argument("--input", type=str, help="Input JSON results file")
    parser.add_argument("--all", action="store_true",
                        help="Generate all plots from results directory")
    parser.add_argument("--output", type=str,
                        default=str(BENCHMARK_DIR / "results" / "plots"),
                        help="Output directory for plots")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.all:
        results_dir = BENCHMARK_DIR / "results"

        # Comprehensive comparison results
        comprehensive_file = results_dir / "comprehensive_comparison.json"
        if comprehensive_file.exists():
            print("Generating comprehensive plots...")
            with open(comprehensive_file) as f:
                results = json.load(f)

            # Generate summary plot
            plot_comprehensive_summary(results, output_dir)

            # Generate individual category plots
            categories = results.get("categories", {})

            if "matmul" in categories:
                print("  Generating matmul plots...")
                plot_matmul_comparison(categories["matmul"], output_dir)
                plot_matmul_scaling(categories["matmul"], output_dir)

            for cat in ["elementwise", "unary", "linalg", "fft"]:
                if cat in categories:
                    print(f"  Generating {cat} plots...")
                    plot_ops_comparison(categories[cat], output_dir, cat)
                    # Bar chart at largest size
                    sizes = categories[cat].get("sizes", [])
                    if sizes:
                        plot_ops_bar_chart(categories[cat], output_dir, cat, max(sizes))

        # Legacy comparison results (backward compatibility)
        comparison_file = results_dir / "comparison_results.json"
        if comparison_file.exists() and not comprehensive_file.exists():
            print("Generating comparison plots from legacy file...")
            with open(comparison_file) as f:
                results = json.load(f)
            plot_matmul_comparison(results, output_dir)
            plot_matmul_scaling(results, output_dir)

        # Individual category files
        for cat in ["elementwise", "unary", "linalg", "fft"]:
            cat_file = results_dir / f"{cat}_comparison.json"
            if cat_file.exists():
                print(f"Generating {cat} plots from individual file...")
                with open(cat_file) as f:
                    results = json.load(f)
                plot_ops_comparison(results, output_dir, cat)

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
        if "categories" in results:
            # Comprehensive results
            plot_comprehensive_summary(results, output_dir)
            for cat_name, cat_data in results["categories"].items():
                if cat_name == "matmul":
                    plot_matmul_comparison(cat_data, output_dir)
                    plot_matmul_scaling(cat_data, output_dir)
                else:
                    plot_ops_comparison(cat_data, output_dir, cat_name)
        elif "sizes" in results and "axiom" in results:
            plot_matmul_comparison(results, output_dir)
            plot_matmul_scaling(results, output_dir)
        elif "ops" in results:
            category = results.get("category", "unknown")
            plot_ops_comparison(results, output_dir, category)
        elif "results" in results:
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
