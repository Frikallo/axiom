#!/usr/bin/env python3
"""
GPU Benchmark Comparison: Axiom CUDA vs PyTorch CUDA

Runs square matmul benchmarks at multiple sizes and prints a side-by-side
comparison table.

Usage:
    python benchmarks/tools/gpu_compare.py [--sizes 128,256,...] [--json out.json]
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

BENCHMARKS_DIR = Path(__file__).parent.parent
BUILD_DIR = BENCHMARKS_DIR.parent / "build"
COMPARE_BIN = BUILD_DIR / "benchmarks" / "compare" / "matmul_axiom_cuda"
PYTORCH_SCRIPT = BENCHMARKS_DIR / "tools" / "baselines" / "pytorch_cuda_baseline.py"


def run_axiom(size):
    """Run Axiom CUDA benchmark, return GFLOPS or None."""
    if not COMPARE_BIN.exists():
        return None
    try:
        result = subprocess.run(
            [str(COMPARE_BIN), str(size)],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode == 0:
            return float(result.stdout.strip())
    except Exception:
        pass
    return None


def run_pytorch(size):
    """Run PyTorch CUDA benchmark, return GFLOPS or None."""
    try:
        result = subprocess.run(
            [sys.executable, str(PYTORCH_SCRIPT), str(size)],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode == 0:
            return float(result.stdout.strip())
    except Exception:
        pass
    return None


def main():
    parser = argparse.ArgumentParser(description="Axiom vs PyTorch GPU matmul benchmark")
    parser.add_argument(
        "--sizes", type=str, default="128,256,512,1024,2048,4096",
        help="Comma-separated matrix sizes",
    )
    parser.add_argument("--json", type=str, help="Save results to JSON")
    args = parser.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")]

    # Header
    print("=" * 72)
    print("GPU Matmul Benchmark: Axiom CUDA vs PyTorch CUDA")
    print("=" * 72)

    # Check availability
    has_axiom = COMPARE_BIN.exists()
    has_pytorch = True
    try:
        import torch
        has_pytorch = torch.cuda.is_available()
        if has_pytorch:
            print(f"GPU:     {torch.cuda.get_device_name(0)}")
            print(f"PyTorch: {torch.__version__} (CUDA {torch.version.cuda})")
    except ImportError:
        has_pytorch = False

    if has_axiom:
        print(f"Axiom:   {COMPARE_BIN}")
    else:
        print(f"Axiom:   NOT FOUND at {COMPARE_BIN}")

    print("=" * 72)
    print(f"{'Size':>6}  {'Axiom (GFLOPS)':>16}  {'PyTorch (GFLOPS)':>16}  {'Ratio (A/P)':>12}")
    print("-" * 72)

    results = []
    for n in sizes:
        axiom_gf = run_axiom(n) if has_axiom else None
        pytorch_gf = run_pytorch(n) if has_pytorch else None

        axiom_str = f"{axiom_gf:>14.1f}" if axiom_gf else "         N/A"
        pytorch_str = f"{pytorch_gf:>14.1f}" if pytorch_gf else "         N/A"

        if axiom_gf and pytorch_gf:
            ratio = axiom_gf / pytorch_gf
            ratio_str = f"{ratio:>10.2f}x"
        else:
            ratio_str = "        N/A"

        print(f"{n:>6}  {axiom_str}    {pytorch_str}    {ratio_str}")
        results.append({
            "size": n,
            "axiom_gflops": axiom_gf,
            "pytorch_gflops": pytorch_gf,
            "ratio": axiom_gf / pytorch_gf if axiom_gf and pytorch_gf else None,
        })

    print("=" * 72)

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"benchmark": "gpu_matmul", "results": results}, f, indent=2)
        print(f"Results saved to {args.json}")


if __name__ == "__main__":
    main()
