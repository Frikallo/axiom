#!/usr/bin/env python3
"""
PyTorch CUDA Baseline Benchmarks

Runs matrix multiplication benchmarks on CUDA GPU for comparison with Axiom.
Outputs one GFLOPS value per line matching the Axiom compare binary format,
or full results as JSON.

Usage:
    python pytorch_cuda_baseline.py <size>           # single size, prints GFLOPS
    python pytorch_cuda_baseline.py --all [--json out.json]  # full suite
"""

import argparse
import json
import sys
import time

try:
    import torch
except ImportError:
    print("Error: PyTorch not installed", file=sys.stderr)
    sys.exit(1)


def benchmark_matmul(n, warmup=5, iterations=20):
    """Benchmark NxN matmul on CUDA, return (avg_ms, gflops)."""
    a = torch.randn(n, n, device="cuda", dtype=torch.float32)
    b = torch.randn(n, n, device="cuda", dtype=torch.float32)

    for _ in range(warmup):
        c = torch.matmul(a, b)
        torch.cuda.synchronize()

    times = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append(end - start)

    avg_s = sum(times) / len(times)
    avg_ms = avg_s * 1000
    flops = 2.0 * n * n * n
    gflops = flops / avg_s / 1e9
    return avg_ms, gflops


def main():
    parser = argparse.ArgumentParser(description="PyTorch CUDA matmul benchmark")
    parser.add_argument("size", nargs="?", type=int, help="Matrix size N (NxN matmul)")
    parser.add_argument("--all", action="store_true", help="Run full benchmark suite")
    parser.add_argument("--json", type=str, help="Output JSON results to file")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("Error: CUDA not available", file=sys.stderr)
        sys.exit(1)

    if args.size and not args.all:
        # Single-size mode: match Axiom binary output format (just print GFLOPS)
        _, gflops = benchmark_matmul(args.size, args.warmup, args.iterations)
        print(f"{gflops:.6f}")
        return

    if not args.all:
        parser.print_help()
        sys.exit(1)

    # Full suite mode
    device_name = torch.cuda.get_device_name(0)
    sizes = [128, 256, 512, 1024, 2048, 4096]

    print("=" * 78)
    print(f"PyTorch CUDA Benchmark — {device_name}")
    print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}")
    print("=" * 78)
    print(f"{'Size':>6}  {'Time (ms)':>12}  {'GFLOPS':>12}")
    print("-" * 78)

    results = []
    for n in sizes:
        avg_ms, gflops = benchmark_matmul(n, args.warmup, args.iterations)
        print(f"{n:>6}  {avg_ms:>12.3f}  {gflops:>12.1f}")
        results.append({
            "size": n,
            "time_ms": avg_ms,
            "gflops": gflops,
        })

    if args.json:
        output = {
            "framework": "pytorch",
            "device": device_name,
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "results": results,
        }
        with open(args.json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults written to {args.json}")


if __name__ == "__main__":
    main()
