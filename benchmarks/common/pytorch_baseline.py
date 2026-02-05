#!/usr/bin/env python3
"""
PyTorch MPS Baseline Benchmarks

This script runs equivalent matrix multiplication benchmarks using PyTorch
with MPS (Metal Performance Shaders) backend for comparison with Axiom.

Usage:
    python pytorch_baseline.py [--json output.json] [--warmup N] [--iterations N]

Requirements:
    - PyTorch >= 2.0 with MPS support
    - macOS 12.3+ with Apple Silicon or AMD GPU
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass
from typing import Optional

try:
    import torch
except ImportError:
    print("Error: PyTorch not installed. Install with: pip install torch")
    sys.exit(1)


@dataclass
class BenchmarkResult:
    name: str
    m: int
    n: int
    k: int
    time_ms: float
    gflops: float
    device: str
    dtype: str


def check_mps_available() -> bool:
    """Check if MPS backend is available."""
    if not torch.backends.mps.is_available():
        print("Warning: MPS not available on this system")
        return False
    if not torch.backends.mps.is_built():
        print("Warning: PyTorch not built with MPS support")
        return False
    return True


def warmup_device(device: str, iterations: int = 5):
    """Warm up the device with some operations."""
    if device == "mps":
        a = torch.randn(256, 256, device=device)
        b = torch.randn(256, 256, device=device)
        for _ in range(iterations):
            c = torch.matmul(a, b)
            torch.mps.synchronize()
    elif device == "cpu":
        a = torch.randn(256, 256)
        b = torch.randn(256, 256)
        for _ in range(iterations):
            c = torch.matmul(a, b)


def benchmark_matmul(
    m: int,
    n: int,
    k: int,
    device: str,
    dtype: torch.dtype,
    warmup_iters: int = 5,
    bench_iters: int = 20,
) -> BenchmarkResult:
    """Benchmark matrix multiplication C = A @ B where A is (M x K), B is (K x N)."""
    # Create tensors
    a = torch.randn(m, k, device=device, dtype=dtype)
    b = torch.randn(k, n, device=device, dtype=dtype)

    # Warmup
    for _ in range(warmup_iters):
        c = torch.matmul(a, b)
        if device == "mps":
            torch.mps.synchronize()

    # Benchmark
    times = []
    for _ in range(bench_iters):
        if device == "mps":
            torch.mps.synchronize()

        start = time.perf_counter()
        c = torch.matmul(a, b)
        if device == "mps":
            torch.mps.synchronize()
        end = time.perf_counter()

        times.append(end - start)

    # Calculate statistics
    avg_time = sum(times) / len(times)
    time_ms = avg_time * 1000

    # GFLOPS = 2*M*N*K / time_in_seconds / 1e9
    flops = 2 * m * n * k
    gflops = flops / avg_time / 1e9

    dtype_str = str(dtype).replace("torch.", "")
    return BenchmarkResult(
        name=f"BM_PyTorch_{device.upper()}_{dtype_str}/{m}/{n}/{k}",
        m=m,
        n=n,
        k=k,
        time_ms=time_ms,
        gflops=gflops,
        device=device,
        dtype=dtype_str,
    )


def benchmark_batched_matmul(
    batch: int,
    m: int,
    n: int,
    k: int,
    device: str,
    dtype: torch.dtype,
    warmup_iters: int = 5,
    bench_iters: int = 20,
) -> BenchmarkResult:
    """Benchmark batched matrix multiplication."""
    a = torch.randn(batch, m, k, device=device, dtype=dtype)
    b = torch.randn(batch, k, n, device=device, dtype=dtype)

    # Warmup
    for _ in range(warmup_iters):
        c = torch.matmul(a, b)
        if device == "mps":
            torch.mps.synchronize()

    # Benchmark
    times = []
    for _ in range(bench_iters):
        if device == "mps":
            torch.mps.synchronize()

        start = time.perf_counter()
        c = torch.matmul(a, b)
        if device == "mps":
            torch.mps.synchronize()
        end = time.perf_counter()

        times.append(end - start)

    avg_time = sum(times) / len(times)
    time_ms = avg_time * 1000

    # Total FLOPS = batch * 2*M*N*K
    flops = batch * 2 * m * n * k
    gflops = flops / avg_time / 1e9

    dtype_str = str(dtype).replace("torch.", "")
    return BenchmarkResult(
        name=f"BM_PyTorch_{device.upper()}_Batched_{dtype_str}/{batch}/{m}/{n}/{k}",
        m=m,
        n=n,
        k=k,
        time_ms=time_ms,
        gflops=gflops,
        device=device,
        dtype=dtype_str,
    )


def print_result(result: BenchmarkResult):
    """Print a single benchmark result."""
    print(f"{result.name:<60} {result.time_ms:>10.3f} ms {result.gflops:>10.1f} GFLOPS")


def main():
    parser = argparse.ArgumentParser(
        description="PyTorch MPS baseline benchmarks for comparison with Axiom"
    )
    parser.add_argument(
        "--json", type=str, help="Output results to JSON file", default=None
    )
    parser.add_argument(
        "--warmup", type=int, help="Number of warmup iterations", default=5
    )
    parser.add_argument(
        "--iterations", type=int, help="Number of benchmark iterations", default=20
    )
    parser.add_argument(
        "--cpu-only", action="store_true", help="Only run CPU benchmarks"
    )
    parser.add_argument(
        "--gpu-only", action="store_true", help="Only run GPU benchmarks"
    )
    args = parser.parse_args()

    results = []

    # Check MPS availability
    mps_available = check_mps_available() and not args.cpu_only

    # Devices to benchmark
    devices = []
    if not args.gpu_only:
        devices.append("cpu")
    if mps_available and not args.cpu_only:
        devices.append("mps")

    if not devices:
        print("Error: No devices available for benchmarking")
        sys.exit(1)

    print("=" * 82)
    print("PyTorch MPS Baseline Benchmarks")
    print("=" * 82)
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {mps_available}")
    print(f"Warmup iterations: {args.warmup}")
    print(f"Benchmark iterations: {args.iterations}")
    print("=" * 82)
    print()

    # Square matrix sizes
    square_sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096]

    # Transformer sizes (M, N, K, name)
    transformer_sizes = [
        (512, 64, 512, "attention_head"),
        (2048, 64, 2048, "attention_large"),
        (2048, 768, 768, "ffn_bert_base"),
        (2048, 3072, 768, "ffn_bert_up"),
        (2048, 768, 3072, "ffn_bert_down"),
        (1, 2048, 4096, "llm_decode_small"),
        (1, 4096, 4096, "llm_decode_med"),
        (1, 4096, 11008, "llm_decode_llama_up"),
        (512, 4096, 4096, "llm_prefill"),
    ]

    # Batched sizes (batch, M, N, K)
    batched_sizes = [
        (8, 512, 64, 512),
        (12, 512, 64, 512),
        (16, 512, 64, 512),
        (32, 2048, 64, 2048),
    ]

    for device in devices:
        print(f"\n{'='*82}")
        print(f"Device: {device.upper()}")
        print("=" * 82)

        # Warmup
        print(f"Warming up {device}...")
        warmup_device(device, args.warmup)
        print()

        # Square matrix benchmarks
        print("-" * 82)
        print("Square Matrices (Float32)")
        print("-" * 82)
        for size in square_sizes:
            result = benchmark_matmul(
                size,
                size,
                size,
                device,
                torch.float32,
                args.warmup,
                args.iterations,
            )
            print_result(result)
            results.append(result)

        # Float16 benchmarks (GPU only, as CPU float16 is slow)
        if device == "mps":
            print()
            print("-" * 82)
            print("Square Matrices (Float16)")
            print("-" * 82)
            for size in square_sizes:
                result = benchmark_matmul(
                    size,
                    size,
                    size,
                    device,
                    torch.float16,
                    args.warmup,
                    args.iterations,
                )
                print_result(result)
                results.append(result)

        # Transformer workloads
        print()
        print("-" * 82)
        print("Transformer Workloads (Float32)")
        print("-" * 82)
        for m, n, k, name in transformer_sizes:
            result = benchmark_matmul(
                m, n, k, device, torch.float32, args.warmup, args.iterations
            )
            print_result(result)
            results.append(result)

        # Batched matmul (GPU only typically)
        if device == "mps":
            print()
            print("-" * 82)
            print("Batched Matmul (Attention-style)")
            print("-" * 82)
            for batch, m, n, k in batched_sizes:
                result = benchmark_batched_matmul(
                    batch, m, n, k, device, torch.float32, args.warmup, args.iterations
                )
                print_result(result)
                results.append(result)

    # Output JSON if requested
    if args.json:
        output = {
            "context": {
                "pytorch_version": torch.__version__,
                "mps_available": mps_available,
                "warmup_iterations": args.warmup,
                "benchmark_iterations": args.iterations,
            },
            "benchmarks": [
                {
                    "name": r.name,
                    "m": r.m,
                    "n": r.n,
                    "k": r.k,
                    "time_ms": r.time_ms,
                    "gflops": r.gflops,
                    "device": r.device,
                    "dtype": r.dtype,
                }
                for r in results
            ],
        }
        with open(args.json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults written to {args.json}")

    print()
    print("=" * 82)
    print("Benchmark complete!")
    print("=" * 82)


if __name__ == "__main__":
    main()
