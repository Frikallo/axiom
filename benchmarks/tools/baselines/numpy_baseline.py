#!/usr/bin/env python3
"""NumPy and PyTorch matmul benchmarks"""

import sys
import time
import numpy as np

def benchmark_numpy(n: int, warmup: int = 3, iterations: int = 10) -> float:
    """Benchmark NumPy matmul and return GFLOPS"""
    A = np.random.randn(n, n).astype(np.float32)
    B = np.random.randn(n, n).astype(np.float32)

    # Warmup
    for _ in range(warmup):
        C = A @ B

    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        C = A @ B
    end = time.perf_counter()

    avg_seconds = (end - start) / iterations
    flops = 2.0 * n * n * n
    gflops = flops / avg_seconds / 1e9

    return gflops


def benchmark_pytorch(n: int, warmup: int = 3, iterations: int = 10) -> float:
    """Benchmark PyTorch matmul and return GFLOPS"""
    try:
        import torch
    except ImportError:
        return -1.0

    A = torch.randn(n, n, dtype=torch.float32)
    B = torch.randn(n, n, dtype=torch.float32)

    # Warmup
    for _ in range(warmup):
        C = torch.mm(A, B)

    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        C = torch.mm(A, B)
    end = time.perf_counter()

    avg_seconds = (end - start) / iterations
    flops = 2.0 * n * n * n
    gflops = flops / avg_seconds / 1e9

    return gflops


def benchmark_pytorch_mps(n: int, warmup: int = 3, iterations: int = 10) -> float:
    """Benchmark PyTorch MPS (Metal) matmul and return GFLOPS"""
    try:
        import torch
        if not torch.backends.mps.is_available():
            return -1.0
    except (ImportError, AttributeError):
        return -1.0

    device = torch.device("mps")
    A = torch.randn(n, n, dtype=torch.float32, device=device)
    B = torch.randn(n, n, dtype=torch.float32, device=device)

    # Warmup
    for _ in range(warmup):
        C = torch.mm(A, B)
        torch.mps.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        C = torch.mm(A, B)
    torch.mps.synchronize()
    end = time.perf_counter()

    avg_seconds = (end - start) / iterations
    flops = 2.0 * n * n * n
    gflops = flops / avg_seconds / 1e9

    return gflops


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python bench_python_matmul.py <library> <size>", file=sys.stderr)
        sys.exit(1)

    library = sys.argv[1]
    n = int(sys.argv[2])

    if library == "numpy":
        print(benchmark_numpy(n))
    elif library == "pytorch":
        print(benchmark_pytorch(n))
    elif library == "pytorch_mps":
        print(benchmark_pytorch_mps(n))
    else:
        print(f"Unknown library: {library}", file=sys.stderr)
        sys.exit(1)
