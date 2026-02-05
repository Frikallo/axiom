#!/usr/bin/env python3
"""
Comprehensive NumPy and PyTorch benchmarks for comparison.

Covers:
- Element-wise binary operations (add, sub, mul, div)
- Element-wise unary operations (exp, log, sqrt, sin, cos, tanh, etc.)
- Linear algebra operations (svd, qr, solve, cholesky, eig, inv, det)
- FFT operations (fft, ifft, fft2, ifft2, rfft, rfft2)
- Matrix multiplication (matmul)
"""

import json
import sys
import time
from typing import Any, Dict, Optional

import numpy as np


# ============================================================================
# Element-wise Binary Operations
# ============================================================================

def benchmark_elementwise_numpy(op: str, n: int, warmup: int = 3,
                                 iterations: int = 10) -> float:
    """Benchmark NumPy element-wise binary ops, return GB/s"""
    A = np.random.randn(n, n).astype(np.float32)
    B = np.random.randn(n, n).astype(np.float32)

    op_func = {
        "add": lambda: A + B,
        "sub": lambda: A - B,
        "mul": lambda: A * B,
        "div": lambda: A / B,
    }[op]

    for _ in range(warmup):
        op_func()

    start = time.perf_counter()
    for _ in range(iterations):
        C = op_func()
    end = time.perf_counter()

    avg_seconds = (end - start) / iterations
    bytes_processed = 3.0 * n * n * 4  # 2 reads + 1 write, float32
    gbps = bytes_processed / avg_seconds / 1e9
    return gbps


def benchmark_elementwise_pytorch(op: str, n: int, device: str = "cpu",
                                   warmup: int = 3, iterations: int = 10) -> float:
    """Benchmark PyTorch element-wise binary ops, return GB/s"""
    try:
        import torch
    except ImportError:
        return -1.0

    if device == "mps":
        if not torch.backends.mps.is_available():
            return -1.0
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")

    A = torch.randn(n, n, dtype=torch.float32, device=dev)
    B = torch.randn(n, n, dtype=torch.float32, device=dev)

    op_func = {
        "add": lambda: A + B,
        "sub": lambda: A - B,
        "mul": lambda: A * B,
        "div": lambda: A / B,
    }[op]

    for _ in range(warmup):
        C = op_func()
        if device == "mps":
            torch.mps.synchronize()

    start = time.perf_counter()
    for _ in range(iterations):
        C = op_func()
    if device == "mps":
        torch.mps.synchronize()
    end = time.perf_counter()

    avg_seconds = (end - start) / iterations
    bytes_processed = 3.0 * n * n * 4
    gbps = bytes_processed / avg_seconds / 1e9
    return gbps


# ============================================================================
# Element-wise Unary Operations
# ============================================================================

def benchmark_unary_numpy(op: str, n: int, warmup: int = 3,
                          iterations: int = 10) -> float:
    """Benchmark NumPy unary ops, return GB/s"""
    A = np.abs(np.random.randn(n, n).astype(np.float32)) + 0.1

    op_func = {
        "exp": lambda: np.exp(A),
        "log": lambda: np.log(A),
        "sqrt": lambda: np.sqrt(A),
        "sin": lambda: np.sin(A),
        "cos": lambda: np.cos(A),
        "tanh": lambda: np.tanh(A),
        "abs": lambda: np.abs(A),
        "neg": lambda: -A,
        "relu": lambda: np.maximum(A, 0),
        "sigmoid": lambda: 1 / (1 + np.exp(-A)),
    }[op]

    for _ in range(warmup):
        op_func()

    start = time.perf_counter()
    for _ in range(iterations):
        C = op_func()
    end = time.perf_counter()

    avg_seconds = (end - start) / iterations
    bytes_processed = 2.0 * n * n * 4  # 1 read + 1 write
    gbps = bytes_processed / avg_seconds / 1e9
    return gbps


def benchmark_unary_pytorch(op: str, n: int, device: str = "cpu",
                            warmup: int = 3, iterations: int = 10) -> float:
    """Benchmark PyTorch unary ops, return GB/s"""
    try:
        import torch
    except ImportError:
        return -1.0

    if device == "mps":
        if not torch.backends.mps.is_available():
            return -1.0
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")

    A = torch.abs(torch.randn(n, n, dtype=torch.float32, device=dev)) + 0.1

    op_func = {
        "exp": lambda: torch.exp(A),
        "log": lambda: torch.log(A),
        "sqrt": lambda: torch.sqrt(A),
        "sin": lambda: torch.sin(A),
        "cos": lambda: torch.cos(A),
        "tanh": lambda: torch.tanh(A),
        "abs": lambda: torch.abs(A),
        "neg": lambda: -A,
        "relu": lambda: torch.relu(A),
        "sigmoid": lambda: torch.sigmoid(A),
    }[op]

    for _ in range(warmup):
        C = op_func()
        if device == "mps":
            torch.mps.synchronize()

    start = time.perf_counter()
    for _ in range(iterations):
        C = op_func()
    if device == "mps":
        torch.mps.synchronize()
    end = time.perf_counter()

    avg_seconds = (end - start) / iterations
    bytes_processed = 2.0 * n * n * 4
    gbps = bytes_processed / avg_seconds / 1e9
    return gbps


# ============================================================================
# Linear Algebra Operations
# ============================================================================

def benchmark_linalg_numpy(op: str, n: int, warmup: int = 2,
                           iterations: int = 5) -> float:
    """Benchmark NumPy linalg ops, return time in ms"""
    if op in ["solve", "cholesky", "eig"]:
        R = np.random.randn(n, n).astype(np.float32)
        A = R @ R.T + np.eye(n, dtype=np.float32) * n
        B = np.random.randn(n, 1).astype(np.float32)
    else:
        A = np.random.randn(n, n).astype(np.float32)
        B = np.random.randn(n, 1).astype(np.float32)

    op_func = {
        "svd": lambda: np.linalg.svd(A, full_matrices=False),
        "qr": lambda: np.linalg.qr(A),
        "solve": lambda: np.linalg.solve(A, B),
        "cholesky": lambda: np.linalg.cholesky(A),
        "eig": lambda: np.linalg.eigh(A),
        "inv": lambda: np.linalg.inv(A),
        "det": lambda: np.linalg.det(A),
    }[op]

    for _ in range(warmup):
        op_func()

    start = time.perf_counter()
    for _ in range(iterations):
        result = op_func()
    end = time.perf_counter()

    avg_ms = (end - start) * 1000 / iterations
    return avg_ms


def benchmark_linalg_pytorch(op: str, n: int, device: str = "cpu",
                             warmup: int = 2, iterations: int = 5) -> float:
    """Benchmark PyTorch linalg ops, return time in ms"""
    try:
        import torch
    except ImportError:
        return -1.0

    if device == "mps":
        if not torch.backends.mps.is_available():
            return -1.0
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")

    if op in ["solve", "cholesky", "eig"]:
        R = torch.randn(n, n, dtype=torch.float32, device=dev)
        A = R @ R.T + torch.eye(n, dtype=torch.float32, device=dev) * n
        B = torch.randn(n, 1, dtype=torch.float32, device=dev)
    else:
        A = torch.randn(n, n, dtype=torch.float32, device=dev)
        B = torch.randn(n, 1, dtype=torch.float32, device=dev)

    op_func = {
        "svd": lambda: torch.linalg.svd(A, full_matrices=False),
        "qr": lambda: torch.linalg.qr(A),
        "solve": lambda: torch.linalg.solve(A, B),
        "cholesky": lambda: torch.linalg.cholesky(A),
        "eig": lambda: torch.linalg.eigh(A),
        "inv": lambda: torch.linalg.inv(A),
        "det": lambda: torch.linalg.det(A),
    }[op]

    for _ in range(warmup):
        result = op_func()
        if device == "mps":
            torch.mps.synchronize()

    start = time.perf_counter()
    for _ in range(iterations):
        result = op_func()
    if device == "mps":
        torch.mps.synchronize()
    end = time.perf_counter()

    avg_ms = (end - start) * 1000 / iterations
    return avg_ms


# ============================================================================
# FFT Operations
# ============================================================================

def benchmark_fft_numpy(op: str, n: int, warmup: int = 3,
                        iterations: int = 10) -> float:
    """Benchmark NumPy FFT ops, return time in ms"""
    if op in ["fft", "ifft"]:
        A = (np.random.randn(n) + 1j * np.random.randn(n)).astype(np.complex64)
    elif op == "rfft":
        A = np.random.randn(n).astype(np.float32)
    elif op in ["fft2", "ifft2"]:
        A = (np.random.randn(n, n) + 1j * np.random.randn(n, n)).astype(np.complex64)
    else:  # rfft2
        A = np.random.randn(n, n).astype(np.float32)

    op_func = {
        "fft": lambda: np.fft.fft(A),
        "ifft": lambda: np.fft.ifft(A),
        "rfft": lambda: np.fft.rfft(A),
        "fft2": lambda: np.fft.fft2(A),
        "ifft2": lambda: np.fft.ifft2(A),
        "rfft2": lambda: np.fft.rfft2(A),
    }[op]

    for _ in range(warmup):
        op_func()

    start = time.perf_counter()
    for _ in range(iterations):
        result = op_func()
    end = time.perf_counter()

    avg_ms = (end - start) * 1000 / iterations
    return avg_ms


def benchmark_fft_pytorch(op: str, n: int, device: str = "cpu",
                          warmup: int = 3, iterations: int = 10) -> float:
    """Benchmark PyTorch FFT ops, return time in ms"""
    try:
        import torch
    except ImportError:
        return -1.0

    if device == "mps":
        if not torch.backends.mps.is_available():
            return -1.0
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")

    if op in ["fft", "ifft"]:
        A = torch.randn(n, dtype=torch.complex64, device=dev)
    elif op == "rfft":
        A = torch.randn(n, dtype=torch.float32, device=dev)
    elif op in ["fft2", "ifft2"]:
        A = torch.randn(n, n, dtype=torch.complex64, device=dev)
    else:  # rfft2
        A = torch.randn(n, n, dtype=torch.float32, device=dev)

    op_func = {
        "fft": lambda: torch.fft.fft(A),
        "ifft": lambda: torch.fft.ifft(A),
        "rfft": lambda: torch.fft.rfft(A),
        "fft2": lambda: torch.fft.fft2(A),
        "ifft2": lambda: torch.fft.ifft2(A),
        "rfft2": lambda: torch.fft.rfft2(A),
    }[op]

    for _ in range(warmup):
        result = op_func()
        if device == "mps":
            torch.mps.synchronize()

    start = time.perf_counter()
    for _ in range(iterations):
        result = op_func()
    if device == "mps":
        torch.mps.synchronize()
    end = time.perf_counter()

    avg_ms = (end - start) * 1000 / iterations
    return avg_ms


# ============================================================================
# Matmul (kept for compatibility)
# ============================================================================

def benchmark_matmul_numpy(n: int, warmup: int = 3, iterations: int = 10) -> float:
    """Benchmark NumPy matmul, return GFLOPS"""
    A = np.random.randn(n, n).astype(np.float32)
    B = np.random.randn(n, n).astype(np.float32)

    for _ in range(warmup):
        C = A @ B

    start = time.perf_counter()
    for _ in range(iterations):
        C = A @ B
    end = time.perf_counter()

    avg_seconds = (end - start) / iterations
    flops = 2.0 * n * n * n
    gflops = flops / avg_seconds / 1e9
    return gflops


def benchmark_matmul_pytorch(n: int, device: str = "cpu",
                             warmup: int = 3, iterations: int = 10) -> float:
    """Benchmark PyTorch matmul, return GFLOPS"""
    try:
        import torch
    except ImportError:
        return -1.0

    if device == "mps":
        if not torch.backends.mps.is_available():
            return -1.0
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")

    A = torch.randn(n, n, dtype=torch.float32, device=dev)
    B = torch.randn(n, n, dtype=torch.float32, device=dev)

    for _ in range(warmup):
        C = torch.mm(A, B)
        if device == "mps":
            torch.mps.synchronize()

    start = time.perf_counter()
    for _ in range(iterations):
        C = torch.mm(A, B)
    if device == "mps":
        torch.mps.synchronize()
    end = time.perf_counter()

    avg_seconds = (end - start) / iterations
    flops = 2.0 * n * n * n
    gflops = flops / avg_seconds / 1e9
    return gflops


# ============================================================================
# Comprehensive Benchmark Runner
# ============================================================================

def run_comprehensive_benchmark(category: str, size: int, library: str = "numpy") -> Dict[str, Any]:
    """Run all benchmarks in a category and return results"""
    results = {}

    if category == "elementwise":
        ops = ["add", "sub", "mul", "div"]
        for op in ops:
            if library == "numpy":
                results[op] = benchmark_elementwise_numpy(op, size)
            elif library == "pytorch":
                results[op] = benchmark_elementwise_pytorch(op, size, "cpu")
            elif library == "pytorch_mps":
                results[op] = benchmark_elementwise_pytorch(op, size, "mps")

    elif category == "unary":
        ops = ["exp", "log", "sqrt", "sin", "cos", "tanh", "abs", "neg", "relu", "sigmoid"]
        for op in ops:
            if library == "numpy":
                results[op] = benchmark_unary_numpy(op, size)
            elif library == "pytorch":
                results[op] = benchmark_unary_pytorch(op, size, "cpu")
            elif library == "pytorch_mps":
                results[op] = benchmark_unary_pytorch(op, size, "mps")

    elif category == "linalg":
        ops = ["svd", "qr", "solve", "cholesky", "eig", "inv", "det"]
        for op in ops:
            if library == "numpy":
                results[op] = benchmark_linalg_numpy(op, size)
            elif library == "pytorch":
                results[op] = benchmark_linalg_pytorch(op, size, "cpu")
            elif library == "pytorch_mps":
                results[op] = benchmark_linalg_pytorch(op, size, "mps")

    elif category == "fft":
        ops = ["fft", "ifft", "rfft", "fft2", "ifft2", "rfft2"]
        for op in ops:
            if library == "numpy":
                results[op] = benchmark_fft_numpy(op, size)
            elif library == "pytorch":
                results[op] = benchmark_fft_pytorch(op, size, "cpu")
            elif library == "pytorch_mps":
                results[op] = benchmark_fft_pytorch(op, size, "mps")

    elif category == "matmul":
        if library == "numpy":
            results["matmul"] = benchmark_matmul_numpy(size)
        elif library == "pytorch":
            results["matmul"] = benchmark_matmul_pytorch(size, "cpu")
        elif library == "pytorch_mps":
            results["matmul"] = benchmark_matmul_pytorch(size, "mps")

    return results


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python comprehensive_baseline.py <category> <library> <size> [op]",
              file=sys.stderr)
        print("  categories: elementwise, unary, linalg, fft, matmul, all", file=sys.stderr)
        print("  libraries: numpy, pytorch, pytorch_mps", file=sys.stderr)
        sys.exit(1)

    category = sys.argv[1]
    library = sys.argv[2]
    size = int(sys.argv[3])
    specific_op = sys.argv[4] if len(sys.argv) > 4 else None

    if category == "all":
        # Run all categories
        all_results = {}
        for cat in ["elementwise", "unary", "linalg", "fft", "matmul"]:
            all_results[cat] = run_comprehensive_benchmark(cat, size, library)
        print(json.dumps(all_results))
    elif specific_op:
        # Run specific operation
        if category == "elementwise":
            if library == "numpy":
                result = benchmark_elementwise_numpy(specific_op, size)
            elif library == "pytorch":
                result = benchmark_elementwise_pytorch(specific_op, size, "cpu")
            elif library == "pytorch_mps":
                result = benchmark_elementwise_pytorch(specific_op, size, "mps")
        elif category == "unary":
            if library == "numpy":
                result = benchmark_unary_numpy(specific_op, size)
            elif library == "pytorch":
                result = benchmark_unary_pytorch(specific_op, size, "cpu")
            elif library == "pytorch_mps":
                result = benchmark_unary_pytorch(specific_op, size, "mps")
        elif category == "linalg":
            if library == "numpy":
                result = benchmark_linalg_numpy(specific_op, size)
            elif library == "pytorch":
                result = benchmark_linalg_pytorch(specific_op, size, "cpu")
            elif library == "pytorch_mps":
                result = benchmark_linalg_pytorch(specific_op, size, "mps")
        elif category == "fft":
            if library == "numpy":
                result = benchmark_fft_numpy(specific_op, size)
            elif library == "pytorch":
                result = benchmark_fft_pytorch(specific_op, size, "cpu")
            elif library == "pytorch_mps":
                result = benchmark_fft_pytorch(specific_op, size, "mps")
        else:
            result = -1.0
        print(result)
    else:
        # Run all ops in category
        results = run_comprehensive_benchmark(category, size, library)
        print(json.dumps(results))
