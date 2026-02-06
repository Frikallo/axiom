# Axiom Benchmark Results

*Generated: 2026-02-05 19:08*

*Platform: Darwin arm64*

## Contents

- [Summary](#summary)
- [Matrix Multiplication](#matrix-multiplication)
- [Element-wise Operations](#element-wise-operations)
- [Unary Operations](#unary-operations)
- [Linear Algebra](#linear-algebra)
- [FFT Operations](#fft-operations)
- [Fusion Patterns](#fusion-patterns)
- [Test Environment](#test-environment)

---

## Summary

Comprehensive performance comparison across tensor operations.

![Comprehensive Summary](../benchmarks/results/plots/comprehensive_summary.png)


---

## Matrix Multiplication

Performance comparison for square matrix multiplication (GFLOPS, higher is better).

| Size | Axiom | Eigen3 | PyTorch | NumPy | Armadillo |
|---:|---:|---:|---:|---:|---:|
| 32×32 | 66.1 | 89.9 | 56.0 | 60.0 | 98.9 |
| 64×64 | 332 | 410 | 320 | 323 | 33.6 |
| 128×128 | 969 | 959 | 943 | 955 | 46.5 |
| 256×256 | 1,355 | 1,347 | 1,411 | 1,412 | 201 |
| 512×512 | 2,610 | 2,253 | 2,580 | 2,542 | 403 |
| 1024×1024 | 3,149 | 2,504 | 2,468 | 2,420 | 434 |
| 2048×2048 | 3,196 | 2,911 | 2,433 | 2,684 | 553 |
| 4096×4096 | 2,771 | 2,858 | 2,833 | 2,938 | 675 |

### Performance Comparison

![Matmul Comparison](../benchmarks/results/plots/matmul_comparison.png)

### Scaling Analysis

![Matmul Scaling](../benchmarks/results/plots/matmul_scaling.png)


---

## Element-wise Operations

Binary element-wise operations (add, sub, mul, div) measured in GB/s throughput.

*Results at 4096×4096 (GB/s)*

| Operation | Axiom | Eigen3 | PyTorch | NumPy |
|:---|---:|---:|---:|---:|
| add | 56.3 | 118 | 90.7 | 35.5 |
| sub | 104 | 119 | 91.1 | 34.8 |
| mul | 119 | 117 | 91.2 | 37.0 |
| div | 97.6 | 116 | 90.7 | 36.7 |

### Performance by Operation

![Elementwise Comparison](../benchmarks/results/plots/elementwise_comparison.png)

### Bar Chart Comparison

![Elementwise Bar](../benchmarks/results/plots/elementwise_bar_4096.png)


---

## Unary Operations

Unary operations (exp, log, sqrt, sin, cos, tanh, abs, neg, relu, sigmoid) measured in GB/s.

*Results at 4096×4096 (GB/s)*

| Operation | Axiom | Eigen3 | PyTorch | NumPy |
|:---|---:|---:|---:|---:|
| exp | 18.6 | 18.4 | 48.8 | 5.61 |
| log | 17.5 | 12.9 | 33.1 | 4.86 |
| sqrt | 39.7 | 66.3 | 70.8 | 25.8 |
| sin | 26.1 | 11.0 | 38.1 | 6.05 |
| cos | 25.6 | 11.0 | 33.8 | 6.61 |
| tanh | 14.1 | 21.5 | 20.9 | 9.10 |
| abs | 103 | 117 | 71.1 | 28.3 |
| neg | 106 | 67.3 | 69.2 | 26.8 |
| relu | 123 | 117 | 69.8 | 17.6 |
| sigmoid | 14.3 | 16.0 | 45.0 | 3.45 |

### Performance by Operation

![Unary Comparison](../benchmarks/results/plots/unary_comparison.png)

### Bar Chart Comparison

![Unary Bar](../benchmarks/results/plots/unary_bar_4096.png)


---

## Linear Algebra

Linear algebra operations (SVD, QR, solve, Cholesky, eigendecomposition, inverse, determinant).
Measured in milliseconds (lower is better).

*Results at 512×512 (time_ms)*

| Operation | Axiom | Eigen3 | PyTorch | NumPy |
|:---|---:|---:|---:|---:|
| svd | 18.6 | 2,207 | 16.2 | 28.7 |
| qr | 4.76 | 1.51 | 4.05 | 8.56 |
| solve | 0.99 | 2.21 | 0.47 | 1.21 |
| cholesky | 0.69 | 0.23 | 0.23 | 1.45 |
| eig | 153 | 21.9 | 9.25 | 15.5 |
| inv | 1.55 | 2.29 | 0.95 | 3.89 |
| det | 1.02 | 1.46 | 0.55 | 1.77 |

### Performance by Operation

![Linalg Comparison](../benchmarks/results/plots/linalg_comparison.png)

### Bar Chart Comparison

![Linalg Bar](../benchmarks/results/plots/linalg_bar_512.png)


---

## FFT Operations

Fast Fourier Transform operations (fft, ifft, rfft, fft2, ifft2, rfft2).
Measured in milliseconds (lower is better).

*Results at 2048×2048 (time_ms)*

| Operation | Axiom | PyTorch | NumPy |
|:---|---:|---:|---:|
| fft | 0.09 | 0.01 | 0.01 |
| ifft | 0.09 | 0.01 | 0.01 |
| rfft | 0.09 | 0.01 | 0.01 |
| fft2 | 14.9 | 27.6 | 63.5 |
| ifft2 | 14.8 | 28.4 | 30.3 |
| rfft2 | 14.2 | 8.04 | 24.2 |

### Performance by Operation

![FFT Comparison](../benchmarks/results/plots/fft_comparison.png)

### Bar Chart Comparison

![FFT Bar](../benchmarks/results/plots/fft_bar_2048.png)


---

## Fusion Patterns

Lazy evaluation with operation fusion vs eager mode execution.

*Run `make benchmark-fusion` to generate fusion data.*


---

## Test Environment

```
OS: Darwin 25.2.0
Architecture: arm64
Python: 3.14.2
Timestamp: 2026-02-05T19:04:35.860936
```

## Notes

- All benchmarks run on CPU
- Axiom uses Accelerate framework (BLAS) on macOS
- Higher GFLOPS/GB/s = better for throughput metrics
- Lower ms = better for time metrics
- Results may vary based on system load and thermal conditions
