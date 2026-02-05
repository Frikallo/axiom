# Axiom Benchmark Results

*Generated: 2026-02-05 13:04*

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
| 32×32 | 58.3 | 88.9 | 57.2 | 57.6 | 87.4 |
| 64×64 | 269 | 438 | 314 | 333 | 32.3 |
| 128×128 | 998 | 957 | 961 | 961 | 30.8 |
| 256×256 | 1,418 | 1,341 | 1,411 | 1,490 | 169 |
| 512×512 | 1,486 | 2,334 | 1,503 | 2,368 | 465 |
| 1024×1024 | 3,162 | 2,476 | 2,040 | 2,197 | 547 |
| 2048×2048 | 3,089 | 2,961 | 2,837 | 2,802 | 636 |
| 4096×4096 | 3,023 | 2,960 | 2,962 | 2,925 | 739 |

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
| add | 57.8 | 112 | 92.2 | 38.0 |
| sub | 112 | 119 | 93.6 | 36.4 |
| mul | 120 | 120 | 84.1 | 39.3 |
| div | 98.9 | 118 | 92.3 | 37.1 |

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
| exp | 18.8 | 18.6 | 48.8 | 5.69 |
| log | 17.3 | 12.9 | 31.2 | 4.84 |
| sqrt | 39.3 | 63.2 | 69.5 | 28.1 |
| sin | 25.9 | 11.0 | 39.3 | 6.72 |
| cos | 25.6 | 11.0 | 32.1 | 6.66 |
| tanh | 14.1 | 21.5 | 20.7 | 9.27 |
| abs | 104 | 118 | 70.4 | 31.7 |
| neg | 109 | 65.3 | 66.7 | 30.4 |
| relu | 118 | 118 | 71.1 | 18.4 |
| sigmoid | 15.3 | 15.9 | 46.0 | 3.48 |

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
| svd | 18.9 | 2,220 | 16.5 | 28.7 |
| qr | 4.83 | 1.52 | 4.10 | 7.90 |
| solve | 0.96 | 2.18 | 0.46 | 1.19 |
| cholesky | 0.69 | 0.23 | 0.27 | 1.46 |
| eig | 187 | 22.1 | 9.49 | 15.4 |
| inv | 1.54 | 2.39 | 1.03 | 3.66 |
| det | 1.00 | 1.46 | 0.60 | 1.69 |

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
| fft | 0.05 | 0.01 | 0.01 |
| ifft | 0.05 | 0.01 | 0.01 |
| rfft | 0.05 | 0.01 | 0.01 |
| fft2 | 250 | 27.6 | 61.8 |
| ifft2 | 249 | 28.0 | 30.6 |
| rfft2 | 163 | 7.90 | 23.6 |

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
Timestamp: 2026-02-05T12:55:44.390795
```

## Notes

- All benchmarks run on CPU
- Axiom uses Accelerate framework (BLAS) on macOS
- Higher GFLOPS/GB/s = better for throughput metrics
- Lower ms = better for time metrics
- Results may vary based on system load and thermal conditions
