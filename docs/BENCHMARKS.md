# Axiom Benchmark Results

*Generated: 2026-02-21 16:02*

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
| 32×32 | 55.2 | 88.4 | 62.2 | 56.4 | 98.3 |
| 64×64 | 371 | 440 | 336 | 332 | 42.1 |
| 128×128 | 923 | 954 | 930 | 951 | 44.7 |
| 256×256 | 1,421 | 1,251 | 1,412 | 1,488 | 162 |
| 512×512 | 2,389 | 2,345 | 1,310 | 2,358 | 434 |
| 1024×1024 | 2,820 | 2,445 | 2,423 | 2,299 | 524 |
| 2048×2048 | 3,218 | 2,982 | 2,801 | 2,795 | 608 |
| 4096×4096 | 3,087 | 2,961 | 2,961 | 2,959 | 754 |

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
| add | 92.4 | 121 | 94.2 | 40.0 |
| sub | 112 | 119 | 90.4 | 40.5 |
| mul | 117 | 119 | 96.1 | 42.9 |
| div | 99.1 | 120 | 95.4 | 41.2 |

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
| exp | 23.0 | 16.4 | 50.7 | 5.69 |
| log | 17.0 | 13.1 | 33.7 | 4.93 |
| sqrt | 39.0 | 66.3 | 73.2 | 29.7 |
| sin | 26.0 | 11.0 | 39.1 | 6.73 |
| cos | 25.2 | 11.0 | 33.7 | 6.46 |
| tanh | 14.3 | 21.5 | 21.0 | 9.43 |
| abs | 104 | 118 | 75.1 | 27.3 |
| neg | 109 | 66.1 | 75.4 | 32.7 |
| relu | 121 | 122 | 74.2 | 18.5 |
| sigmoid | 14.3 | 15.9 | 47.5 | 3.71 |

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
| svd | 18.2 | 2,200 | 16.2 | 25.9 |
| qr | 4.72 | 1.50 | 4.06 | 7.89 |
| solve | 0.96 | 2.18 | 0.47 | 1.20 |
| cholesky | 0.69 | 0.24 | 0.22 | 1.43 |
| eig | 151 | 22.5 | 9.50 | 15.4 |
| inv | 1.55 | 2.34 | 1.06 | 3.83 |
| det | 1.01 | 1.46 | 0.57 | 1.69 |

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
| fft | 0.00 | 0.01 | 0.01 |
| ifft | 0.00 | 0.01 | 0.01 |
| rfft | 0.00 | 0.01 | 0.01 |
| fft2 | 14.3 | 27.2 | 60.6 |
| ifft2 | 14.3 | 27.6 | 29.6 |
| rfft2 | 10.0 | 7.82 | 22.8 |

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
OS: Darwin 25.3.0
Architecture: arm64
Python: 3.12.7
Timestamp: 2026-02-21T15:56:58.080886
```

## Notes

- All benchmarks run on CPU
- Axiom uses Accelerate framework (BLAS) on macOS
- Higher GFLOPS/GB/s = better for throughput metrics
- Lower ms = better for time metrics
- Results may vary based on system load and thermal conditions
