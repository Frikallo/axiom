# Axiom Benchmark Results

*Generated: 2026-02-06 11:49*

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
| 32×32 | 12.5 | 72.2 | 59.4 | 55.2 | 97.1 |
| 64×64 | 102 | 422 | 319 | 329 | 32.3 |
| 128×128 | 485 | 956 | 949 | 957 | 31.3 |
| 256×256 | 1,165 | 1,348 | 1,553 | 1,412 | 162 |
| 512×512 | 1,535 | 2,288 | 1,255 | 2,580 | 462 |
| 1024×1024 | 2,614 | 2,484 | 2,227 | 2,333 | 519 |
| 2048×2048 | 3,189 | 2,941 | 2,754 | 2,745 | 606 |
| 4096×4096 | 3,090 | 2,974 | 2,902 | 2,940 | 683 |

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
| add | 89.4 | 116 | 91.1 | 36.4 |
| sub | 109 | 118 | 92.4 | 36.4 |
| mul | 117 | 117 | 92.1 | 36.0 |
| div | 98.1 | 120 | 92.3 | 35.7 |

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
| exp | 23.8 | 18.7 | 47.9 | 5.36 |
| log | 17.5 | 13.0 | 33.5 | 4.65 |
| sqrt | 40.0 | 66.4 | 68.7 | 22.1 |
| sin | 26.2 | 11.0 | 36.7 | 6.58 |
| cos | 26.0 | 11.1 | 24.4 | 6.63 |
| tanh | 14.3 | 21.6 | 18.5 | 9.06 |
| abs | 104 | 119 | 67.5 | 26.1 |
| neg | 108 | 66.8 | 66.2 | 26.2 |
| relu | 121 | 117 | 61.5 | 18.1 |
| sigmoid | 14.0 | 16.0 | 43.5 | 3.50 |

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
| svd | 18.6 | 2,182 | 16.5 | 28.9 |
| qr | 4.73 | 1.57 | 4.11 | 7.96 |
| solve | 0.99 | 2.17 | 0.44 | 1.21 |
| cholesky | 0.71 | 0.24 | 0.21 | 1.48 |
| eig | 160 | 22.4 | 9.51 | 15.5 |
| inv | 1.63 | 2.34 | 1.05 | 3.89 |
| det | 1.03 | 1.47 | 0.61 | 1.68 |

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
| fft | 0.10 | 0.01 | 0.01 |
| ifft | 0.09 | 0.01 | 0.01 |
| rfft | 0.09 | 0.01 | 0.01 |
| fft2 | 14.8 | 28.0 | 62.2 |
| ifft2 | 14.7 | 27.9 | 30.1 |
| rfft2 | 14.2 | 7.83 | 23.7 |

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
Timestamp: 2026-02-06T11:47:28.078194
```

## Notes

- All benchmarks run on CPU
- Axiom uses Accelerate framework (BLAS) on macOS
- Higher GFLOPS/GB/s = better for throughput metrics
- Lower ms = better for time metrics
- Results may vary based on system load and thermal conditions
