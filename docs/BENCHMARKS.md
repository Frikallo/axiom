# Axiom Benchmark Results
*Generated: 2026-02-05 01:30*
*Platform: Darwin arm64*

## Contents
1. [Matrix Multiplication (CPU)](#matrix-multiplication-cpu)
2. [Matrix Multiplication (GPU)](#matrix-multiplication-gpu)
3. [Fusion Patterns](#fusion-patterns)
4. [SIMD Kernels](#simd-kernels)

---

## Matrix Multiplication (CPU)

Performance comparison for square matrix multiplication (GFLOPS, higher is better).

| Size | Axiom (CPU) | Eigen3 | Armadillo | NumPy | PyTorch (CPU) |
|---:|---:|---:|---:|---:|---:|
| 32×32 | 61.9 | 88.9 | 98.3 | 54.2 | 55.0 |
| 48×48 | 137.9 | 220.3 | 92.3 | 159.9 | 162.3 |
| 64×64 | 342.9 | 432.4 | 29.1 | 317.0 | 326.8 |
| 96×96 | 737.3 | 719.8 | 25.7 | 645.4 | 687.2 |
| 128×128 | 972.6 | 935.5 | 43.8 | 917.6 | 894.8 |
| 192×192 | 1236.8 | 1161.5 | 70.1 | 1361.1 | 1379.9 |
| 256×256 | 1515.4 | 1347.1 | 195.7 | 1595.9 | 1606.1 |
| 384×384 | 1710.6 | 1505.2 | 374.1 | 1690.7 | 1791.5 |
| 512×512 | 2549.8 | 2271.6 | 378.1 | 2300.8 | 2531.4 |
| 768×768 | 3381.3 | 2954.1 | 459.8 | 2400.9 | 2415.0 |
| 1024×1024 | 3095.2 | 2493.9 | 474.7 | 2335.3 | 2144.0 |
| 1536×1536 | 3480.1 | 3059.8 | 494.4 | 2457.9 | 2929.8 |
| 2048×2048 | 3062.8 | 2933.2 | 456.1 | 2752.7 | 2631.4 |
| 3072×3072 | 2943.0 | 2751.6 | 626.9 | 2697.0 | 2700.6 |
| 4096×4096 | 2949.3 | 2735.4 | 624.9 | 2866.7 | 2824.5 |

![CPU Comparison](../benchmarks/results/plots/matmul_comparison.png)

---

## Matrix Multiplication (GPU)

GPU performance comparison on Apple Silicon (GFLOPS, higher is better).

| Size | Axiom (Metal) | PyTorch (MPS) | Speedup |
|---:|---:|---:|---:|
| 32×32 | 0.4 | 1.3 | 0.31x |
| 48×48 | 1.6 | 4.4 | 0.36x |
| 64×64 | 2.9 | 11.9 | 0.25x |
| 96×96 | 5.1 | 35.3 | 0.14x |
| 128×128 | 10.7 | 47.5 | 0.23x |
| 192×192 | 45.1 | 128.4 | 0.35x |
| 256×256 | 104.0 | 303.1 | 0.34x |
| 384×384 | 240.2 | 561.1 | 0.43x |
| 512×512 | 380.9 | 900.1 | 0.42x |
| 768×768 | 977.6 | 981.3 | 1.00x |
| 1024×1024 | 1310.5 | 2041.3 | 0.64x |
| 1536×1536 | 2834.3 | 4699.1 | 0.60x |
| 2048×2048 | 3417.9 | 5166.6 | 0.66x |
| 3072×3072 | 3977.3 | 5269.0 | 0.75x |
| 4096×4096 | 4141.2 | 5293.0 | 0.78x |

![Scaling](../benchmarks/results/plots/matmul_scaling.png)

---

## Fusion Patterns

Lazy evaluation with fusion vs eager mode execution.

*Run `make benchmark-fusion` to generate fusion data.*

---

## SIMD Kernels

Element-wise operation throughput using Highway SIMD.

*Run `make benchmark-simd` to generate SIMD data.*

---

## Test Environment

```
OS: Darwin 25.2.0
Architecture: arm64
Python: 3.12.7
```
