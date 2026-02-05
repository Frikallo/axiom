# Axiom Benchmarks

Comprehensive benchmarking suite for the Axiom tensor library.

## Quick Start

```bash
# Build all benchmarks
make benchmarks

# Run comparison benchmarks and generate plots
make benchmark-compare

# Run specific benchmark suites
make benchmark-matmul    # Matrix multiplication
make benchmark-simd      # SIMD kernel throughput
make benchmark-fusion    # Lazy evaluation / fusion
```

## Directory Structure

```
benchmarks/
├── core/                 # Axiom internal benchmarks
│   ├── matmul/          # GEMM benchmarks (CPU + GPU)
│   ├── simd/            # SIMD kernel throughput
│   └── fusion/          # Lazy evaluation and fusion
│
├── compare/             # Library comparison benchmarks
│   ├── matmul_axiom.cpp
│   ├── matmul_eigen.cpp
│   └── matmul_armadillo.cpp
│
├── common/              # Shared utilities
│   └── benchmark_utils.hpp
│
├── tools/               # Python automation
│   ├── runner.py        # Unified benchmark runner
│   ├── plotter.py       # Plot generation
│   └── baselines/       # PyTorch/NumPy baselines
│
└── results/             # Output directory (gitignored)
```

## Benchmark Categories

### Core Benchmarks (Google Benchmark)

Internal performance benchmarks using Google Benchmark framework:

- **GEMM** (`core/matmul/`): Matrix multiplication on CPU and GPU
- **SIMD** (`core/simd/`): Element-wise ops, reductions, activations
- **Fusion** (`core/fusion/`): Lazy evaluation vs eager mode

Run with:
```bash
./build/benchmarks/bench_gemm --benchmark_format=console
./build/benchmarks/bench_simd_kernels
./build/benchmarks/bench_fusion
```

### Library Comparisons

Compare Axiom against other libraries:

```bash
make benchmark-compare
```

This runs matmul benchmarks against:
- Eigen3 (with Accelerate BLAS)
- Armadillo (with Accelerate BLAS)
- NumPy (with Accelerate BLAS)
- PyTorch CPU and MPS (GPU)

Results are saved to `results/` and plots to `results/plots/`.

## Adding New Benchmarks

1. Create a new `.cpp` file in the appropriate `core/` subdirectory
2. Use Google Benchmark macros (`BENCHMARK(...)`)
3. Add the target to `CMakeLists.txt`
4. Run `make benchmarks` to build

Example:
```cpp
#include <benchmark/benchmark.h>
#include <axiom/axiom.hpp>

static void BM_MyOperation(benchmark::State& state) {
    auto tensor = axiom::Tensor::randn({1024, 1024});
    for (auto _ : state) {
        auto result = axiom::ops::my_operation(tensor);
        benchmark::DoNotOptimize(result.data());
    }
}
BENCHMARK(BM_MyOperation);
```

## Output Formats

All benchmarks support JSON output for programmatic analysis:

```bash
./build/benchmarks/bench_gemm --benchmark_out=results/gemm.json --benchmark_out_format=json
```

## Environment Variables

- `AXIOM_EAGER_MODE=1`: Disable lazy evaluation for comparison
- `AXIOM_SKIP_GPU_TESTS=1`: Skip GPU benchmarks (useful in CI)
