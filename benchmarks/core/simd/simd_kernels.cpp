// SIMD Kernel Benchmarks
// Benchmarks element-wise ops, reductions, and activations that use SIMD dispatch.
// Run before/after Highway migration to compare performance.
//
// Usage:
//   make clean && make release AXIOM_BUILD_BENCHMARKS=ON
//   ./build/release/benchmarks/bench_simd_kernels

// Include axiom first to avoid namespace collision with C random()
#include "axiom/axiom.hpp"

#include <benchmark/benchmark.h>

#include <cmath>
#include <random>
#include <vector>

namespace {

// Generate random data
template <typename T>
std::vector<T> random_data(size_t n, T min_val = T(-1), T max_val = T(1)) {
    std::vector<T> data(n);
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<T> dist(min_val, max_val);
    for (size_t i = 0; i < n; ++i) {
        data[i] = dist(rng);
    }
    return data;
}

// ============================================================================
// Binary Operations
// ============================================================================

static void BM_BinaryAdd_Float(benchmark::State &state) {
    const size_t n = static_cast<size_t>(state.range(0));
    auto a = axiom::Tensor::randn({n});
    auto b = axiom::Tensor::randn({n});
    axiom::Tensor c;

    for (auto _ : state) {
        c = a + b;
        benchmark::DoNotOptimize(c.data());
    }

    state.SetBytesProcessed(state.iterations() * n * sizeof(float) * 3);
    state.SetItemsProcessed(state.iterations() * n);
}

static void BM_BinaryMul_Float(benchmark::State &state) {
    const size_t n = static_cast<size_t>(state.range(0));
    auto a = axiom::Tensor::randn({n});
    auto b = axiom::Tensor::randn({n});
    axiom::Tensor c;

    for (auto _ : state) {
        c = axiom::ops::multiply(a, b);
        benchmark::DoNotOptimize(c.data());
    }

    state.SetBytesProcessed(state.iterations() * n * sizeof(float) * 3);
    state.SetItemsProcessed(state.iterations() * n);
}

// ============================================================================
// Unary Operations (Transcendentals)
// ============================================================================

static void BM_UnaryExp_Float(benchmark::State &state) {
    const size_t n = static_cast<size_t>(state.range(0));
    // Use smaller values to avoid overflow
    auto a = axiom::Tensor::uniform(-2.0f, 2.0f, {n});
    axiom::Tensor b;

    for (auto _ : state) {
        b = axiom::ops::exp(a);
        benchmark::DoNotOptimize(b.data());
    }

    state.SetBytesProcessed(state.iterations() * n * sizeof(float) * 2);
    state.SetItemsProcessed(state.iterations() * n);
}

static void BM_UnaryLog_Float(benchmark::State &state) {
    const size_t n = static_cast<size_t>(state.range(0));
    // Use positive values for log
    auto a = axiom::Tensor::uniform(0.1f, 10.0f, {n});
    axiom::Tensor b;

    for (auto _ : state) {
        b = axiom::ops::log(a);
        benchmark::DoNotOptimize(b.data());
    }

    state.SetBytesProcessed(state.iterations() * n * sizeof(float) * 2);
    state.SetItemsProcessed(state.iterations() * n);
}

static void BM_UnarySqrt_Float(benchmark::State &state) {
    const size_t n = static_cast<size_t>(state.range(0));
    auto a = axiom::Tensor::uniform(0.0f, 100.0f, {n});
    axiom::Tensor b;

    for (auto _ : state) {
        b = axiom::ops::sqrt(a);
        benchmark::DoNotOptimize(b.data());
    }

    state.SetBytesProcessed(state.iterations() * n * sizeof(float) * 2);
    state.SetItemsProcessed(state.iterations() * n);
}

static void BM_UnaryTanh_Float(benchmark::State &state) {
    const size_t n = static_cast<size_t>(state.range(0));
    auto a = axiom::Tensor::randn({n});
    axiom::Tensor b;

    for (auto _ : state) {
        b = axiom::ops::tanh(a);
        benchmark::DoNotOptimize(b.data());
    }

    state.SetBytesProcessed(state.iterations() * n * sizeof(float) * 2);
    state.SetItemsProcessed(state.iterations() * n);
}

static void BM_UnarySin_Float(benchmark::State &state) {
    const size_t n = static_cast<size_t>(state.range(0));
    auto a = axiom::Tensor::randn({n});
    axiom::Tensor b;

    for (auto _ : state) {
        b = axiom::ops::sin(a);
        benchmark::DoNotOptimize(b.data());
    }

    state.SetBytesProcessed(state.iterations() * n * sizeof(float) * 2);
    state.SetItemsProcessed(state.iterations() * n);
}

// ============================================================================
// Reductions
// ============================================================================

static void BM_ReduceSum_Float(benchmark::State &state) {
    const size_t n = static_cast<size_t>(state.range(0));
    auto a = axiom::Tensor::randn({n});
    axiom::Tensor b;

    for (auto _ : state) {
        b = axiom::ops::sum(a);
        benchmark::DoNotOptimize(b.data());
    }

    state.SetBytesProcessed(state.iterations() * n * sizeof(float));
    state.SetItemsProcessed(state.iterations() * n);
}

static void BM_ReduceMax_Float(benchmark::State &state) {
    const size_t n = static_cast<size_t>(state.range(0));
    auto a = axiom::Tensor::randn({n});
    axiom::Tensor b;

    for (auto _ : state) {
        b = axiom::ops::max(a);
        benchmark::DoNotOptimize(b.data());
    }

    state.SetBytesProcessed(state.iterations() * n * sizeof(float));
    state.SetItemsProcessed(state.iterations() * n);
}

// ============================================================================
// Activation Functions
// ============================================================================

static void BM_ActivationReLU_Float(benchmark::State &state) {
    const size_t n = static_cast<size_t>(state.range(0));
    auto a = axiom::Tensor::randn({n});
    axiom::Tensor b;

    for (auto _ : state) {
        b = axiom::ops::relu(a);
        benchmark::DoNotOptimize(b.data());
    }

    state.SetBytesProcessed(state.iterations() * n * sizeof(float) * 2);
    state.SetItemsProcessed(state.iterations() * n);
}

static void BM_ActivationGELU_Float(benchmark::State &state) {
    const size_t n = static_cast<size_t>(state.range(0));
    auto a = axiom::Tensor::randn({n});
    axiom::Tensor b;

    for (auto _ : state) {
        b = axiom::ops::gelu(a);
        benchmark::DoNotOptimize(b.data());
    }

    state.SetBytesProcessed(state.iterations() * n * sizeof(float) * 2);
    state.SetItemsProcessed(state.iterations() * n);
}

static void BM_ActivationSigmoid_Float(benchmark::State &state) {
    const size_t n = static_cast<size_t>(state.range(0));
    auto a = axiom::Tensor::randn({n});
    axiom::Tensor b;

    for (auto _ : state) {
        b = axiom::ops::sigmoid(a);
        benchmark::DoNotOptimize(b.data());
    }

    state.SetBytesProcessed(state.iterations() * n * sizeof(float) * 2);
    state.SetItemsProcessed(state.iterations() * n);
}

static void BM_ActivationSiLU_Float(benchmark::State &state) {
    const size_t n = static_cast<size_t>(state.range(0));
    auto a = axiom::Tensor::randn({n});
    axiom::Tensor b;

    for (auto _ : state) {
        b = axiom::ops::silu(a);
        benchmark::DoNotOptimize(b.data());
    }

    state.SetBytesProcessed(state.iterations() * n * sizeof(float) * 2);
    state.SetItemsProcessed(state.iterations() * n);
}

// ============================================================================
// Double Precision Variants
// ============================================================================

static void BM_BinaryAdd_Double(benchmark::State &state) {
    const size_t n = static_cast<size_t>(state.range(0));
    auto a = axiom::Tensor::randn({n}, axiom::DType::Float64);
    auto b = axiom::Tensor::randn({n}, axiom::DType::Float64);
    axiom::Tensor c;

    for (auto _ : state) {
        c = a + b;
        benchmark::DoNotOptimize(c.data());
    }

    state.SetBytesProcessed(state.iterations() * n * sizeof(double) * 3);
    state.SetItemsProcessed(state.iterations() * n);
}

static void BM_UnaryExp_Double(benchmark::State &state) {
    const size_t n = static_cast<size_t>(state.range(0));
    auto a = axiom::Tensor::uniform(-2.0, 2.0, {n}, axiom::DType::Float64);
    axiom::Tensor b;

    for (auto _ : state) {
        b = axiom::ops::exp(a);
        benchmark::DoNotOptimize(b.data());
    }

    state.SetBytesProcessed(state.iterations() * n * sizeof(double) * 2);
    state.SetItemsProcessed(state.iterations() * n);
}

static void BM_ReduceSum_Double(benchmark::State &state) {
    const size_t n = static_cast<size_t>(state.range(0));
    auto a = axiom::Tensor::randn({n}, axiom::DType::Float64);
    axiom::Tensor b;

    for (auto _ : state) {
        b = axiom::ops::sum(a);
        benchmark::DoNotOptimize(b.data());
    }

    state.SetBytesProcessed(state.iterations() * n * sizeof(double));
    state.SetItemsProcessed(state.iterations() * n);
}

static void BM_ActivationGELU_Double(benchmark::State &state) {
    const size_t n = static_cast<size_t>(state.range(0));
    auto a = axiom::Tensor::randn({n}, axiom::DType::Float64);
    axiom::Tensor b;

    for (auto _ : state) {
        b = axiom::ops::gelu(a);
        benchmark::DoNotOptimize(b.data());
    }

    state.SetBytesProcessed(state.iterations() * n * sizeof(double) * 2);
    state.SetItemsProcessed(state.iterations() * n);
}

} // namespace

// Sizes: 1K, 16K, 256K, 1M, 4M elements
#define SIMD_BENCHMARK_SIZES \
    ->Arg(1 << 10)           \
    ->Arg(1 << 14)           \
    ->Arg(1 << 18)           \
    ->Arg(1 << 20)           \
    ->Arg(1 << 22)           \
    ->Unit(benchmark::kMicrosecond)

// Binary ops
BENCHMARK(BM_BinaryAdd_Float) SIMD_BENCHMARK_SIZES;
BENCHMARK(BM_BinaryMul_Float) SIMD_BENCHMARK_SIZES;

// Unary transcendentals
BENCHMARK(BM_UnaryExp_Float) SIMD_BENCHMARK_SIZES;
BENCHMARK(BM_UnaryLog_Float) SIMD_BENCHMARK_SIZES;
BENCHMARK(BM_UnarySqrt_Float) SIMD_BENCHMARK_SIZES;
BENCHMARK(BM_UnaryTanh_Float) SIMD_BENCHMARK_SIZES;
BENCHMARK(BM_UnarySin_Float) SIMD_BENCHMARK_SIZES;

// Reductions
BENCHMARK(BM_ReduceSum_Float) SIMD_BENCHMARK_SIZES;
BENCHMARK(BM_ReduceMax_Float) SIMD_BENCHMARK_SIZES;

// Activations
BENCHMARK(BM_ActivationReLU_Float) SIMD_BENCHMARK_SIZES;
BENCHMARK(BM_ActivationGELU_Float) SIMD_BENCHMARK_SIZES;
BENCHMARK(BM_ActivationSigmoid_Float) SIMD_BENCHMARK_SIZES;
BENCHMARK(BM_ActivationSiLU_Float) SIMD_BENCHMARK_SIZES;

// Double precision
BENCHMARK(BM_BinaryAdd_Double) SIMD_BENCHMARK_SIZES;
BENCHMARK(BM_UnaryExp_Double) SIMD_BENCHMARK_SIZES;
BENCHMARK(BM_ReduceSum_Double) SIMD_BENCHMARK_SIZES;
BENCHMARK(BM_ActivationGELU_Double) SIMD_BENCHMARK_SIZES;

BENCHMARK_MAIN();
