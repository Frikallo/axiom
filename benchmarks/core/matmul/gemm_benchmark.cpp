#include <benchmark/benchmark.h>

#include <axiom/axiom.hpp>
#include <benchmark_utils.hpp>

using namespace axiom;
using namespace axiom::ops;

// ============================================================================
// CPU GEMM Benchmarks
// ============================================================================

static void BM_GEMM_CPU_Float32(benchmark::State& state) {
    const int64_t m = state.range(0);
    const int64_t n = state.range(1);
    const int64_t k = state.range(2);

    auto a = Tensor::randn({static_cast<size_t>(m), static_cast<size_t>(k)},
                           DType::Float32, Device::CPU);
    auto b = Tensor::randn({static_cast<size_t>(k), static_cast<size_t>(n)},
                           DType::Float32, Device::CPU);

    // Warmup
    for (int i = 0; i < 3; ++i) {
        auto c = matmul(a, b);
    }

    for (auto _ : state) {
        auto c = matmul(a, b);
        benchmark::DoNotOptimize(c.data());
    }

    bench::set_matmul_counters(state, m, n, k, sizeof(float));
}

static void BM_GEMM_CPU_Float64(benchmark::State& state) {
    const int64_t m = state.range(0);
    const int64_t n = state.range(1);
    const int64_t k = state.range(2);

    auto a = Tensor::randn({static_cast<size_t>(m), static_cast<size_t>(k)},
                           DType::Float64, Device::CPU);
    auto b = Tensor::randn({static_cast<size_t>(k), static_cast<size_t>(n)},
                           DType::Float64, Device::CPU);

    // Warmup
    for (int i = 0; i < 3; ++i) {
        auto c = matmul(a, b);
    }

    for (auto _ : state) {
        auto c = matmul(a, b);
        benchmark::DoNotOptimize(c.data());
    }

    bench::set_matmul_counters(state, m, n, k, sizeof(double));
}

// ============================================================================
// GPU GEMM Benchmarks (Metal)
// ============================================================================

#ifdef AXIOM_METAL_SUPPORT

static void BM_GEMM_GPU_Float32(benchmark::State& state) {
    if (!bench::gpu_available()) {
        state.SkipWithError("GPU not available");
        return;
    }

    const int64_t m = state.range(0);
    const int64_t n = state.range(1);
    const int64_t k = state.range(2);

    auto a = Tensor::randn({static_cast<size_t>(m), static_cast<size_t>(k)},
                           DType::Float32, Device::GPU);
    auto b = Tensor::randn({static_cast<size_t>(k), static_cast<size_t>(n)},
                           DType::Float32, Device::GPU);

    // Warmup
    bench::warmup_gpu();
    for (int i = 0; i < 3; ++i) {
        auto c = matmul(a, b);
        c.cpu();  // Sync
    }

    for (auto _ : state) {
        auto c = matmul(a, b);
        // Sync to ensure operation completes
        auto c_cpu = c.cpu();
        benchmark::DoNotOptimize(c_cpu.data());
    }

    bench::set_matmul_counters(state, m, n, k, sizeof(float));
}

static void BM_GEMM_GPU_Float16(benchmark::State& state) {
    if (!bench::gpu_available()) {
        state.SkipWithError("GPU not available");
        return;
    }

    const int64_t m = state.range(0);
    const int64_t n = state.range(1);
    const int64_t k = state.range(2);

    auto a = Tensor::randn({static_cast<size_t>(m), static_cast<size_t>(k)},
                           DType::Float16, Device::GPU);
    auto b = Tensor::randn({static_cast<size_t>(k), static_cast<size_t>(n)},
                           DType::Float16, Device::GPU);

    // Warmup
    bench::warmup_gpu();
    for (int i = 0; i < 3; ++i) {
        auto c = matmul(a, b);
        c.cpu();  // Sync
    }

    for (auto _ : state) {
        auto c = matmul(a, b);
        // Sync to ensure operation completes
        auto c_cpu = c.cpu();
        benchmark::DoNotOptimize(c_cpu.data());
    }

    bench::set_matmul_counters(state, m, n, k, sizeof(uint16_t));
}

static void BM_GEMM_GPU_TransposeA(benchmark::State& state) {
    if (!bench::gpu_available()) {
        state.SkipWithError("GPU not available");
        return;
    }

    const int64_t m = state.range(0);
    const int64_t n = state.range(1);
    const int64_t k = state.range(2);

    // Create A as (K x M) so transpose gives (M x K)
    auto a = Tensor::randn({static_cast<size_t>(k), static_cast<size_t>(m)},
                           DType::Float32, Device::GPU);
    auto b = Tensor::randn({static_cast<size_t>(k), static_cast<size_t>(n)},
                           DType::Float32, Device::GPU);

    // Warmup
    bench::warmup_gpu();
    for (int i = 0; i < 3; ++i) {
        auto c = matmul(a, b, /*transpose_a=*/true);
        c.cpu();
    }

    for (auto _ : state) {
        auto c = matmul(a, b, /*transpose_a=*/true);
        auto c_cpu = c.cpu();
        benchmark::DoNotOptimize(c_cpu.data());
    }

    bench::set_matmul_counters(state, m, n, k, sizeof(float));
}

static void BM_GEMM_GPU_TransposeB(benchmark::State& state) {
    if (!bench::gpu_available()) {
        state.SkipWithError("GPU not available");
        return;
    }

    const int64_t m = state.range(0);
    const int64_t n = state.range(1);
    const int64_t k = state.range(2);

    auto a = Tensor::randn({static_cast<size_t>(m), static_cast<size_t>(k)},
                           DType::Float32, Device::GPU);
    // Create B as (N x K) so transpose gives (K x N)
    auto b = Tensor::randn({static_cast<size_t>(n), static_cast<size_t>(k)},
                           DType::Float32, Device::GPU);

    // Warmup
    bench::warmup_gpu();
    for (int i = 0; i < 3; ++i) {
        auto c = matmul(a, b, /*transpose_a=*/false, /*transpose_b=*/true);
        c.cpu();
    }

    for (auto _ : state) {
        auto c = matmul(a, b, /*transpose_a=*/false, /*transpose_b=*/true);
        auto c_cpu = c.cpu();
        benchmark::DoNotOptimize(c_cpu.data());
    }

    bench::set_matmul_counters(state, m, n, k, sizeof(float));
}

static void BM_GEMM_GPU_Batched(benchmark::State& state) {
    if (!bench::gpu_available()) {
        state.SkipWithError("GPU not available");
        return;
    }

    const int64_t batch = state.range(0);
    const int64_t m = state.range(1);
    const int64_t n = state.range(2);
    const int64_t k = state.range(3);

    auto a = Tensor::randn({static_cast<size_t>(batch), static_cast<size_t>(m),
                            static_cast<size_t>(k)},
                           DType::Float32, Device::GPU);
    auto b = Tensor::randn({static_cast<size_t>(batch), static_cast<size_t>(k),
                            static_cast<size_t>(n)},
                           DType::Float32, Device::GPU);

    // Warmup
    bench::warmup_gpu();
    for (int i = 0; i < 3; ++i) {
        auto c = matmul(a, b);
        c.cpu();
    }

    for (auto _ : state) {
        auto c = matmul(a, b);
        auto c_cpu = c.cpu();
        benchmark::DoNotOptimize(c_cpu.data());
    }

    // Total FLOPs = batch * 2 * M * N * K
    int64_t total_m = batch * m;
    bench::set_matmul_counters(state, total_m, n, k, sizeof(float));
}

#endif  // AXIOM_METAL_SUPPORT

// ============================================================================
// CPU vs GPU Comparison (same sizes)
// ============================================================================

#ifdef AXIOM_METAL_SUPPORT

static void BM_Compare_CPU(benchmark::State& state) {
    const int64_t size = state.range(0);

    auto a =
        Tensor::randn({static_cast<size_t>(size), static_cast<size_t>(size)},
                      DType::Float32, Device::CPU);
    auto b =
        Tensor::randn({static_cast<size_t>(size), static_cast<size_t>(size)},
                      DType::Float32, Device::CPU);

    for (int i = 0; i < 3; ++i) {
        auto c = matmul(a, b);
    }

    for (auto _ : state) {
        auto c = matmul(a, b);
        benchmark::DoNotOptimize(c.data());
    }

    bench::set_matmul_counters(state, size, size, size, sizeof(float));
}

static void BM_Compare_GPU(benchmark::State& state) {
    if (!bench::gpu_available()) {
        state.SkipWithError("GPU not available");
        return;
    }

    const int64_t size = state.range(0);

    auto a =
        Tensor::randn({static_cast<size_t>(size), static_cast<size_t>(size)},
                      DType::Float32, Device::GPU);
    auto b =
        Tensor::randn({static_cast<size_t>(size), static_cast<size_t>(size)},
                      DType::Float32, Device::GPU);

    bench::warmup_gpu();
    for (int i = 0; i < 3; ++i) {
        auto c = matmul(a, b);
        c.cpu();
    }

    for (auto _ : state) {
        auto c = matmul(a, b);
        auto c_cpu = c.cpu();
        benchmark::DoNotOptimize(c_cpu.data());
    }

    bench::set_matmul_counters(state, size, size, size, sizeof(float));
}

#endif  // AXIOM_METAL_SUPPORT

// ============================================================================
// Benchmark Registration
// ============================================================================

// CPU benchmarks - square matrices
BENCHMARK(BM_GEMM_CPU_Float32)
    ->Apply(bench::square_matrix_args)
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_GEMM_CPU_Float64)
    ->Apply(bench::square_matrix_args)
    ->Unit(benchmark::kMillisecond);

// CPU benchmarks - transformer sizes
BENCHMARK(BM_GEMM_CPU_Float32)
    ->Apply(bench::transformer_args)
    ->Unit(benchmark::kMillisecond);

#ifdef AXIOM_METAL_SUPPORT

// GPU benchmarks - square matrices
BENCHMARK(BM_GEMM_GPU_Float32)
    ->Apply(bench::square_matrix_args)
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_GEMM_GPU_Float16)
    ->Apply(bench::square_matrix_args)
    ->Unit(benchmark::kMillisecond);

// GPU benchmarks - transformer sizes
BENCHMARK(BM_GEMM_GPU_Float32)
    ->Apply(bench::transformer_args)
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_GEMM_GPU_Float16)
    ->Apply(bench::transformer_args)
    ->Unit(benchmark::kMillisecond);

// Transpose overhead benchmarks
BENCHMARK(BM_GEMM_GPU_TransposeA)
    ->Args({1024, 1024, 1024})
    ->Args({2048, 2048, 2048})
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_GEMM_GPU_TransposeB)
    ->Args({1024, 1024, 1024})
    ->Args({2048, 2048, 2048})
    ->Unit(benchmark::kMillisecond);

// Batched matmul (attention-style)
BENCHMARK(BM_GEMM_GPU_Batched)
    ->Args({8, 512, 64, 512})    // 8 attention heads
    ->Args({12, 512, 64, 512})   // BERT-base heads
    ->Args({16, 512, 64, 512})   // More heads
    ->Args({32, 2048, 64, 2048}) // Large batch attention
    ->Unit(benchmark::kMillisecond);

// CPU vs GPU comparison
BENCHMARK(BM_Compare_CPU)
    ->Arg(64)
    ->Arg(128)
    ->Arg(256)
    ->Arg(512)
    ->Arg(1024)
    ->Arg(2048)
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_Compare_GPU)
    ->Arg(64)
    ->Arg(128)
    ->Arg(256)
    ->Arg(512)
    ->Arg(1024)
    ->Arg(2048)
    ->Unit(benchmark::kMillisecond);

#endif  // AXIOM_METAL_SUPPORT
