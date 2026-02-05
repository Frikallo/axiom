#include <benchmark/benchmark.h>

#include <axiom/axiom.hpp>
#include <benchmark_utils.hpp>

using namespace axiom;
using namespace axiom::ops;

#ifdef AXIOM_METAL_SUPPORT

// ============================================================================
// GPU Overhead Benchmarks
// These benchmarks are designed to identify specific sources of overhead
// in the Axiom GPU implementation compared to PyTorch MPS.
// ============================================================================

// ----------------------------------------------------------------------------
// Graph Creation Overhead
// Measures the cost of creating a new MPSGraph for each operation
// ----------------------------------------------------------------------------

static void BM_GPU_GraphCreationOverhead(benchmark::State& state) {
    if (!bench::gpu_available()) {
        state.SkipWithError("GPU not available");
        return;
    }

    const int64_t size = state.range(0);

    // Create fresh tensors each iteration to force new graph creation
    bench::warmup_gpu();

    for (auto _ : state) {
        auto a =
            Tensor::randn({static_cast<size_t>(size), static_cast<size_t>(size)},
                          DType::Float32, Device::GPU);
        auto b =
            Tensor::randn({static_cast<size_t>(size), static_cast<size_t>(size)},
                          DType::Float32, Device::GPU);

        auto c = matmul(a, b);
        auto c_cpu = c.cpu();  // Force completion
        benchmark::DoNotOptimize(c_cpu.data());
    }

    bench::set_matmul_counters(state, size, size, size, sizeof(float));
}

// ----------------------------------------------------------------------------
// Repeated Operation Overhead
// Measures performance when the same operation shape is repeated
// (potential graph caching benefit)
// ----------------------------------------------------------------------------

static void BM_GPU_RepeatedOperation(benchmark::State& state) {
    if (!bench::gpu_available()) {
        state.SkipWithError("GPU not available");
        return;
    }

    const int64_t size = state.range(0);

    // Pre-create tensors so graph shape is constant
    auto a =
        Tensor::randn({static_cast<size_t>(size), static_cast<size_t>(size)},
                      DType::Float32, Device::GPU);
    auto b =
        Tensor::randn({static_cast<size_t>(size), static_cast<size_t>(size)},
                      DType::Float32, Device::GPU);

    bench::warmup_gpu();

    // Warmup with same shapes to potentially trigger caching
    for (int i = 0; i < 10; ++i) {
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

// ----------------------------------------------------------------------------
// Contiguity Overhead
// Measures the overhead of making non-contiguous tensors contiguous
// (GPU gather kernel copy cost)
// ----------------------------------------------------------------------------

static void BM_GPU_ContiguousInput(benchmark::State& state) {
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

static void BM_GPU_TransposedInput(benchmark::State& state) {
    if (!bench::gpu_available()) {
        state.SkipWithError("GPU not available");
        return;
    }

    const int64_t size = state.range(0);

    // Create (size x size) then transpose to get non-contiguous view
    auto a_base =
        Tensor::randn({static_cast<size_t>(size), static_cast<size_t>(size)},
                      DType::Float32, Device::GPU);
    auto b_base =
        Tensor::randn({static_cast<size_t>(size), static_cast<size_t>(size)},
                      DType::Float32, Device::GPU);

    // Transpose creates non-contiguous views
    auto a = a_base.transpose();
    auto b = b_base.transpose();

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

// ----------------------------------------------------------------------------
// Synchronization Overhead
// Measures the cost of waitUntilCompleted() calls
// ----------------------------------------------------------------------------

static void BM_GPU_ChainedOperations(benchmark::State& state) {
    if (!bench::gpu_available()) {
        state.SkipWithError("GPU not available");
        return;
    }

    const int64_t size = state.range(0);
    const int64_t chain_length = state.range(1);

    auto a =
        Tensor::randn({static_cast<size_t>(size), static_cast<size_t>(size)},
                      DType::Float32, Device::GPU);
    auto b =
        Tensor::randn({static_cast<size_t>(size), static_cast<size_t>(size)},
                      DType::Float32, Device::GPU);

    bench::warmup_gpu();

    for (auto _ : state) {
        Tensor result = a;
        for (int64_t i = 0; i < chain_length; ++i) {
            result = matmul(result, b);
        }
        auto result_cpu = result.cpu();  // Only sync at the end
        benchmark::DoNotOptimize(result_cpu.data());
    }

    // Total FLOPs = chain_length * 2 * size^3
    int64_t total_flops = chain_length * 2 * size * size * size;
    state.counters["ChainLength"] = static_cast<double>(chain_length);
    state.counters["GFLOPS"] =
        benchmark::Counter(static_cast<double>(total_flops),
                           benchmark::Counter::kIsIterationInvariantRate,
                           benchmark::Counter::kIs1000);
}

// ----------------------------------------------------------------------------
// Data Transfer Overhead
// Measures CPU-GPU data transfer costs
// ----------------------------------------------------------------------------

static void BM_GPU_TransferToGPU(benchmark::State& state) {
    const int64_t size = state.range(0);

    auto cpu_tensor =
        Tensor::randn({static_cast<size_t>(size), static_cast<size_t>(size)},
                      DType::Float32, Device::CPU);

    bench::warmup_gpu();

    for (auto _ : state) {
        auto gpu_tensor = cpu_tensor.to(Device::GPU);
        // Force transfer completion
        benchmark::DoNotOptimize(gpu_tensor.device());
    }

    int64_t bytes = size * size * static_cast<int64_t>(sizeof(float));
    state.counters["Bandwidth"] =
        benchmark::Counter(static_cast<double>(bytes),
                           benchmark::Counter::kIsIterationInvariantRate,
                           benchmark::Counter::kIs1024);
}

static void BM_GPU_TransferToCPU(benchmark::State& state) {
    if (!bench::gpu_available()) {
        state.SkipWithError("GPU not available");
        return;
    }

    const int64_t size = state.range(0);

    auto gpu_tensor =
        Tensor::randn({static_cast<size_t>(size), static_cast<size_t>(size)},
                      DType::Float32, Device::GPU);

    bench::warmup_gpu();

    for (auto _ : state) {
        auto cpu_tensor = gpu_tensor.cpu();
        benchmark::DoNotOptimize(cpu_tensor.data());
    }

    int64_t bytes = size * size * static_cast<int64_t>(sizeof(float));
    state.counters["Bandwidth"] =
        benchmark::Counter(static_cast<double>(bytes),
                           benchmark::Counter::kIsIterationInvariantRate,
                           benchmark::Counter::kIs1024);
}

// ----------------------------------------------------------------------------
// DType Promotion Overhead
// Measures overhead when non-Float32 types are promoted to Float32
// ----------------------------------------------------------------------------

static void BM_GPU_Float32Native(benchmark::State& state) {
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

    for (auto _ : state) {
        auto c = matmul(a, b);
        auto c_cpu = c.cpu();
        benchmark::DoNotOptimize(c_cpu.data());
    }

    bench::set_matmul_counters(state, size, size, size, sizeof(float));
}

static void BM_GPU_Float16Native(benchmark::State& state) {
    if (!bench::gpu_available()) {
        state.SkipWithError("GPU not available");
        return;
    }

    const int64_t size = state.range(0);

    auto a =
        Tensor::randn({static_cast<size_t>(size), static_cast<size_t>(size)},
                      DType::Float16, Device::GPU);
    auto b =
        Tensor::randn({static_cast<size_t>(size), static_cast<size_t>(size)},
                      DType::Float16, Device::GPU);

    bench::warmup_gpu();

    for (auto _ : state) {
        auto c = matmul(a, b);
        auto c_cpu = c.cpu();
        benchmark::DoNotOptimize(c_cpu.data());
    }

    bench::set_matmul_counters(state, size, size, size, sizeof(uint16_t));
}

// ============================================================================
// Benchmark Registration
// ============================================================================

// Graph creation overhead
BENCHMARK(BM_GPU_GraphCreationOverhead)
    ->Arg(256)
    ->Arg(512)
    ->Arg(1024)
    ->Unit(benchmark::kMillisecond);

// Repeated operation (caching benefit)
BENCHMARK(BM_GPU_RepeatedOperation)
    ->Arg(256)
    ->Arg(512)
    ->Arg(1024)
    ->Arg(2048)
    ->Unit(benchmark::kMillisecond);

// Contiguity overhead comparison
BENCHMARK(BM_GPU_ContiguousInput)
    ->Arg(512)
    ->Arg(1024)
    ->Arg(2048)
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_GPU_TransposedInput)
    ->Arg(512)
    ->Arg(1024)
    ->Arg(2048)
    ->Unit(benchmark::kMillisecond);

// Chained operations (sync overhead)
BENCHMARK(BM_GPU_ChainedOperations)
    ->Args({256, 1})
    ->Args({256, 5})
    ->Args({256, 10})
    ->Args({512, 1})
    ->Args({512, 5})
    ->Args({512, 10})
    ->Unit(benchmark::kMillisecond);

// Data transfer overhead
BENCHMARK(BM_GPU_TransferToGPU)
    ->Arg(256)
    ->Arg(512)
    ->Arg(1024)
    ->Arg(2048)
    ->Arg(4096)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_GPU_TransferToCPU)
    ->Arg(256)
    ->Arg(512)
    ->Arg(1024)
    ->Arg(2048)
    ->Arg(4096)
    ->Unit(benchmark::kMicrosecond);

// DType comparison
BENCHMARK(BM_GPU_Float32Native)
    ->Arg(512)
    ->Arg(1024)
    ->Arg(2048)
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_GPU_Float16Native)
    ->Arg(512)
    ->Arg(1024)
    ->Arg(2048)
    ->Unit(benchmark::kMillisecond);

#else

// Stub for non-Metal platforms
static void BM_GPU_NotAvailable(benchmark::State& state) {
    state.SkipWithError("GPU benchmarks require Metal support (macOS only)");
}

BENCHMARK(BM_GPU_NotAvailable);

#endif  // AXIOM_METAL_SUPPORT
