#pragma once

#include <benchmark/benchmark.h>

#include <axiom/axiom.hpp>
#include <cmath>
#include <string>
#include <vector>

namespace axiom::bench {

// ============================================================================
// Standard Matrix Sizes for Benchmarking
// ============================================================================

// Small sizes (overhead-dominated)
constexpr int kSmallSizes[] = {32, 64, 128};

// Medium sizes (typical neural network layers)
constexpr int kMediumSizes[] = {256, 512};

// Large sizes (compute-bound)
constexpr int kLargeSizes[] = {1024, 2048, 4096};

// All standard square sizes
constexpr int kAllSquareSizes[] = {32, 64, 128, 256, 512, 1024, 2048, 4096};

// Transformer-specific sizes
struct TransformerSize {
    int m, n, k;
    const char* name;
};

// Common transformer workload shapes
constexpr TransformerSize kTransformerSizes[] = {
    // Attention: batch * heads * seq_len, head_dim, seq_len
    {512, 64, 512, "attention_head"},
    {2048, 64, 2048, "attention_large"},

    // FFN: batch * seq_len, hidden_dim, model_dim
    {2048, 768, 768, "ffn_bert_base"},
    {2048, 3072, 768, "ffn_bert_up"},
    {2048, 768, 3072, "ffn_bert_down"},
    {2048, 4096, 1024, "ffn_bert_large_up"},
    {2048, 1024, 4096, "ffn_bert_large_down"},

    // LLM decode: single token inference
    {1, 2048, 4096, "llm_decode_small"},
    {1, 4096, 4096, "llm_decode_med"},
    {1, 8192, 4096, "llm_decode_large"},
    {1, 4096, 11008, "llm_decode_llama_up"},
    {1, 11008, 4096, "llm_decode_llama_down"},

    // LLM prefill: batch processing
    {512, 4096, 4096, "llm_prefill"},
    {2048, 4096, 4096, "llm_prefill_large"},
};

// ============================================================================
// Performance Metrics
// ============================================================================

/// Calculate GFLOPS for matrix multiplication C = A @ B
/// where A is (M x K) and B is (K x N)
inline double matmul_gflops(int64_t m, int64_t n, int64_t k, double seconds) {
    // 2*M*N*K operations (multiply + add for each output element)
    double flops = 2.0 * static_cast<double>(m) * static_cast<double>(n) *
                   static_cast<double>(k);
    return (flops / seconds) / 1e9;
}

/// Calculate bandwidth in GB/s for matrix multiplication
/// Assumes reading A, B and writing C once
inline double matmul_bandwidth_gbps(int64_t m,
                                    int64_t n,
                                    int64_t k,
                                    size_t element_bytes,
                                    double seconds) {
    // Read A (M*K), read B (K*N), write C (M*N)
    double bytes = static_cast<double>(element_bytes) *
                   (static_cast<double>(m * k) + static_cast<double>(k * n) +
                    static_cast<double>(m * n));
    return (bytes / seconds) / 1e9;
}

// ============================================================================
// Benchmark Registration Helpers
// ============================================================================

/// Generate arguments for square matrix benchmarks
inline void square_matrix_args(benchmark::internal::Benchmark* b) {
    for (int size : kAllSquareSizes) {
        b->Args({size, size, size});
    }
}

/// Generate arguments for transformer workload benchmarks
inline void transformer_args(benchmark::internal::Benchmark* b) {
    for (const auto& size : kTransformerSizes) {
        b->Args({size.m, size.n, size.k});
    }
}

/// Generate arguments for both square and transformer workloads
inline void all_matrix_args(benchmark::internal::Benchmark* b) {
    square_matrix_args(b);
    transformer_args(b);
}

// ============================================================================
// GPU Availability Check
// ============================================================================

/// Check if GPU (Metal) is available for benchmarking
inline bool gpu_available() {
#ifdef AXIOM_METAL_SUPPORT
    try {
        // Try to create a small tensor on GPU
        auto t = Tensor::zeros({2, 2}, DType::Float32, Device::GPU);
        return true;
    } catch (...) {
        return false;
    }
#else
    return false;
#endif
}

// ============================================================================
// Warmup Helpers
// ============================================================================

/// Warmup GPU by running a few operations
inline void warmup_gpu(int iterations = 5) {
#ifdef AXIOM_METAL_SUPPORT
    if (!gpu_available())
        return;

    auto a = Tensor::randn({256, 256}, DType::Float32, Device::GPU);
    auto b = Tensor::randn({256, 256}, DType::Float32, Device::GPU);

    for (int i = 0; i < iterations; ++i) {
        auto c = ops::matmul(a, b);
        // Ensure completion
        c.cpu();
    }
#else
    (void)iterations;
#endif
}

/// Warmup CPU caches
inline void warmup_cpu(int iterations = 5) {
    auto a = Tensor::randn({256, 256}, DType::Float32, Device::CPU);
    auto b = Tensor::randn({256, 256}, DType::Float32, Device::CPU);

    for (int i = 0; i < iterations; ++i) {
        auto c = ops::matmul(a, b);
    }
}

// ============================================================================
// Benchmark State Helpers
// ============================================================================

/// Set standard counters for matmul benchmarks
inline void set_matmul_counters(benchmark::State& state,
                                int64_t m,
                                int64_t n,
                                int64_t k,
                                size_t element_bytes = 4) {
    // Total FLOPs per iteration
    int64_t flops = 2 * m * n * k;
    state.counters["M"] = static_cast<double>(m);
    state.counters["N"] = static_cast<double>(n);
    state.counters["K"] = static_cast<double>(k);

    // GFLOPS = FLOPs / (time in seconds * 1e9)
    state.counters["GFLOPS"] =
        benchmark::Counter(static_cast<double>(flops),
                           benchmark::Counter::kIsIterationInvariantRate,
                           benchmark::Counter::kIs1000);

    // Bytes processed (read A, B; write C)
    int64_t bytes =
        static_cast<int64_t>(element_bytes) * (m * k + k * n + m * n);
    state.counters["Bandwidth"] =
        benchmark::Counter(static_cast<double>(bytes),
                           benchmark::Counter::kIsIterationInvariantRate,
                           benchmark::Counter::kIs1024);
}

// ============================================================================
// Custom Benchmark Names
// ============================================================================

/// Generate a descriptive name for matrix dimensions
inline std::string matrix_name(int64_t m, int64_t n, int64_t k) {
    if (m == n && n == k) {
        return std::to_string(m) + "x" + std::to_string(m);
    }
    return std::to_string(m) + "x" + std::to_string(n) + "x" +
           std::to_string(k);
}

}  // namespace axiom::bench
