// Fusion Pattern Benchmarks
// Tests the performance of fused SIMD kernels and general fusion pass.
//
// Combines benchmarks for:
// - Specific fused patterns (AddReLU, SubAbs, MulAdd, etc.)
// - General fusion chain benchmarks
//
// Usage:
//   make benchmarks
//   ./build/benchmarks/bench_fusion
//
// To compare lazy vs eager mode, run twice:
//   ./build/benchmarks/bench_fusion                    # Lazy mode (default)
//   AXIOM_EAGER_MODE=1 ./build/benchmarks/bench_fusion # Eager mode

#include "axiom/axiom.hpp"
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>

using namespace axiom;
using namespace std::chrono;

// =============================================================================
// Configuration
// =============================================================================

bool is_eager_mode() {
    const char* env = std::getenv("AXIOM_EAGER_MODE");
    return env != nullptr && std::string(env) == "1";
}

// =============================================================================
// Timing Infrastructure
// =============================================================================

template <typename F>
double time_ms(F&& func, int warmup = 5, int iterations = 20) {
    for (int i = 0; i < warmup; ++i)
        func();

    auto start = high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i)
        func();
    auto end = high_resolution_clock::now();

    return duration<double, std::milli>(end - start).count() / iterations;
}

struct BenchResult {
    std::string name;
    double time_ms;
};

std::vector<BenchResult> results;

// =============================================================================
// Binary + Unary Fused Patterns (2 inputs)
// =============================================================================

void bench_add_relu(size_t size) {
    auto a = Tensor::randn({size, size});
    auto b = Tensor::randn({size, size});

    double time = time_ms([&]() {
        auto result = ops::relu(ops::add(a, b));
        (void)result.item<float>({0, 0});
    });

    results.push_back({"AddReLU " + std::to_string(size), time});
}

void bench_sub_abs(size_t size) {
    auto a = Tensor::randn({size, size});
    auto b = Tensor::randn({size, size});

    double time = time_ms([&]() {
        auto result = ops::abs(ops::subtract(a, b));
        (void)result.item<float>({0, 0});
    });

    results.push_back({"SubAbs " + std::to_string(size), time});
}

void bench_add_square(size_t size) {
    auto a = Tensor::randn({size, size});
    auto b = Tensor::randn({size, size});

    double time = time_ms([&]() {
        auto result = ops::square(ops::add(a, b));
        (void)result.item<float>({0, 0});
    });

    results.push_back({"AddSquare " + std::to_string(size), time});
}

void bench_sub_square(size_t size) {
    auto a = Tensor::randn({size, size});
    auto b = Tensor::randn({size, size});

    double time = time_ms([&]() {
        auto result = ops::square(ops::subtract(a, b));
        (void)result.item<float>({0, 0});
    });

    results.push_back({"SubSquare " + std::to_string(size), time});
}

void bench_mul_relu(size_t size) {
    auto a = Tensor::randn({size, size});
    auto b = Tensor::randn({size, size});

    double time = time_ms([&]() {
        auto result = ops::relu(ops::multiply(a, b));
        (void)result.item<float>({0, 0});
    });

    results.push_back({"MulReLU " + std::to_string(size), time});
}

void bench_add_sigmoid(size_t size) {
    auto a = Tensor::randn({size, size});
    auto b = Tensor::randn({size, size});

    double time = time_ms([&]() {
        auto result = ops::sigmoid(ops::add(a, b));
        (void)result.item<float>({0, 0});
    });

    results.push_back({"AddSigmoid " + std::to_string(size), time});
}

// =============================================================================
// Ternary Fused Patterns (3 inputs)
// =============================================================================

void bench_mul_add(size_t size) {
    auto a = Tensor::randn({size, size});
    auto b = Tensor::randn({size, size});
    auto c = Tensor::randn({size, size});

    double time = time_ms([&]() {
        auto result = ops::add(ops::multiply(a, b), c);
        (void)result.item<float>({0, 0});
    });

    results.push_back({"MulAdd(FMA) " + std::to_string(size), time});
}

void bench_scale_shift_relu(size_t size) {
    auto x = Tensor::randn({size, size});
    auto scale = Tensor::randn({size, size});
    auto bias = Tensor::randn({size, size});

    double time = time_ms([&]() {
        auto result = ops::relu(ops::add(ops::multiply(x, scale), bias));
        (void)result.item<float>({0, 0});
    });

    results.push_back({"ScaleShiftReLU " + std::to_string(size), time});
}

void bench_add_mul_relu(size_t size) {
    auto a = Tensor::randn({size, size});
    auto b = Tensor::randn({size, size});
    auto c = Tensor::randn({size, size});

    double time = time_ms([&]() {
        auto result = ops::relu(ops::multiply(ops::add(a, b), c));
        (void)result.item<float>({0, 0});
    });

    results.push_back({"AddMulReLU " + std::to_string(size), time});
}

// =============================================================================
// Integer Pattern Benchmarks
// =============================================================================

void bench_add_relu_int32(size_t size) {
    auto a = Tensor::full<int32_t>({size, size}, -5);
    auto b = Tensor::full<int32_t>({size, size}, 10);

    double time = time_ms([&]() {
        auto result = ops::relu(ops::add(a, b));
        (void)result.item<int32_t>({0, 0});
    });

    results.push_back({"AddReLU(i32) " + std::to_string(size), time});
}

void bench_sub_abs_int32(size_t size) {
    auto a = Tensor::full<int32_t>({size, size}, 5);
    auto b = Tensor::full<int32_t>({size, size}, 10);

    double time = time_ms([&]() {
        auto result = ops::abs(ops::subtract(a, b));
        (void)result.item<int32_t>({0, 0});
    });

    results.push_back({"SubAbs(i32) " + std::to_string(size), time});
}

// =============================================================================
// General Fusion Chain Benchmarks (no specific SIMD kernel)
// =============================================================================

void bench_unary_chain(size_t size) {
    auto x = Tensor::randn({size, size});

    double time = time_ms([&]() {
        auto result =
            ops::tanh(ops::sigmoid(ops::relu(ops::sqrt(ops::abs(x)))));
        (void)result.item<float>({0, 0});
    });

    results.push_back({"UnaryChain(5) " + std::to_string(size), time});
}

void bench_long_chain(size_t size) {
    auto x = Tensor::randn({size, size});

    double time = time_ms([&]() {
        auto result = ops::tanh(ops::sigmoid(
            ops::relu(ops::exp(ops::log(ops::abs(ops::sqrt(ops::abs(x))))))));
        (void)result.item<float>({0, 0});
    });

    results.push_back({"LongChain(8) " + std::to_string(size), time});
}

// =============================================================================
// Output
// =============================================================================

void print_results() {
    std::cout << "\n" << std::string(56, '=') << "\n";
    std::cout << "              FUSION BENCHMARK RESULTS\n";
    std::cout << std::string(56, '=') << "\n\n";

    std::string mode = is_eager_mode() ? "EAGER" : "LAZY";
    std::cout << "Mode: " << mode << "\n\n";

    std::cout << std::left << std::setw(30) << "Pattern" << std::right
              << std::setw(14) << "Time (ms)"
              << "\n";
    std::cout << std::string(44, '-') << "\n";

    double total = 0;
    for (const auto& r : results) {
        std::cout << std::left << std::setw(30) << r.name << std::right
                  << std::fixed << std::setprecision(3) << std::setw(14)
                  << r.time_ms << "\n";
        total += r.time_ms;
    }

    std::cout << std::string(44, '-') << "\n";
    std::cout << std::left << std::setw(30) << "TOTAL" << std::right
              << std::fixed << std::setprecision(3) << std::setw(14)
              << total << "\n";

    std::cout << "\n" << std::string(56, '=') << "\n";
    std::cout << "To compare modes:\n";
    std::cout << "  ./bench_fusion                    # Lazy mode\n";
    std::cout << "  AXIOM_EAGER_MODE=1 ./bench_fusion # Eager mode\n";
    std::cout << std::string(56, '=') << "\n\n";
}

int main() {
    ops::OperationRegistry::initialize_builtin_operations();

    std::cout << "\n";
    std::cout << std::string(56, '=') << "\n";
    std::cout << "           AXIOM FUSION PATTERN BENCHMARK\n";
    std::cout << std::string(56, '=') << "\n";
    std::cout << "\nBenchmarking fused SIMD kernels.\n";
    std::cout << "(warmup=5, iterations=20 per measurement)\n\n";

    std::vector<size_t> sizes = {256, 512, 1024, 2048};

    for (size_t size : sizes) {
        std::cout << "Running size " << size << "x" << size << "...\n";

        // Binary + Unary patterns
        bench_add_relu(size);
        bench_sub_abs(size);
        bench_add_square(size);
        bench_sub_square(size);
        bench_mul_relu(size);
        bench_add_sigmoid(size);

        // Ternary patterns
        bench_mul_add(size);
        bench_scale_shift_relu(size);
        bench_add_mul_relu(size);

        // Integer patterns
        bench_add_relu_int32(size);
        bench_sub_abs_int32(size);

        // General fusion chains
        bench_unary_chain(size);
        bench_long_chain(size);
    }

    print_results();

    return 0;
}
