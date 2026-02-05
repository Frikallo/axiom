// Lazy vs Eager Mode Comprehensive Benchmark
//
// This benchmark compares lazy evaluation (with fusion) against eager mode
// for a comprehensive set of operation patterns.
//
// Usage:
//   make benchmarks
//   ./build/benchmarks/bench_lazy_vs_eager
//
// To compare modes, run twice:
//   ./bench_lazy_vs_eager                    # Lazy mode (default)
//   AXIOM_EAGER_MODE=1 ./bench_lazy_vs_eager # Eager mode

#include "axiom/axiom.hpp"
#include "axiom/tensor_operators.hpp"
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

struct BenchResult {
    std::string name;
    double time_ms;
};

template <typename F>
double time_ms(F&& func, int warmup = 3, int iterations = 10) {
    for (int i = 0; i < warmup; ++i) {
        func();
    }

    auto start = high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        func();
    }
    auto end = high_resolution_clock::now();

    return duration<double, std::milli>(end - start).count() / iterations;
}

// Force materialization - handles both 1D and 2D results
template <typename T>
void materialize(Tensor& t) {
    if (t.ndim() >= 2) {
        (void)t.item<T>({0, 0});
    } else if (t.ndim() == 1) {
        (void)t.item<T>({0});
    } else {
        (void)t.item<T>({});
    }
}

std::vector<BenchResult> results;

// =============================================================================
// Benchmark Functions
// =============================================================================

void bench_add_relu(size_t size) {
    auto a = Tensor::randn({size, size});
    auto b = Tensor::randn({size, size});

    double time = time_ms([&]() {
        auto result = ops::relu(ops::add(a, b));
        materialize<float>(result);
    });

    results.push_back(
        {"AddReLU " + std::to_string(size) + "x" + std::to_string(size), time});
}

void bench_sub_abs(size_t size) {
    auto a = Tensor::randn({size, size});
    auto b = Tensor::randn({size, size});

    double time = time_ms([&]() {
        auto result = ops::abs(ops::subtract(a, b));
        materialize<float>(result);
    });

    results.push_back(
        {"SubAbs " + std::to_string(size) + "x" + std::to_string(size), time});
}

void bench_add_square(size_t size) {
    auto a = Tensor::randn({size, size});
    auto b = Tensor::randn({size, size});

    double time = time_ms([&]() {
        auto result = ops::square(ops::add(a, b));
        materialize<float>(result);
    });

    results.push_back({"AddSquare " + std::to_string(size) + "x" +
                           std::to_string(size),
                       time});
}

void bench_mul_add(size_t size) {
    auto a = Tensor::randn({size, size});
    auto b = Tensor::randn({size, size});
    auto c = Tensor::randn({size, size});

    double time = time_ms([&]() {
        auto result = ops::add(ops::multiply(a, b), c);
        materialize<float>(result);
    });

    results.push_back({"MulAdd (FMA) " + std::to_string(size) + "x" +
                           std::to_string(size),
                       time});
}

void bench_scale_shift_relu(size_t size) {
    auto x = Tensor::randn({size, size});
    auto scale = Tensor::randn({size, size});
    auto bias = Tensor::randn({size, size});

    double time = time_ms([&]() {
        auto result = ops::relu(ops::add(ops::multiply(x, scale), bias));
        materialize<float>(result);
    });

    results.push_back({"ScaleShiftReLU " + std::to_string(size) + "x" +
                           std::to_string(size),
                       time});
}

void bench_long_chain(size_t size) {
    auto x = Tensor::randn({size, size});

    double time = time_ms([&]() {
        auto result = ops::exp(ops::sqrt(ops::abs(ops::relu(x))));
        materialize<float>(result);
    });

    results.push_back({"LongChain(4ops) " + std::to_string(size) + "x" +
                           std::to_string(size),
                       time});
}

void bench_operator_syntax(size_t size) {
    auto a = Tensor::randn({size, size});
    auto b = Tensor::randn({size, size});
    auto c = Tensor::randn({size, size});

    double time = time_ms([&]() {
        auto result = ((a + b) * c).relu();
        materialize<float>(result);
    });

    results.push_back({"((a+b)*c).relu() " + std::to_string(size) + "x" +
                           std::to_string(size),
                       time});
}

void bench_sigmoid_chain(size_t size) {
    auto x = Tensor::randn({size, size});

    double time = time_ms([&]() {
        auto result = ops::multiply(x, ops::sigmoid(x));
        materialize<float>(result);
    });

    results.push_back({"x*sigmoid(x) " + std::to_string(size) + "x" +
                           std::to_string(size),
                       time});
}

void bench_reduction_chain(size_t size) {
    auto x = Tensor::randn({size, size});

    double time = time_ms([&]() {
        auto result = ops::sqrt(ops::sum(ops::relu(x)));
        materialize<float>(result);
    });

    results.push_back({"relu->sum->sqrt " + std::to_string(size) + "x" +
                           std::to_string(size),
                       time});
}

void bench_tanh_exp(size_t size) {
    auto x = Tensor::randn({size, size});

    double time = time_ms([&]() {
        auto result = ops::exp(ops::tanh(x));
        materialize<float>(result);
    });

    results.push_back({"tanh->exp " + std::to_string(size) + "x" +
                           std::to_string(size),
                       time});
}

void bench_sub_square(size_t size) {
    auto a = Tensor::randn({size, size});
    auto b = Tensor::randn({size, size});

    double time = time_ms([&]() {
        auto result = ops::square(ops::subtract(a, b));
        materialize<float>(result);
    });

    results.push_back(
        {"SubSquare " + std::to_string(size) + "x" + std::to_string(size),
         time});
}

void bench_mul_relu(size_t size) {
    auto a = Tensor::randn({size, size});
    auto b = Tensor::randn({size, size});

    double time = time_ms([&]() {
        auto result = ops::relu(ops::multiply(a, b));
        materialize<float>(result);
    });

    results.push_back(
        {"MulReLU " + std::to_string(size) + "x" + std::to_string(size), time});
}

void bench_add_sigmoid(size_t size) {
    auto a = Tensor::randn({size, size});
    auto b = Tensor::randn({size, size});

    double time = time_ms([&]() {
        auto result = ops::sigmoid(ops::add(a, b));
        materialize<float>(result);
    });

    results.push_back({"AddSigmoid " + std::to_string(size) + "x" +
                           std::to_string(size),
                       time});
}

void bench_add_relu_int32(size_t size) {
    auto a = Tensor::full<int32_t>({size, size}, -5);
    auto b = Tensor::full<int32_t>({size, size}, 10);

    double time = time_ms([&]() {
        auto result = ops::relu(ops::add(a, b));
        (void)result.item<int32_t>({0, 0});
    });

    results.push_back({"AddReLU(i32) " + std::to_string(size) + "x" +
                           std::to_string(size),
                       time});
}

void bench_sub_abs_int32(size_t size) {
    auto a = Tensor::full<int32_t>({size, size}, 5);
    auto b = Tensor::full<int32_t>({size, size}, 10);

    double time = time_ms([&]() {
        auto result = ops::abs(ops::subtract(a, b));
        (void)result.item<int32_t>({0, 0});
    });

    results.push_back({"SubAbs(i32) " + std::to_string(size) + "x" +
                           std::to_string(size),
                       time});
}

void bench_add_mul_relu(size_t size) {
    auto a = Tensor::randn({size, size});
    auto b = Tensor::randn({size, size});
    auto c = Tensor::randn({size, size});

    double time = time_ms([&]() {
        auto result = ops::relu(ops::multiply(ops::add(a, b), c));
        materialize<float>(result);
    });

    results.push_back({"AddMulReLU " + std::to_string(size) + "x" +
                           std::to_string(size),
                       time});
}

void bench_broadcast_add_relu(size_t size) {
    auto a = Tensor::randn({size, 1});
    auto b = Tensor::randn({1, size});

    double time = time_ms([&]() {
        auto result = ops::relu(ops::add(a, b));
        materialize<float>(result);
    });

    results.push_back({"Broadcast AddReLU " + std::to_string(size) + "x" +
                           std::to_string(size),
                       time});
}

void print_results() {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "                    LAZY vs EAGER BENCHMARK RESULTS\n";
    std::cout << std::string(80, '=') << "\n\n";

    std::string mode = is_eager_mode() ? "EAGER" : "LAZY";
    std::cout << "Mode: " << mode << "\n\n";

    std::cout << std::left << std::setw(35) << "Pattern" << std::right
              << std::setw(15) << "Time (ms)"
              << "\n";
    std::cout << std::string(50, '-') << "\n";

    double total = 0;
    for (const auto& r : results) {
        std::cout << std::left << std::setw(35) << r.name << std::right
                  << std::fixed << std::setprecision(3) << std::setw(15)
                  << r.time_ms << "\n";
        total += r.time_ms;
    }

    std::cout << std::string(50, '-') << "\n";
    std::cout << std::left << std::setw(35) << "TOTAL" << std::right
              << std::fixed << std::setprecision(3) << std::setw(15) << total
              << "\n";

    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "                           USAGE\n";
    std::cout << std::string(80, '=') << "\n\n";

    std::cout << "To compare lazy vs eager mode:\n";
    std::cout
        << "  ./bench_lazy_vs_eager                    # Lazy mode (default)\n";
    std::cout << "  AXIOM_EAGER_MODE=1 ./bench_lazy_vs_eager # Eager mode\n\n";
}

int main() {
    ops::OperationRegistry::initialize_builtin_operations();

    std::cout << "\n";
    std::cout << std::string(80, '=') << "\n";
    std::cout
        << "           AXIOM LAZY EVALUATION vs EAGER MODE BENCHMARK\n";
    std::cout << std::string(80, '=') << "\n";
    std::cout << "\nBenchmarking fused patterns at multiple sizes...\n";
    std::cout << "(warmup=3, iterations=10 per measurement)\n\n";

    std::vector<size_t> sizes = {256, 512, 1024, 2048};

    for (size_t size : sizes) {
        std::cout << "Running size " << size << "x" << size << "...\n";

        // Original fused SIMD patterns
        bench_add_relu(size);
        bench_sub_abs(size);
        bench_add_square(size);
        bench_mul_add(size);
        bench_scale_shift_relu(size);

        // NEW patterns
        bench_sub_square(size);
        bench_mul_relu(size);
        bench_add_sigmoid(size);
        bench_add_mul_relu(size);

        // Integer SIMD patterns
        bench_add_relu_int32(size);
        bench_sub_abs_int32(size);

        // Broadcasting patterns
        bench_broadcast_add_relu(size);

        // General fusion (no SIMD kernel)
        bench_long_chain(size);
        bench_operator_syntax(size);
        bench_sigmoid_chain(size);
        bench_tanh_exp(size);

        // Reduction (no fusion expected)
        bench_reduction_chain(size);
    }

    print_results();

    return 0;
}
