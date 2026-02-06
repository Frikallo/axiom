// Axiom FFT operations benchmark
#include <axiom/axiom.hpp>
#include <axiom/fft.hpp>
#include <chrono>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

using namespace axiom;

struct BenchResult {
    std::string op;
    double time_ms;
    double throughput_gflops;
};

BenchResult benchmark_op(const std::string& op, size_t n, int warmup = 3,
                         int iterations = 10) {
    ops::OperationRegistry::initialize_builtin_operations();

    // Create appropriate input tensor for this specific operation
    Tensor A;
    Tensor complex_input;

    bool is_2d = (op == "fft2" || op == "ifft2" || op == "rfft2");

    if (op == "ifft") {
        auto real_input = Tensor::randn({n}, DType::Float32, Device::CPU);
        complex_input = fft::fft(real_input);
    } else if (op == "ifft2") {
        auto real_input = Tensor::randn({n, n}, DType::Float32, Device::CPU);
        complex_input = fft::fft2(real_input);
    } else if (is_2d) {
        A = Tensor::randn({n, n}, DType::Float32, Device::CPU);
    } else {
        A = Tensor::randn({n}, DType::Float32, Device::CPU);
    }

    // Warmup
    for (int i = 0; i < warmup; i++) {
        Tensor C;
        if (op == "fft")
            C = fft::fft(A);
        else if (op == "ifft")
            C = fft::ifft(complex_input);
        else if (op == "fft2")
            C = fft::fft2(A);
        else if (op == "ifft2")
            C = fft::ifft2(complex_input);
        else if (op == "rfft")
            C = fft::rfft(A);
        else if (op == "rfft2")
            C = fft::rfft2(A);
        (void)C.data();  // Force materialization of lazy tensor
    }

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        Tensor C;
        if (op == "fft")
            C = fft::fft(A);
        else if (op == "ifft")
            C = fft::ifft(complex_input);
        else if (op == "fft2")
            C = fft::fft2(A);
        else if (op == "ifft2")
            C = fft::ifft2(complex_input);
        else if (op == "rfft")
            C = fft::rfft(A);
        else if (op == "rfft2")
            C = fft::rfft2(A);
        (void)C.data();  // Force materialization of lazy tensor
    }
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed_ms =
        std::chrono::duration<double, std::milli>(end - start).count();
    double avg_ms = elapsed_ms / iterations;

    // FFT has O(n log n) complexity for 1D, O(n^2 log n) for 2D
    double flops;
    if (is_2d)
        flops = 5.0 * n * n * std::log2(static_cast<double>(n * n));
    else
        flops = 5.0 * n * std::log2(static_cast<double>(n));

    double gflops = (flops / (avg_ms / 1000.0)) / 1e9;

    return {op, avg_ms, gflops};
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <op> <size>" << std::endl;
        std::cerr << "  ops: fft, ifft, fft2, ifft2, rfft, rfft2, all"
                  << std::endl;
        return 1;
    }

    std::string op = argv[1];
    size_t n = static_cast<size_t>(std::atoi(argv[2]));

    if (op == "all") {
        std::cout << "{";
        std::vector<std::string> ops = {"fft", "ifft", "rfft", "fft2", "ifft2", "rfft2"};
        for (size_t i = 0; i < ops.size(); i++) {
            auto result = benchmark_op(ops[i], n);
            std::cout << "\"" << ops[i] << "\":{\"time_ms\":" << result.time_ms
                      << ",\"gflops\":" << result.throughput_gflops << "}";
            if (i < ops.size() - 1) std::cout << ",";
        }
        std::cout << "}" << std::endl;
    } else {
        auto result = benchmark_op(op, n);
        std::cout << result.time_ms << std::endl;
    }

    return 0;
}
