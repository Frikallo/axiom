// Axiom matmul benchmark
#include <axiom/axiom.hpp>
#include <chrono>
#include <iostream>
#include <vector>

using namespace axiom;

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <size>" << std::endl;
        return 1;
    }

    size_t n = static_cast<size_t>(std::atoi(argv[1]));
    int warmup = 3;
    int iterations = 10;

    ops::OperationRegistry::initialize_builtin_operations();

    // Create matrices
    auto A = Tensor::randn({n, n}, DType::Float32, Device::CPU);
    auto B = Tensor::randn({n, n}, DType::Float32, Device::CPU);

    // Warmup
    for (int i = 0; i < warmup; i++) {
        auto C = A.matmul(B);
    }

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        auto C = A.matmul(B);
    }
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed_ms =
        std::chrono::duration<double, std::milli>(end - start).count();
    double avg_ms = elapsed_ms / iterations;

    // Calculate GFLOPS: 2*N^3 operations for matmul
    double flops = 2.0 * static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(n);
    double gflops = (flops / (avg_ms / 1000.0)) / 1e9;

    std::cout << gflops << std::endl;

    return 0;
}
