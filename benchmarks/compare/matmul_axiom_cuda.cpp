// Axiom CUDA GPU matmul benchmark
#include <axiom/axiom.hpp>
#include <chrono>
#include <iostream>

using namespace axiom;

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <size>" << std::endl;
        return 1;
    }

    size_t n = static_cast<size_t>(std::atoi(argv[1]));
    int warmup = 5;
    int iterations = 20;

    ops::OperationRegistry::initialize_builtin_operations();

    auto A = Tensor::randn({n, n}, DType::Float32, Device::GPU);
    auto B = Tensor::randn({n, n}, DType::Float32, Device::GPU);

    Tensor C;

    // Warmup
    for (int i = 0; i < warmup; i++) {
        C = A.matmul(B);
        C.storage();  // Force lazy graph execution
        system::synchronize();
    }

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        C = A.matmul(B);
        C.storage();  // Force lazy graph execution (no D2H copy)
        system::synchronize();
    }
    auto end = std::chrono::high_resolution_clock::now();

    // Prevent dead-code elimination
    volatile auto ptr = C.storage().get();
    (void)ptr;

    double elapsed_ms =
        std::chrono::duration<double, std::milli>(end - start).count();
    double avg_ms = elapsed_ms / iterations;

    double flops = 2.0 * static_cast<double>(n) * static_cast<double>(n) *
                   static_cast<double>(n);
    double gflops = (flops / (avg_ms / 1000.0)) / 1e9;

    std::cout << gflops << std::endl;

    return 0;
}
