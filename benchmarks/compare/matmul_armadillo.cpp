// Armadillo matmul benchmark
#include <armadillo>
#include <chrono>
#include <iostream>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <size>" << std::endl;
        return 1;
    }

    int n = std::atoi(argv[1]);
    int warmup = 3;
    int iterations = 10;

    // Create random matrices
    arma::fmat A = arma::randn<arma::fmat>(n, n);
    arma::fmat B = arma::randn<arma::fmat>(n, n);
    arma::fmat C(n, n);

    // Warmup
    for (int i = 0; i < warmup; i++) {
        C = A * B;
    }

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        C = A * B;
    }
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed_ms =
        std::chrono::duration<double, std::milli>(end - start).count();
    double avg_ms = elapsed_ms / iterations;

    // Calculate GFLOPS: 2*N^3 operations for matmul
    double flops = 2.0 * n * n * n;
    double gflops = (flops / (avg_ms / 1000.0)) / 1e9;

    std::cout << gflops << std::endl;

    // Prevent optimization
    volatile float sink = C(0, 0);
    (void)sink;

    return 0;
}
