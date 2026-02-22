// Axiom linear algebra operations benchmark
#include <axiom/axiom.hpp>
#include <axiom/linalg.hpp>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>

using namespace axiom;

struct BenchResult {
    std::string op;
    double time_ms;
    double gflops;
};

BenchResult benchmark_op(const std::string& op, size_t n, int warmup = 2,
                         int iterations = 5) {
    ops::OperationRegistry::initialize_builtin_operations();

    // Create appropriate matrices for each operation
    Tensor A, B;
    if (op == "solve" || op == "cholesky" || op == "eig") {
        // Create symmetric positive definite matrix
        auto R = Tensor::randn({n, n}, DType::Float32, Device::CPU);
        A = R.matmul(R.transpose()) +
            Tensor::eye(n, DType::Float32, Device::CPU) * static_cast<float>(n);
        (void)A.data();  // Force materialization of lazy tensor
        B = Tensor::randn({n, 1}, DType::Float32, Device::CPU);
    } else {
        A = Tensor::randn({n, n}, DType::Float32, Device::CPU);
        B = Tensor::randn({n, 1}, DType::Float32, Device::CPU);
    }

    // Warmup
    for (int i = 0; i < warmup; i++) {
        Tensor result_tensor;
        if (op == "svd") {
            auto [U, S, Vt] = linalg::svd(A);
            result_tensor = U;
        } else if (op == "qr") {
            auto [Q, R] = linalg::qr(A);
            result_tensor = Q;
        } else if (op == "solve") {
            result_tensor = linalg::solve(A, B);
        } else if (op == "cholesky") {
            result_tensor = linalg::cholesky(A);
        } else if (op == "eig") {
            auto [vals, vecs] = linalg::eigh(A);
            result_tensor = vals;
        } else if (op == "inv") {
            result_tensor = linalg::inv(A);
        } else if (op == "det") {
            result_tensor = linalg::det(A);
        }
        (void)result_tensor.data();  // Force materialization
    }

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        Tensor result_tensor;
        if (op == "svd") {
            auto [U, S, Vt] = linalg::svd(A);
            result_tensor = U;
        } else if (op == "qr") {
            auto [Q, R] = linalg::qr(A);
            result_tensor = Q;
        } else if (op == "solve") {
            result_tensor = linalg::solve(A, B);
        } else if (op == "cholesky") {
            result_tensor = linalg::cholesky(A);
        } else if (op == "eig") {
            auto [vals, vecs] = linalg::eigh(A);
            result_tensor = vals;
        } else if (op == "inv") {
            result_tensor = linalg::inv(A);
        } else if (op == "det") {
            result_tensor = linalg::det(A);
        }
        (void)result_tensor.data();  // Force materialization
    }
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed_ms =
        std::chrono::duration<double, std::milli>(end - start).count();
    double avg_ms = elapsed_ms / iterations;

    // Approximate GFLOPS (varies by algorithm)
    double flops;
    if (op == "svd")
        flops = 4.0 * n * n * n;  // Rough approximation
    else if (op == "qr")
        flops = (4.0 / 3.0) * n * n * n;
    else if (op == "solve")
        flops = (2.0 / 3.0) * n * n * n;
    else if (op == "cholesky")
        flops = (1.0 / 3.0) * n * n * n;
    else if (op == "eig")
        flops = 10.0 * n * n * n;  // Very rough
    else if (op == "inv")
        flops = 2.0 * n * n * n;
    else if (op == "det")
        flops = (2.0 / 3.0) * n * n * n;
    else
        flops = n * n * n;

    double gflops = (flops / (avg_ms / 1000.0)) / 1e9;

    return {op, avg_ms, gflops};
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <op> <size>" << std::endl;
        std::cerr << "  ops: svd, qr, solve, cholesky, eig, inv, det, all"
                  << std::endl;
        return 1;
    }

    std::string op = argv[1];
    size_t n = static_cast<size_t>(std::atoi(argv[2]));

    if (op == "all") {
        std::cout << "{";
        std::vector<std::string> ops = {"svd",  "qr",  "solve", "cholesky",
                                        "eig",  "inv", "det"};
        for (size_t i = 0; i < ops.size(); i++) {
            auto result = benchmark_op(ops[i], n);
            std::cout << "\"" << ops[i] << "\":{\"time_ms\":" << result.time_ms
                      << ",\"gflops\":" << result.gflops << "}";
            if (i < ops.size() - 1) std::cout << ",";
        }
        std::cout << "}" << std::endl;
    } else {
        auto result = benchmark_op(op, n);
        // Output time_ms for comparison
        std::cout << result.time_ms << std::endl;
    }

    return 0;
}
