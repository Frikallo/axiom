// Eigen linear algebra operations benchmark
// Note: Only use BLAS, not LAPACKE (not available in macOS Accelerate)
#define EIGEN_USE_BLAS
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>

struct BenchResult {
    std::string op;
    double time_ms;
    double gflops;
};

BenchResult benchmark_op(const std::string& op, int n, int warmup = 2,
                         int iterations = 5) {
    Eigen::MatrixXf A, B;

    if (op == "solve" || op == "cholesky" || op == "eig") {
        // Create symmetric positive definite matrix
        Eigen::MatrixXf R = Eigen::MatrixXf::Random(n, n);
        A = R * R.transpose() +
            Eigen::MatrixXf::Identity(n, n) * static_cast<float>(n);
        B = Eigen::MatrixXf::Random(n, 1);
    } else {
        A = Eigen::MatrixXf::Random(n, n);
        B = Eigen::MatrixXf::Random(n, 1);
    }

    // Warmup
    for (int i = 0; i < warmup; i++) {
        if (op == "svd") {
            Eigen::JacobiSVD<Eigen::MatrixXf> svd(
                A, Eigen::ComputeThinU | Eigen::ComputeThinV);
        } else if (op == "qr") {
            Eigen::HouseholderQR<Eigen::MatrixXf> qr(A);
        } else if (op == "solve") {
            Eigen::MatrixXf X = A.ldlt().solve(B);
        } else if (op == "cholesky") {
            Eigen::LLT<Eigen::MatrixXf> llt(A);
        } else if (op == "eig") {
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> es(A);
        } else if (op == "inv") {
            Eigen::MatrixXf Ainv = A.inverse();
        } else if (op == "det") {
            float d = A.determinant();
            (void)d;
        }
    }

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        if (op == "svd") {
            Eigen::JacobiSVD<Eigen::MatrixXf> svd(
                A, Eigen::ComputeThinU | Eigen::ComputeThinV);
        } else if (op == "qr") {
            Eigen::HouseholderQR<Eigen::MatrixXf> qr(A);
        } else if (op == "solve") {
            Eigen::MatrixXf X = A.ldlt().solve(B);
        } else if (op == "cholesky") {
            Eigen::LLT<Eigen::MatrixXf> llt(A);
        } else if (op == "eig") {
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> es(A);
        } else if (op == "inv") {
            Eigen::MatrixXf Ainv = A.inverse();
        } else if (op == "det") {
            float d = A.determinant();
            (void)d;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed_ms =
        std::chrono::duration<double, std::milli>(end - start).count();
    double avg_ms = elapsed_ms / iterations;

    // Approximate GFLOPS
    double flops;
    if (op == "svd")
        flops = 4.0 * n * n * n;
    else if (op == "qr")
        flops = (4.0 / 3.0) * n * n * n;
    else if (op == "solve")
        flops = (2.0 / 3.0) * n * n * n;
    else if (op == "cholesky")
        flops = (1.0 / 3.0) * n * n * n;
    else if (op == "eig")
        flops = 10.0 * n * n * n;
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
    int n = std::atoi(argv[2]);

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
        std::cout << result.time_ms << std::endl;
    }

    return 0;
}
