// Eigen unary operations benchmark
#include <Eigen/Dense>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>

struct BenchResult {
    std::string op;
    double throughput_gbps;
};

BenchResult benchmark_op(const std::string& op, int n, int warmup = 3,
                         int iterations = 10) {
    // Use positive values for log/sqrt
    Eigen::MatrixXf A =
        Eigen::MatrixXf::Random(n, n).cwiseAbs() +
        Eigen::MatrixXf::Constant(n, n, 0.1f);
    Eigen::MatrixXf C(n, n);

    // Warmup
    for (int i = 0; i < warmup; i++) {
        if (op == "exp")
            C = A.array().exp();
        else if (op == "log")
            C = A.array().log();
        else if (op == "sqrt")
            C = A.array().sqrt();
        else if (op == "sin")
            C = A.array().sin();
        else if (op == "cos")
            C = A.array().cos();
        else if (op == "tanh")
            C = A.array().tanh();
        else if (op == "abs")
            C = A.array().abs();
        else if (op == "neg")
            C = -A;
        else if (op == "relu")
            C = A.cwiseMax(0.0f);
        else if (op == "sigmoid")
            C = (1.0f / (1.0f + (-A.array()).exp())).matrix();
    }

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        if (op == "exp")
            C = A.array().exp();
        else if (op == "log")
            C = A.array().log();
        else if (op == "sqrt")
            C = A.array().sqrt();
        else if (op == "sin")
            C = A.array().sin();
        else if (op == "cos")
            C = A.array().cos();
        else if (op == "tanh")
            C = A.array().tanh();
        else if (op == "abs")
            C = A.array().abs();
        else if (op == "neg")
            C = -A;
        else if (op == "relu")
            C = A.cwiseMax(0.0f);
        else if (op == "sigmoid")
            C = (1.0f / (1.0f + (-A.array()).exp())).matrix();
    }
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed_ms =
        std::chrono::duration<double, std::milli>(end - start).count();
    double avg_seconds = (elapsed_ms / 1000.0) / iterations;

    // GB/s: 1 read + 1 write
    double bytes = 2.0 * n * n * sizeof(float);
    double gbps = bytes / avg_seconds / 1e9;

    volatile float sink = C(0, 0);
    (void)sink;

    return {op, gbps};
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <op> <size>" << std::endl;
        std::cerr << "  ops: exp, log, sqrt, sin, cos, tanh, abs, neg, relu, "
                     "sigmoid, all"
                  << std::endl;
        return 1;
    }

    std::string op = argv[1];
    int n = std::atoi(argv[2]);

    if (op == "all") {
        std::cout << "{";
        std::vector<std::string> ops = {"exp",  "log",     "sqrt", "sin",
                                        "cos",  "tanh",    "abs",  "neg",
                                        "relu", "sigmoid"};
        for (size_t i = 0; i < ops.size(); i++) {
            auto result = benchmark_op(ops[i], n);
            std::cout << "\"" << ops[i] << "\":" << result.throughput_gbps;
            if (i < ops.size() - 1) std::cout << ",";
        }
        std::cout << "}" << std::endl;
    } else {
        auto result = benchmark_op(op, n);
        std::cout << result.throughput_gbps << std::endl;
    }

    return 0;
}
