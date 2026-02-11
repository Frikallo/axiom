// Axiom unary operations benchmark
#include <axiom/axiom.hpp>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>

using namespace axiom;

struct BenchResult {
    std::string op;
    double throughput_gbps;
};

BenchResult benchmark_op(const std::string& op, size_t n, int warmup = 3,
                         int iterations = 10) {
    // Use positive values for ops that need them (log, sqrt)
    auto A = Tensor::rand({n, n}, DType::Float32, Device::CPU) + 0.1f;
    (void)A.data();  // Force materialization of lazy tensor

    // Warmup
    for (int i = 0; i < warmup; i++) {
        Tensor C;
        if (op == "exp")
            C = ops::exp(A);
        else if (op == "log")
            C = ops::log(A);
        else if (op == "sqrt")
            C = ops::sqrt(A);
        else if (op == "sin")
            C = ops::sin(A);
        else if (op == "cos")
            C = ops::cos(A);
        else if (op == "tanh")
            C = ops::tanh(A);
        else if (op == "abs")
            C = ops::abs(A);
        else if (op == "neg")
            C = -A;
        else if (op == "relu")
            C = ops::relu(A);
        else if (op == "sigmoid")
            C = ops::sigmoid(A);
        (void)C.data();  // Force materialization of lazy tensor
    }

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        Tensor C;
        if (op == "exp")
            C = ops::exp(A);
        else if (op == "log")
            C = ops::log(A);
        else if (op == "sqrt")
            C = ops::sqrt(A);
        else if (op == "sin")
            C = ops::sin(A);
        else if (op == "cos")
            C = ops::cos(A);
        else if (op == "tanh")
            C = ops::tanh(A);
        else if (op == "abs")
            C = ops::abs(A);
        else if (op == "neg")
            C = -A;
        else if (op == "relu")
            C = ops::relu(A);
        else if (op == "sigmoid")
            C = ops::sigmoid(A);
        (void)C.data();  // Force materialization of lazy tensor
    }
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed_ms =
        std::chrono::duration<double, std::milli>(end - start).count();
    double avg_seconds = (elapsed_ms / 1000.0) / iterations;

    // GB/s: 1 read + 1 write, each n*n float32s
    double bytes = 2.0 * n * n * sizeof(float);
    double gbps = bytes / avg_seconds / 1e9;

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
    size_t n = static_cast<size_t>(std::atoi(argv[2]));

    ops::OperationRegistry::initialize_builtin_operations();

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
