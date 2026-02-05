// Eigen element-wise operations benchmark
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
    Eigen::MatrixXf A = Eigen::MatrixXf::Random(n, n);
    Eigen::MatrixXf B = Eigen::MatrixXf::Random(n, n);
    Eigen::MatrixXf C(n, n);

    // Warmup
    for (int i = 0; i < warmup; i++) {
        if (op == "add")
            C.noalias() = A + B;
        else if (op == "sub")
            C.noalias() = A - B;
        else if (op == "mul")
            C.noalias() = A.cwiseProduct(B);
        else if (op == "div")
            C.noalias() = A.cwiseQuotient(B);
    }

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        if (op == "add")
            C.noalias() = A + B;
        else if (op == "sub")
            C.noalias() = A - B;
        else if (op == "mul")
            C.noalias() = A.cwiseProduct(B);
        else if (op == "div")
            C.noalias() = A.cwiseQuotient(B);
    }
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed_ms =
        std::chrono::duration<double, std::milli>(end - start).count();
    double avg_seconds = (elapsed_ms / 1000.0) / iterations;

    // GB/s: 2 reads + 1 write, each n*n float32s
    double bytes = 3.0 * n * n * sizeof(float);
    double gbps = bytes / avg_seconds / 1e9;

    // Prevent optimization
    volatile float sink = C(0, 0);
    (void)sink;

    return {op, gbps};
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <op> <size>" << std::endl;
        std::cerr << "  ops: add, sub, mul, div, all" << std::endl;
        return 1;
    }

    std::string op = argv[1];
    int n = std::atoi(argv[2]);

    if (op == "all") {
        std::cout << "{";
        std::vector<std::string> ops = {"add", "sub", "mul", "div"};
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
