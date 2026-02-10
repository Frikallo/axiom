// Axiom element-wise operations benchmark
#include <axiom/axiom.hpp>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using namespace axiom;

using BenchFn = Tensor (*)(const Tensor &, const Tensor &);

static BenchFn get_bench_fn(const std::string &name) {
    static const std::unordered_map<std::string, BenchFn> fns = {
        {"add", ops::add},
        {"sub", ops::subtract},
        {"mul", ops::multiply},
        {"div", ops::divide},
    };
    auto it = fns.find(name);
    if (it == fns.end())
        throw std::runtime_error("Unknown op: " + name);
    return it->second;
}

struct BenchResult {
    std::string op;
    double throughput_gbps;
};

BenchResult benchmark_op(const std::string& op, size_t n, int warmup = 3,
                         int iterations = 10) {
    auto A = Tensor::randn({n, n}, DType::Float32, Device::CPU);
    auto B = Tensor::randn({n, n}, DType::Float32, Device::CPU);
    BenchFn fn = get_bench_fn(op);

    // Warmup
    for (int i = 0; i < warmup; i++) {
        Tensor C = fn(A, B);
        (void)C.data();  // Force materialization of lazy tensor
    }

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        Tensor C = fn(A, B);
        (void)C.data();  // Force materialization of lazy tensor
    }
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed_ms =
        std::chrono::duration<double, std::milli>(end - start).count();
    double avg_seconds = (elapsed_ms / 1000.0) / iterations;

    // GB/s: 2 reads + 1 write, each n*n float32s
    double bytes = 3.0 * n * n * sizeof(float);
    double gbps = bytes / avg_seconds / 1e9;

    return {op, gbps};
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <op> <size>" << std::endl;
        std::cerr << "  ops: add, sub, mul, div, all" << std::endl;
        return 1;
    }

    std::string op = argv[1];
    size_t n = static_cast<size_t>(std::atoi(argv[2]));

    ops::OperationRegistry::initialize_builtin_operations();

    if (op == "all") {
        // Output JSON for all ops
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
