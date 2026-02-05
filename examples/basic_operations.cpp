#include <axiom/axiom.hpp>
#include <iostream>

using namespace axiom;

int main() {
    // Create tensors with NumPy-like syntax
    auto x = Tensor::randn({64, 128, 256}, Float32(), Device::CPU);
    auto y = Tensor::ones({256, 512}, Float32(), Device::CPU);

    // Matrix operations
    auto result = x.matmul(y);

    // Einops-style rearrangement
    auto reshaped = x.rearrange("b h w -> b (h w)");

    // Broadcasting and element-wise ops
    auto scaled = (x * 2.0f + 1.0f).relu();

    std::cout << result << std::endl;

    // Benchmark matmul of huge tensors on GPU
    auto a = Tensor::randn({1000, 1000}, Float32(), Device::CPU);
    auto b = Tensor::randn({1000, 1000}, Float32(), Device::CPU);
    auto start = std::chrono::high_resolution_clock::now();
    auto c = a.matmul(b);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Matmul of 1000x1000 tensors on GPU took "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                       start)
                     .count()
              << "ms" << std::endl;

    // Benchmark matmul of huge tensors on CPU with proper warmup
    auto a_cpu = Tensor::randn({1000, 1000}, Float32(), Device::CPU);
    auto b_cpu = Tensor::randn({1000, 1000}, Float32(), Device::CPU);

    // Warmup runs
    for (int i = 0; i < 3; ++i) {
        auto warmup = a_cpu.matmul(b_cpu);
    }

    // Timed runs
    constexpr int num_runs = 10;
    auto start_cpu = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_runs; ++i) {
        auto c_cpu = a_cpu.matmul(b_cpu);
    }
    auto end_cpu = std::chrono::high_resolution_clock::now();
    auto total_us = std::chrono::duration_cast<std::chrono::microseconds>(
                        end_cpu - start_cpu)
                        .count();
    std::cout << "Matmul of 1000x1000 tensors on CPU took "
              << (total_us / num_runs) << "us (avg of " << num_runs << " runs)"
              << std::endl;

    return 0;
}