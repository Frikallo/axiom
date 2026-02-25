#include <axiom/axiom.hpp>
#include <chrono>
#include <iostream>

using namespace axiom;

int main() {
    auto a = Tensor::randn({1, 128, 2048}, DType::Float32, Device::GPU,
                           MemoryOrder::RowMajor);
    std::cout << a << std::endl;
    std::cout << a.shape() << std::endl;
    std::cout << a.repr() << std::endl;
    // Transfer to CPU
    auto start = std::chrono::high_resolution_clock::now();
    auto b = a.to(Device::CPU);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Time taken: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                       start)
                     .count()
              << "ms" << std::endl;
    std::cout << b.shape() << std::endl;
    std::cout << b.repr() << std::endl;
    // Transfer to GPU
    start = std::chrono::high_resolution_clock::now();
    auto c = b.to(Device::GPU);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Time taken: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                       start)
                     .count()
              << "ms" << std::endl;
    std::cout << c.shape() << std::endl;
    std::cout << c.repr() << std::endl;
    return 0;
}