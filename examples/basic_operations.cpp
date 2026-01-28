#include <axiom/axiom.hpp>
#include <iostream>

using namespace axiom;

int main() {
    // Create tensors with NumPy-like syntax
    auto x = Tensor::randn({64, 128, 256}, DType::Float32, Device::GPU);
    auto y = Tensor::ones({256, 512}, DType::Float32, Device::GPU);

    // Matrix operations
    auto result = x.matmul(y);

    // Einops-style rearrangement
    auto reshaped = x.rearrange("b h w -> b (h w)");

    // Broadcasting and element-wise ops
    auto scaled = (x * 2.0f + 1.0f).relu();

    std::cout << result << std::endl;

    return 0;
}