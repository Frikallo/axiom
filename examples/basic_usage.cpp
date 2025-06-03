#include <axiom/tensor.hpp>
#include <iostream>

using namespace axiom;

int main() {
    // Create tensors with NumPy-like syntax
    auto x = Tensor::randn({64, 128, 256}, DType::Float16, Device::GPU, MemoryOrder::RowMajor);
    auto y = Tensor::ones({256, 512});
    
    // Matrix operations
    // auto result = x.matmul(y);
    
    // Einops-style rearrangement
    auto reshaped = x.rearrange("b h w -> b (h w)");
    std::cout << reshaped.repr() << std::endl;
    reshaped.save("reshaped.axm");
    auto reshaped_loaded = Tensor::load("reshaped.axm", Device::GPU);
    std::cout << reshaped_loaded.repr() << std::endl;
    
    // Broadcasting and element-wise ops
    // auto scaled = (x * 2.0f + 1.0f).relu();
    
    return 0;
}