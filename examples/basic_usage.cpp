#include <axiom/tensor.hpp>
#include <iostream>

using namespace axiom;

int main() {
    std::cout << "Axiom Tensor Library - Basic Usage Example\n" << std::endl;
    
    // Show available backends
    std::cout << "0. Available backends:" << std::endl;
    
    std::cout << "  CPU: Available" << std::endl;
#ifdef __APPLE__
    std::cout << "  Metal GPU: " << (Device::GPU == Device::GPU ? "Available" : "Not Available") << std::endl;
#else
    std::cout << "  Metal GPU: Not available (requires macOS)" << std::endl;
#endif
    
    // Create tensors with different shapes and data types
    std::cout << "1. Creating tensors:" << std::endl;
    
    auto x = zeros({3, 4}, DType::Float32, Device::CPU);
    std::cout << "  x (zeros): " << x.repr() << std::endl;
    
    auto y = ones({4, 2}, DType::Float32, Device::CPU);
    std::cout << "  y (ones): " << y.repr() << std::endl;
    
    auto identity_mat = eye(3, DType::Float32, Device::CPU);
    std::cout << "  I (identity): " << identity_mat.repr() << std::endl;
    
    // Fill tensor with custom values
    std::cout << "\n2. Setting values:" << std::endl;
    
    auto data_tensor = empty({2, 3}, DType::Float32, Device::CPU);
    data_tensor.fill<float>(42.0f);
    std::cout << "  Filled with 42.0: " << data_tensor.repr() << std::endl;
    
    // Individual element access
    data_tensor.set_item<float>({0, 1}, 3.14f);
    data_tensor.set_item<float>({1, 2}, 2.71f);
    
    std::cout << "  Element [0,1]: " << data_tensor.item<float>({0, 1}) << std::endl;
    std::cout << "  Element [1,2]: " << data_tensor.item<float>({1, 2}) << std::endl;
    
    // Shape manipulation
    std::cout << "\n3. Shape manipulation:" << std::endl;
    
    auto original = zeros({2, 3, 4}, DType::Float32, Device::CPU);
    std::cout << "  Original: " << original.repr() << std::endl;
    
    auto reshaped = original.reshape({6, 4});
    std::cout << "  Reshaped: " << reshaped.repr() << std::endl;
    
    auto transposed = y.transpose();
    std::cout << "  y transposed: " << transposed.repr() << std::endl;
    
    auto squeezed = original.reshape({1, 2, 3, 4, 1}).squeeze();
    std::cout << "  Squeezed: " << squeezed.repr() << std::endl;
    
    auto unsqueezed = y.unsqueeze(1);
    std::cout << "  Unsqueezed: " << unsqueezed.repr() << std::endl;
    
    // Memory operations
    std::cout << "\n4. Memory operations:" << std::endl;
    
    auto source = ones({2, 2}, DType::Float32, Device::CPU);
    source.set_item<float>({0, 0}, 99.0f);
    
    auto copied = source.copy();
    std::cout << "  Source: " << source.repr() << std::endl;
    std::cout << "  Copy: " << copied.repr() << std::endl;
    
    // Modify source - copy should remain unchanged
    source.fill<float>(0.0f);
    std::cout << "  After modifying source:" << std::endl;
    std::cout << "    Source[0,0]: " << source.item<float>({0, 0}) << std::endl;
    std::cout << "    Copy[0,0]: " << copied.item<float>({0, 0}) << std::endl;
    
    // Views and memory sharing
    std::cout << "\n5. Views (memory sharing):" << std::endl;
    
    auto base_tensor = zeros({3, 4}, DType::Float32, Device::CPU);
    base_tensor.fill<float>(1.0f);
    
    auto view_tensor = base_tensor.reshape({12});
    std::cout << "  Base: " << base_tensor.repr() << std::endl;
    std::cout << "  View: " << view_tensor.repr() << std::endl;
    
    // Modify through view
    view_tensor.set_item<float>({0}, 777.0f);
    std::cout << "  After modifying view[0] = 777:" << std::endl;
    std::cout << "    Base[0,0]: " << base_tensor.item<float>({0, 0}) << std::endl;
    std::cout << "    View[0]: " << view_tensor.item<float>({0}) << std::endl;
    
    // Data type information
    std::cout << "\n6. Data type information:" << std::endl;
    
    auto int_tensor = zeros({2, 2}, DType::Int32, Device::CPU);
    auto float_tensor = zeros({2, 2}, DType::Float64, Device::CPU);
    
    std::cout << "  Int32 tensor itemsize: " << int_tensor.itemsize() << " bytes" << std::endl;
    std::cout << "  Float64 tensor itemsize: " << float_tensor.itemsize() << " bytes" << std::endl;
    std::cout << "  Int32 tensor total bytes: " << int_tensor.nbytes() << " bytes" << std::endl;
    
#ifdef __APPLE__
    // GPU operations (Metal)
    std::cout << "\n7. GPU operations (Metal):" << std::endl;
    
    try {
        auto cpu_tensor = ones({2, 3}, DType::Float32, Device::CPU);
        auto gpu_tensor = cpu_tensor.gpu();
        
        std::cout << "  CPU tensor device: " << (cpu_tensor.device() == Device::CPU ? "CPU" : "GPU") << std::endl;
        std::cout << "  GPU tensor device: " << (gpu_tensor.device() == Device::CPU ? "CPU" : "GPU") << std::endl;
        
        // Move back to CPU
        auto back_to_cpu = gpu_tensor.cpu();
        std::cout << "  Moved back to CPU: " << (back_to_cpu.device() == Device::CPU ? "CPU" : "GPU") << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "  Metal operations failed: " << e.what() << std::endl;
    }
#else
    std::cout << "\n7. GPU operations: Not available (Metal requires macOS)" << std::endl;
#endif
    
    std::cout << "\nBasic usage example completed successfully!" << std::endl;
    return 0;
}