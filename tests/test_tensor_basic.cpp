#include <axiom/tensor.hpp>
#include <iostream>
#include <cassert>
#include <vector>

using namespace axiom;

void test_tensor_creation() {
    std::cout << "Testing tensor creation..." << std::endl;
    
    // Test basic tensor creation
    auto t1 = Tensor({3, 4}, DType::Float32, Device::CPU);
    assert(t1.ndim() == 2);
    assert(t1.shape()[0] == 3);
    assert(t1.shape()[1] == 4);
    assert(t1.size() == 12);
    assert(t1.dtype() == DType::Float32);
    assert(t1.device() == Device::CPU);
    
    // Test initializer list constructor
    auto t2 = Tensor({2, 3, 4}, DType::Int32);
    assert(t2.ndim() == 3);
    assert(t2.size() == 24);
    
    std::cout << "  Basic creation: PASSED" << std::endl;
}

void test_factory_functions() {
    std::cout << "Testing factory functions..." << std::endl;
    
    // Test zeros
    auto z = zeros({2, 3}, DType::Float32, Device::CPU);
    assert(z.shape()[0] == 2);
    assert(z.shape()[1] == 3);
    
    // Verify zeros are actually zero
    auto data = z.typed_data<float>();
    for (size_t i = 0; i < z.size(); ++i) {
        assert(data[i] == 0.0f);
    }
    
    // Test ones
    auto o = ones({2, 2}, DType::Float32, Device::CPU);
    auto ones_data = o.typed_data<float>();
    for (size_t i = 0; i < o.size(); ++i) {
        assert(ones_data[i] == 1.0f);
    }
    
    // Test eye
    auto eye_mat = eye(3, DType::Float32, Device::CPU);
    auto eye_data = eye_mat.typed_data<float>();
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            float expected = (i == j) ? 1.0f : 0.0f;
            assert(eye_data[i * 3 + j] == expected);
        }
    }
    
    std::cout << "  Factory functions: PASSED" << std::endl;
}

void test_indexing_and_access() {
    std::cout << "Testing indexing and access..." << std::endl;
    
    auto t = zeros({2, 3}, DType::Float32, Device::CPU);
    
    // Test single element access
    t.set_item<float>({0, 1}, 5.0f);
    float val = t.item<float>({0, 1});
    assert(val == 5.0f);
    
    // Test fill
    auto t2 = empty({3, 3}, DType::Int32, Device::CPU);
    t2.fill<int32_t>(42);
    auto data = t2.typed_data<int32_t>();
    for (size_t i = 0; i < t2.size(); ++i) {
        assert(data[i] == 42);
    }
    
    std::cout << "  Indexing and access: PASSED" << std::endl;
}

void test_shape_manipulation() {
    std::cout << "Testing shape manipulation..." << std::endl;
    
    auto t = zeros({2, 3, 4}, DType::Float32, Device::CPU);
    
    // Test reshape
    auto reshaped = t.reshape({6, 4});
    assert(reshaped.shape()[0] == 6);
    assert(reshaped.shape()[1] == 4);
    assert(reshaped.size() == t.size());
    
    // Test transpose
    auto t2d = zeros({3, 4}, DType::Float32, Device::CPU);
    auto transposed = t2d.transpose();
    assert(transposed.shape()[0] == 4);
    assert(transposed.shape()[1] == 3);
    
    // Test squeeze
    auto t_with_ones = zeros({1, 3, 1, 4}, DType::Float32, Device::CPU);
    auto squeezed = t_with_ones.squeeze();
    assert(squeezed.ndim() == 2);
    assert(squeezed.shape()[0] == 3);
    assert(squeezed.shape()[1] == 4);
    
    // Test unsqueeze
    auto t2 = zeros({3, 4}, DType::Float32, Device::CPU);
    auto unsqueezed = t2.unsqueeze(1);
    assert(unsqueezed.ndim() == 3);
    assert(unsqueezed.shape()[0] == 3);
    assert(unsqueezed.shape()[1] == 1);
    assert(unsqueezed.shape()[2] == 4);
    
    std::cout << "  Shape manipulation: PASSED" << std::endl;
}

void test_memory_operations() {
    std::cout << "Testing memory operations..." << std::endl;
    
    auto t1 = ones({2, 3}, DType::Float32, Device::CPU);
    
    // Test copy
    auto t2 = t1.copy();
    assert(t2.same_shape(t1));
    assert(t2.same_dtype(t1));
    assert(t2.same_device(t1));
    
    // Modify original, copy should be unchanged
    t1.fill<float>(5.0f);
    auto t1_data = t1.typed_data<float>();
    auto t2_data = t2.typed_data<float>();
    
    assert(t1_data[0] == 5.0f);
    assert(t2_data[0] == 1.0f);
    
    std::cout << "  Memory operations: PASSED" << std::endl;
}

void test_views() {
    std::cout << "Testing tensor views..." << std::endl;
    
    auto base = zeros({2, 3, 4}, DType::Float32, Device::CPU);
    base.fill<float>(1.0f);
    
    // Test reshape view
    auto reshaped = base.reshape({6, 4});
    assert(reshaped.is_view() || base.is_contiguous()); // Should be a view if base is contiguous
    
    // Modify via reshaped view
    reshaped.set_item<float>({0, 0}, 99.0f);
    float val = base.item<float>({0, 0, 0});
    assert(val == 99.0f); // Change should be visible in base
    
    std::cout << "  Views: PASSED" << std::endl;
}

void test_dtype_system() {
    std::cout << "Testing dtype system..." << std::endl;
    
    // Test different dtypes
    auto t_f32 = zeros({2, 2}, DType::Float32, Device::CPU);
    auto t_f64 = zeros({2, 2}, DType::Float64, Device::CPU);
    auto t_i32 = zeros({2, 2}, DType::Int32, Device::CPU);
    
    assert(t_f32.itemsize() == 4);
    assert(t_f64.itemsize() == 8);
    assert(t_i32.itemsize() == 4);
    
    assert(t_f32.dtype_name() == "float32");
    assert(t_f64.dtype_name() == "float64");
    assert(t_i32.dtype_name() == "int32");
    
    // Test automatic dtype deduction
    static_assert(dtype_of_v<float> == DType::Float32);
    static_assert(dtype_of_v<double> == DType::Float64);
    static_assert(dtype_of_v<int32_t> == DType::Int32);
    
    std::cout << "  Dtype system: PASSED" << std::endl;
}

void test_metal_backend() {
    std::cout << "Testing Metal backend..." << std::endl;
    
#ifdef __APPLE__
    try {
        auto cpu_tensor = ones({2, 3}, DType::Float32, Device::CPU);
        auto gpu_tensor = cpu_tensor.gpu();
        
        assert(gpu_tensor.device() == Device::GPU);
        assert(gpu_tensor.same_shape(cpu_tensor));
        assert(gpu_tensor.same_dtype(cpu_tensor));
        
        // Move back to CPU and verify data
        auto back_to_cpu = gpu_tensor.cpu();
        assert(back_to_cpu.device() == Device::CPU);
        
        auto data = back_to_cpu.typed_data<float>();
        for (size_t i = 0; i < back_to_cpu.size(); ++i) {
            assert(data[i] == 1.0f);
        }
        
        std::cout << "  Metal operations: PASSED" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "  Metal operations: SKIPPED (" << e.what() << ")" << std::endl;
    }
#else
    std::cout << "  Metal operations: SKIPPED (not on macOS)" << std::endl;
#endif
}

int main() {
    std::cout << "Running Axiom tensor tests..." << std::endl << std::endl;
    
    try {
        test_tensor_creation();
        test_factory_functions();
        test_indexing_and_access();
        test_shape_manipulation();
        test_memory_operations();
        test_views();
        test_dtype_system();
        test_metal_backend();
        
        std::cout << std::endl << "All tests PASSED! ✓" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test FAILED: " << e.what() << std::endl;
        return 1;
    }
}