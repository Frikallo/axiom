#include <iostream>
#include <string>
#include <vector>
#include <functional>
#include <stdexcept>
#include <cassert>
#include <cmath>

#include <axiom/axiom.hpp>

// Test runner state
static int tests_run = 0;
static int tests_passed = 0;

// Test runner helper
void run_test(const std::function<void()>& test_func, const std::string& test_name) {
    tests_run++;
    std::cout << "--- Running: " << test_name << " ---" << std::endl;
    try {
        test_func();
        std::cout << "--- PASSED: " << test_name << " ---" << std::endl;
        tests_passed++;
    } catch (const std::exception& e) {
        std::cerr << "--- FAILED: " << test_name << " ---" << std::endl;
        std::cerr << "    Error: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "--- FAILED: " << test_name << " ---" << std::endl;
        std::cerr << "    Error: Unknown exception caught." << std::endl;
    }
    std::cout << std::endl;
}

// Helper to check if two floats are close enough
void assert_close(float a, float b, float epsilon = 1e-6) {
    assert(std::abs(a - b) < epsilon);
}

// ==================================
// 1. Basic Success Cases
// ==================================

void test_cpu_add_success() {
    auto a = axiom::Tensor::full({2, 2}, 2.0f);
    auto b = axiom::Tensor::full({2, 2}, 3.0f);
    auto c = axiom::ops::add(a, b);
    assert(c.device() == axiom::Device::CPU);
    const float* c_data = c.typed_data<float>();
    for (size_t i = 0; i < 4; ++i) assert_close(c_data[i], 5.0f);
}

void test_metal_add_success() {
#ifdef __APPLE__
    if (!axiom::system::is_metal_available()) return;
    auto a = axiom::Tensor::full({2, 2}, 2.0f).to(axiom::Device::GPU);
    auto b = axiom::Tensor::full({2, 2}, 3.0f).to(axiom::Device::GPU);
    auto c = axiom::ops::add(a, b);
    assert(c.device() == axiom::Device::GPU);
    auto c_cpu = c.cpu();
    const float* c_data = c_cpu.typed_data<float>();
    for (size_t i = 0; i < 4; ++i) assert_close(c_data[i], 5.0f);
#endif
}

void test_cpu_sub_success() {
    auto a = axiom::Tensor::full({2, 2}, 10.0f);
    auto b = axiom::Tensor::full({2, 2}, 3.0f);
    auto c = axiom::ops::subtract(a, b);
    assert(c.device() == axiom::Device::CPU);
    const float* c_data = c.typed_data<float>();
    for (size_t i = 0; i < 4; ++i) assert_close(c_data[i], 7.0f);
}

void test_metal_sub_success() {
#ifdef __APPLE__
    if (!axiom::system::is_metal_available()) return;
    auto a = axiom::Tensor::full({2, 2}, 10.0f).to(axiom::Device::GPU);
    auto b = axiom::Tensor::full({2, 2}, 3.0f).to(axiom::Device::GPU);
    auto c = axiom::ops::subtract(a, b);
    assert(c.device() == axiom::Device::GPU);
    auto c_cpu = c.cpu();
    const float* c_data = c_cpu.typed_data<float>();
    for (size_t i = 0; i < 4; ++i) assert_close(c_data[i], 7.0f);
#endif
}

// ==================================
// 2. Broadcasting and Type Promotion
// ==================================

void test_cpu_broadcasting_success() {
    auto a = axiom::Tensor::full({2, 2}, 10.0f);
    auto b = axiom::Tensor::full({1}, 5.0f); // Scalar
    auto c = axiom::ops::add(a, b);
    assert(c.shape() == axiom::Shape({2, 2}));
    const float* c_data = c.typed_data<float>();
    for (size_t i = 0; i < 4; ++i) assert_close(c_data[i], 15.0f);
}

void test_cpu_type_promotion_success() {
    auto a = axiom::Tensor::full({2, 2}, static_cast<int32_t>(10));
    auto b = axiom::Tensor::full({2, 2}, 5.5f);
    assert(a.dtype() == axiom::DType::Int32);
    assert(b.dtype() == axiom::DType::Float32);

    auto c = axiom::ops::add(a, b);
    assert(c.dtype() == axiom::DType::Float32);
    const float* c_data = c.typed_data<float>();
    for (size_t i = 0; i < 4; ++i) assert_close(c_data[i], 15.5f);
}


// ==================================
// 3. Edge Cases and Error Handling
// ==================================

void test_unsupported_gpu_op_fallback() {
#ifdef __APPLE__
    if (!axiom::system::is_metal_available()) return;
    // Multiply is not implemented on Metal, should fall back to CPU.
    auto a = axiom::Tensor::full({2, 2}, 3.0f).to(axiom::Device::GPU);
    auto b = axiom::Tensor::full({2, 2}, 4.0f).to(axiom::Device::GPU);
    auto c = axiom::ops::multiply(a, b);
    
    // The result should be on the CPU as it's the fallback device
    assert(c.device() == axiom::Device::CPU);
    
    const float* c_data = c.typed_data<float>();
    for (size_t i = 0; i < 4; ++i) assert_close(c_data[i], 12.0f);
#endif
}

void test_mixed_device_op_success() {
#ifdef __APPLE__
    if (!axiom::system::is_metal_available()) return;
    // Operation between CPU and GPU tensor should work, preferring GPU.
    auto a = axiom::Tensor::full({2, 2}, 3.0f, axiom::Device::CPU);
    auto b = axiom::Tensor::full({2, 2}, 4.0f).to(axiom::Device::GPU);
    auto c = axiom::ops::add(a, b);

    // The result should be on the GPU
    assert(c.device() == axiom::Device::GPU);
    auto c_cpu = c.cpu();
    const float* c_data = c_cpu.typed_data<float>();
    for (size_t i = 0; i < 4; ++i) assert_close(c_data[i], 7.0f);
#endif
}

void test_shape_mismatch_error() {
    auto a = axiom::Tensor::full({2, 3}, 1.0f);
    auto b = axiom::Tensor::full({2, 4}, 1.0f);
    try {
        axiom::ops::add(a, b);
        assert(false); // Should have thrown
    } catch (const std::runtime_error& e) {
        // Expected
        std::string msg = e.what();
        assert(msg.find("not broadcastable") != std::string::npos);
    }
}

void test_metal_type_mismatch_error() {
#ifdef __APPLE__
    if (!axiom::system::is_metal_available()) return;
    auto a = axiom::Tensor::full({2, 2}, static_cast<int32_t>(1)).to(axiom::Device::GPU);
    auto b = axiom::Tensor::full({2, 2}, static_cast<int32_t>(2)).to(axiom::Device::GPU);
    assert(a.dtype() == axiom::DType::Int32);
    assert(b.dtype() == axiom::DType::Int32);

    try {
        axiom::ops::add(a, b);
        assert(false); // Should have thrown
    } catch (const std::runtime_error& e) {
        // Expected
        std::string msg = e.what();
        assert(msg.find("supports float32") != std::string::npos);
    }
#endif
}

void test_metal_broadcast_error() {
#ifdef __APPLE__
    if (!axiom::system::is_metal_available()) return;
    auto a = axiom::Tensor::full({2, 2}, 10.0f).to(axiom::Device::GPU);
    auto b = axiom::Tensor::full({1}, 5.0f).to(axiom::Device::GPU); // Scalar
    
    // The current Metal implementation does NOT support broadcasting.
    // This should fall back to the CPU implementation.
    auto c = axiom::ops::add(a, b);
    
    // The result should be on the CPU.
    assert(c.device() == axiom::Device::CPU);
    
    const float* c_data = c.typed_data<float>();
    for (size_t i = 0; i < 4; ++i) assert_close(c_data[i], 15.0f);
#endif
}


int main() {
    axiom::ops::OperationRegistry::initialize_builtin_operations();

    run_test(test_cpu_add_success, "CPU Add Success");
    run_test(test_metal_add_success, "Metal Add Success");
    run_test(test_cpu_sub_success, "CPU Subtract Success");
    run_test(test_metal_sub_success, "Metal Subtract Success");

    run_test(test_cpu_broadcasting_success, "CPU Broadcasting Success");
    run_test(test_cpu_type_promotion_success, "CPU Type Promotion Success");

    run_test(test_unsupported_gpu_op_fallback, "GPU Op Fallback to CPU");
    run_test(test_mixed_device_op_success, "Mixed Device Operation Success");
    run_test(test_shape_mismatch_error, "Shape Mismatch Error");
    run_test(test_metal_type_mismatch_error, "Metal Type Mismatch Error");
    run_test(test_metal_broadcast_error, "Metal Broadcasting Fallback to CPU");

    std::cout << "========================================" << std::endl;
    std::cout << "Test Suite Summary:" << std::endl;
    std::cout << "    " << tests_passed << " / " << tests_run << " tests passed." << std::endl;
    std::cout << "========================================" << std::endl;

    return (tests_passed == tests_run) ? 0 : 1;
} 