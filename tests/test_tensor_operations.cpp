#include <iostream>
#include <string>
#include <vector>
#include <functional>
#include <stdexcept>
#include <cassert>
#include <cmath>
#include <map>
#include <variant>
#include <iomanip>

#include <axiom/axiom.hpp>

// ==================================
//
//      TEST HARNESS
//
// ==================================

static int tests_run = 0;
static int tests_passed = 0;
static std::string current_test_name;

#define RUN_TEST(test_func, ...) run_test([&]() { test_func(__VA_ARGS__); }, #test_func)

void run_test(const std::function<void()>& test_func, const std::string& test_name) {
    tests_run++;
    current_test_name = test_name;
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

// ==================================
//
//      CUSTOM ASSERTIONS
//
// ==================================

#define ASSERT(condition, msg) \
    do { \
        if (!(condition)) { \
            throw std::runtime_error("Assertion failed: (" #condition ") - " + std::string(msg)); \
        } \
    } while (0)

#define ASSERT_THROWS(expression) \
    do { \
        try { \
            (expression); \
            throw std::runtime_error("Expected exception was not thrown for: " #expression); \
        } catch (const std::exception& e) { \
            (void)e; /* Caught expected exception */ \
        } \
    } while (0)

template<typename T>
void assert_tensor_equals_cpu(const axiom::Tensor& t, const std::vector<T>& expected_data, double epsilon = 1e-6) {
    auto t_cpu = t.cpu();
    ASSERT(t_cpu.device() == axiom::Device::CPU, "Tensor is not on CPU");
    ASSERT(t_cpu.size() == expected_data.size(), "Tensor size mismatch");
    
    const T* t_data = t_cpu.template typed_data<T>();
    for (size_t i = 0; i < expected_data.size(); ++i) {
        if constexpr (std::is_floating_point_v<T>) {
            if (std::abs(static_cast<double>(t_data[i]) - static_cast<double>(expected_data[i])) >= epsilon) {
                throw std::runtime_error("Tensor data mismatch at index " + std::to_string(i));
            }
        } else {
            if (t_data[i] != expected_data[i]) {
                 throw std::runtime_error("Tensor data mismatch at index " + std::to_string(i));
            }
        }
    }
}

// ==================================
//
//      ORIGINAL TEST BED
//
// ==================================

void assert_close(float a, float b, float epsilon = 1e-6) {
    ASSERT(std::abs(a - b) < epsilon, "Floats are not close.");
}

void test_cpu_add_success() {
    auto a = axiom::Tensor::full({2, 2}, 2.0f);
    auto b = axiom::Tensor::full({2, 2}, 3.0f);
    auto c = axiom::ops::add(a, b);
    ASSERT(c.device() == axiom::Device::CPU, "Device mismatch");
    const float* c_data = c.typed_data<float>();
    for (size_t i = 0; i < 4; ++i) assert_close(c_data[i], 5.0f);
}

void test_metal_add_success() {
#ifdef __APPLE__
    if (!axiom::system::is_metal_available()) return;
    auto a = axiom::Tensor::full({2, 2}, 2.0f).to(axiom::Device::GPU);
    auto b = axiom::Tensor::full({2, 2}, 3.0f).to(axiom::Device::GPU);
    auto c = axiom::ops::add(a, b);
    ASSERT(c.device() == axiom::Device::GPU, "Device mismatch");
    auto c_cpu = c.cpu();
    const float* c_data = c_cpu.typed_data<float>();
    for (size_t i = 0; i < 4; ++i) assert_close(c_data[i], 5.0f);
#endif
}

void test_cpu_sub_success() {
    auto a = axiom::Tensor::full({2, 2}, 10.0f);
    auto b = axiom::Tensor::full({2, 2}, 3.0f);
    auto c = axiom::ops::subtract(a, b);
    ASSERT(c.device() == axiom::Device::CPU, "Device mismatch");
    const float* c_data = c.typed_data<float>();
    for (size_t i = 0; i < 4; ++i) assert_close(c_data[i], 7.0f);
}

void test_metal_sub_success() {
#ifdef __APPLE__
    if (!axiom::system::is_metal_available()) return;
    auto a = axiom::Tensor::full({2, 2}, 10.0f).to(axiom::Device::GPU);
    auto b = axiom::Tensor::full({2, 2}, 3.0f).to(axiom::Device::GPU);
    auto c = axiom::ops::subtract(a, b);
    ASSERT(c.device() == axiom::Device::GPU, "Device mismatch");
    auto c_cpu = c.cpu();
    const float* c_data = c_cpu.typed_data<float>();
    for (size_t i = 0; i < 4; ++i) assert_close(c_data[i], 7.0f);
#endif
}

void test_metal_mul_success() {
#ifdef __APPLE__
    if (!axiom::system::is_metal_available()) return;
    auto a = axiom::Tensor::full({2, 2}, 10.0f).to(axiom::Device::GPU);
    auto b = axiom::Tensor::full({2, 2}, 3.0f).to(axiom::Device::GPU);
    auto c = axiom::ops::multiply(a, b);
    ASSERT(c.device() == axiom::Device::GPU, "Device mismatch");
    auto c_cpu = c.cpu();
    const float* c_data = c_cpu.typed_data<float>();
    for (size_t i = 0; i < 4; ++i) assert_close(c_data[i], 30.0f);
#endif
}

void test_metal_div_success() {
#ifdef __APPLE__
    if (!axiom::system::is_metal_available()) return;
    auto a = axiom::Tensor::full({2, 2}, 30.0f).to(axiom::Device::GPU);
    auto b = axiom::Tensor::full({2, 2}, 3.0f).to(axiom::Device::GPU);
    auto c = axiom::ops::divide(a, b);
    ASSERT(c.device() == axiom::Device::GPU, "Device mismatch");
    auto c_cpu = c.cpu();
    const float* c_data = c_cpu.typed_data<float>();
    for (size_t i = 0; i < 4; ++i) assert_close(c_data[i], 10.0f);
#endif
}

void test_cpu_broadcasting_success() {
    auto a = axiom::Tensor::full({2, 2}, 10.0f);
    auto b = axiom::Tensor::full({}, 5.0f); // Scalar
    auto c = axiom::ops::add(a, b);
    ASSERT(c.shape() == axiom::Shape({2, 2}), "Shape mismatch");
    const float* c_data = c.typed_data<float>();
    for (size_t i = 0; i < 4; ++i) assert_close(c_data[i], 15.0f);
}

void test_cpu_type_promotion_success() {
    auto a = axiom::Tensor::full({2, 2}, static_cast<int32_t>(10));
    auto b = axiom::Tensor::full({2, 2}, 5.5f);
    ASSERT(a.dtype() == axiom::DType::Int32, "DType mismatch");
    ASSERT(b.dtype() == axiom::DType::Float32, "DType mismatch");

    auto c = axiom::ops::add(a, b);
    ASSERT(c.dtype() == axiom::DType::Float32, "Promotion failed");
    const float* c_data = c.typed_data<float>();
    for (size_t i = 0; i < 4; ++i) assert_close(c_data[i], 15.5f);
}

void test_unsupported_gpu_op_fallback() {
#ifdef __APPLE__
    if (!axiom::system::is_metal_available()) return;
    // Power is not implemented on Metal, should fall back to CPU.
    auto a = axiom::Tensor::full({2, 2}, 3.0f).to(axiom::Device::GPU);
    auto b = axiom::Tensor::full({2, 2}, 4.0f).to(axiom::Device::GPU);
    auto c = axiom::ops::power(a, b);
    
    ASSERT(c.device() == axiom::Device::CPU, "Device mismatch");
    
    auto c_cpu = c.cpu();
    const float* c_data = c_cpu.typed_data<float>();
    for (size_t i = 0; i < 4; ++i) assert_close(c_data[i], 81.0f);
#endif
}

void test_mixed_device_op_success() {
#ifdef __APPLE__
    if (!axiom::system::is_metal_available()) return;
    auto a = axiom::Tensor::full({2, 2}, 3.0f, axiom::Device::CPU);
    auto b = axiom::Tensor::full({2, 2}, 4.0f).to(axiom::Device::GPU);
    auto c = axiom::ops::add(a, b);

    ASSERT(c.device() == axiom::Device::GPU, "Device mismatch");
    auto c_cpu = c.cpu();
    const float* c_data = c_cpu.typed_data<float>();
    for (size_t i = 0; i < 4; ++i) assert_close(c_data[i], 7.0f);
#endif
}

void test_shape_mismatch_error() {
    auto a = axiom::Tensor::full({2, 3}, 1.0f);
    auto b = axiom::Tensor::full({2, 4}, 1.0f);
    ASSERT_THROWS(axiom::ops::add(a, b));
}

void test_metal_add_int_success() {
#ifdef __APPLE__
    if (!axiom::system::is_metal_available()) return;
    auto a = axiom::Tensor::full({2, 2}, static_cast<int32_t>(1)).to(axiom::Device::GPU);
    auto b = axiom::Tensor::full({2, 2}, static_cast<int32_t>(2)).to(axiom::Device::GPU);
    ASSERT(a.dtype() == axiom::DType::Int32, "DType mismatch");
    ASSERT(b.dtype() == axiom::DType::Int32, "DType mismatch");

    auto c = axiom::ops::add(a, b);
    ASSERT(c.device() == axiom::Device::GPU, "Device mismatch");
    ASSERT(c.dtype() == axiom::DType::Int32, "DType mismatch");

    auto c_cpu = c.cpu();
    const int32_t* c_data = c_cpu.typed_data<int32_t>();
    for (size_t i = 0; i < 4; ++i) ASSERT(c_data[i] == 3, "Data mismatch");
#endif
}

void test_metal_broadcasting_success() {
#ifdef __APPLE__
    if (!axiom::system::is_metal_available()) return;
    auto a = axiom::Tensor::full({2, 2}, 10.0f).to(axiom::Device::GPU);
    auto b = axiom::Tensor::full({}, 5.0f).to(axiom::Device::GPU); 
    
    auto c = axiom::ops::add(a, b);
    
    ASSERT(c.device() == axiom::Device::GPU, "Device mismatch, expected GPU");
    ASSERT(c.shape() == a.shape(), "Broadcast shape mismatch");

    auto c_cpu = c.cpu();
    const float* c_data = c_cpu.typed_data<float>();
    for (size_t i = 0; i < 4; ++i) assert_close(c_data[i], 15.0f);
#endif
}

void test_inplace_operations() {
    // Basic in-place addition
    auto a = axiom::Tensor::full({2, 2}, (int32_t)5);
    auto b = axiom::Tensor::full({2, 2}, (int32_t)3);
    a += b;
    assert_tensor_equals_cpu<int32_t>(a, {8, 8, 8, 8});

    // In-place subtraction
    auto c = axiom::Tensor::full({2, 2}, (int32_t)8);
    auto d = axiom::Tensor::full({2, 2}, (int32_t)2);
    c -= d;
    assert_tensor_equals_cpu<int32_t>(c, {6, 6, 6, 6});

    // In-place multiplication
    auto e = axiom::Tensor::full({2, 2}, (int32_t)6);
    auto f = axiom::Tensor::full({2, 2}, (int32_t)3);
    e *= f;
    assert_tensor_equals_cpu<int32_t>(e, {18, 18, 18, 18});

    // In-place division
    auto g = axiom::Tensor::full({2, 2}, (int32_t)18);
    auto h = axiom::Tensor::full({2, 2}, (int32_t)3);
    g /= h;
    assert_tensor_equals_cpu<int32_t>(g, {6, 6, 6, 6});
    
    // In-place with scalar
    auto i = axiom::Tensor::full({3, 3}, 10.f);
    i += 5.f;
    assert_tensor_equals_cpu<float>(i, {15.f, 15.f, 15.f, 15.f, 15.f, 15.f, 15.f, 15.f, 15.f});

    // Test unsafe type cast
    auto j = axiom::Tensor::full({2, 2}, (int32_t)5);
    auto k = axiom::Tensor::full({2, 2}, 3.f);
    ASSERT_THROWS(j += k);

    // Test shape mismatch
    auto l = axiom::Tensor::full({2, 2}, (int32_t)5);
    auto m = axiom::Tensor::full({3, 3}, (int32_t)5);
    ASSERT_THROWS(l += m);
    
    // Test broadcasting that would change shape
    auto n = axiom::Tensor::full({2, 2}, (int32_t)5);
    auto o = axiom::Tensor::full({2}, (int32_t)5);
    ASSERT_THROWS(n += o);
}

// ==================================
//
//      FULL COVERAGE TESTS
//
// ==================================

// DType list for iteration
const std::vector<axiom::DType> all_dtypes = {
    axiom::DType::Bool, axiom::DType::Int8, axiom::DType::Int16, axiom::DType::Int32, axiom::DType::Int64,
    axiom::DType::UInt8, axiom::DType::UInt16, axiom::DType::UInt32, axiom::DType::UInt64,
    axiom::DType::Float16, axiom::DType::Float32, axiom::DType::Float64,
};

template<typename T_a, typename T_b, typename T_exp>
void test_arithmetic_op(const std::string& op_name, 
                        std::function<axiom::Tensor(const axiom::Tensor&, const axiom::Tensor&)> op,
                        T_a val_a, T_b val_b, T_exp val_exp) {
    auto a = axiom::Tensor::full({2, 2}, val_a);
    auto b = axiom::Tensor::full({2, 2}, val_b);
    
    auto result = op(a, b);
    
    auto expected_dtype = axiom::ops::promote_types(a.dtype(), b.dtype());
    ASSERT(result.dtype() == expected_dtype, "DType promotion failed for " + op_name);
    
    std::vector<T_exp> expected_data(4, val_exp);
    assert_tensor_equals_cpu<T_exp>(result, expected_data);
}

void test_all_arithmetic_ops() {
    test_arithmetic_op<float, int, float>("add", axiom::ops::add, 2.5f, 3, 5.5f);
    test_arithmetic_op<int, float, float>("sub", axiom::ops::subtract, 10, 3.5f, 6.5f);
    test_arithmetic_op<double, int, double>("mul", axiom::ops::multiply, 2.5, 4, 10.0);
    test_arithmetic_op<int, int, int>("div", axiom::ops::divide, 10, 3, 3);
    test_arithmetic_op<uint8_t, int8_t, int16_t>("add_mixed_sign", axiom::ops::add, 10, -5, 5);
}

void test_full_broadcasting(axiom::Device device) {
    if (device == axiom::Device::GPU && !axiom::system::is_metal_available()) {
        return;
    }

    // (2, 3) + scalar
    auto a_data = std::vector<float>{1, 2, 3, 4, 5, 6};
    auto a = axiom::Tensor::from_data(a_data.data(), {2, 3}).to(device);
    auto b_scalar = axiom::Tensor::full({}, 10.0f).to(device);
    auto c = axiom::ops::add(a, b_scalar);
    ASSERT(c.shape() == a.shape(), "Scalar broadcast shape mismatch");
    assert_tensor_equals_cpu<float>(c, {11, 12, 13, 14, 15, 16});
    
    // (2, 3) + (3,)
    auto b_vec_data = std::vector<float>{10, 20, 30};
    auto b_vec = axiom::Tensor::from_data(b_vec_data.data(), {3}).to(device);
    auto d = axiom::ops::add(a, b_vec);
    ASSERT(d.shape() == a.shape(), "Vector broadcast shape mismatch");
    assert_tensor_equals_cpu<float>(d, {11, 22, 33, 14, 25, 36});
    
    // (2, 3) + (2, 1)
    auto b_col_data = std::vector<float>{10, 20};
    auto b_col = axiom::Tensor::from_data(b_col_data.data(), {2, 1}).to(device);
    auto e = axiom::ops::add(a, b_col);
    ASSERT(e.shape() == a.shape(), "Column broadcast shape mismatch");
    assert_tensor_equals_cpu<float>(e, {11, 12, 13, 24, 25, 26});
    
    // (1, 3) + (2, 1) -> (2, 3)
    auto a_row_data = std::vector<float>{1, 2, 3};
    auto a_row = axiom::Tensor::from_data(a_row_data.data(), {1, 3}).to(device);
    auto f = axiom::ops::add(a_row, b_col);
    axiom::Shape expected_shape = {2, 3};
    ASSERT(f.shape() == expected_shape, "2D-2D broadcast shape mismatch");
    assert_tensor_equals_cpu<float>(f, {11, 12, 13, 21, 22, 23});
    
    // Incompatible broadcast
    auto b_wrong = axiom::Tensor::full({4}, 1.0f).to(device);
    ASSERT_THROWS(axiom::ops::add(a, b_wrong));
}

void test_type_promotion_grid() {
    for (auto dtype_a : all_dtypes) {
        for (auto dtype_b : all_dtypes) {
            // Create dummy tensors of the specified types
            auto a = axiom::Tensor::empty({1}, dtype_a);
            auto b = axiom::Tensor::empty({1}, dtype_b);
            
            // Get expected promotion result
            auto expected_dtype = axiom::ops::promote_types(dtype_a, dtype_b);
            
            // Perform an operation (add is fine, we only care about the resulting type)
            auto result = axiom::ops::add(a, b);
            
            std::string test_msg = "Promotion failed for " + axiom::dtype_name(dtype_a) +
                                   " + " + axiom::dtype_name(dtype_b);
            ASSERT(result.dtype() == expected_dtype, test_msg);
        }
    }
}

int main(int argc, char** argv) {
    axiom::ops::OperationRegistry::initialize_builtin_operations();

    std::cout << "========================================" << std::endl;
    std::cout << "         Axiom Test Suite" << std::endl;
    std::cout << "========================================" << std::endl;
    
    RUN_TEST(test_cpu_add_success);
    RUN_TEST(test_metal_add_success);
    RUN_TEST(test_cpu_sub_success);
    RUN_TEST(test_metal_sub_success);
    RUN_TEST(test_metal_mul_success);
    RUN_TEST(test_metal_div_success);
    RUN_TEST(test_cpu_broadcasting_success);
    RUN_TEST(test_cpu_type_promotion_success);
    RUN_TEST(test_unsupported_gpu_op_fallback);
    RUN_TEST(test_mixed_device_op_success);
    RUN_TEST(test_shape_mismatch_error);
    RUN_TEST(test_metal_add_int_success);
    RUN_TEST(test_metal_broadcasting_success);
    RUN_TEST(test_inplace_operations);
    std::cout << "--- Finished Original Test Bed ---\n" << std::endl;

    std::cout << "--- Running Full Coverage Tests ---" << std::endl;
    RUN_TEST(test_all_arithmetic_ops);
    RUN_TEST(test_full_broadcasting, axiom::Device::CPU);
    RUN_TEST(test_full_broadcasting, axiom::Device::GPU);
    RUN_TEST(test_type_promotion_grid);
    std::cout << "--- Finished Full Coverage Tests ---\n" << std::endl;

    std::cout << "========================================" << std::endl;
    std::cout << "Test Suite Summary:" << std::endl;
    std::cout << "    " << tests_passed << " / " << tests_run << " tests passed." << std::endl;
    std::cout << "========================================" << std::endl;

    return (tests_passed == tests_run) ? 0 : 1;
} 