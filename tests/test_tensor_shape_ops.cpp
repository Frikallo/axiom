#include <iostream>
#include <string>
#include <vector>
#include <functional>
#include <stdexcept>
#include <cmath>

#include <axiom/axiom.hpp>

// Test harness
static int tests_run = 0;
static int tests_passed = 0;

#define RUN_TEST(test_func) run_test([&]() { test_func(); }, #test_func)

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
    }
    std::cout << std::endl;
}

#define ASSERT(condition, msg) \
    do { \
        if (!(condition)) { \
            throw std::runtime_error("Assertion failed: (" #condition ") - " + std::string(msg)); \
        } \
    } while (0)

template<typename T>
void assert_tensor_equals_cpu(const axiom::Tensor& t, const std::vector<T>& expected_data, double epsilon = 1e-6) {
    auto t_cpu = t.cpu();
    ASSERT(t_cpu.device() == axiom::Device::CPU, "Tensor is not on CPU");
    ASSERT(t_cpu.size() == expected_data.size(), "Tensor size mismatch");
    
    // For contiguous tensors, use direct pointer access for speed
    if (t_cpu.is_contiguous()) {
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
    } else {
        // For non-contiguous tensors (like expand with zero strides), use item()
        std::vector<size_t> indices(t_cpu.ndim(), 0);
        for (size_t i = 0; i < expected_data.size(); ++i) {
            T val = t_cpu.item<T>(indices);
            if constexpr (std::is_floating_point_v<T>) {
                if (std::abs(static_cast<double>(val) - static_cast<double>(expected_data[i])) >= epsilon) {
                    throw std::runtime_error("Tensor data mismatch at index " + std::to_string(i));
                }
            } else {
                if (val != expected_data[i]) {
                     throw std::runtime_error("Tensor data mismatch at index " + std::to_string(i));
                }
            }
            
            // Increment indices in row-major order
            for (int j = t_cpu.ndim() - 1; j >= 0; --j) {
                if (++indices[j] < t_cpu.shape()[j]) {
                    break;
                }
                indices[j] = 0;
            }
        }
    }
}

// Test arange
void test_arange_basic() {
    auto t = axiom::Tensor::arange(5);
    ASSERT(t.shape() == axiom::Shape({5}), "Shape mismatch");
    assert_tensor_equals_cpu<int>(t, {0, 1, 2, 3, 4});
}

void test_arange_start_end() {
    auto t = axiom::Tensor::arange(2, 8);
    ASSERT(t.shape() == axiom::Shape({6}), "Shape mismatch");
    assert_tensor_equals_cpu<int>(t, {2, 3, 4, 5, 6, 7});
}

void test_arange_with_step() {
    auto t = axiom::Tensor::arange(0, 10, 2);
    ASSERT(t.shape() == axiom::Shape({5}), "Shape mismatch");
    assert_tensor_equals_cpu<int>(t, {0, 2, 4, 6, 8});
}

// Test flatten
void test_flatten_default() {
    auto t = axiom::Tensor::arange(24).reshape({2, 3, 4});
    auto flat = t.flatten();
    ASSERT(flat.ndim() == 1, "Should be 1D");
    ASSERT(flat.shape()[0] == 24, "Size mismatch");
}

void test_flatten_partial() {
    auto t = axiom::Tensor::arange(24).reshape({2, 3, 4});
    auto flat = t.flatten(1, 2);  // Flatten dims 1 and 2
    ASSERT(flat.shape() == axiom::Shape({2, 12}), "Shape mismatch");
}

void test_flatten_negative_index() {
    auto t = axiom::Tensor::arange(24).reshape({2, 3, 4});
    auto flat = t.flatten(0, -1);  // Flatten all dims
    ASSERT(flat.ndim() == 1, "Should be 1D");
    ASSERT(flat.shape()[0] == 24, "Size mismatch");
}

// Test expand (zero-copy broadcast)
void test_expand_basic() {
    auto t = axiom::Tensor::ones({1, 4});
    auto expanded = t.expand({3, 4});
    
    ASSERT(expanded.shape() == axiom::Shape({3, 4}), "Shape mismatch");
    ASSERT(expanded.has_zero_stride(), "Should have zero stride");
    
    // Verify data
    std::vector<float> expected(12, 1.0f);
    assert_tensor_equals_cpu<float>(expanded, expected);
}

void test_expand_multidim() {
    auto t = axiom::Tensor::ones({1, 1, 4});
    auto expanded = t.expand({2, 3, 4});
    
    ASSERT(expanded.shape() == axiom::Shape({2, 3, 4}), "Shape mismatch");
    ASSERT(expanded.has_zero_stride(), "Should have zero stride");
}

void test_expand_as() {
    auto t = axiom::Tensor::ones({1, 4});
    auto target = axiom::Tensor::zeros({3, 4});
    auto expanded = t.expand_as(target);
    
    ASSERT(expanded.shape() == target.shape(), "Shape should match target");
}

void test_broadcast_to() {
    auto t = axiom::Tensor::ones({1, 4});
    auto broadcasted = t.broadcast_to({3, 4});
    
    ASSERT(broadcasted.shape() == axiom::Shape({3, 4}), "Shape mismatch");
    ASSERT(broadcasted.has_zero_stride(), "Should have zero stride");
}

// Test repeat (copies data)
void test_repeat_basic() {
    auto t = axiom::Tensor::arange(4).reshape({2, 2});
    auto repeated = t.repeat({2, 3});
    
    ASSERT(repeated.shape() == axiom::Shape({4, 6}), "Shape mismatch");
    ASSERT(!repeated.has_zero_stride(), "Should NOT have zero stride (data copied)");
}

void test_repeat_single_dim() {
    auto t = axiom::Tensor::arange(4);
    auto repeated = t.repeat({3});
    
    ASSERT(repeated.shape() == axiom::Shape({12}), "Shape mismatch");
    // Expected: [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
    assert_tensor_equals_cpu<int>(repeated, {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3});
}

void test_tile_alias() {
    auto t = axiom::Tensor::arange(4);
    auto tiled = t.tile({2});
    
    ASSERT(tiled.shape() == axiom::Shape({8}), "Shape mismatch");
}

// Test from_data
void test_from_data() {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    auto t = axiom::Tensor::from_data(data.data(), {2, 3});
    
    ASSERT(t.shape() == axiom::Shape({2, 3}), "Shape mismatch");
    assert_tensor_equals_cpu<float>(t, data);
}

// Test rearrange (if implemented)
void test_rearrange_flatten() {
    auto t = axiom::Tensor::arange(6).reshape({2, 3}).astype(axiom::DType::Float32);
    auto r = t.rearrange("h w -> (h w)");
    
    ASSERT(r.ndim() == 1, "Should be 1D");
    ASSERT(r.shape()[0] == 6, "Size mismatch");
}

void test_rearrange_transpose() {
    auto t = axiom::Tensor::arange(6).reshape({2, 3}).astype(axiom::DType::Float32);
    auto r = t.rearrange("h w -> w h");
    
    ASSERT(r.shape() == axiom::Shape({3, 2}), "Shape mismatch");
}

int main() {
    axiom::ops::OperationRegistry::initialize_builtin_operations();
    
    std::cout << "========================================" << std::endl;
    std::cout << "   Shape Operations Test Suite" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    // arange tests
    RUN_TEST(test_arange_basic);
    RUN_TEST(test_arange_start_end);
    RUN_TEST(test_arange_with_step);
    
    // flatten tests
    RUN_TEST(test_flatten_default);
    RUN_TEST(test_flatten_partial);
    RUN_TEST(test_flatten_negative_index);
    
    // expand tests
    RUN_TEST(test_expand_basic);
    RUN_TEST(test_expand_multidim);
    RUN_TEST(test_expand_as);
    RUN_TEST(test_broadcast_to);
    
    // repeat/tile tests
    RUN_TEST(test_repeat_basic);
    RUN_TEST(test_repeat_single_dim);
    RUN_TEST(test_tile_alias);
    
    // from_data test
    RUN_TEST(test_from_data);
    
    // rearrange tests
    RUN_TEST(test_rearrange_flatten);
    RUN_TEST(test_rearrange_transpose);
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "Test Suite Summary:" << std::endl;
    std::cout << "    " << tests_passed << " / " << tests_run << " tests passed." << std::endl;
    std::cout << "========================================" << std::endl;
    
    return (tests_passed == tests_run) ? 0 : 1;
}
