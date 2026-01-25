#include <axiom/axiom.hpp>
#include <cassert>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace axiom;

// ==================================
//
//      TEST HARNESS
//
// ==================================

static int tests_run = 0;
static int tests_passed = 0;
static std::string current_test_name;

#define RUN_TEST(test_func) run_test([&]() { test_func(); }, #test_func)

void run_test(const std::function<void()> &test_func,
              const std::string &test_name) {
    tests_run++;
    current_test_name = test_name;
    std::cout << "--- Running: " << test_name << " ---" << std::endl;
    try {
        test_func();
        std::cout << "--- PASSED: " << test_name << " ---" << std::endl;
        tests_passed++;
    } catch (const std::exception &e) {
        std::cerr << "--- FAILED: " << test_name << " ---" << std::endl;
        std::cerr << "    Error: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "--- FAILED: " << test_name << " ---" << std::endl;
        std::cerr << "    Error: Unknown exception caught." << std::endl;
    }
    std::cout << std::endl;
}

#define ASSERT(condition, msg)                                                 \
    do {                                                                       \
        if (!(condition)) {                                                    \
            throw std::runtime_error("Assertion failed: (" #condition ") - " + \
                                     std::string(msg));                        \
        }                                                                      \
    } while (0)

// ==================================
//
//      PARITY COMPARISON HELPERS
//
// ==================================

// Helper to convert shape to string
static std::string shape_to_string(const Shape &shape) {
    std::string s = "(";
    for (size_t i = 0; i < shape.size(); ++i) {
        s += std::to_string(shape[i]);
        if (i < shape.size() - 1)
            s += ", ";
    }
    s += ")";
    return s;
}

// Compare two tensors element-wise within tolerance
bool tensors_equal(const Tensor &cpu, const Tensor &gpu,
                   float tolerance = 1e-5f) {
    auto gpu_cpu = gpu.cpu();

    if (cpu.shape() != gpu_cpu.shape()) {
        std::cerr << "Shape mismatch: CPU " << shape_to_string(cpu.shape())
                  << " vs GPU " << shape_to_string(gpu_cpu.shape())
                  << std::endl;
        return false;
    }

    if (cpu.dtype() != gpu_cpu.dtype()) {
        std::cerr << "DType mismatch: CPU " << dtype_name(cpu.dtype())
                  << " vs GPU " << dtype_name(gpu_cpu.dtype()) << std::endl;
        return false;
    }

    // Compare element-wise based on dtype
    switch (cpu.dtype()) {
    case DType::Float32: {
        const float *cpu_data = cpu.typed_data<float>();
        const float *gpu_data = gpu_cpu.typed_data<float>();
        for (size_t i = 0; i < cpu.size(); ++i) {
            float diff = std::abs(cpu_data[i] - gpu_data[i]);
            if (diff > tolerance &&
                diff > tolerance * std::max(std::abs(cpu_data[i]),
                                            std::abs(gpu_data[i]))) {
                std::cerr << "Mismatch at index " << i
                          << ": CPU=" << cpu_data[i] << " GPU=" << gpu_data[i]
                          << " diff=" << diff << std::endl;
                return false;
            }
        }
        break;
    }
    case DType::Float64: {
        const double *cpu_data = cpu.typed_data<double>();
        const double *gpu_data = gpu_cpu.typed_data<double>();
        for (size_t i = 0; i < cpu.size(); ++i) {
            double diff = std::abs(cpu_data[i] - gpu_data[i]);
            if (diff > tolerance &&
                diff > tolerance * std::max(std::abs(cpu_data[i]),
                                            std::abs(gpu_data[i]))) {
                std::cerr << "Mismatch at index " << i
                          << ": CPU=" << cpu_data[i] << " GPU=" << gpu_data[i]
                          << " diff=" << diff << std::endl;
                return false;
            }
        }
        break;
    }
    case DType::Int32: {
        const int32_t *cpu_data = cpu.typed_data<int32_t>();
        const int32_t *gpu_data = gpu_cpu.typed_data<int32_t>();
        for (size_t i = 0; i < cpu.size(); ++i) {
            if (cpu_data[i] != gpu_data[i]) {
                std::cerr << "Mismatch at index " << i
                          << ": CPU=" << cpu_data[i] << " GPU=" << gpu_data[i]
                          << std::endl;
                return false;
            }
        }
        break;
    }
    case DType::Int64: {
        const int64_t *cpu_data = cpu.typed_data<int64_t>();
        const int64_t *gpu_data = gpu_cpu.typed_data<int64_t>();
        for (size_t i = 0; i < cpu.size(); ++i) {
            if (cpu_data[i] != gpu_data[i]) {
                std::cerr << "Mismatch at index " << i
                          << ": CPU=" << cpu_data[i] << " GPU=" << gpu_data[i]
                          << std::endl;
                return false;
            }
        }
        break;
    }
    case DType::Bool: {
        const bool *cpu_data = cpu.typed_data<bool>();
        const bool *gpu_data = gpu_cpu.typed_data<bool>();
        for (size_t i = 0; i < cpu.size(); ++i) {
            if (cpu_data[i] != gpu_data[i]) {
                std::cerr << "Mismatch at index " << i
                          << ": CPU=" << cpu_data[i] << " GPU=" << gpu_data[i]
                          << std::endl;
                return false;
            }
        }
        break;
    }
    default:
        std::cerr << "Unsupported dtype for comparison: "
                  << dtype_name(cpu.dtype()) << std::endl;
        return false;
    }

    return true;
}

// ==================================
//
//      BINARY OPERATION PARITY TESTS
//
// ==================================

void test_add_parity() {
    if (!system::should_run_gpu_tests()) {
        std::cout << "  Skipping (GPU tests disabled)" << std::endl;
        return;
    }

    auto a = Tensor::randn({4, 5}, DType::Float32, Device::CPU);
    auto b = Tensor::randn({4, 5}, DType::Float32, Device::CPU);

    auto cpu_result = ops::add(a, b);
    auto gpu_result = ops::add(a.gpu(), b.gpu());

    ASSERT(tensors_equal(cpu_result, gpu_result), "Add parity failed");
}

void test_subtract_parity() {
    if (!system::should_run_gpu_tests())
        return;

    auto a = Tensor::randn({4, 5}, DType::Float32, Device::CPU);
    auto b = Tensor::randn({4, 5}, DType::Float32, Device::CPU);

    auto cpu_result = ops::subtract(a, b);
    auto gpu_result = ops::subtract(a.gpu(), b.gpu());

    ASSERT(tensors_equal(cpu_result, gpu_result), "Subtract parity failed");
}

void test_multiply_parity() {
    if (!system::should_run_gpu_tests())
        return;

    auto a = Tensor::randn({4, 5}, DType::Float32, Device::CPU);
    auto b = Tensor::randn({4, 5}, DType::Float32, Device::CPU);

    auto cpu_result = ops::multiply(a, b);
    auto gpu_result = ops::multiply(a.gpu(), b.gpu());

    ASSERT(tensors_equal(cpu_result, gpu_result), "Multiply parity failed");
}

void test_divide_parity() {
    if (!system::should_run_gpu_tests())
        return;

    auto a = Tensor::randn({4, 5}, DType::Float32, Device::CPU);
    auto b = Tensor::randn({4, 5}, DType::Float32, Device::CPU);
    // Avoid division by very small numbers
    b = ops::add(b, Tensor::full({4, 5}, 1.0f, Device::CPU));

    auto cpu_result = ops::divide(a, b);
    auto gpu_result = ops::divide(a.gpu(), b.gpu());

    ASSERT(tensors_equal(cpu_result, gpu_result), "Divide parity failed");
}

void test_maximum_parity() {
    if (!system::should_run_gpu_tests())
        return;

    auto a = Tensor::randn({4, 5}, DType::Float32, Device::CPU);
    auto b = Tensor::randn({4, 5}, DType::Float32, Device::CPU);

    auto cpu_result = ops::maximum(a, b);
    auto gpu_result = ops::maximum(a.gpu(), b.gpu());

    ASSERT(tensors_equal(cpu_result, gpu_result), "Maximum parity failed");
}

void test_minimum_parity() {
    if (!system::should_run_gpu_tests())
        return;

    auto a = Tensor::randn({4, 5}, DType::Float32, Device::CPU);
    auto b = Tensor::randn({4, 5}, DType::Float32, Device::CPU);

    auto cpu_result = ops::minimum(a, b);
    auto gpu_result = ops::minimum(a.gpu(), b.gpu());

    ASSERT(tensors_equal(cpu_result, gpu_result), "Minimum parity failed");
}

void test_hypot_parity() {
    if (!system::should_run_gpu_tests())
        return;

    auto a = Tensor::randn({4, 5}, DType::Float32, Device::CPU);
    auto b = Tensor::randn({4, 5}, DType::Float32, Device::CPU);

    auto cpu_result = ops::hypot(a, b);
    auto gpu_result = ops::hypot(a.gpu(), b.gpu());

    ASSERT(tensors_equal(cpu_result, gpu_result, 1e-4f), "Hypot parity failed");
}

// ==================================
//
//      COMPARISON OPERATION PARITY TESTS
//
// ==================================

void test_equal_parity() {
    if (!system::should_run_gpu_tests())
        return;

    auto a = Tensor::randn({4, 5}, DType::Float32, Device::CPU);
    auto b = a.copy(); // Same values

    auto cpu_result = ops::equal(a, b);
    auto gpu_result = ops::equal(a.gpu(), b.gpu());

    ASSERT(tensors_equal(cpu_result, gpu_result), "Equal parity failed");
}

void test_less_parity() {
    if (!system::should_run_gpu_tests())
        return;

    auto a = Tensor::randn({4, 5}, DType::Float32, Device::CPU);
    auto b = Tensor::randn({4, 5}, DType::Float32, Device::CPU);

    auto cpu_result = ops::less(a, b);
    auto gpu_result = ops::less(a.gpu(), b.gpu());

    ASSERT(tensors_equal(cpu_result, gpu_result), "Less parity failed");
}

void test_greater_parity() {
    if (!system::should_run_gpu_tests())
        return;

    auto a = Tensor::randn({4, 5}, DType::Float32, Device::CPU);
    auto b = Tensor::randn({4, 5}, DType::Float32, Device::CPU);

    auto cpu_result = ops::greater(a, b);
    auto gpu_result = ops::greater(a.gpu(), b.gpu());

    ASSERT(tensors_equal(cpu_result, gpu_result), "Greater parity failed");
}

// ==================================
//
//      LOGICAL OPERATION PARITY TESTS
//
// ==================================

void test_logical_and_parity() {
    if (!system::should_run_gpu_tests())
        return;

    auto a = Tensor::randn({4, 5}, DType::Float32, Device::CPU);
    auto b = Tensor::randn({4, 5}, DType::Float32, Device::CPU);

    auto cpu_result = ops::logical_and(a, b);
    auto gpu_result = ops::logical_and(a.gpu(), b.gpu());

    ASSERT(tensors_equal(cpu_result, gpu_result), "LogicalAnd parity failed");
}

void test_logical_or_parity() {
    if (!system::should_run_gpu_tests())
        return;

    auto a = Tensor::randn({4, 5}, DType::Float32, Device::CPU);
    auto b = Tensor::randn({4, 5}, DType::Float32, Device::CPU);

    auto cpu_result = ops::logical_or(a, b);
    auto gpu_result = ops::logical_or(a.gpu(), b.gpu());

    ASSERT(tensors_equal(cpu_result, gpu_result), "LogicalOr parity failed");
}

void test_logical_not_parity() {
    if (!system::should_run_gpu_tests())
        return;

    auto a = Tensor::randn({4, 5}, DType::Float32, Device::CPU);

    auto cpu_result = ops::logical_not(a);
    auto gpu_result = ops::logical_not(a.gpu());

    ASSERT(tensors_equal(cpu_result, gpu_result), "LogicalNot parity failed");
}

// ==================================
//
//      UNARY OPERATION PARITY TESTS
//
// ==================================

void test_negate_parity() {
    if (!system::should_run_gpu_tests())
        return;

    auto a = Tensor::randn({4, 5}, DType::Float32, Device::CPU);

    auto cpu_result = ops::negate(a);
    auto gpu_result = ops::negate(a.gpu());

    ASSERT(tensors_equal(cpu_result, gpu_result), "Negate parity failed");
}

void test_abs_parity() {
    if (!system::should_run_gpu_tests())
        return;

    auto a = Tensor::randn({4, 5}, DType::Float32, Device::CPU);

    auto cpu_result = ops::abs(a);
    auto gpu_result = ops::abs(a.gpu());

    ASSERT(tensors_equal(cpu_result, gpu_result), "Abs parity failed");
}

void test_sqrt_parity() {
    if (!system::should_run_gpu_tests())
        return;

    // Use positive values for sqrt
    auto a = ops::abs(Tensor::randn({4, 5}, DType::Float32, Device::CPU));
    a = ops::add(a, Tensor::full({4, 5}, 0.1f, Device::CPU)); // Avoid sqrt(0)

    auto cpu_result = ops::sqrt(a);
    auto gpu_result = ops::sqrt(a.gpu());

    ASSERT(tensors_equal(cpu_result, gpu_result), "Sqrt parity failed");
}

void test_exp_parity() {
    if (!system::should_run_gpu_tests())
        return;

    // Use small values to avoid overflow
    auto a = ops::divide(Tensor::randn({4, 5}, DType::Float32, Device::CPU),
                         Tensor::full({4, 5}, 10.0f, Device::CPU));

    auto cpu_result = ops::exp(a);
    auto gpu_result = ops::exp(a.gpu());

    ASSERT(tensors_equal(cpu_result, gpu_result, 1e-4f), "Exp parity failed");
}

void test_log_parity() {
    if (!system::should_run_gpu_tests())
        return;

    // Use positive values for log
    auto a = ops::abs(Tensor::randn({4, 5}, DType::Float32, Device::CPU));
    a = ops::add(a, Tensor::full({4, 5}, 1.0f, Device::CPU));

    auto cpu_result = ops::log(a);
    auto gpu_result = ops::log(a.gpu());

    ASSERT(tensors_equal(cpu_result, gpu_result), "Log parity failed");
}

void test_sin_parity() {
    if (!system::should_run_gpu_tests())
        return;

    auto a = Tensor::randn({4, 5}, DType::Float32, Device::CPU);

    auto cpu_result = ops::sin(a);
    auto gpu_result = ops::sin(a.gpu());

    ASSERT(tensors_equal(cpu_result, gpu_result), "Sin parity failed");
}

void test_cos_parity() {
    if (!system::should_run_gpu_tests())
        return;

    auto a = Tensor::randn({4, 5}, DType::Float32, Device::CPU);

    auto cpu_result = ops::cos(a);
    auto gpu_result = ops::cos(a.gpu());

    ASSERT(tensors_equal(cpu_result, gpu_result), "Cos parity failed");
}

// ==================================
//
//      REDUCTION OPERATION PARITY TESTS
//
// ==================================

void test_sum_parity() {
    if (!system::should_run_gpu_tests())
        return;

    auto a = Tensor::randn({4, 5, 3}, DType::Float32, Device::CPU);

    // Full reduction
    auto cpu_result = ops::sum(a, {}, false);
    auto gpu_result = ops::sum(a.gpu(), {}, false);
    ASSERT(tensors_equal(cpu_result, gpu_result, 1e-4f),
           "Sum full parity failed");

    // Axis reduction
    cpu_result = ops::sum(a, {1}, false);
    gpu_result = ops::sum(a.gpu(), {1}, false);
    ASSERT(tensors_equal(cpu_result, gpu_result, 1e-4f),
           "Sum axis parity failed");

    // Keep dims
    cpu_result = ops::sum(a, {1}, true);
    gpu_result = ops::sum(a.gpu(), {1}, true);
    ASSERT(tensors_equal(cpu_result, gpu_result, 1e-4f),
           "Sum keep_dims parity failed");
}

void test_mean_parity() {
    if (!system::should_run_gpu_tests())
        return;

    auto a = Tensor::randn({4, 5, 3}, DType::Float32, Device::CPU);

    auto cpu_result = ops::mean(a, {1}, false);
    auto gpu_result = ops::mean(a.gpu(), {1}, false);

    ASSERT(tensors_equal(cpu_result, gpu_result, 1e-4f), "Mean parity failed");
}

void test_max_parity() {
    if (!system::should_run_gpu_tests())
        return;

    auto a = Tensor::randn({4, 5, 3}, DType::Float32, Device::CPU);

    auto cpu_result = ops::max(a, {1}, false);
    auto gpu_result = ops::max(a.gpu(), {1}, false);

    ASSERT(tensors_equal(cpu_result, gpu_result), "Max parity failed");
}

void test_min_parity() {
    if (!system::should_run_gpu_tests())
        return;

    auto a = Tensor::randn({4, 5, 3}, DType::Float32, Device::CPU);

    auto cpu_result = ops::min(a, {1}, false);
    auto gpu_result = ops::min(a.gpu(), {1}, false);

    ASSERT(tensors_equal(cpu_result, gpu_result), "Min parity failed");
}

void test_argmax_parity() {
    if (!system::should_run_gpu_tests())
        return;

    auto a = Tensor::randn({4, 5}, DType::Float32, Device::CPU);

    auto cpu_result = ops::argmax(a, 1, false);
    auto gpu_result = ops::argmax(a.gpu(), 1, false);

    ASSERT(tensors_equal(cpu_result, gpu_result), "ArgMax parity failed");
}

void test_argmin_parity() {
    if (!system::should_run_gpu_tests())
        return;

    auto a = Tensor::randn({4, 5}, DType::Float32, Device::CPU);

    auto cpu_result = ops::argmin(a, 1, false);
    auto gpu_result = ops::argmin(a.gpu(), 1, false);

    ASSERT(tensors_equal(cpu_result, gpu_result), "ArgMin parity failed");
}

// ==================================
//
//      MATMUL PARITY TESTS
//
// ==================================

void test_matmul_2d_parity() {
    if (!system::should_run_gpu_tests())
        return;

    auto a = Tensor::randn({4, 5}, DType::Float32, Device::CPU);
    auto b = Tensor::randn({5, 3}, DType::Float32, Device::CPU);

    auto cpu_result = ops::matmul(a, b);
    auto gpu_result = ops::matmul(a.gpu(), b.gpu());

    ASSERT(tensors_equal(cpu_result, gpu_result, 1e-4f),
           "MatMul 2D parity failed");
}

void test_matmul_batched_parity() {
    if (!system::should_run_gpu_tests())
        return;

    auto a = Tensor::randn({2, 4, 5}, DType::Float32, Device::CPU);
    auto b = Tensor::randn({2, 5, 3}, DType::Float32, Device::CPU);

    auto cpu_result = ops::matmul(a, b);
    auto gpu_result = ops::matmul(a.gpu(), b.gpu());

    ASSERT(tensors_equal(cpu_result, gpu_result, 1e-4f),
           "MatMul batched parity failed");
}

// ==================================
//
//      SPECIAL OPERATION PARITY TESTS
//
// ==================================

void test_where_parity() {
    if (!system::should_run_gpu_tests())
        return;

    auto a = Tensor::randn({4, 5}, DType::Float32, Device::CPU);
    auto b = Tensor::randn({4, 5}, DType::Float32, Device::CPU);
    auto cond =
        ops::greater(Tensor::randn({4, 5}, DType::Float32, Device::CPU),
                     Tensor::zeros({4, 5}, DType::Float32, Device::CPU));

    auto cpu_result = ops::where(cond, a, b);
    auto gpu_result = ops::where(cond.gpu(), a.gpu(), b.gpu());

    ASSERT(tensors_equal(cpu_result, gpu_result), "Where parity failed");
}

void test_softmax_parity() {
    if (!system::should_run_gpu_tests())
        return;

    auto a = Tensor::randn({4, 5}, DType::Float32, Device::CPU);

    auto cpu_result = ops::softmax(a, -1);
    auto gpu_result = ops::softmax(a.gpu(), -1);

    ASSERT(tensors_equal(cpu_result, gpu_result, 1e-4f),
           "Softmax parity failed");
}

void test_gather_parity() {
    if (!system::should_run_gpu_tests())
        return;

    auto input = Tensor::randn({4, 5}, DType::Float32, Device::CPU);
    int64_t idx_data[] = {0, 2, 1, 3};
    auto indices = Tensor::from_data(idx_data, {4});

    auto cpu_result =
        ops::gather(input, 1, indices.unsqueeze(1).expand({4, 1}));
    auto gpu_result =
        ops::gather(input.gpu(), 1, indices.unsqueeze(1).expand({4, 1}).gpu());

    ASSERT(tensors_equal(cpu_result, gpu_result), "Gather parity failed");
}

void test_index_select_parity() {
    if (!system::should_run_gpu_tests())
        return;

    auto input = Tensor::randn({4, 5}, DType::Float32, Device::CPU);
    int64_t idx_data[] = {0, 2, 1};
    auto indices = Tensor::from_data(idx_data, {3});

    auto cpu_result = ops::index_select(input, 0, indices);
    auto gpu_result = ops::index_select(input.gpu(), 0, indices.gpu());

    ASSERT(tensors_equal(cpu_result, gpu_result), "IndexSelect parity failed");
}

// ==================================
//
//      BROADCASTING PARITY TESTS
//
// ==================================

void test_broadcast_add_parity() {
    if (!system::should_run_gpu_tests())
        return;

    auto a = Tensor::randn({4, 5, 3}, DType::Float32, Device::CPU);
    auto b = Tensor::randn({1, 3}, DType::Float32, Device::CPU);

    auto cpu_result = ops::add(a, b);
    auto gpu_result = ops::add(a.gpu(), b.gpu());

    ASSERT(tensors_equal(cpu_result, gpu_result),
           "Broadcast add parity failed");
}

void test_broadcast_multiply_parity() {
    if (!system::should_run_gpu_tests())
        return;

    auto a = Tensor::randn({4, 5}, DType::Float32, Device::CPU);
    auto b = Tensor::randn({5}, DType::Float32, Device::CPU);

    auto cpu_result = ops::multiply(a, b);
    auto gpu_result = ops::multiply(a.gpu(), b.gpu());

    ASSERT(tensors_equal(cpu_result, gpu_result),
           "Broadcast multiply parity failed");
}

// ==================================
//
//      MAIN
//
// ==================================

int main() {
    ops::OperationRegistry::initialize_builtin_operations();

    std::cout << "=== CPU/GPU Parity Tests ===" << std::endl << std::endl;

    if (!system::should_run_gpu_tests()) {
        std::cout << "GPU tests disabled or Metal not available, skipping"
                  << std::endl;
        return 0;
    }

    // Binary operations
    std::cout << "--- Binary Operations ---" << std::endl;
    RUN_TEST(test_add_parity);
    RUN_TEST(test_subtract_parity);
    RUN_TEST(test_multiply_parity);
    RUN_TEST(test_divide_parity);
    RUN_TEST(test_maximum_parity);
    RUN_TEST(test_minimum_parity);
    RUN_TEST(test_hypot_parity);

    // Comparison operations
    std::cout << "--- Comparison Operations ---" << std::endl;
    RUN_TEST(test_equal_parity);
    RUN_TEST(test_less_parity);
    RUN_TEST(test_greater_parity);

    // Logical operations
    std::cout << "--- Logical Operations ---" << std::endl;
    RUN_TEST(test_logical_and_parity);
    RUN_TEST(test_logical_or_parity);
    RUN_TEST(test_logical_not_parity);

    // Unary operations
    std::cout << "--- Unary Operations ---" << std::endl;
    RUN_TEST(test_negate_parity);
    RUN_TEST(test_abs_parity);
    RUN_TEST(test_sqrt_parity);
    RUN_TEST(test_exp_parity);
    RUN_TEST(test_log_parity);
    RUN_TEST(test_sin_parity);
    RUN_TEST(test_cos_parity);

    // Reduction operations
    std::cout << "--- Reduction Operations ---" << std::endl;
    RUN_TEST(test_sum_parity);
    RUN_TEST(test_mean_parity);
    RUN_TEST(test_max_parity);
    RUN_TEST(test_min_parity);
    RUN_TEST(test_argmax_parity);
    RUN_TEST(test_argmin_parity);

    // MatMul operations
    std::cout << "--- MatMul Operations ---" << std::endl;
    RUN_TEST(test_matmul_2d_parity);
    RUN_TEST(test_matmul_batched_parity);

    // Special operations
    std::cout << "--- Special Operations ---" << std::endl;
    RUN_TEST(test_where_parity);
    RUN_TEST(test_softmax_parity);
    RUN_TEST(test_gather_parity);
    RUN_TEST(test_index_select_parity);

    // Broadcasting
    std::cout << "--- Broadcasting ---" << std::endl;
    RUN_TEST(test_broadcast_add_parity);
    RUN_TEST(test_broadcast_multiply_parity);

    // Print summary
    std::cout << "=== Results ===" << std::endl;
    std::cout << "Passed: " << tests_passed << "/" << tests_run << std::endl;

    return (tests_passed == tests_run) ? 0 : 1;
}
