// Test for parallel functionality
#include <chrono>
#include <iostream>

#include "axiom/axiom.hpp"
#include "axiom/parallel.hpp"

using namespace axiom;

#define ASSERT(cond)                                                           \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::cerr << "ASSERTION FAILED: " #cond << " at " << __FILE__      \
                      << ":" << __LINE__ << std::endl;                         \
            return false;                                                      \
        }                                                                      \
    } while (0)

bool test_parallel_api() {
    std::cout << "test_parallel_api... ";

    // Test that we can get/set thread counts
    size_t orig_threads = parallel::get_num_threads();

#ifdef AXIOM_USE_OPENMP
    ASSERT(orig_threads >= 1);
    std::cout << "(OpenMP enabled, " << orig_threads << " threads) ";

    // Test should_parallelize
    ASSERT(!parallel::should_parallelize(100));   // Too small
    ASSERT(!parallel::should_parallelize(1000));  // Still too small
    ASSERT(parallel::should_parallelize(100000)); // Large enough

    // Test ThreadGuard RAII
    {
        parallel::ThreadGuard guard(2);
        ASSERT(parallel::get_num_threads() == 2);
    }
    ASSERT(parallel::get_num_threads() == orig_threads);
#else
    ASSERT(orig_threads == 1);
    std::cout << "(OpenMP disabled) ";

    // Without OpenMP, should_parallelize always returns false
    ASSERT(!parallel::should_parallelize(100));
    ASSERT(!parallel::should_parallelize(1000000));
#endif

    std::cout << "PASSED" << std::endl;
    return true;
}

bool test_parallel_binary_ops() {
    std::cout << "test_parallel_binary_ops... ";

    // Create large tensors that will trigger parallelization
    size_t n = 1000000; // 1M elements
    auto a = Tensor::randn({n}, DType::Float32);
    auto b = Tensor::randn({n}, DType::Float32);

    // Perform binary operation
    auto c = ops::add(a, b);

    // Verify correctness (spot check)
    const float *a_data = a.typed_data<float>();
    const float *b_data = b.typed_data<float>();
    const float *c_data = c.typed_data<float>();

    for (size_t i = 0; i < 100; i += 10) {
        float expected = a_data[i] + b_data[i];
        float actual = c_data[i];
        if (std::abs(expected - actual) > 1e-5f) {
            std::cerr << "Mismatch at " << i << ": expected " << expected
                      << ", got " << actual << std::endl;
            return false;
        }
    }

    std::cout << "PASSED" << std::endl;
    return true;
}

bool test_parallel_unary_ops() {
    std::cout << "test_parallel_unary_ops... ";

    // Create large tensor
    size_t n = 1000000;
    auto a = Tensor::randn({n}, DType::Float32);

    // Perform unary operation
    auto c = ops::exp(a);

    // Verify correctness
    const float *a_data = a.typed_data<float>();
    const float *c_data = c.typed_data<float>();

    for (size_t i = 0; i < 100; i += 10) {
        float expected = std::exp(a_data[i]);
        float actual = c_data[i];
        if (std::abs(expected - actual) / std::max(1.0f, std::abs(expected)) >
            1e-5f) {
            std::cerr << "Mismatch at " << i << ": expected " << expected
                      << ", got " << actual << std::endl;
            return false;
        }
    }

    std::cout << "PASSED" << std::endl;
    return true;
}

bool test_parallel_broadcast() {
    std::cout << "test_parallel_broadcast... ";

    // Create tensors that need broadcasting
    auto a = Tensor::randn({1000, 1000}, DType::Float32);
    auto b = Tensor::randn({1, 1000}, DType::Float32);

    // Perform operation with broadcasting
    auto c = ops::add(a, b);

    ASSERT(c.shape() == Shape({1000, 1000}));

    // Verify correctness
    const float *a_data = a.typed_data<float>();
    const float *b_data = b.typed_data<float>();
    const float *c_data = c.typed_data<float>();

    for (size_t i = 0; i < 10; ++i) {
        for (size_t j = 0; j < 10; ++j) {
            float expected = a_data[i * 1000 + j] + b_data[j];
            float actual = c_data[i * 1000 + j];
            if (std::abs(expected - actual) > 1e-5f) {
                std::cerr << "Mismatch at [" << i << "," << j << "]: expected "
                          << expected << ", got " << actual << std::endl;
                return false;
            }
        }
    }

    std::cout << "PASSED" << std::endl;
    return true;
}

bool test_parallel_matmul() {
    std::cout << "test_parallel_matmul... ";

    // Create batch of matrices
    size_t batch = 8;
    size_t m = 128, n = 128, k = 128;

    auto a = Tensor::randn({batch, m, k}, DType::Float32);
    auto b = Tensor::randn({batch, k, n}, DType::Float32);

    // Perform batch matmul
    auto c = ops::matmul(a, b);

    ASSERT(c.shape() == Shape({batch, m, n}));

    // Verify one element as spot check
    // c[0, 0, 0] should be sum of a[0, 0, :] * b[0, :, 0]
    const float *a_data = a.typed_data<float>();
    const float *b_data = b.typed_data<float>();
    const float *c_data = c.typed_data<float>();

    float expected = 0.0f;
    for (size_t i = 0; i < k; ++i) {
        expected += a_data[i] * b_data[i * n];
    }
    float actual = c_data[0];
    if (std::abs(expected - actual) / std::max(1.0f, std::abs(expected)) >
        1e-3f) {
        std::cerr << "Mismatch: expected " << expected << ", got " << actual
                  << std::endl;
        return false;
    }

    std::cout << "PASSED" << std::endl;
    return true;
}

#define RUN_TEST(test)                                                         \
    do {                                                                       \
        if (!test()) {                                                         \
            std::cerr << "TEST FAILED: " #test << std::endl;                   \
            return 1;                                                          \
        }                                                                      \
    } while (0)

int main() {
    ops::OperationRegistry::initialize_builtin_operations();

    std::cout << "\n=== Parallel Tests ===\n";
    std::cout << "OpenMP enabled: ";
#ifdef AXIOM_USE_OPENMP
    std::cout << "YES" << std::endl;
    std::cout << "Max threads: " << parallel::get_num_threads() << std::endl;
#else
    std::cout << "NO" << std::endl;
#endif
    std::cout << std::endl;

    RUN_TEST(test_parallel_api);
    RUN_TEST(test_parallel_binary_ops);
    RUN_TEST(test_parallel_unary_ops);
    RUN_TEST(test_parallel_broadcast);
    RUN_TEST(test_parallel_matmul);

    std::cout << "\nAll parallel tests passed!" << std::endl;
    return 0;
}
