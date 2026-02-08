#include "axiom_test_utils.hpp"

#include "axiom/parallel.hpp"

using namespace axiom;

TEST(Parallel, ParallelApi) {
    // Test that we can get/set thread counts
    size_t orig_threads = parallel::get_num_threads();

#ifdef AXIOM_USE_OPENMP
    ASSERT_TRUE(orig_threads >= 1);

    // Test should_parallelize
    ASSERT_TRUE(!parallel::should_parallelize(100));   // Too small
    ASSERT_TRUE(!parallel::should_parallelize(1000));  // Still too small
    ASSERT_TRUE(parallel::should_parallelize(100000)); // Large enough

    // Test ThreadGuard RAII
    {
        parallel::ThreadGuard guard(2);
        ASSERT_TRUE(parallel::get_num_threads() == 2);
    }
    ASSERT_TRUE(parallel::get_num_threads() == orig_threads);
#else
    ASSERT_TRUE(orig_threads == 1);

    // Without OpenMP, should_parallelize always returns false
    ASSERT_TRUE(!parallel::should_parallelize(100));
    ASSERT_TRUE(!parallel::should_parallelize(1000000));
#endif
}

TEST(Parallel, ParallelBinaryOps) {
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
        EXPECT_NEAR(expected, actual, 1e-5f) << "Mismatch at " << i;
    }
}

TEST(Parallel, ParallelUnaryOps) {
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
        EXPECT_NEAR(expected, actual,
                    1e-5f * std::max(1.0f, std::abs(expected)))
            << "Mismatch at " << i;
    }
}

TEST(Parallel, ParallelBroadcast) {
    // Create tensors that need broadcasting
    auto a = Tensor::randn({1000, 1000}, DType::Float32);
    auto b = Tensor::randn({1, 1000}, DType::Float32);

    // Perform operation with broadcasting
    auto c = ops::add(a, b);

    ASSERT_TRUE(c.shape() == Shape({1000, 1000}));

    // Verify correctness
    const float *a_data = a.typed_data<float>();
    const float *b_data = b.typed_data<float>();
    const float *c_data = c.typed_data<float>();

    for (size_t i = 0; i < 10; ++i) {
        for (size_t j = 0; j < 10; ++j) {
            float expected = a_data[i * 1000 + j] + b_data[j];
            float actual = c_data[i * 1000 + j];
            EXPECT_NEAR(expected, actual, 1e-5f)
                << "Mismatch at [" << i << "," << j << "]";
        }
    }
}

TEST(Parallel, ParallelMatmul) {
    // Create batch of matrices
    size_t batch = 8;
    size_t m = 128, n = 128, k = 128;

    auto a = Tensor::randn({batch, m, k}, DType::Float32);
    auto b = Tensor::randn({batch, k, n}, DType::Float32);

    // Perform batch matmul
    auto c = ops::matmul(a, b);

    ASSERT_TRUE(c.shape() == Shape({batch, m, n}));

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
    EXPECT_NEAR(expected, actual, 1e-3f * std::max(1.0f, std::abs(expected)))
        << "Matmul spot check mismatch";
}
