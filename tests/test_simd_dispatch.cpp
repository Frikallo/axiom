// Test for runtime SIMD dispatch functionality (Highway backend)
#include "../src/backends/cpu/simd/simd_dispatch.hpp"
#include "axiom_test_utils.hpp"
#include <cmath>
#include <cstdio>
#include <vector>

using namespace axiom::backends::cpu::simd;

bool nearly_equal(float a, float b, float eps = 1e-5f) {
    return std::abs(a - b) < eps;
}

bool nearly_equal(double a, double b, double eps = 1e-10) {
    return std::abs(a - b) < eps;
}

TEST(SimdDispatch, RuntimeArchDetection) {
    auto info = get_simd_info();
    printf("[detected: %s] ", info.arch_name);

    // Should return something valid
    ASSERT_TRUE(info.arch_name != nullptr);
    ASSERT_TRUE(info.arch_name[0] != '\0');
    ASSERT_TRUE(info.alignment > 0);
    ASSERT_TRUE(info.float32_width > 0);
}

TEST(SimdDispatch, BinaryAddFloat) {
    const size_t n = 1024;
    std::vector<float> a(n), b(n), result(n);

    for (size_t i = 0; i < n; ++i) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(i * 2);
    }

    dispatch_binary_add(a.data(), b.data(), result.data(), n);

    for (size_t i = 0; i < n; ++i) {
        ASSERT_TRUE(nearly_equal(result[i], a[i] + b[i]));
    }
}

TEST(SimdDispatch, BinaryMulDouble) {
    const size_t n = 512;
    std::vector<double> a(n), b(n), result(n);

    for (size_t i = 0; i < n; ++i) {
        a[i] = static_cast<double>(i) * 0.1;
        b[i] = static_cast<double>(i) * 0.5;
    }

    dispatch_binary_mul(a.data(), b.data(), result.data(), n);

    for (size_t i = 0; i < n; ++i) {
        ASSERT_TRUE(nearly_equal(result[i], a[i] * b[i]));
    }
}

TEST(SimdDispatch, ReduceSumFloat) {
    const size_t n = 1000;
    std::vector<float> data(n);

    float expected = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        data[i] = static_cast<float>(i + 1);
        expected += data[i];
    }

    float result = dispatch_reduce_sum(data.data(), n);
    ASSERT_TRUE(
        nearly_equal(result, expected, 1e-3f)); // Allow some FP tolerance
}

TEST(SimdDispatch, ReduceMaxFloat) {
    const size_t n = 1000;
    std::vector<float> data(n);

    for (size_t i = 0; i < n; ++i) {
        data[i] = static_cast<float>(i);
    }
    data[500] = 9999.0f; // Max value

    float result = dispatch_reduce_max(data.data(), n);
    ASSERT_TRUE(nearly_equal(result, 9999.0f));
}

TEST(SimdDispatch, UnaryExpFloat) {
    const size_t n = 256;
    std::vector<float> input(n), output(n);

    for (size_t i = 0; i < n; ++i) {
        input[i] =
            static_cast<float>(i) * 0.01f; // Small values to avoid overflow
    }

    dispatch_unary_exp(input.data(), output.data(), n);

    for (size_t i = 0; i < n; ++i) {
        ASSERT_TRUE(nearly_equal(output[i], std::exp(input[i]), 1e-4f));
    }
}

TEST(SimdDispatch, ActivationReluFloat) {
    const size_t n = 256;
    std::vector<float> input(n), output(n);

    for (size_t i = 0; i < n; ++i) {
        input[i] = static_cast<float>(i) - 128.0f; // Range: [-128, 127]
    }

    dispatch_activation_relu(input.data(), output.data(), n);

    for (size_t i = 0; i < n; ++i) {
        float expected = input[i] > 0 ? input[i] : 0;
        ASSERT_TRUE(nearly_equal(output[i], expected));
    }
}

TEST(SimdDispatch, SmallArrays) {
    // Test with arrays smaller than SIMD width to ensure scalar fallback works
    std::vector<float> a = {1, 2, 3};
    std::vector<float> b = {4, 5, 6};
    std::vector<float> result(3);

    dispatch_binary_add(a.data(), b.data(), result.data(), 3);

    ASSERT_TRUE(nearly_equal(result[0], 5.0f));
    ASSERT_TRUE(nearly_equal(result[1], 7.0f));
    ASSERT_TRUE(nearly_equal(result[2], 9.0f));
}

TEST(SimdDispatch, UnalignedAccess) {
    // Test with odd-sized arrays that won't be aligned to SIMD boundaries
    const size_t n = 17; // Prime number, won't align to any SIMD width
    std::vector<float> a(n), b(n), result(n);

    for (size_t i = 0; i < n; ++i) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(i * 3);
    }

    dispatch_binary_add(a.data(), b.data(), result.data(), n);

    for (size_t i = 0; i < n; ++i) {
        ASSERT_TRUE(nearly_equal(result[i], a[i] + b[i]));
    }
}
