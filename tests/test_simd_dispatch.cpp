// Test for runtime SIMD dispatch functionality
#include "../src/backends/cpu/simd/simd_arch_list.hpp"
#include "../src/backends/cpu/simd/simd_dispatch.hpp"
#include <cmath>
#include <cstdio>
#include <vector>

using namespace axiom::backends::cpu::simd;

#define ASSERT(cond)                                                           \
    do {                                                                       \
        if (!(cond)) {                                                         \
            printf("FAIL: %s at %s:%d\n", #cond, __FILE__, __LINE__);          \
            return 1;                                                          \
        }                                                                      \
    } while (0)

#define RUN_TEST(test_func)                                                    \
    do {                                                                       \
        printf("Running %s... ", #test_func);                                  \
        fflush(stdout);                                                        \
        if (test_func() == 0) {                                                \
            printf("OK\n");                                                    \
        } else {                                                               \
            printf("FAILED\n");                                                \
            return 1;                                                          \
        }                                                                      \
    } while (0)

bool nearly_equal(float a, float b, float eps = 1e-5f) {
    return std::abs(a - b) < eps;
}

bool nearly_equal(double a, double b, double eps = 1e-10) {
    return std::abs(a - b) < eps;
}

int test_runtime_arch_detection() {
    const char *arch = get_runtime_arch_name();
    printf("[detected: %s] ", arch);

    // Should return something valid
    ASSERT(arch != nullptr);
    ASSERT(arch[0] != '\0');

    return 0;
}

int test_binary_add_float() {
    const size_t n = 1024;
    std::vector<float> a(n), b(n), result(n);

    for (size_t i = 0; i < n; ++i) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(i * 2);
    }

    dispatch_binary_add(a.data(), b.data(), result.data(), n);

    for (size_t i = 0; i < n; ++i) {
        ASSERT(nearly_equal(result[i], a[i] + b[i]));
    }

    return 0;
}

int test_binary_mul_double() {
    const size_t n = 512;
    std::vector<double> a(n), b(n), result(n);

    for (size_t i = 0; i < n; ++i) {
        a[i] = static_cast<double>(i) * 0.1;
        b[i] = static_cast<double>(i) * 0.5;
    }

    dispatch_binary_mul(a.data(), b.data(), result.data(), n);

    for (size_t i = 0; i < n; ++i) {
        ASSERT(nearly_equal(result[i], a[i] * b[i]));
    }

    return 0;
}

int test_reduce_sum_float() {
    const size_t n = 1000;
    std::vector<float> data(n);

    float expected = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        data[i] = static_cast<float>(i + 1);
        expected += data[i];
    }

    float result = dispatch_reduce_sum(data.data(), n);
    ASSERT(nearly_equal(result, expected, 1e-3f)); // Allow some FP tolerance

    return 0;
}

int test_reduce_max_float() {
    const size_t n = 1000;
    std::vector<float> data(n);

    for (size_t i = 0; i < n; ++i) {
        data[i] = static_cast<float>(i);
    }
    data[500] = 9999.0f; // Max value

    float result = dispatch_reduce_max(data.data(), n);
    ASSERT(nearly_equal(result, 9999.0f));

    return 0;
}

int test_unary_exp_float() {
    const size_t n = 256;
    std::vector<float> input(n), output(n);

    for (size_t i = 0; i < n; ++i) {
        input[i] =
            static_cast<float>(i) * 0.01f; // Small values to avoid overflow
    }

    dispatch_unary_exp(input.data(), output.data(), n);

    for (size_t i = 0; i < n; ++i) {
        ASSERT(nearly_equal(output[i], std::exp(input[i]), 1e-4f));
    }

    return 0;
}

int test_activation_relu_float() {
    const size_t n = 256;
    std::vector<float> input(n), output(n);

    for (size_t i = 0; i < n; ++i) {
        input[i] = static_cast<float>(i) - 128.0f; // Range: [-128, 127]
    }

    dispatch_activation_relu(input.data(), output.data(), n);

    for (size_t i = 0; i < n; ++i) {
        float expected = input[i] > 0 ? input[i] : 0;
        ASSERT(nearly_equal(output[i], expected));
    }

    return 0;
}

int test_small_arrays() {
    // Test with arrays smaller than SIMD width to ensure scalar fallback works
    std::vector<float> a = {1, 2, 3};
    std::vector<float> b = {4, 5, 6};
    std::vector<float> result(3);

    dispatch_binary_add(a.data(), b.data(), result.data(), 3);

    ASSERT(nearly_equal(result[0], 5.0f));
    ASSERT(nearly_equal(result[1], 7.0f));
    ASSERT(nearly_equal(result[2], 9.0f));

    return 0;
}

int test_unaligned_access() {
    // Test with odd-sized arrays that won't be aligned to SIMD boundaries
    const size_t n = 17; // Prime number, won't align to any SIMD width
    std::vector<float> a(n), b(n), result(n);

    for (size_t i = 0; i < n; ++i) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(i * 3);
    }

    dispatch_binary_add(a.data(), b.data(), result.data(), n);

    for (size_t i = 0; i < n; ++i) {
        ASSERT(nearly_equal(result[i], a[i] + b[i]));
    }

    return 0;
}

int main(int argc, char **argv) {
    (void)argc;
    (void)argv;

    printf("=== SIMD Runtime Dispatch Tests ===\n");
    printf("Build type: ");
#ifdef AXIOM_SIMD_MULTI_ARCH
    printf("Multi-arch runtime dispatch\n");
#else
    printf("Compile-time architecture\n");
#endif
    printf("\n");

    RUN_TEST(test_runtime_arch_detection);
    RUN_TEST(test_binary_add_float);
    RUN_TEST(test_binary_mul_double);
    RUN_TEST(test_reduce_sum_float);
    RUN_TEST(test_reduce_max_float);
    RUN_TEST(test_unary_exp_float);
    RUN_TEST(test_activation_relu_float);
    RUN_TEST(test_small_arrays);
    RUN_TEST(test_unaligned_access);

    printf("\n=== All SIMD dispatch tests passed! ===\n");
    return 0;
}
