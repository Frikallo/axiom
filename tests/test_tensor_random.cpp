#include <axiom/axiom.hpp>
#include <cmath>
#include <functional>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

// Test harness
static int tests_run = 0;
static int tests_passed = 0;

#define RUN_TEST(test_func) run_test([&]() { test_func(); }, #test_func)

void run_test(const std::function<void()> &test_func,
              const std::string &test_name) {
    tests_run++;
    std::cout << "--- Running: " << test_name << " ---" << std::endl;
    try {
        test_func();
        std::cout << "--- PASSED: " << test_name << " ---" << std::endl;
        tests_passed++;
    } catch (const std::exception &e) {
        std::cerr << "--- FAILED: " << test_name << " ---" << std::endl;
        std::cerr << "    Error: " << e.what() << std::endl;
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

// Test randn creates tensors with expected shape
void test_randn_shape() {
    auto t = axiom::Tensor::randn({3, 4});
    ASSERT(t.shape() == axiom::Shape({3, 4}), "Shape mismatch");
    ASSERT(t.dtype() == axiom::DType::Float32, "DType should be Float32");
    ASSERT(t.size() == 12, "Size mismatch");
}

// Test randn creates different values
void test_randn_randomness() {
    auto t1 = axiom::Tensor::randn({100});
    auto t2 = axiom::Tensor::randn({100});

    // Check that t1 and t2 are not identical
    const float *data1 = t1.typed_data<float>();
    const float *data2 = t2.typed_data<float>();

    int differences = 0;
    for (size_t i = 0; i < 100; ++i) {
        if (std::abs(data1[i] - data2[i]) > 1e-6) {
            differences++;
        }
    }

    ASSERT(differences > 90, "Random tensors should be different");
}

// Test manual_seed produces reproducible results
void test_manual_seed_reproducibility() {
    axiom::Tensor::manual_seed(42);
    auto t1 = axiom::Tensor::randn({10});

    axiom::Tensor::manual_seed(42);
    auto t2 = axiom::Tensor::randn({10});

    // Check that t1 and t2 are identical
    const float *data1 = t1.typed_data<float>();
    const float *data2 = t2.typed_data<float>();

    for (size_t i = 0; i < 10; ++i) {
        ASSERT(std::abs(data1[i] - data2[i]) < 1e-6,
               "Seeded random should be reproducible");
    }
}

// Test manual_seed with different seeds produces different results
void test_manual_seed_different_seeds() {
    axiom::Tensor::manual_seed(42);
    auto t1 = axiom::Tensor::randn({100});

    axiom::Tensor::manual_seed(123);
    auto t2 = axiom::Tensor::randn({100});

    // Check that t1 and t2 are different
    const float *data1 = t1.typed_data<float>();
    const float *data2 = t2.typed_data<float>();

    int differences = 0;
    for (size_t i = 0; i < 100; ++i) {
        if (std::abs(data1[i] - data2[i]) > 1e-6) {
            differences++;
        }
    }

    ASSERT(differences > 90,
           "Different seeds should produce different results");
}

// Test get_seed returns the current seed
void test_get_seed() {
    uint64_t seed = 12345;
    axiom::manual_seed(seed);
    uint64_t retrieved_seed = axiom::get_seed();
    ASSERT(retrieved_seed == seed, "get_seed should return the set seed");
}

// Test randn statistical properties (rough check)
void test_randn_statistics() {
    axiom::Tensor::manual_seed(42);
    auto t = axiom::Tensor::randn({10000});

    // Calculate mean and stddev
    const float *data = t.typed_data<float>();
    double sum = 0.0;
    for (size_t i = 0; i < 10000; ++i) {
        sum += data[i];
    }
    double mean = sum / 10000.0;

    double variance = 0.0;
    for (size_t i = 0; i < 10000; ++i) {
        double diff = data[i] - mean;
        variance += diff * diff;
    }
    variance /= 10000.0;
    double stddev = std::sqrt(variance);

    // Check that mean is close to 0 and stddev is close to 1
    ASSERT(std::abs(mean) < 0.05, "Mean should be close to 0");
    ASSERT(std::abs(stddev - 1.0) < 0.05, "Stddev should be close to 1");
}

// Test randn on GPU if available
void test_randn_gpu() {
    if (!axiom::system::should_run_gpu_tests()) {
        std::cout << "Skipping GPU test (Metal not available)" << std::endl;
        return;
    }

    auto t =
        axiom::Tensor::randn({3, 4}, axiom::DType::Float32, axiom::Device::GPU);
    ASSERT(t.device() == axiom::Device::GPU, "Should be on GPU");
    ASSERT(t.shape() == axiom::Shape({3, 4}), "Shape mismatch");
}

// Test that free function manual_seed works
void test_free_function_manual_seed() {
    axiom::manual_seed(999);
    auto t1 = axiom::Tensor::randn({10});

    axiom::manual_seed(999);
    auto t2 = axiom::Tensor::randn({10});

    const float *data1 = t1.typed_data<float>();
    const float *data2 = t2.typed_data<float>();

    for (size_t i = 0; i < 10; ++i) {
        ASSERT(std::abs(data1[i] - data2[i]) < 1e-6,
               "Free function manual_seed should work");
    }
}

int main() {
    axiom::ops::OperationRegistry::initialize_builtin_operations();

    std::cout << "========================================" << std::endl;
    std::cout << "   Random Number Generation Test Suite" << std::endl;
    std::cout << "========================================\n" << std::endl;

    RUN_TEST(test_randn_shape);
    RUN_TEST(test_randn_randomness);
    RUN_TEST(test_manual_seed_reproducibility);
    RUN_TEST(test_manual_seed_different_seeds);
    RUN_TEST(test_get_seed);
    RUN_TEST(test_randn_statistics);
    RUN_TEST(test_randn_gpu);
    RUN_TEST(test_free_function_manual_seed);

    std::cout << "\n========================================" << std::endl;
    std::cout << "Test Suite Summary:" << std::endl;
    std::cout << "    " << tests_passed << " / " << tests_run
              << " tests passed." << std::endl;
    std::cout << "========================================" << std::endl;

    return (tests_passed == tests_run) ? 0 : 1;
}
