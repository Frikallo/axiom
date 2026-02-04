// Tests for expanded random module (Phase 2)

#include <axiom/axiom.hpp>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>

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

void test_rand() {
    auto t = axiom::Tensor::rand({3, 4});
    ASSERT(t.shape() == axiom::Shape({3, 4}), "Shape mismatch");
    ASSERT(t.dtype() == axiom::DType::Float32, "DType should be Float32");

    // Check values are in [0, 1)
    const float *data = t.typed_data<float>();
    for (size_t i = 0; i < t.size(); ++i) {
        ASSERT(data[i] >= 0.0f && data[i] < 1.0f,
               "rand values should be in [0, 1)");
    }
}

void test_rand_float64() {
    auto t = axiom::Tensor::rand({5, 5}, axiom::DType::Float64);
    ASSERT(t.dtype() == axiom::DType::Float64, "DType should be Float64");

    const double *data = t.typed_data<double>();
    for (size_t i = 0; i < t.size(); ++i) {
        ASSERT(data[i] >= 0.0 && data[i] < 1.0,
               "rand float64 values should be in [0, 1)");
    }
}

void test_uniform() {
    auto t = axiom::Tensor::uniform(5.0, 10.0, {100});
    const float *data = t.typed_data<float>();

    for (size_t i = 0; i < t.size(); ++i) {
        ASSERT(data[i] >= 5.0f && data[i] < 10.0f,
               "uniform values should be in [5, 10)");
    }
}

void test_uniform_negative() {
    auto t = axiom::Tensor::uniform(-5.0, 5.0, {100});
    const float *data = t.typed_data<float>();

    for (size_t i = 0; i < t.size(); ++i) {
        ASSERT(data[i] >= -5.0f && data[i] < 5.0f,
               "uniform values should be in [-5, 5)");
    }
}

void test_randint() {
    auto t = axiom::Tensor::randint(0, 10, {100}, axiom::DType::Int64);
    const int64_t *data = t.typed_data<int64_t>();

    for (size_t i = 0; i < t.size(); ++i) {
        ASSERT(data[i] >= 0 && data[i] < 10,
               "randint values should be in [0, 10)");
    }
}

void test_randint_int32() {
    auto t = axiom::Tensor::randint(10, 20, {50}, axiom::DType::Int32);
    ASSERT(t.dtype() == axiom::DType::Int32, "DType should be Int32");

    const int32_t *data = t.typed_data<int32_t>();
    for (size_t i = 0; i < t.size(); ++i) {
        ASSERT(data[i] >= 10 && data[i] < 20,
               "randint int32 values should be in [10, 20)");
    }
}

void test_rand_like() {
    auto proto = axiom::Tensor::zeros({3, 4}, axiom::DType::Float64);
    auto t = axiom::Tensor::rand_like(proto);

    ASSERT(t.shape() == proto.shape(), "Shape should match prototype");
    ASSERT(t.dtype() == proto.dtype(), "DType should match prototype");

    const double *data = t.typed_data<double>();
    for (size_t i = 0; i < t.size(); ++i) {
        ASSERT(data[i] >= 0.0 && data[i] < 1.0,
               "rand_like values should be in [0, 1)");
    }
}

void test_randn_like() {
    auto proto = axiom::Tensor::zeros({5, 5});
    auto t = axiom::Tensor::randn_like(proto);

    ASSERT(t.shape() == proto.shape(), "Shape should match prototype");
    ASSERT(t.dtype() == proto.dtype(), "DType should match prototype");
}

void test_randint_like() {
    auto proto = axiom::Tensor::zeros({10}, axiom::DType::Int32);
    auto t = axiom::Tensor::randint_like(proto, 0, 100);

    ASSERT(t.shape() == proto.shape(), "Shape should match prototype");
    ASSERT(t.dtype() == proto.dtype(), "DType should match prototype");

    const int32_t *data = t.typed_data<int32_t>();
    for (size_t i = 0; i < t.size(); ++i) {
        ASSERT(data[i] >= 0 && data[i] < 100,
               "randint_like values should be in [0, 100)");
    }
}

int main() {
    axiom::ops::OperationRegistry::initialize_builtin_operations();

    std::cout << "========================================" << std::endl;
    std::cout << "   Extended Random Module Tests" << std::endl;
    std::cout << "========================================\n" << std::endl;

    RUN_TEST(test_rand);
    RUN_TEST(test_rand_float64);
    RUN_TEST(test_uniform);
    RUN_TEST(test_uniform_negative);
    RUN_TEST(test_randint);
    RUN_TEST(test_randint_int32);
    RUN_TEST(test_rand_like);
    RUN_TEST(test_randn_like);
    RUN_TEST(test_randint_like);

    std::cout << "\n========================================" << std::endl;
    std::cout << "Test Summary: " << tests_passed << " / " << tests_run
              << " tests passed." << std::endl;
    std::cout << "========================================" << std::endl;

    return (tests_passed == tests_run) ? 0 : 1;
}
