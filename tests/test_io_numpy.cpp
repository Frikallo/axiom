#include <cmath>
#include <filesystem>
#include <iostream>

#include "axiom/axiom.hpp"

using namespace axiom;

int tests_run = 0;
int tests_passed = 0;

#define TEST(name)                                                             \
    void name();                                                               \
    struct name##_registrar {                                                  \
        name##_registrar() {                                                   \
            std::cout << "Running " #name "..." << std::endl;                  \
            tests_run++;                                                       \
            try {                                                              \
                name();                                                        \
                tests_passed++;                                                \
                std::cout << "  PASSED" << std::endl;                          \
            } catch (const std::exception &e) {                                \
                std::cout << "  FAILED: " << e.what() << std::endl;            \
            }                                                                  \
        }                                                                      \
    } name##_instance;                                                         \
    void name()

#define ASSERT(cond)                                                           \
    if (!(cond)) {                                                             \
        throw std::runtime_error("Assertion failed: " #cond);                  \
    }

#define ASSERT_EQ(a, b)                                                        \
    if ((a) != (b)) {                                                          \
        throw std::runtime_error("Assertion failed: " #a " == " #b);           \
    }

#define ASSERT_NEAR(a, b, tol)                                                 \
    if (std::abs((a) - (b)) > (tol)) {                                         \
        throw std::runtime_error("Assertion failed: " #a " near " #b);         \
    }

namespace fs = std::filesystem;

// Path to fixtures directory
const std::string FIXTURES_DIR = "fixtures/";

std::string fixture_path(const std::string &name) {
    return FIXTURES_DIR + name;
}

// ============================================================================
// Format Detection Tests
// ============================================================================

TEST(test_is_npy_file) {
    ASSERT(io::numpy::is_npy_file(fixture_path("float32_2d.npy")));
    ASSERT(io::detect_format(fixture_path("float32_2d.npy")) ==
           io::FileFormat::NumPy);
}

TEST(test_format_name) {
    ASSERT(io::format_name(io::FileFormat::NumPy) == "NumPy");
    ASSERT(io::format_name(io::FileFormat::Axiom) == "Axiom FlatBuffers");
    ASSERT(io::format_name(io::FileFormat::Unknown) == "Unknown");
}

// ============================================================================
// Float Type Tests
// ============================================================================

TEST(test_load_float32) {
    auto t = io::numpy::load(fixture_path("float32_2d.npy"));

    ASSERT(t.dtype() == DType::Float32);
    ASSERT(t.shape() == Shape({2, 3}));

    // Check values: [[1, 2, 3], [4, 5, 6]]
    ASSERT_NEAR(t.item<float>({0, 0}), 1.0f, 1e-6);
    ASSERT_NEAR(t.item<float>({0, 2}), 3.0f, 1e-6);
    ASSERT_NEAR(t.item<float>({1, 0}), 4.0f, 1e-6);
    ASSERT_NEAR(t.item<float>({1, 2}), 6.0f, 1e-6);
}

TEST(test_load_float64) {
    auto t = io::numpy::load(fixture_path("float64_2d.npy"));

    ASSERT(t.dtype() == DType::Float64);
    ASSERT(t.shape() == Shape({3, 2}));

    ASSERT_NEAR(t.item<double>({0, 0}), 1.0, 1e-12);
    ASSERT_NEAR(t.item<double>({2, 1}), 6.0, 1e-12);
}

TEST(test_load_float16) {
    auto t = io::numpy::load(fixture_path("float16_2d.npy"));

    ASSERT(t.dtype() == DType::Float16);
    ASSERT(t.shape() == Shape({2, 2}));

    // Convert to float32 for comparison
    auto tf = t.to_float();
    ASSERT_NEAR(tf.item<float>({0, 0}), 0.5f, 1e-2);
    ASSERT_NEAR(tf.item<float>({1, 1}), 3.5f, 1e-2);
}

// ============================================================================
// Integer Type Tests
// ============================================================================

TEST(test_load_int8) {
    auto t = io::numpy::load(fixture_path("int8_1d.npy"));

    ASSERT(t.dtype() == DType::Int8);
    ASSERT(t.shape() == Shape({5}));

    ASSERT_EQ(t.item<int8_t>({0}), -128);
    ASSERT_EQ(t.item<int8_t>({2}), 0);
    ASSERT_EQ(t.item<int8_t>({4}), 127);
}

TEST(test_load_int16) {
    auto t = io::numpy::load(fixture_path("int16_1d.npy"));

    ASSERT(t.dtype() == DType::Int16);
    ASSERT(t.shape() == Shape({3}));

    ASSERT_EQ(t.item<int16_t>({0}), -32768);
    ASSERT_EQ(t.item<int16_t>({2}), 32767);
}

TEST(test_load_int32) {
    auto t = io::numpy::load(fixture_path("int32_2d.npy"));

    ASSERT(t.dtype() == DType::Int32);
    ASSERT(t.shape() == Shape({3, 4}));

    // Values are arange(12).reshape(3,4): [[0,1,2,3],[4,5,6,7],[8,9,10,11]]
    ASSERT_EQ(t.item<int32_t>({0, 0}), 0);
    ASSERT_EQ(t.item<int32_t>({1, 1}), 5);
    ASSERT_EQ(t.item<int32_t>({2, 3}), 11);
}

TEST(test_load_int64) {
    auto t = io::numpy::load(fixture_path("int64_2d.npy"));

    ASSERT(t.dtype() == DType::Int64);
    ASSERT(t.shape() == Shape({2, 3}));

    ASSERT_EQ(t.item<int64_t>({0, 0}), 0);
    ASSERT_EQ(t.item<int64_t>({1, 2}), 5);
}

// ============================================================================
// Unsigned Integer Type Tests
// ============================================================================

TEST(test_load_uint8) {
    auto t = io::numpy::load(fixture_path("uint8_1d.npy"));

    ASSERT(t.dtype() == DType::UInt8);
    ASSERT(t.shape() == Shape({3}));

    ASSERT_EQ(t.item<uint8_t>({0}), 0);
    ASSERT_EQ(t.item<uint8_t>({1}), 128);
    ASSERT_EQ(t.item<uint8_t>({2}), 255);
}

TEST(test_load_uint16) {
    auto t = io::numpy::load(fixture_path("uint16_1d.npy"));

    ASSERT(t.dtype() == DType::UInt16);
    ASSERT(t.shape() == Shape({3}));

    ASSERT_EQ(t.item<uint16_t>({0}), 0);
    ASSERT_EQ(t.item<uint16_t>({2}), 65535);
}

TEST(test_load_uint32) {
    auto t = io::numpy::load(fixture_path("uint32_1d.npy"));

    ASSERT(t.dtype() == DType::UInt32);
    ASSERT(t.shape() == Shape({3}));

    ASSERT_EQ(t.item<uint32_t>({0}), 0u);
    ASSERT_EQ(t.item<uint32_t>({2}), 4294967295u);
}

TEST(test_load_uint64) {
    auto t = io::numpy::load(fixture_path("uint64_1d.npy"));

    ASSERT(t.dtype() == DType::UInt64);
    ASSERT(t.shape() == Shape({2}));

    ASSERT_EQ(t.item<uint64_t>({0}), 0ull);
    ASSERT_EQ(t.item<uint64_t>({1}), 1000000000000ull);
}

// ============================================================================
// Boolean Test
// ============================================================================

TEST(test_load_bool) {
    auto t = io::numpy::load(fixture_path("bool_2d.npy"));

    ASSERT(t.dtype() == DType::Bool);
    ASSERT(t.shape() == Shape({2, 2}));

    // [[True, False], [False, True]]
    ASSERT(t.item<bool>({0, 0}) == true);
    ASSERT(t.item<bool>({0, 1}) == false);
    ASSERT(t.item<bool>({1, 0}) == false);
    ASSERT(t.item<bool>({1, 1}) == true);
}

// ============================================================================
// Complex Type Tests
// ============================================================================

TEST(test_load_complex64) {
    auto t = io::numpy::load(fixture_path("complex64_1d.npy"));

    ASSERT(t.dtype() == DType::Complex64);
    ASSERT(t.shape() == Shape({3}));

    auto val0 = t.item<complex64_t>({0});
    auto val1 = t.item<complex64_t>({1});

    ASSERT_NEAR(val0.real(), 1.0f, 1e-6);
    ASSERT_NEAR(val0.imag(), 2.0f, 1e-6);
    ASSERT_NEAR(val1.real(), 3.0f, 1e-6);
    ASSERT_NEAR(val1.imag(), 4.0f, 1e-6);
}

TEST(test_load_complex128) {
    auto t = io::numpy::load(fixture_path("complex128_1d.npy"));

    ASSERT(t.dtype() == DType::Complex128);
    ASSERT(t.shape() == Shape({2}));

    auto val0 = t.item<complex128_t>({0});
    ASSERT_NEAR(val0.real(), 1.5, 1e-12);
    ASSERT_NEAR(val0.imag(), 2.5, 1e-12);
}

// ============================================================================
// Memory Order Tests
// ============================================================================

TEST(test_load_c_order) {
    auto t = io::numpy::load(fixture_path("c_order.npy"));

    ASSERT(t.memory_order() == MemoryOrder::RowMajor);
    ASSERT(t.is_c_contiguous());
    ASSERT(t.shape() == Shape({2, 3}));
}

TEST(test_load_f_order) {
    auto t = io::numpy::load(fixture_path("f_order.npy"));

    ASSERT(t.memory_order() == MemoryOrder::ColMajor);
    ASSERT(t.is_f_contiguous());
    ASSERT(t.shape() == Shape({2, 3}));

    // Values should still be correct
    ASSERT_NEAR(t.item<float>({0, 0}), 1.0f, 1e-6);
    ASSERT_NEAR(t.item<float>({1, 2}), 6.0f, 1e-6);
}

// ============================================================================
// Shape Tests
// ============================================================================

TEST(test_load_scalar) {
    auto t = io::numpy::load(fixture_path("scalar.npy"));

    ASSERT(t.ndim() == 0);
    ASSERT(t.shape().empty());
    ASSERT(t.size() == 1);
    ASSERT_NEAR(t.item<float>(), 42.0f, 1e-6);
}

TEST(test_load_1d) {
    auto t = io::numpy::load(fixture_path("1d.npy"));

    ASSERT(t.ndim() == 1);
    ASSERT(t.shape() == Shape({10}));

    // arange(10)
    for (int i = 0; i < 10; i++) {
        ASSERT_NEAR(t.item<float>({static_cast<size_t>(i)}),
                    static_cast<float>(i), 1e-6);
    }
}

TEST(test_load_3d) {
    auto t = io::numpy::load(fixture_path("3d.npy"));

    ASSERT(t.ndim() == 3);
    ASSERT(t.shape() == Shape({2, 3, 4}));
    ASSERT(t.size() == 24);
}

TEST(test_load_4d) {
    auto t = io::numpy::load(fixture_path("4d.npy"));

    ASSERT(t.ndim() == 4);
    ASSERT(t.shape() == Shape({2, 3, 4, 5}));
    ASSERT(t.size() == 120);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST(test_load_empty) {
    auto t = io::numpy::load(fixture_path("empty_1d.npy"));

    ASSERT(t.dtype() == DType::Float32);
    ASSERT(t.shape() == Shape({0}));
    ASSERT(t.size() == 0);
}

TEST(test_load_single_element) {
    auto t = io::numpy::load(fixture_path("single_element.npy"));

    ASSERT(t.shape() == Shape({1}));
    ASSERT(t.size() == 1);
    ASSERT_NEAR(t.item<float>({0}), 42.0f, 1e-6);
}

TEST(test_load_large) {
    auto t = io::numpy::load(fixture_path("large_1d.npy"));

    ASSERT(t.shape() == Shape({10000}));
    ASSERT(t.size() == 10000);

    // Check some values
    ASSERT_NEAR(t.item<float>({0}), 0.0f, 1e-6);
    ASSERT_NEAR(t.item<float>({9999}), 9999.0f, 1e-6);
}

// ============================================================================
// Special Values
// ============================================================================

TEST(test_load_special_values) {
    auto t = io::numpy::load(fixture_path("special_float32.npy"));

    ASSERT(t.dtype() == DType::Float32);
    ASSERT(t.shape() == Shape({5}));

    // [0.0, -0.0, inf, -inf, nan]
    ASSERT_NEAR(t.item<float>({0}), 0.0f, 1e-6);
    // -0.0 compares equal to 0.0
    ASSERT(std::isinf(t.item<float>({2})) && t.item<float>({2}) > 0);
    ASSERT(std::isinf(t.item<float>({3})) && t.item<float>({3}) < 0);
    ASSERT(std::isnan(t.item<float>({4})));
}

// ============================================================================
// Universal Load Test
// ============================================================================

TEST(test_universal_load_npy) {
    // Test that io::load() auto-detects NumPy format
    auto t = io::load(fixture_path("float32_2d.npy"));

    ASSERT(t.dtype() == DType::Float32);
    ASSERT(t.shape() == Shape({2, 3}));
}

TEST(test_load_archive_npy) {
    // Test that load_archive works with .npy files
    auto tensors = io::load_archive(fixture_path("float32_2d.npy"));

    ASSERT(tensors.size() == 1);
    // Key should be filename without .npy extension
    ASSERT(tensors.count("float32_2d") == 1);
    ASSERT(tensors["float32_2d"].shape() == Shape({2, 3}));
}

int main() {
    std::cout << "\n========================================\n";
    std::cout << "NumPy I/O Tests\n";
    std::cout << "========================================\n\n";

    // Tests are run automatically via static initialization

    std::cout << "\n========================================\n";
    std::cout << "Results: " << tests_passed << "/" << tests_run << " passed\n";
    std::cout << "========================================\n";

    return (tests_passed == tests_run) ? 0 : 1;
}
