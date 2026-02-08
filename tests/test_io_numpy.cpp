#include "axiom_test_utils.hpp"
#include <cmath>
#include <filesystem>
#include <iostream>

using namespace axiom;

namespace fs = std::filesystem;

// Path to fixtures directory
const std::string FIXTURES_DIR = "fixtures/";

std::string fixture_path(const std::string &name) {
    return FIXTURES_DIR + name;
}

// ============================================================================
// Format Detection Tests
// ============================================================================

TEST(IoNumpy, IsNpyFile) {
    ASSERT_TRUE(io::numpy::is_npy_file(fixture_path("float32_2d.npy")));
    ASSERT_TRUE(io::detect_format(fixture_path("float32_2d.npy")) ==
                io::FileFormat::NumPy);
}

TEST(IoNumpy, FormatName) {
    ASSERT_TRUE(io::format_name(io::FileFormat::NumPy) == "NumPy");
    ASSERT_TRUE(io::format_name(io::FileFormat::Axiom) == "Axiom FlatBuffers");
    ASSERT_TRUE(io::format_name(io::FileFormat::Unknown) == "Unknown");
}

// ============================================================================
// Float Type Tests
// ============================================================================

TEST(IoNumpy, LoadFloat32) {
    auto t = io::numpy::load(fixture_path("float32_2d.npy"));

    ASSERT_TRUE(t.dtype() == DType::Float32);
    ASSERT_TRUE(t.shape() == Shape({2, 3}));

    // Check values: [[1, 2, 3], [4, 5, 6]]
    ASSERT_NEAR(t.item<float>({0, 0}), 1.0f, 1e-6);
    ASSERT_NEAR(t.item<float>({0, 2}), 3.0f, 1e-6);
    ASSERT_NEAR(t.item<float>({1, 0}), 4.0f, 1e-6);
    ASSERT_NEAR(t.item<float>({1, 2}), 6.0f, 1e-6);
}

TEST(IoNumpy, LoadFloat64) {
    auto t = io::numpy::load(fixture_path("float64_2d.npy"));

    ASSERT_TRUE(t.dtype() == DType::Float64);
    ASSERT_TRUE(t.shape() == Shape({3, 2}));

    ASSERT_NEAR(t.item<double>({0, 0}), 1.0, 1e-12);
    ASSERT_NEAR(t.item<double>({2, 1}), 6.0, 1e-12);
}

TEST(IoNumpy, LoadFloat16) {
    auto t = io::numpy::load(fixture_path("float16_2d.npy"));

    ASSERT_TRUE(t.dtype() == DType::Float16);
    ASSERT_TRUE(t.shape() == Shape({2, 2}));

    // Convert to float32 for comparison
    auto tf = t.to_float();
    ASSERT_NEAR(tf.item<float>({0, 0}), 0.5f, 1e-2);
    ASSERT_NEAR(tf.item<float>({1, 1}), 3.5f, 1e-2);
}

// ============================================================================
// Integer Type Tests
// ============================================================================

TEST(IoNumpy, LoadInt8) {
    auto t = io::numpy::load(fixture_path("int8_1d.npy"));

    ASSERT_TRUE(t.dtype() == DType::Int8);
    ASSERT_TRUE(t.shape() == Shape({5}));

    ASSERT_EQ(t.item<int8_t>({0}), -128);
    ASSERT_EQ(t.item<int8_t>({2}), 0);
    ASSERT_EQ(t.item<int8_t>({4}), 127);
}

TEST(IoNumpy, LoadInt16) {
    auto t = io::numpy::load(fixture_path("int16_1d.npy"));

    ASSERT_TRUE(t.dtype() == DType::Int16);
    ASSERT_TRUE(t.shape() == Shape({3}));

    ASSERT_EQ(t.item<int16_t>({0}), -32768);
    ASSERT_EQ(t.item<int16_t>({2}), 32767);
}

TEST(IoNumpy, LoadInt32) {
    auto t = io::numpy::load(fixture_path("int32_2d.npy"));

    ASSERT_TRUE(t.dtype() == DType::Int32);
    ASSERT_TRUE(t.shape() == Shape({3, 4}));

    // Values are arange(12).reshape(3,4): [[0,1,2,3],[4,5,6,7],[8,9,10,11]]
    ASSERT_EQ(t.item<int32_t>({0, 0}), 0);
    ASSERT_EQ(t.item<int32_t>({1, 1}), 5);
    ASSERT_EQ(t.item<int32_t>({2, 3}), 11);
}

TEST(IoNumpy, LoadInt64) {
    auto t = io::numpy::load(fixture_path("int64_2d.npy"));

    ASSERT_TRUE(t.dtype() == DType::Int64);
    ASSERT_TRUE(t.shape() == Shape({2, 3}));

    ASSERT_EQ(t.item<int64_t>({0, 0}), 0);
    ASSERT_EQ(t.item<int64_t>({1, 2}), 5);
}

// ============================================================================
// Unsigned Integer Type Tests
// ============================================================================

TEST(IoNumpy, LoadUint8) {
    auto t = io::numpy::load(fixture_path("uint8_1d.npy"));

    ASSERT_TRUE(t.dtype() == DType::UInt8);
    ASSERT_TRUE(t.shape() == Shape({3}));

    ASSERT_EQ(t.item<uint8_t>({0}), 0);
    ASSERT_EQ(t.item<uint8_t>({1}), 128);
    ASSERT_EQ(t.item<uint8_t>({2}), 255);
}

TEST(IoNumpy, LoadUint16) {
    auto t = io::numpy::load(fixture_path("uint16_1d.npy"));

    ASSERT_TRUE(t.dtype() == DType::UInt16);
    ASSERT_TRUE(t.shape() == Shape({3}));

    ASSERT_EQ(t.item<uint16_t>({0}), 0);
    ASSERT_EQ(t.item<uint16_t>({2}), 65535);
}

TEST(IoNumpy, LoadUint32) {
    auto t = io::numpy::load(fixture_path("uint32_1d.npy"));

    ASSERT_TRUE(t.dtype() == DType::UInt32);
    ASSERT_TRUE(t.shape() == Shape({3}));

    ASSERT_EQ(t.item<uint32_t>({0}), 0u);
    ASSERT_EQ(t.item<uint32_t>({2}), 4294967295u);
}

TEST(IoNumpy, LoadUint64) {
    auto t = io::numpy::load(fixture_path("uint64_1d.npy"));

    ASSERT_TRUE(t.dtype() == DType::UInt64);
    ASSERT_TRUE(t.shape() == Shape({2}));

    ASSERT_EQ(t.item<uint64_t>({0}), 0ull);
    ASSERT_EQ(t.item<uint64_t>({1}), 1000000000000ull);
}

// ============================================================================
// Boolean Test
// ============================================================================

TEST(IoNumpy, LoadBool) {
    auto t = io::numpy::load(fixture_path("bool_2d.npy"));

    ASSERT_TRUE(t.dtype() == DType::Bool);
    ASSERT_TRUE(t.shape() == Shape({2, 2}));

    // [[True, False], [False, True]]
    ASSERT_TRUE(t.item<bool>({0, 0}) == true);
    ASSERT_TRUE(t.item<bool>({0, 1}) == false);
    ASSERT_TRUE(t.item<bool>({1, 0}) == false);
    ASSERT_TRUE(t.item<bool>({1, 1}) == true);
}

// ============================================================================
// Complex Type Tests
// ============================================================================

TEST(IoNumpy, LoadComplex64) {
    auto t = io::numpy::load(fixture_path("complex64_1d.npy"));

    ASSERT_TRUE(t.dtype() == DType::Complex64);
    ASSERT_TRUE(t.shape() == Shape({3}));

    auto val0 = t.item<complex64_t>({0});
    auto val1 = t.item<complex64_t>({1});

    ASSERT_NEAR(val0.real(), 1.0f, 1e-6);
    ASSERT_NEAR(val0.imag(), 2.0f, 1e-6);
    ASSERT_NEAR(val1.real(), 3.0f, 1e-6);
    ASSERT_NEAR(val1.imag(), 4.0f, 1e-6);
}

TEST(IoNumpy, LoadComplex128) {
    auto t = io::numpy::load(fixture_path("complex128_1d.npy"));

    ASSERT_TRUE(t.dtype() == DType::Complex128);
    ASSERT_TRUE(t.shape() == Shape({2}));

    auto val0 = t.item<complex128_t>({0});
    ASSERT_NEAR(val0.real(), 1.5, 1e-12);
    ASSERT_NEAR(val0.imag(), 2.5, 1e-12);
}

// ============================================================================
// Memory Order Tests
// ============================================================================

TEST(IoNumpy, LoadCOrder) {
    auto t = io::numpy::load(fixture_path("c_order.npy"));

    ASSERT_TRUE(t.memory_order() == MemoryOrder::RowMajor);
    ASSERT_TRUE(t.is_c_contiguous());
    ASSERT_TRUE(t.shape() == Shape({2, 3}));
}

TEST(IoNumpy, LoadFOrder) {
    auto t = io::numpy::load(fixture_path("f_order.npy"));

    ASSERT_TRUE(t.memory_order() == MemoryOrder::ColMajor);
    ASSERT_TRUE(t.is_f_contiguous());
    ASSERT_TRUE(t.shape() == Shape({2, 3}));

    // Values should still be correct
    ASSERT_NEAR(t.item<float>({0, 0}), 1.0f, 1e-6);
    ASSERT_NEAR(t.item<float>({1, 2}), 6.0f, 1e-6);
}

// ============================================================================
// Shape Tests
// ============================================================================

TEST(IoNumpy, LoadScalar) {
    auto t = io::numpy::load(fixture_path("scalar.npy"));

    ASSERT_TRUE(t.ndim() == 0);
    ASSERT_TRUE(t.shape().empty());
    ASSERT_TRUE(t.size() == 1);
    ASSERT_NEAR(t.item<float>(), 42.0f, 1e-6);
}

TEST(IoNumpy, Load1d) {
    auto t = io::numpy::load(fixture_path("1d.npy"));

    ASSERT_TRUE(t.ndim() == 1);
    ASSERT_TRUE(t.shape() == Shape({10}));

    // arange(10)
    for (int i = 0; i < 10; i++) {
        ASSERT_NEAR(t.item<float>({static_cast<size_t>(i)}),
                    static_cast<float>(i), 1e-6);
    }
}

TEST(IoNumpy, Load3d) {
    auto t = io::numpy::load(fixture_path("3d.npy"));

    ASSERT_TRUE(t.ndim() == 3);
    ASSERT_TRUE(t.shape() == Shape({2, 3, 4}));
    ASSERT_TRUE(t.size() == 24);
}

TEST(IoNumpy, Load4d) {
    auto t = io::numpy::load(fixture_path("4d.npy"));

    ASSERT_TRUE(t.ndim() == 4);
    ASSERT_TRUE(t.shape() == Shape({2, 3, 4, 5}));
    ASSERT_TRUE(t.size() == 120);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST(IoNumpy, LoadEmpty) {
    auto t = io::numpy::load(fixture_path("empty_1d.npy"));

    ASSERT_TRUE(t.dtype() == DType::Float32);
    ASSERT_TRUE(t.shape() == Shape({0}));
    ASSERT_TRUE(t.size() == 0);
}

TEST(IoNumpy, LoadSingleElement) {
    auto t = io::numpy::load(fixture_path("single_element.npy"));

    ASSERT_TRUE(t.shape() == Shape({1}));
    ASSERT_TRUE(t.size() == 1);
    ASSERT_NEAR(t.item<float>({0}), 42.0f, 1e-6);
}

TEST(IoNumpy, LoadLarge) {
    auto t = io::numpy::load(fixture_path("large_1d.npy"));

    ASSERT_TRUE(t.shape() == Shape({10000}));
    ASSERT_TRUE(t.size() == 10000);

    // Check some values
    ASSERT_NEAR(t.item<float>({0}), 0.0f, 1e-6);
    ASSERT_NEAR(t.item<float>({9999}), 9999.0f, 1e-6);
}

// ============================================================================
// Special Values
// ============================================================================

TEST(IoNumpy, LoadSpecialValues) {
    auto t = io::numpy::load(fixture_path("special_float32.npy"));

    ASSERT_TRUE(t.dtype() == DType::Float32);
    ASSERT_TRUE(t.shape() == Shape({5}));

    // [0.0, -0.0, inf, -inf, nan]
    ASSERT_NEAR(t.item<float>({0}), 0.0f, 1e-6);
    // -0.0 compares equal to 0.0
    ASSERT_TRUE(std::isinf(t.item<float>({2})) && t.item<float>({2}) > 0);
    ASSERT_TRUE(std::isinf(t.item<float>({3})) && t.item<float>({3}) < 0);
    ASSERT_TRUE(std::isnan(t.item<float>({4})));
}

// ============================================================================
// Universal Load Test
// ============================================================================

TEST(IoNumpy, UniversalLoadNpy) {
    // Test that io::load() auto-detects NumPy format
    auto t = io::load(fixture_path("float32_2d.npy"));

    ASSERT_TRUE(t.dtype() == DType::Float32);
    ASSERT_TRUE(t.shape() == Shape({2, 3}));
}

TEST(IoNumpy, LoadArchiveNpy) {
    // Test that load_archive works with .npy files
    auto tensors = io::load_archive(fixture_path("float32_2d.npy"));

    ASSERT_TRUE(tensors.size() == 1);
    // Key should be filename without .npy extension
    ASSERT_TRUE(tensors.count("float32_2d") == 1);
    ASSERT_TRUE(tensors["float32_2d"].shape() == Shape({2, 3}));
}
