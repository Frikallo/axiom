#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>

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

// Temporary test file path
std::string get_test_path(const std::string &name) {
    return "/tmp/axiom_test_" + name + ".axfb";
}

void cleanup_file(const std::string &path) {
    if (fs::exists(path)) {
        fs::remove(path);
    }
}

// ============================================================================
// Basic Round-trip Tests
// ============================================================================

TEST(test_save_load_float32) {
    auto t = Tensor::randn({3, 4, 5}, DType::Float32);
    std::string path = get_test_path("float32");

    io::save(t, path);
    auto loaded = io::load(path);

    ASSERT(loaded.shape() == t.shape());
    ASSERT(loaded.dtype() == t.dtype());
    ASSERT(loaded.allclose(t));

    cleanup_file(path);
}

TEST(test_save_load_float64) {
    auto t = Tensor::randn({10, 20}, DType::Float64);
    std::string path = get_test_path("float64");

    io::save(t, path);
    auto loaded = io::load(path);

    ASSERT(loaded.shape() == t.shape());
    ASSERT(loaded.dtype() == DType::Float64);
    ASSERT(loaded.allclose(t));

    cleanup_file(path);
}

TEST(test_save_load_float16) {
    auto t = Tensor::randn({5, 6}, DType::Float32).half();
    std::string path = get_test_path("float16");

    io::save(t, path);
    auto loaded = io::load(path);

    ASSERT(loaded.shape() == t.shape());
    ASSERT(loaded.dtype() == DType::Float16);
    // Float16 has lower precision, use wider tolerance
    ASSERT(loaded.to_float().allclose(t.to_float(), 1e-2, 1e-2));

    cleanup_file(path);
}

TEST(test_save_load_int32) {
    auto t = Tensor::arange(0, 100).reshape({10, 10});
    std::string path = get_test_path("int32");

    io::save(t, path);
    auto loaded = io::load(path);

    ASSERT(loaded.shape() == t.shape());
    ASSERT(loaded.dtype() == DType::Int32);
    ASSERT(loaded.array_equal(t));

    cleanup_file(path);
}

TEST(test_save_load_int64) {
    auto t = Tensor::arange(0, 50, 1, DType::Int64).reshape({5, 10});
    std::string path = get_test_path("int64");

    io::save(t, path);
    auto loaded = io::load(path);

    ASSERT(loaded.shape() == t.shape());
    ASSERT(loaded.dtype() == DType::Int64);
    ASSERT(loaded.array_equal(t));

    cleanup_file(path);
}

TEST(test_save_load_uint8) {
    auto t = Tensor({256}, DType::UInt8);
    for (int i = 0; i < 256; i++) {
        t.set_item<uint8_t>({static_cast<size_t>(i)}, static_cast<uint8_t>(i));
    }
    std::string path = get_test_path("uint8");

    io::save(t, path);
    auto loaded = io::load(path);

    ASSERT(loaded.shape() == t.shape());
    ASSERT(loaded.dtype() == DType::UInt8);
    ASSERT(loaded.array_equal(t));

    cleanup_file(path);
}

TEST(test_save_load_bool) {
    auto t = Tensor({4, 5}, DType::Bool);
    for (size_t i = 0; i < 4; i++) {
        for (size_t j = 0; j < 5; j++) {
            t.set_item<bool>({i, j}, (i + j) % 2 == 0);
        }
    }
    std::string path = get_test_path("bool");

    io::save(t, path);
    auto loaded = io::load(path);

    ASSERT(loaded.shape() == t.shape());
    ASSERT(loaded.dtype() == DType::Bool);
    ASSERT(loaded.array_equal(t));

    cleanup_file(path);
}

// ============================================================================
// Memory Order Tests
// ============================================================================

TEST(test_save_load_row_major) {
    auto t = Tensor({3, 4}, DType::Float32, Device::CPU, MemoryOrder::RowMajor);
    t.fill(1.0f);
    std::string path = get_test_path("row_major");

    io::save(t, path);
    auto loaded = io::load(path);

    ASSERT(loaded.memory_order() == MemoryOrder::RowMajor);
    ASSERT(loaded.is_c_contiguous());

    cleanup_file(path);
}

TEST(test_save_load_col_major) {
    auto t = Tensor({3, 4}, DType::Float32, Device::CPU, MemoryOrder::ColMajor);
    t.fill(2.0f);
    std::string path = get_test_path("col_major");

    io::save(t, path);
    auto loaded = io::load(path);

    ASSERT(loaded.memory_order() == MemoryOrder::ColMajor);
    ASSERT(loaded.is_f_contiguous());

    cleanup_file(path);
}

// ============================================================================
// Archive Tests
// ============================================================================

TEST(test_save_load_archive) {
    std::map<std::string, Tensor> tensors;
    tensors["weights"] = Tensor::randn({10, 5}, DType::Float32);
    tensors["biases"] = Tensor::randn({5}, DType::Float32);
    tensors["indices"] = Tensor::arange(0, 100, 1, DType::Int64);

    std::string path = get_test_path("archive");

    io::save_archive(tensors, path);
    auto loaded = io::load_archive(path);

    ASSERT(loaded.size() == 3);
    ASSERT(loaded.count("weights") == 1);
    ASSERT(loaded.count("biases") == 1);
    ASSERT(loaded.count("indices") == 1);

    ASSERT(loaded["weights"].shape() == tensors["weights"].shape());
    ASSERT(loaded["biases"].shape() == tensors["biases"].shape());
    ASSERT(loaded["indices"].shape() == tensors["indices"].shape());

    ASSERT(loaded["weights"].allclose(tensors["weights"]));
    ASSERT(loaded["biases"].allclose(tensors["biases"]));
    ASSERT(loaded["indices"].array_equal(tensors["indices"]));

    cleanup_file(path);
}

TEST(test_list_archive) {
    std::map<std::string, Tensor> tensors;
    tensors["alpha"] = Tensor::ones({2, 2});
    tensors["beta"] = Tensor::zeros({3, 3});
    tensors["gamma"] = Tensor::randn({4});

    std::string path = get_test_path("list_archive");

    io::save_archive(tensors, path);
    auto names = io::flatbuffers::list_archive(path);

    ASSERT(names.size() == 3);
    // Names should be in alphabetical order (std::map)
    ASSERT(std::find(names.begin(), names.end(), "alpha") != names.end());
    ASSERT(std::find(names.begin(), names.end(), "beta") != names.end());
    ASSERT(std::find(names.begin(), names.end(), "gamma") != names.end());

    cleanup_file(path);
}

TEST(test_load_from_archive) {
    std::map<std::string, Tensor> tensors;
    tensors["first"] = Tensor::full<float>({3, 3}, 1.0f);
    tensors["second"] = Tensor::full<float>({4, 4}, 2.0f);

    std::string path = get_test_path("load_from_archive");

    io::save_archive(tensors, path);

    auto first = io::flatbuffers::load_from_archive(path, "first");
    auto second = io::flatbuffers::load_from_archive(path, "second");

    ASSERT(first.shape() == Shape({3, 3}));
    ASSERT(second.shape() == Shape({4, 4}));
    ASSERT_NEAR(first.item<float>({0, 0}), 1.0f, 1e-6);
    ASSERT_NEAR(second.item<float>({0, 0}), 2.0f, 1e-6);

    cleanup_file(path);
}

// ============================================================================
// Edge Case Tests
// ============================================================================

TEST(test_save_load_scalar) {
    auto t = Tensor::full<float>({}, 42.0f); // Scalar
    std::string path = get_test_path("scalar");

    io::save(t, path);
    auto loaded = io::load(path);

    ASSERT(loaded.shape().empty());
    ASSERT(loaded.size() == 1);
    ASSERT_NEAR(loaded.item<float>(), 42.0f, 1e-6);

    cleanup_file(path);
}

TEST(test_save_load_1d) {
    auto t = Tensor::randn({100});
    std::string path = get_test_path("1d");

    io::save(t, path);
    auto loaded = io::load(path);

    ASSERT(loaded.ndim() == 1);
    ASSERT(loaded.shape()[0] == 100);
    ASSERT(loaded.allclose(t));

    cleanup_file(path);
}

TEST(test_save_load_high_dim) {
    auto t = Tensor::randn({2, 3, 4, 5, 6});
    std::string path = get_test_path("high_dim");

    io::save(t, path);
    auto loaded = io::load(path);

    ASSERT(loaded.ndim() == 5);
    ASSERT(loaded.shape() == t.shape());
    ASSERT(loaded.allclose(t));

    cleanup_file(path);
}

TEST(test_save_load_large) {
    auto t = Tensor::randn({1000, 1000}); // 1M elements
    std::string path = get_test_path("large");

    io::save(t, path);
    auto loaded = io::load(path);

    ASSERT(loaded.shape() == t.shape());
    ASSERT(loaded.allclose(t));

    cleanup_file(path);
}

// ============================================================================
// Format Detection Tests
// ============================================================================

TEST(test_is_axfb_file) {
    auto t = Tensor::ones({2, 2});
    std::string path = get_test_path("detect");

    io::save(t, path);

    ASSERT(io::flatbuffers::is_axfb_file(path));
    ASSERT(io::detect_format(path) == io::FileFormat::Axiom);

    cleanup_file(path);
}

TEST(test_format_detection_unknown) {
    // Create a file with random content
    std::string path = "/tmp/axiom_test_unknown.bin";
    std::ofstream f(path, std::ios::binary);
    f << "random content that is not a valid format";
    f.close();

    ASSERT(io::detect_format(path) == io::FileFormat::Unknown);
    ASSERT(!io::flatbuffers::is_axfb_file(path));

    cleanup_file(path);
}

// ============================================================================
// Tensor Class Methods Tests
// ============================================================================

TEST(test_tensor_save_load) {
    auto t = Tensor::randn({5, 5});
    std::string path = get_test_path("tensor_method");

    t.save(path);
    auto loaded = Tensor::load(path);

    ASSERT(loaded.allclose(t));

    cleanup_file(path);
}

TEST(test_tensor_save_load_tensors) {
    std::map<std::string, Tensor> tensors;
    tensors["a"] = Tensor::randn({3, 3});
    tensors["b"] = Tensor::randn({4, 4});

    std::string path = get_test_path("tensor_archive_method");

    Tensor::save_tensors(tensors, path);
    auto loaded = Tensor::load_tensors(path);

    ASSERT(loaded.size() == 2);
    ASSERT(loaded["a"].allclose(tensors["a"]));
    ASSERT(loaded["b"].allclose(tensors["b"]));

    cleanup_file(path);
}

int main() {
    std::cout << "\n========================================\n";
    std::cout << "FlatBuffers I/O Tests\n";
    std::cout << "========================================\n\n";

    // Tests are run automatically via static initialization

    std::cout << "\n========================================\n";
    std::cout << "Results: " << tests_passed << "/" << tests_run << " passed\n";
    std::cout << "========================================\n";

    return (tests_passed == tests_run) ? 0 : 1;
}
