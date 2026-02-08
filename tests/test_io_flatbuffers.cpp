#include "axiom_test_utils.hpp"
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>

using namespace axiom;

namespace fs = std::filesystem;

// Temporary test file path (cross-platform)
std::string get_test_path(const std::string &name) {
    return (fs::temp_directory_path() / ("axiom_test_" + name + ".axfb"))
        .string();
}

void cleanup_file(const std::string &path) {
    if (fs::exists(path)) {
        fs::remove(path);
    }
}

// ============================================================================
// Basic Round-trip Tests
// ============================================================================

TEST(IoFlatbuffers, SaveLoadFloat32) {
    auto t = Tensor::randn({3, 4, 5}, DType::Float32);
    std::string path = get_test_path("float32");

    io::save(t, path);
    auto loaded = io::load(path);

    ASSERT_TRUE(loaded.shape() == t.shape());
    ASSERT_TRUE(loaded.dtype() == t.dtype());
    ASSERT_TRUE(loaded.allclose(t));

    cleanup_file(path);
}

TEST(IoFlatbuffers, SaveLoadFloat64) {
    auto t = Tensor::randn({10, 20}, DType::Float64);
    std::string path = get_test_path("float64");

    io::save(t, path);
    auto loaded = io::load(path);

    ASSERT_TRUE(loaded.shape() == t.shape());
    ASSERT_TRUE(loaded.dtype() == DType::Float64);
    ASSERT_TRUE(loaded.allclose(t));

    cleanup_file(path);
}

TEST(IoFlatbuffers, SaveLoadFloat16) {
    auto t = Tensor::randn({5, 6}, DType::Float32).half();
    std::string path = get_test_path("float16");

    io::save(t, path);
    auto loaded = io::load(path);

    ASSERT_TRUE(loaded.shape() == t.shape());
    ASSERT_TRUE(loaded.dtype() == DType::Float16);
    // Float16 has lower precision, use wider tolerance
    ASSERT_TRUE(loaded.to_float().allclose(t.to_float(), 1e-2, 1e-2));

    cleanup_file(path);
}

TEST(IoFlatbuffers, SaveLoadInt32) {
    auto t = Tensor::arange(0, 100).reshape({10, 10});
    std::string path = get_test_path("int32");

    io::save(t, path);
    auto loaded = io::load(path);

    ASSERT_TRUE(loaded.shape() == t.shape());
    ASSERT_TRUE(loaded.dtype() == DType::Int32);
    ASSERT_TRUE(loaded.array_equal(t));

    cleanup_file(path);
}

TEST(IoFlatbuffers, SaveLoadInt64) {
    auto t = Tensor::arange(0, 50, 1, DType::Int64).reshape({5, 10});
    std::string path = get_test_path("int64");

    io::save(t, path);
    auto loaded = io::load(path);

    ASSERT_TRUE(loaded.shape() == t.shape());
    ASSERT_TRUE(loaded.dtype() == DType::Int64);
    ASSERT_TRUE(loaded.array_equal(t));

    cleanup_file(path);
}

TEST(IoFlatbuffers, SaveLoadUint8) {
    auto t = Tensor({256}, DType::UInt8);
    for (int i = 0; i < 256; i++) {
        t.set_item<uint8_t>({static_cast<size_t>(i)}, static_cast<uint8_t>(i));
    }
    std::string path = get_test_path("uint8");

    io::save(t, path);
    auto loaded = io::load(path);

    ASSERT_TRUE(loaded.shape() == t.shape());
    ASSERT_TRUE(loaded.dtype() == DType::UInt8);
    ASSERT_TRUE(loaded.array_equal(t));

    cleanup_file(path);
}

TEST(IoFlatbuffers, SaveLoadBool) {
    auto t = Tensor({4, 5}, DType::Bool);
    for (size_t i = 0; i < 4; i++) {
        for (size_t j = 0; j < 5; j++) {
            t.set_item<bool>({i, j}, (i + j) % 2 == 0);
        }
    }
    std::string path = get_test_path("bool");

    io::save(t, path);
    auto loaded = io::load(path);

    ASSERT_TRUE(loaded.shape() == t.shape());
    ASSERT_TRUE(loaded.dtype() == DType::Bool);
    ASSERT_TRUE(loaded.array_equal(t));

    cleanup_file(path);
}

// ============================================================================
// Memory Order Tests
// ============================================================================

TEST(IoFlatbuffers, SaveLoadRowMajor) {
    auto t = Tensor({3, 4}, DType::Float32, Device::CPU, MemoryOrder::RowMajor);
    t.fill(1.0f);
    std::string path = get_test_path("row_major");

    io::save(t, path);
    auto loaded = io::load(path);

    ASSERT_TRUE(loaded.memory_order() == MemoryOrder::RowMajor);
    ASSERT_TRUE(loaded.is_c_contiguous());

    cleanup_file(path);
}

TEST(IoFlatbuffers, SaveLoadColMajor) {
    auto t = Tensor({3, 4}, DType::Float32, Device::CPU, MemoryOrder::ColMajor);
    t.fill(2.0f);
    std::string path = get_test_path("col_major");

    io::save(t, path);
    auto loaded = io::load(path);

    ASSERT_TRUE(loaded.memory_order() == MemoryOrder::ColMajor);
    ASSERT_TRUE(loaded.is_f_contiguous());

    cleanup_file(path);
}

// ============================================================================
// Archive Tests
// ============================================================================

TEST(IoFlatbuffers, SaveLoadArchive) {
    std::map<std::string, Tensor> tensors;
    tensors["weights"] = Tensor::randn({10, 5}, DType::Float32);
    tensors["biases"] = Tensor::randn({5}, DType::Float32);
    tensors["indices"] = Tensor::arange(0, 100, 1, DType::Int64);

    std::string path = get_test_path("archive");

    io::save_archive(tensors, path);
    auto loaded = io::load_archive(path);

    ASSERT_TRUE(loaded.size() == 3);
    ASSERT_TRUE(loaded.count("weights") == 1);
    ASSERT_TRUE(loaded.count("biases") == 1);
    ASSERT_TRUE(loaded.count("indices") == 1);

    ASSERT_TRUE(loaded["weights"].shape() == tensors["weights"].shape());
    ASSERT_TRUE(loaded["biases"].shape() == tensors["biases"].shape());
    ASSERT_TRUE(loaded["indices"].shape() == tensors["indices"].shape());

    ASSERT_TRUE(loaded["weights"].allclose(tensors["weights"]));
    ASSERT_TRUE(loaded["biases"].allclose(tensors["biases"]));
    ASSERT_TRUE(loaded["indices"].array_equal(tensors["indices"]));

    cleanup_file(path);
}

TEST(IoFlatbuffers, ListArchive) {
    std::map<std::string, Tensor> tensors;
    tensors["alpha"] = Tensor::ones({2, 2});
    tensors["beta"] = Tensor::zeros({3, 3});
    tensors["gamma"] = Tensor::randn({4});

    std::string path = get_test_path("list_archive");

    io::save_archive(tensors, path);
    auto names = io::flatbuffers::list_archive(path);

    ASSERT_TRUE(names.size() == 3);
    // Names should be in alphabetical order (std::map)
    ASSERT_TRUE(std::find(names.begin(), names.end(), "alpha") != names.end());
    ASSERT_TRUE(std::find(names.begin(), names.end(), "beta") != names.end());
    ASSERT_TRUE(std::find(names.begin(), names.end(), "gamma") != names.end());

    cleanup_file(path);
}

TEST(IoFlatbuffers, LoadFromArchive) {
    std::map<std::string, Tensor> tensors;
    tensors["first"] = Tensor::full<float>({3, 3}, 1.0f);
    tensors["second"] = Tensor::full<float>({4, 4}, 2.0f);

    std::string path = get_test_path("load_from_archive");

    io::save_archive(tensors, path);

    auto first = io::flatbuffers::load_from_archive(path, "first");
    auto second = io::flatbuffers::load_from_archive(path, "second");

    ASSERT_TRUE(first.shape() == Shape({3, 3}));
    ASSERT_TRUE(second.shape() == Shape({4, 4}));
    ASSERT_NEAR(first.item<float>({0, 0}), 1.0f, 1e-6);
    ASSERT_NEAR(second.item<float>({0, 0}), 2.0f, 1e-6);

    cleanup_file(path);
}

// ============================================================================
// Edge Case Tests
// ============================================================================

TEST(IoFlatbuffers, SaveLoadScalar) {
    auto t = Tensor::full<float>({}, 42.0f); // Scalar
    std::string path = get_test_path("scalar");

    io::save(t, path);
    auto loaded = io::load(path);

    ASSERT_TRUE(loaded.shape().empty());
    ASSERT_TRUE(loaded.size() == 1);
    ASSERT_NEAR(loaded.item<float>(), 42.0f, 1e-6);

    cleanup_file(path);
}

TEST(IoFlatbuffers, SaveLoad1d) {
    auto t = Tensor::randn({100});
    std::string path = get_test_path("1d");

    io::save(t, path);
    auto loaded = io::load(path);

    ASSERT_TRUE(loaded.ndim() == 1);
    ASSERT_TRUE(loaded.shape()[0] == 100);
    ASSERT_TRUE(loaded.allclose(t));

    cleanup_file(path);
}

TEST(IoFlatbuffers, SaveLoadHighDim) {
    auto t = Tensor::randn({2, 3, 4, 5, 6});
    std::string path = get_test_path("high_dim");

    io::save(t, path);
    auto loaded = io::load(path);

    ASSERT_TRUE(loaded.ndim() == 5);
    ASSERT_TRUE(loaded.shape() == t.shape());
    ASSERT_TRUE(loaded.allclose(t));

    cleanup_file(path);
}

TEST(IoFlatbuffers, SaveLoadLarge) {
    auto t = Tensor::randn({1000, 1000}); // 1M elements
    std::string path = get_test_path("large");

    io::save(t, path);
    auto loaded = io::load(path);

    ASSERT_TRUE(loaded.shape() == t.shape());
    ASSERT_TRUE(loaded.allclose(t));

    cleanup_file(path);
}

// ============================================================================
// Format Detection Tests
// ============================================================================

TEST(IoFlatbuffers, IsAxfbFile) {
    auto t = Tensor::ones({2, 2});
    std::string path = get_test_path("detect");

    io::save(t, path);

    ASSERT_TRUE(io::flatbuffers::is_axfb_file(path));
    ASSERT_TRUE(io::detect_format(path) == io::FileFormat::Axiom);

    cleanup_file(path);
}

TEST(IoFlatbuffers, FormatDetectionUnknown) {
    // Create a file with random content
    std::string path =
        (fs::temp_directory_path() / "axiom_test_unknown.bin").string();
    std::ofstream f(path, std::ios::binary);
    f << "random content that is not a valid format";
    f.close();

    ASSERT_TRUE(io::detect_format(path) == io::FileFormat::Unknown);
    ASSERT_TRUE(!io::flatbuffers::is_axfb_file(path));

    cleanup_file(path);
}

// ============================================================================
// Tensor Class Methods Tests
// ============================================================================

TEST(IoFlatbuffers, TensorSaveLoad) {
    auto t = Tensor::randn({5, 5});
    std::string path = get_test_path("tensor_method");

    t.save(path);
    auto loaded = Tensor::load(path);

    ASSERT_TRUE(loaded.allclose(t));

    cleanup_file(path);
}

TEST(IoFlatbuffers, TensorSaveLoadTensors) {
    std::map<std::string, Tensor> tensors;
    tensors["a"] = Tensor::randn({3, 3});
    tensors["b"] = Tensor::randn({4, 4});

    std::string path = get_test_path("tensor_archive_method");

    Tensor::save_tensors(tensors, path);
    auto loaded = Tensor::load_tensors(path);

    ASSERT_TRUE(loaded.size() == 2);
    ASSERT_TRUE(loaded["a"].allclose(tensors["a"]));
    ASSERT_TRUE(loaded["b"].allclose(tensors["b"]));

    cleanup_file(path);
}
