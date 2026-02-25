#include "axiom_test_utils.hpp"

#include <cstring>
#include <fstream>

namespace {

// Helper: write a minimal safetensors file with given tensors
void write_safetensors(const std::string &path, const std::string &header_json,
                       const std::vector<uint8_t> &data) {
    std::ofstream file(path, std::ios::binary);
    ASSERT_TRUE(file.is_open()) << "Failed to create test file: " << path;

    uint64_t header_len = header_json.size();
    file.write(reinterpret_cast<const char *>(&header_len), 8);
    file.write(header_json.data(),
               static_cast<std::streamsize>(header_json.size()));
    file.write(reinterpret_cast<const char *>(data.data()),
               static_cast<std::streamsize>(data.size()));
}

std::string test_file_path(const std::string &name) {
    return "/tmp/axiom_test_" + name + ".safetensors";
}

} // namespace

TEST(SafeTensors, IsFileDetection) {
    EXPECT_TRUE(
        axiom::io::safetensors::is_safetensors_file("model.safetensors"));
    EXPECT_TRUE(axiom::io::safetensors::is_safetensors_file(
        "/path/to/weights.safetensors"));
    EXPECT_FALSE(axiom::io::safetensors::is_safetensors_file("model.bin"));
    EXPECT_FALSE(axiom::io::safetensors::is_safetensors_file("model.npy"));
}

TEST(SafeTensors, LoadSingleFloat32Tensor) {
    std::string path = test_file_path("single_f32");

    // Create a 2x3 float32 tensor: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    std::vector<float> values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<uint8_t> data(values.size() * sizeof(float));
    std::memcpy(data.data(), values.data(), data.size());

    std::string header =
        R"({"weight":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}})";

    write_safetensors(path, header, data);

    auto tensors = axiom::io::safetensors::load(path);
    ASSERT_EQ(tensors.size(), 1u);
    ASSERT_TRUE(tensors.count("weight") > 0);

    auto &t = tensors["weight"];
    ASSERT_TRUE(t.shape() == axiom::Shape({2, 3}));
    ASSERT_EQ(t.dtype(), axiom::DType::Float32);

    axiom::testing::ExpectTensorEquals<float>(
        t, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
}

TEST(SafeTensors, LoadMultipleTensors) {
    std::string path = test_file_path("multi");

    // Two tensors: a (2,) int32 and b (3,) float32
    // a = [10, 20], b = [1.5, 2.5, 3.5]
    std::vector<uint8_t> data(2 * 4 + 3 * 4); // 8 + 12 = 20 bytes
    int32_t a_vals[] = {10, 20};
    float b_vals[] = {1.5f, 2.5f, 3.5f};
    std::memcpy(data.data(), a_vals, 8);
    std::memcpy(data.data() + 8, b_vals, 12);

    std::string header =
        R"({"a":{"dtype":"I32","shape":[2],"data_offsets":[0,8]},"b":{"dtype":"F32","shape":[3],"data_offsets":[8,20]}})";

    write_safetensors(path, header, data);

    auto tensors = axiom::io::safetensors::load(path);
    ASSERT_EQ(tensors.size(), 2u);

    ASSERT_TRUE(tensors["a"].shape() == axiom::Shape({2}));
    ASSERT_EQ(tensors["a"].dtype(), axiom::DType::Int32);
    axiom::testing::ExpectTensorEquals<int32_t>(tensors["a"], {10, 20});

    ASSERT_TRUE(tensors["b"].shape() == axiom::Shape({3}));
    ASSERT_EQ(tensors["b"].dtype(), axiom::DType::Float32);
    axiom::testing::ExpectTensorEquals<float>(tensors["b"], {1.5f, 2.5f, 3.5f});
}

TEST(SafeTensors, ListTensors) {
    std::string path = test_file_path("list");

    std::vector<uint8_t> data(16);
    std::string header =
        R"({"encoder.weight":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}})";

    write_safetensors(path, header, data);

    auto names = axiom::io::safetensors::list_tensors(path);
    ASSERT_EQ(names.size(), 1u);
    EXPECT_EQ(names[0], "encoder.weight");
}

TEST(SafeTensors, LoadSpecificTensor) {
    std::string path = test_file_path("specific");

    std::vector<uint8_t> data(8 + 12);
    int32_t a_vals[] = {1, 2};
    float b_vals[] = {10.0f, 20.0f, 30.0f};
    std::memcpy(data.data(), a_vals, 8);
    std::memcpy(data.data() + 8, b_vals, 12);

    std::string header =
        R"({"a":{"dtype":"I32","shape":[2],"data_offsets":[0,8]},"b":{"dtype":"F32","shape":[3],"data_offsets":[8,20]}})";

    write_safetensors(path, header, data);

    auto t = axiom::io::safetensors::load_tensor(path, "b");
    ASSERT_TRUE(t.shape() == axiom::Shape({3}));
    axiom::testing::ExpectTensorEquals<float>(t, {10.0f, 20.0f, 30.0f});
}

TEST(SafeTensors, MetadataSkipped) {
    std::string path = test_file_path("metadata");

    std::vector<float> values = {1.0f, 2.0f};
    std::vector<uint8_t> data(8);
    std::memcpy(data.data(), values.data(), 8);

    std::string header =
        R"({"__metadata__":{"format":"pt"},"w":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}})";

    write_safetensors(path, header, data);

    auto tensors = axiom::io::safetensors::load(path);
    ASSERT_EQ(tensors.size(), 1u);
    ASSERT_TRUE(tensors.count("w") > 0);
}

TEST(SafeTensors, FormatDetection) {
    std::string path = test_file_path("detect");

    std::vector<uint8_t> data(4);
    std::string header =
        R"({"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}})";
    write_safetensors(path, header, data);

    auto format = axiom::io::detect_format(path);
    EXPECT_EQ(format, axiom::io::FileFormat::SafeTensors);
}

TEST(SafeTensors, UniversalLoad) {
    std::string path = test_file_path("universal");

    std::vector<float> values = {42.0f};
    std::vector<uint8_t> data(4);
    std::memcpy(data.data(), values.data(), 4);

    std::string header =
        R"({"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}})";
    write_safetensors(path, header, data);

    // Test through universal load
    auto t = axiom::io::load(path);
    EXPECT_FLOAT_EQ(t.item<float>({0}), 42.0f);

    // Test through universal load_archive
    auto archive = axiom::io::load_archive(path);
    ASSERT_EQ(archive.size(), 1u);
}
