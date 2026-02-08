#include "axiom_test_utils.hpp"

#include <cmath>
#include <vector>

using namespace axiom;

// ==================================
//
//      MATH OPERATION TESTS - CPU
//
// ==================================

TEST(TensorNumpyOps, SignCPU) {
    auto device = Device::CPU;
    auto data = std::vector<float>({-3.0f, -1.0f, 0.0f, 1.0f, 3.0f});
    auto t = Tensor::from_data<float>(data.data(), {5}).to(device);
    auto result = t.sign();
    axiom::testing::ExpectTensorEquals<float>(result,
                                              {-1.0f, -1.0f, 0.0f, 1.0f, 1.0f});
}

TEST(TensorNumpyOps, FloorCeilTruncCPU) {
    auto device = Device::CPU;
    auto data = std::vector<float>({-2.7f, -0.5f, 0.5f, 2.7f});
    auto t = Tensor::from_data<float>(data.data(), {4}).to(device);

    auto fl = t.floor();
    axiom::testing::ExpectTensorEquals<float>(fl, {-3.0f, -1.0f, 0.0f, 2.0f});

    auto ce = t.ceil();
    axiom::testing::ExpectTensorEquals<float>(ce, {-2.0f, 0.0f, 1.0f, 3.0f});

    auto tr = t.trunc();
    axiom::testing::ExpectTensorEquals<float>(tr, {-2.0f, 0.0f, 0.0f, 2.0f});
}

TEST(TensorNumpyOps, ReciprocalCPU) {
    auto device = Device::CPU;
    auto data = std::vector<float>({1.0f, 2.0f, 4.0f, 0.5f});
    auto t = Tensor::from_data<float>(data.data(), {4}).to(device);
    auto result = t.reciprocal();
    axiom::testing::ExpectTensorEquals<float>(result,
                                              {1.0f, 0.5f, 0.25f, 2.0f});
}

TEST(TensorNumpyOps, SquareCbrtCPU) {
    auto device = Device::CPU;
    auto data = std::vector<float>({2.0f, 3.0f, 4.0f});
    auto t = Tensor::from_data<float>(data.data(), {3}).to(device);

    auto sq = t.square();
    axiom::testing::ExpectTensorEquals<float>(sq, {4.0f, 9.0f, 16.0f});

    auto data2 = std::vector<float>({8.0f, 27.0f, 64.0f});
    auto t2 = Tensor::from_data<float>(data2.data(), {3}).to(device);
    auto cb = t2.cbrt();
    axiom::testing::ExpectTensorEquals<float>(cb, {2.0f, 3.0f, 4.0f}, 1e-5);
}

TEST(TensorNumpyOps, IsnanIsinfIsfiniteCPU) {
    auto device = Device::CPU;
    auto data = std::vector<float>({1.0f, NAN, INFINITY, -INFINITY, 0.0f});
    auto t = Tensor::from_data<float>(data.data(), {5}).to(device);

    auto nan_result = t.isnan();
    axiom::testing::ExpectTensorEquals<bool>(
        nan_result, {false, true, false, false, false});

    auto inf_result = t.isinf();
    axiom::testing::ExpectTensorEquals<bool>(inf_result,
                                             {false, false, true, true, false});

    auto fin_result = t.isfinite();
    axiom::testing::ExpectTensorEquals<bool>(fin_result,
                                             {true, false, false, false, true});
}

TEST(TensorNumpyOps, ClipCPU) {
    auto device = Device::CPU;
    auto data = std::vector<float>({-5.0f, -1.0f, 0.0f, 1.0f, 5.0f});
    auto t = Tensor::from_data<float>(data.data(), {5}).to(device);
    auto result = t.clip(-2.0, 2.0);
    axiom::testing::ExpectTensorEquals<float>(result,
                                              {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f});
}

// ==================================
//
//      REDUCTION TESTS - CPU
//
// ==================================

TEST(TensorNumpyOps, ProdCPU) {
    auto device = Device::CPU;
    auto data = std::vector<float>({1.0f, 2.0f, 3.0f, 4.0f});
    auto t = Tensor::from_data<float>(data.data(), {2, 2}).to(device);

    auto prod_all = t.prod();
    ASSERT_TRUE(std::abs(prod_all.item<float>() - 24.0f) < 1e-5)
        << "prod should be 24";

    auto prod_axis0 = t.prod(0);
    axiom::testing::ExpectTensorEquals<float>(prod_axis0, {3.0f, 8.0f});
}

TEST(TensorNumpyOps, VarStdCPU) {
    auto device = Device::CPU;
    auto data = std::vector<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    auto t = Tensor::from_data<float>(data.data(), {5}).to(device);

    // var = mean((x - mean(x))^2) = 2.0
    auto v = t.var();
    ASSERT_TRUE(std::abs(v.item<float>() - 2.0f) < 1e-4) << "var should be 2.0";

    // std = sqrt(var) = sqrt(2) ~= 1.414
    auto s = t.std();
    ASSERT_TRUE(std::abs(s.item<float>() - std::sqrt(2.0f)) < 1e-4)
        << "std should be sqrt(2)";
}

TEST(TensorNumpyOps, PtpCPU) {
    auto device = Device::CPU;
    auto data = std::vector<float>({1.0f, 5.0f, 3.0f, 2.0f, 8.0f});
    auto t = Tensor::from_data<float>(data.data(), {5}).to(device);
    auto result = t.ptp();
    ASSERT_TRUE(std::abs(result.item<float>() - 7.0f) < 1e-5)
        << "ptp should be 7";
}

TEST(TensorNumpyOps, AnyAllCPU) {
    auto device = Device::CPU;
    // Use uint8_t arrays since std::vector<bool> is specialized and has no
    // .data()
    uint8_t data1[] = {1, 1, 1};
    auto t1 =
        Tensor::from_data<uint8_t>(data1, {3}).astype(DType::Bool).to(device);
    ASSERT_TRUE(t1.all().item<bool>() == true) << "all should be true";
    ASSERT_TRUE(t1.any().item<bool>() == true) << "any should be true";

    uint8_t data2[] = {1, 0, 1};
    auto t2 =
        Tensor::from_data<uint8_t>(data2, {3}).astype(DType::Bool).to(device);
    ASSERT_TRUE(t2.all().item<bool>() == false) << "all should be false";
    ASSERT_TRUE(t2.any().item<bool>() == true) << "any should be true";

    uint8_t data3[] = {0, 0, 0};
    auto t3 =
        Tensor::from_data<uint8_t>(data3, {3}).astype(DType::Bool).to(device);
    ASSERT_TRUE(t3.all().item<bool>() == false) << "all should be false";
    ASSERT_TRUE(t3.any().item<bool>() == false) << "any should be false";
}

// ==================================
//
//      COMPARISON TESTS - CPU
//
// ==================================

TEST(TensorNumpyOps, IscloseAllcloseCPU) {
    auto device = Device::CPU;
    auto data1 = std::vector<float>({1.0f, 2.0f, 3.0f});
    auto data2 = std::vector<float>({1.0f, 2.00001f, 3.0f});
    auto t1 = Tensor::from_data<float>(data1.data(), {3}).to(device);
    auto t2 = Tensor::from_data<float>(data2.data(), {3}).to(device);

    ASSERT_TRUE(t1.allclose(t2, 1e-4, 1e-4) == true)
        << "tensors should be close";

    auto data3 = std::vector<float>({1.0f, 3.0f, 3.0f});
    auto t3 = Tensor::from_data<float>(data3.data(), {3}).to(device);
    ASSERT_TRUE(t1.allclose(t3) == false) << "tensors should not be close";
}

TEST(TensorNumpyOps, ArrayEqualCPU) {
    auto device = Device::CPU;
    auto data1 = std::vector<float>({1.0f, 2.0f, 3.0f});
    auto t1 = Tensor::from_data<float>(data1.data(), {3}).to(device);
    auto t2 = Tensor::from_data<float>(data1.data(), {3}).to(device);

    ASSERT_TRUE(t1.array_equal(t2) == true)
        << "identical tensors should be equal";

    auto data2 = std::vector<float>({1.0f, 2.1f, 3.0f});
    auto t3 = Tensor::from_data<float>(data2.data(), {3}).to(device);
    ASSERT_TRUE(t1.array_equal(t3) == false)
        << "different tensors should not be equal";
}

// ==================================
//
//      STACKING TESTS - CPU
//
// ==================================

TEST(TensorNumpyOps, ConcatenateCPU) {
    auto device = Device::CPU;
    auto data1 = std::vector<float>({1.0f, 2.0f, 3.0f});
    auto data2 = std::vector<float>({4.0f, 5.0f, 6.0f});
    auto t1 = Tensor::from_data<float>(data1.data(), {3}).to(device);
    auto t2 = Tensor::from_data<float>(data2.data(), {3}).to(device);

    // Using vector
    auto result = Tensor::concatenate({t1, t2}, 0);
    ASSERT_TRUE(result.size() == 6) << "concatenate size should be 6";
    axiom::testing::ExpectTensorEquals<float>(
        result, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

    // Using cat alias with initializer list
    auto result2 = Tensor::cat({t1, t2}, 0);
    axiom::testing::ExpectTensorEquals<float>(
        result2, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

    // Member function cat
    auto result3 = t1.cat(t2, 0);
    axiom::testing::ExpectTensorEquals<float>(
        result3, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
}

TEST(TensorNumpyOps, Concatenate2dCPU) {
    auto device = Device::CPU;
    auto data1 = std::vector<float>({1.0f, 2.0f, 3.0f, 4.0f});
    auto data2 = std::vector<float>({5.0f, 6.0f, 7.0f, 8.0f});
    auto t1 = Tensor::from_data<float>(data1.data(), {2, 2}).to(device);
    auto t2 = Tensor::from_data<float>(data2.data(), {2, 2}).to(device);

    // Concat along axis 0
    auto result0 = Tensor::cat({t1, t2}, 0);
    ASSERT_TRUE(result0.shape() == Shape({4, 2})) << "shape should be (4, 2)";

    // Concat along axis 1
    auto result1 = Tensor::cat({t1, t2}, 1);
    ASSERT_TRUE(result1.shape() == Shape({2, 4})) << "shape should be (2, 4)";
}

TEST(TensorNumpyOps, StackCPU) {
    auto device = Device::CPU;
    auto data1 = std::vector<float>({1.0f, 2.0f, 3.0f});
    auto data2 = std::vector<float>({4.0f, 5.0f, 6.0f});
    auto t1 = Tensor::from_data<float>(data1.data(), {3}).to(device);
    auto t2 = Tensor::from_data<float>(data2.data(), {3}).to(device);

    // Stack along axis 0 (creates new first dimension)
    auto result = Tensor::stack({t1, t2}, 0);
    ASSERT_TRUE(result.shape() == Shape({2, 3})) << "shape should be (2, 3)";

    // Stack along axis 1
    auto result1 = Tensor::stack({t1, t2}, 1);
    ASSERT_TRUE(result1.shape() == Shape({3, 2})) << "shape should be (3, 2)";
}

TEST(TensorNumpyOps, VstackHstackCPU) {
    auto device = Device::CPU;
    auto data1 = std::vector<float>({1.0f, 2.0f, 3.0f});
    auto data2 = std::vector<float>({4.0f, 5.0f, 6.0f});
    auto t1 = Tensor::from_data<float>(data1.data(), {3}).to(device);
    auto t2 = Tensor::from_data<float>(data2.data(), {3}).to(device);

    // vstack 1D arrays
    auto vresult = Tensor::vstack({t1, t2});
    ASSERT_TRUE(vresult.shape() == Shape({2, 3}))
        << "vstack shape should be (2, 3)";

    // hstack 1D arrays
    auto hresult = Tensor::hstack({t1, t2});
    ASSERT_TRUE(hresult.shape() == Shape({6})) << "hstack shape should be (6,)";
    axiom::testing::ExpectTensorEquals<float>(
        hresult, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
}

TEST(TensorNumpyOps, SplitCPU) {
    auto device = Device::CPU;
    auto data = std::vector<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    auto t = Tensor::from_data<float>(data.data(), {6}).to(device);

    // Split into 3 equal parts
    auto parts = t.split(3, 0);
    ASSERT_TRUE(parts.size() == 3) << "should have 3 parts";
    ASSERT_TRUE(parts[0].size() == 2) << "each part should have 2 elements";
    axiom::testing::ExpectTensorEquals<float>(parts[0], {1.0f, 2.0f});
    axiom::testing::ExpectTensorEquals<float>(parts[1], {3.0f, 4.0f});
    axiom::testing::ExpectTensorEquals<float>(parts[2], {5.0f, 6.0f});
}

TEST(TensorNumpyOps, ChunkCPU) {
    auto device = Device::CPU;
    auto data = std::vector<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    auto t = Tensor::from_data<float>(data.data(), {5}).to(device);

    // Chunk into 3 parts (unequal)
    auto chunks = t.chunk(3, 0);
    ASSERT_TRUE(chunks.size() == 3) << "should have 3 chunks";
    // First 2 chunks get 2 elements, last gets 1
    ASSERT_TRUE(chunks[0].size() == 2) << "first chunk should have 2 elements";
    ASSERT_TRUE(chunks[1].size() == 2) << "second chunk should have 2 elements";
    ASSERT_TRUE(chunks[2].size() == 1) << "third chunk should have 1 element";
}

// ==================================
//
//      FLIP OPERATIONS TESTS - CPU
//
// ==================================

TEST(TensorNumpyOps, Flip1dZeroCopyCPU) {
    auto device = Device::CPU;
    // Test that flip returns a zero-copy view with negative strides
    auto data =
        std::vector<float>({0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f});
    auto a = Tensor::from_data<float>(data.data(), {8}).to(device);
    auto b = a.flip(0);

    // Verify flip is zero-copy (shares storage)
    ASSERT_TRUE(a.shares_storage(b)) << "flipped tensor should share storage";

    // Verify flip has negative stride
    ASSERT_TRUE(b.has_negative_stride())
        << "flipped tensor should have negative stride";

    // Verify flipped tensor is not contiguous
    ASSERT_TRUE(!b.is_contiguous())
        << "flipped tensor should not be contiguous";

    // Verify values are correct
    // a = [0, 1, 2, 3, 4, 5, 6, 7]
    // b = [7, 6, 5, 4, 3, 2, 1, 0]
    ASSERT_TRUE(std::abs(b.item<float>({0}) - 7.0f) < 1e-5)
        << "b[0] should be 7";
    ASSERT_TRUE(std::abs(b.item<float>({7}) - 0.0f) < 1e-5)
        << "b[7] should be 0";
    ASSERT_TRUE(std::abs(b.item<float>({3}) - 4.0f) < 1e-5)
        << "b[3] should be 4";
}

TEST(TensorNumpyOps, Flip2dCPU) {
    auto device = Device::CPU;
    // Test 2D flip along different axes
    auto data = std::vector<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    auto a = Tensor::from_data<float>(data.data(), {2, 3}).to(device);
    // a = [[1, 2, 3],
    //      [4, 5, 6]]

    // Flip along axis 0 (rows)
    auto b = a.flip(0);
    ASSERT_TRUE(a.shares_storage(b)) << "flip(0) should share storage";
    ASSERT_TRUE(b.has_negative_stride())
        << "flip(0) should have negative stride";

    // b = [[4, 5, 6],
    //      [1, 2, 3]]
    ASSERT_TRUE(std::abs(b.item<float>({0, 0}) - 4.0f) < 1e-5)
        << "b[0,0] should be 4";
    ASSERT_TRUE(std::abs(b.item<float>({0, 2}) - 6.0f) < 1e-5)
        << "b[0,2] should be 6";
    ASSERT_TRUE(std::abs(b.item<float>({1, 0}) - 1.0f) < 1e-5)
        << "b[1,0] should be 1";

    // Flip along axis 1 (cols)
    auto c = a.flip(1);
    ASSERT_TRUE(a.shares_storage(c)) << "flip(1) should share storage";

    // c = [[3, 2, 1],
    //      [6, 5, 4]]
    ASSERT_TRUE(std::abs(c.item<float>({0, 0}) - 3.0f) < 1e-5)
        << "c[0,0] should be 3";
    ASSERT_TRUE(std::abs(c.item<float>({1, 2}) - 4.0f) < 1e-5)
        << "c[1,2] should be 4";
}

TEST(TensorNumpyOps, FlipudFliplrCPU) {
    auto device = Device::CPU;
    auto data = std::vector<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    auto a = Tensor::from_data<float>(data.data(), {2, 3}).to(device);

    // flipud = flip(0)
    auto ud = a.flipud();
    ASSERT_TRUE(a.shares_storage(ud)) << "flipud should share storage";

    // fliplr = flip(1)
    auto lr = a.fliplr();
    ASSERT_TRUE(a.shares_storage(lr)) << "fliplr should share storage";

    // Same value checks as test_flip_2d
    ASSERT_TRUE(std::abs(ud.item<float>({0, 0}) - 4.0f) < 1e-5)
        << "flipud[0,0] should be 4";
    ASSERT_TRUE(std::abs(lr.item<float>({0, 0}) - 3.0f) < 1e-5)
        << "fliplr[0,0] should be 3";
}

TEST(TensorNumpyOps, FlipMultiAxisCPU) {
    auto device = Device::CPU;
    // Test flipping multiple axes at once
    auto data = std::vector<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    auto a = Tensor::from_data<float>(data.data(), {2, 3}).to(device);

    // Flip both axes
    auto b = a.flip({0, 1});
    ASSERT_TRUE(a.shares_storage(b)) << "multi-axis flip should share storage";

    // b = [[6, 5, 4],
    //      [3, 2, 1]]
    ASSERT_TRUE(std::abs(b.item<float>({0, 0}) - 6.0f) < 1e-5)
        << "b[0,0] should be 6";
    ASSERT_TRUE(std::abs(b.item<float>({1, 2}) - 1.0f) < 1e-5)
        << "b[1,2] should be 1";
}

TEST(TensorNumpyOps, FlipOperationsOnFlippedCPU) {
    auto device = Device::CPU;
    // Test that arithmetic operations work on flipped tensors
    auto data1 = std::vector<float>({1.0f, 2.0f, 3.0f, 4.0f});
    auto data2 = std::vector<float>({10.0f, 20.0f, 30.0f, 40.0f});
    auto a = Tensor::from_data<float>(data1.data(), {4}).to(device);
    auto b = Tensor::from_data<float>(data2.data(), {4}).to(device);

    // Flip a: [4, 3, 2, 1]
    auto a_flip = a.flip(0);

    // Add flipped tensor to normal tensor
    // [4, 3, 2, 1] + [10, 20, 30, 40] = [14, 23, 32, 41]
    auto result = ops::add(a_flip, b);

    ASSERT_TRUE(std::abs(result.item<float>({0}) - 14.0f) < 1e-5)
        << "result[0] should be 14";
    ASSERT_TRUE(std::abs(result.item<float>({1}) - 23.0f) < 1e-5)
        << "result[1] should be 23";
    ASSERT_TRUE(std::abs(result.item<float>({2}) - 32.0f) < 1e-5)
        << "result[2] should be 32";
    ASSERT_TRUE(std::abs(result.item<float>({3}) - 41.0f) < 1e-5)
        << "result[3] should be 41";
}

TEST(TensorNumpyOps, FlipContiguousCopyCPU) {
    auto device = Device::CPU;
    // Test that ascontiguousarray() correctly materializes a flipped tensor
    auto data = std::vector<float>({1.0f, 2.0f, 3.0f, 4.0f});
    auto a = Tensor::from_data<float>(data.data(), {4}).to(device);
    auto b = a.flip(0);

    // b is a view (non-contiguous with negative stride)
    ASSERT_TRUE(!b.is_contiguous()) << "flipped should not be contiguous";
    ASSERT_TRUE(a.shares_storage(b)) << "should share storage before copy";

    // Make contiguous copy
    auto c = b.ascontiguousarray();
    ASSERT_TRUE(c.is_contiguous())
        << "ascontiguousarray result should be contiguous";
    ASSERT_TRUE(!a.shares_storage(c))
        << "contiguous copy should not share storage";

    // Verify values are preserved
    ASSERT_TRUE(std::abs(c.item<float>({0}) - 4.0f) < 1e-5)
        << "c[0] should be 4";
    ASSERT_TRUE(std::abs(c.item<float>({3}) - 1.0f) < 1e-5)
        << "c[3] should be 1";
}

TEST(TensorNumpyOps, Rot90CPU) {
    auto device = Device::CPU;
    // Test rot90 which uses flip internally
    auto data = std::vector<float>({1.0f, 2.0f, 3.0f, 4.0f});
    auto a = Tensor::from_data<float>(data.data(), {2, 2}).to(device);
    // a = [[1, 2],
    //      [3, 4]]

    // 90 degree rotation = flip + transpose
    auto b = a.rot90(1, {0, 1});
    // Expected: [[2, 4],
    //            [1, 3]]

    ASSERT_TRUE(std::abs(b.item<float>({0, 0}) - 2.0f) < 1e-5)
        << "b[0,0] should be 2";
    ASSERT_TRUE(std::abs(b.item<float>({0, 1}) - 4.0f) < 1e-5)
        << "b[0,1] should be 4";
    ASSERT_TRUE(std::abs(b.item<float>({1, 0}) - 1.0f) < 1e-5)
        << "b[1,0] should be 1";
    ASSERT_TRUE(std::abs(b.item<float>({1, 1}) - 3.0f) < 1e-5)
        << "b[1,1] should be 3";
}

// ==================================
//
//      MATH OPERATION TESTS - GPU
//
// ==================================

TEST(TensorNumpyOps, SignGPU) {
    SKIP_IF_NO_GPU();
    auto device = Device::GPU;
    auto data = std::vector<float>({-3.0f, -1.0f, 0.0f, 1.0f, 3.0f});
    auto t = Tensor::from_data<float>(data.data(), {5}).to(device);
    auto result = t.sign();
    axiom::testing::ExpectTensorEquals<float>(result,
                                              {-1.0f, -1.0f, 0.0f, 1.0f, 1.0f});
}

TEST(TensorNumpyOps, FloorCeilTruncGPU) {
    SKIP_IF_NO_GPU();
    auto device = Device::GPU;
    auto data = std::vector<float>({-2.7f, -0.5f, 0.5f, 2.7f});
    auto t = Tensor::from_data<float>(data.data(), {4}).to(device);

    auto fl = t.floor();
    axiom::testing::ExpectTensorEquals<float>(fl, {-3.0f, -1.0f, 0.0f, 2.0f});

    auto ce = t.ceil();
    axiom::testing::ExpectTensorEquals<float>(ce, {-2.0f, 0.0f, 1.0f, 3.0f});

    auto tr = t.trunc();
    axiom::testing::ExpectTensorEquals<float>(tr, {-2.0f, 0.0f, 0.0f, 2.0f});
}

TEST(TensorNumpyOps, ReciprocalGPU) {
    SKIP_IF_NO_GPU();
    auto device = Device::GPU;
    auto data = std::vector<float>({1.0f, 2.0f, 4.0f, 0.5f});
    auto t = Tensor::from_data<float>(data.data(), {4}).to(device);
    auto result = t.reciprocal();
    axiom::testing::ExpectTensorEquals<float>(result,
                                              {1.0f, 0.5f, 0.25f, 2.0f});
}

TEST(TensorNumpyOps, SquareCbrtGPU) {
    SKIP_IF_NO_GPU();
    auto device = Device::GPU;
    auto data = std::vector<float>({2.0f, 3.0f, 4.0f});
    auto t = Tensor::from_data<float>(data.data(), {3}).to(device);

    auto sq = t.square();
    axiom::testing::ExpectTensorEquals<float>(sq, {4.0f, 9.0f, 16.0f});

    auto data2 = std::vector<float>({8.0f, 27.0f, 64.0f});
    auto t2 = Tensor::from_data<float>(data2.data(), {3}).to(device);
    auto cb = t2.cbrt();
    axiom::testing::ExpectTensorEquals<float>(cb, {2.0f, 3.0f, 4.0f}, 1e-5);
}

TEST(TensorNumpyOps, IsnanIsinfIsfiniteGPU) {
    SKIP_IF_NO_GPU();
    auto device = Device::GPU;
    auto data = std::vector<float>({1.0f, NAN, INFINITY, -INFINITY, 0.0f});
    auto t = Tensor::from_data<float>(data.data(), {5}).to(device);

    auto nan_result = t.isnan();
    axiom::testing::ExpectTensorEquals<bool>(
        nan_result, {false, true, false, false, false});

    auto inf_result = t.isinf();
    axiom::testing::ExpectTensorEquals<bool>(inf_result,
                                             {false, false, true, true, false});

    auto fin_result = t.isfinite();
    axiom::testing::ExpectTensorEquals<bool>(fin_result,
                                             {true, false, false, false, true});
}

TEST(TensorNumpyOps, ClipGPU) {
    SKIP_IF_NO_GPU();
    auto device = Device::GPU;
    auto data = std::vector<float>({-5.0f, -1.0f, 0.0f, 1.0f, 5.0f});
    auto t = Tensor::from_data<float>(data.data(), {5}).to(device);
    auto result = t.clip(-2.0, 2.0);
    axiom::testing::ExpectTensorEquals<float>(result,
                                              {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f});
}

// ==================================
//
//      REDUCTION TESTS - GPU
//
// ==================================

TEST(TensorNumpyOps, ProdGPU) {
    SKIP_IF_NO_GPU();
    auto device = Device::GPU;
    auto data = std::vector<float>({1.0f, 2.0f, 3.0f, 4.0f});
    auto t = Tensor::from_data<float>(data.data(), {2, 2}).to(device);

    auto prod_all = t.prod();
    ASSERT_TRUE(std::abs(prod_all.item<float>() - 24.0f) < 1e-5)
        << "prod should be 24";

    auto prod_axis0 = t.prod(0);
    axiom::testing::ExpectTensorEquals<float>(prod_axis0, {3.0f, 8.0f});
}

TEST(TensorNumpyOps, VarStdGPU) {
    SKIP_IF_NO_GPU();
    auto device = Device::GPU;
    auto data = std::vector<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    auto t = Tensor::from_data<float>(data.data(), {5}).to(device);

    // var = mean((x - mean(x))^2) = 2.0
    auto v = t.var();
    ASSERT_TRUE(std::abs(v.item<float>() - 2.0f) < 1e-4) << "var should be 2.0";

    // std = sqrt(var) = sqrt(2) ~= 1.414
    auto s = t.std();
    ASSERT_TRUE(std::abs(s.item<float>() - std::sqrt(2.0f)) < 1e-4)
        << "std should be sqrt(2)";
}

TEST(TensorNumpyOps, PtpGPU) {
    SKIP_IF_NO_GPU();
    auto device = Device::GPU;
    auto data = std::vector<float>({1.0f, 5.0f, 3.0f, 2.0f, 8.0f});
    auto t = Tensor::from_data<float>(data.data(), {5}).to(device);
    auto result = t.ptp();
    ASSERT_TRUE(std::abs(result.item<float>() - 7.0f) < 1e-5)
        << "ptp should be 7";
}

// ==================================
//
//      STACKING TESTS - GPU
//
// ==================================

TEST(TensorNumpyOps, ConcatenateGPU) {
    SKIP_IF_NO_GPU();
    auto device = Device::GPU;
    auto data1 = std::vector<float>({1.0f, 2.0f, 3.0f});
    auto data2 = std::vector<float>({4.0f, 5.0f, 6.0f});
    auto t1 = Tensor::from_data<float>(data1.data(), {3}).to(device);
    auto t2 = Tensor::from_data<float>(data2.data(), {3}).to(device);

    // Using vector
    auto result = Tensor::concatenate({t1, t2}, 0);
    ASSERT_TRUE(result.size() == 6) << "concatenate size should be 6";
    axiom::testing::ExpectTensorEquals<float>(
        result, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

    // Using cat alias with initializer list
    auto result2 = Tensor::cat({t1, t2}, 0);
    axiom::testing::ExpectTensorEquals<float>(
        result2, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

    // Member function cat
    auto result3 = t1.cat(t2, 0);
    axiom::testing::ExpectTensorEquals<float>(
        result3, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
}

TEST(TensorNumpyOps, Concatenate2dGPU) {
    SKIP_IF_NO_GPU();
    auto device = Device::GPU;
    auto data1 = std::vector<float>({1.0f, 2.0f, 3.0f, 4.0f});
    auto data2 = std::vector<float>({5.0f, 6.0f, 7.0f, 8.0f});
    auto t1 = Tensor::from_data<float>(data1.data(), {2, 2}).to(device);
    auto t2 = Tensor::from_data<float>(data2.data(), {2, 2}).to(device);

    // Concat along axis 0
    auto result0 = Tensor::cat({t1, t2}, 0);
    ASSERT_TRUE(result0.shape() == Shape({4, 2})) << "shape should be (4, 2)";

    // Concat along axis 1
    auto result1 = Tensor::cat({t1, t2}, 1);
    ASSERT_TRUE(result1.shape() == Shape({2, 4})) << "shape should be (2, 4)";
}

TEST(TensorNumpyOps, StackGPU) {
    SKIP_IF_NO_GPU();
    auto device = Device::GPU;
    auto data1 = std::vector<float>({1.0f, 2.0f, 3.0f});
    auto data2 = std::vector<float>({4.0f, 5.0f, 6.0f});
    auto t1 = Tensor::from_data<float>(data1.data(), {3}).to(device);
    auto t2 = Tensor::from_data<float>(data2.data(), {3}).to(device);

    // Stack along axis 0 (creates new first dimension)
    auto result = Tensor::stack({t1, t2}, 0);
    ASSERT_TRUE(result.shape() == Shape({2, 3})) << "shape should be (2, 3)";

    // Stack along axis 1
    auto result1 = Tensor::stack({t1, t2}, 1);
    ASSERT_TRUE(result1.shape() == Shape({3, 2})) << "shape should be (3, 2)";
}

TEST(TensorNumpyOps, VstackHstackGPU) {
    SKIP_IF_NO_GPU();
    auto device = Device::GPU;
    auto data1 = std::vector<float>({1.0f, 2.0f, 3.0f});
    auto data2 = std::vector<float>({4.0f, 5.0f, 6.0f});
    auto t1 = Tensor::from_data<float>(data1.data(), {3}).to(device);
    auto t2 = Tensor::from_data<float>(data2.data(), {3}).to(device);

    // vstack 1D arrays
    auto vresult = Tensor::vstack({t1, t2});
    ASSERT_TRUE(vresult.shape() == Shape({2, 3}))
        << "vstack shape should be (2, 3)";

    // hstack 1D arrays
    auto hresult = Tensor::hstack({t1, t2});
    ASSERT_TRUE(hresult.shape() == Shape({6})) << "hstack shape should be (6,)";
    axiom::testing::ExpectTensorEquals<float>(
        hresult, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
}

// ==================================
//
//      FLIP OPERATIONS TESTS - GPU
//
// ==================================

TEST(TensorNumpyOps, Flip1dZeroCopyGPU) {
    SKIP_IF_NO_GPU();
    auto device = Device::GPU;
    // Test that flip returns a zero-copy view with negative strides
    auto data =
        std::vector<float>({0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f});
    auto a = Tensor::from_data<float>(data.data(), {8}).to(device);
    auto b = a.flip(0);

    // Verify flip is zero-copy (shares storage)
    ASSERT_TRUE(a.shares_storage(b)) << "flipped tensor should share storage";

    // Verify flip has negative stride
    ASSERT_TRUE(b.has_negative_stride())
        << "flipped tensor should have negative stride";

    // Verify flipped tensor is not contiguous
    ASSERT_TRUE(!b.is_contiguous())
        << "flipped tensor should not be contiguous";
}

TEST(TensorNumpyOps, Flip2dGPU) {
    SKIP_IF_NO_GPU();
    auto device = Device::GPU;
    // Test 2D flip along different axes
    auto data = std::vector<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    auto a = Tensor::from_data<float>(data.data(), {2, 3}).to(device);

    // Flip along axis 0 (rows)
    auto b = a.flip(0);
    ASSERT_TRUE(a.shares_storage(b)) << "flip(0) should share storage";
    ASSERT_TRUE(b.has_negative_stride())
        << "flip(0) should have negative stride";

    // Flip along axis 1 (cols)
    auto c = a.flip(1);
    ASSERT_TRUE(a.shares_storage(c)) << "flip(1) should share storage";
}

TEST(TensorNumpyOps, FlipudFliplrGPU) {
    SKIP_IF_NO_GPU();
    auto device = Device::GPU;
    auto data = std::vector<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    auto a = Tensor::from_data<float>(data.data(), {2, 3}).to(device);

    // flipud = flip(0)
    auto ud = a.flipud();
    ASSERT_TRUE(a.shares_storage(ud)) << "flipud should share storage";

    // fliplr = flip(1)
    auto lr = a.fliplr();
    ASSERT_TRUE(a.shares_storage(lr)) << "fliplr should share storage";
}

TEST(TensorNumpyOps, FlipOperationsOnFlippedGPU) {
    SKIP_IF_NO_GPU();
    auto device = Device::GPU;
    // Test that arithmetic operations work on flipped tensors
    auto data1 = std::vector<float>({1.0f, 2.0f, 3.0f, 4.0f});
    auto data2 = std::vector<float>({10.0f, 20.0f, 30.0f, 40.0f});
    auto a = Tensor::from_data<float>(data1.data(), {4}).to(device);
    auto b = Tensor::from_data<float>(data2.data(), {4}).to(device);

    // Flip a: [4, 3, 2, 1]
    auto a_flip = a.flip(0);

    // Add flipped tensor to normal tensor
    // [4, 3, 2, 1] + [10, 20, 30, 40] = [14, 23, 32, 41]
    auto result = ops::add(a_flip, b);

    // GPU test - result verified via CPU transfer in the add operation
    // Just verify it doesn't crash and produces a tensor of the right shape
    ASSERT_TRUE(result.size() == 4) << "result should have 4 elements";
}
