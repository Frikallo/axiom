// Tests for extended shape operations

#include "axiom_test_utils.hpp"

using namespace axiom;

TEST(ShapeOpsExtended, MeshgridXY) {
    auto x = Tensor::arange(3);
    auto y = Tensor::arange(4);

    auto grids = ops::meshgrid({x, y}, "xy");
    ASSERT_TRUE(grids.size() == 2) << "Should return 2 grids";
    ASSERT_TRUE(grids[0].shape() == Shape({4, 3}))
        << "X grid shape wrong for xy";
    ASSERT_TRUE(grids[1].shape() == Shape({4, 3}))
        << "Y grid shape wrong for xy";
}

TEST(ShapeOpsExtended, MeshgridIJ) {
    auto x = Tensor::arange(3);
    auto y = Tensor::arange(4);

    auto grids = ops::meshgrid({x, y}, "ij");
    ASSERT_TRUE(grids[0].shape() == Shape({3, 4}))
        << "X grid shape wrong for ij";
    ASSERT_TRUE(grids[1].shape() == Shape({3, 4}))
        << "Y grid shape wrong for ij";
}

TEST(ShapeOpsExtended, MeshgridThree) {
    auto x = Tensor::arange(2);
    auto y = Tensor::arange(3);
    auto z = Tensor::arange(4);

    auto grids = ops::meshgrid({x, y, z}, "ij");
    ASSERT_TRUE(grids.size() == 3) << "Should return 3 grids";
    ASSERT_TRUE(grids[0].shape() == Shape({2, 3, 4})) << "X grid shape wrong";
    ASSERT_TRUE(grids[1].shape() == Shape({2, 3, 4})) << "Y grid shape wrong";
    ASSERT_TRUE(grids[2].shape() == Shape({2, 3, 4})) << "Z grid shape wrong";
}

TEST(ShapeOpsExtended, PadConstant) {
    auto t = Tensor::ones({3, 3});
    auto padded = ops::pad(t, {{1, 1}, {1, 1}}, "constant", 0.0);

    ASSERT_TRUE(padded.shape() == Shape({5, 5})) << "Padded shape wrong";
    // Check corner is 0
    ASSERT_TRUE(padded.item<float>({0, 0}) == 0.0f)
        << "Padding value should be 0";
    // Check center is 1
    ASSERT_TRUE(padded.item<float>({2, 2}) == 1.0f)
        << "Original value should be 1";
}

TEST(ShapeOpsExtended, PadAsymmetric) {
    auto t = Tensor::ones({2, 2});
    auto padded = ops::pad(t, {{1, 2}, {0, 3}}, "constant", 5.0);

    ASSERT_TRUE(padded.shape() == Shape({5, 5}))
        << "Asymmetric padded shape wrong";
    ASSERT_TRUE(padded.item<float>({0, 0}) == 5.0f)
        << "Top padding value wrong";
    ASSERT_TRUE(padded.item<float>({4, 4}) == 5.0f)
        << "Bottom-right padding value wrong";
    ASSERT_TRUE(padded.item<float>({1, 0}) == 1.0f)
        << "Original value position wrong";
}

TEST(ShapeOpsExtended, Pad1D) {
    auto t = Tensor::arange(5);
    auto padded = ops::pad(t, {{2, 2}}, "constant", -1.0);

    ASSERT_TRUE(padded.shape() == Shape({9})) << "1D padded shape wrong";
    ASSERT_TRUE(padded.item<int32_t>({0}) == -1) << "Left padding wrong";
    ASSERT_TRUE(padded.item<int32_t>({2}) == 0) << "First original value wrong";
    ASSERT_TRUE(padded.item<int32_t>({8}) == -1) << "Right padding wrong";
}

TEST(ShapeOpsExtended, Atleast1D) {
    // Scalar to 1D
    auto scalar = Tensor::full({}, 1.0f);
    auto t1d = ops::atleast_1d(scalar);
    ASSERT_TRUE(t1d.ndim() == 1) << "atleast_1d should make scalar 1D";
    ASSERT_TRUE(t1d.shape() == Shape({1})) << "atleast_1d scalar shape wrong";

    // Already 1D
    auto vec = Tensor::arange(5);
    auto vec1d = ops::atleast_1d(vec);
    ASSERT_TRUE(vec1d.shape() == vec.shape())
        << "atleast_1d should not change 1D tensor";

    // Higher dim unchanged
    auto mat = Tensor::ones({3, 4});
    auto mat1d = ops::atleast_1d(mat);
    ASSERT_TRUE(mat1d.shape() == mat.shape())
        << "atleast_1d should not change 2D tensor";
}

TEST(ShapeOpsExtended, Atleast2D) {
    // Scalar to 2D
    auto scalar = Tensor::full({}, 2.0f);
    auto t2d = ops::atleast_2d(scalar);
    ASSERT_TRUE(t2d.ndim() == 2) << "atleast_2d should make scalar 2D";
    ASSERT_TRUE(t2d.shape() == Shape({1, 1}))
        << "atleast_2d scalar shape wrong";

    // 1D to 2D
    auto vec = Tensor::arange(5);
    auto vec2d = ops::atleast_2d(vec);
    ASSERT_TRUE(vec2d.ndim() == 2) << "atleast_2d should make 1D tensor 2D";
    ASSERT_TRUE(vec2d.shape() == Shape({1, 5})) << "atleast_2d 1D shape wrong";

    // Already 2D
    auto mat = Tensor::ones({3, 4});
    auto mat2d = ops::atleast_2d(mat);
    ASSERT_TRUE(mat2d.shape() == mat.shape())
        << "atleast_2d should not change 2D tensor";
}

TEST(ShapeOpsExtended, Atleast3D) {
    // Scalar to 3D
    auto scalar = Tensor::full({}, 3.0f);
    auto t3d = ops::atleast_3d(scalar);
    ASSERT_TRUE(t3d.ndim() == 3) << "atleast_3d should make scalar 3D";
    ASSERT_TRUE(t3d.shape() == Shape({1, 1, 1}))
        << "atleast_3d scalar shape wrong";

    // 1D to 3D
    auto vec = Tensor::arange(5);
    auto vec3d = ops::atleast_3d(vec);
    ASSERT_TRUE(vec3d.ndim() == 3) << "atleast_3d should make 1D tensor 3D";
    ASSERT_TRUE(vec3d.shape() == Shape({1, 5, 1}))
        << "atleast_3d 1D shape wrong";

    // 2D to 3D
    auto mat = Tensor::ones({3, 4});
    auto mat3d = ops::atleast_3d(mat);
    ASSERT_TRUE(mat3d.ndim() == 3) << "atleast_3d should make 2D tensor 3D";
    ASSERT_TRUE(mat3d.shape() == Shape({3, 4, 1}))
        << "atleast_3d 2D shape wrong";
}
