//=============================================================================
// tests/test_shape_utils.cpp - Tests for ShapeUtils iteration helpers
//=============================================================================

#include "axiom_test_utils.hpp"
#include <vector>

using namespace axiom;

//=============================================================================
// increment_coords
//=============================================================================

TEST(ShapeUtils, IncrementCoords2D) {
    Shape shape = {2, 3};
    std::vector<size_t> coords = {0, 0};

    // Walk every element of a 2x3 grid and record the sequence
    std::vector<std::vector<size_t>> visited;
    do {
        visited.push_back(coords);
    } while (ShapeUtils::increment_coords(coords, shape));

    ASSERT_EQ(visited.size(), 6u);
    EXPECT_EQ(visited[0], (std::vector<size_t>{0, 0}));
    EXPECT_EQ(visited[1], (std::vector<size_t>{0, 1}));
    EXPECT_EQ(visited[2], (std::vector<size_t>{0, 2}));
    EXPECT_EQ(visited[3], (std::vector<size_t>{1, 0}));
    EXPECT_EQ(visited[4], (std::vector<size_t>{1, 1}));
    EXPECT_EQ(visited[5], (std::vector<size_t>{1, 2}));

    // coords should be back to all-zeros after wrap
    EXPECT_EQ(coords, (std::vector<size_t>{0, 0}));
}

TEST(ShapeUtils, IncrementCoords3D) {
    Shape shape = {2, 2, 2};
    std::vector<size_t> coords = {0, 0, 0};

    size_t count = 0;
    do {
        ++count;
    } while (ShapeUtils::increment_coords(coords, shape));

    EXPECT_EQ(count, 8u);
}

TEST(ShapeUtils, IncrementCoords1D) {
    Shape shape = {5};
    std::vector<size_t> coords = {0};

    std::vector<size_t> indices;
    do {
        indices.push_back(coords[0]);
    } while (ShapeUtils::increment_coords(coords, shape));

    ASSERT_EQ(indices.size(), 5u);
    for (size_t i = 0; i < 5; ++i) {
        EXPECT_EQ(indices[i], i);
    }
}

TEST(ShapeUtils, IncrementCoordsScalar) {
    // 0-D: single element, first call returns false
    Shape shape = {};
    std::vector<size_t> coords = {};

    EXPECT_FALSE(ShapeUtils::increment_coords(coords, shape));
}

TEST(ShapeUtils, IncrementCoordsSingleElement) {
    Shape shape = {1, 1, 1};
    std::vector<size_t> coords = {0, 0, 0};

    // Only one element — first increment wraps
    EXPECT_FALSE(ShapeUtils::increment_coords(coords, shape));
}

//=============================================================================
// broadcast_strides
//=============================================================================

TEST(ShapeUtils, BroadcastStridesIdentity) {
    // Same shape, strides pass through
    Shape input_shape = {3, 4};
    Strides input_strides = {16, 4}; // row-major float32
    Shape result_shape = {3, 4};

    auto out =
        ShapeUtils::broadcast_strides(input_shape, input_strides, result_shape);

    ASSERT_EQ(out.size(), 2u);
    EXPECT_EQ(out[0], 16);
    EXPECT_EQ(out[1], 4);
}

TEST(ShapeUtils, BroadcastStridesDimExpansion) {
    // (4,) broadcast to (3, 4)
    Shape input_shape = {4};
    Strides input_strides = {4};
    Shape result_shape = {3, 4};

    auto out =
        ShapeUtils::broadcast_strides(input_shape, input_strides, result_shape);

    ASSERT_EQ(out.size(), 2u);
    EXPECT_EQ(out[0], 0); // new leading dim → stride 0
    EXPECT_EQ(out[1], 4); // trailing dim passes through
}

TEST(ShapeUtils, BroadcastStridesSizeOneDim) {
    // (1, 4) broadcast to (3, 4) — size-1 dim gets stride 0
    Shape input_shape = {1, 4};
    Strides input_strides = {16, 4};
    Shape result_shape = {3, 4};

    auto out =
        ShapeUtils::broadcast_strides(input_shape, input_strides, result_shape);

    ASSERT_EQ(out.size(), 2u);
    EXPECT_EQ(out[0], 0); // broadcast dim
    EXPECT_EQ(out[1], 4);
}

TEST(ShapeUtils, BroadcastStridesMultipleBroadcast) {
    // (1, 3, 1) broadcast to (2, 3, 4)
    Shape input_shape = {1, 3, 1};
    Strides input_strides = {12, 4, 4};
    Shape result_shape = {2, 3, 4};

    auto out =
        ShapeUtils::broadcast_strides(input_shape, input_strides, result_shape);

    ASSERT_EQ(out.size(), 3u);
    EXPECT_EQ(out[0], 0); // size-1 → broadcast
    EXPECT_EQ(out[1], 4); // size-3 → passes through
    EXPECT_EQ(out[2], 0); // size-1 → broadcast
}

TEST(ShapeUtils, BroadcastStridesFewerDims) {
    // (5,) broadcast to (2, 3, 5)
    Shape input_shape = {5};
    Strides input_strides = {4};
    Shape result_shape = {2, 3, 5};

    auto out =
        ShapeUtils::broadcast_strides(input_shape, input_strides, result_shape);

    ASSERT_EQ(out.size(), 3u);
    EXPECT_EQ(out[0], 0);
    EXPECT_EQ(out[1], 0);
    EXPECT_EQ(out[2], 4);
}

//=============================================================================
// axis_outer_inner
//=============================================================================

TEST(ShapeUtils, AxisOuterInner3DMiddle) {
    Shape shape = {2, 3, 4};
    auto s = ShapeUtils::axis_outer_inner(shape, 1);

    EXPECT_EQ(s.outer, 2u);
    EXPECT_EQ(s.axis, 3u);
    EXPECT_EQ(s.inner, 4u);
}

TEST(ShapeUtils, AxisOuterInnerFirstAxis) {
    Shape shape = {5, 3, 4};
    auto s = ShapeUtils::axis_outer_inner(shape, 0);

    EXPECT_EQ(s.outer, 1u);
    EXPECT_EQ(s.axis, 5u);
    EXPECT_EQ(s.inner, 12u);
}

TEST(ShapeUtils, AxisOuterInnerLastAxis) {
    Shape shape = {2, 3, 4};
    auto s = ShapeUtils::axis_outer_inner(shape, 2);

    EXPECT_EQ(s.outer, 6u);
    EXPECT_EQ(s.axis, 4u);
    EXPECT_EQ(s.inner, 1u);
}

TEST(ShapeUtils, AxisOuterInner1D) {
    Shape shape = {7};
    auto s = ShapeUtils::axis_outer_inner(shape, 0);

    EXPECT_EQ(s.outer, 1u);
    EXPECT_EQ(s.axis, 7u);
    EXPECT_EQ(s.inner, 1u);
}

TEST(ShapeUtils, AxisOuterInner4D) {
    Shape shape = {2, 3, 4, 5};
    auto s = ShapeUtils::axis_outer_inner(shape, 2);

    EXPECT_EQ(s.outer, 6u); // 2 * 3
    EXPECT_EQ(s.axis, 4u);
    EXPECT_EQ(s.inner, 5u);
}

TEST(ShapeUtils, AxisOuterInnerProductEqualsTotal) {
    // outer * axis * inner should always equal total element count
    Shape shape = {3, 5, 7, 2};
    for (int ax = 0; ax < 4; ++ax) {
        auto s = ShapeUtils::axis_outer_inner(shape, ax);
        EXPECT_EQ(s.outer * s.axis * s.inner, ShapeUtils::size(shape));
    }
}
