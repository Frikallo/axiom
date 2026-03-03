#include "axiom_test_utils.hpp"
#include <axiom/axiom.hpp>

using namespace axiom;

TEST(SortTopk, Sort1D) {
    float data[] = {3.0f, 1.0f, 4.0f, 1.0f, 5.0f, 9.0f};
    auto t = Tensor::from_data(data, {6});
    auto sorted = ops::sort(t);
    auto s = sorted.typed_data<float>();
    ASSERT_FLOAT_EQ(s[0], 1.0f);
    ASSERT_FLOAT_EQ(s[1], 1.0f);
    ASSERT_FLOAT_EQ(s[2], 3.0f);
    ASSERT_FLOAT_EQ(s[3], 4.0f);
    ASSERT_FLOAT_EQ(s[4], 5.0f);
    ASSERT_FLOAT_EQ(s[5], 9.0f);
}

TEST(SortTopk, SortDescending) {
    float data[] = {3.0f, 1.0f, 4.0f};
    auto t = Tensor::from_data(data, {3});
    auto sorted = ops::sort(t, -1, true);
    auto s = sorted.typed_data<float>();
    ASSERT_FLOAT_EQ(s[0], 4.0f);
    ASSERT_FLOAT_EQ(s[1], 3.0f);
    ASSERT_FLOAT_EQ(s[2], 1.0f);
}

TEST(SortTopk, Sort2DAxis0) {
    float data[] = {3.0f, 1.0f, 2.0f, 4.0f};
    auto t = Tensor::from_data(data, {2, 2});
    auto sorted = ops::sort(t, 0);
    auto s = sorted.typed_data<float>();
    // Column 0: [3,2] → [2,3], Column 1: [1,4] → [1,4]
    ASSERT_FLOAT_EQ(s[0], 2.0f);
    ASSERT_FLOAT_EQ(s[1], 1.0f);
    ASSERT_FLOAT_EQ(s[2], 3.0f);
    ASSERT_FLOAT_EQ(s[3], 4.0f);
}

TEST(SortTopk, Sort2DAxis1) {
    float data[] = {3.0f, 1.0f, 2.0f, 4.0f};
    auto t = Tensor::from_data(data, {2, 2});
    auto sorted = ops::sort(t, 1);
    auto s = sorted.typed_data<float>();
    // Row 0: [3,1] → [1,3], Row 1: [2,4] → [2,4]
    ASSERT_FLOAT_EQ(s[0], 1.0f);
    ASSERT_FLOAT_EQ(s[1], 3.0f);
    ASSERT_FLOAT_EQ(s[2], 2.0f);
    ASSERT_FLOAT_EQ(s[3], 4.0f);
}

TEST(SortTopk, Argsort1D) {
    float data[] = {3.0f, 1.0f, 4.0f};
    auto t = Tensor::from_data(data, {3});
    auto indices = ops::argsort(t);
    auto idx = indices.typed_data<int64_t>();
    ASSERT_EQ(idx[0], 1); // 1.0
    ASSERT_EQ(idx[1], 0); // 3.0
    ASSERT_EQ(idx[2], 2); // 4.0
}

TEST(SortTopk, ArgsortDescending) {
    float data[] = {3.0f, 1.0f, 4.0f};
    auto t = Tensor::from_data(data, {3});
    auto indices = ops::argsort(t, -1, true);
    auto idx = indices.typed_data<int64_t>();
    ASSERT_EQ(idx[0], 2); // 4.0
    ASSERT_EQ(idx[1], 0); // 3.0
    ASSERT_EQ(idx[2], 1); // 1.0
}

TEST(SortTopk, TopK1D) {
    float data[] = {3.0f, 1.0f, 4.0f, 1.0f, 5.0f};
    auto t = Tensor::from_data(data, {5});
    auto [values, indices] = ops::topk(t, 3);
    auto v = values.typed_data<float>();
    auto idx = indices.typed_data<int64_t>();
    // Top 3 largest: 5, 4, 3
    ASSERT_FLOAT_EQ(v[0], 5.0f);
    ASSERT_FLOAT_EQ(v[1], 4.0f);
    ASSERT_FLOAT_EQ(v[2], 3.0f);
    ASSERT_EQ(idx[0], 4);
    ASSERT_EQ(idx[1], 2);
    ASSERT_EQ(idx[2], 0);
}

TEST(SortTopk, TopKSmallest) {
    float data[] = {3.0f, 1.0f, 4.0f, 1.0f, 5.0f};
    auto t = Tensor::from_data(data, {5});
    auto [values, indices] = ops::topk(t, 2, -1, false);
    auto v = values.typed_data<float>();
    // Smallest 2: 1, 1
    ASSERT_FLOAT_EQ(v[0], 1.0f);
    ASSERT_FLOAT_EQ(v[1], 1.0f);
}

TEST(SortTopk, TopK2D) {
    float data[] = {3.0f, 1.0f, 4.0f, 2.0f, 5.0f, 0.0f};
    auto t = Tensor::from_data(data, {2, 3});
    auto [values, indices] = ops::topk(t, 2, -1);
    ASSERT_EQ(values.shape()[0], 2);
    ASSERT_EQ(values.shape()[1], 2);
    // Row 0: [3,1,4] → top2 = [4,3]
    auto v = values.typed_data<float>();
    ASSERT_FLOAT_EQ(v[0], 4.0f);
    ASSERT_FLOAT_EQ(v[1], 3.0f);
    // Row 1: [2,5,0] → top2 = [5,2]
    ASSERT_FLOAT_EQ(v[2], 5.0f);
    ASSERT_FLOAT_EQ(v[3], 2.0f);
}

TEST(SortTopk, SortNonContiguous) {
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    auto t = Tensor::from_data(data, {2, 3});
    auto transposed = t.transpose(); // (3, 2)
    auto sorted = ops::sort(transposed, -1);
    // Each row of transposed: [1,4], [2,5], [3,6] — all already sorted
    ASSERT_EQ(sorted.shape()[0], 3);
    ASSERT_EQ(sorted.shape()[1], 2);
}
