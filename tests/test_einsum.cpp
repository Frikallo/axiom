// Tests for einsum (Einstein summation) operations

#include "axiom_test_utils.hpp"

using namespace axiom;

TEST(Einsum, Matmul) {
    auto A = Tensor::randn({2, 3});
    auto B = Tensor::randn({3, 4});

    auto result = einops::einsum("ij,jk->ik", {A, B});
    auto expected = ops::matmul(A, B);

    ASSERT_TRUE(result.shape() == expected.shape())
        << "einsum matmul shape mismatch";
    ASSERT_TRUE(result.allclose(expected, 1e-4, 1e-4))
        << "einsum matmul values mismatch";
}

TEST(Einsum, MatmulLarger) {
    auto A = Tensor::randn({10, 20});
    auto B = Tensor::randn({20, 15});

    auto result = einops::einsum("ij,jk->ik", {A, B});
    auto expected = ops::matmul(A, B);

    ASSERT_TRUE(result.shape() == expected.shape())
        << "einsum large matmul shape mismatch";
    ASSERT_TRUE(result.allclose(expected, 1e-4, 1e-4))
        << "einsum large matmul values mismatch";
}

TEST(Einsum, Transpose) {
    auto A = Tensor::randn({3, 4});

    auto result = einops::einsum("ij->ji", {A});
    auto expected = A.transpose();

    ASSERT_TRUE(result.shape() == expected.shape())
        << "einsum transpose shape mismatch";
    ASSERT_TRUE(result.allclose(expected))
        << "einsum transpose values mismatch";
}

TEST(Einsum, Trace) {
    auto A = Tensor::randn({4, 4});

    auto result = einops::einsum("ii->", {A});
    auto expected = A.trace();

    ASSERT_TRUE(result.shape() == expected.shape())
        << "einsum trace shape mismatch";
    ASSERT_TRUE(result.allclose(expected, 1e-4, 1e-4))
        << "einsum trace values mismatch";
}

TEST(Einsum, SumAll) {
    auto A = Tensor::randn({3, 4});

    auto result = einops::einsum("ij->", {A});
    auto expected = ops::sum(A);

    ASSERT_TRUE(result.shape() == expected.shape())
        << "einsum sum all shape mismatch";
    ASSERT_TRUE(result.allclose(expected, 1e-4, 1e-4))
        << "einsum sum all values mismatch";
}

TEST(Einsum, SumAxis) {
    auto A = Tensor::randn({2, 3, 4});

    auto result = einops::einsum("ijk->j", {A});
    auto expected = ops::sum(A, {0, 2});

    ASSERT_TRUE(result.shape() == expected.shape())
        << "einsum sum axis shape mismatch";
    ASSERT_TRUE(result.allclose(expected, 1e-4, 1e-4))
        << "einsum sum axis values mismatch";
}

TEST(Einsum, SumKeepdim) {
    auto A = Tensor::randn({3, 4});

    // Sum over second axis, keep first
    auto result = einops::einsum("ij->i", {A});
    auto expected = ops::sum(A, {1});

    ASSERT_TRUE(result.shape() == expected.shape())
        << "einsum sum keepdim shape mismatch";
    ASSERT_TRUE(result.allclose(expected, 1e-4, 1e-4))
        << "einsum sum keepdim values mismatch";
}

TEST(Einsum, Elementwise) {
    auto A = Tensor::randn({3, 4});
    auto B = Tensor::randn({3, 4});

    auto result = einops::einsum("ij,ij->ij", {A, B});
    auto expected = ops::multiply(A, B);

    ASSERT_TRUE(result.shape() == expected.shape())
        << "einsum elementwise shape mismatch";
    ASSERT_TRUE(result.allclose(expected))
        << "einsum elementwise values mismatch";
}

TEST(Einsum, BatchedMatmul) {
    auto A = Tensor::randn({2, 3, 4});
    auto B = Tensor::randn({2, 4, 5});

    auto result = einops::einsum("bij,bjk->bik", {A, B});

    ASSERT_TRUE(result.shape() == Shape({2, 3, 5}))
        << "einsum batched matmul shape mismatch";

    // Verify by computing batch elements separately
    for (size_t b = 0; b < 2; ++b) {
        auto A_b = A.slice({{b, b + 1}}).squeeze(0);
        auto B_b = B.slice({{b, b + 1}}).squeeze(0);
        auto expected_b = ops::matmul(A_b, B_b);

        auto result_b = result.slice({{b, b + 1}}).squeeze(0);
        ASSERT_TRUE(result_b.allclose(expected_b, 1e-4, 1e-4))
            << "einsum batched matmul batch element mismatch";
    }
}

TEST(Einsum, DotProduct) {
    auto a = Tensor::randn({5});
    auto b = Tensor::randn({5});

    // Dot product: i,i->
    auto result = einops::einsum("i,i->", {a, b});
    auto expected = ops::sum(ops::multiply(a, b));

    ASSERT_TRUE(result.shape() == expected.shape())
        << "einsum dot product shape mismatch";
    ASSERT_TRUE(result.allclose(expected, 1e-4, 1e-4))
        << "einsum dot product value mismatch";
}
