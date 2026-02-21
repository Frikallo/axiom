#include "axiom_test_utils.hpp"

#include <axiom/axiom.hpp>

using namespace axiom;
using namespace axiom::testing;

template <typename DT>
class TypedReductions : public TypedTensorTest<DT> {};

TYPED_TEST_SUITE(TypedReductions, NumericTypes, AxiomTypeName);

TYPED_TEST(TypedReductions, SumOnes) {
    auto a = Tensor::ones({2, 3}, this->dtype);
    auto result = ops::sum(a);
    auto expected = Tensor::full({}, 6.0f).astype(this->dtype);
    this->assert_tensors_close(result, expected);
}

TYPED_TEST(TypedReductions, MeanOnes) {
    if constexpr (TestFixture::DT::is_float()) {
        auto a = Tensor::ones({2, 3}, this->dtype);
        auto result = ops::mean(a);
        auto expected = Tensor::full({}, 1.0f).astype(this->dtype);
        this->assert_tensors_close(result, expected);
    } else {
        GTEST_SKIP() << "Mean skipped for integer types";
    }
}

TYPED_TEST(TypedReductions, MaxOnes) {
    auto a = Tensor::ones({2, 3}, this->dtype);
    auto result = ops::max(a);
    auto expected = Tensor::full({}, 1.0f).astype(this->dtype);
    this->assert_tensors_close(result, expected);
}

TYPED_TEST(TypedReductions, MinOnes) {
    auto a = Tensor::ones({2, 3}, this->dtype);
    auto result = ops::min(a);
    auto expected = Tensor::full({}, 1.0f).astype(this->dtype);
    this->assert_tensors_close(result, expected);
}

TYPED_TEST(TypedReductions, SumAxis) {
    auto a = Tensor::ones({2, 3}, this->dtype);
    auto result = ops::sum(a, {1});
    ASSERT_EQ(result.shape(), Shape({2}));
    auto expected = Tensor::full({2}, 3.0f).astype(this->dtype);
    this->assert_tensors_close(result, expected);
}

TYPED_TEST(TypedReductions, ProdOnes) {
    auto a = Tensor::ones({2, 3}, this->dtype);
    auto result = ops::prod(a);
    auto expected = Tensor::full({}, 1.0f).astype(this->dtype);
    this->assert_tensors_close(result, expected);
}
