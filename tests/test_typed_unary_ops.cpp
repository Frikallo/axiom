#include "axiom_test_utils.hpp"

#include <axiom/axiom.hpp>

using namespace axiom;
using namespace axiom::testing;

template <typename DT>
class TypedUnaryOps : public TypedTensorTest<DT> {};

TYPED_TEST_SUITE(TypedUnaryOps, AllFloatTypes, AxiomTypeName);

TYPED_TEST(TypedUnaryOps, SqrtPerfectSquare) {
    auto a = Tensor::full({2, 3}, 4.0f).astype(this->dtype);
    auto result = ops::sqrt(a);
    auto expected = Tensor::full({2, 3}, 2.0f).astype(this->dtype);
    this->assert_tensors_close(result, expected);
}

TYPED_TEST(TypedUnaryOps, ExpZero) {
    auto a = Tensor::zeros({2, 3}, this->dtype);
    auto result = ops::exp(a);
    auto expected = Tensor::ones({2, 3}, this->dtype);
    this->assert_tensors_close(result, expected);
}

TYPED_TEST(TypedUnaryOps, LogOne) {
    auto a = Tensor::ones({2, 3}, this->dtype);
    auto result = ops::log(a);
    auto expected = Tensor::zeros({2, 3}, this->dtype);
    this->assert_tensors_close(result, expected);
}

TYPED_TEST(TypedUnaryOps, SinZero) {
    auto a = Tensor::zeros({2, 3}, this->dtype);
    auto result = ops::sin(a);
    auto expected = Tensor::zeros({2, 3}, this->dtype);
    this->assert_tensors_close(result, expected);
}

TYPED_TEST(TypedUnaryOps, CosZero) {
    auto a = Tensor::zeros({2, 3}, this->dtype);
    auto result = ops::cos(a);
    auto expected = Tensor::ones({2, 3}, this->dtype);
    this->assert_tensors_close(result, expected);
}

TYPED_TEST(TypedUnaryOps, NegateOnes) {
    auto a = Tensor::ones({2, 3}, this->dtype);
    auto result = ops::negate(a);
    auto expected = Tensor::full({2, 3}, -1.0f).astype(this->dtype);
    this->assert_tensors_close(result, expected);
}

TYPED_TEST(TypedUnaryOps, AbsNegative) {
    auto a = Tensor::full({2, 3}, -1.0f).astype(this->dtype);
    auto result = ops::abs(a);
    auto expected = Tensor::ones({2, 3}, this->dtype);
    this->assert_tensors_close(result, expected);
}

TYPED_TEST(TypedUnaryOps, TanhZero) {
    auto a = Tensor::zeros({2, 3}, this->dtype);
    auto result = ops::tanh(a);
    auto expected = Tensor::zeros({2, 3}, this->dtype);
    this->assert_tensors_close(result, expected);
}
