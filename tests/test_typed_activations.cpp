#include "axiom_test_utils.hpp"

#include <axiom/axiom.hpp>

using namespace axiom;
using namespace axiom::testing;

template <typename DT>
class TypedActivations : public TypedTensorTest<DT> {};

TYPED_TEST_SUITE(TypedActivations, AllFloatTypes, AxiomTypeName);

TYPED_TEST(TypedActivations, ReluPositive) {
    auto a = Tensor::ones({2, 3}, this->dtype);
    auto result = ops::relu(a);
    this->assert_tensors_close(result, a);
}

TYPED_TEST(TypedActivations, ReluNegative) {
    auto a = Tensor::full({2, 3}, -1.0f).astype(this->dtype);
    auto result = ops::relu(a);
    auto expected = Tensor::zeros({2, 3}, this->dtype);
    this->assert_tensors_close(result, expected);
}

TYPED_TEST(TypedActivations, SigmoidZero) {
    auto a = Tensor::zeros({2, 3}, this->dtype);
    auto result = ops::sigmoid(a);
    auto expected = Tensor::full({2, 3}, 0.5f).astype(this->dtype);
    this->assert_tensors_close(result, expected);
}

TYPED_TEST(TypedActivations, SoftmaxUniform) {
    auto a = Tensor::ones({2, 3}, this->dtype);
    auto result = ops::softmax(a, -1);
    // softmax of uniform values = 1/n along that axis
    auto expected =
        Tensor::full({2, 3}, 1.0f / 3.0f).astype(this->dtype);
    this->assert_tensors_close(result, expected);
}

TYPED_TEST(TypedActivations, TanhZero) {
    auto a = Tensor::zeros({2, 3}, this->dtype);
    auto result = ops::tanh(a);
    auto expected = Tensor::zeros({2, 3}, this->dtype);
    this->assert_tensors_close(result, expected);
}

TYPED_TEST(TypedActivations, GeluZero) {
    auto a = Tensor::zeros({2, 3}, this->dtype);
    auto result = ops::gelu(a);
    auto expected = Tensor::zeros({2, 3}, this->dtype);
    this->assert_tensors_close(result, expected);
}
