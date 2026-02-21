#include "axiom_test_utils.hpp"

#include <axiom/axiom.hpp>

using namespace axiom;
using namespace axiom::testing;

template <typename DT> class TypedBinaryOps : public TypedTensorTest<DT> {};

TYPED_TEST_SUITE(TypedBinaryOps, NumericTypes, AxiomTypeName);

TYPED_TEST(TypedBinaryOps, AddOnes) {
    auto a = Tensor::ones({2, 3}, this->dtype);
    auto b = Tensor::ones({2, 3}, this->dtype);
    auto c = a + b;
    auto expected = Tensor::full({2, 3}, 2.0f).astype(this->dtype);
    this->assert_tensors_close(c, expected);
}

TYPED_TEST(TypedBinaryOps, SubtractSelf) {
    auto a = Tensor::ones({2, 3}, this->dtype);
    auto c = a - a;
    auto expected = Tensor::zeros({2, 3}, this->dtype);
    this->assert_tensors_close(c, expected);
}

TYPED_TEST(TypedBinaryOps, MultiplyByOne) {
    auto a = Tensor::full({2, 3}, 2.0f).astype(this->dtype);
    auto one = Tensor::ones({2, 3}, this->dtype);
    auto c = a * one;
    this->assert_tensors_close(c, a);
}

TYPED_TEST(TypedBinaryOps, DivideByOne) {
    if constexpr (TestFixture::DT::is_float()) {
        auto a = Tensor::full({2, 3}, 2.0f).astype(this->dtype);
        auto one = Tensor::ones({2, 3}, this->dtype);
        auto c = a / one;
        this->assert_tensors_close(c, a);
    } else {
        GTEST_SKIP() << "Division skipped for integer types";
    }
}

TYPED_TEST(TypedBinaryOps, AddZero) {
    auto a = Tensor::ones({2, 3}, this->dtype);
    auto z = Tensor::zeros({2, 3}, this->dtype);
    auto c = a + z;
    this->assert_tensors_close(c, a);
}

TYPED_TEST(TypedBinaryOps, MultiplyByZero) {
    auto a = Tensor::ones({2, 3}, this->dtype);
    auto z = Tensor::zeros({2, 3}, this->dtype);
    auto c = a * z;
    this->assert_tensors_close(c, z);
}
