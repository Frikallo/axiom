#include "axiom_test_utils.hpp"

#include <axiom/axiom.hpp>

using namespace axiom;
using namespace axiom::testing;

template <typename DT> class TypedComparisons : public TypedTensorTest<DT> {};

TYPED_TEST_SUITE(TypedComparisons, NumericTypes, AxiomTypeName);

TYPED_TEST(TypedComparisons, EqualSelf) {
    auto a = Tensor::ones({2, 3}, this->dtype);
    auto result = ops::equal(a, a);
    ASSERT_EQ(result.dtype(), DType::Bool);
    auto all_true = ops::all(result);
    bool val = all_true.template item<bool>();
    ASSERT_TRUE(val) << "equal(a, a) should be all true for "
                     << TestFixture::DT::name();
}

TYPED_TEST(TypedComparisons, NotEqualDifferent) {
    auto a = Tensor::zeros({2, 3}, this->dtype);
    auto b = Tensor::ones({2, 3}, this->dtype);
    auto result = ops::not_equal(a, b);
    ASSERT_EQ(result.dtype(), DType::Bool);
    auto all_true = ops::all(result);
    bool val = all_true.template item<bool>();
    ASSERT_TRUE(val) << "not_equal(zeros, ones) should be all true for "
                     << TestFixture::DT::name();
}

TYPED_TEST(TypedComparisons, LessThanGreater) {
    auto a = Tensor::zeros({2, 3}, this->dtype);
    auto b = Tensor::ones({2, 3}, this->dtype);
    auto result = ops::less(a, b);
    ASSERT_EQ(result.dtype(), DType::Bool);
    auto all_true = ops::all(result);
    bool val = all_true.template item<bool>();
    ASSERT_TRUE(val) << "less(zeros, ones) should be all true for "
                     << TestFixture::DT::name();
}

TYPED_TEST(TypedComparisons, GreaterThanLess) {
    auto a = Tensor::ones({2, 3}, this->dtype);
    auto b = Tensor::zeros({2, 3}, this->dtype);
    auto result = ops::greater(a, b);
    ASSERT_EQ(result.dtype(), DType::Bool);
    auto all_true = ops::all(result);
    bool val = all_true.template item<bool>();
    ASSERT_TRUE(val) << "greater(ones, zeros) should be all true for "
                     << TestFixture::DT::name();
}
