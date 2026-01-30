#include "axiom/expr/base.hpp"
#include "axiom/expr/binary.hpp"
#include "axiom/expr/matmul.hpp"
#include "axiom/expr/traits.hpp"
#include "axiom/expr/unary.hpp"
#include "axiom/operations.hpp"
#include "axiom/tensor.hpp"

namespace axiom {
namespace expr {

// ============================================================================
// TensorRef implementation
// ============================================================================

DType TensorRef::dtype_impl() const { return tensor_->dtype(); }

Shape TensorRef::shape_impl() const { return tensor_->shape(); }

Device TensorRef::device_impl() const { return tensor_->device(); }

Tensor TensorRef::eval_impl() const { return *tensor_; }

// ============================================================================
// ScalarExpr implementation
// ============================================================================

template <typename T> Tensor ScalarExpr<T>::eval_impl() const {
    return Tensor::full({1}, value_, device_);
}

// Explicit instantiations for common scalar types
template Tensor ScalarExpr<float>::eval_impl() const;
template Tensor ScalarExpr<double>::eval_impl() const;
template Tensor ScalarExpr<int>::eval_impl() const;
template Tensor ScalarExpr<int64_t>::eval_impl() const;

// Note: BinaryExpr, UnaryExpr, and MatMulExpr eval_impl() are defined
// inline in their respective headers to support arbitrary expression tree
// compositions without requiring explicit template instantiations.

} // namespace expr
} // namespace axiom
