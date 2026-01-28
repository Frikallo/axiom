#pragma once

#include <type_traits>
#include <utility>

#include "axiom/dtype.hpp"
#include "axiom/shape.hpp"
#include "axiom/storage.hpp"
#include "axiom/tensor.hpp"

namespace axiom {
namespace expr {

// Forward declarations for fluent API
template <typename Op, typename Operand> class UnaryExpr;

struct ReluOp;
struct SigmoidOp;
struct TanhActivationOp;
struct GeluOp;
struct SiluOp;
struct SqrtOp;
struct ExpOp;
struct LogOp;
struct AbsOp;
struct NegOp;
struct SquareOp;
struct ReciprocalOp;
struct SinOp;
struct CosOp;
struct TanOp;
struct FloorOp;
struct CeilOp;
struct RoundOp;
struct SignOp;
struct ErfOp;

// CRTP base class for expression templates
// All expressions derive from this using the Curiously Recurring Template
// Pattern
template <typename Derived> class ExprBase {
  public:
    // Access the derived type
    const Derived &derived() const {
        return static_cast<const Derived &>(*this);
    }

    // Common interface - delegated to derived class
    DType dtype() const { return derived().dtype_impl(); }
    Shape shape() const { return derived().shape_impl(); }
    Device device() const { return derived().device_impl(); }

    // Evaluate the expression to a Tensor
    Tensor eval() const { return derived().eval_impl(); }

    // Implicit conversion to Tensor for backward compatibility
    operator Tensor() const { return eval(); }

    // Get size of the result
    size_t size() const { return ShapeUtils::size(shape()); }

    // Get number of dimensions
    size_t ndim() const { return shape().size(); }

    // ========================================================================
    // Fluent unary methods - enable (a + b).relu().sigmoid() syntax
    // These return UnaryExpr wrapping this expression (still lazy!)
    // ========================================================================

    // Activation functions
    inline auto relu() const;
    inline auto sigmoid() const;
    inline auto tanh() const;
    inline auto gelu() const;
    inline auto silu() const;

    // Math functions
    inline auto sqrt() const;
    inline auto exp() const;
    inline auto log() const;
    inline auto abs() const;
    inline auto neg() const;
    inline auto square() const;
    inline auto reciprocal() const;

    // Trigonometric
    inline auto sin() const;
    inline auto cos() const;
    inline auto tan() const;

    // Rounding
    inline auto floor() const;
    inline auto ceil() const;
    inline auto round() const;

    // Other
    inline auto sign() const;
    inline auto erf() const;
};

// TensorRef: Leaf expression that wraps an existing Tensor
// Stores a pointer to the tensor (does not own it)
class TensorRef : public ExprBase<TensorRef> {
    const Tensor *tensor_;

  public:
    explicit TensorRef(const Tensor &t) : tensor_(&t) {}

    // No copy/move restrictions - TensorRef is just a pointer wrapper
    TensorRef(const TensorRef &) = default;
    TensorRef &operator=(const TensorRef &) = default;
    TensorRef(TensorRef &&) = default;
    TensorRef &operator=(TensorRef &&) = default;

    // Expression interface implementation
    DType dtype_impl() const;
    Shape shape_impl() const;
    Device device_impl() const;
    Tensor eval_impl() const;

    // Direct access to the wrapped tensor
    const Tensor &tensor() const { return *tensor_; }
    const Tensor *tensor_ptr() const { return tensor_; }
};

// ScalarExpr: Expression representing a scalar value
// Used for scalar-tensor operations like (tensor + 5.0)
template <typename T> class ScalarExpr : public ExprBase<ScalarExpr<T>> {
    T value_;
    Device device_;

  public:
    explicit ScalarExpr(T value, Device device = Device::CPU)
        : value_(value), device_(device) {}

    // Expression interface implementation
    DType dtype_impl() const { return dtype_of_v<T>; }
    Shape shape_impl() const { return {1}; } // Scalar is shape {1}
    Device device_impl() const { return device_; }

    // Evaluation creates a scalar tensor
    Tensor eval_impl() const;

    // Access the scalar value
    T value() const { return value_; }
};

// Helper to check if a type is an expression
template <typename T, typename = void>
struct is_expression : std::false_type {};

template <typename T>
struct is_expression<T,
                     std::void_t<decltype(std::declval<const T &>().dtype()),
                                 decltype(std::declval<const T &>().shape()),
                                 decltype(std::declval<const T &>().device()),
                                 decltype(std::declval<const T &>().eval())>>
    : std::true_type {};

template <typename T>
inline constexpr bool is_expression_v = is_expression<T>::value;

// Helper to convert scalar types to ScalarExpr
template <typename T> struct to_expr {
    using type = T; // Default: keep as-is
};

template <> struct to_expr<float> {
    using type = ScalarExpr<float>;
};

template <> struct to_expr<double> {
    using type = ScalarExpr<double>;
};

template <> struct to_expr<int> {
    using type = ScalarExpr<int>;
};

template <> struct to_expr<int64_t> {
    using type = ScalarExpr<int64_t>;
};

template <typename T> using to_expr_t = typename to_expr<T>::type;

// Helper to wrap a value as an expression if needed
template <typename T> auto make_expr(T &&val, Device device = Device::CPU) {
    if constexpr (is_expression_v<std::decay_t<T>>) {
        return std::forward<T>(val);
    } else if constexpr (std::is_arithmetic_v<std::decay_t<T>>) {
        return ScalarExpr<std::decay_t<T>>(val, device);
    } else {
        return std::forward<T>(val);
    }
}

} // namespace expr
} // namespace axiom
