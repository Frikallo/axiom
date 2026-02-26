#pragma once

#include <cmath>
#include <type_traits>

#include "axiom/error.hpp"
#include "axiom/operations.hpp"

namespace axiom {
namespace backends {
namespace cpu {

// ============================================================================
// CPU Binary Operation Base Class
// ============================================================================

template <typename Func> class CPUBinaryOperation : public ops::Operation {
  private:
    ops::OpType op_type_;
    std::string name_;
    Func func_;

  public:
    CPUBinaryOperation(ops::OpType op_type, const std::string &name, Func func)
        : op_type_(op_type), name_(name), func_(func) {}

    ops::OpType type() const override { return op_type_; }
    std::string name() const override { return name_; }
    Device device() const override { return Device::CPU; }

    bool supports_binary(const Tensor &lhs, const Tensor &rhs) const override {
        return ops::are_broadcastable(lhs.shape(), rhs.shape());
    }

    Tensor execute_binary(const Tensor &lhs, const Tensor &rhs) const override;

    void execute_binary_inplace(Tensor &lhs, const Tensor &rhs) const override;

  private:
    template <typename T>
    void execute_binary_typed(const Tensor &lhs, const Tensor &rhs,
                              Tensor &result) const;

    template <typename T>
    void
    execute_binary_broadcast(const Tensor &lhs, const Tensor &rhs,
                             Tensor &result,
                             const ops::BroadcastInfo &broadcast_info) const;

    template <typename T>
    void execute_binary_same_shape(const Tensor &lhs, const Tensor &rhs,
                                   Tensor &result) const;

    template <typename T>
    void execute_inplace_typed(Tensor &lhs, const Tensor &rhs) const;

    template <typename T>
    void execute_inplace_broadcast(Tensor &lhs, const Tensor &rhs) const;

    template <typename InputT, typename OutputT>
    void execute_broadcast_loop(const InputT *lhs_data, const InputT *rhs_data,
                                OutputT *result_data, const Shape &lhs_shape,
                                const Shape &rhs_shape,
                                const Shape &result_shape,
                                const Strides &lhs_strides,
                                const Strides &rhs_strides) const;
};

// ============================================================================
// CPU Complex Binary Operation Class (dedicated for complex arithmetic)
// ============================================================================

class CPUComplexBinaryOperation : public ops::Operation {
  private:
    ops::OpType op_type_;
    std::string name_;

  public:
    CPUComplexBinaryOperation(ops::OpType op_type, const std::string &name)
        : op_type_(op_type), name_(name) {}

    ops::OpType type() const override { return op_type_; }
    std::string name() const override { return name_; }
    Device device() const override { return Device::CPU; }

    bool supports_binary(const Tensor &lhs, const Tensor &rhs) const override {
        return ops::are_broadcastable(lhs.shape(), rhs.shape());
    }

    Tensor execute_binary(const Tensor &lhs, const Tensor &rhs) const override;

  private:
    template <typename T>
    void execute_complex_typed(const Tensor &lhs, const Tensor &rhs,
                               Tensor &result) const;
};

// ============================================================================
// CPU Unary Operation Base Class
// ============================================================================

template <typename Func> class CPUUnaryOperation : public ops::Operation {
  private:
    ops::OpType op_type_;
    std::string name_;
    Func func_;

  public:
    CPUUnaryOperation(ops::OpType op_type, const std::string &name, Func func)
        : op_type_(op_type), name_(name), func_(func) {}

    ops::OpType type() const override { return op_type_; }
    std::string name() const override { return name_; }
    Device device() const override { return Device::CPU; }

    Tensor execute_binary(const Tensor &lhs, const Tensor &rhs) const override {
        (void)lhs;
        (void)rhs;
        throw RuntimeError::internal(
            "execute_binary called on unary operation");
    }

    Tensor execute_unary(const Tensor &input) const override;

  private:
    template <typename T>
    void execute_unary_typed(const Tensor &input, Tensor &result) const;
};

// ============================================================================
// CPU Reduction Operation Base Class
// ============================================================================

template <typename Func> class CPUReductionOperation : public ops::Operation {
  private:
    ops::OpType op_type_;
    std::string name_;
    Func func_;

  public:
    CPUReductionOperation(ops::OpType op_type, const std::string &name,
                          Func func)
        : op_type_(op_type), name_(name), func_(func) {}

    ops::OpType type() const override { return op_type_; }
    std::string name() const override { return name_; }
    Device device() const override { return Device::CPU; }

    Tensor execute_binary(const Tensor &lhs, const Tensor &rhs) const override {
        (void)lhs;
        (void)rhs;
        throw RuntimeError::internal(
            "execute_binary called on reduction operation");
    }

    Tensor execute_unary(const Tensor &input) const override {
        // This will be called by the high-level reduction function
        // We will need a new execute_reduction method
        (void)input;
        throw RuntimeError::internal(
            "Use execute_reduction for reduction operations");
    }

    Tensor execute_reduction(const Tensor &input, const std::vector<int> &axis,
                             bool keep_dims) const override;

  private:
    template <typename T>
    Tensor execute_reduction_typed(const Tensor &input,
                                   const std::vector<int> &axis,
                                   bool keep_dims) const;

    template <typename T>
    static void reduction_recursive_helper(const Tensor &input, Tensor &result,
                                           const std::vector<int> &axes,
                                           std::vector<size_t> &current_coords,
                                           int current_dim, const Func &func,
                                           bool keep_dims);
};

// ============================================================================
// Operation Function Objects
// ============================================================================

// Arithmetic operations
struct AddFunc {
    template <typename T> T operator()(const T &a, const T &b) const {
        return a + b;
    }
};

struct SubtractFunc {
    template <typename T> T operator()(const T &a, const T &b) const {
        return a - b;
    }
};

struct MultiplyFunc {
    template <typename T> T operator()(const T &a, const T &b) const {
        return a * b;
    }
};

struct DivideFunc {
    template <typename T> T operator()(const T &a, const T &b) const {
        if constexpr (std::is_same_v<T, bool>) {
            return static_cast<bool>(static_cast<int>(a) / static_cast<int>(b));
        } else {
            return a / b;
        }
    }
};

struct PowerFunc {
    template <typename T> T operator()(const T &a, const T &b) const {
        if constexpr (std::is_integral_v<T>) {
            return static_cast<T>(
                std::pow(static_cast<double>(a), static_cast<double>(b)));
        } else {
            return static_cast<T>(std::pow(a, b));
        }
    }
};

struct ModuloFunc {
    template <typename T> T operator()(const T &a, const T &b) const {
        if constexpr (std::is_same_v<T, bool>) {
            return static_cast<bool>(static_cast<int>(a) % static_cast<int>(b));
        } else if constexpr (std::is_integral_v<T>) {
            return a % b;
        } else {
            return static_cast<T>(std::fmod(a, b));
        }
    }
};

// Comparison operations
struct EqualFunc {
    template <typename T> bool operator()(const T &a, const T &b) const {
        return a == b;
    }
};

struct NotEqualFunc {
    template <typename T> bool operator()(const T &a, const T &b) const {
        return a != b;
    }
};

struct LessFunc {
    template <typename T> bool operator()(const T &a, const T &b) const {
        return a < b;
    }
};

struct LessEqualFunc {
    template <typename T> bool operator()(const T &a, const T &b) const {
        return a <= b;
    }
};

struct GreaterFunc {
    template <typename T> bool operator()(const T &a, const T &b) const {
        return a > b;
    }
};

struct GreaterEqualFunc {
    template <typename T> bool operator()(const T &a, const T &b) const {
        return a >= b;
    }
};

// Logical operations
struct LogicalAndFunc {
    template <typename T> bool operator()(const T &a, const T &b) const {
        return static_cast<bool>(a) && static_cast<bool>(b);
    }
};

struct LogicalOrFunc {
    template <typename T> bool operator()(const T &a, const T &b) const {
        return static_cast<bool>(a) || static_cast<bool>(b);
    }
};

struct LogicalXorFunc {
    template <typename T> bool operator()(const T &a, const T &b) const {
        return static_cast<bool>(a) != static_cast<bool>(b);
    }
};

// Bitwise operations (only for integer types)
// Use if constexpr to avoid compile errors when instantiated for non-integer
// types (runtime checks in execute_binary prevent non-integer types from
// reaching here)
struct BitwiseAndFunc {
    template <typename T> T operator()(const T &a, const T &b) const {
        if constexpr (std::is_integral_v<T>) {
            return a & b;
        } else {
            return T{}; // Never reached at runtime
        }
    }
};

struct BitwiseOrFunc {
    template <typename T> T operator()(const T &a, const T &b) const {
        if constexpr (std::is_integral_v<T>) {
            return a | b;
        } else {
            return T{};
        }
    }
};

struct BitwiseXorFunc {
    template <typename T> T operator()(const T &a, const T &b) const {
        if constexpr (std::is_integral_v<T>) {
            return a ^ b;
        } else {
            return T{};
        }
    }
};

struct LeftShiftFunc {
    template <typename T> T operator()(const T &a, const T &b) const {
        if constexpr (std::is_integral_v<T> && !std::is_same_v<T, bool>) {
            return a << b;
        } else {
            return T{};
        }
    }
};

struct RightShiftFunc {
    template <typename T> T operator()(const T &a, const T &b) const {
        if constexpr (std::is_integral_v<T> && !std::is_same_v<T, bool>) {
            return a >> b;
        } else {
            return T{};
        }
    }
};

// Math operations
struct MaximumFunc {
    template <typename T> T operator()(const T &a, const T &b) const {
        return std::max(a, b);
    }
};

struct MinimumFunc {
    template <typename T> T operator()(const T &a, const T &b) const {
        return std::min(a, b);
    }
};

struct Atan2Func {
    template <typename T> T operator()(const T &a, const T &b) const {
        if constexpr (std::is_integral_v<T>) {
            return static_cast<T>(
                std::atan2(static_cast<double>(a), static_cast<double>(b)));
        } else {
            return static_cast<T>(std::atan2(a, b));
        }
    }
};

struct HypotFunc {
    template <typename T> T operator()(const T &a, const T &b) const {
        if constexpr (std::is_integral_v<T>) {
            return static_cast<T>(
                std::hypot(static_cast<double>(a), static_cast<double>(b)));
        } else {
            return static_cast<T>(std::hypot(a, b));
        }
    }
};

// Unary operations
struct NegateFunc {
    template <typename T> T operator()(const T &a) const {
        if constexpr (std::is_same_v<T, bool>) {
            return !a;
        } else if constexpr (std::is_unsigned_v<T>) {
            return static_cast<T>(T{0} - a);
        } else {
            return -a; // Works for complex types too
        }
    }
};

struct LogicalNotFunc {
    // LogicalNot always returns bool: !static_cast<bool>(a)
    template <typename T> bool operator()(const T &a) const {
        return !static_cast<bool>(a);
    }
};

struct AbsFunc {
    template <typename T> auto operator()(const T &a) const {
        if constexpr (std::is_unsigned_v<T> && std::is_integral_v<T>) {
            return a;
        } else if constexpr (std::is_same_v<T, float16_t>) {
            return static_cast<T>(std::abs(static_cast<float>(a)));
        } else if constexpr (std::is_same_v<T, std::complex<float>>) {
            // For complex, abs returns the magnitude as float
            return std::abs(a);
        } else if constexpr (std::is_same_v<T, std::complex<double>>) {
            return std::abs(a);
        } else {
            return std::abs(a);
        }
    }
};

struct SqrtFunc {
    template <typename T> T operator()(const T &a) const {
        if constexpr (std::is_floating_point_v<T>) {
            return std::sqrt(a);
        } else if constexpr (std::is_same_v<T, std::complex<float>> ||
                             std::is_same_v<T, std::complex<double>>) {
            return std::sqrt(a);
        } else {
            return static_cast<T>(std::sqrt(static_cast<double>(a)));
        }
    }
};

struct ExpFunc {
    template <typename T> T operator()(const T &a) const {
        if constexpr (std::is_floating_point_v<T>) {
            return std::exp(a);
        } else if constexpr (std::is_same_v<T, std::complex<float>> ||
                             std::is_same_v<T, std::complex<double>>) {
            return std::exp(a);
        } else {
            return static_cast<T>(std::exp(static_cast<double>(a)));
        }
    }
};

struct LogFunc {
    template <typename T> T operator()(const T &a) const {
        if constexpr (std::is_floating_point_v<T>) {
            return std::log(a);
        } else if constexpr (std::is_same_v<T, std::complex<float>> ||
                             std::is_same_v<T, std::complex<double>>) {
            return std::log(a);
        } else {
            return static_cast<T>(std::log(static_cast<double>(a)));
        }
    }
};

struct SinFunc {
    template <typename T> T operator()(const T &a) const {
        if constexpr (std::is_floating_point_v<T>) {
            return std::sin(a);
        } else if constexpr (std::is_same_v<T, std::complex<float>> ||
                             std::is_same_v<T, std::complex<double>>) {
            return std::sin(a);
        } else {
            return static_cast<T>(std::sin(static_cast<double>(a)));
        }
    }
};

struct CosFunc {
    template <typename T> T operator()(const T &a) const {
        if constexpr (std::is_floating_point_v<T>) {
            return std::cos(a);
        } else if constexpr (std::is_same_v<T, std::complex<float>> ||
                             std::is_same_v<T, std::complex<double>>) {
            return std::cos(a);
        } else {
            return static_cast<T>(std::cos(static_cast<double>(a)));
        }
    }
};

struct TanFunc {
    template <typename T> T operator()(const T &a) const {
        if constexpr (std::is_floating_point_v<T>) {
            return std::tan(a);
        } else if constexpr (std::is_same_v<T, std::complex<float>> ||
                             std::is_same_v<T, std::complex<double>>) {
            return std::tan(a);
        } else {
            return static_cast<T>(std::tan(static_cast<double>(a)));
        }
    }
};

struct ErfFunc {
    template <typename T> T operator()(const T &a) const {
        if constexpr (std::is_floating_point_v<T>) {
            return std::erf(a);
        } else {
            return static_cast<T>(std::erf(static_cast<double>(a)));
        }
    }
};

struct SinhFunc {
    template <typename T> T operator()(const T &a) const {
        if constexpr (std::is_floating_point_v<T>) {
            return std::sinh(a);
        } else if constexpr (std::is_same_v<T, std::complex<float>> ||
                             std::is_same_v<T, std::complex<double>>) {
            return std::sinh(a);
        } else {
            return static_cast<T>(std::sinh(static_cast<double>(a)));
        }
    }
};

struct CoshFunc {
    template <typename T> T operator()(const T &a) const {
        if constexpr (std::is_floating_point_v<T>) {
            return std::cosh(a);
        } else if constexpr (std::is_same_v<T, std::complex<float>> ||
                             std::is_same_v<T, std::complex<double>>) {
            return std::cosh(a);
        } else {
            return static_cast<T>(std::cosh(static_cast<double>(a)));
        }
    }
};

struct AsinFunc {
    template <typename T> T operator()(const T &a) const {
        if constexpr (std::is_floating_point_v<T>) {
            return std::asin(a);
        } else if constexpr (std::is_same_v<T, std::complex<float>> ||
                             std::is_same_v<T, std::complex<double>>) {
            return std::asin(a);
        } else {
            return static_cast<T>(std::asin(static_cast<double>(a)));
        }
    }
};

struct AcosFunc {
    template <typename T> T operator()(const T &a) const {
        if constexpr (std::is_floating_point_v<T>) {
            return std::acos(a);
        } else if constexpr (std::is_same_v<T, std::complex<float>> ||
                             std::is_same_v<T, std::complex<double>>) {
            return std::acos(a);
        } else {
            return static_cast<T>(std::acos(static_cast<double>(a)));
        }
    }
};

struct AtanFunc {
    template <typename T> T operator()(const T &a) const {
        if constexpr (std::is_floating_point_v<T>) {
            return std::atan(a);
        } else if constexpr (std::is_same_v<T, std::complex<float>> ||
                             std::is_same_v<T, std::complex<double>>) {
            return std::atan(a);
        } else {
            return static_cast<T>(std::atan(static_cast<double>(a)));
        }
    }
};

struct Log2Func {
    template <typename T> T operator()(const T &a) const {
        if constexpr (std::is_floating_point_v<T>) {
            return std::log2(a);
        } else {
            return static_cast<T>(std::log2(static_cast<double>(a)));
        }
    }
};

struct Log10Func {
    template <typename T> T operator()(const T &a) const {
        if constexpr (std::is_floating_point_v<T>) {
            return std::log10(a);
        } else if constexpr (std::is_same_v<T, std::complex<float>> ||
                             std::is_same_v<T, std::complex<double>>) {
            return std::log10(a);
        } else {
            return static_cast<T>(std::log10(static_cast<double>(a)));
        }
    }
};

struct Log1pFunc {
    template <typename T> T operator()(const T &a) const {
        if constexpr (std::is_floating_point_v<T>) {
            return std::log1p(a);
        } else {
            return static_cast<T>(std::log1p(static_cast<double>(a)));
        }
    }
};

struct Exp2Func {
    template <typename T> T operator()(const T &a) const {
        if constexpr (std::is_floating_point_v<T>) {
            return std::exp2(a);
        } else {
            return static_cast<T>(std::exp2(static_cast<double>(a)));
        }
    }
};

struct Expm1Func {
    template <typename T> T operator()(const T &a) const {
        if constexpr (std::is_floating_point_v<T>) {
            return std::expm1(a);
        } else {
            return static_cast<T>(std::expm1(static_cast<double>(a)));
        }
    }
};

struct RsqrtFunc {
    template <typename T> T operator()(const T &a) const {
        if constexpr (std::is_floating_point_v<T>) {
            return static_cast<T>(1) / std::sqrt(a);
        } else {
            return static_cast<T>(1.0 / std::sqrt(static_cast<double>(a)));
        }
    }
};

struct GELUFunc {
    template <typename T> T operator()(const T &a) const {
        // GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
        constexpr double sqrt2 = 1.4142135623730951;
        if constexpr (std::is_floating_point_v<T>) {
            return static_cast<T>(0.5) * a *
                   (static_cast<T>(1.0) + std::erf(a / static_cast<T>(sqrt2)));
        } else {
            double da = static_cast<double>(a);
            return static_cast<T>(0.5 * da * (1.0 + std::erf(da / sqrt2)));
        }
    }
};

struct ReLUFunc {
    template <typename T> T operator()(const T &a) const {
        return a > T{0} ? a : T{0};
    }
};

struct LeakyReLUFunc {
    float negative_slope = 0.01f;
    template <typename T> T operator()(const T &a) const {
        return a > T{0} ? a : static_cast<T>(negative_slope) * a;
    }
};

struct SigmoidFunc {
    template <typename T> T operator()(const T &a) const {
        // Numerically stable: compute exp(-|x|) to avoid overflow.
        if constexpr (std::is_floating_point_v<T>) {
            T neg_abs = -std::abs(a);
            T e = std::exp(neg_abs);
            T denom = static_cast<T>(1) + e;
            return a >= static_cast<T>(0) ? static_cast<T>(1) / denom
                                          : e / denom;
        } else {
            double da = static_cast<double>(a);
            double neg_abs = -std::abs(da);
            double e = std::exp(neg_abs);
            double denom = 1.0 + e;
            return static_cast<T>(da >= 0.0 ? 1.0 / denom : e / denom);
        }
    }
};

struct TanhFunc {
    template <typename T> T operator()(const T &a) const {
        if constexpr (std::is_floating_point_v<T>) {
            return std::tanh(a);
        } else {
            return static_cast<T>(std::tanh(static_cast<double>(a)));
        }
    }
};

struct SiLUFunc {
    template <typename T> T operator()(const T &a) const {
        // SiLU(x) = x * sigmoid(x)
        // Numerically stable: compute exp(-|x|) which is always in (0,1],
        // avoiding overflow in exp() for large |x|.
        if constexpr (std::is_floating_point_v<T>) {
            T neg_abs = -std::abs(a);
            T e = std::exp(neg_abs); // exp(-|x|), always in (0, 1]
            T denom = static_cast<T>(1) + e;
            return a >= static_cast<T>(0) ? a / denom : (a * e) / denom;
        } else {
            double da = static_cast<double>(a);
            double neg_abs = -std::abs(da);
            double e = std::exp(neg_abs);
            double denom = 1.0 + e;
            double result = da >= 0.0 ? da / denom : (da * e) / denom;
            return static_cast<T>(result);
        }
    }
};

struct ConjFunc {
    template <typename T> T operator()(const T &a) const {
        if constexpr (std::is_same_v<T, std::complex<float>> ||
                      std::is_same_v<T, std::complex<double>>) {
            return std::conj(a);
        } else {
            return a; // Real types return themselves
        }
    }
};

// NumPy-like math operations
struct SignFunc {
    template <typename T> T operator()(const T &a) const {
        if constexpr (std::is_same_v<T, bool>) {
            return a; // bool: true->true, false->false
        } else if constexpr (std::is_unsigned_v<T>) {
            // Unsigned types can only be 0 or positive
            return a > T{0} ? T{1} : T{0};
        } else if (a > T{0}) {
            return T{1};
        } else if (a < T{0}) {
            return static_cast<T>(-1);
        } else {
            return T{0};
        }
    }
};

struct FloorFunc {
    template <typename T> T operator()(const T &a) const {
        if constexpr (std::is_floating_point_v<T>) {
            return std::floor(a);
        } else if constexpr (std::is_same_v<T, float16_t>) {
            return static_cast<T>(std::floor(static_cast<float>(a)));
        } else {
            return a; // Integer types are already "floored"
        }
    }
};

struct CeilFunc {
    template <typename T> T operator()(const T &a) const {
        if constexpr (std::is_floating_point_v<T>) {
            return std::ceil(a);
        } else if constexpr (std::is_same_v<T, float16_t>) {
            return static_cast<T>(std::ceil(static_cast<float>(a)));
        } else {
            return a; // Integer types are already "ceiled"
        }
    }
};

struct TruncFunc {
    template <typename T> T operator()(const T &a) const {
        if constexpr (std::is_floating_point_v<T>) {
            return std::trunc(a);
        } else if constexpr (std::is_same_v<T, float16_t>) {
            return static_cast<T>(std::trunc(static_cast<float>(a)));
        } else {
            return a; // Integer types are already truncated
        }
    }
};

struct RoundFunc {
    template <typename T> T operator()(const T &a) const {
        if constexpr (std::is_floating_point_v<T>) {
            return std::round(a);
        } else if constexpr (std::is_same_v<T, float16_t>) {
            return static_cast<T>(std::round(static_cast<float>(a)));
        } else {
            return a; // Integer types are already rounded
        }
    }
};

struct ReciprocalFunc {
    template <typename T> T operator()(const T &a) const {
        if constexpr (std::is_floating_point_v<T>) {
            return T{1} / a;
        } else if constexpr (std::is_same_v<T, float16_t>) {
            return static_cast<T>(1.0f / static_cast<float>(a));
        } else if constexpr (std::is_same_v<T, bool>) {
            return a; // 1/true = true
        } else {
            // Integer division: 1/a (truncated)
            return (a != T{0}) ? (T{1} / a) : T{0};
        }
    }
};

struct SquareFunc {
    template <typename T> T operator()(const T &a) const { return a * a; }
};

struct CbrtFunc {
    template <typename T> T operator()(const T &a) const {
        if constexpr (std::is_floating_point_v<T>) {
            return std::cbrt(a);
        } else if constexpr (std::is_same_v<T, float16_t>) {
            return static_cast<T>(std::cbrt(static_cast<float>(a)));
        } else {
            return static_cast<T>(std::cbrt(static_cast<double>(a)));
        }
    }
};

// Element-wise testing operations (return bool)
struct IsNaNFunc {
    template <typename T> bool operator()(const T &a) const {
        if constexpr (std::is_floating_point_v<T>) {
            return std::isnan(a);
        } else if constexpr (std::is_same_v<T, float16_t>) {
            return std::isnan(static_cast<float>(a));
        } else {
            return false; // Integer types cannot be NaN
        }
    }
};

struct IsInfFunc {
    template <typename T> bool operator()(const T &a) const {
        if constexpr (std::is_floating_point_v<T>) {
            return std::isinf(a);
        } else if constexpr (std::is_same_v<T, float16_t>) {
            return std::isinf(static_cast<float>(a));
        } else {
            return false; // Integer types cannot be Inf
        }
    }
};

struct IsFiniteFunc {
    template <typename T> bool operator()(const T &a) const {
        if constexpr (std::is_floating_point_v<T>) {
            return std::isfinite(a);
        } else if constexpr (std::is_same_v<T, float16_t>) {
            return std::isfinite(static_cast<float>(a));
        } else {
            return true; // Integer types are always finite
        }
    }
};

// Reduction operations
struct SumFunc {
    template <typename T> T operator()(const T &a, const T &b) const {
        return a + b;
    }
    template <typename T> static T identity() { return static_cast<T>(0); }
};

// ============================================================================
// CPU ArgMax/ArgMin Operation
// ============================================================================

class CPUArgMaxOperation : public ops::Operation {
  public:
    ops::OpType type() const override { return ops::OpType::ArgMax; }
    std::string name() const override { return "argmax"; }
    Device device() const override { return Device::CPU; }

    Tensor execute_binary(const Tensor &lhs, const Tensor &rhs) const override {
        (void)lhs;
        (void)rhs;
        throw RuntimeError::internal(
            "execute_binary called on ArgMax operation");
    }

    Tensor execute_reduction(const Tensor &input, const std::vector<int> &axis,
                             bool keep_dims) const override;

  private:
    template <typename T>
    Tensor execute_argmax_typed(const Tensor &input, int axis,
                                bool keep_dims) const;
};

class CPUArgMinOperation : public ops::Operation {
  public:
    ops::OpType type() const override { return ops::OpType::ArgMin; }
    std::string name() const override { return "argmin"; }
    Device device() const override { return Device::CPU; }

    Tensor execute_binary(const Tensor &lhs, const Tensor &rhs) const override {
        (void)lhs;
        (void)rhs;
        throw RuntimeError::internal(
            "execute_binary called on ArgMin operation");
    }

    Tensor execute_reduction(const Tensor &input, const std::vector<int> &axis,
                             bool keep_dims) const override;

  private:
    template <typename T>
    Tensor execute_argmin_typed(const Tensor &input, int axis,
                                bool keep_dims) const;
};

struct MaxFunc {
    template <typename T> T operator()(const T &a, const T &b) const {
        return std::max(a, b);
    }
    template <typename T> static T identity() {
        return std::numeric_limits<T>::lowest();
    }
};

struct MinFunc {
    template <typename T> T operator()(const T &a, const T &b) const {
        return std::min(a, b);
    }
    template <typename T> static T identity() {
        return std::numeric_limits<T>::max();
    }
};

// Boolean reduction functions
struct AnyFunc {
    template <typename T> T operator()(const T &a, const T &b) const {
        return static_cast<bool>(a) || static_cast<bool>(b);
    }
    template <typename T> static T identity() { return T(0); } // false
};

struct AllFunc {
    template <typename T> T operator()(const T &a, const T &b) const {
        return static_cast<bool>(a) && static_cast<bool>(b);
    }
    template <typename T> static T identity() { return T(1); } // true
};

struct ProdFunc {
    template <typename T> T operator()(const T &a, const T &b) const {
        return a * b;
    }
    template <typename T> static T identity() { return T{1}; }
};

// ============================================================================
// CPU MatMul Operation
// ============================================================================

class CPUMatMulOperation : public ops::Operation {
  public:
    ops::OpType type() const override { return ops::OpType::MatMul; }
    std::string name() const override { return "matmul"; }
    Device device() const override { return Device::CPU; }

    Tensor execute_binary(const Tensor &lhs, const Tensor &rhs) const override {
        (void)lhs;
        (void)rhs;
        throw RuntimeError::internal(
            "Use execute_matmul for MatMul operations");
    }

    Tensor execute_matmul(const Tensor &a, const Tensor &b, bool transpose_a,
                          bool transpose_b) const override;

  private:
    // Helper to get the logical dimensions considering transpose flags
    static void get_matmul_dims(const Tensor &a, const Tensor &b,
                                bool transpose_a, bool transpose_b, size_t &M,
                                size_t &N, size_t &K, size_t &K_b);

    // Compute broadcasted batch shape
    static Shape compute_batch_shape(const Tensor &a, const Tensor &b);

    // Get element from tensor with proper stride handling
    template <typename T>
    static T get_element(const T *data, const std::vector<size_t> &coords,
                         const Strides &strides, size_t itemsize);

    // Set element in tensor
    template <typename T>
    static void set_element(T *data, const std::vector<size_t> &coords,
                            const Strides &strides, size_t itemsize, T value);

    // Compute single matrix multiplication C = A @ B for one batch element
    template <typename T>
    static void matmul_2d(const T *a_data, const T *b_data, T *c_data, size_t M,
                          size_t N, size_t K, size_t a_row_stride,
                          size_t a_col_stride, size_t b_row_stride,
                          size_t b_col_stride, size_t c_row_stride,
                          size_t c_col_stride);

    template <typename T>
    Tensor execute_matmul_typed(const Tensor &a, const Tensor &b,
                                bool transpose_a, bool transpose_b) const;
};

// ============================================================================
// CPU Where Operation
// ============================================================================

class CPUWhereOperation : public ops::Operation {
  public:
    ops::OpType type() const override { return ops::OpType::Where; }
    std::string name() const override { return "where"; }
    Device device() const override { return Device::CPU; }

    Tensor execute_binary(const Tensor &lhs, const Tensor &rhs) const override {
        (void)lhs;
        (void)rhs;
        throw RuntimeError::internal(
            "execute_binary called on Where operation");
    }

    Tensor execute_where(const Tensor &condition, const Tensor &a,
                         const Tensor &b) const override;

  private:
    template <typename T>
    Tensor execute_where_typed(const Tensor &condition, const Tensor &a,
                               const Tensor &b) const;
};

// ============================================================================
// CPU MaskedFill Operation
// ============================================================================

class CPUMaskedFillOperation : public ops::Operation {
  public:
    ops::OpType type() const override { return ops::OpType::MaskedFill; }
    std::string name() const override { return "masked_fill"; }
    Device device() const override { return Device::CPU; }

    Tensor execute_binary(const Tensor &lhs, const Tensor &rhs) const override {
        (void)lhs;
        (void)rhs;
        throw RuntimeError::internal(
            "execute_binary called on MaskedFill operation");
    }

    Tensor execute_masked_fill(const Tensor &input, const Tensor &mask,
                               const Tensor &value) const override;

  private:
    template <typename T>
    Tensor execute_masked_fill_typed(const Tensor &input, const Tensor &mask,
                                     const Tensor &value) const;
};

// ============================================================================
// CPU MaskedSelect Operation
// ============================================================================

class CPUMaskedSelectOperation : public ops::Operation {
  public:
    ops::OpType type() const override { return ops::OpType::MaskedSelect; }
    std::string name() const override { return "masked_select"; }
    Device device() const override { return Device::CPU; }

    Tensor execute_binary(const Tensor &lhs, const Tensor &rhs) const override {
        (void)lhs;
        (void)rhs;
        throw RuntimeError::internal(
            "execute_binary called on MaskedSelect operation");
    }

    Tensor execute_masked_select(const Tensor &input,
                                 const Tensor &mask) const override;

  private:
    template <typename T>
    Tensor execute_masked_select_typed(const Tensor &input,
                                       const Tensor &mask) const;
};

// ============================================================================
// CPU Gather Operation
// ============================================================================

class CPUGatherOperation : public ops::Operation {
  public:
    ops::OpType type() const override { return ops::OpType::Gather; }
    std::string name() const override { return "gather"; }
    Device device() const override { return Device::CPU; }

    Tensor execute_binary(const Tensor &lhs, const Tensor &rhs) const override {
        (void)lhs;
        (void)rhs;
        throw RuntimeError::internal(
            "execute_binary called on Gather operation");
    }

    Tensor execute_gather(const Tensor &input, int dim,
                          const Tensor &indices) const override;

  private:
    template <typename T>
    Tensor execute_gather_typed(const Tensor &input, int dim,
                                const Tensor &indices) const;
};

// ============================================================================
// CPU Scatter Operation
// ============================================================================

class CPUScatterOperation : public ops::Operation {
  public:
    ops::OpType type() const override { return ops::OpType::Scatter; }
    std::string name() const override { return "scatter"; }
    Device device() const override { return Device::CPU; }

    Tensor execute_binary(const Tensor &lhs, const Tensor &rhs) const override {
        (void)lhs;
        (void)rhs;
        throw RuntimeError::internal(
            "execute_binary called on Scatter operation");
    }

    Tensor execute_scatter(const Tensor &input, int dim, const Tensor &indices,
                           const Tensor &src) const override;

  private:
    template <typename T>
    Tensor execute_scatter_typed(const Tensor &input, int dim,
                                 const Tensor &indices,
                                 const Tensor &src) const;
};

// ============================================================================
// CPU IndexSelect Operation
// ============================================================================

class CPUIndexSelectOperation : public ops::Operation {
  public:
    ops::OpType type() const override { return ops::OpType::IndexSelect; }
    std::string name() const override { return "index_select"; }
    Device device() const override { return Device::CPU; }

    Tensor execute_binary(const Tensor &lhs, const Tensor &rhs) const override {
        (void)lhs;
        (void)rhs;
        throw RuntimeError::internal(
            "execute_binary called on IndexSelect operation");
    }

    Tensor execute_index_select(const Tensor &input, int dim,
                                const Tensor &indices) const override;

  private:
    template <typename T>
    Tensor execute_index_select_typed(const Tensor &input, int dim,
                                      const Tensor &indices) const;
};

// ============================================================================
// CPU Softmax/LogSoftmax Operations
// ============================================================================

class CPUSoftmaxOperation : public ops::Operation {
    bool is_log_;

  public:
    CPUSoftmaxOperation(bool is_log) : is_log_(is_log) {}
    ops::OpType type() const override {
        return is_log_ ? ops::OpType::LogSoftmax : ops::OpType::Softmax;
    }
    std::string name() const override {
        return is_log_ ? "log_softmax" : "softmax";
    }
    Device device() const override { return Device::CPU; }

    Tensor execute_binary(const Tensor &lhs, const Tensor &rhs) const override {
        (void)lhs;
        (void)rhs;
        throw RuntimeError::internal(
            "execute_binary called on Softmax operation");
    }

    Tensor execute_reduction(const Tensor &input, const std::vector<int> &axis,
                             bool keep_dims) const override;

  private:
    template <typename T>
    Tensor execute_softmax_typed(const Tensor &input, int axis) const;
};

// ============================================================================
// Factory functions
// ============================================================================

void register_cpu_operations();

void register_cpu_backend();

} // namespace cpu
} // namespace backends
} // namespace axiom