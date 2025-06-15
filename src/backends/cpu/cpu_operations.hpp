#pragma once

#include "axiom/operations.hpp"
#include <type_traits>
#include <cmath>

namespace axiom {
namespace backends {
namespace cpu {

// ============================================================================
// CPU Binary Operation Base Class
// ============================================================================

template<typename Func>
class CPUBinaryOperation : public ops::Operation {
 private:
  ops::OpType op_type_;
  std::string name_;
  Func func_;

 public:
  CPUBinaryOperation(ops::OpType op_type, const std::string& name, Func func)
      : op_type_(op_type), name_(name), func_(func) {}

  ops::OpType type() const override { return op_type_; }
  std::string name() const override { return name_; }
  Device device() const override { return Device::CPU; }
  
  bool supports_binary(const Tensor& lhs, const Tensor& rhs) const override {
      return ops::are_broadcastable(lhs.shape(), rhs.shape());
  }

  Tensor execute_binary(const Tensor& lhs, const Tensor& rhs) const override;

  void execute_binary_inplace(Tensor& lhs, const Tensor& rhs) const override;

 private:
  template<typename T>
  void execute_binary_typed(const Tensor& lhs, const Tensor& rhs, Tensor& result) const;
  
  template<typename T>
  void execute_binary_broadcast(const Tensor& lhs, const Tensor& rhs, Tensor& result,
                               const ops::BroadcastInfo& broadcast_info) const;
  
  template<typename T>
  void execute_binary_same_shape(const Tensor& lhs, const Tensor& rhs, Tensor& result) const;
  
  template<typename T>
  void execute_inplace_typed(Tensor& lhs, const Tensor& rhs) const;

  template<typename T>
  void execute_inplace_broadcast(Tensor& lhs, const Tensor& rhs) const;
  
  template<typename InputT, typename OutputT>
  void execute_broadcast_loop(const InputT* lhs_data, const InputT* rhs_data, OutputT* result_data,
                             const Shape& lhs_shape, const Shape& rhs_shape, const Shape& result_shape,
                             const Strides& lhs_strides, const Strides& rhs_strides) const;
};

// ============================================================================
// CPU Unary Operation Base Class
// ============================================================================

template<typename Func>
class CPUUnaryOperation : public ops::Operation {
 private:
  ops::OpType op_type_;
  std::string name_;
  Func func_;

 public:
  CPUUnaryOperation(ops::OpType op_type, const std::string& name, Func func)
      : op_type_(op_type), name_(name), func_(func) {}

  ops::OpType type() const override { return op_type_; }
  std::string name() const override { return name_; }
  Device device() const override { return Device::CPU; }

  Tensor execute_binary(const Tensor& lhs, const Tensor& rhs) const override {
    (void)lhs; (void)rhs;
    throw std::runtime_error("Not a binary operation");
  }

  Tensor execute_unary(const Tensor& input) const override;

 private:
  template<typename T>
  void execute_unary_typed(const Tensor& input, Tensor& result) const;
};

// ============================================================================
// CPU Reduction Operation Base Class
// ============================================================================

template<typename Func>
class CPUReductionOperation : public ops::Operation {
 private:
  ops::OpType op_type_;
  std::string name_;
  Func func_;

 public:
  CPUReductionOperation(ops::OpType op_type, const std::string& name, Func func)
      : op_type_(op_type), name_(name), func_(func) {}

  ops::OpType type() const override { return op_type_; }
  std::string name() const override { return name_; }
  Device device() const override { return Device::CPU; }

  Tensor execute_binary(const Tensor& lhs, const Tensor& rhs) const override {
    (void)lhs; (void)rhs;
    throw std::runtime_error("Not a binary operation");
  }

  Tensor execute_unary(const Tensor& input) const override {
    // This will be called by the high-level reduction function
    // We will need a new execute_reduction method
    (void)input;
    throw std::runtime_error("Use execute_reduction for reduction operations");
  }

  Tensor execute_reduction(const Tensor& input, const std::vector<int>& axis, bool keep_dims) const override;

 private:
  template<typename T>
  Tensor execute_reduction_typed(const Tensor& input, const std::vector<int>& axis, bool keep_dims) const;

  template<typename T>
  static void reduction_recursive_helper(const Tensor& input, Tensor& result, const std::vector<int>& axes,
                                          std::vector<size_t>& current_coords, int current_dim, const Func& func, bool keep_dims);
};

// ============================================================================
// Operation Function Objects
// ============================================================================

// Arithmetic operations
struct AddFunc {
  template<typename T>
  T operator()(const T& a, const T& b) const { return a + b; }
};

struct SubtractFunc {
  template<typename T>
  T operator()(const T& a, const T& b) const { return a - b; }
};

struct MultiplyFunc {
  template<typename T>
  T operator()(const T& a, const T& b) const { return a * b; }
};

struct DivideFunc {
  template<typename T>
  T operator()(const T& a, const T& b) const { return a / b; }
};

struct PowerFunc {
  template<typename T>
  T operator()(const T& a, const T& b) const {
    if constexpr (std::is_integral_v<T>) {
      return static_cast<T>(std::pow(static_cast<double>(a), static_cast<double>(b)));
    } else {
      return static_cast<T>(std::pow(a, b));
    }
  }
};

struct ModuloFunc {
  template<typename T>
  T operator()(const T& a, const T& b) const {
    if constexpr (std::is_integral_v<T>) {
      return a % b;
    } else {
      return static_cast<T>(std::fmod(a, b));
    }
  }
};

// Comparison operations
struct EqualFunc {
  template<typename T>
  bool operator()(const T& a, const T& b) const { return a == b; }
};

struct NotEqualFunc {
  template<typename T>
  bool operator()(const T& a, const T& b) const { return a != b; }
};

struct LessFunc {
  template<typename T>
  bool operator()(const T& a, const T& b) const { return a < b; }
};

struct LessEqualFunc {
  template<typename T>
  bool operator()(const T& a, const T& b) const { return a <= b; }
};

struct GreaterFunc {
  template<typename T>
  bool operator()(const T& a, const T& b) const { return a > b; }
};

struct GreaterEqualFunc {
  template<typename T>
  bool operator()(const T& a, const T& b) const { return a >= b; }
};

// Logical operations
struct LogicalAndFunc {
  template<typename T>
  bool operator()(const T& a, const T& b) const { 
    return static_cast<bool>(a) && static_cast<bool>(b);
  }
};

struct LogicalOrFunc {
  template<typename T>
  bool operator()(const T& a, const T& b) const { 
    return static_cast<bool>(a) || static_cast<bool>(b);
  }
};

struct LogicalXorFunc {
  template<typename T>
  bool operator()(const T& a, const T& b) const { 
    return static_cast<bool>(a) != static_cast<bool>(b);
  }
};

// Bitwise operations (only for integer types)
struct BitwiseAndFunc {
  template<typename T>
  T operator()(const T& a, const T& b) const {
    static_assert(std::is_integral_v<T>, "Bitwise operations only supported for integer types");
    return a & b;
  }
};

struct BitwiseOrFunc {
  template<typename T>
  T operator()(const T& a, const T& b) const {
    static_assert(std::is_integral_v<T>, "Bitwise operations only supported for integer types");
    return a | b;
  }
};

struct BitwiseXorFunc {
  template<typename T>
  T operator()(const T& a, const T& b) const {
    static_assert(std::is_integral_v<T>, "Bitwise operations only supported for integer types");
    return a ^ b;
  }
};

struct LeftShiftFunc {
  template<typename T>
  T operator()(const T& a, const T& b) const {
    static_assert(std::is_integral_v<T>, "Shift operations only supported for integer types");
    return a << b;
  }
};

struct RightShiftFunc {
  template<typename T>
  T operator()(const T& a, const T& b) const {
    static_assert(std::is_integral_v<T>, "Shift operations only supported for integer types");
    return a >> b;
  }
};

// Math operations
struct MaximumFunc {
  template<typename T>
  T operator()(const T& a, const T& b) const { return std::max(a, b); }
};

struct MinimumFunc {
  template<typename T>
  T operator()(const T& a, const T& b) const { return std::min(a, b); }
};

struct Atan2Func {
  template<typename T>
  T operator()(const T& a, const T& b) const {
    if constexpr (std::is_integral_v<T>) {
      return static_cast<T>(std::atan2(static_cast<double>(a), static_cast<double>(b)));
    } else {
      return static_cast<T>(std::atan2(a, b));
    }
  }
};

struct HypotFunc {
  template<typename T>
  T operator()(const T& a, const T& b) const {
    if constexpr (std::is_integral_v<T>) {
      return static_cast<T>(std::hypot(static_cast<double>(a), static_cast<double>(b)));
    } else {
      return static_cast<T>(std::hypot(a, b));
    }
  }
};

// Unary operations
struct NegateFunc {
  template<typename T>
  T operator()(const T& a) const {
    if constexpr (std::is_same_v<T, bool>) {
        return !a;
    } else {
        return -a;
    }
  }
};

struct AbsFunc {
  template<typename T>
  T operator()(const T& a) const {
    if constexpr (std::is_unsigned_v<T> && std::is_integral_v<T>) {
      return a;
    } else if constexpr (std::is_same_v<T, half_float::half>) {
      return static_cast<T>(std::abs(static_cast<float>(a)));
    } else {
      return std::abs(a);
    }
  }
};

struct SqrtFunc {
  template<typename T>
  T operator()(const T& a) const {
    if constexpr (std::is_floating_point_v<T>) {
        return std::sqrt(a);
    } else {
        return static_cast<T>(std::sqrt(static_cast<double>(a)));
    }
  }
};

struct ExpFunc {
  template<typename T>
  T operator()(const T& a) const {
    if constexpr (std::is_floating_point_v<T>) {
        return std::exp(a);
    } else {
        return static_cast<T>(std::exp(static_cast<double>(a)));
    }
  }
};

struct LogFunc {
  template<typename T>
  T operator()(const T& a) const {
    if constexpr (std::is_floating_point_v<T>) {
        return std::log(a);
    } else {
        return static_cast<T>(std::log(static_cast<double>(a)));
    }
  }
};

struct SinFunc {
  template<typename T>
  T operator()(const T& a) const {
    if constexpr (std::is_floating_point_v<T>) {
        return std::sin(a);
    } else {
        return static_cast<T>(std::sin(static_cast<double>(a)));
    }
  }
};

struct CosFunc {
  template<typename T>
  T operator()(const T& a) const {
    if constexpr (std::is_floating_point_v<T>) {
        return std::cos(a);
    } else {
        return static_cast<T>(std::cos(static_cast<double>(a)));
    }
  }
};

struct TanFunc {
  template<typename T>
  T operator()(const T& a) const {
    if constexpr (std::is_floating_point_v<T>) {
        return std::tan(a);
    } else {
        return static_cast<T>(std::tan(static_cast<double>(a)));
    }
  }
};

// Reduction operations
struct SumFunc {
  template<typename T>
  T operator()(const T& a, const T& b) const { return a + b; }
  template<typename T>
  static T identity() { return static_cast<T>(0); }
};

struct MaxFunc {
  template<typename T>
  T operator()(const T& a, const T& b) const { return std::max(a, b); }
  template<typename T>
  static T identity() { return std::numeric_limits<T>::lowest(); }
};

struct MinFunc {
  template<typename T>
  T operator()(const T& a, const T& b) const { return std::min(a, b); }
  template<typename T>
  static T identity() { return std::numeric_limits<T>::max(); }
};

// ============================================================================
// Factory functions
// ============================================================================

void register_cpu_operations();

void add(Tensor& a, const Tensor& b);

void register_cpu_backend();

}  // namespace cpu
}  // namespace backends
}  // namespace axiom 