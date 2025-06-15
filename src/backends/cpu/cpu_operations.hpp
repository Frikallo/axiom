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
                             const Shape& lhs_shape, const Shape& rhs_shape, const Shape& result_shape) const;
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

// ============================================================================
// Factory functions
// ============================================================================

void register_cpu_operations();

void add(Tensor& a, const Tensor& b);

void register_cpu_backend();

}  // namespace cpu
}  // namespace backends
}  // namespace axiom 