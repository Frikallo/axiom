#pragma once

#include <memory>
#include <string>
#include <vector>
#include <functional>
#include <type_traits>
#include <map>

#include "dtype.hpp"
#include "shape.hpp"
#include "storage.hpp"
#include "tensor.hpp"

namespace axiom {

namespace ops {

// ============================================================================
// Broadcasting utilities
// ============================================================================

struct BroadcastInfo {
  Shape result_shape;
  std::vector<int> lhs_strides_adjustment;
  std::vector<int> rhs_strides_adjustment;
  bool needs_broadcast;
};

BroadcastInfo compute_broadcast_info(const Shape& lhs_shape, const Shape& rhs_shape);
bool are_broadcastable(const Shape& lhs_shape, const Shape& rhs_shape);

// ============================================================================
// Type promotion utilities
// ============================================================================

DType promote_types(DType lhs_dtype, DType rhs_dtype);
DType result_type(const Tensor& lhs, const Tensor& rhs);

// ============================================================================
// Operation Interface
// ============================================================================

enum class OpType {
  // Binary operations
  Add,
  Subtract,
  Multiply,
  Divide,
  Power,
  Modulo,
  
  // Comparison operations
  Equal,
  NotEqual,
  Less,
  LessEqual,
  Greater,
  GreaterEqual,
  
  // Logical operations
  LogicalAnd,
  LogicalOr,
  LogicalXor,
  
  // Bitwise operations
  BitwiseAnd,
  BitwiseOr,
  BitwiseXor,
  LeftShift,
  RightShift,
  
  // Math operations
  Maximum,
  Minimum,
  Atan2,
  Hypot,
  
  // Future unary operations
  Negate,
  Abs,
  Sqrt,
  Exp,
  Log,
  Sin,
  Cos,
  Tan
};

class Operation {
 public:
  virtual ~Operation() = default;
  
  virtual OpType type() const = 0;
  virtual std::string name() const = 0;
  virtual Device device() const = 0;
  
  // A way to check for feature support like broadcasting
  virtual bool supports_binary(const Tensor& lhs, const Tensor& rhs) const {
      // By default, assume basic support (same shapes, no broadcasting)
      return lhs.shape() == rhs.shape();
  }
  
  // For binary operations
  virtual Tensor execute_binary(const Tensor& lhs, const Tensor& rhs) const = 0;
  
  // For unary operations (future extension)
  virtual Tensor execute_unary(const Tensor& input) const;
  
  // For in-place operations (future extension)
  virtual void execute_binary_inplace(Tensor& lhs, const Tensor& rhs) const;
};

// ============================================================================
// Operation Registry
// ============================================================================

class OperationRegistry {
 public:
  static void register_operation(OpType op_type, Device device, 
                                std::unique_ptr<Operation> operation);
  
  static const Operation* get_operation(OpType op_type, Device device);
  
  static std::vector<Device> available_devices_for_operation(OpType op_type);
  
  static bool is_operation_available(OpType op_type, Device device);
  
  // Initialize built-in operations
  static void initialize_builtin_operations();
  
 private:
  static std::map<std::pair<OpType, Device>, std::unique_ptr<Operation>>& 
    get_registry();
};

// ============================================================================
// High-level operation functions
// ============================================================================

// Binary operations
Tensor add(const Tensor& lhs, const Tensor& rhs);
Tensor subtract(const Tensor& lhs, const Tensor& rhs);
Tensor multiply(const Tensor& lhs, const Tensor& rhs);
Tensor divide(const Tensor& lhs, const Tensor& rhs);
Tensor power(const Tensor& lhs, const Tensor& rhs);
Tensor modulo(const Tensor& lhs, const Tensor& rhs);

// Comparison operations
Tensor equal(const Tensor& lhs, const Tensor& rhs);
Tensor not_equal(const Tensor& lhs, const Tensor& rhs);
Tensor less(const Tensor& lhs, const Tensor& rhs);
Tensor less_equal(const Tensor& lhs, const Tensor& rhs);
Tensor greater(const Tensor& lhs, const Tensor& rhs);
Tensor greater_equal(const Tensor& lhs, const Tensor& rhs);

// Logical operations
Tensor logical_and(const Tensor& lhs, const Tensor& rhs);
Tensor logical_or(const Tensor& lhs, const Tensor& rhs);
Tensor logical_xor(const Tensor& lhs, const Tensor& rhs);

// Bitwise operations
Tensor bitwise_and(const Tensor& lhs, const Tensor& rhs);
Tensor bitwise_or(const Tensor& lhs, const Tensor& rhs);
Tensor bitwise_xor(const Tensor& lhs, const Tensor& rhs);
Tensor left_shift(const Tensor& lhs, const Tensor& rhs);
Tensor right_shift(const Tensor& lhs, const Tensor& rhs);

// Math operations
Tensor maximum(const Tensor& lhs, const Tensor& rhs);
Tensor minimum(const Tensor& lhs, const Tensor& rhs);
Tensor atan2(const Tensor& lhs, const Tensor& rhs);
Tensor hypot(const Tensor& lhs, const Tensor& rhs);

}  // namespace ops
}  // namespace axiom 