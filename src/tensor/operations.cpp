#include "axiom/operations.hpp"
#include "axiom/tensor.hpp"
#include "../backends/cpu/cpu_operations.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace axiom {
namespace ops {

// ============================================================================
// Broadcasting utilities
// ============================================================================

BroadcastInfo compute_broadcast_info(const Shape& lhs_shape, const Shape& rhs_shape) {
  BroadcastInfo info;
  info.needs_broadcast = false;
  
  // Get the maximum number of dimensions
  size_t max_dims = std::max(lhs_shape.size(), rhs_shape.size());
  
  // Start with the result shape
  info.result_shape.resize(max_dims);
  info.lhs_strides_adjustment.resize(max_dims, 0);
  info.rhs_strides_adjustment.resize(max_dims, 0);
  
  // Work backwards from the last dimension
  for (int i = max_dims - 1; i >= 0; --i) {
    int lhs_idx = static_cast<int>(lhs_shape.size()) - (max_dims - i);
    int rhs_idx = static_cast<int>(rhs_shape.size()) - (max_dims - i);
    
    size_t lhs_dim = (lhs_idx >= 0) ? lhs_shape[lhs_idx] : 1;
    size_t rhs_dim = (rhs_idx >= 0) ? rhs_shape[rhs_idx] : 1;
    
    if (lhs_dim == rhs_dim) {
      info.result_shape[i] = lhs_dim;
    } else if (lhs_dim == 1) {
      info.result_shape[i] = rhs_dim;
      info.lhs_strides_adjustment[i] = 0;  // Stride becomes 0 for broadcasting
      info.needs_broadcast = true;
    } else if (rhs_dim == 1) {
      info.result_shape[i] = lhs_dim;
      info.rhs_strides_adjustment[i] = 0;  // Stride becomes 0 for broadcasting
      info.needs_broadcast = true;
    } else {
      throw std::runtime_error("Cannot broadcast shapes: dimension mismatch");
    }
  }
  
  return info;
}

bool are_broadcastable(const Shape& lhs_shape, const Shape& rhs_shape) {
  try {
    compute_broadcast_info(lhs_shape, rhs_shape);
    return true;
  } catch (const std::runtime_error&) {
    return false;
  }
}

// ============================================================================
// Type promotion utilities
// ============================================================================

DType promote_types(DType lhs_dtype, DType rhs_dtype) {
  // If types are the same, return that type
  if (lhs_dtype == rhs_dtype) {
    return lhs_dtype;
  }
  
  // Complex types take precedence
  if (is_complex_dtype(lhs_dtype) || is_complex_dtype(rhs_dtype)) {
    if (lhs_dtype == DType::Complex128 || rhs_dtype == DType::Complex128) {
      return DType::Complex128;
    }
    return DType::Complex64;
  }
  
  // Float types take precedence over integers
  if (is_floating_dtype(lhs_dtype) || is_floating_dtype(rhs_dtype)) {
    DType float_type = is_floating_dtype(lhs_dtype) ? lhs_dtype : rhs_dtype;
    DType other_type = is_floating_dtype(lhs_dtype) ? rhs_dtype : lhs_dtype;
    
    // Promote to highest precision float
    if (float_type == DType::Float64 || other_type == DType::Float64) {
      return DType::Float64;
    }
    if (float_type == DType::Float32 || other_type == DType::Float32) {
      return DType::Float32;
    }
    return DType::Float16;
  }
  
  // Integer promotion rules
  if (is_integer_dtype(lhs_dtype) && is_integer_dtype(rhs_dtype)) {
    // Bool promotes to any integer type
    if (lhs_dtype == DType::Bool) return rhs_dtype;
    if (rhs_dtype == DType::Bool) return lhs_dtype;
    
    // Promote to largest integer type
    auto get_int_priority = [](DType dtype) -> int {
      switch (dtype) {
        case DType::Int8: return 1;
        case DType::UInt8: return 2;
        case DType::Int16: return 3;
        case DType::UInt16: return 4;
        case DType::Int32: return 5;
        case DType::UInt32: return 6;
        case DType::Int64: return 7;
        case DType::UInt64: return 8;
        default: return 0;
      }
    };
    
    return get_int_priority(lhs_dtype) >= get_int_priority(rhs_dtype) ? lhs_dtype : rhs_dtype;
  }
  
  // Default case - promote to Float32
  return DType::Float32;
}

DType result_type(const Tensor& lhs, const Tensor& rhs) {
  return promote_types(lhs.dtype(), rhs.dtype());
}

// ============================================================================
// Operation Registry
// ============================================================================

std::map<std::pair<OpType, Device>, std::unique_ptr<Operation>>& 
OperationRegistry::get_registry() {
  static std::map<std::pair<OpType, Device>, std::unique_ptr<Operation>> registry;
  return registry;
}

void OperationRegistry::register_operation(OpType op_type, Device device, 
                                          std::unique_ptr<Operation> operation) {
  auto& registry = get_registry();
  registry[{op_type, device}] = std::move(operation);
}

const Operation* OperationRegistry::get_operation(OpType op_type, Device device) {
  auto& registry = get_registry();
  auto it = registry.find({op_type, device});
  return (it != registry.end()) ? it->second.get() : nullptr;
}

std::vector<Device> OperationRegistry::available_devices_for_operation(OpType op_type) {
  std::vector<Device> devices;
  auto& registry = get_registry();
  
  for (const auto& [key, operation] : registry) {
    if (key.first == op_type) {
      devices.push_back(key.second);
    }
  }
  
  return devices;
}

bool OperationRegistry::is_operation_available(OpType op_type, Device device) {
  return get_operation(op_type, device) != nullptr;
}

void OperationRegistry::initialize_builtin_operations() {
  // Register CPU operations
  backends::cpu::register_cpu_operations();
}

// ============================================================================
// Default Operation implementations
// ============================================================================

Tensor Operation::execute_unary(const Tensor& input) const {
  (void)input; // Suppress unused parameter warning
  throw std::runtime_error("Unary operations not implemented yet");
}

void Operation::execute_binary_inplace(Tensor& lhs, const Tensor& rhs) const {
  (void)lhs; // Suppress unused parameter warning
  (void)rhs; // Suppress unused parameter warning
  throw std::runtime_error("In-place operations not implemented yet");
}

// ============================================================================
// Helper function for executing binary operations
// ============================================================================

static Tensor execute_binary_operation(OpType op_type, const Tensor& lhs, const Tensor& rhs) {
  // Check if tensors are broadcastable
  if (!are_broadcastable(lhs.shape(), rhs.shape())) {
    throw std::runtime_error("Tensors are not broadcastable");
  }
  
  // Determine the target device (prefer GPU if available)
  Device target_device = Device::CPU;
  if (lhs.device() == Device::GPU || rhs.device() == Device::GPU) {
    target_device = Device::GPU;
  }
  
  // Get the operation implementation
  const Operation* op = OperationRegistry::get_operation(op_type, target_device);
  if (!op) {
    // Fallback to CPU if GPU operation not available
    if (target_device == Device::GPU) {
      target_device = Device::CPU;
      op = OperationRegistry::get_operation(op_type, target_device);
    }
    
    if (!op) {
      throw std::runtime_error("Operation not available for any device");
    }
  }
  
  // Move tensors to target device if needed
  Tensor lhs_target = (lhs.device() == target_device) ? lhs : lhs.to(target_device);
  Tensor rhs_target = (rhs.device() == target_device) ? rhs : rhs.to(target_device);
  
  // Execute the operation
  return op->execute_binary(lhs_target, rhs_target);
}

// ============================================================================
// High-level operation functions
// ============================================================================

// Binary operations
Tensor add(const Tensor& lhs, const Tensor& rhs) {
  return execute_binary_operation(OpType::Add, lhs, rhs);
}

Tensor subtract(const Tensor& lhs, const Tensor& rhs) {
  return execute_binary_operation(OpType::Subtract, lhs, rhs);
}

Tensor multiply(const Tensor& lhs, const Tensor& rhs) {
  return execute_binary_operation(OpType::Multiply, lhs, rhs);
}

Tensor divide(const Tensor& lhs, const Tensor& rhs) {
  return execute_binary_operation(OpType::Divide, lhs, rhs);
}

Tensor power(const Tensor& lhs, const Tensor& rhs) {
  return execute_binary_operation(OpType::Power, lhs, rhs);
}

Tensor modulo(const Tensor& lhs, const Tensor& rhs) {
  return execute_binary_operation(OpType::Modulo, lhs, rhs);
}

// Comparison operations
Tensor equal(const Tensor& lhs, const Tensor& rhs) {
  return execute_binary_operation(OpType::Equal, lhs, rhs);
}

Tensor not_equal(const Tensor& lhs, const Tensor& rhs) {
  return execute_binary_operation(OpType::NotEqual, lhs, rhs);
}

Tensor less(const Tensor& lhs, const Tensor& rhs) {
  return execute_binary_operation(OpType::Less, lhs, rhs);
}

Tensor less_equal(const Tensor& lhs, const Tensor& rhs) {
  return execute_binary_operation(OpType::LessEqual, lhs, rhs);
}

Tensor greater(const Tensor& lhs, const Tensor& rhs) {
  return execute_binary_operation(OpType::Greater, lhs, rhs);
}

Tensor greater_equal(const Tensor& lhs, const Tensor& rhs) {
  return execute_binary_operation(OpType::GreaterEqual, lhs, rhs);
}

// Logical operations
Tensor logical_and(const Tensor& lhs, const Tensor& rhs) {
  return execute_binary_operation(OpType::LogicalAnd, lhs, rhs);
}

Tensor logical_or(const Tensor& lhs, const Tensor& rhs) {
  return execute_binary_operation(OpType::LogicalOr, lhs, rhs);
}

Tensor logical_xor(const Tensor& lhs, const Tensor& rhs) {
  return execute_binary_operation(OpType::LogicalXor, lhs, rhs);
}

// Bitwise operations
Tensor bitwise_and(const Tensor& lhs, const Tensor& rhs) {
  return execute_binary_operation(OpType::BitwiseAnd, lhs, rhs);
}

Tensor bitwise_or(const Tensor& lhs, const Tensor& rhs) {
  return execute_binary_operation(OpType::BitwiseOr, lhs, rhs);
}

Tensor bitwise_xor(const Tensor& lhs, const Tensor& rhs) {
  return execute_binary_operation(OpType::BitwiseXor, lhs, rhs);
}

Tensor left_shift(const Tensor& lhs, const Tensor& rhs) {
  return execute_binary_operation(OpType::LeftShift, lhs, rhs);
}

Tensor right_shift(const Tensor& lhs, const Tensor& rhs) {
  return execute_binary_operation(OpType::RightShift, lhs, rhs);
}

// Math operations
Tensor maximum(const Tensor& lhs, const Tensor& rhs) {
  return execute_binary_operation(OpType::Maximum, lhs, rhs);
}

Tensor minimum(const Tensor& lhs, const Tensor& rhs) {
  return execute_binary_operation(OpType::Minimum, lhs, rhs);
}

Tensor atan2(const Tensor& lhs, const Tensor& rhs) {
  return execute_binary_operation(OpType::Atan2, lhs, rhs);
}

Tensor hypot(const Tensor& lhs, const Tensor& rhs) {
  return execute_binary_operation(OpType::Hypot, lhs, rhs);
}

}  // namespace ops
}  // namespace axiom 