#include "axiom/operations.hpp"
#include "axiom/tensor.hpp"
#include "backends/cpu/cpu_operations.hpp"
#include "backends/metal/metal_operations.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace axiom {
namespace ops {

// ============================================================================
// Broadcasting utilities
// ============================================================================

BroadcastInfo compute_broadcast_info(const Shape& lhs_shape, const Shape& rhs_shape) {
    if (lhs_shape.empty() && rhs_shape.empty()) {
        return {{}, {}, {}, false};
    }
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

DType promote_types(DType lhs, DType rhs) {
    if (lhs == rhs) return lhs;

    // Highest rank wins
    static const std::map<DType, int> rank = {
        {DType::Bool, 0}, {DType::Int8, 1}, {DType::UInt8, 2},
        {DType::Int16, 3}, {DType::UInt16, 4}, {DType::Int32, 5},
        {DType::UInt32, 6}, {DType::Int64, 7}, {DType::UInt64, 8},
        {DType::Float16, 9}, {DType::Float32, 10}, {DType::Float64, 11},
        {DType::Complex64, 12}, {DType::Complex128, 13}
    };
    
    DType t1 = rank.at(lhs) > rank.at(rhs) ? lhs : rhs;
    DType t2 = rank.at(lhs) > rank.at(rhs) ? rhs : lhs;

    if (is_complex_dtype(t1)) return t1;
    if (is_floating_dtype(t1)) return t1;

    // Both are integers
    bool signed1 = is_signed_integer_dtype(t1);
    bool signed2 = is_signed_integer_dtype(t2);

    size_t size1 = dtype_size(t1);
    size_t size2 = dtype_size(t2);

    if (signed1 == signed2) return t1;

    // t1 is higher rank. if signed, it can hold t2.
    if (signed1 && size1 > size2) return t1;

    // t1 is unsigned. if bigger, it can hold t2.
    if (!signed1 && size1 > size2) return t1;
    
    // if sizes are equal, promote to next size signed
    // if t2 is signed and bigger, t1 wins (e.g. uint8+int16 -> int16)
    // if we are here, we need to promote.
    switch(std::max(size1, size2)) {
        case 1: return DType::Int16;
        case 2: return DType::Int32;
        case 4: return DType::Int64;
        default: return DType::Float64;
    }
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

#ifdef __APPLE__
    backends::metal::register_metal_operations();
#endif
}

// ============================================================================
// Default Operation implementations
// ============================================================================

Tensor Operation::execute_unary(const Tensor& input) const {
  (void)input; // Suppress unused parameter warning
  throw std::runtime_error("Unary operations not implemented yet");
}

Tensor Operation::execute_reduction(const Tensor& input, const std::vector<int>& axis, bool keep_dims) const {
    (void)input; (void)axis; (void)keep_dims;
    throw std::runtime_error("Reduction operations not implemented yet");
}

Tensor Operation::execute_matmul(const Tensor& a, const Tensor& b,
                                 bool transpose_a, bool transpose_b) const {
    (void)a; (void)b; (void)transpose_a; (void)transpose_b;
    throw std::runtime_error("MatMul operations not implemented yet");
}

void Operation::execute_binary_inplace(Tensor& lhs, const Tensor& rhs) const {
  (void)lhs; // Suppress unused parameter warning
  (void)rhs; // Suppress unused parameter warning
  throw std::runtime_error("In-place operations not implemented yet");
}

// ============================================================================
// Helper function for executing unary operations
// ============================================================================

static Tensor execute_unary_operation(OpType op_type, const Tensor& input) {
  Device target_device = input.device();
  
  const Operation* op = OperationRegistry::get_operation(op_type, target_device);
  
  if (!op) {
    throw std::runtime_error("Operation not available for the tensor's device");
  }
  
  return op->execute_unary(input);
}

// ============================================================================
// Helper function for executing reduction operations
// ============================================================================
static Tensor execute_reduction_operation(OpType op_type, const Tensor& input, const std::vector<int>& axis, bool keep_dims) {
    Device target_device = input.device();

    const auto* op = OperationRegistry::get_operation(op_type, target_device);
    if (!op) {
        throw std::runtime_error("Reduction operation not available for the tensor's device");
    }

    return op->execute_reduction(input, axis, keep_dims);
}

// ============================================================================
// Helper function for executing matmul operations
// ============================================================================
static Tensor execute_matmul_operation(const Tensor& a, const Tensor& b,
                                       bool transpose_a, bool transpose_b) {
    // Determine target device (prefer GPU if available)
    Device target_device = (a.device() == Device::GPU || b.device() == Device::GPU)
                           ? Device::GPU : Device::CPU;

    const Operation* op = OperationRegistry::get_operation(OpType::MatMul, target_device);

    // Fallback to CPU if GPU op not available
    if (target_device == Device::GPU && !op) {
        target_device = Device::CPU;
        op = OperationRegistry::get_operation(OpType::MatMul, target_device);
    }

    if (!op) {
        throw std::runtime_error("MatMul operation not available for any device");
    }

    // Move tensors to target device if needed
    Tensor a_target = (a.device() == target_device) ? a : a.to(target_device);
    Tensor b_target = (b.device() == target_device) ? b : b.to(target_device);

    return op->execute_matmul(a_target, b_target, transpose_a, transpose_b);
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
  Device target_device = (lhs.device() == Device::GPU || rhs.device() == Device::GPU) ? Device::GPU : Device::CPU;
  
  // Get the operation implementation
  const Operation* op = OperationRegistry::get_operation(op_type, target_device);

  // Fallback to CPU if GPU op is not available or doesn't support the inputs
  if (target_device == Device::GPU && (!op || !op->supports_binary(lhs, rhs))) {
    target_device = Device::CPU;
    op = OperationRegistry::get_operation(op_type, target_device);
  }
  
  if (!op) {
    throw std::runtime_error("Operation not available for any device");
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

// Unary operations
Tensor negate(const Tensor& input) {
  return execute_unary_operation(OpType::Negate, input);
}

Tensor abs(const Tensor& input) {
  return execute_unary_operation(OpType::Abs, input);
}

Tensor sqrt(const Tensor& input) {
  return execute_unary_operation(OpType::Sqrt, input);
}

Tensor exp(const Tensor& input) {
  return execute_unary_operation(OpType::Exp, input);
}

Tensor log(const Tensor& input) {
  return execute_unary_operation(OpType::Log, input);
}

Tensor sin(const Tensor& input) {
  return execute_unary_operation(OpType::Sin, input);
}

Tensor cos(const Tensor& input) {
  return execute_unary_operation(OpType::Cos, input);
}

Tensor tan(const Tensor& input) {
  return execute_unary_operation(OpType::Tan, input);
}

// Reduction operations
Tensor sum(const Tensor& input, const std::vector<int>& axis, bool keep_dims) {
    return execute_reduction_operation(OpType::Sum, input, axis, keep_dims);
}

Tensor mean(const Tensor& input, const std::vector<int>& axis, bool keep_dims) {
    return execute_reduction_operation(OpType::Mean, input, axis, keep_dims);
}

Tensor max(const Tensor& input, const std::vector<int>& axis, bool keep_dims) {
    return execute_reduction_operation(OpType::Max, input, axis, keep_dims);
}

Tensor min(const Tensor& input, const std::vector<int>& axis, bool keep_dims) {
    return execute_reduction_operation(OpType::Min, input, axis, keep_dims);
}

Tensor argmax(const Tensor& input, int axis, bool keep_dims) {
    return execute_reduction_operation(OpType::ArgMax, input, {axis}, keep_dims);
}

Tensor argmin(const Tensor& input, int axis, bool keep_dims) {
    return execute_reduction_operation(OpType::ArgMin, input, {axis}, keep_dims);
}

void execute_binary_inplace(OpType op_type, Tensor& lhs, const Tensor& rhs) {
    auto device = lhs.device(); // In-place ops run on the device of the lhs
    auto op = OperationRegistry::get_operation(op_type, device);
    if (!op) {
        throw std::runtime_error("Operation not available for the given device.");
    }
    op->execute_binary_inplace(lhs, rhs);
}

// In-place operations
void add_inplace(Tensor& lhs, const Tensor& rhs) { execute_binary_inplace(OpType::Add, lhs, rhs); }
void subtract_inplace(Tensor& lhs, const Tensor& rhs) { execute_binary_inplace(OpType::Subtract, lhs, rhs); }
void multiply_inplace(Tensor& lhs, const Tensor& rhs) { execute_binary_inplace(OpType::Multiply, lhs, rhs); }
void divide_inplace(Tensor& lhs, const Tensor& rhs) { execute_binary_inplace(OpType::Divide, lhs, rhs); }

// Matrix multiplication
Tensor matmul(const Tensor& a, const Tensor& b, bool transpose_a, bool transpose_b) {
    return execute_matmul_operation(a, b, transpose_a, transpose_b);
}

}  // namespace ops
}  // namespace axiom 