#include "cpu_operations.hpp"
#include "axiom/tensor.hpp"
#include "axiom/shape.hpp"
#include <stdexcept>
#include "axiom/operations.hpp"

namespace axiom {
namespace backends {
namespace cpu {

// ============================================================================
// Simple implementation focusing on basic types
// ============================================================================

template<typename Func>
Tensor CPUBinaryOperation<Func>::execute_binary(const Tensor& lhs, const Tensor& rhs) const {
  // Ensure tensors are on CPU
  if (lhs.device() != Device::CPU || rhs.device() != Device::CPU) {
    throw std::runtime_error("CPU operations require CPU tensors");
  }
  
  // Compute broadcast information
  auto broadcast_info = ops::compute_broadcast_info(lhs.shape(), rhs.shape());
  
  // Determine result type
  DType result_dtype = ops::result_type(lhs, rhs);
  
  // Handle comparison and logical operations - they always return bool
  if (op_type_ == ops::OpType::Equal || op_type_ == ops::OpType::NotEqual ||
      op_type_ == ops::OpType::Less || op_type_ == ops::OpType::LessEqual ||
      op_type_ == ops::OpType::Greater || op_type_ == ops::OpType::GreaterEqual ||
      op_type_ == ops::OpType::LogicalAnd || op_type_ == ops::OpType::LogicalOr ||
      op_type_ == ops::OpType::LogicalXor) {
    result_dtype = DType::Bool;
  }
  
  // Create result tensor
  Tensor result(broadcast_info.result_shape, result_dtype, Device::CPU);
  
  // For now, support Float32 primarily
  if (result_dtype == DType::Float32) {
    execute_binary_typed<float>(lhs, rhs, result);
  } else if (result_dtype == DType::Bool) {
    execute_binary_typed<bool>(lhs, rhs, result);
  } else if (result_dtype == DType::Int32) {
    execute_binary_typed<int32_t>(lhs, rhs, result);
  } else if (result_dtype == DType::Float64) {
    execute_binary_typed<double>(lhs, rhs, result);
  } else {
    throw std::runtime_error("Unsupported data type for CPU operations");
  }
  
  return result;
}

template<typename Func>
template<typename T>
void CPUBinaryOperation<Func>::execute_binary_typed(const Tensor& lhs, const Tensor& rhs, Tensor& result) const {
  auto broadcast_info = ops::compute_broadcast_info(lhs.shape(), rhs.shape());
  
  if (broadcast_info.needs_broadcast) {
    execute_binary_broadcast<T>(lhs, rhs, result, broadcast_info);
  } else {
    execute_binary_same_shape<T>(lhs, rhs, result);
  }
}

template<typename Func>
template<typename T>
void CPUBinaryOperation<Func>::execute_binary_same_shape(const Tensor& lhs, const Tensor& rhs, Tensor& result) const {
  size_t total_elements = result.size();
  
  // For comparison operations with bool output
  if constexpr (std::is_same_v<T, bool>) {
    // Convert inputs to Float32 for comparison
    Tensor lhs_float = lhs.astype(DType::Float32);
    Tensor rhs_float = rhs.astype(DType::Float32);
    
    const float* lhs_data = lhs_float.template typed_data<float>();
    const float* rhs_data = rhs_float.template typed_data<float>();
    bool* result_data = result.template typed_data<bool>();
    
    for (size_t i = 0; i < total_elements; ++i) {
      result_data[i] = func_(lhs_data[i], rhs_data[i]);
    }
  } else {
    // Convert tensors to result type
    Tensor lhs_converted = lhs.astype(result.dtype());
    Tensor rhs_converted = rhs.astype(result.dtype());
    
    const T* lhs_data = lhs_converted.template typed_data<T>();
    const T* rhs_data = rhs_converted.template typed_data<T>();
    T* result_data = result.template typed_data<T>();
    
    for (size_t i = 0; i < total_elements; ++i) {
      result_data[i] = func_(lhs_data[i], rhs_data[i]);
    }
  }
}

template<typename Func>
template<typename T>
void CPUBinaryOperation<Func>::execute_binary_broadcast(const Tensor& lhs, const Tensor& rhs, Tensor& result,
                                                       const ops::BroadcastInfo& broadcast_info) const {
  // For simplicity, implement basic broadcasting
  const Shape& result_shape = broadcast_info.result_shape;
  
  // For comparison operations with bool output
  if constexpr (std::is_same_v<T, bool>) {
    Tensor lhs_float = lhs.astype(DType::Float32);
    Tensor rhs_float = rhs.astype(DType::Float32);
    
    const float* lhs_data = lhs_float.template typed_data<float>();
    const float* rhs_data = rhs_float.template typed_data<float>();
    bool* result_data = result.template typed_data<bool>();
    
    execute_broadcast_loop<float, bool>(lhs_data, rhs_data, result_data,
                                       lhs.shape(), rhs.shape(), result_shape);
  } else {
    Tensor lhs_converted = lhs.astype(result.dtype());
    Tensor rhs_converted = rhs.astype(result.dtype());
    
    const T* lhs_data = lhs_converted.template typed_data<T>();
    const T* rhs_data = rhs_converted.template typed_data<T>();
    T* result_data = result.template typed_data<T>();
    
    execute_broadcast_loop<T, T>(lhs_data, rhs_data, result_data,
                                lhs.shape(), rhs.shape(), result_shape);
  }
}

template<typename Func>
template<typename InputT, typename OutputT>
void CPUBinaryOperation<Func>::execute_broadcast_loop(const InputT* lhs_data, const InputT* rhs_data, OutputT* result_data,
                                                     const Shape& lhs_shape, const Shape& rhs_shape, const Shape& result_shape) const {
  size_t total_elements = ShapeUtils::size(result_shape);
  
  for (size_t i = 0; i < total_elements; ++i) {
    // Simple broadcasting implementation
    size_t lhs_idx = 0;
    size_t rhs_idx = 0;
    
    // For now, handle simple cases where one tensor is scalar or same shape
    if (ShapeUtils::size(lhs_shape) == 1) {
      lhs_idx = 0;
    } else if (lhs_shape == result_shape) {
      lhs_idx = i;
    } else {
      lhs_idx = i % ShapeUtils::size(lhs_shape);
    }
    
    if (ShapeUtils::size(rhs_shape) == 1) {
      rhs_idx = 0;
    } else if (rhs_shape == result_shape) {
      rhs_idx = i;
    } else {
      rhs_idx = i % ShapeUtils::size(rhs_shape);
    }
    
    result_data[i] = func_(lhs_data[lhs_idx], rhs_data[rhs_idx]);
  }
}

// ============================================================================
// Registration function
// ============================================================================

void register_cpu_operations() {
  using namespace ops;
  
  // Register arithmetic operations
  OperationRegistry::register_operation(
    OpType::Add, Device::CPU, 
    std::make_unique<CPUBinaryOperation<AddFunc>>(OpType::Add, "add", AddFunc{}));
  
  OperationRegistry::register_operation(
    OpType::Subtract, Device::CPU, 
    std::make_unique<CPUBinaryOperation<SubtractFunc>>(OpType::Subtract, "subtract", SubtractFunc{}));
  
  OperationRegistry::register_operation(
    OpType::Multiply, Device::CPU, 
    std::make_unique<CPUBinaryOperation<MultiplyFunc>>(OpType::Multiply, "multiply", MultiplyFunc{}));
  
  OperationRegistry::register_operation(
    OpType::Divide, Device::CPU, 
    std::make_unique<CPUBinaryOperation<DivideFunc>>(OpType::Divide, "divide", DivideFunc{}));
  
  OperationRegistry::register_operation(
    OpType::Power, Device::CPU, 
    std::make_unique<CPUBinaryOperation<PowerFunc>>(OpType::Power, "power", PowerFunc{}));
  
  OperationRegistry::register_operation(
    OpType::Modulo, Device::CPU, 
    std::make_unique<CPUBinaryOperation<ModuloFunc>>(OpType::Modulo, "modulo", ModuloFunc{}));
  
  // Register comparison operations
  OperationRegistry::register_operation(
    OpType::Equal, Device::CPU, 
    std::make_unique<CPUBinaryOperation<EqualFunc>>(OpType::Equal, "equal", EqualFunc{}));
  
  OperationRegistry::register_operation(
    OpType::NotEqual, Device::CPU, 
    std::make_unique<CPUBinaryOperation<NotEqualFunc>>(OpType::NotEqual, "not_equal", NotEqualFunc{}));
  
  OperationRegistry::register_operation(
    OpType::Less, Device::CPU, 
    std::make_unique<CPUBinaryOperation<LessFunc>>(OpType::Less, "less", LessFunc{}));
  
  OperationRegistry::register_operation(
    OpType::LessEqual, Device::CPU, 
    std::make_unique<CPUBinaryOperation<LessEqualFunc>>(OpType::LessEqual, "less_equal", LessEqualFunc{}));
  
  OperationRegistry::register_operation(
    OpType::Greater, Device::CPU, 
    std::make_unique<CPUBinaryOperation<GreaterFunc>>(OpType::Greater, "greater", GreaterFunc{}));
  
  OperationRegistry::register_operation(
    OpType::GreaterEqual, Device::CPU, 
    std::make_unique<CPUBinaryOperation<GreaterEqualFunc>>(OpType::GreaterEqual, "greater_equal", GreaterEqualFunc{}));
  
  // Register logical operations
  OperationRegistry::register_operation(
    OpType::LogicalAnd, Device::CPU, 
    std::make_unique<CPUBinaryOperation<LogicalAndFunc>>(OpType::LogicalAnd, "logical_and", LogicalAndFunc{}));
  
  OperationRegistry::register_operation(
    OpType::LogicalOr, Device::CPU, 
    std::make_unique<CPUBinaryOperation<LogicalOrFunc>>(OpType::LogicalOr, "logical_or", LogicalOrFunc{}));
  
  OperationRegistry::register_operation(
    OpType::LogicalXor, Device::CPU, 
    std::make_unique<CPUBinaryOperation<LogicalXorFunc>>(OpType::LogicalXor, "logical_xor", LogicalXorFunc{}));
  
  // Register math operations
  OperationRegistry::register_operation(
    OpType::Maximum, Device::CPU, 
    std::make_unique<CPUBinaryOperation<MaximumFunc>>(OpType::Maximum, "maximum", MaximumFunc{}));
  
  OperationRegistry::register_operation(
    OpType::Minimum, Device::CPU, 
    std::make_unique<CPUBinaryOperation<MinimumFunc>>(OpType::Minimum, "minimum", MinimumFunc{}));
  
  OperationRegistry::register_operation(
    OpType::Atan2, Device::CPU, 
    std::make_unique<CPUBinaryOperation<Atan2Func>>(OpType::Atan2, "atan2", Atan2Func{}));
  
  OperationRegistry::register_operation(
    OpType::Hypot, Device::CPU, 
    std::make_unique<CPUBinaryOperation<HypotFunc>>(OpType::Hypot, "hypot", HypotFunc{}));
}

}  // namespace cpu
}  // namespace backends
}  // namespace axiom 