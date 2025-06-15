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

#define DISPATCH_CPU_BINARY_OP(TYPE_ENUM, TYPE) \
  case TYPE_ENUM: \
    execute_binary_typed<TYPE>(lhs, rhs, result); \
    break;

  switch (result_dtype) {
    DISPATCH_CPU_BINARY_OP(DType::Float32, float)
    DISPATCH_CPU_BINARY_OP(DType::Float64, double)
    DISPATCH_CPU_BINARY_OP(DType::Float16, float16_t)
    DISPATCH_CPU_BINARY_OP(DType::Int8, int8_t)
    DISPATCH_CPU_BINARY_OP(DType::Int16, int16_t)
    DISPATCH_CPU_BINARY_OP(DType::Int32, int32_t)
    DISPATCH_CPU_BINARY_OP(DType::Int64, int64_t)
    DISPATCH_CPU_BINARY_OP(DType::UInt8, uint8_t)
    DISPATCH_CPU_BINARY_OP(DType::UInt16, uint16_t)
    DISPATCH_CPU_BINARY_OP(DType::UInt32, uint32_t)
    DISPATCH_CPU_BINARY_OP(DType::UInt64, uint64_t)
    case DType::Bool:
      execute_binary_typed<bool>(lhs, rhs, result);
      break;
    case DType::Complex64: // Fallthrough
    case DType::Complex128:
      throw std::runtime_error("Complex types are not yet supported by CPU operations.");
    default:
      throw std::runtime_error("Unsupported data type for CPU operations");
  }
#undef DISPATCH_CPU_BINARY_OP

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
                                       lhs.shape(), rhs.shape(), result_shape,
                                       lhs_float.strides(), rhs_float.strides());
  } else {
    Tensor lhs_converted = lhs.astype(result.dtype());
    Tensor rhs_converted = rhs.astype(result.dtype());
    
    const T* lhs_data = lhs_converted.template typed_data<T>();
    const T* rhs_data = rhs_converted.template typed_data<T>();
    T* result_data = result.template typed_data<T>();
    
    execute_broadcast_loop<T, T>(lhs_data, rhs_data, result_data,
                                lhs.shape(), rhs.shape(), result_shape,
                                lhs_converted.strides(), rhs_converted.strides());
  }
}

template<typename Func>
template<typename InputT, typename OutputT>
void CPUBinaryOperation<Func>::execute_broadcast_loop(const InputT* lhs_data, const InputT* rhs_data, OutputT* result_data,
                                                     const Shape& lhs_shape, const Shape& rhs_shape, const Shape& result_shape,
                                                     const Strides& lhs_strides_in, const Strides& rhs_strides_in) const {
    size_t total_elements = ShapeUtils::size(result_shape);
    size_t ndim = result_shape.size();

    // Prepare broadcasted strides
    Strides lhs_bcast_strides(ndim, 0);
    Strides rhs_bcast_strides(ndim, 0);
    
    int lhs_dim_offset = ndim - lhs_shape.size();
    for (size_t i = 0; i < lhs_shape.size(); ++i) {
        if (lhs_shape[i] != 1) {
            lhs_bcast_strides[i + lhs_dim_offset] = lhs_strides_in[i];
        }
    }

    int rhs_dim_offset = ndim - rhs_shape.size();
    for (size_t i = 0; i < rhs_shape.size(); ++i) {
        if (rhs_shape[i] != 1) {
            rhs_bcast_strides[i + rhs_dim_offset] = rhs_strides_in[i];
        }
    }

    std::vector<size_t> result_coords(ndim, 0);

    for (size_t i = 0; i < total_elements; ++i) {
        size_t lhs_byte_offset = 0;
        size_t rhs_byte_offset = 0;

        for (size_t j = 0; j < ndim; ++j) {
            lhs_byte_offset += result_coords[j] * lhs_bcast_strides[j];
            rhs_byte_offset += result_coords[j] * rhs_bcast_strides[j];
        }
        
        const auto& lhs_val = *reinterpret_cast<const InputT*>(reinterpret_cast<const uint8_t*>(lhs_data) + lhs_byte_offset);
        const auto& rhs_val = *reinterpret_cast<const InputT*>(reinterpret_cast<const uint8_t*>(rhs_data) + rhs_byte_offset);
        
        result_data[i] = func_(lhs_val, rhs_val);

        // Increment coordinates
        for (int j = ndim - 1; j >= 0; --j) {
            if (++result_coords[j] < result_shape[j]) {
                break;
            }
            result_coords[j] = 0;
        }
    }
}

template<typename Func>
void CPUBinaryOperation<Func>::execute_binary_inplace(Tensor& lhs, const Tensor& rhs) const {
    // In-place operations require lhs to be writeable
    if (!lhs.flags().writeable) {
        throw std::runtime_error("Cannot perform in-place operation on a non-writeable tensor.");
    }

    // Check for type safety. In-place ops do not promote the lhs tensor.
    DType promoted_dtype = ops::promote_types(lhs.dtype(), rhs.dtype());
    if (promoted_dtype != lhs.dtype()) {
        throw std::runtime_error("In-place operation would require unsafe type casting.");
    }

    // Check for broadcast safety. In-place ops cannot change the lhs shape.
    if (!ops::are_broadcastable(lhs.shape(), rhs.shape()) || 
        ops::compute_broadcast_info(lhs.shape(), rhs.shape()).result_shape != lhs.shape()) {
        throw std::runtime_error("In-place operation with broadcasting cannot change tensor shape.");
    }

    // Dispatch to the typed implementation
    #define DISPATCH_CPU_INPLACE_OP(TYPE_ENUM, TYPE) \
        case TYPE_ENUM: \
            execute_inplace_typed<TYPE>(lhs, rhs); \
            break;

    switch (lhs.dtype()) {
        DISPATCH_CPU_INPLACE_OP(DType::Float32, float)
        DISPATCH_CPU_INPLACE_OP(DType::Float64, double)
        DISPATCH_CPU_INPLACE_OP(DType::Float16, float16_t)
        DISPATCH_CPU_INPLACE_OP(DType::Int8, int8_t)
        DISPATCH_CPU_INPLACE_OP(DType::Int16, int16_t)
        DISPATCH_CPU_INPLACE_OP(DType::Int32, int32_t)
        DISPATCH_CPU_INPLACE_OP(DType::Int64, int64_t)
        DISPATCH_CPU_INPLACE_OP(DType::UInt8, uint8_t)
        DISPATCH_CPU_INPLACE_OP(DType::UInt16, uint16_t)
        DISPATCH_CPU_INPLACE_OP(DType::UInt32, uint32_t)
        DISPATCH_CPU_INPLACE_OP(DType::UInt64, uint64_t)
        DISPATCH_CPU_INPLACE_OP(DType::Bool, bool)
        default:
            throw std::runtime_error("Unsupported data type for CPU in-place operations");
    }
    #undef DISPATCH_CPU_INPLACE_OP
}

template<typename Func>
template<typename T>
void CPUBinaryOperation<Func>::execute_inplace_typed(Tensor& lhs, const Tensor& rhs) const {
    if (lhs.shape() == rhs.shape()) {
        T* lhs_data = lhs.template typed_data<T>();
        const T* rhs_data = rhs.template typed_data<T>();
        for (size_t i = 0; i < lhs.size(); ++i) {
            lhs_data[i] = func_(lhs_data[i], rhs_data[i]);
        }
    } else {
        execute_inplace_broadcast<T>(lhs, rhs);
    }
}

template<typename Func>
template<typename T>
void CPUBinaryOperation<Func>::execute_inplace_broadcast(Tensor& lhs, const Tensor& rhs) const {
    // This simplified version only handles broadcasting from a scalar
    if (rhs.size() != 1) {
        throw std::runtime_error("In-place broadcasting is only supported for scalar rhs.");
    }
    T* lhs_data = lhs.template typed_data<T>();
    const T rhs_val = *rhs.template typed_data<T>();
    for (size_t i = 0; i < lhs.size(); ++i) {
        lhs_data[i] = func_(lhs_data[i], rhs_val);
    }
}

// ============================================================================
// CPU Unary Operation Implementation
// ============================================================================

template<typename Func>
Tensor CPUUnaryOperation<Func>::execute_unary(const Tensor& input) const {
  if (input.device() != Device::CPU) {
    throw std::runtime_error("CPU operations require CPU tensors");
  }

  // Unary ops usually return the same dtype as input, except for some functions
  // which might promote it (e.g., if we had a function that always returns float)
  DType result_dtype = input.dtype();
  Tensor result(input.shape(), result_dtype, Device::CPU);

#define DISPATCH_CPU_UNARY_OP(TYPE_ENUM, TYPE) \
  case TYPE_ENUM: \
    execute_unary_typed<TYPE>(input, result); \
    break;

  switch (result_dtype) {
    DISPATCH_CPU_UNARY_OP(DType::Float32, float)
    DISPATCH_CPU_UNARY_OP(DType::Float64, double)
    DISPATCH_CPU_UNARY_OP(DType::Float16, float16_t)
    DISPATCH_CPU_UNARY_OP(DType::Int8, int8_t)
    DISPATCH_CPU_UNARY_OP(DType::Int16, int16_t)
    DISPATCH_CPU_UNARY_OP(DType::Int32, int32_t)
    DISPATCH_CPU_UNARY_OP(DType::Int64, int64_t)
    DISPATCH_CPU_UNARY_OP(DType::UInt8, uint8_t)
    DISPATCH_CPU_UNARY_OP(DType::UInt16, uint16_t)
    DISPATCH_CPU_UNARY_OP(DType::UInt32, uint32_t)
    DISPATCH_CPU_UNARY_OP(DType::UInt64, uint64_t)
    DISPATCH_CPU_UNARY_OP(DType::Bool, bool)
    case DType::Complex64: // Fallthrough
    case DType::Complex128:
      throw std::runtime_error("Complex types are not yet supported by CPU operations.");
    default:
      throw std::runtime_error("Unsupported data type for CPU operations");
  }
#undef DISPATCH_CPU_UNARY_OP

  return result;
}

template<typename Func>
template<typename T>
void CPUUnaryOperation<Func>::execute_unary_typed(const Tensor& input, Tensor& result) const {
  size_t total_elements = input.size();
  const T* input_data = input.template typed_data<T>();
  T* result_data = result.template typed_data<T>();

  for (size_t i = 0; i < total_elements; ++i) {
    result_data[i] = func_(input_data[i]);
  }
}

// ============================================================================
// CPU Reduction Operation Implementation
// ============================================================================
namespace { // Anonymous namespace for helpers

Shape calculate_reduction_shape(const Shape& input_shape, const std::vector<int>& axes, bool keep_dims) {
    if (axes.empty()) {
        return keep_dims ? Shape(input_shape.size(), 1) : Shape{1};
    }

    Shape output_shape;
    std::vector<bool> is_reduced_axis(input_shape.size(), false);
    for (int axis : axes) {
        is_reduced_axis[axis] = true;
    }

    for (size_t i = 0; i < input_shape.size(); ++i) {
        if (is_reduced_axis[i]) {
            if (keep_dims) {
                output_shape.push_back(1);
            }
        } else {
            output_shape.push_back(input_shape[i]);
        }
    }
    return output_shape;
}
} // namespace

template<typename Func>
Tensor CPUReductionOperation<Func>::execute_reduction(const Tensor& input, const std::vector<int>& axis, bool keep_dims) const {
    if (input.device() != Device::CPU) {
        throw std::runtime_error("CPU operations require CPU tensors");
    }

    // For now, we only support float32 for reductions
    DType result_dtype = DType::Float32;
    if (input.dtype() != DType::Float32) {
       // In future, we can support more types
    }

    return execute_reduction_typed<float>(input, axis, keep_dims);
}

template<typename Func>
template<typename T>
void CPUReductionOperation<Func>::reduction_recursive_helper(const Tensor& input, Tensor& result, const std::vector<int>& axes,
                                                              std::vector<size_t>& current_coords, int current_dim, const Func& func, bool keep_dims) {
    if (current_dim == static_cast<int>(input.ndim())) {
        std::vector<size_t> result_coords;
        if (keep_dims) {
            result_coords = current_coords;
            for (int axis : axes) {
                result_coords[axis] = 0;
            }
        } else {
            for (size_t i = 0; i < input.ndim(); ++i) {
                bool is_reduced = false;
                for (int axis : axes) {
                    if (i == static_cast<size_t>(axis)) {
                        is_reduced = true;
                        break;
                    }
                }
                if (!is_reduced) {
                    result_coords.push_back(current_coords[i]);
                }
            }
        }
        
        if (result_coords.empty()) {
            result_coords.push_back(0);
        }

        size_t result_offset = ShapeUtils::linear_index(result_coords, result.strides()) / result.itemsize();
        T& result_val = result.template typed_data<T>()[result_offset];
        
        size_t input_offset = ShapeUtils::linear_index(current_coords, input.strides()) / input.itemsize();
        const T& input_val = input.template typed_data<T>()[input_offset];

        result_val = func(result_val, input_val);
        return;
    }

    for (size_t i = 0; i < input.shape()[current_dim]; ++i) {
        current_coords[current_dim] = i;
        reduction_recursive_helper<T>(input, result, axes, current_coords, current_dim + 1, func, keep_dims);
    }
}

template<typename Func>
template<typename T>
Tensor CPUReductionOperation<Func>::execute_reduction_typed(const Tensor& input, const std::vector<int>& axes, bool keep_dims) const {
    auto input_float = input.astype(DType::Float32);
    Shape result_shape = calculate_reduction_shape(input.shape(), axes, keep_dims);
    Tensor result = Tensor::full(result_shape, Func::template identity<T>());
    result.fill(Func::template identity<T>());

    std::vector<int> norm_axes = axes;
    if (norm_axes.empty()) {
        for(size_t i = 0; i < input.ndim(); ++i) norm_axes.push_back(i);
    }
    
    std::vector<size_t> current_coords(input.ndim(), 0);
    reduction_recursive_helper<T>(input_float, result, norm_axes, current_coords, 0, func_, keep_dims);


    if (op_type_ == ops::OpType::Mean) {
        size_t reduction_size = 1;
        for(int axis : norm_axes) {
            reduction_size *= input.shape()[axis];
        }

        T* result_data = result.template typed_data<T>();
        for(size_t i = 0; i < result.size(); ++i) {
            result_data[i] /= static_cast<T>(reduction_size);
        }
    }

    return result;
}

// ============================================================================
// Factory functions
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

  // Register unary operations
  OperationRegistry::register_operation(OpType::Negate, Device::CPU, std::make_unique<CPUUnaryOperation<NegateFunc>>(OpType::Negate, "negate", NegateFunc{}));
  OperationRegistry::register_operation(OpType::Abs, Device::CPU, std::make_unique<CPUUnaryOperation<AbsFunc>>(OpType::Abs, "abs", AbsFunc{}));
  OperationRegistry::register_operation(OpType::Sqrt, Device::CPU, std::make_unique<CPUUnaryOperation<SqrtFunc>>(OpType::Sqrt, "sqrt", SqrtFunc{}));
  OperationRegistry::register_operation(OpType::Exp, Device::CPU, std::make_unique<CPUUnaryOperation<ExpFunc>>(OpType::Exp, "exp", ExpFunc{}));
  OperationRegistry::register_operation(OpType::Log, Device::CPU, std::make_unique<CPUUnaryOperation<LogFunc>>(OpType::Log, "log", LogFunc{}));
  OperationRegistry::register_operation(OpType::Sin, Device::CPU, std::make_unique<CPUUnaryOperation<SinFunc>>(OpType::Sin, "sin", SinFunc{}));
  OperationRegistry::register_operation(OpType::Cos, Device::CPU, std::make_unique<CPUUnaryOperation<CosFunc>>(OpType::Cos, "cos", CosFunc{}));
  OperationRegistry::register_operation(OpType::Tan, Device::CPU, std::make_unique<CPUUnaryOperation<TanFunc>>(OpType::Tan, "tan", TanFunc{}));

  // Register reduction operations
  OperationRegistry::register_operation(OpType::Sum, Device::CPU, std::make_unique<CPUReductionOperation<SumFunc>>(OpType::Sum, "sum", SumFunc{}));
  OperationRegistry::register_operation(OpType::Mean, Device::CPU, std::make_unique<CPUReductionOperation<SumFunc>>(OpType::Mean, "mean", SumFunc{}));
  OperationRegistry::register_operation(OpType::Max, Device::CPU, std::make_unique<CPUReductionOperation<MaxFunc>>(OpType::Max, "max", MaxFunc{}));
  OperationRegistry::register_operation(OpType::Min, Device::CPU, std::make_unique<CPUReductionOperation<MinFunc>>(OpType::Min, "min", MinFunc{}));
}

void add(Tensor& a, const Tensor& b) {
  // Example of a potential external function if needed
}

}  // namespace cpu
}  // namespace backends
}  // namespace axiom 