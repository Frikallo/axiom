#include "cpu_operations.hpp"
#include "axiom/tensor.hpp"
#include "axiom/shape.hpp"
#include "axiom/error.hpp"
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
    throw DeviceError::cpu_only("CPU binary operations");
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
      throw TypeError::unsupported_dtype(dtype_name(result_dtype), "CPU binary operations");
    default:
      throw TypeError::unsupported_dtype(dtype_name(result_dtype), "CPU binary operations");
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
        throw MemoryError("Cannot perform in-place operation on a non-writeable tensor");
    }

    // Check for type safety. In-place ops do not promote the lhs tensor.
    DType promoted_dtype = ops::promote_types(lhs.dtype(), rhs.dtype());
    if (promoted_dtype != lhs.dtype()) {
        throw TypeError("In-place operation would require unsafe type casting from " + 
                       dtype_name(lhs.dtype()) + " to " + dtype_name(promoted_dtype));
    }

    // Check for broadcast safety. In-place ops cannot change the lhs shape.
    if (!ops::are_broadcastable(lhs.shape(), rhs.shape()) || 
        ops::compute_broadcast_info(lhs.shape(), rhs.shape()).result_shape != lhs.shape()) {
        throw ShapeError("In-place operation with broadcasting cannot change tensor shape");
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
            throw TypeError::unsupported_dtype(dtype_name(lhs.dtype()), "CPU in-place operations");
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
    const Shape& lhs_shape = lhs.shape();
    const Shape& rhs_shape = rhs.shape();
    size_t lhs_ndim = lhs_shape.size();
    size_t rhs_ndim = rhs_shape.size();

    // Prepare broadcasted strides for rhs (lhs strides remain as-is)
    Strides rhs_bcast_strides(lhs_ndim, 0);
    size_t rhs_dim_offset = lhs_ndim - rhs_ndim;

    for (size_t i = 0; i < rhs_ndim; ++i) {
        if (rhs_shape[i] != 1) {
            rhs_bcast_strides[i + rhs_dim_offset] = rhs.strides()[i];
        }
    }

    T* lhs_data = lhs.template typed_data<T>();
    const T* rhs_data = rhs.template typed_data<T>();
    const Strides& lhs_strides = lhs.strides();
    size_t total_elements = lhs.size();

    std::vector<size_t> coords(lhs_ndim, 0);

    for (size_t i = 0; i < total_elements; ++i) {
        size_t lhs_byte_offset = 0;
        size_t rhs_byte_offset = 0;

        for (size_t j = 0; j < lhs_ndim; ++j) {
            lhs_byte_offset += coords[j] * lhs_strides[j];
            rhs_byte_offset += coords[j] * rhs_bcast_strides[j];
        }

        T& lhs_val = *reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(lhs_data) + lhs_byte_offset);
        const T& rhs_val = *reinterpret_cast<const T*>(reinterpret_cast<const uint8_t*>(rhs_data) + rhs_byte_offset);

        lhs_val = func_(lhs_val, rhs_val);

        // Increment coordinates
        for (int j = static_cast<int>(lhs_ndim) - 1; j >= 0; --j) {
            if (++coords[j] < lhs_shape[j]) {
                break;
            }
            coords[j] = 0;
        }
    }
}

// ============================================================================
// CPU Unary Operation Implementation
// ============================================================================

template<typename Func>
Tensor CPUUnaryOperation<Func>::execute_unary(const Tensor& input) const {
  if (input.device() != Device::CPU) {
    throw DeviceError::cpu_only("CPU unary operations");
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
      throw TypeError::unsupported_dtype(dtype_name(result_dtype), "CPU unary operations");
    default:
      throw TypeError::unsupported_dtype(dtype_name(result_dtype), "CPU unary operations");
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
        throw DeviceError::cpu_only("CPU reduction operations");
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
    // Only convert if necessary - avoid unnecessary copies
    const Tensor& input_typed = (input.dtype() == DType::Float32) ? input : input.astype(DType::Float32);

    Shape result_shape = calculate_reduction_shape(input.shape(), axes, keep_dims);

    // Create result tensor initialized with identity value
    Tensor result(result_shape, DType::Float32, Device::CPU);
    result.fill(Func::template identity<T>());

    std::vector<int> norm_axes = axes;
    if (norm_axes.empty()) {
        for(size_t i = 0; i < input.ndim(); ++i) norm_axes.push_back(static_cast<int>(i));
    }

    std::vector<size_t> current_coords(input.ndim(), 0);
    reduction_recursive_helper<T>(input_typed, result, norm_axes, current_coords, 0, func_, keep_dims);

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
// CPU MatMul Operation Implementation
// ============================================================================

void CPUMatMulOperation::get_matmul_dims(const Tensor& a, const Tensor& b,
                                          bool transpose_a, bool transpose_b,
                                          size_t& M, size_t& N, size_t& K,
                                          size_t& K_b) {
    size_t a_ndim = a.ndim();
    size_t b_ndim = b.ndim();

    // For 1D tensors, treat as row/column vector
    size_t a_rows, a_cols, b_rows, b_cols;

    if (a_ndim == 1) {
        a_rows = 1;
        a_cols = a.shape()[0];
    } else {
        a_rows = a.shape()[a_ndim - 2];
        a_cols = a.shape()[a_ndim - 1];
    }

    if (b_ndim == 1) {
        b_rows = b.shape()[0];
        b_cols = 1;
    } else {
        b_rows = b.shape()[b_ndim - 2];
        b_cols = b.shape()[b_ndim - 1];
    }

    // Apply transpose flags
    if (transpose_a) std::swap(a_rows, a_cols);
    if (transpose_b) std::swap(b_rows, b_cols);

    M = a_rows;
    K = a_cols;
    K_b = b_rows;
    N = b_cols;
}

Shape CPUMatMulOperation::compute_batch_shape(const Tensor& a, const Tensor& b) {
    // Get batch dimensions (all dims except last 2)
    size_t a_batch_dims = a.ndim() > 2 ? a.ndim() - 2 : 0;
    size_t b_batch_dims = b.ndim() > 2 ? b.ndim() - 2 : 0;

    Shape a_batch, b_batch;
    for (size_t i = 0; i < a_batch_dims; ++i) a_batch.push_back(a.shape()[i]);
    for (size_t i = 0; i < b_batch_dims; ++i) b_batch.push_back(b.shape()[i]);

    // Broadcast batch dimensions
    return ShapeUtils::broadcast_shape(a_batch, b_batch);
}

template<typename T>
void CPUMatMulOperation::matmul_2d(const T* a_data, const T* b_data, T* c_data,
                                    size_t M, size_t N, size_t K,
                                    size_t a_row_stride, size_t a_col_stride,
                                    size_t b_row_stride, size_t b_col_stride,
                                    size_t c_row_stride, size_t c_col_stride) {
    // Standard triple-loop matrix multiplication
    // This handles arbitrary strides for transposed views
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            T sum = T(0);
            for (size_t k = 0; k < K; ++k) {
                T a_val = a_data[i * a_row_stride + k * a_col_stride];
                T b_val = b_data[k * b_row_stride + j * b_col_stride];
                sum += a_val * b_val;
            }
            c_data[i * c_row_stride + j * c_col_stride] = sum;
        }
    }
}

template<typename T>
Tensor CPUMatMulOperation::execute_matmul_typed(const Tensor& a, const Tensor& b,
                                                 bool transpose_a, bool transpose_b) const {
    size_t M, N, K, K_b;
    get_matmul_dims(a, b, transpose_a, transpose_b, M, N, K, K_b);

    if (K != K_b) {
        throw ShapeError(
            "MatMul dimension mismatch: A has " + std::to_string(K) +
            " columns but B has " + std::to_string(K_b) + " rows");
    }

    size_t a_ndim = a.ndim();
    size_t b_ndim = b.ndim();

    // Compute output shape with broadcasted batch dimensions
    Shape result_shape;

    if (a_ndim > 2 || b_ndim > 2) {
        Shape batch_shape = compute_batch_shape(a, b);
        result_shape = batch_shape;
    }

    // Handle 1D cases for output shape
    if (a_ndim == 1 && b_ndim == 1) {
        // Vector dot product: returns scalar
        result_shape = {};
    } else if (a_ndim == 1) {
        // (K,) @ (..., K, N) -> (..., N)
        result_shape.push_back(N);
    } else if (b_ndim == 1) {
        // (..., M, K) @ (K,) -> (..., M)
        result_shape.push_back(M);
    } else {
        // Standard case: (..., M, K) @ (..., K, N) -> (..., M, N)
        result_shape.push_back(M);
        result_shape.push_back(N);
    }

    // Scalar result (rank-0 tensor) is handled naturally with empty shape
    Tensor result = Tensor::zeros(result_shape, a.dtype(), Device::CPU);

    // Get strides for the matrix dimensions
    size_t a_itemsize = a.itemsize();
    size_t b_itemsize = b.itemsize();
    size_t c_itemsize = result.itemsize();

    // Calculate element strides (converting byte strides to element strides)
    size_t a_row_stride, a_col_stride;
    size_t b_row_stride, b_col_stride;
    size_t c_row_stride, c_col_stride;

    if (a_ndim == 1) {
        a_row_stride = 0;
        a_col_stride = a.strides()[0] / a_itemsize;
    } else {
        a_row_stride = a.strides()[a_ndim - 2] / a_itemsize;
        a_col_stride = a.strides()[a_ndim - 1] / a_itemsize;
    }

    if (b_ndim == 1) {
        b_row_stride = b.strides()[0] / b_itemsize;
        b_col_stride = 0;
    } else {
        b_row_stride = b.strides()[b_ndim - 2] / b_itemsize;
        b_col_stride = b.strides()[b_ndim - 1] / b_itemsize;
    }

    // Handle transpose via stride swapping (zero-copy!)
    if (transpose_a) std::swap(a_row_stride, a_col_stride);
    if (transpose_b) std::swap(b_row_stride, b_col_stride);

    size_t result_ndim = result.ndim();
    if (result_ndim >= 2) {
        c_row_stride = result.strides()[result_ndim - 2] / c_itemsize;
        c_col_stride = result.strides()[result_ndim - 1] / c_itemsize;
    } else if (result_ndim == 1) {
        // For 1D result (from 1D @ 2D), treat as row vector: c_row=0, c_col=stride
        // Or as column vector (from 2D @ 1D): c_row=stride, c_col=0
        // We need to determine which based on input shapes
        if (a_ndim == 1 && b_ndim >= 2) {
            // (K,) @ (..., K, N) -> (..., N) - result is conceptually a row, so col varies
            c_row_stride = 0;
            c_col_stride = result.strides()[0] / c_itemsize;
        } else {
            // (..., M, K) @ (K,) -> (..., M) - result is conceptually a column, so row varies
            c_row_stride = result.strides()[0] / c_itemsize;
            c_col_stride = 0;
        }
    } else {
        c_row_stride = 0;
        c_col_stride = 0;
    }

    const T* a_base = a.typed_data<T>();
    const T* b_base = b.typed_data<T>();
    T* c_base = result.typed_data<T>();

    // For simple 2D case without batching
    if (a_ndim <= 2 && b_ndim <= 2) {
        matmul_2d<T>(a_base, b_base, c_base, M, N, K,
                     a_row_stride, a_col_stride,
                     b_row_stride, b_col_stride,
                     c_row_stride, c_col_stride);
    } else {
        // Batch matmul with broadcasting
        Shape batch_shape = compute_batch_shape(a, b);
        size_t batch_size = ShapeUtils::size(batch_shape);
        size_t batch_ndim = batch_shape.size();

        // Compute batch strides for a, b, c
        Strides a_batch_strides(batch_ndim, 0);
        Strides b_batch_strides(batch_ndim, 0);
        Strides c_batch_strides(batch_ndim, 0);

        size_t a_batch_offset = batch_ndim - (a_ndim > 2 ? a_ndim - 2 : 0);
        size_t b_batch_offset = batch_ndim - (b_ndim > 2 ? b_ndim - 2 : 0);

        for (size_t i = 0; i < batch_ndim; ++i) {
            if (i >= a_batch_offset && a_ndim > 2) {
                size_t a_dim_idx = i - a_batch_offset;
                if (a.shape()[a_dim_idx] != 1) {
                    a_batch_strides[i] = a.strides()[a_dim_idx] / a_itemsize;
                }
            }
            if (i >= b_batch_offset && b_ndim > 2) {
                size_t b_dim_idx = i - b_batch_offset;
                if (b.shape()[b_dim_idx] != 1) {
                    b_batch_strides[i] = b.strides()[b_dim_idx] / b_itemsize;
                }
            }
            c_batch_strides[i] = result.strides()[i] / c_itemsize;
        }

        // Iterate over batch dimensions
        std::vector<size_t> batch_coords(batch_ndim, 0);
        for (size_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
            // Compute batch offsets
            size_t a_batch_off = 0, b_batch_off = 0, c_batch_off = 0;
            for (size_t i = 0; i < batch_ndim; ++i) {
                a_batch_off += batch_coords[i] * a_batch_strides[i];
                b_batch_off += batch_coords[i] * b_batch_strides[i];
                c_batch_off += batch_coords[i] * c_batch_strides[i];
            }

            matmul_2d<T>(a_base + a_batch_off, b_base + b_batch_off, c_base + c_batch_off,
                         M, N, K,
                         a_row_stride, a_col_stride,
                         b_row_stride, b_col_stride,
                         c_row_stride, c_col_stride);

            // Increment batch coordinates
            for (int i = batch_ndim - 1; i >= 0; --i) {
                if (++batch_coords[i] < batch_shape[i]) break;
                batch_coords[i] = 0;
            }
        }
    }

    return result;
}

Tensor CPUMatMulOperation::execute_matmul(const Tensor& a, const Tensor& b,
                                           bool transpose_a, bool transpose_b) const {
    if (a.device() != Device::CPU || b.device() != Device::CPU) {
        throw DeviceError::cpu_only("CPU MatMul");
    }

    if (a.ndim() == 0 || b.ndim() == 0) {
        throw ShapeError("MatMul does not support 0-dimensional tensors");
    }

    // Type promote and dispatch
    DType result_dtype = ops::promote_types(a.dtype(), b.dtype());
    Tensor a_promoted = (a.dtype() == result_dtype) ? a : a.astype(result_dtype);
    Tensor b_promoted = (b.dtype() == result_dtype) ? b : b.astype(result_dtype);

#define DISPATCH_MATMUL(DTYPE, CTYPE) \
    case DTYPE: return execute_matmul_typed<CTYPE>(a_promoted, b_promoted, transpose_a, transpose_b);

    switch (result_dtype) {
        DISPATCH_MATMUL(DType::Float32, float)
        DISPATCH_MATMUL(DType::Float64, double)
        DISPATCH_MATMUL(DType::Int32, int32_t)
        DISPATCH_MATMUL(DType::Int64, int64_t)
        default:
            throw TypeError::unsupported_dtype(dtype_name(result_dtype), "MatMul");
    }
#undef DISPATCH_MATMUL
}

// ============================================================================
// CPU ArgMax/ArgMin Operation Implementation
// ============================================================================

template<typename T>
Tensor CPUArgMaxOperation::execute_argmax_typed(const Tensor& input, int axis, bool keep_dims) const {
    size_t ndim = input.ndim();

    // Normalize axis
    if (axis < 0) axis += static_cast<int>(ndim);
    if (axis < 0 || axis >= static_cast<int>(ndim)) {
        throw ShapeError::invalid_axis(axis, ndim);
    }

    // Calculate output shape
    Shape output_shape;
    for (size_t i = 0; i < ndim; ++i) {
        if (static_cast<int>(i) == axis) {
            if (keep_dims) output_shape.push_back(1);
        } else {
            output_shape.push_back(input.shape()[i]);
        }
    }
    // Scalar result (rank-0 tensor) is handled naturally with empty shape

    // Create output tensor with Int64 dtype for indices
    Tensor result = Tensor::zeros(output_shape, DType::Int64, Device::CPU);
    int64_t* result_data = result.typed_data<int64_t>();
    const T* input_data = input.typed_data<T>();

    // Calculate sizes
    size_t outer_size = 1;
    for (int i = 0; i < axis; ++i) outer_size *= input.shape()[i];

    size_t axis_size = input.shape()[axis];

    size_t inner_size = 1;
    for (size_t i = axis + 1; i < ndim; ++i) inner_size *= input.shape()[i];

    // Get strides in elements
    size_t axis_stride = input.strides()[axis] / input.itemsize();

    // Iterate over all positions
    for (size_t outer = 0; outer < outer_size; ++outer) {
        for (size_t inner = 0; inner < inner_size; ++inner) {
            // Find max along axis
            size_t best_idx = 0;
            T best_val = std::numeric_limits<T>::lowest();

            for (size_t k = 0; k < axis_size; ++k) {
                // Calculate input index
                std::vector<size_t> coords(ndim);
                size_t temp_outer = outer;
                for (int i = axis - 1; i >= 0; --i) {
                    coords[i] = temp_outer % input.shape()[i];
                    temp_outer /= input.shape()[i];
                }
                coords[axis] = k;
                size_t temp_inner = inner;
                for (int i = static_cast<int>(ndim) - 1; i > axis; --i) {
                    coords[i] = temp_inner % input.shape()[i];
                    temp_inner /= input.shape()[i];
                }

                size_t input_offset = 0;
                for (size_t i = 0; i < ndim; ++i) {
                    input_offset += coords[i] * (input.strides()[i] / input.itemsize());
                }

                T val = input_data[input_offset];
                if (val > best_val) {
                    best_val = val;
                    best_idx = k;
                }
            }

            // Store result
            size_t result_idx = outer * inner_size + inner;
            result_data[result_idx] = static_cast<int64_t>(best_idx);
        }
    }

    return result;
}

Tensor CPUArgMaxOperation::execute_reduction(const Tensor& input, const std::vector<int>& axis, bool keep_dims) const {
    if (input.device() != Device::CPU) {
        throw DeviceError::cpu_only("CPU ArgMax");
    }

    int ax = axis.empty() ? -1 : axis[0];

    // For full reduction (axis=-1 or all axes), flatten first
    if (ax == -1 || axis.size() > 1) {
        auto flat = input.flatten();
        ax = 0;
    }

#define DISPATCH_ARGMAX(DTYPE, CTYPE) \
    case DTYPE: return execute_argmax_typed<CTYPE>(input, ax, keep_dims);

    switch (input.dtype()) {
        DISPATCH_ARGMAX(DType::Float32, float)
        DISPATCH_ARGMAX(DType::Float64, double)
        DISPATCH_ARGMAX(DType::Int32, int32_t)
        DISPATCH_ARGMAX(DType::Int64, int64_t)
        default:
            throw TypeError::unsupported_dtype(dtype_name(input.dtype()), "ArgMax");
    }
#undef DISPATCH_ARGMAX
}

template<typename T>
Tensor CPUArgMinOperation::execute_argmin_typed(const Tensor& input, int axis, bool keep_dims) const {
    size_t ndim = input.ndim();

    // Normalize axis
    if (axis < 0) axis += static_cast<int>(ndim);
    if (axis < 0 || axis >= static_cast<int>(ndim)) {
        throw ShapeError::invalid_axis(axis, ndim);
    }

    // Calculate output shape
    Shape output_shape;
    for (size_t i = 0; i < ndim; ++i) {
        if (static_cast<int>(i) == axis) {
            if (keep_dims) output_shape.push_back(1);
        } else {
            output_shape.push_back(input.shape()[i]);
        }
    }
    // Scalar result (rank-0 tensor) is handled naturally with empty shape

    // Create output tensor with Int64 dtype for indices
    Tensor result = Tensor::zeros(output_shape, DType::Int64, Device::CPU);
    int64_t* result_data = result.typed_data<int64_t>();
    const T* input_data = input.typed_data<T>();

    // Calculate sizes
    size_t outer_size = 1;
    for (int i = 0; i < axis; ++i) outer_size *= input.shape()[i];

    size_t axis_size = input.shape()[axis];

    size_t inner_size = 1;
    for (size_t i = axis + 1; i < ndim; ++i) inner_size *= input.shape()[i];

    // Iterate over all positions
    for (size_t outer = 0; outer < outer_size; ++outer) {
        for (size_t inner = 0; inner < inner_size; ++inner) {
            // Find min along axis
            size_t best_idx = 0;
            T best_val = std::numeric_limits<T>::max();

            for (size_t k = 0; k < axis_size; ++k) {
                // Calculate input index
                std::vector<size_t> coords(ndim);
                size_t temp_outer = outer;
                for (int i = axis - 1; i >= 0; --i) {
                    coords[i] = temp_outer % input.shape()[i];
                    temp_outer /= input.shape()[i];
                }
                coords[axis] = k;
                size_t temp_inner = inner;
                for (int i = static_cast<int>(ndim) - 1; i > axis; --i) {
                    coords[i] = temp_inner % input.shape()[i];
                    temp_inner /= input.shape()[i];
                }

                size_t input_offset = 0;
                for (size_t i = 0; i < ndim; ++i) {
                    input_offset += coords[i] * (input.strides()[i] / input.itemsize());
                }

                T val = input_data[input_offset];
                if (val < best_val) {
                    best_val = val;
                    best_idx = k;
                }
            }

            // Store result
            size_t result_idx = outer * inner_size + inner;
            result_data[result_idx] = static_cast<int64_t>(best_idx);
        }
    }

    return result;
}

Tensor CPUArgMinOperation::execute_reduction(const Tensor& input, const std::vector<int>& axis, bool keep_dims) const {
    if (input.device() != Device::CPU) {
        throw DeviceError::cpu_only("CPU ArgMin");
    }

    int ax = axis.empty() ? -1 : axis[0];

    // For full reduction (axis=-1 or all axes), flatten first
    if (ax == -1 || axis.size() > 1) {
        auto flat = input.flatten();
        ax = 0;
    }

#define DISPATCH_ARGMIN(DTYPE, CTYPE) \
    case DTYPE: return execute_argmin_typed<CTYPE>(input, ax, keep_dims);

    switch (input.dtype()) {
        DISPATCH_ARGMIN(DType::Float32, float)
        DISPATCH_ARGMIN(DType::Float64, double)
        DISPATCH_ARGMIN(DType::Int32, int32_t)
        DISPATCH_ARGMIN(DType::Int64, int64_t)
        default:
            throw TypeError::unsupported_dtype(dtype_name(input.dtype()), "ArgMin");
    }
#undef DISPATCH_ARGMIN
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

  // Register argmax/argmin operations
  OperationRegistry::register_operation(OpType::ArgMax, Device::CPU, std::make_unique<CPUArgMaxOperation>());
  OperationRegistry::register_operation(OpType::ArgMin, Device::CPU, std::make_unique<CPUArgMinOperation>());

  // Register matrix multiplication operation
  OperationRegistry::register_operation(OpType::MatMul, Device::CPU, std::make_unique<CPUMatMulOperation>());
}

void add(Tensor& a, const Tensor& b) {
  // Example of a potential external function if needed
}

}  // namespace cpu
}  // namespace backends
}  // namespace axiom 