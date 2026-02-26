#pragma once

#include <cstddef>
#include <cstdint>

#ifdef AXIOM_CUDA_SUPPORT
#include <cuda_runtime.h>
#endif

namespace axiom {
namespace backends {
namespace cuda {

static constexpr int MAX_DIMS = 8;

struct GatherStridedParams {
    unsigned int ndim;
    unsigned int numel;
    unsigned int offset;   // Byte offset into source buffer
    unsigned int itemsize; // Size of each element in bytes
    unsigned int shape[MAX_DIMS];
    unsigned int src_strides[MAX_DIMS]; // Strides in ELEMENTS, always positive
    unsigned int flip_mask; // Bitmask: bit i set if axis i has negative stride
};

// Identifies which binary operation to launch.
enum class BinaryOpKind : uint8_t {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Mod,
    Max,
    Min,
    Atan2,
    Hypot,
    // Comparison ops — output is uint8_t (bool)
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    // Logical ops — inputs cast to bool, output uint8_t
    LogicalAnd,
    LogicalOr,
    LogicalXor,
    // Bitwise ops — integer types only, output same type
    BitwiseAnd,
    BitwiseOr,
    BitwiseXor,
    LeftShift,
    RightShift,
};

#ifdef AXIOM_CUDA_SUPPORT
// Launch gather_strided for the given dtype.  Dispatches to the correct
// template instantiation defined in cuda_kernels.cu.
void launch_gather_strided(const void *src, void *dst,
                           const GatherStridedParams &params,
                           size_t element_size, cudaStream_t stream);

// Launch a binary element-wise kernel (no broadcast, flat arrays).
// For arithmetic ops (Add..Hypot) src_a, src_b, and dst share the same
// element_size.  For comparison/logical ops dst is always uint8_t.
void launch_binary_elementwise(BinaryOpKind op, const void *src_a,
                               const void *src_b, void *dst, size_t n,
                               size_t element_size, cudaStream_t stream);

// Broadcast parameters copied to device memory once per launch.
struct BroadcastParams {
    int64_t a_strides[MAX_DIMS]; // 0 where a is broadcast
    int64_t b_strides[MAX_DIMS]; // 0 where b is broadcast
    int64_t out_shape[MAX_DIMS]; // output shape
    int ndim;
};

// Launch a binary element-wise kernel with NumPy-style broadcasting.
// Strides of 0 indicate broadcast dimensions.
void launch_binary_broadcast(BinaryOpKind op, const void *src_a,
                             const void *src_b, void *dst, size_t n,
                             const BroadcastParams &params, size_t element_size,
                             cudaStream_t stream);

// Identifies which unary operation to launch.
enum class UnaryOpKind : uint8_t {
    Negate,
    Abs,
    Sqrt,
    Exp,
    Log,
    Sin,
    Cos,
    Tan,
    Tanh,
    Sign,
    Floor,
    Ceil,
    Trunc,
    Round,
    Reciprocal,
    Square,
    Cbrt,
    Erf,
    // Testing ops — output is uint8_t
    IsNaN,
    IsInf,
    IsFinite,
    // Activations
    ReLU,
    LeakyReLU,
    Sigmoid,
    SiLU,
    GELU,
    // Logical
    LogicalNot,
};

// Launch a unary element-wise kernel.
// For most ops input and output share element_size.
// For IsNaN/IsInf/IsFinite the output is uint8_t.
void launch_unary_elementwise(UnaryOpKind op, const void *src, void *dst,
                              size_t n, size_t element_size,
                              cudaStream_t stream);

// ============================================================================
// Reduction kernels
// ============================================================================

enum class ReduceOpKind : uint8_t {
    Sum,
    Prod,
    Max,
    Min,
    Any,
    All,
};

// Full reduction — reduce all elements to a single scalar.
// `temp_bytes` is an in/out: pass 0 to query required temp size, then
// call again with allocated temp buffer.
void launch_full_reduce(ReduceOpKind op, const void *src, void *dst, size_t n,
                        size_t element_size, void *temp, size_t &temp_bytes,
                        cudaStream_t stream);

// Axis reduction — reduce along one axis decomposed as
// (outer, axis_len, inner).  Output has outer*inner elements.
void launch_axis_reduce(ReduceOpKind op, const void *src, void *dst,
                        size_t outer, size_t axis_len, size_t inner,
                        size_t element_size, cudaStream_t stream);

// Full ArgMax / ArgMin — returns the linear index of the extreme
// element (int64_t output).
void launch_full_argreduce(bool is_max, const void *src, void *dst, size_t n,
                           size_t element_size, void *temp, size_t &temp_bytes,
                           cudaStream_t stream);

// Axis ArgMax / ArgMin — returns indices along one axis.
void launch_axis_argreduce(bool is_max, const void *src, void *dst,
                           size_t outer, size_t axis_len, size_t inner,
                           size_t element_size, cudaStream_t stream);

// ============================================================================
// Where / MaskedFill / MaskedSelect kernels
// ============================================================================

// Where: out[i] = cond[i] ? a[i] : b[i]
// cond is uint8_t, a/b/out share element_size.
void launch_where(const void *cond, const void *a, const void *b, void *dst,
                  size_t n, size_t element_size, cudaStream_t stream);

// MaskedFill: out[i] = mask[i] ? fill_value_ptr[0] : src[i]
// mask is uint8_t, src/out/fill_value share element_size.
void launch_masked_fill(const void *src, const void *mask,
                        const void *fill_value, void *dst, size_t n,
                        size_t element_size, cudaStream_t stream);

// MaskedSelect: compact src elements where mask is true into dst.
// Uses CUB DeviceSelect::Flagged.  Returns selected count via
// d_num_selected (device pointer to size_t).
// Two-pass API: call with temp==nullptr to query temp_bytes, then
// call again with allocated temp.
void launch_masked_select(const void *src, const void *mask, void *dst,
                          size_t n, size_t element_size, void *d_num_selected,
                          void *temp, size_t &temp_bytes, cudaStream_t stream);

// ============================================================================
// Gather / Scatter / IndexSelect kernels
// ============================================================================

// Gather along `dim`.  For each output element at flat index i, decompose
// into N-d coords using `out_shape`, replace coord[dim] with
// indices[i], then read from `src` using `src_strides`.
// out_shape/src_strides are element-count strides of length ndim.
void launch_gather(const void *src, const int64_t *indices, void *dst,
                   size_t numel, int ndim, int dim, const int64_t *out_shape,
                   const int64_t *src_strides, int64_t dim_size,
                   size_t element_size, cudaStream_t stream);

// Scatter along `dim`.  dst starts as a copy of `input`.  For each
// position i in indices (total `numel` elements), decompose into coords
// using `idx_shape`, replace coord[dim] with indices[i], then write
// src_vals[i] into dst at that position.  Uses atomicExch to handle
// races (last-write-wins, matching PyTorch semantics).
void launch_scatter(const void *src_vals, const int64_t *indices, void *dst,
                    size_t numel, int ndim, int dim, const int64_t *idx_shape,
                    const int64_t *dst_strides, int64_t dim_size,
                    size_t element_size, cudaStream_t stream);

// IndexSelect along `dim`.  Simpler than Gather: output has the same
// shape as input except dim-size is replaced by num_indices.
// indices is a 1-D int64 tensor of length num_indices.
void launch_index_select(const void *src, const int64_t *indices, void *dst,
                         size_t numel, int ndim, int dim,
                         const int64_t *out_shape, const int64_t *src_strides,
                         int64_t dim_size, size_t element_size,
                         cudaStream_t stream);

// ============================================================================
// Cast kernel
// ============================================================================

// Type identifiers for the cast kernel — maps to concrete C++ types in the
// kernel implementation.  The CudaCastOperation translates axiom::DType to
// these before calling launch_cast.
enum class CastDType : uint8_t {
    Float16,
    Float32,
    Float64,
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    Bool, // uint8_t with != 0 → 1 semantics on output
};

// Element-wise type cast.  Handles all pairs of the above types.
// When dst_dtype is Bool, any non-zero source value maps to 1.
void launch_cast(CastDType src_dtype, CastDType dst_dtype, const void *src,
                 void *dst, size_t n, cudaStream_t stream);

// ============================================================================
// Softmax / LogSoftmax kernels
// ============================================================================

// Numerically stable softmax along one axis, decomposed as
// (outer, axis_len, inner).  Input and output have the same shape.
// is_log: if true, compute log_softmax instead.
void launch_softmax(const void *src, void *dst, size_t outer, size_t axis_len,
                    size_t inner, bool is_log, size_t element_size,
                    cudaStream_t stream);
#endif

} // namespace cuda
} // namespace backends
} // namespace axiom
