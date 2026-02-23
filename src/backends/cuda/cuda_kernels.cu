#include "cuda_kernels.hpp"

#ifdef AXIOM_CUDA_SUPPORT
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace axiom {
namespace backends {
namespace cuda {

static constexpr int BLOCK_SIZE = 256;

// ============================================================================
// Gather Strided Kernel
// ============================================================================

template <typename T>
__global__ void gather_strided(const T *src, T *dst,
                               GatherStridedParams p) {
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= p.numel) return;

    unsigned int coords[MAX_DIMS];
    unsigned int temp = gid;

    #pragma unroll
    for (int i = static_cast<int>(p.ndim) - 1; i >= 0; --i) {
        coords[i] = temp % p.shape[i];
        temp /= p.shape[i];
    }

    unsigned int src_idx = 0;
    #pragma unroll
    for (unsigned int i = 0; i < p.ndim; ++i) {
        unsigned int coord = coords[i];
        if (p.flip_mask & (1u << i)) {
            coord = p.shape[i] - 1 - coord;
        }
        src_idx += coord * p.src_strides[i];
    }

    dst[gid] = src[src_idx];
}

// Explicit template instantiations — gather_strided
template __global__ void gather_strided<float>(const float *, float *,
                                               GatherStridedParams);
template __global__ void gather_strided<double>(const double *, double *,
                                                GatherStridedParams);
template __global__ void gather_strided<__half>(const __half *, __half *,
                                                GatherStridedParams);
template __global__ void gather_strided<int32_t>(const int32_t *, int32_t *,
                                                 GatherStridedParams);
template __global__ void gather_strided<int64_t>(const int64_t *, int64_t *,
                                                 GatherStridedParams);
template __global__ void gather_strided<int16_t>(const int16_t *, int16_t *,
                                                 GatherStridedParams);
template __global__ void gather_strided<int8_t>(const int8_t *, int8_t *,
                                                GatherStridedParams);
template __global__ void gather_strided<uint8_t>(const uint8_t *, uint8_t *,
                                                 GatherStridedParams);
template __global__ void gather_strided<uint16_t>(const uint16_t *,
                                                  uint16_t *,
                                                  GatherStridedParams);
template __global__ void gather_strided<uint32_t>(const uint32_t *,
                                                  uint32_t *,
                                                  GatherStridedParams);

void launch_gather_strided(const void *src, void *dst,
                           const GatherStridedParams &params,
                           size_t element_size, cudaStream_t stream) {
    unsigned int grid = (params.numel + BLOCK_SIZE - 1) / BLOCK_SIZE;

    switch (element_size) {
    case 1:
        gather_strided<uint8_t><<<grid, BLOCK_SIZE, 0, stream>>>(
            static_cast<const uint8_t *>(src),
            static_cast<uint8_t *>(dst), params);
        break;
    case 2:
        gather_strided<uint16_t><<<grid, BLOCK_SIZE, 0, stream>>>(
            static_cast<const uint16_t *>(src),
            static_cast<uint16_t *>(dst), params);
        break;
    case 4:
        gather_strided<uint32_t><<<grid, BLOCK_SIZE, 0, stream>>>(
            static_cast<const uint32_t *>(src),
            static_cast<uint32_t *>(dst), params);
        break;
    case 8:
        gather_strided<int64_t><<<grid, BLOCK_SIZE, 0, stream>>>(
            static_cast<const int64_t *>(src),
            static_cast<int64_t *>(dst), params);
        break;
    default:
        throw std::runtime_error(
            "gather_strided: unsupported element size " +
            std::to_string(element_size));
    }
}

// ============================================================================
// Binary Element-wise Functors
// ============================================================================

struct AddOp {
    template <typename T>
    __device__ T operator()(T a, T b) const { return a + b; }
};

struct SubOp {
    template <typename T>
    __device__ T operator()(T a, T b) const { return a - b; }
};

struct MulOp {
    template <typename T>
    __device__ T operator()(T a, T b) const { return a * b; }
};

struct DivOp {
    template <typename T>
    __device__ T operator()(T a, T b) const { return a / b; }
};

struct PowOp {
    template <typename T>
    __device__ T operator()(T a, T b) const { return pow(static_cast<double>(a), static_cast<double>(b)); }
    __device__ float operator()(float a, float b) const { return powf(a, b); }
    __device__ double operator()(double a, double b) const { return pow(a, b); }
};

struct ModOp {
    template <typename T>
    __device__ T operator()(T a, T b) const { return a % b; }
    __device__ float operator()(float a, float b) const { return fmodf(a, b); }
    __device__ double operator()(double a, double b) const { return fmod(a, b); }
};

struct MaxOp {
    template <typename T>
    __device__ T operator()(T a, T b) const { return a > b ? a : b; }
    __device__ float operator()(float a, float b) const { return fmaxf(a, b); }
    __device__ double operator()(double a, double b) const { return fmax(a, b); }
};

struct MinOp {
    template <typename T>
    __device__ T operator()(T a, T b) const { return a < b ? a : b; }
    __device__ float operator()(float a, float b) const { return fminf(a, b); }
    __device__ double operator()(double a, double b) const { return fmin(a, b); }
};

struct Atan2Op {
    __device__ float operator()(float a, float b) const { return atan2f(a, b); }
    __device__ double operator()(double a, double b) const { return atan2(a, b); }
    template <typename T>
    __device__ T operator()(T a, T b) const {
        return static_cast<T>(atan2(static_cast<double>(a),
                                    static_cast<double>(b)));
    }
};

struct HypotOp {
    __device__ float operator()(float a, float b) const { return hypotf(a, b); }
    __device__ double operator()(double a, double b) const { return hypot(a, b); }
    template <typename T>
    __device__ T operator()(T a, T b) const {
        return static_cast<T>(hypot(static_cast<double>(a),
                                    static_cast<double>(b)));
    }
};

// ============================================================================
// Comparison Functors — output is uint8_t
// ============================================================================

struct EqualOp {
    template <typename T>
    __device__ uint8_t operator()(T a, T b) const { return a == b ? 1 : 0; }
};

struct NotEqualOp {
    template <typename T>
    __device__ uint8_t operator()(T a, T b) const { return a != b ? 1 : 0; }
};

struct LessOp {
    template <typename T>
    __device__ uint8_t operator()(T a, T b) const { return a < b ? 1 : 0; }
};

struct LessEqualOp {
    template <typename T>
    __device__ uint8_t operator()(T a, T b) const { return a <= b ? 1 : 0; }
};

struct GreaterOp {
    template <typename T>
    __device__ uint8_t operator()(T a, T b) const { return a > b ? 1 : 0; }
};

struct GreaterEqualOp {
    template <typename T>
    __device__ uint8_t operator()(T a, T b) const { return a >= b ? 1 : 0; }
};

// ============================================================================
// Logical Functors — inputs cast to bool, output is uint8_t
// ============================================================================

struct LogicalAndOp {
    template <typename T>
    __device__ uint8_t operator()(T a, T b) const {
        return (a != T(0)) && (b != T(0)) ? 1 : 0;
    }
};

struct LogicalOrOp {
    template <typename T>
    __device__ uint8_t operator()(T a, T b) const {
        return (a != T(0)) || (b != T(0)) ? 1 : 0;
    }
};

struct LogicalXorOp {
    template <typename T>
    __device__ uint8_t operator()(T a, T b) const {
        return ((a != T(0)) != (b != T(0))) ? 1 : 0;
    }
};

// ============================================================================
// Binary Element-wise Kernels
// ============================================================================

// Arithmetic: same-type input/output
template <typename T, typename Op>
__global__ void binary_elementwise(const T *a, const T *b, T *out,
                                   size_t n, Op op) {
    size_t gid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    out[gid] = op(a[gid], b[gid]);
}

// Comparison / logical: input T, output uint8_t
template <typename T, typename Op>
__global__ void binary_elementwise_cmp(const T *a, const T *b,
                                       uint8_t *out, size_t n, Op op) {
    size_t gid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    out[gid] = op(a[gid], b[gid]);
}

// ============================================================================
// Typed dispatch helpers
// ============================================================================

// Arithmetic dispatch — instantiates binary_elementwise<T,Op> for a given Op
// across float / double / int32 / int64.
template <typename Op>
static void launch_arith(Op op, const void *a, const void *b, void *dst,
                         size_t n, size_t elem, cudaStream_t s) {
    unsigned int grid = static_cast<unsigned int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    switch (elem) {
    case 4:
        binary_elementwise<float, Op><<<grid, BLOCK_SIZE, 0, s>>>(
            static_cast<const float *>(a), static_cast<const float *>(b),
            static_cast<float *>(dst), n, op);
        break;
    case 8:
        binary_elementwise<double, Op><<<grid, BLOCK_SIZE, 0, s>>>(
            static_cast<const double *>(a), static_cast<const double *>(b),
            static_cast<double *>(dst), n, op);
        break;
    case 2:
        binary_elementwise<int16_t, Op><<<grid, BLOCK_SIZE, 0, s>>>(
            static_cast<const int16_t *>(a),
            static_cast<const int16_t *>(b),
            static_cast<int16_t *>(dst), n, op);
        break;
    case 1:
        binary_elementwise<int8_t, Op><<<grid, BLOCK_SIZE, 0, s>>>(
            static_cast<const int8_t *>(a),
            static_cast<const int8_t *>(b),
            static_cast<int8_t *>(dst), n, op);
        break;
    default:
        throw std::runtime_error(
            "binary_elementwise: unsupported element size " +
            std::to_string(elem));
    }
}

// Comparison / logical dispatch — output is always uint8_t.
template <typename Op>
static void launch_cmp(Op op, const void *a, const void *b, void *dst,
                       size_t n, size_t elem, cudaStream_t s) {
    unsigned int grid = static_cast<unsigned int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    switch (elem) {
    case 4:
        binary_elementwise_cmp<float, Op><<<grid, BLOCK_SIZE, 0, s>>>(
            static_cast<const float *>(a), static_cast<const float *>(b),
            static_cast<uint8_t *>(dst), n, op);
        break;
    case 8:
        binary_elementwise_cmp<double, Op><<<grid, BLOCK_SIZE, 0, s>>>(
            static_cast<const double *>(a), static_cast<const double *>(b),
            static_cast<uint8_t *>(dst), n, op);
        break;
    case 2:
        binary_elementwise_cmp<int16_t, Op><<<grid, BLOCK_SIZE, 0, s>>>(
            static_cast<const int16_t *>(a),
            static_cast<const int16_t *>(b),
            static_cast<uint8_t *>(dst), n, op);
        break;
    case 1:
        binary_elementwise_cmp<int8_t, Op><<<grid, BLOCK_SIZE, 0, s>>>(
            static_cast<const int8_t *>(a),
            static_cast<const int8_t *>(b),
            static_cast<uint8_t *>(dst), n, op);
        break;
    default:
        throw std::runtime_error(
            "binary_elementwise_cmp: unsupported element size " +
            std::to_string(elem));
    }
}

// ============================================================================
// Public launcher
// ============================================================================

void launch_binary_elementwise(BinaryOpKind op, const void *src_a,
                               const void *src_b, void *dst, size_t n,
                               size_t element_size, cudaStream_t stream) {
    switch (op) {
    // Arithmetic
    case BinaryOpKind::Add:
        launch_arith(AddOp{}, src_a, src_b, dst, n, element_size, stream);
        break;
    case BinaryOpKind::Sub:
        launch_arith(SubOp{}, src_a, src_b, dst, n, element_size, stream);
        break;
    case BinaryOpKind::Mul:
        launch_arith(MulOp{}, src_a, src_b, dst, n, element_size, stream);
        break;
    case BinaryOpKind::Div:
        launch_arith(DivOp{}, src_a, src_b, dst, n, element_size, stream);
        break;
    case BinaryOpKind::Pow:
        launch_arith(PowOp{}, src_a, src_b, dst, n, element_size, stream);
        break;
    case BinaryOpKind::Mod:
        launch_arith(ModOp{}, src_a, src_b, dst, n, element_size, stream);
        break;
    case BinaryOpKind::Max:
        launch_arith(MaxOp{}, src_a, src_b, dst, n, element_size, stream);
        break;
    case BinaryOpKind::Min:
        launch_arith(MinOp{}, src_a, src_b, dst, n, element_size, stream);
        break;
    case BinaryOpKind::Atan2:
        launch_arith(Atan2Op{}, src_a, src_b, dst, n, element_size, stream);
        break;
    case BinaryOpKind::Hypot:
        launch_arith(HypotOp{}, src_a, src_b, dst, n, element_size, stream);
        break;
    // Comparison
    case BinaryOpKind::Equal:
        launch_cmp(EqualOp{}, src_a, src_b, dst, n, element_size, stream);
        break;
    case BinaryOpKind::NotEqual:
        launch_cmp(NotEqualOp{}, src_a, src_b, dst, n, element_size, stream);
        break;
    case BinaryOpKind::Less:
        launch_cmp(LessOp{}, src_a, src_b, dst, n, element_size, stream);
        break;
    case BinaryOpKind::LessEqual:
        launch_cmp(LessEqualOp{}, src_a, src_b, dst, n, element_size, stream);
        break;
    case BinaryOpKind::Greater:
        launch_cmp(GreaterOp{}, src_a, src_b, dst, n, element_size, stream);
        break;
    case BinaryOpKind::GreaterEqual:
        launch_cmp(GreaterEqualOp{}, src_a, src_b, dst, n, element_size, stream);
        break;
    // Logical
    case BinaryOpKind::LogicalAnd:
        launch_cmp(LogicalAndOp{}, src_a, src_b, dst, n, element_size, stream);
        break;
    case BinaryOpKind::LogicalOr:
        launch_cmp(LogicalOrOp{}, src_a, src_b, dst, n, element_size, stream);
        break;
    case BinaryOpKind::LogicalXor:
        launch_cmp(LogicalXorOp{}, src_a, src_b, dst, n, element_size, stream);
        break;
    }
}

// ============================================================================
// Binary Broadcast Kernels
// ============================================================================
// Stride-based indexing: a stride of 0 means the dimension is broadcast.
// The output linear index is decomposed into N-d coords which are then
// dotted with per-operand strides to find the source element.
// ============================================================================

template <typename T, typename Op>
__global__ void binary_broadcast(const T *a, const T *b, T *out,
                                 size_t n, BroadcastParams p, Op op) {
    size_t gid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (gid >= n) return;

    // Decompose linear index into N-d coords (row-major)
    size_t tmp = gid;
    int64_t a_idx = 0;
    int64_t b_idx = 0;
    #pragma unroll
    for (int i = p.ndim - 1; i >= 0; --i) {
        int64_t coord = static_cast<int64_t>(tmp % static_cast<size_t>(p.out_shape[i]));
        tmp /= static_cast<size_t>(p.out_shape[i]);
        a_idx += coord * p.a_strides[i];
        b_idx += coord * p.b_strides[i];
    }

    out[gid] = op(a[a_idx], b[b_idx]);
}

template <typename T, typename Op>
__global__ void binary_broadcast_cmp(const T *a, const T *b, uint8_t *out,
                                     size_t n, BroadcastParams p, Op op) {
    size_t gid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (gid >= n) return;

    size_t tmp = gid;
    int64_t a_idx = 0;
    int64_t b_idx = 0;
    #pragma unroll
    for (int i = p.ndim - 1; i >= 0; --i) {
        int64_t coord = static_cast<int64_t>(tmp % static_cast<size_t>(p.out_shape[i]));
        tmp /= static_cast<size_t>(p.out_shape[i]);
        a_idx += coord * p.a_strides[i];
        b_idx += coord * p.b_strides[i];
    }

    out[gid] = op(a[a_idx], b[b_idx]);
}

// ============================================================================
// Broadcast typed dispatch helpers
// ============================================================================

template <typename Op>
static void launch_bcast_arith(Op op, const void *a, const void *b,
                               void *dst, size_t n,
                               const BroadcastParams &p, size_t elem,
                               cudaStream_t s) {
    unsigned int grid =
        static_cast<unsigned int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    switch (elem) {
    case 4:
        binary_broadcast<float, Op><<<grid, BLOCK_SIZE, 0, s>>>(
            static_cast<const float *>(a), static_cast<const float *>(b),
            static_cast<float *>(dst), n, p, op);
        break;
    case 8:
        binary_broadcast<double, Op><<<grid, BLOCK_SIZE, 0, s>>>(
            static_cast<const double *>(a),
            static_cast<const double *>(b),
            static_cast<double *>(dst), n, p, op);
        break;
    case 2:
        binary_broadcast<int16_t, Op><<<grid, BLOCK_SIZE, 0, s>>>(
            static_cast<const int16_t *>(a),
            static_cast<const int16_t *>(b),
            static_cast<int16_t *>(dst), n, p, op);
        break;
    case 1:
        binary_broadcast<int8_t, Op><<<grid, BLOCK_SIZE, 0, s>>>(
            static_cast<const int8_t *>(a),
            static_cast<const int8_t *>(b),
            static_cast<int8_t *>(dst), n, p, op);
        break;
    default:
        throw std::runtime_error(
            "binary_broadcast: unsupported element size " +
            std::to_string(elem));
    }
}

template <typename Op>
static void launch_bcast_cmp(Op op, const void *a, const void *b,
                             void *dst, size_t n,
                             const BroadcastParams &p, size_t elem,
                             cudaStream_t s) {
    unsigned int grid =
        static_cast<unsigned int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    switch (elem) {
    case 4:
        binary_broadcast_cmp<float, Op><<<grid, BLOCK_SIZE, 0, s>>>(
            static_cast<const float *>(a), static_cast<const float *>(b),
            static_cast<uint8_t *>(dst), n, p, op);
        break;
    case 8:
        binary_broadcast_cmp<double, Op><<<grid, BLOCK_SIZE, 0, s>>>(
            static_cast<const double *>(a),
            static_cast<const double *>(b),
            static_cast<uint8_t *>(dst), n, p, op);
        break;
    case 2:
        binary_broadcast_cmp<int16_t, Op><<<grid, BLOCK_SIZE, 0, s>>>(
            static_cast<const int16_t *>(a),
            static_cast<const int16_t *>(b),
            static_cast<uint8_t *>(dst), n, p, op);
        break;
    case 1:
        binary_broadcast_cmp<int8_t, Op><<<grid, BLOCK_SIZE, 0, s>>>(
            static_cast<const int8_t *>(a),
            static_cast<const int8_t *>(b),
            static_cast<uint8_t *>(dst), n, p, op);
        break;
    default:
        throw std::runtime_error(
            "binary_broadcast_cmp: unsupported element size " +
            std::to_string(elem));
    }
}

// ============================================================================
// Public broadcast launcher
// ============================================================================

void launch_binary_broadcast(BinaryOpKind op, const void *src_a,
                             const void *src_b, void *dst, size_t n,
                             const BroadcastParams &params,
                             size_t element_size, cudaStream_t stream) {
    switch (op) {
    // Arithmetic
    case BinaryOpKind::Add:
        launch_bcast_arith(AddOp{}, src_a, src_b, dst, n, params, element_size, stream); break;
    case BinaryOpKind::Sub:
        launch_bcast_arith(SubOp{}, src_a, src_b, dst, n, params, element_size, stream); break;
    case BinaryOpKind::Mul:
        launch_bcast_arith(MulOp{}, src_a, src_b, dst, n, params, element_size, stream); break;
    case BinaryOpKind::Div:
        launch_bcast_arith(DivOp{}, src_a, src_b, dst, n, params, element_size, stream); break;
    case BinaryOpKind::Pow:
        launch_bcast_arith(PowOp{}, src_a, src_b, dst, n, params, element_size, stream); break;
    case BinaryOpKind::Mod:
        launch_bcast_arith(ModOp{}, src_a, src_b, dst, n, params, element_size, stream); break;
    case BinaryOpKind::Max:
        launch_bcast_arith(MaxOp{}, src_a, src_b, dst, n, params, element_size, stream); break;
    case BinaryOpKind::Min:
        launch_bcast_arith(MinOp{}, src_a, src_b, dst, n, params, element_size, stream); break;
    case BinaryOpKind::Atan2:
        launch_bcast_arith(Atan2Op{}, src_a, src_b, dst, n, params, element_size, stream); break;
    case BinaryOpKind::Hypot:
        launch_bcast_arith(HypotOp{}, src_a, src_b, dst, n, params, element_size, stream); break;
    // Comparison
    case BinaryOpKind::Equal:
        launch_bcast_cmp(EqualOp{}, src_a, src_b, dst, n, params, element_size, stream); break;
    case BinaryOpKind::NotEqual:
        launch_bcast_cmp(NotEqualOp{}, src_a, src_b, dst, n, params, element_size, stream); break;
    case BinaryOpKind::Less:
        launch_bcast_cmp(LessOp{}, src_a, src_b, dst, n, params, element_size, stream); break;
    case BinaryOpKind::LessEqual:
        launch_bcast_cmp(LessEqualOp{}, src_a, src_b, dst, n, params, element_size, stream); break;
    case BinaryOpKind::Greater:
        launch_bcast_cmp(GreaterOp{}, src_a, src_b, dst, n, params, element_size, stream); break;
    case BinaryOpKind::GreaterEqual:
        launch_bcast_cmp(GreaterEqualOp{}, src_a, src_b, dst, n, params, element_size, stream); break;
    // Logical
    case BinaryOpKind::LogicalAnd:
        launch_bcast_cmp(LogicalAndOp{}, src_a, src_b, dst, n, params, element_size, stream); break;
    case BinaryOpKind::LogicalOr:
        launch_bcast_cmp(LogicalOrOp{}, src_a, src_b, dst, n, params, element_size, stream); break;
    case BinaryOpKind::LogicalXor:
        launch_bcast_cmp(LogicalXorOp{}, src_a, src_b, dst, n, params, element_size, stream); break;
    }
}

} // namespace cuda
} // namespace backends
} // namespace axiom

#endif // AXIOM_CUDA_SUPPORT
