#include "cuda_kernels.hpp"

#ifdef AXIOM_CUDA_SUPPORT
#include <cub/cub.cuh>
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

// ============================================================================
// Unary Element-wise Functors
// ============================================================================

struct NegateOp {
    template <typename T>
    __device__ T operator()(T x) const { return -x; }
};

struct AbsOp {
    __device__ float operator()(float x) const { return fabsf(x); }
    __device__ double operator()(double x) const { return fabs(x); }
    template <typename T>
    __device__ T operator()(T x) const { return x < T(0) ? -x : x; }
};

struct SqrtOp {
    __device__ float operator()(float x) const { return sqrtf(x); }
    __device__ double operator()(double x) const { return sqrt(x); }
    template <typename T>
    __device__ T operator()(T x) const { return static_cast<T>(sqrtf(static_cast<float>(x))); }
};

struct ExpOp {
    __device__ float operator()(float x) const { return __expf(x); }
    __device__ double operator()(double x) const { return exp(x); }
    template <typename T>
    __device__ T operator()(T x) const { return static_cast<T>(__expf(static_cast<float>(x))); }
};

struct LogOp {
    __device__ float operator()(float x) const { return __logf(x); }
    __device__ double operator()(double x) const { return log(x); }
    template <typename T>
    __device__ T operator()(T x) const { return static_cast<T>(__logf(static_cast<float>(x))); }
};

struct SinOp {
    __device__ float operator()(float x) const { return __sinf(x); }
    __device__ double operator()(double x) const { return sin(x); }
    template <typename T>
    __device__ T operator()(T x) const { return static_cast<T>(__sinf(static_cast<float>(x))); }
};

struct CosOp {
    __device__ float operator()(float x) const { return __cosf(x); }
    __device__ double operator()(double x) const { return cos(x); }
    template <typename T>
    __device__ T operator()(T x) const { return static_cast<T>(__cosf(static_cast<float>(x))); }
};

struct TanOp {
    __device__ float operator()(float x) const { return tanf(x); }
    __device__ double operator()(double x) const { return tan(x); }
    template <typename T>
    __device__ T operator()(T x) const { return static_cast<T>(tanf(static_cast<float>(x))); }
};

struct TanhOp {
    __device__ float operator()(float x) const { return tanhf(x); }
    __device__ double operator()(double x) const { return tanh(x); }
    template <typename T>
    __device__ T operator()(T x) const { return static_cast<T>(tanhf(static_cast<float>(x))); }
};

struct SignOp {
    template <typename T>
    __device__ T operator()(T x) const {
        return (x > T(0)) ? T(1) : ((x < T(0)) ? T(-1) : T(0));
    }
};

struct FloorOp {
    __device__ float operator()(float x) const { return floorf(x); }
    __device__ double operator()(double x) const { return floor(x); }
    template <typename T>
    __device__ T operator()(T x) const { return x; } // no-op for integers
};

struct CeilOp {
    __device__ float operator()(float x) const { return ceilf(x); }
    __device__ double operator()(double x) const { return ceil(x); }
    template <typename T>
    __device__ T operator()(T x) const { return x; }
};

struct TruncOp {
    __device__ float operator()(float x) const { return truncf(x); }
    __device__ double operator()(double x) const { return trunc(x); }
    template <typename T>
    __device__ T operator()(T x) const { return x; }
};

struct RoundOp {
    __device__ float operator()(float x) const { return rintf(x); }
    __device__ double operator()(double x) const { return rint(x); }
    template <typename T>
    __device__ T operator()(T x) const { return x; }
};

struct ReciprocalOp {
    __device__ float operator()(float x) const { return __fdividef(1.0f, x); }
    __device__ double operator()(double x) const { return 1.0 / x; }
    template <typename T>
    __device__ T operator()(T x) const { return T(1) / x; }
};

struct SquareOp {
    template <typename T>
    __device__ T operator()(T x) const { return x * x; }
};

struct CbrtOp {
    __device__ float operator()(float x) const { return cbrtf(x); }
    __device__ double operator()(double x) const { return cbrt(x); }
    template <typename T>
    __device__ T operator()(T x) const { return static_cast<T>(cbrtf(static_cast<float>(x))); }
};

struct ErfOp {
    __device__ float operator()(float x) const { return erff(x); }
    __device__ double operator()(double x) const { return erf(x); }
    template <typename T>
    __device__ T operator()(T x) const { return static_cast<T>(erff(static_cast<float>(x))); }
};

// Testing ops — output is uint8_t
struct IsNaNOp {
    __device__ uint8_t operator()(float x) const { return isnan(x) ? 1 : 0; }
    __device__ uint8_t operator()(double x) const { return isnan(x) ? 1 : 0; }
    template <typename T>
    __device__ uint8_t operator()(T /*x*/) const { return 0; } // integers are never NaN
};

struct IsInfOp {
    __device__ uint8_t operator()(float x) const { return isinf(x) ? 1 : 0; }
    __device__ uint8_t operator()(double x) const { return isinf(x) ? 1 : 0; }
    template <typename T>
    __device__ uint8_t operator()(T /*x*/) const { return 0; }
};

struct IsFiniteOp {
    __device__ uint8_t operator()(float x) const { return isfinite(x) ? 1 : 0; }
    __device__ uint8_t operator()(double x) const { return isfinite(x) ? 1 : 0; }
    template <typename T>
    __device__ uint8_t operator()(T /*x*/) const { return 1; } // integers are always finite
};

// Activation functors
struct ReLUOp {
    template <typename T>
    __device__ T operator()(T x) const { return x > T(0) ? x : T(0); }
};

struct LeakyReLUOp {
    template <typename T>
    __device__ T operator()(T x) const {
        // Default negative slope = 0.01
        return x > T(0) ? x : static_cast<T>(static_cast<float>(x) * 0.01f);
    }
    __device__ float operator()(float x) const { return x > 0.0f ? x : x * 0.01f; }
    __device__ double operator()(double x) const { return x > 0.0 ? x : x * 0.01; }
};

struct SigmoidOp {
    __device__ float operator()(float x) const { return 1.0f / (1.0f + __expf(-x)); }
    __device__ double operator()(double x) const { return 1.0 / (1.0 + exp(-x)); }
    template <typename T>
    __device__ T operator()(T x) const {
        return static_cast<T>(1.0f / (1.0f + __expf(-static_cast<float>(x))));
    }
};

struct SiLUOp {
    __device__ float operator()(float x) const { return x / (1.0f + __expf(-x)); }
    __device__ double operator()(double x) const { return x / (1.0 + exp(-x)); }
    template <typename T>
    __device__ T operator()(T x) const {
        float fx = static_cast<float>(x);
        return static_cast<T>(fx / (1.0f + __expf(-fx)));
    }
};

struct GELUOp {
    // Approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    __device__ float operator()(float x) const {
        constexpr float kAlpha = 0.7978845608f; // sqrt(2/pi)
        constexpr float kBeta = 0.044715f;
        float inner = kAlpha * (x + kBeta * x * x * x);
        return 0.5f * x * (1.0f + tanhf(inner));
    }
    __device__ double operator()(double x) const {
        constexpr double kAlpha = 0.7978845608028654;
        constexpr double kBeta = 0.044715;
        double inner = kAlpha * (x + kBeta * x * x * x);
        return 0.5 * x * (1.0 + tanh(inner));
    }
    template <typename T>
    __device__ T operator()(T x) const {
        return static_cast<T>(this->operator()(static_cast<float>(x)));
    }
};

// ============================================================================
// Unary Element-wise Kernels
// ============================================================================

// Same-type input/output
template <typename T, typename Op>
__global__ void unary_elementwise(const T *in, T *out, size_t n, Op op) {
    size_t gid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    out[gid] = op(in[gid]);
}

// Input T, output uint8_t (for IsNaN, IsInf, IsFinite)
template <typename T, typename Op>
__global__ void unary_elementwise_test(const T *in, uint8_t *out,
                                       size_t n, Op op) {
    size_t gid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    out[gid] = op(in[gid]);
}

// ============================================================================
// Unary typed dispatch helpers
// ============================================================================

template <typename Op>
static void launch_unary(Op op, const void *src, void *dst, size_t n,
                         size_t elem, cudaStream_t s) {
    unsigned int grid =
        static_cast<unsigned int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    switch (elem) {
    case 4:
        unary_elementwise<float, Op><<<grid, BLOCK_SIZE, 0, s>>>(
            static_cast<const float *>(src), static_cast<float *>(dst),
            n, op);
        break;
    case 8:
        unary_elementwise<double, Op><<<grid, BLOCK_SIZE, 0, s>>>(
            static_cast<const double *>(src),
            static_cast<double *>(dst), n, op);
        break;
    case 2:
        unary_elementwise<int16_t, Op><<<grid, BLOCK_SIZE, 0, s>>>(
            static_cast<const int16_t *>(src),
            static_cast<int16_t *>(dst), n, op);
        break;
    case 1:
        unary_elementwise<int8_t, Op><<<grid, BLOCK_SIZE, 0, s>>>(
            static_cast<const int8_t *>(src),
            static_cast<int8_t *>(dst), n, op);
        break;
    default:
        throw std::runtime_error(
            "unary_elementwise: unsupported element size " +
            std::to_string(elem));
    }
}

template <typename Op>
static void launch_unary_test(Op op, const void *src, void *dst, size_t n,
                              size_t elem, cudaStream_t s) {
    unsigned int grid =
        static_cast<unsigned int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    switch (elem) {
    case 4:
        unary_elementwise_test<float, Op><<<grid, BLOCK_SIZE, 0, s>>>(
            static_cast<const float *>(src), static_cast<uint8_t *>(dst),
            n, op);
        break;
    case 8:
        unary_elementwise_test<double, Op><<<grid, BLOCK_SIZE, 0, s>>>(
            static_cast<const double *>(src),
            static_cast<uint8_t *>(dst), n, op);
        break;
    case 2:
        unary_elementwise_test<int16_t, Op><<<grid, BLOCK_SIZE, 0, s>>>(
            static_cast<const int16_t *>(src),
            static_cast<uint8_t *>(dst), n, op);
        break;
    case 1:
        unary_elementwise_test<int8_t, Op><<<grid, BLOCK_SIZE, 0, s>>>(
            static_cast<const int8_t *>(src),
            static_cast<uint8_t *>(dst), n, op);
        break;
    default:
        throw std::runtime_error(
            "unary_elementwise_test: unsupported element size " +
            std::to_string(elem));
    }
}

// ============================================================================
// Public unary launcher
// ============================================================================

void launch_unary_elementwise(UnaryOpKind op, const void *src, void *dst,
                              size_t n, size_t element_size,
                              cudaStream_t stream) {
    switch (op) {
    case UnaryOpKind::Negate:
        launch_unary(NegateOp{}, src, dst, n, element_size, stream); break;
    case UnaryOpKind::Abs:
        launch_unary(AbsOp{}, src, dst, n, element_size, stream); break;
    case UnaryOpKind::Sqrt:
        launch_unary(SqrtOp{}, src, dst, n, element_size, stream); break;
    case UnaryOpKind::Exp:
        launch_unary(ExpOp{}, src, dst, n, element_size, stream); break;
    case UnaryOpKind::Log:
        launch_unary(LogOp{}, src, dst, n, element_size, stream); break;
    case UnaryOpKind::Sin:
        launch_unary(SinOp{}, src, dst, n, element_size, stream); break;
    case UnaryOpKind::Cos:
        launch_unary(CosOp{}, src, dst, n, element_size, stream); break;
    case UnaryOpKind::Tan:
        launch_unary(TanOp{}, src, dst, n, element_size, stream); break;
    case UnaryOpKind::Tanh:
        launch_unary(TanhOp{}, src, dst, n, element_size, stream); break;
    case UnaryOpKind::Sign:
        launch_unary(SignOp{}, src, dst, n, element_size, stream); break;
    case UnaryOpKind::Floor:
        launch_unary(FloorOp{}, src, dst, n, element_size, stream); break;
    case UnaryOpKind::Ceil:
        launch_unary(CeilOp{}, src, dst, n, element_size, stream); break;
    case UnaryOpKind::Trunc:
        launch_unary(TruncOp{}, src, dst, n, element_size, stream); break;
    case UnaryOpKind::Round:
        launch_unary(RoundOp{}, src, dst, n, element_size, stream); break;
    case UnaryOpKind::Reciprocal:
        launch_unary(ReciprocalOp{}, src, dst, n, element_size, stream); break;
    case UnaryOpKind::Square:
        launch_unary(SquareOp{}, src, dst, n, element_size, stream); break;
    case UnaryOpKind::Cbrt:
        launch_unary(CbrtOp{}, src, dst, n, element_size, stream); break;
    case UnaryOpKind::Erf:
        launch_unary(ErfOp{}, src, dst, n, element_size, stream); break;
    // Testing ops — output uint8_t
    case UnaryOpKind::IsNaN:
        launch_unary_test(IsNaNOp{}, src, dst, n, element_size, stream); break;
    case UnaryOpKind::IsInf:
        launch_unary_test(IsInfOp{}, src, dst, n, element_size, stream); break;
    case UnaryOpKind::IsFinite:
        launch_unary_test(IsFiniteOp{}, src, dst, n, element_size, stream); break;
    // Activations
    case UnaryOpKind::ReLU:
        launch_unary(ReLUOp{}, src, dst, n, element_size, stream); break;
    case UnaryOpKind::LeakyReLU:
        launch_unary(LeakyReLUOp{}, src, dst, n, element_size, stream); break;
    case UnaryOpKind::Sigmoid:
        launch_unary(SigmoidOp{}, src, dst, n, element_size, stream); break;
    case UnaryOpKind::SiLU:
        launch_unary(SiLUOp{}, src, dst, n, element_size, stream); break;
    case UnaryOpKind::GELU:
        launch_unary(GELUOp{}, src, dst, n, element_size, stream); break;
    }
}

// ============================================================================
// Where Kernel: out[i] = cond[i] ? a[i] : b[i]
// ============================================================================

template <typename T>
__global__ void where_kernel(const uint8_t *cond, const T *a, const T *b,
                             T *out, size_t n) {
    size_t gid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    out[gid] = cond[gid] ? a[gid] : b[gid];
}

void launch_where(const void *cond, const void *a, const void *b,
                  void *dst, size_t n, size_t element_size,
                  cudaStream_t stream) {
    unsigned int grid =
        static_cast<unsigned int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    const auto *cond_ptr = static_cast<const uint8_t *>(cond);

    switch (element_size) {
    case 4:
        where_kernel<float><<<grid, BLOCK_SIZE, 0, stream>>>(
            cond_ptr, static_cast<const float *>(a),
            static_cast<const float *>(b), static_cast<float *>(dst), n);
        break;
    case 8:
        where_kernel<double><<<grid, BLOCK_SIZE, 0, stream>>>(
            cond_ptr, static_cast<const double *>(a),
            static_cast<const double *>(b), static_cast<double *>(dst), n);
        break;
    case 2:
        where_kernel<int16_t><<<grid, BLOCK_SIZE, 0, stream>>>(
            cond_ptr, static_cast<const int16_t *>(a),
            static_cast<const int16_t *>(b),
            static_cast<int16_t *>(dst), n);
        break;
    case 1:
        where_kernel<int8_t><<<grid, BLOCK_SIZE, 0, stream>>>(
            cond_ptr, static_cast<const int8_t *>(a),
            static_cast<const int8_t *>(b),
            static_cast<int8_t *>(dst), n);
        break;
    default:
        throw std::runtime_error(
            "launch_where: unsupported element size " +
            std::to_string(element_size));
    }
}

// ============================================================================
// MaskedFill Kernel: out[i] = mask[i] ? fill_value : src[i]
// ============================================================================

template <typename T>
__global__ void masked_fill_kernel(const T *src, const uint8_t *mask,
                                   T fill_value, T *out, size_t n) {
    size_t gid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    out[gid] = mask[gid] ? fill_value : src[gid];
}

void launch_masked_fill(const void *src, const void *mask,
                        const void *fill_value, void *dst, size_t n,
                        size_t element_size, cudaStream_t stream) {
    unsigned int grid =
        static_cast<unsigned int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    const auto *mask_ptr = static_cast<const uint8_t *>(mask);

    switch (element_size) {
    case 4: {
        // Read the scalar fill value from device memory via a small
        // device-side kernel parameter.  fill_value points to device memory
        // holding one T element, so we copy it to host for the kernel param.
        float fv;
        cudaMemcpyAsync(&fv, fill_value, sizeof(float),
                         cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        masked_fill_kernel<float><<<grid, BLOCK_SIZE, 0, stream>>>(
            static_cast<const float *>(src), mask_ptr, fv,
            static_cast<float *>(dst), n);
        break;
    }
    case 8: {
        double fv;
        cudaMemcpyAsync(&fv, fill_value, sizeof(double),
                         cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        masked_fill_kernel<double><<<grid, BLOCK_SIZE, 0, stream>>>(
            static_cast<const double *>(src), mask_ptr, fv,
            static_cast<double *>(dst), n);
        break;
    }
    case 2: {
        int16_t fv;
        cudaMemcpyAsync(&fv, fill_value, sizeof(int16_t),
                         cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        masked_fill_kernel<int16_t><<<grid, BLOCK_SIZE, 0, stream>>>(
            static_cast<const int16_t *>(src), mask_ptr, fv,
            static_cast<int16_t *>(dst), n);
        break;
    }
    case 1: {
        int8_t fv;
        cudaMemcpyAsync(&fv, fill_value, sizeof(int8_t),
                         cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        masked_fill_kernel<int8_t><<<grid, BLOCK_SIZE, 0, stream>>>(
            static_cast<const int8_t *>(src), mask_ptr, fv,
            static_cast<int8_t *>(dst), n);
        break;
    }
    default:
        throw std::runtime_error(
            "launch_masked_fill: unsupported element size " +
            std::to_string(element_size));
    }
}

// ============================================================================
// MaskedSelect via CUB DeviceSelect::Flagged
// ============================================================================
// CUB expects a "flags" iterator of the same length as input where
// flag[i] != 0 means select.  Our mask is uint8_t which works directly.
// ============================================================================

template <typename T>
static void masked_select_typed(const T *src, const uint8_t *flags, T *dst,
                                size_t n, int *d_num_selected,
                                void *temp, size_t &temp_bytes,
                                cudaStream_t stream) {
    cub::DeviceSelect::Flagged(temp, temp_bytes, src, flags, dst,
                                d_num_selected, static_cast<int>(n),
                                stream);
}

void launch_masked_select(const void *src, const void *mask, void *dst,
                          size_t n, size_t element_size,
                          void *d_num_selected,
                          void *temp, size_t &temp_bytes,
                          cudaStream_t stream) {
    const auto *flags = static_cast<const uint8_t *>(mask);
    auto *d_count = static_cast<int *>(d_num_selected);

    switch (element_size) {
    case 4:
        masked_select_typed(static_cast<const float *>(src), flags,
                            static_cast<float *>(dst), n, d_count,
                            temp, temp_bytes, stream);
        break;
    case 8:
        masked_select_typed(static_cast<const double *>(src), flags,
                            static_cast<double *>(dst), n, d_count,
                            temp, temp_bytes, stream);
        break;
    case 2:
        masked_select_typed(static_cast<const int16_t *>(src), flags,
                            static_cast<int16_t *>(dst), n, d_count,
                            temp, temp_bytes, stream);
        break;
    case 1:
        masked_select_typed(static_cast<const int8_t *>(src), flags,
                            static_cast<int8_t *>(dst), n, d_count,
                            temp, temp_bytes, stream);
        break;
    default:
        throw std::runtime_error(
            "launch_masked_select: unsupported element size " +
            std::to_string(element_size));
    }
}

} // namespace cuda
} // namespace backends
} // namespace axiom

#endif // AXIOM_CUDA_SUPPORT
