#include "cuda_kernels.hpp"
#include "cuda_launch_cache.hpp"

#ifdef AXIOM_CUDA_SUPPORT
#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include <cfloat>
#include <climits>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace axiom {
namespace backends {
namespace cuda {

static constexpr int REDUCE_BLOCK = 256;

// ============================================================================
// CUB reduction identity values
// ============================================================================

template <typename T> struct SumIdentity { static __host__ __device__ T value() { return T(0); } };
template <typename T> struct ProdIdentity { static __host__ __device__ T value() { return T(1); } };
template <typename T> struct MaxIdentity;
template <> struct MaxIdentity<float>   { static __host__ __device__ float   value() { return -FLT_MAX; } };
template <> struct MaxIdentity<double>  { static __host__ __device__ double  value() { return -DBL_MAX; } };
template <> struct MaxIdentity<int32_t> { static __host__ __device__ int32_t value() { return INT_MIN; } };
template <> struct MaxIdentity<int64_t> { static __host__ __device__ int64_t value() { return LLONG_MIN; } };
template <> struct MaxIdentity<int16_t> { static __host__ __device__ int16_t value() { return SHRT_MIN; } };
template <> struct MaxIdentity<int8_t>  { static __host__ __device__ int8_t  value() { return SCHAR_MIN; } };

template <typename T> struct MinIdentity;
template <> struct MinIdentity<float>   { static __host__ __device__ float   value() { return FLT_MAX; } };
template <> struct MinIdentity<double>  { static __host__ __device__ double  value() { return DBL_MAX; } };
template <> struct MinIdentity<int32_t> { static __host__ __device__ int32_t value() { return INT_MAX; } };
template <> struct MinIdentity<int64_t> { static __host__ __device__ int64_t value() { return LLONG_MAX; } };
template <> struct MinIdentity<int16_t> { static __host__ __device__ int16_t value() { return SHRT_MAX; } };
template <> struct MinIdentity<int8_t>  { static __host__ __device__ int8_t  value() { return SCHAR_MAX; } };

// Lightweight reduction functors (avoid CUB-internal types that were removed
// in CUDA 13.x CCCL — cub::Sum, cub::Max, cub::Min no longer exist).
struct ReduceSumOp {
    template <typename T>
    __device__ __forceinline__ T operator()(const T &a, const T &b) const {
        return a + b;
    }
};

struct ReduceMaxOp {
    template <typename T>
    __device__ __forceinline__ T operator()(const T &a, const T &b) const {
        return a > b ? a : b;
    }
};

struct ReduceMinOp {
    template <typename T>
    __device__ __forceinline__ T operator()(const T &a, const T &b) const {
        return a < b ? a : b;
    }
};

struct ReduceProdOp {
    template <typename T>
    __device__ __forceinline__ T operator()(const T &a, const T &b) const {
        return a * b;
    }
};

struct ReduceAnyOp {
    template <typename T>
    __device__ __forceinline__ T operator()(const T &a, const T &b) const {
        return (a != T(0) || b != T(0)) ? T(1) : T(0);
    }
};

struct ReduceAllOp {
    template <typename T>
    __device__ __forceinline__ T operator()(const T &a, const T &b) const {
        return (a != T(0) && b != T(0)) ? T(1) : T(0);
    }
};

// ============================================================================
// Full reduction via CUB
// ============================================================================

template <typename T>
static void full_reduce_typed(ReduceOpKind op, const T *src, T *dst,
                              size_t n, void *temp, size_t &temp_bytes,
                              cudaStream_t stream) {
    switch (op) {
    case ReduceOpKind::Sum:
        cub::DeviceReduce::Reduce(temp, temp_bytes, src, dst,
                                  static_cast<int>(n), ReduceSumOp{},
                                  SumIdentity<T>::value(), stream);
        break;
    case ReduceOpKind::Max:
        cub::DeviceReduce::Reduce(temp, temp_bytes, src, dst,
                                  static_cast<int>(n), ReduceMaxOp{},
                                  MaxIdentity<T>::value(), stream);
        break;
    case ReduceOpKind::Min:
        cub::DeviceReduce::Reduce(temp, temp_bytes, src, dst,
                                  static_cast<int>(n), ReduceMinOp{},
                                  MinIdentity<T>::value(), stream);
        break;
    case ReduceOpKind::Prod:
        cub::DeviceReduce::Reduce(temp, temp_bytes, src, dst,
                                  static_cast<int>(n), ReduceProdOp{},
                                  ProdIdentity<T>::value(), stream);
        break;
    case ReduceOpKind::Any:
        cub::DeviceReduce::Reduce(temp, temp_bytes, src, dst,
                                  static_cast<int>(n), ReduceAnyOp{},
                                  T(0), stream);
        break;
    case ReduceOpKind::All:
        cub::DeviceReduce::Reduce(temp, temp_bytes, src, dst,
                                  static_cast<int>(n), ReduceAllOp{},
                                  T(1), stream);
        break;
    }
}

void launch_full_reduce(ReduceOpKind op, const void *src, void *dst,
                        size_t n, size_t element_size,
                        void *temp, size_t &temp_bytes,
                        cudaStream_t stream) {
    switch (element_size) {
    case 4:
        full_reduce_typed(op, static_cast<const float *>(src),
                          static_cast<float *>(dst), n, temp, temp_bytes,
                          stream);
        break;
    case 8:
        full_reduce_typed(op, static_cast<const double *>(src),
                          static_cast<double *>(dst), n, temp, temp_bytes,
                          stream);
        break;
    case 2:
        full_reduce_typed(op, static_cast<const int16_t *>(src),
                          static_cast<int16_t *>(dst), n, temp, temp_bytes,
                          stream);
        break;
    case 1:
        full_reduce_typed(op, static_cast<const int8_t *>(src),
                          static_cast<int8_t *>(dst), n, temp, temp_bytes,
                          stream);
        break;
    default:
        throw std::runtime_error("launch_full_reduce: unsupported element size " +
                                 std::to_string(element_size));
    }
}

// ============================================================================
// Axis reduction kernel (shared-memory based)
// ============================================================================
// Layout: input is (outer, axis_len, inner) in row-major order.
// Each thread block handles one (outer_idx, inner_idx) pair and
// reduces across axis_len elements.
// ============================================================================

template <typename T, typename Op, typename Identity>
__global__ void axis_reduce_kernel(const T *src, T *dst,
                                   size_t outer, size_t axis_len,
                                   size_t inner, Op op) {
    extern __shared__ char smem_raw[];
    T *smem = reinterpret_cast<T *>(smem_raw);

    // Each block covers one output element (outer_idx, inner_idx)
    size_t out_idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (out_idx >= outer * inner) return;

    size_t outer_idx = out_idx / inner;
    size_t inner_idx = out_idx % inner;

    // Base offset in the flattened (outer, axis_len, inner) layout
    size_t base = outer_idx * axis_len * inner + inner_idx;

    // Sequential reduction across axis_len for this thread
    T acc = Identity::value();
    for (size_t a = 0; a < axis_len; ++a) {
        acc = op(acc, src[base + a * inner]);
    }

    dst[out_idx] = acc;
}

template <typename T>
static void axis_reduce_typed(ReduceOpKind op, const T *src, T *dst,
                              size_t outer, size_t axis_len, size_t inner,
                              cudaStream_t stream) {
    size_t total = outer * inner;
    auto &lc = CudaLaunchCache::instance();

    switch (op) {
    case ReduceOpKind::Sum: {
        auto lp = lc.params_for(
            axis_reduce_kernel<T, ReduceSumOp, SumIdentity<T>>, total);
        axis_reduce_kernel<T, ReduceSumOp, SumIdentity<T>>
            <<<lp.grid, lp.block, 0, stream>>>(
                src, dst, outer, axis_len, inner, ReduceSumOp{});
        break;
    }
    case ReduceOpKind::Max: {
        auto lp = lc.params_for(
            axis_reduce_kernel<T, ReduceMaxOp, MaxIdentity<T>>, total);
        axis_reduce_kernel<T, ReduceMaxOp, MaxIdentity<T>>
            <<<lp.grid, lp.block, 0, stream>>>(
                src, dst, outer, axis_len, inner, ReduceMaxOp{});
        break;
    }
    case ReduceOpKind::Min: {
        auto lp = lc.params_for(
            axis_reduce_kernel<T, ReduceMinOp, MinIdentity<T>>, total);
        axis_reduce_kernel<T, ReduceMinOp, MinIdentity<T>>
            <<<lp.grid, lp.block, 0, stream>>>(
                src, dst, outer, axis_len, inner, ReduceMinOp{});
        break;
    }
    case ReduceOpKind::Prod: {
        auto lp = lc.params_for(
            axis_reduce_kernel<T, ReduceProdOp, ProdIdentity<T>>, total);
        axis_reduce_kernel<T, ReduceProdOp, ProdIdentity<T>>
            <<<lp.grid, lp.block, 0, stream>>>(
                src, dst, outer, axis_len, inner, ReduceProdOp{});
        break;
    }
    case ReduceOpKind::Any: {
        auto lp = lc.params_for(
            axis_reduce_kernel<T, ReduceAnyOp, SumIdentity<T>>, total);
        axis_reduce_kernel<T, ReduceAnyOp, SumIdentity<T>>
            <<<lp.grid, lp.block, 0, stream>>>(
                src, dst, outer, axis_len, inner, ReduceAnyOp{});
        break;
    }
    case ReduceOpKind::All: {
        auto lp = lc.params_for(
            axis_reduce_kernel<T, ReduceAllOp, ProdIdentity<T>>, total);
        axis_reduce_kernel<T, ReduceAllOp, ProdIdentity<T>>
            <<<lp.grid, lp.block, 0, stream>>>(
                src, dst, outer, axis_len, inner, ReduceAllOp{});
        break;
    }
    }
}

void launch_axis_reduce(ReduceOpKind op, const void *src, void *dst,
                        size_t outer, size_t axis_len, size_t inner,
                        size_t element_size, cudaStream_t stream) {
    switch (element_size) {
    case 4:
        axis_reduce_typed(op, static_cast<const float *>(src),
                          static_cast<float *>(dst), outer, axis_len,
                          inner, stream);
        break;
    case 8:
        axis_reduce_typed(op, static_cast<const double *>(src),
                          static_cast<double *>(dst), outer, axis_len,
                          inner, stream);
        break;
    case 2:
        axis_reduce_typed(op, static_cast<const int16_t *>(src),
                          static_cast<int16_t *>(dst), outer, axis_len,
                          inner, stream);
        break;
    case 1:
        axis_reduce_typed(op, static_cast<const int8_t *>(src),
                          static_cast<int8_t *>(dst), outer, axis_len,
                          inner, stream);
        break;
    default:
        throw std::runtime_error("launch_axis_reduce: unsupported element size " +
                                 std::to_string(element_size));
    }
}

// ============================================================================
// Full ArgMax / ArgMin via CUB
// ============================================================================

template <typename T>
static void full_argreduce_typed(bool is_max, const T *src, int64_t *dst,
                                 size_t n, void *temp, size_t &temp_bytes,
                                 cudaStream_t stream) {
    // CUB returns cub::KeyValuePair<int, T>; we need a device buffer
    // for the pair, then copy the key (index) to dst.
    using KV = cub::KeyValuePair<int, T>;

    // We store the KV result in the first sizeof(KV) bytes of dst's
    // backing memory — but dst is int64_t*, so we use temp space.
    // On the first call (temp == nullptr) we query temp_bytes only.
    // On the second call we need temp_bytes + sizeof(KV).
    size_t cub_temp = 0;
    KV *d_kv = nullptr;

    if (temp == nullptr) {
        // Query phase — ask CUB how much it needs, add KV storage
        if (is_max)
            cub::DeviceReduce::ArgMax(nullptr, cub_temp, src, d_kv,
                                      static_cast<int>(n), stream);
        else
            cub::DeviceReduce::ArgMin(nullptr, cub_temp, src, d_kv,
                                      static_cast<int>(n), stream);
        temp_bytes = cub_temp + sizeof(KV);
        return;
    }

    // Execute phase
    cub_temp = temp_bytes - sizeof(KV);
    d_kv = reinterpret_cast<KV *>(static_cast<uint8_t *>(temp) + cub_temp);

    if (is_max)
        cub::DeviceReduce::ArgMax(temp, cub_temp, src, d_kv,
                                  static_cast<int>(n), stream);
    else
        cub::DeviceReduce::ArgMin(temp, cub_temp, src, d_kv,
                                  static_cast<int>(n), stream);

    // Copy the index from the KV pair to dst
    // We launch a tiny kernel to avoid a device→host→device round trip.
    // Use cudaMemcpyAsync for the 4 bytes we need.
    cudaMemcpyAsync(dst, &d_kv->key, sizeof(int), cudaMemcpyDeviceToDevice,
                    stream);
}

void launch_full_argreduce(bool is_max, const void *src, void *dst,
                           size_t n, size_t element_size,
                           void *temp, size_t &temp_bytes,
                           cudaStream_t stream) {
    switch (element_size) {
    case 4:
        full_argreduce_typed(is_max, static_cast<const float *>(src),
                             static_cast<int64_t *>(dst), n, temp,
                             temp_bytes, stream);
        break;
    case 8:
        full_argreduce_typed(is_max, static_cast<const double *>(src),
                             static_cast<int64_t *>(dst), n, temp,
                             temp_bytes, stream);
        break;
    case 2:
        full_argreduce_typed(is_max, static_cast<const int16_t *>(src),
                             static_cast<int64_t *>(dst), n, temp,
                             temp_bytes, stream);
        break;
    case 1:
        full_argreduce_typed(is_max, static_cast<const int8_t *>(src),
                             static_cast<int64_t *>(dst), n, temp,
                             temp_bytes, stream);
        break;
    default:
        throw std::runtime_error(
            "launch_full_argreduce: unsupported element size " +
            std::to_string(element_size));
    }
}

// ============================================================================
// Axis ArgMax / ArgMin kernel
// ============================================================================

template <typename T>
__global__ void axis_argreduce_kernel(const T *src, int64_t *dst,
                                      size_t outer, size_t axis_len,
                                      size_t inner, bool is_max) {
    size_t out_idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (out_idx >= outer * inner) return;

    size_t outer_idx = out_idx / inner;
    size_t inner_idx = out_idx % inner;
    size_t base = outer_idx * axis_len * inner + inner_idx;

    T best_val = src[base];
    int64_t best_idx = 0;

    for (size_t a = 1; a < axis_len; ++a) {
        T val = src[base + a * inner];
        bool better = is_max ? (val > best_val) : (val < best_val);
        if (better) {
            best_val = val;
            best_idx = static_cast<int64_t>(a);
        }
    }

    dst[out_idx] = best_idx;
}

void launch_axis_argreduce(bool is_max, const void *src, void *dst,
                           size_t outer, size_t axis_len, size_t inner,
                           size_t element_size, cudaStream_t stream) {
    size_t total = outer * inner;
    auto &lc = CudaLaunchCache::instance();

    switch (element_size) {
    case 4: {
        auto lp = lc.params_for(axis_argreduce_kernel<float>, total);
        axis_argreduce_kernel<float><<<lp.grid, lp.block, 0, stream>>>(
            static_cast<const float *>(src), static_cast<int64_t *>(dst),
            outer, axis_len, inner, is_max);
        break;
    }
    case 8: {
        auto lp = lc.params_for(axis_argreduce_kernel<double>, total);
        axis_argreduce_kernel<double><<<lp.grid, lp.block, 0, stream>>>(
            static_cast<const double *>(src), static_cast<int64_t *>(dst),
            outer, axis_len, inner, is_max);
        break;
    }
    case 2: {
        auto lp = lc.params_for(axis_argreduce_kernel<int16_t>, total);
        axis_argreduce_kernel<int16_t><<<lp.grid, lp.block, 0, stream>>>(
            static_cast<const int16_t *>(src), static_cast<int64_t *>(dst),
            outer, axis_len, inner, is_max);
        break;
    }
    case 1: {
        auto lp = lc.params_for(axis_argreduce_kernel<int8_t>, total);
        axis_argreduce_kernel<int8_t><<<lp.grid, lp.block, 0, stream>>>(
            static_cast<const int8_t *>(src), static_cast<int64_t *>(dst),
            outer, axis_len, inner, is_max);
        break;
    }
    default:
        throw std::runtime_error(
            "launch_axis_argreduce: unsupported element size " +
            std::to_string(element_size));
    }
}

} // namespace cuda
} // namespace backends
} // namespace axiom

#endif // AXIOM_CUDA_SUPPORT
