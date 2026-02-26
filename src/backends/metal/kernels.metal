//
//  kernels.metal
//  axiom
//
//  Created by Noah Kay on 2/10/24.
//

#include <metal_stdlib>
using namespace metal;

#define MAX_DIMS 8

// ============================================================================
// Gather Strided Kernel
// ============================================================================
// Copies non-contiguous (strided) tensor data to a contiguous buffer.
// This is needed because MPSGraph requires contiguous input data.
// 
// The kernel converts each output index to N-dimensional coordinates,
// then uses the source strides to compute the correct input offset.
// ============================================================================

struct GatherStridedParams {
    uint ndim;
    uint numel;
    uint offset;           // Byte offset into source buffer
    uint itemsize;         // Size of each element in bytes
    uint shape[MAX_DIMS];
    uint src_strides[MAX_DIMS];  // Strides in ELEMENTS (not bytes), always positive
    uint flip_mask;        // Bitmask: bit i set if axis i has negative stride (flipped)
};

template<typename T>
kernel void gather_strided(
    device const T* src [[buffer(0)]],
    device T* dst [[buffer(1)]],
    constant GatherStridedParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= params.numel) return;

    // Convert linear index to N-dimensional coordinates
    uint coords[MAX_DIMS];
    uint temp = gid;

    #pragma unroll
    for (int i = int(params.ndim) - 1; i >= 0; --i) {
        coords[i] = temp % params.shape[i];
        temp /= params.shape[i];
    }

    // Compute strided source offset using element strides
    // For flipped axes (negative stride), transform coord to (shape - 1 - coord)
    uint src_idx = 0;
    #pragma unroll
    for (uint i = 0; i < params.ndim; ++i) {
        uint coord = coords[i];
        // Check if this axis is flipped (has negative stride)
        if (params.flip_mask & (1u << i)) {
            coord = params.shape[i] - 1 - coord;
        }
        src_idx += coord * params.src_strides[i];
    }

    // Read from strided source, write to contiguous destination
    dst[gid] = src[src_idx];
}

// Explicit instantiations for supported types
template
[[host_name("gather_strided_float")]]
kernel void gather_strided<float>(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant GatherStridedParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]);

template
[[host_name("gather_strided_half")]]
kernel void gather_strided<half>(
    device const half* src [[buffer(0)]],
    device half* dst [[buffer(1)]],
    constant GatherStridedParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]);

template
[[host_name("gather_strided_int")]]
kernel void gather_strided<int>(
    device const int* src [[buffer(0)]],
    device int* dst [[buffer(1)]],
    constant GatherStridedParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]);

template
[[host_name("gather_strided_uint")]]
kernel void gather_strided<uint>(
    device const uint* src [[buffer(0)]],
    device uint* dst [[buffer(1)]],
    constant GatherStridedParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]);

template
[[host_name("gather_strided_short")]]
kernel void gather_strided<short>(
    device const short* src [[buffer(0)]],
    device short* dst [[buffer(1)]],
    constant GatherStridedParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]);

template
[[host_name("gather_strided_ushort")]]
kernel void gather_strided<ushort>(
    device const ushort* src [[buffer(0)]],
    device ushort* dst [[buffer(1)]],
    constant GatherStridedParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]);

template
[[host_name("gather_strided_char")]]
kernel void gather_strided<char>(
    device const char* src [[buffer(0)]],
    device char* dst [[buffer(1)]],
    constant GatherStridedParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]);

template
[[host_name("gather_strided_uchar")]]
kernel void gather_strided<uchar>(
    device const uchar* src [[buffer(0)]],
    device uchar* dst [[buffer(1)]],
    constant GatherStridedParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]);

// 64-bit types
template
[[host_name("gather_strided_long")]]
kernel void gather_strided<long>(
    device const long* src [[buffer(0)]],
    device long* dst [[buffer(1)]],
    constant GatherStridedParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]);

template
[[host_name("gather_strided_ulong")]]
kernel void gather_strided<ulong>(
    device const ulong* src [[buffer(0)]],
    device ulong* dst [[buffer(1)]],
    constant GatherStridedParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]);

// ============================================================================
// LayerNorm Kernel
// ============================================================================
// Custom Metal kernel for LayerNorm, replacing MPSGraph-based implementation.
// One threadgroup per normalization row. Uses vectorized 4-element reads and
// SIMD reduction for efficient mean/variance computation.
//
// For d_model=512: 128 threads × 4 reads = 512 elements (perfect fit).
// Adapted from PyTorch MPS backend's layer_norm_single_row pattern.
// ============================================================================

struct LayerNormParams {
    uint axis_size;   // number of elements per row (e.g. d_model=512)
    float eps;
};

template<typename T>
kernel void layer_norm_fwd(
    device const T* input   [[buffer(0)]],
    device T* output        [[buffer(1)]],
    device const T* weight  [[buffer(2)]],
    device const T* bias    [[buffer(3)]],
    constant LayerNormParams& params [[buffer(4)]],
    uint tg_id      [[threadgroup_position_in_grid]],
    uint tid         [[thread_position_in_threadgroup]],
    uint simd_lane   [[thread_index_in_simdgroup]],
    uint simd_group  [[simdgroup_index_in_threadgroup]])
{
    constexpr int N_READS = 4;

    uint row_offset = tg_id * params.axis_size;
    device const T* x = input + row_offset + tid * N_READS;
    device T* out = output + row_offset + tid * N_READS;

    // Partial sums for mean & variance (Welford-style via sum and sum_sq)
    float partial_sum = 0.0f;
    float partial_sum_sq = 0.0f;
    uint base_lane = tid * N_READS;

    if (base_lane + N_READS <= params.axis_size) {
        float4 v4 = float4(x[0], x[1], x[2], x[3]);
        partial_sum = v4.x + v4.y + v4.z + v4.w;
        partial_sum_sq = dot(v4, v4);
    } else {
        int remaining = params.axis_size - base_lane;
        if (remaining >= 3) {
            float3 v3 = float3(x[0], x[1], x[2]);
            partial_sum = v3.x + v3.y + v3.z;
            partial_sum_sq = dot(v3, v3);
        } else if (remaining >= 2) {
            float2 v2 = float2(x[0], x[1]);
            partial_sum = v2.x + v2.y;
            partial_sum_sq = dot(v2, v2);
        } else if (remaining >= 1) {
            float v = float(x[0]);
            partial_sum = v;
            partial_sum_sq = v * v;
        }
    }

    // Threadgroup-wide reduction via SIMD + shared memory
    constexpr uint SIMD_SIZE = 32;
    threadgroup float local_sums[SIMD_SIZE];
    threadgroup float local_sums_sq[SIMD_SIZE];
    threadgroup float tg_mean[1];
    threadgroup float tg_inv_std[1];

    if (simd_group == 0) {
        local_sums[simd_lane] = 0.0f;
        local_sums_sq[simd_lane] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each simdgroup reduces its partial sums
    float group_sum = simd_sum(partial_sum);
    float group_sum_sq = simd_sum(partial_sum_sq);
    if (simd_lane == 0) {
        local_sums[simd_group] = group_sum;
        local_sums_sq[simd_group] = group_sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // First simdgroup reduces across all simdgroups
    if (simd_group == 0) {
        float sum = simd_sum(local_sums[simd_lane]);
        float sum_sq = simd_sum(local_sums_sq[simd_lane]);
        if (simd_lane == 0) {
            float mean = sum / float(params.axis_size);
            float var = sum_sq / float(params.axis_size) - mean * mean;
            var = var < 1e-6f ? 0.0f : var;
            float inv_std = metal::precise::rsqrt(var + params.eps);
            tg_mean[0] = mean;
            tg_inv_std[0] = inv_std;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float mean = tg_mean[0];
    float inv_std = tg_inv_std[0];

    // Normalize, scale, and shift
    if (base_lane + N_READS <= params.axis_size) {
        #pragma unroll
        for (int i = 0; i < N_READS; i++) {
            float v = float(x[i]);
            float norm = (v - mean) * inv_std;
            norm = norm * float(weight[base_lane + i]) + float(bias[base_lane + i]);
            out[i] = static_cast<T>(norm);
        }
    } else {
        #pragma unroll
        for (int i = 0; i < N_READS; i++) {
            uint lane_idx = base_lane + i;
            if (lane_idx < params.axis_size) {
                float v = float(x[i]);
                float norm = (v - mean) * inv_std;
                norm = norm * float(weight[lane_idx]) + float(bias[lane_idx]);
                out[i] = static_cast<T>(norm);
            }
        }
    }
}

// Looped variant for large axis_size (> 1024 threads × 4 = 4096 elements)
template<typename T>
kernel void layer_norm_fwd_looped(
    device const T* input   [[buffer(0)]],
    device T* output        [[buffer(1)]],
    device const T* weight  [[buffer(2)]],
    device const T* bias    [[buffer(3)]],
    constant LayerNormParams& params [[buffer(4)]],
    uint tg_id      [[threadgroup_position_in_grid]],
    uint tid         [[thread_position_in_threadgroup]],
    uint lsize       [[threads_per_threadgroup]],
    uint simd_lane   [[thread_index_in_simdgroup]],
    uint simd_group  [[simdgroup_index_in_threadgroup]])
{
    constexpr int N_READS = 4;

    uint row_offset = tg_id * params.axis_size;
    device const T* x = input + row_offset;
    device T* out = output + row_offset;

    float partial_sum = 0.0f;
    float partial_sum_sq = 0.0f;

    for (uint r = 0; r < params.axis_size; r += lsize * N_READS) {
        uint base = r + tid * N_READS;
        if (base + N_READS <= params.axis_size) {
            float4 v4 = float4(x[base], x[base + 1], x[base + 2], x[base + 3]);
            partial_sum += v4.x + v4.y + v4.z + v4.w;
            partial_sum_sq += dot(v4, v4);
        } else {
            int remaining = params.axis_size - base;
            if (remaining >= 3) {
                float3 v3 = float3(x[base], x[base + 1], x[base + 2]);
                partial_sum += v3.x + v3.y + v3.z;
                partial_sum_sq += dot(v3, v3);
            } else if (remaining >= 2) {
                float2 v2 = float2(x[base], x[base + 1]);
                partial_sum += v2.x + v2.y;
                partial_sum_sq += dot(v2, v2);
            } else if (remaining >= 1) {
                float v = float(x[base]);
                partial_sum += v;
                partial_sum_sq += v * v;
            }
        }
    }

    partial_sum = simd_sum(partial_sum);
    partial_sum_sq = simd_sum(partial_sum_sq);

    constexpr uint SIMD_SIZE = 32;
    threadgroup float local_sums[SIMD_SIZE];
    threadgroup float local_sums_sq[SIMD_SIZE];
    threadgroup float tg_mean[1];
    threadgroup float tg_inv_std[1];

    if (simd_lane == 0) {
        local_sums[simd_group] = 0.0f;
        local_sums_sq[simd_group] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_lane == 0) {
        local_sums[simd_group] = partial_sum;
        local_sums_sq[simd_group] = partial_sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
        float s = simd_sum(local_sums[simd_lane]);
        float ss = simd_sum(local_sums_sq[simd_lane]);
        if (simd_lane == 0) {
            float mean = s / float(params.axis_size);
            float var = ss / float(params.axis_size) - mean * mean;
            var = var < 1e-6f ? 0.0f : var;
            float inv_std = metal::precise::rsqrt(var + params.eps);
            tg_mean[0] = mean;
            tg_inv_std[0] = inv_std;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float mean = tg_mean[0];
    float inv_std = tg_inv_std[0];

    for (uint r = 0; r < params.axis_size; r += lsize * N_READS) {
        uint base = r + tid * N_READS;
        if (base + N_READS <= params.axis_size) {
            #pragma unroll
            for (int i = 0; i < N_READS; i++) {
                float v = float(x[base + i]);
                float norm = (v - mean) * inv_std;
                norm = norm * float(weight[base + i]) + float(bias[base + i]);
                out[base + i] = static_cast<T>(norm);
            }
        } else {
            #pragma unroll
            for (int i = 0; i < N_READS; i++) {
                if (base + i < params.axis_size) {
                    float v = float(x[base + i]);
                    float norm = (v - mean) * inv_std;
                    norm = norm * float(weight[base + i]) + float(bias[base + i]);
                    out[base + i] = static_cast<T>(norm);
                }
            }
        }
    }
}

// Explicit instantiations
template
[[host_name("layer_norm_fwd_float")]]
kernel void layer_norm_fwd<float>(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const float* weight [[buffer(2)]],
    device const float* bias [[buffer(3)]],
    constant LayerNormParams& params [[buffer(4)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]);

template
[[host_name("layer_norm_fwd_half")]]
kernel void layer_norm_fwd<half>(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    device const half* weight [[buffer(2)]],
    device const half* bias [[buffer(3)]],
    constant LayerNormParams& params [[buffer(4)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]);

template
[[host_name("layer_norm_fwd_looped_float")]]
kernel void layer_norm_fwd_looped<float>(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const float* weight [[buffer(2)]],
    device const float* bias [[buffer(3)]],
    constant LayerNormParams& params [[buffer(4)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]);

template
[[host_name("layer_norm_fwd_looped_half")]]
kernel void layer_norm_fwd_looped<half>(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    device const half* weight [[buffer(2)]],
    device const half* bias [[buffer(3)]],
    constant LayerNormParams& params [[buffer(4)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]);
