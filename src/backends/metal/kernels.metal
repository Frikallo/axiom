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
// All operations migrated to MPSGraph
// ============================================================================
// Binary arithmetic, unary operations, reductions, and matrix multiplication
// have all been migrated to MPSGraph for:
// - Automatic kernel fusion
// - Apple Silicon optimizations
// - Reduced code complexity
// - Better maintainability
//
// The only remaining kernel is gather_strided for copying non-contiguous
// tensor data to contiguous buffers (required for MPSGraph compatibility).
// ============================================================================

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
    uint src_strides[MAX_DIMS];  // Strides in ELEMENTS (not bytes)
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
    uint src_idx = 0;
    #pragma unroll
    for (uint i = 0; i < params.ndim; ++i) {
        src_idx += coords[i] * params.src_strides[i];
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
