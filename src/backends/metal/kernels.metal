//
//  kernels.metal
//  axiom
//
//  Created by Noah Kay on 2/10/24.
//

#include <metal_stdlib>
using namespace metal;

#define MAX_DIMS 8

// Must match the order in metal_operations.hpp
enum MetalOpType {
    Add,
    Subtract,
    Multiply,
    Divide,
    Negate,
    Abs,
    Sqrt,
    Exp,
    Log,
    Sin,
    Cos,
    Tan,

    // Reductions
    Sum,
    Mean,
    Max,
    Min
};

struct KernelParams {
    MetalOpType op_type;
    uint rank;
    uint lhs_shape[MAX_DIMS];
    uint lhs_strides[MAX_DIMS];
    uint rhs_shape[MAX_DIMS];
    uint rhs_strides[MAX_DIMS];
    uint result_shape[MAX_DIMS];
    uint result_strides[MAX_DIMS];
};

struct ReductionKernelParams {
    MetalOpType op_type;
    uint rank;
    uint output_rank;
    uint input_shape[MAX_DIMS];
    uint input_strides[MAX_DIMS];
    uint output_shape[MAX_DIMS];
    uint output_strides[MAX_DIMS];
    uint reduction_axes[MAX_DIMS];
    uint num_reduction_axes;
    uint reduction_size;
};

template<typename T>
kernel void binary_kernel(device const T* a [[buffer(0)]],
                          device const T* b [[buffer(1)]],
                          device T* result [[buffer(2)]],
                          constant KernelParams& params [[buffer(3)]],
                          uint index [[thread_position_in_grid]])
{
    uint coords[MAX_DIMS];
    uint temp_index = index;
    
    // Unravel the 1D index into N-dimensional coordinates
    #pragma unroll
    for (int i = params.rank - 1; i >= 0; --i) {
        coords[i] = temp_index % params.result_shape[i];
        temp_index /= params.result_shape[i];
    }
    
    uint lhs_index = 0;
    uint rhs_index = 0;
    
    // Calculate 1D indices for lhs and rhs using strides
    #pragma unroll
    for (int i = 0; i < params.rank; ++i) {
        lhs_index += coords[i] * params.lhs_strides[i];
        rhs_index += coords[i] * params.rhs_strides[i];
    }

    switch(params.op_type) {
        case MetalOpType::Add:
            result[index] = a[lhs_index] + b[rhs_index];
            break;
        case MetalOpType::Subtract:
            result[index] = a[lhs_index] - b[rhs_index];
            break;
        case MetalOpType::Multiply:
            result[index] = a[lhs_index] * b[rhs_index];
            break;
        case MetalOpType::Divide:
            result[index] = a[lhs_index] / b[rhs_index];
            break;
    }
}

// Instantiate kernels for each supported data type
#define INSTANTIATE_BINARY_KERNEL(TYPE) \
    template [[host_name("binary_kernel_" #TYPE )]] \
    kernel void binary_kernel<TYPE>(device const TYPE* a [[buffer(0)]], \
                                    device const TYPE* b [[buffer(1)]], \
                                    device TYPE* result [[buffer(2)]], \
                                    constant KernelParams& params [[buffer(3)]], \
                                    uint index [[thread_position_in_grid]]);

INSTANTIATE_BINARY_KERNEL(float)
INSTANTIATE_BINARY_KERNEL(int)
// INSTANTIATE_BINARY_KERNEL(half) - TODO: Add half support
INSTANTIATE_BINARY_KERNEL(uint8_t)
INSTANTIATE_BINARY_KERNEL(int8_t)

// ============================================================================
// Unary Kernels
// ============================================================================

// Must match C++ struct
struct UnaryKernelParams {
    MetalOpType op_type;
    uint rank;
    uint input_shape[MAX_DIMS];
    uint input_strides[MAX_DIMS];
};

// Templated unary kernel
template<typename T>
kernel void unary_kernel(device const T *input [[buffer(0)]],
                         device T *result [[buffer(1)]],
                         constant UnaryKernelParams &params [[buffer(2)]],
                         uint index [[thread_position_in_grid]]) {

    uint coords[MAX_DIMS];
    uint temp_index = index;
    
    // Unravel the 1D index into N-dimensional coordinates based on result shape (which is same as input shape)
    #pragma unroll
    for (int i = params.rank - 1; i >= 0; --i) {
        coords[i] = temp_index % params.input_shape[i];
        temp_index /= params.input_shape[i];
    }
    
    uint input_offset = 0;
    
    // Calculate 1D index for input using strides
    #pragma unroll
    for (int i = 0; i < params.rank; ++i) {
        input_offset += coords[i] * params.input_strides[i];
    }
    
    T val = input[input_offset];

    switch (params.op_type) {
        case MetalOpType::Negate:
            result[index] = -val;
            break;
        case MetalOpType::Abs:
            result[index] = abs(val);
            break;
        case MetalOpType::Sqrt:
            result[index] = sqrt(val);
            break;
        case MetalOpType::Exp:
            result[index] = exp(val);
            break;
        case MetalOpType::Log:
            result[index] = log(val);
            break;
        case MetalOpType::Sin:
            result[index] = sin(val);
            break;
        case MetalOpType::Cos:
            result[index] = cos(val);
            break;
        case MetalOpType::Tan:
            result[index] = tan(val);
            break;
        default:
            // Should not happen
            break;
    }
}

// Explicit instantiations for supported types
template
[[host_name("unary_kernel_float")]]
kernel void unary_kernel<float>(device const float *input [[buffer(0)]],
                                device float *result [[buffer(1)]],
                                constant UnaryKernelParams &params [[buffer(2)]],
                                uint gid [[thread_position_in_grid]]);

template
[[host_name("unary_kernel_half")]]
kernel void unary_kernel<half>(device const half *input [[buffer(0)]],
                               device half *result [[buffer(1)]],
                               constant UnaryKernelParams &params [[buffer(2)]],
                               uint gid [[thread_position_in_grid]]);

// ============================================================================
// Reduction Kernels
// ============================================================================

// Utility to check if an axis is a reduction axis
bool is_reduction_axis(uint axis, constant ReductionKernelParams& params) {
    for (uint i = 0; i < params.num_reduction_axes; ++i) {
        if (params.reduction_axes[i] == axis) {
            return true;
        }
    }
    return false;
}

// Maps a global thread ID to an output coordinate
void get_output_coords(uint gid, constant ReductionKernelParams& params, thread uint* output_coords) {
    uint temp_index = gid;

    // Unravel gid into a compact coordinate array based on the output shape.
    uint compact_coords[MAX_DIMS];
    for (int i = params.output_rank - 1; i >= 0; --i) {
        compact_coords[i] = temp_index % params.output_shape[i];
        temp_index /= params.output_shape[i];
    }

    // Map the compact coordinates to the full-rank output_coords.
    // Non-reduction axes get the calculated coordinate, reduction axes are set to 0.
    int compact_idx = 0;
    for (uint i = 0; i < params.rank; ++i) {
        if (!is_reduction_axis(i, params)) {
            output_coords[i] = compact_coords[compact_idx++];
        } else {
            output_coords[i] = 0;
        }
    }
}


template<typename T>
kernel void reduction_kernel(device const T* input [[buffer(0)]],
                             device T* result [[buffer(1)]],
                             constant ReductionKernelParams& params [[buffer(2)]],
                             uint gid [[thread_position_in_grid]])
{
    // Each thread computes one element of the output tensor.
    uint output_coords[MAX_DIMS];
    get_output_coords(gid, params, output_coords);

    if (params.reduction_size == 0) {
        // Define identity elements for reductions over empty sets.
        switch (params.op_type) {
            case MetalOpType::Sum:
            case MetalOpType::Mean:
                result[gid] = T(0);
                break;
            case MetalOpType::Min:
                 result[gid] = INFINITY;
                 break;
            case MetalOpType::Max:
                 result[gid] = -INFINITY;
                 break;
            default:
                 break; // Should not be reached
        }
        return;
    }

    // Initialize accumulator
    T acc;
    if (params.op_type == MetalOpType::Sum || params.op_type == MetalOpType::Mean) {
        acc = T(0);
    }
    bool first_element = true;

    // Simulate nested loops over the reduction axes.
    // `reduction_coords` holds the indices for the dimensions being reduced.
    uint reduction_coords[MAX_DIMS];
    #pragma unroll
    for (uint i = 0; i < params.rank; ++i) {
        reduction_coords[i] = 0;
    }

    for (uint iter = 0; iter < params.reduction_size; ++iter) {
        uint current_input_coords[MAX_DIMS];
        uint input_offset = 0;
        
        // Combine the base output coordinate with the current reduction coordinate
        #pragma unroll
        for(uint j = 0; j < params.rank; ++j) {
            current_input_coords[j] = is_reduction_axis(j, params) ? reduction_coords[j] : output_coords[j];
        }
        
        // Calculate 1D offset from the full coordinate using input strides
        #pragma unroll
        for (uint j = 0; j < params.rank; ++j) {
            input_offset += current_input_coords[j] * params.input_strides[j];
        }
        
        T val = input[input_offset];

        // Accumulate
        if (params.op_type == MetalOpType::Min || params.op_type == MetalOpType::Max) {
            if (first_element) {
                acc = val;
                first_element = false;
            } else {
                if (params.op_type == MetalOpType::Min) acc = min(acc, val);
                else acc = max(acc, val);
            }
        } else { // Sum or Mean
            acc += val;
        }

        // Increment the reduction_coords to the next logical element, handling carry.
        #pragma unroll
        for (int j = params.rank - 1; j >= 0; --j) {
            if (is_reduction_axis(j, params)) {
                reduction_coords[j]++;
                if (reduction_coords[j] < params.input_shape[j]) {
                    // This dimension did not overflow, so we're done incrementing.
                    break;
                }
                // This dimension overflowed. Reset to 0 and continue loop to carry over the increment.
                reduction_coords[j] = 0;
            }
        }
    }

    if (params.op_type == MetalOpType::Mean && params.reduction_size > 0) {
        acc /= T(params.reduction_size);
    }
    
    result[gid] = acc;
}


// Explicit instantiations for supported types
template
[[host_name("reduction_kernel_float")]]
kernel void reduction_kernel<float>(device const float *input [[buffer(0)]],
                                    device float *result [[buffer(1)]],
                                    constant ReductionKernelParams &params [[buffer(2)]],
                                    uint gid [[thread_position_in_grid]]);

template
[[host_name("reduction_kernel_half")]]
kernel void reduction_kernel<half>(device const half *input [[buffer(0)]],
                                   device half *result [[buffer(1)]],
                                   constant ReductionKernelParams &params [[buffer(2)]],
                                   uint gid [[thread_position_in_grid]]);

// ============================================================================
// Matrix Multiplication Kernels
// ============================================================================

// Tile size for shared memory tiling - optimized for Apple Silicon
// 8x8 tiles work well for general cases; could be 16x16 or 32x32 for larger matrices
constant uint TILE_SIZE = 8;

struct MatMulKernelParams {
    // Matrix dimensions: C[M, N] = A[M, K] @ B[K, N]
    uint M;              // Output rows
    uint N;              // Output cols
    uint K;              // Inner dimension (reduction)

    // Strides for A matrix (in elements, not bytes)
    // These handle both contiguous and transposed views
    uint a_row_stride;   // Stride to move one row in A
    uint a_col_stride;   // Stride to move one column in A

    // Strides for B matrix (in elements)
    uint b_row_stride;   // Stride to move one row in B
    uint b_col_stride;   // Stride to move one column in B

    // Strides for C matrix (in elements)
    uint c_row_stride;   // Stride to move one row in C
    uint c_col_stride;   // Stride to move one column in C

    // Batch information
    uint batch_size;             // Total number of batch elements
    uint batch_ndim;             // Number of batch dimensions
    uint batch_shape[MAX_DIMS];  // Shape of batch dimensions
    uint a_batch_strides[MAX_DIMS];  // Batch strides for A
    uint b_batch_strides[MAX_DIMS];  // Batch strides for B
    uint c_batch_strides[MAX_DIMS];  // Batch strides for C
};

// Optimized tiled matrix multiplication kernel
// Uses threadgroup memory to reduce global memory accesses
// Handles arbitrary strides for zero-copy transposed views
template<typename T>
kernel void matmul_tiled_kernel(
    device const T* A [[buffer(0)]],
    device const T* B [[buffer(1)]],
    device T* C [[buffer(2)]],
    constant MatMulKernelParams& params [[buffer(3)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]]
) {
    uint batch_id = tgid.z;

    // Threadgroup shared memory for tiles
    threadgroup T A_tile[TILE_SIZE][TILE_SIZE];
    threadgroup T B_tile[TILE_SIZE][TILE_SIZE];

    // Global row and column for this thread's output element
    uint row = tgid.y * TILE_SIZE + tid.y;
    uint col = tgid.x * TILE_SIZE + tid.x;

    // Compute batch offsets using broadcasted batch strides
    uint a_batch_offset = 0;
    uint b_batch_offset = 0;
    uint c_batch_offset = 0;

    if (params.batch_size > 1) {
        // Unravel batch_id into batch coordinates
        uint batch_coords[MAX_DIMS];
        uint temp_batch_id = batch_id;
        for (int i = int(params.batch_ndim) - 1; i >= 0; --i) {
            batch_coords[i] = temp_batch_id % params.batch_shape[i];
            temp_batch_id /= params.batch_shape[i];
        }

        // Compute batch offsets (strides of 0 enable broadcasting)
        for (uint i = 0; i < params.batch_ndim; ++i) {
            a_batch_offset += batch_coords[i] * params.a_batch_strides[i];
            b_batch_offset += batch_coords[i] * params.b_batch_strides[i];
            c_batch_offset += batch_coords[i] * params.c_batch_strides[i];
        }
    }

    // Offset pointers for this batch
    device const T* A_batch = A + a_batch_offset;
    device const T* B_batch = B + b_batch_offset;
    device T* C_batch = C + c_batch_offset;

    // Accumulator for the dot product
    T sum = T(0);

    // Number of tiles needed to cover K dimension
    uint num_tiles = (params.K + TILE_SIZE - 1) / TILE_SIZE;

    for (uint tile = 0; tile < num_tiles; ++tile) {
        // Load A tile: A[row, tile * TILE_SIZE + tid.x]
        uint a_col = tile * TILE_SIZE + tid.x;
        if (row < params.M && a_col < params.K) {
            // Use strides for zero-copy transposed access
            A_tile[tid.y][tid.x] = A_batch[row * params.a_row_stride + a_col * params.a_col_stride];
        } else {
            A_tile[tid.y][tid.x] = T(0);
        }

        // Load B tile: B[tile * TILE_SIZE + tid.y, col]
        uint b_row = tile * TILE_SIZE + tid.y;
        if (b_row < params.K && col < params.N) {
            // Use strides for zero-copy transposed access
            B_tile[tid.y][tid.x] = B_batch[b_row * params.b_row_stride + col * params.b_col_stride];
        } else {
            B_tile[tid.y][tid.x] = T(0);
        }

        // Synchronize to ensure tiles are loaded
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute partial dot product for this tile
        #pragma unroll
        for (uint k = 0; k < TILE_SIZE; ++k) {
            sum += A_tile[tid.y][k] * B_tile[k][tid.x];
        }

        // Synchronize before loading next tile
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write result to C using strides
    if (row < params.M && col < params.N) {
        C_batch[row * params.c_row_stride + col * params.c_col_stride] = sum;
    }
}

// Alternative: Simple non-tiled kernel for small matrices or fallback
// Each thread computes one element of C
template<typename T>
kernel void matmul_simple_kernel(
    device const T* A [[buffer(0)]],
    device const T* B [[buffer(1)]],
    device T* C [[buffer(2)]],
    constant MatMulKernelParams& params [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;
    uint batch_id = gid.z;

    if (row >= params.M || col >= params.N || batch_id >= params.batch_size) {
        return;
    }

    // Compute batch offsets
    uint a_batch_offset = 0;
    uint b_batch_offset = 0;
    uint c_batch_offset = 0;

    if (params.batch_size > 1) {
        uint batch_coords[MAX_DIMS];
        uint temp_batch_id = batch_id;
        for (int i = int(params.batch_ndim) - 1; i >= 0; --i) {
            batch_coords[i] = temp_batch_id % params.batch_shape[i];
            temp_batch_id /= params.batch_shape[i];
        }

        for (uint i = 0; i < params.batch_ndim; ++i) {
            a_batch_offset += batch_coords[i] * params.a_batch_strides[i];
            b_batch_offset += batch_coords[i] * params.b_batch_strides[i];
            c_batch_offset += batch_coords[i] * params.c_batch_strides[i];
        }
    }

    device const T* A_batch = A + a_batch_offset;
    device const T* B_batch = B + b_batch_offset;
    device T* C_batch = C + c_batch_offset;

    // Compute dot product for this element
    T sum = T(0);
    for (uint k = 0; k < params.K; ++k) {
        T a_val = A_batch[row * params.a_row_stride + k * params.a_col_stride];
        T b_val = B_batch[k * params.b_row_stride + col * params.b_col_stride];
        sum += a_val * b_val;
    }

    C_batch[row * params.c_row_stride + col * params.c_col_stride] = sum;
}

// Explicit instantiations for tiled kernel
template
[[host_name("matmul_tiled_kernel_float")]]
kernel void matmul_tiled_kernel<float>(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant MatMulKernelParams& params [[buffer(3)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]]);

template
[[host_name("matmul_tiled_kernel_half")]]
kernel void matmul_tiled_kernel<half>(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* C [[buffer(2)]],
    constant MatMulKernelParams& params [[buffer(3)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]]);

// Explicit instantiations for simple kernel
template
[[host_name("matmul_simple_kernel_float")]]
kernel void matmul_simple_kernel<float>(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant MatMulKernelParams& params [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]);

template
[[host_name("matmul_simple_kernel_half")]]
kernel void matmul_simple_kernel<half>(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* C [[buffer(2)]],
    constant MatMulKernelParams& params [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]);