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