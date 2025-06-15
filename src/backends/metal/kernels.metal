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
    Tan
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