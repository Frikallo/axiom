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
    Divide
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