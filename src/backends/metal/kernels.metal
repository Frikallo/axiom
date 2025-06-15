#include <metal_stdlib>

using namespace metal;

kernel void add_kernel(device float *result [[buffer(0)]],
                       const device float *a [[buffer(1)]],
                       const device float *b [[buffer(2)]],
                       uint index [[thread_position_in_grid]]) {
    result[index] = a[index] + b[index];
}

kernel void sub_kernel(device float *result [[buffer(0)]],
                       const device float *a [[buffer(1)]],
                       const device float *b [[buffer(2)]],
                       uint index [[thread_position_in_grid]]) {
    result[index] = a[index] - b[index];
} 