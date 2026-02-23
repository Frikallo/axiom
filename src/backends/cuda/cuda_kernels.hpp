#pragma once

#include <cstddef>
#include <cstdint>

#ifdef AXIOM_CUDA_SUPPORT
#include <cuda_runtime.h>
#endif

namespace axiom {
namespace backends {
namespace cuda {

static constexpr int MAX_DIMS = 8;

struct GatherStridedParams {
    unsigned int ndim;
    unsigned int numel;
    unsigned int offset;    // Byte offset into source buffer
    unsigned int itemsize;  // Size of each element in bytes
    unsigned int shape[MAX_DIMS];
    unsigned int src_strides[MAX_DIMS]; // Strides in ELEMENTS, always positive
    unsigned int flip_mask; // Bitmask: bit i set if axis i has negative stride
};

// Identifies which binary operation to launch.
enum class BinaryOpKind : uint8_t {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Mod,
    Max,
    Min,
    Atan2,
    Hypot,
    // Comparison ops — output is uint8_t (bool)
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    // Logical ops — inputs cast to bool, output uint8_t
    LogicalAnd,
    LogicalOr,
    LogicalXor,
};

#ifdef AXIOM_CUDA_SUPPORT
// Launch gather_strided for the given dtype.  Dispatches to the correct
// template instantiation defined in cuda_kernels.cu.
void launch_gather_strided(const void *src, void *dst,
                           const GatherStridedParams &params,
                           size_t element_size, cudaStream_t stream);

// Launch a binary element-wise kernel.
// For arithmetic ops (Add..Hypot) src_a, src_b, and dst share the same
// element_size.  For comparison/logical ops dst is always uint8_t.
void launch_binary_elementwise(BinaryOpKind op, const void *src_a,
                               const void *src_b, void *dst, size_t n,
                               size_t element_size, cudaStream_t stream);
#endif

} // namespace cuda
} // namespace backends
} // namespace axiom
