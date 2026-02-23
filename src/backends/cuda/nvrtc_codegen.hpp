#pragma once

#ifdef AXIOM_CUDA_SUPPORT

#include "axiom/dtype.hpp"
#include "axiom/operations.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace axiom {
namespace backends {
namespace cuda {

struct FusedKernelSpec {
    std::vector<ops::OpType> op_chain;
    // Per-op input slot indices.  -1 means "use previous chain result".
    std::vector<std::vector<int>> input_slot_indices;
    DType compute_dtype{};
    DType output_dtype{};
    size_t num_external_inputs = 0;
    // Maps external input ordinal to buffer slot.
    std::vector<int> external_slots;
    bool needs_broadcast = false;
    int ndim = 0;
    // Per-input strides (only used when needs_broadcast == true).
    std::vector<std::vector<int64_t>> input_strides;
    std::vector<int64_t> output_shape;
};

struct GeneratedKernel {
    std::string source;
    std::string entry_point;
};

// Generate CUDA C source for a fused elementwise kernel.
GeneratedKernel generate_fused_kernel(const FusedKernelSpec &spec);

// Check if an op can be included in an NVRTC fused kernel.
bool is_nvrtc_supported_op(ops::OpType op);

// Check if a dtype is supported by NVRTC codegen.
bool is_nvrtc_supported_dtype(DType dtype);

} // namespace cuda
} // namespace backends
} // namespace axiom

#endif
