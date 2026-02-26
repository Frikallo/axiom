#pragma once

#ifdef AXIOM_CUDA_SUPPORT
#include <cuda.h>

#include <string>

namespace axiom {
namespace backends {
namespace cuda {

struct CompiledKernel {
    CUmodule module = nullptr;
    CUfunction function = nullptr;
};

// Compile a CUDA C source string via NVRTC and load the resulting PTX.
// Returns a module/function pair ready for cuLaunchKernel.
CompiledKernel nvrtc_compile(const std::string &source,
                             const std::string &entry_point,
                             const std::string &compute_cap = "");

// Unload the module and null both fields.
void nvrtc_release(CompiledKernel &kernel);

// Query the compute capability of the current device (e.g. "sm_86").
std::string current_compute_capability();

} // namespace cuda
} // namespace backends
} // namespace axiom

#endif
