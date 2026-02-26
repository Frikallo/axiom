#include "nvrtc_compiler.hpp"

#ifdef AXIOM_CUDA_SUPPORT
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace axiom {
namespace backends {
namespace cuda {

static void check_nvrtc(nvrtcResult result, const char *msg) {
    if (result != NVRTC_SUCCESS)
        throw std::runtime_error(std::string(msg) + ": " +
                                 nvrtcGetErrorString(result));
}

static void check_driver(CUresult result, const char *msg) {
    if (result != CUDA_SUCCESS) {
        const char *err_str = nullptr;
        cuGetErrorString(result, &err_str);
        throw std::runtime_error(
            std::string(msg) + ": " +
            (err_str ? std::string(err_str) : "unknown driver error"));
    }
}

std::string current_compute_capability() {
    static std::string cached;
    static std::once_flag flag;
    std::call_once(flag, [] {
        cudaDeviceProp props{};
        cudaGetDeviceProperties(&props, 0);
        int arch = props.major * 10 + props.minor;

        // NVRTC may not support the device's architecture (e.g. Blackwell
        // on CUDA 12.0).  Query the NVRTC version and cap accordingly.
        int nvrtc_major = 0, nvrtc_minor = 0;
        nvrtcVersion(&nvrtc_major, &nvrtc_minor);
        int nvrtc_ver = nvrtc_major * 10 + nvrtc_minor;

        // Conservative max-arch lookup: NVRTC 12.x supports up to sm_90,
        // NVRTC 12.8+ adds sm_100/120.  Fall back to 90 when unsure.
        int max_arch = 90;
        if (nvrtc_ver >= 128)
            max_arch = 120;

        if (arch > max_arch)
            arch = max_arch;

        // Use compute_ prefix (PTX) for forward compatibility — the driver
        // will JIT-compile to the actual GPU architecture.
        cached = "compute_" + std::to_string(arch);
    });
    return cached;
}

static void ensure_driver_init() {
    static std::once_flag flag;
    std::call_once(flag, [] { check_driver(cuInit(0), "cuInit"); });
}

CompiledKernel nvrtc_compile(const std::string &source,
                             const std::string &entry_point,
                             const std::string &compute_cap) {
    std::string arch =
        compute_cap.empty() ? current_compute_capability() : compute_cap;
    std::string arch_flag = "--gpu-architecture=" + arch;

    nvrtcProgram prog = nullptr;
    check_nvrtc(
        nvrtcCreateProgram(&prog, source.c_str(), "fused.cu", 0, nullptr,
                           nullptr),
        "nvrtcCreateProgram");

    const char *opts[] = {arch_flag.c_str(), "--std=c++17", "--use_fast_math"};
    nvrtcResult compile_result =
        nvrtcCompileProgram(prog, 3, opts);

    if (compile_result != NVRTC_SUCCESS) {
        size_t log_size = 0;
        nvrtcGetProgramLogSize(prog, &log_size);
        std::string log(log_size, '\0');
        nvrtcGetProgramLog(prog, log.data());
        nvrtcDestroyProgram(&prog);
        throw std::runtime_error("NVRTC compilation failed:\n" + log);
    }

    size_t ptx_size = 0;
    check_nvrtc(nvrtcGetPTXSize(prog, &ptx_size), "nvrtcGetPTXSize");
    std::string ptx(ptx_size, '\0');
    check_nvrtc(nvrtcGetPTX(prog, ptx.data()), "nvrtcGetPTX");
    nvrtcDestroyProgram(&prog);

    ensure_driver_init();

    CUmodule mod = nullptr;
    check_driver(cuModuleLoadData(&mod, ptx.data()), "cuModuleLoadData");

    CUfunction func = nullptr;
    check_driver(cuModuleGetFunction(&func, mod, entry_point.c_str()),
                 "cuModuleGetFunction");

    return {mod, func};
}

void nvrtc_release(CompiledKernel &kernel) {
    if (kernel.module)
        cuModuleUnload(kernel.module);
    kernel.module = nullptr;
    kernel.function = nullptr;
}

} // namespace cuda
} // namespace backends
} // namespace axiom

#endif
