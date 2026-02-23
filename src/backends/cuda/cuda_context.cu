#include "cuda_context.hpp"

#include <mutex>

#ifdef AXIOM_CUDA_SUPPORT
#include <cuda_runtime.h>
#endif

namespace axiom {
namespace backends {
namespace cuda {

CudaContext &CudaContext::instance() {
    static CudaContext ctx;
    return ctx;
}

int CudaContext::device_id() const { return device_id_; }

void *CudaContext::stream() const { return stream_; }

void CudaContext::synchronize() {
#ifdef AXIOM_CUDA_SUPPORT
    cudaStreamSynchronize(static_cast<cudaStream_t>(stream_));
#endif
}

CudaContext::CudaContext() : device_id_(0), stream_(nullptr) {
#ifdef AXIOM_CUDA_SUPPORT
    cudaSetDevice(device_id_);
    cudaStream_t s = nullptr;
    cudaStreamCreate(&s);
    stream_ = static_cast<void *>(s);
#endif
}

CudaContext::~CudaContext() {
#ifdef AXIOM_CUDA_SUPPORT
    if (stream_) {
        cudaStreamDestroy(static_cast<cudaStream_t>(stream_));
    }
#endif
}

bool is_cuda_available() {
    static std::once_flag flag;
    static bool available = false;

    std::call_once(flag, [] {
#ifdef AXIOM_CUDA_SUPPORT
        int count = 0;
        cudaError_t err = cudaGetDeviceCount(&count);
        if (err != cudaSuccess || count == 0) return;

        // Require compute capability >= 7.0 (Volta)
        cudaDeviceProp props{};
        err = cudaGetDeviceProperties(&props, 0);
        if (err != cudaSuccess) return;
        if (props.major < 7) return;

        available = true;
#endif
    });

    return available;
}

} // namespace cuda
} // namespace backends
} // namespace axiom
