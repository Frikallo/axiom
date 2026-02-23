#include "cuda_context.hpp"

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
#ifdef AXIOM_CUDA_SUPPORT
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    return err == cudaSuccess && count > 0;
#else
    return false;
#endif
}

} // namespace cuda
} // namespace backends
} // namespace axiom
