#include "cuda_storage.hpp"
#include "cuda_context.hpp"

#include "axiom/error.hpp"

#ifdef AXIOM_CUDA_SUPPORT
#include <cuda_runtime.h>
#endif

namespace axiom {
namespace backends {
namespace cuda {

CudaStorage::CudaStorage(size_t size_bytes)
    : device_ptr_(nullptr), size_bytes_(size_bytes), offset_(0) {
#ifdef AXIOM_CUDA_SUPPORT
    cudaError_t err = cudaMalloc(&device_ptr_, size_bytes);
    if (err != cudaSuccess) {
        throw DeviceError(std::string("cudaMalloc failed: ") +
                          cudaGetErrorString(err));
    }
#else
    throw DeviceError("CUDA support not compiled");
#endif
}

CudaStorage::~CudaStorage() {
#ifdef AXIOM_CUDA_SUPPORT
    if (device_ptr_) {
        cudaFree(device_ptr_);
    }
#endif
}

void *CudaStorage::data() {
    throw DeviceError(
        "CUDA GPU memory is not directly accessible from the CPU");
}

const void *CudaStorage::data() const {
    throw DeviceError(
        "CUDA GPU memory is not directly accessible from the CPU");
}

size_t CudaStorage::size_bytes() const { return size_bytes_; }

void CudaStorage::copy_to(Storage &other) const {
#ifdef AXIOM_CUDA_SUPPORT
    auto stream =
        static_cast<cudaStream_t>(CudaContext::instance().stream());

    if (other.device() == Device::GPU) {
        // D2D — extract device pointer from the target
        auto *provider = dynamic_cast<CudaBufferProvider *>(&other);
        if (!provider) {
            throw DeviceError("Target GPU storage is not CUDA-backed");
        }
        auto *dst = static_cast<uint8_t *>(provider->device_ptr()) +
                    provider->offset();
        auto *src = static_cast<const uint8_t *>(device_ptr_) + offset_;
        cudaMemcpyAsync(dst, src, size_bytes_, cudaMemcpyDeviceToDevice,
                        stream);
    } else {
        // D2H — target is CPU storage, data() gives host pointer
        auto *src = static_cast<const uint8_t *>(device_ptr_) + offset_;
        cudaMemcpyAsync(other.data(), src, size_bytes_,
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
    }
#else
    (void)other;
    throw DeviceError("CUDA support not compiled");
#endif
}

void CudaStorage::copy_from(const Storage &other) {
#ifdef AXIOM_CUDA_SUPPORT
    auto stream =
        static_cast<cudaStream_t>(CudaContext::instance().stream());
    auto *dst = static_cast<uint8_t *>(device_ptr_) + offset_;

    if (other.device() == Device::GPU) {
        // D2D — extract device pointer from the source
        auto *provider =
            dynamic_cast<const CudaBufferProvider *>(&other);
        if (!provider) {
            throw DeviceError("Source GPU storage is not CUDA-backed");
        }
        auto *src =
            static_cast<const uint8_t *>(provider->device_ptr()) +
            provider->offset();
        cudaMemcpyAsync(dst, src, size_bytes_, cudaMemcpyDeviceToDevice,
                        stream);
    } else {
        // H2D — source is CPU storage, data() gives host pointer
        cudaMemcpyAsync(dst, other.data(), size_bytes_,
                        cudaMemcpyHostToDevice, stream);
    }
#else
    (void)other;
    throw DeviceError("CUDA support not compiled");
#endif
}

std::unique_ptr<Storage> CudaStorage::clone() const {
    auto cloned = std::make_unique<CudaStorage>(size_bytes_);
#ifdef AXIOM_CUDA_SUPPORT
    auto stream =
        static_cast<cudaStream_t>(CudaContext::instance().stream());
    auto *src = static_cast<const uint8_t *>(device_ptr_) + offset_;
    cudaMemcpyAsync(cloned->device_ptr_, src, size_bytes_,
                    cudaMemcpyDeviceToDevice, stream);
#endif
    return cloned;
}

std::unique_ptr<Storage> make_cuda_storage(size_t size_bytes) {
    return std::make_unique<CudaStorage>(size_bytes);
}

} // namespace cuda
} // namespace backends
} // namespace axiom
