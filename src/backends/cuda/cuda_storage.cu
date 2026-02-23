#include "cuda_storage.hpp"
#include "cuda_context.hpp"

#ifdef AXIOM_CUDA_SUPPORT
#include <cuda_runtime.h>
#endif

#include <cstring>
#include <stdexcept>

namespace axiom {
namespace backends {
namespace cuda {

CudaStorage::CudaStorage(size_t size_bytes)
    : device_ptr_(nullptr), size_bytes_(size_bytes), offset_(0) {
#ifdef AXIOM_CUDA_SUPPORT
    cudaError_t err = cudaMalloc(&device_ptr_, size_bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMalloc failed: ") +
                                 cudaGetErrorString(err));
    }
#else
    throw std::runtime_error("CUDA support not compiled");
#endif
}

CudaStorage::~CudaStorage() {
#ifdef AXIOM_CUDA_SUPPORT
    if (device_ptr_) {
        cudaFree(device_ptr_);
    }
#endif
}

void *CudaStorage::data() { return device_ptr_; }

const void *CudaStorage::data() const { return device_ptr_; }

size_t CudaStorage::size_bytes() const { return size_bytes_; }

void CudaStorage::copy_to(Storage &other) const {
#ifdef AXIOM_CUDA_SUPPORT
    if (other.device() == Device::GPU) {
        cudaMemcpy(other.data(), device_ptr_, size_bytes_,
                   cudaMemcpyDeviceToDevice);
    } else {
        cudaMemcpy(other.data(), device_ptr_, size_bytes_,
                   cudaMemcpyDeviceToHost);
    }
#else
    (void)other;
    throw std::runtime_error("CUDA support not compiled");
#endif
}

void CudaStorage::copy_from(const Storage &other) {
#ifdef AXIOM_CUDA_SUPPORT
    if (other.device() == Device::GPU) {
        cudaMemcpy(device_ptr_, other.data(), size_bytes_,
                   cudaMemcpyDeviceToDevice);
    } else {
        cudaMemcpy(device_ptr_, other.data(), size_bytes_,
                   cudaMemcpyHostToDevice);
    }
#else
    (void)other;
    throw std::runtime_error("CUDA support not compiled");
#endif
}

std::unique_ptr<Storage> CudaStorage::clone() const {
    auto cloned = std::make_unique<CudaStorage>(size_bytes_);
#ifdef AXIOM_CUDA_SUPPORT
    cudaMemcpy(cloned->device_ptr_, device_ptr_, size_bytes_,
               cudaMemcpyDeviceToDevice);
#endif
    return cloned;
}

std::unique_ptr<Storage> make_cuda_storage(size_t size_bytes) {
    return std::make_unique<CudaStorage>(size_bytes);
}

} // namespace cuda
} // namespace backends
} // namespace axiom
