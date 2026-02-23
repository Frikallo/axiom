#include "cuda_unified_storage.hpp"
#include "cuda_context.hpp"

#include "axiom/error.hpp"

#ifdef AXIOM_CUDA_SUPPORT
#include <cuda_runtime.h>
#endif

#include <cstring>
#include <mutex>

namespace axiom {
namespace backends {
namespace cuda {

// ============================================================================
// Private alias constructor (zero-copy, does not own memory)
// ============================================================================

CudaUnifiedStorage::CudaUnifiedStorage(void *managed_ptr, size_t size_bytes,
                                       size_t offset, Device tag)
    : managed_ptr_(managed_ptr),
      size_bytes_(size_bytes),
      offset_(offset),
      device_tag_(tag),
      owns_memory_(false) {}

// ============================================================================
// Public constructor (allocates managed memory)
// ============================================================================

CudaUnifiedStorage::CudaUnifiedStorage(size_t size_bytes, Device device_tag)
    : managed_ptr_(nullptr),
      size_bytes_(size_bytes),
      offset_(0),
      device_tag_(device_tag),
      owns_memory_(true) {
#ifdef AXIOM_CUDA_SUPPORT
    cudaError_t err =
        cudaMallocManaged(&managed_ptr_, size_bytes, cudaMemAttachGlobal);
    if (err != cudaSuccess) {
        throw DeviceError(std::string("cudaMallocManaged failed: ") +
                          cudaGetErrorString(err));
    }
#else
    throw DeviceError("CUDA support not compiled");
#endif
}

CudaUnifiedStorage::~CudaUnifiedStorage() {
#ifdef AXIOM_CUDA_SUPPORT
    if (owns_memory_ && managed_ptr_) {
        cudaFree(managed_ptr_);
    }
#endif
}

// ============================================================================
// Storage interface
// ============================================================================

void *CudaUnifiedStorage::data() {
    return static_cast<uint8_t *>(managed_ptr_) + offset_;
}

const void *CudaUnifiedStorage::data() const {
    return static_cast<const uint8_t *>(managed_ptr_) + offset_;
}

size_t CudaUnifiedStorage::size_bytes() const { return size_bytes_; }

void CudaUnifiedStorage::copy_to(Storage &other) const {
#ifdef AXIOM_CUDA_SUPPORT
    auto stream =
        static_cast<cudaStream_t>(CudaContext::instance().stream());
    auto *src = static_cast<const uint8_t *>(managed_ptr_) + offset_;

    if (other.device() == Device::GPU) {
        auto *provider = dynamic_cast<CudaBufferProvider *>(&other);
        if (provider) {
            auto *dst = static_cast<uint8_t *>(provider->device_ptr()) +
                        provider->offset();
            cudaMemcpyAsync(dst, src, size_bytes_, cudaMemcpyDefault,
                            stream);
        } else {
            throw DeviceError("Target GPU storage is not CUDA-backed");
        }
    } else {
        // Managed → CPU: memcpy after stream sync
        cudaStreamSynchronize(stream);
        std::memcpy(other.data(), src, size_bytes_);
    }
#else
    (void)other;
    throw DeviceError("CUDA support not compiled");
#endif
}

void CudaUnifiedStorage::copy_from(const Storage &other) {
#ifdef AXIOM_CUDA_SUPPORT
    auto stream =
        static_cast<cudaStream_t>(CudaContext::instance().stream());
    auto *dst = static_cast<uint8_t *>(managed_ptr_) + offset_;

    if (other.device() == Device::GPU) {
        auto *provider =
            dynamic_cast<const CudaBufferProvider *>(&other);
        if (provider) {
            auto *src =
                static_cast<const uint8_t *>(provider->device_ptr()) +
                provider->offset();
            cudaMemcpyAsync(dst, src, size_bytes_, cudaMemcpyDefault,
                            stream);
        } else {
            throw DeviceError("Source GPU storage is not CUDA-backed");
        }
    } else {
        // CPU → managed: memcpy after stream sync
        cudaStreamSynchronize(stream);
        std::memcpy(dst, other.data(), size_bytes_);
    }
#else
    (void)other;
    throw DeviceError("CUDA support not compiled");
#endif
}

std::unique_ptr<Storage> CudaUnifiedStorage::clone() const {
    auto cloned =
        std::make_unique<CudaUnifiedStorage>(size_bytes_, device_tag_);
#ifdef AXIOM_CUDA_SUPPORT
    auto stream =
        static_cast<cudaStream_t>(CudaContext::instance().stream());
    auto *src = static_cast<const uint8_t *>(managed_ptr_) + offset_;
    cudaMemcpyAsync(cloned->managed_ptr_, src, size_bytes_,
                    cudaMemcpyDefault, stream);
#endif
    return cloned;
}

// ============================================================================
// Zero-copy alias
// ============================================================================

std::unique_ptr<CudaUnifiedStorage>
CudaUnifiedStorage::with_device_tag(Device tag) const {
    return std::unique_ptr<CudaUnifiedStorage>(
        new CudaUnifiedStorage(managed_ptr_, size_bytes_, offset_, tag));
}

// ============================================================================
// Availability detection
// ============================================================================

bool is_cuda_unified_memory_available() {
    static std::once_flag flag;
    static bool available = false;

    std::call_once(flag, [] {
#ifdef AXIOM_CUDA_SUPPORT
        int val = 0;
        cudaError_t err = cudaDeviceGetAttribute(
            &val, cudaDevAttrManagedMemory, 0);
        available = (err == cudaSuccess && val != 0);
#endif
    });

    return available;
}

// ============================================================================
// Factory
// ============================================================================

std::unique_ptr<Storage> make_cuda_unified_storage(size_t size_bytes,
                                                   Device device_tag) {
    return std::make_unique<CudaUnifiedStorage>(size_bytes, device_tag);
}

} // namespace cuda
} // namespace backends
} // namespace axiom
