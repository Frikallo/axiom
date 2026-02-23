#include "axiom/storage.hpp"

#include "axiom/error.hpp"
#include "backends/cpu/cpu_storage.hpp"

#ifdef __APPLE__
namespace axiom {
namespace backends {
namespace metal {
std::unique_ptr<Storage> make_metal_storage(size_t size_bytes);
std::unique_ptr<Storage> make_unified_storage(size_t size_bytes,
                                              Device device_tag);
bool is_metal_available();
bool is_unified_memory_available();
} // namespace metal
} // namespace backends
} // namespace axiom
#elif defined(AXIOM_CUDA_SUPPORT)
namespace axiom {
namespace backends {
namespace cuda {
std::unique_ptr<Storage> make_cuda_storage(size_t size_bytes);
std::unique_ptr<Storage> make_cuda_unified_storage(size_t size_bytes,
                                                   Device device_tag);
bool is_cuda_available();
bool is_cuda_unified_memory_available();
} // namespace cuda
} // namespace backends
} // namespace axiom
#endif

namespace axiom {

// ============================================================================
// Factory functions
// ============================================================================

std::unique_ptr<Storage> make_storage(size_t size_bytes, Device device) {
    switch (device) {
    case Device::CPU:
        return backends::cpu::make_cpu_storage(size_bytes);

    case Device::GPU:
#ifdef __APPLE__
        if (backends::metal::is_metal_available()) {
            if (size_bytes > 0 &&
                backends::metal::is_unified_memory_available()) {
                return backends::metal::make_unified_storage(size_bytes,
                                                             Device::GPU);
            }
            return backends::metal::make_metal_storage(size_bytes);
        } else {
            throw DeviceError::not_available("Metal GPU");
        }
#elif defined(AXIOM_CUDA_SUPPORT)
        if (backends::cuda::is_cuda_available()) {
            if (size_bytes > 0 &&
                backends::cuda::is_cuda_unified_memory_available()) {
                return backends::cuda::make_cuda_unified_storage(size_bytes,
                                                                 Device::GPU);
            }
            return backends::cuda::make_cuda_storage(size_bytes);
        } else {
            throw DeviceError::not_available("CUDA GPU");
        }
#else
        throw DeviceError::not_available("GPU storage on this platform");
#endif
    }
    throw DeviceError("Unknown device type");
}

} // namespace axiom
