#include "cuda_operations.hpp"
#include "cublas_operations.hpp"
#include "cuda_buffer_provider.hpp"
#include "cuda_context.hpp"
#include "cuda_kernels.hpp"

#include "axiom/dtype.hpp"
#include "axiom/tensor.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace axiom {
namespace backends {
namespace cuda {

// ============================================================================
// ensure_gpu_contiguous — every CUDA op should call this on its inputs
// ============================================================================

Tensor ensure_gpu_contiguous(const Tensor &t) {
    if (t.is_contiguous()) return t;

#ifdef AXIOM_CUDA_SUPPORT
    Tensor result(t.shape(), t.dtype(), Device::GPU);

    auto *src_provider = as_cuda_buffer_provider(t.storage().get());
    auto *dst_provider = as_cuda_buffer_provider(result.storage().get());
    if (!src_provider || !dst_provider) {
        throw DeviceError("ensure_gpu_contiguous: storage is not CUDA-backed");
    }

    const auto *src_ptr =
        static_cast<const uint8_t *>(src_provider->device_ptr()) +
        src_provider->offset() + t.offset();
    auto *dst_ptr =
        static_cast<uint8_t *>(dst_provider->device_ptr()) +
        dst_provider->offset();

    GatherStridedParams params{};
    params.ndim = static_cast<unsigned int>(t.ndim());
    params.numel = static_cast<unsigned int>(t.size());
    params.offset = 0;
    params.itemsize = static_cast<unsigned int>(t.itemsize());
    params.flip_mask = 0;

    for (size_t i = 0; i < t.ndim(); ++i) {
        params.shape[i] = static_cast<unsigned int>(t.shape()[i]);
        int64_t stride = t.strides()[i];
        if (stride < 0) {
            params.flip_mask |= (1u << i);
        }
        params.src_strides[i] =
            static_cast<unsigned int>(std::abs(stride) /
                                      static_cast<int64_t>(t.itemsize()));
    }

    auto stream =
        static_cast<cudaStream_t>(CudaContext::instance().stream());

    launch_gather_strided(src_ptr, dst_ptr, params, t.itemsize(), stream);

    CudaExecutionStream::instance().increment_batch();
    return result;
#else
    throw DeviceError("CUDA support not compiled");
#endif
}

// ============================================================================
// Operation registration
// ============================================================================

void register_cuda_operations() {
    if (!is_cuda_available()) return;

    register_cublas_operations();

    // TODO: register element-wise, reduction, and custom kernel operations
}

} // namespace cuda
} // namespace backends
} // namespace axiom
