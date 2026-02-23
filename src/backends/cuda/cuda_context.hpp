#pragma once

#include <cstddef>
#include <mutex>
#include <stdexcept>

namespace axiom {
namespace backends {
namespace cuda {

// Singleton managing the CUDA device, stream, cuBLAS handle, and context
// lifetime.
class CudaContext {
  public:
    static CudaContext &instance();

    int device_id() const;
    void *stream() const;         // cudaStream_t
    void *cublas_handle() const;  // cublasHandle_t

    // Synchronize the default stream.
    void synchronize();

  private:
    CudaContext();
    ~CudaContext();

    CudaContext(const CudaContext &) = delete;
    CudaContext &operator=(const CudaContext &) = delete;

    int device_id_;
    void *stream_;         // cudaStream_t
    void *cublas_handle_;  // cublasHandle_t
};

// Check if a CUDA-capable device is present at runtime.
bool is_cuda_available();

} // namespace cuda
} // namespace backends
} // namespace axiom
