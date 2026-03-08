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
    void *stream() const;          // cudaStream_t
    void *cublas_handle() const;   // cublasHandle_t
    void *cusolver_handle() const; // cusolverDnHandle_t

    // Synchronize the default stream.
    void synchronize();

  private:
    CudaContext();
    ~CudaContext();

    CudaContext(const CudaContext &) = delete;
    CudaContext &operator=(const CudaContext &) = delete;

    int device_id_;
    void *stream_;          // cudaStream_t
    void *cublas_handle_;   // cublasHandle_t
    void *cusolver_handle_; // cusolverDnHandle_t
};

// ============================================================================
// CUDA Execution Stream
// ============================================================================

// Thin wrapper around the CudaContext stream that tracks in-flight work
// and auto-synchronizes after MAX_BATCH_SIZE operations.  Simpler than
// MetalExecutionStream because CUDA streams don't require explicit
// command-buffer creation/commit cycles.
class CudaExecutionStream {
  public:
    static CudaExecutionStream &instance();

    // Block until all operations on the stream have completed.
    void synchronize();

    // Record that one more kernel/operation was enqueued.
    // Triggers an automatic synchronize() every MAX_BATCH_SIZE ops.
    void increment_batch();

    // True when at least one operation is pending since the last sync.
    bool has_pending_work() const;

    // Number of operations since the last synchronize().
    size_t current_batch_size() const { return batch_count_; }

    static constexpr size_t MAX_BATCH_SIZE = 64;

  private:
    CudaExecutionStream();
    ~CudaExecutionStream();

    CudaExecutionStream(const CudaExecutionStream &) = delete;
    CudaExecutionStream &operator=(const CudaExecutionStream &) = delete;

    size_t batch_count_;
    mutable std::mutex mutex_;
};

// Check if a CUDA-capable device is present at runtime.
bool is_cuda_available();

} // namespace cuda
} // namespace backends
} // namespace axiom
