#pragma once

#include <cstddef>
#include <mutex>
#include <stdexcept>

// Forward-declare Objective-C types
#ifdef __OBJC__
@class MTLDevice;
@class MTLCommandQueue;
@class MTLLibrary;
@class MTLCommandBuffer;
#else
typedef void MTLDevice;
typedef void MTLCommandQueue;
typedef void MTLLibrary;
typedef void MTLCommandBuffer;
#endif

namespace axiom {
namespace backends {
namespace metal {

// A singleton to manage the application's single Metal device and command
// queue.
class MetalContext {
  public:
    static MetalContext &instance();

    void *device() const;
    void *command_queue() const;

  private:
    MetalContext();
    ~MetalContext();

    MetalContext(const MetalContext &) = delete;
    MetalContext &operator=(const MetalContext &) = delete;

    void *device_; // Using void* to avoid Obj-C headers
    void *command_queue_;
};

// Returns the default Metal library, compiled from kernels.metal
void *get_default_library();

// ============================================================================
// Metal Execution Stream
// ============================================================================

// A singleton to manage asynchronous GPU command execution.
// Batches multiple operations into a single command buffer for efficiency.
class MetalExecutionStream {
  public:
    // Get singleton instance
    static MetalExecutionStream &instance();

    // Get the current command buffer (creates one if needed)
    // The returned buffer is used for encoding GPU operations
    void *current_buffer();

    // Commit the current command buffer and create a new one
    // Call this after encoding a batch of operations
    void commit();

    // Synchronize: commit current buffer and wait for all pending work
    // Must be called before reading GPU data from CPU
    void synchronize();

    // Check if there are pending operations
    bool has_pending_work() const;

    // Get number of operations in current batch
    size_t current_batch_size() const { return batch_count_; }

    // Increment batch counter (call when encoding an operation)
    void increment_batch();

    // Maximum operations before auto-commit (for latency control)
    static constexpr size_t MAX_BATCH_SIZE = 64;

  private:
    MetalExecutionStream();
    ~MetalExecutionStream();

    MetalExecutionStream(const MetalExecutionStream &) = delete;
    MetalExecutionStream &operator=(const MetalExecutionStream &) = delete;

    // Create a new command buffer
    void create_new_buffer();

    void *command_queue_;      // id<MTLCommandQueue>
    void *current_buffer_;     // id<MTLCommandBuffer>
    size_t batch_count_;       // Operations in current batch
    mutable std::mutex mutex_; // Thread safety
};

} // namespace metal
} // namespace backends
} // namespace axiom