#pragma once

#include <stdexcept>

// Forward-declare Objective-C types
#ifdef __OBJC__
@class MTLDevice;
@class MTLCommandQueue;
@class MTLLibrary;
#else
typedef void MTLDevice;
typedef void MTLCommandQueue;
typedef void MTLLibrary;
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

} // namespace metal
} // namespace backends
} // namespace axiom