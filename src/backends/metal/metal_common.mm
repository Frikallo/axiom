#import "metal_common.hpp"

#import <Metal/Metal.h>
#import "axiom/error.hpp"
#import "metal_storage.hpp"

#ifdef AXIOM_METAL_EMBED_LIBRARY
extern "C" {
    extern const char axiom_metal_kernels_start;
    extern const char axiom_metal_kernels_end;
}
#endif

namespace axiom {
namespace backends {
namespace metal {

// ============================================================================
// MetalContext Implementation
// ============================================================================

MetalContext& MetalContext::instance() {
    static MetalContext instance;
    return instance;
}

MetalContext::MetalContext() {
    if (axiom::backends::metal::is_metal_available()) {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        device_ = (void *)CFBridgingRetain(device);
        command_queue_ = (void *)CFBridgingRetain([device newCommandQueue]);
    } else {
        device_ = nil;
        command_queue_ = nil;
    }
}

MetalContext::~MetalContext() {
    if (command_queue_) {
        CFRelease(command_queue_);
    }
    if (device_) {
        CFRelease(device_);
    }
}

void* MetalContext::device() const {
    return device_;
}

void* MetalContext::command_queue() const {
    return command_queue_;
}

static id<MTLLibrary> g_default_library = nil;
static dispatch_once_t g_library_once;

void init_default_library() {
    dispatch_once(&g_library_once, ^{
        if (!axiom::backends::metal::is_metal_available()) return;

        id<MTLDevice> device = (__bridge id<MTLDevice>)MetalContext::instance().device();
        NSError* error = nil;

#ifdef AXIOM_METAL_EMBED_LIBRARY
        dispatch_data_t lib_data = dispatch_data_create(
                &axiom_metal_kernels_start,
                &axiom_metal_kernels_end - &axiom_metal_kernels_start,
                dispatch_get_main_queue(),
                DISPATCH_DATA_DESTRUCTOR_DEFAULT);

        g_default_library = [device newLibraryWithData:lib_data error:&error];
#else
        // Logic from ggml to find the metallib file
        NSString *mainBundlePath = [[NSBundle mainBundle] bundlePath];
        NSString *path = [mainBundlePath stringByAppendingPathComponent:@"default.metallib"];

        if ([[NSFileManager defaultManager] fileExistsAtPath:path]) {
            g_default_library = [device newLibraryWithFile:path error:&error];
        } else {
            // Fallback for when not running in a bundle (e.g. command line tools)
            g_default_library = [device newDefaultLibrary];
        }
#endif
        if (!g_default_library) {
            NSLog(@"Failed to load Metal library: %@", error);
            throw DeviceError("Failed to load default Metal library");
        }
    });
}

void* get_default_library() {
    init_default_library();
    return (__bridge void*)g_default_library;
}

// ============================================================================
// MetalExecutionStream Implementation
// ============================================================================

MetalExecutionStream& MetalExecutionStream::instance() {
    static MetalExecutionStream instance;
    return instance;
}

MetalExecutionStream::MetalExecutionStream()
    : command_queue_(nullptr), current_buffer_(nullptr), batch_count_(0) {
    if (axiom::backends::metal::is_metal_available()) {
        command_queue_ = MetalContext::instance().command_queue();
    }
}

MetalExecutionStream::~MetalExecutionStream() {
    // Synchronize any pending work before destruction
    if (current_buffer_) {
        synchronize();
    }
}

void MetalExecutionStream::create_new_buffer() {
    // Called with mutex held
    if (!command_queue_) {
        throw DeviceError("Metal command queue not available");
    }

    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)command_queue_;
    id<MTLCommandBuffer> buffer = [queue commandBuffer];

    if (!buffer) {
        throw DeviceError("Failed to create Metal command buffer");
    }

    // Retain the buffer
    current_buffer_ = (void *)CFBridgingRetain(buffer);
    batch_count_ = 0;
}

void* MetalExecutionStream::current_buffer() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!current_buffer_) {
        create_new_buffer();
    }

    return current_buffer_;
}

void MetalExecutionStream::commit() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (current_buffer_) {
        id<MTLCommandBuffer> buffer = (__bridge id<MTLCommandBuffer>)current_buffer_;
        [buffer commit];

        // Release our reference
        CFRelease(current_buffer_);
        current_buffer_ = nullptr;
        batch_count_ = 0;
    }
}

void MetalExecutionStream::synchronize() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (current_buffer_) {
        id<MTLCommandBuffer> buffer = (__bridge id<MTLCommandBuffer>)current_buffer_;
        [buffer commit];
        [buffer waitUntilCompleted];

        // Check for errors
        if ([buffer status] == MTLCommandBufferStatusError) {
            NSLog(@"Metal command buffer error: %@", [buffer error]);
            throw DeviceError("Metal command buffer execution failed");
        }

        // Release our reference
        CFRelease(current_buffer_);
        current_buffer_ = nullptr;
        batch_count_ = 0;
    }
}

bool MetalExecutionStream::has_pending_work() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return current_buffer_ != nullptr && batch_count_ > 0;
}

void MetalExecutionStream::increment_batch() {
    std::lock_guard<std::mutex> lock(mutex_);
    ++batch_count_;

    // Auto-commit if batch is full (for latency control)
    if (batch_count_ >= MAX_BATCH_SIZE) {
        if (current_buffer_) {
            id<MTLCommandBuffer> buffer = (__bridge id<MTLCommandBuffer>)current_buffer_;
            [buffer commit];
            CFRelease(current_buffer_);
            current_buffer_ = nullptr;
            batch_count_ = 0;
        }
    }
}

} // namespace metal
} // namespace backends
} // namespace axiom 