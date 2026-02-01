#import "metal_common.hpp"

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#import "axiom/error.hpp"

// Expose private enableCommitAndContinue property (used by PyTorch)
@interface MPSGraphExecutionDescriptor ()
@property(readwrite, atomic) BOOL enableCommitAndContinue;
@end
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
    : command_queue_(nullptr), current_buffer_(nullptr), mps_buffer_(nullptr),
      execution_descriptor_(nullptr), compilation_descriptor_(nullptr),
      compute_encoder_(nullptr), batch_count_(0) {
    if (axiom::backends::metal::is_metal_available()) {
        command_queue_ = MetalContext::instance().command_queue();

        // Create execution descriptor with commitAndContinue enabled
        // This allows continuous encoding without blocking
        MPSGraphExecutionDescriptor* execDesc = [MPSGraphExecutionDescriptor new];
        execDesc.enableCommitAndContinue = YES;

        // Create compilation descriptor with GPU optimization level
        MPSGraphCompilationDescriptor* compDesc = [MPSGraphCompilationDescriptor new];
        compDesc.optimizationLevel = MPSGraphOptimizationLevel0;  // Optimizes for GPU

        execDesc.compilationDescriptor = compDesc;

        execution_descriptor_ = (void*)CFBridgingRetain(execDesc);
        compilation_descriptor_ = (void*)CFBridgingRetain(compDesc);
    }
}

MetalExecutionStream::~MetalExecutionStream() {
    // Synchronize any pending work before destruction
    if (current_buffer_ || mps_buffer_ || compute_encoder_) {
        synchronize();
    }

    // Release descriptors
    if (compilation_descriptor_) {
        CFRelease(compilation_descriptor_);
        compilation_descriptor_ = nullptr;
    }
    if (execution_descriptor_) {
        CFRelease(execution_descriptor_);
        execution_descriptor_ = nullptr;
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

void* MetalExecutionStream::current_mps_buffer() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!mps_buffer_) {
        if (!command_queue_) {
            throw DeviceError("Metal command queue not available");
        }

        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)command_queue_;
        MPSCommandBuffer* buffer =
            [MPSCommandBuffer commandBufferFromCommandQueue:queue];

        if (!buffer) {
            throw DeviceError("Failed to create MPSCommandBuffer");
        }

        mps_buffer_ = (void*)CFBridgingRetain(buffer);
        batch_count_ = 0;
    }

    return mps_buffer_;
}

void* MetalExecutionStream::execution_descriptor() {
    // No locking needed - execution_descriptor_ is created once in constructor
    return execution_descriptor_;
}

void* MetalExecutionStream::compute_encoder() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!compute_encoder_) {
        // Need to create from MPS buffer's underlying MTL command buffer
        if (!mps_buffer_) {
            // Create MPS buffer first
            if (!command_queue_) {
                throw DeviceError("Metal command queue not available");
            }
            id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)command_queue_;
            MPSCommandBuffer* buffer =
                [MPSCommandBuffer commandBufferFromCommandQueue:queue];
            if (!buffer) {
                throw DeviceError("Failed to create MPSCommandBuffer");
            }
            mps_buffer_ = (void*)CFBridgingRetain(buffer);
            batch_count_ = 0;
        }

        MPSCommandBuffer* mps_buffer = (__bridge MPSCommandBuffer*)mps_buffer_;
        id<MTLComputeCommandEncoder> encoder = [mps_buffer.commandBuffer computeCommandEncoder];
        if (!encoder) {
            throw DeviceError("Failed to create compute command encoder");
        }
        compute_encoder_ = (void*)CFBridgingRetain(encoder);
    }

    return compute_encoder_;
}

// Internal helper - ends compute encoder without locking (caller must hold mutex_)
void MetalExecutionStream::end_kernel_coalescing_internal() {
    if (compute_encoder_) {
        id<MTLComputeCommandEncoder> encoder =
            (__bridge id<MTLComputeCommandEncoder>)compute_encoder_;
        [encoder endEncoding];
        CFRelease(compute_encoder_);
        compute_encoder_ = nullptr;
    }
}

void MetalExecutionStream::end_kernel_coalescing() {
    std::lock_guard<std::mutex> lock(mutex_);
    end_kernel_coalescing_internal();
}

void MetalExecutionStream::commit() {
    std::lock_guard<std::mutex> lock(mutex_);

    // End any active compute encoder first
    end_kernel_coalescing_internal();

    // Commit MPSCommandBuffer if active (preferred path for MPSGraph ops)
    // Use commitAndContinue to allow reusing the same command buffer for
    // continuous encoding without blocking
    if (mps_buffer_) {
        MPSCommandBuffer* buffer = (__bridge MPSCommandBuffer*)mps_buffer_;
        [buffer commitAndContinue];
        batch_count_ = 0;
        // Note: Don't release buffer - commitAndContinue allows continued use
        return;
    }

    // Fallback: commit regular MTLCommandBuffer
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

    // End any active compute encoder first
    end_kernel_coalescing_internal();

    // Synchronize MPSCommandBuffer if active (preferred path for MPSGraph ops)
    if (mps_buffer_) {
        MPSCommandBuffer* buffer = (__bridge MPSCommandBuffer*)mps_buffer_;
        [buffer commit];
        [buffer waitUntilCompleted];

        // Check for errors via the underlying MTLCommandBuffer
        if ([buffer.commandBuffer status] == MTLCommandBufferStatusError) {
            NSLog(@"Metal command buffer error: %@", buffer.commandBuffer.error);
            CFRelease(mps_buffer_);
            mps_buffer_ = nullptr;
            batch_count_ = 0;
            throw DeviceError("Metal command buffer execution failed");
        }

        CFRelease(mps_buffer_);
        mps_buffer_ = nullptr;
        batch_count_ = 0;
        return;
    }

    // Fallback: synchronize regular MTLCommandBuffer
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
    return (current_buffer_ != nullptr || mps_buffer_ != nullptr ||
            compute_encoder_ != nullptr) && batch_count_ > 0;
}

void MetalExecutionStream::increment_batch() {
    std::lock_guard<std::mutex> lock(mutex_);
    ++batch_count_;

    // Auto-commit if batch is full (for latency control)
    // Use commitAndContinue to maintain continuous encoding
    if (batch_count_ >= MAX_BATCH_SIZE) {
        // End any active compute encoder first
        end_kernel_coalescing_internal();

        // Prefer committing MPSCommandBuffer if active
        if (mps_buffer_) {
            MPSCommandBuffer* buffer = (__bridge MPSCommandBuffer*)mps_buffer_;
            [buffer commitAndContinue];
            batch_count_ = 0;
            // Note: Don't release buffer - commitAndContinue allows continued use
        } else if (current_buffer_) {
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