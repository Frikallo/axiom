#import "metal_common.hpp"

#import <Metal/Metal.h>
#import "metal_storage.hpp"

namespace axiom {
namespace backends {
namespace metal {

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

        // Logic from ggml to find the metallib file
        NSString *mainBundlePath = [[NSBundle mainBundle] bundlePath];
        NSString *path = [mainBundlePath stringByAppendingPathComponent:@"default.metallib"];

        if ([[NSFileManager defaultManager] fileExistsAtPath:path]) {
            g_default_library = [device newLibraryWithFile:path error:&error];
        } else {
            // Fallback for when not running in a bundle (e.g. command line tools)
            g_default_library = [device newDefaultLibrary];
        }

        if (!g_default_library) {
            NSLog(@"Failed to load Metal library: %@", error);
            throw std::runtime_error("Failed to load default Metal library.");
        }
    });
}

void* get_default_library() {
    init_default_library();
    return (__bridge void*)g_default_library;
}

} // namespace metal
} // namespace backends
} // namespace axiom 