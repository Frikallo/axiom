#import "metal_operations.hpp"

#import <Metal/Metal.h>
#import "axiom/tensor.hpp"
#import "axiom/shape.hpp"
#import "axiom/operations.hpp"
#import "metal_storage.hpp"

namespace axiom {
namespace backends {
namespace metal {

static id<MTLCommandQueue> g_command_queue = nil;
static dispatch_once_t g_metal_queue_once;

void init_metal_queue() {
    dispatch_once(&g_metal_queue_once, ^{
        if (is_metal_available()) {
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            g_command_queue = [device newCommandQueue];
        }
    });
}

struct MetalAdd : public ops::Operation {
    id<MTLComputePipelineState> pipeline_state_;

    MetalAdd() {
        init_metal_queue();
        if (!g_command_queue) {
            throw std::runtime_error("Metal command queue not initialized.");
        }

        const char* kernel_source = R"(
            #include <metal_stdlib>
            using namespace metal;
            kernel void add_kernel(device float *result [[buffer(0)]],
                                   const device float *a [[buffer(1)]],
                                   const device float *b [[buffer(2)]],
                                   uint index [[thread_position_in_grid]]) {
                result[index] = a[index] + b[index];
            }
        )";

        id<MTLDevice> device = [g_command_queue device];
        NSError* error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:kernel_source]
                                                      options:nil
                                                        error:&error];
        if (!library) {
            throw std::runtime_error("Failed to create Metal library");
        }

        id<MTLFunction> add_function = [library newFunctionWithName:@"add_kernel"];
        pipeline_state_ = [device newComputePipelineStateWithFunction:add_function error:&error];
        if (!pipeline_state_) {
            throw std::runtime_error("Failed to create Metal pipeline state");
        }
    }

    ops::OpType type() const override { return ops::OpType::Add; }
    std::string name() const override { return "add"; }
    Device device() const override { return Device::GPU; }

    Tensor execute_binary(const Tensor& lhs, const Tensor& rhs) const override {
        if (lhs.dtype() != DType::Float32 || rhs.dtype() != DType::Float32) {
            throw std::runtime_error("Metal add kernel only supports float32.");
        }
        
        Tensor result = Tensor(lhs.shape(), lhs.dtype(), Device::GPU);

        auto* lhs_storage = static_cast<const MetalStorage*>(lhs.storage().get());
        auto* rhs_storage = static_cast<const MetalStorage*>(rhs.storage().get());
        auto* res_storage = static_cast<MetalStorage*>(result.storage().get());
        
        id<MTLCommandBuffer> command_buffer = [g_command_queue commandBuffer];
        id<MTLComputeCommandEncoder> command_encoder = [command_buffer computeCommandEncoder];

        [command_encoder setComputePipelineState:pipeline_state_];

        [command_encoder setBuffer:(__bridge id<MTLBuffer>)res_storage->buffer() offset:res_storage->offset() atIndex:0];
        [command_encoder setBuffer:(__bridge id<MTLBuffer>)lhs_storage->buffer() offset:lhs_storage->offset() atIndex:1];
        [command_encoder setBuffer:(__bridge id<MTLBuffer>)rhs_storage->buffer() offset:rhs_storage->offset() atIndex:2];

        NSUInteger width = ShapeUtils::size(result.shape());
        NSUInteger max_threads = [pipeline_state_ maxTotalThreadsPerThreadgroup];
        MTLSize threads_per_group = MTLSizeMake(std::min(width, max_threads), 1, 1);
        MTLSize thread_groups = MTLSizeMake((width + threads_per_group.width - 1) / threads_per_group.width, 1, 1);

        [command_encoder dispatchThreadgroups:thread_groups threadsPerThreadgroup:threads_per_group];
        [command_encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        if ([command_buffer status] == MTLCommandBufferStatusError) {
            NSLog(@"Error: %@", [command_buffer error]);
            throw std::runtime_error("Metal command buffer execution failed.");
        }
        
        return result;
    }
};

struct MetalSub : public ops::Operation {
    id<MTLComputePipelineState> pipeline_state_;

    MetalSub() {
        init_metal_queue();
        if (!g_command_queue) {
            throw std::runtime_error("Metal command queue not initialized.");
        }

        const char* kernel_source = R"(
            #include <metal_stdlib>
            using namespace metal;
            kernel void sub_kernel(device float *result [[buffer(0)]],
                                   const device float *a [[buffer(1)]],
                                   const device float *b [[buffer(2)]],
                                   uint index [[thread_position_in_grid]]) {
                result[index] = a[index] - b[index];
            }
        )";

        id<MTLDevice> device = [g_command_queue device];
        NSError* error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:kernel_source]
                                                      options:nil
                                                        error:&error];
        if (!library) {
            throw std::runtime_error("Failed to create Metal library for subtraction");
        }

        id<MTLFunction> sub_function = [library newFunctionWithName:@"sub_kernel"];
        pipeline_state_ = [device newComputePipelineStateWithFunction:sub_function error:&error];
        if (!pipeline_state_) {
            throw std::runtime_error("Failed to create Metal pipeline state for subtraction");
        }
    }

    ops::OpType type() const override { return ops::OpType::Subtract; }
    std::string name() const override { return "subtract"; }
    Device device() const override { return Device::GPU; }

    Tensor execute_binary(const Tensor& lhs, const Tensor& rhs) const override {
        if (lhs.dtype() != DType::Float32 || rhs.dtype() != DType::Float32) {
            throw std::runtime_error("Metal subtract kernel only supports float32.");
        }
        
        Tensor result = Tensor(lhs.shape(), lhs.dtype(), Device::GPU);

        auto* lhs_storage = static_cast<const MetalStorage*>(lhs.storage().get());
        auto* rhs_storage = static_cast<const MetalStorage*>(rhs.storage().get());
        auto* res_storage = static_cast<MetalStorage*>(result.storage().get());
        
        id<MTLCommandBuffer> command_buffer = [g_command_queue commandBuffer];
        id<MTLComputeCommandEncoder> command_encoder = [command_buffer computeCommandEncoder];

        [command_encoder setComputePipelineState:pipeline_state_];

        [command_encoder setBuffer:(__bridge id<MTLBuffer>)res_storage->buffer() offset:res_storage->offset() atIndex:0];
        [command_encoder setBuffer:(__bridge id<MTLBuffer>)lhs_storage->buffer() offset:lhs_storage->offset() atIndex:1];
        [command_encoder setBuffer:(__bridge id<MTLBuffer>)rhs_storage->buffer() offset:rhs_storage->offset() atIndex:2];

        NSUInteger width = ShapeUtils::size(result.shape());
        NSUInteger max_threads = [pipeline_state_ maxTotalThreadsPerThreadgroup];
        MTLSize threads_per_group = MTLSizeMake(std::min(width, max_threads), 1, 1);
        MTLSize thread_groups = MTLSizeMake((width + threads_per_group.width - 1) / threads_per_group.width, 1, 1);

        [command_encoder dispatchThreadgroups:thread_groups threadsPerThreadgroup:threads_per_group];
        [command_encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        
        return result;
    }
};

void register_metal_operations() {
    if (!is_metal_available()) return;
    
    ops::OperationRegistry::register_operation(
        ops::OpType::Add,
        Device::GPU,
        std::make_unique<MetalAdd>());

    ops::OperationRegistry::register_operation(
        ops::OpType::Subtract,
        Device::GPU,
        std::make_unique<MetalSub>());
}

} // namespace metal
} // namespace backends
} // namespace axiom 