#import "metal_operations.hpp"

#import "metal_common.hpp"
#import "metal_storage.hpp"
#import "axiom/operations.hpp"
#import "axiom/shape.hpp"
#import "axiom/tensor.hpp"

#import <Metal/Metal.h>

namespace axiom {
namespace backends {
namespace metal {

// Base class for Metal binary operations
class MetalBinaryOperation : public ops::Operation {
protected:
    id<MTLComputePipelineState> pipeline_state_;

public:
    explicit MetalBinaryOperation(const std::string& kernel_name) {
        id<MTLDevice> device = (__bridge id<MTLDevice>)MetalContext::instance().device();
        if (!device) {
            throw std::runtime_error("No Metal device found.");
        }

        id<MTLLibrary> library = (__bridge id<MTLLibrary>)get_default_library();
        if (!library) {
            throw std::runtime_error("Failed to get default Metal library.");
        }

        NSError* error = nil;
        id<MTLFunction> function = [library newFunctionWithName:[NSString stringWithUTF8String:kernel_name.c_str()]];
        if (!function) {
            throw std::runtime_error("Failed to find kernel: " + kernel_name);
        }

        pipeline_state_ = [device newComputePipelineStateWithFunction:function error:&error];
        if (!pipeline_state_) {
            throw std::runtime_error("Failed to create Metal pipeline state for kernel: " + kernel_name);
        }
    }

    Device device() const override { return Device::GPU; }

    Tensor execute_binary(const Tensor& lhs, const Tensor& rhs) const override {
        if (lhs.dtype() != DType::Float32 || rhs.dtype() != DType::Float32) {
            throw std::runtime_error("Metal kernel only supports float32.");
        }
        if (lhs.device() != Device::GPU || rhs.device() != Device::GPU) {
            throw std::runtime_error("Metal binary op inputs must be on GPU.");
        }
        
        Tensor result = Tensor(lhs.shape(), lhs.dtype(), Device::GPU);

        auto* lhs_storage = static_cast<const MetalStorage*>(lhs.storage().get());
        auto* rhs_storage = static_cast<const MetalStorage*>(rhs.storage().get());
        auto* res_storage = static_cast<MetalStorage*>(result.storage().get());
        
        id<MTLCommandQueue> command_queue = (__bridge id<MTLCommandQueue>)MetalContext::instance().command_queue();
        id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
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

struct MetalAdd : public MetalBinaryOperation {
    MetalAdd() : MetalBinaryOperation("add_kernel") {}
    ops::OpType type() const override { return ops::OpType::Add; }
    std::string name() const override { return "add"; }
};

struct MetalSub : public MetalBinaryOperation {
    MetalSub() : MetalBinaryOperation("sub_kernel") {}
    ops::OpType type() const override { return ops::OpType::Subtract; }
    std::string name() const override { return "subtract"; }
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