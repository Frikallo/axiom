#import "metal_operations.hpp"

#import "metal_common.hpp"
#import "metal_storage.hpp"
#import "axiom/operations.hpp"
#import "axiom/shape.hpp"
#import "axiom/tensor.hpp"
#import "axiom/dtype.hpp"

#import <Metal/Metal.h>
#import <vector>
#import <numeric>
#import <algorithm>

namespace axiom {
namespace backends {
namespace metal {

constexpr int kMaxDims = 8;

// Must match the order in kernels.metal
enum MetalOpType {
    Add,
    Subtract,
    Multiply,
    Divide
};

// Must match layout in kernels.metal
struct KernelParams {
    MetalOpType op_type;
    uint rank;
    uint lhs_shape[kMaxDims];
    uint lhs_strides[kMaxDims];
    uint rhs_shape[kMaxDims];
    uint rhs_strides[kMaxDims];
    uint result_shape[kMaxDims];
    uint result_strides[kMaxDims];
};

MetalOpType to_metal_op_type(ops::OpType op_type) {
    switch(op_type) {
        case ops::OpType::Add: return MetalOpType::Add;
        case ops::OpType::Subtract: return MetalOpType::Subtract;
        case ops::OpType::Multiply: return MetalOpType::Multiply;
        case ops::OpType::Divide: return MetalOpType::Divide;
        default: throw std::runtime_error("Unsupported binary operation for Metal.");
    }
}

// Unified class for all Metal binary operations
class MetalBinaryOperation : public ops::Operation {
private:
    ops::OpType op_type_;
    std::string op_name_;
    
    // Cache for pipeline states
    mutable std::map<DType, id<MTLComputePipelineState>> pipeline_states_;

public:
    MetalBinaryOperation(ops::OpType op_type, std::string op_name)
        : op_type_(op_type), op_name_(std::move(op_name)) {}

    ops::OpType type() const override { return op_type_; }
    std::string name() const override { return op_name_; }
    Device device() const override { return Device::GPU; }

    bool supports_binary(const Tensor& lhs, const Tensor& rhs) const override {
        // Now supports broadcasting
        try {
            ShapeUtils::broadcast_shape(lhs.shape(), rhs.shape());
            return true;
        } catch (const std::exception& e) {
            return false;
        }
    }

    id<MTLComputePipelineState> get_pipeline_state(DType dtype) const {
        if (pipeline_states_.count(dtype)) {
            return pipeline_states_.at(dtype);
        }

        std::string type_suffix;
        switch (dtype) {
            case DType::Float32: type_suffix = "float"; break;
            case DType::Float16: type_suffix = "half"; break;
            case DType::Int32:   type_suffix = "int"; break;
            case DType::UInt8:   type_suffix = "uint8_t"; break;
            case DType::Int8:    type_suffix = "int8_t"; break;
            default:
                throw std::runtime_error("Unsupported data type for Metal operation: " + dtype_name(dtype));
        }

        std::string kernel_name = "binary_kernel_" + type_suffix;

        id<MTLDevice> device = (__bridge id<MTLDevice>)MetalContext::instance().device();
        id<MTLLibrary> library = (__bridge id<MTLLibrary>)get_default_library();
        
        NSError* error = nil;
        id<MTLFunction> function = [library newFunctionWithName:[NSString stringWithUTF8String:kernel_name.c_str()]];
        if (!function) {
            throw std::runtime_error("Failed to find kernel: " + kernel_name);
        }

        id<MTLComputePipelineState> pipeline_state = [device newComputePipelineStateWithFunction:function error:&error];
        if (!pipeline_state) {
            throw std::runtime_error("Failed to create Metal pipeline state for kernel: " + kernel_name);
        }

        pipeline_states_[dtype] = pipeline_state;
        return pipeline_state;
    }

    // Helper to calculate strides for broadcasting
    std::vector<size_t> calculate_broadcast_strides(const Shape& original_shape, const Shape& result_shape, size_t itemsize) const {
        std::vector<size_t> broadcast_strides(result_shape.size(), 0);
        
        int rank_diff = result_shape.size() - original_shape.size();
        auto original_strides = ShapeUtils::get_contiguous_strides(original_shape, itemsize);
        
        for (int i = result_shape.size() - 1; i >= 0; --i) {
            if (i >= rank_diff && original_shape[i - rank_diff] == result_shape[i]) {
                broadcast_strides[i] = (i - rank_diff < original_shape.size()) ? original_strides[i - rank_diff] : 0;
            } else if (i >= rank_diff && original_shape[i - rank_diff] == 1) {
                broadcast_strides[i] = 0; // Broadcast this dimension
            } else if (i < rank_diff) {
                broadcast_strides[i] = 0; // Broadcast new leading dimensions
            } else {
                 // This case should be prevented by broadcast_shape check, but as a safeguard:
                 throw std::logic_error("Incompatible shapes for broadcasting strides calculation.");
            }
        }
        return broadcast_strides;
    }

    Tensor execute_binary(const Tensor& lhs, const Tensor& rhs) const override {
        // Type promotion should happen before this point.
        if (lhs.dtype() != rhs.dtype()) {
             throw std::runtime_error("Metal kernel inputs must have the same dtype.");
        }
        if (lhs.device() != Device::GPU || rhs.device() != Device::GPU) {
            throw std::runtime_error("Metal binary op inputs must be on GPU.");
        }
        
        Shape result_shape = ShapeUtils::broadcast_shape(lhs.shape(), rhs.shape());
        
        DType dtype = lhs.dtype();
        id<MTLComputePipelineState> pipeline_state = get_pipeline_state(dtype);

        Tensor result = Tensor(result_shape, dtype, Device::GPU);

        auto* lhs_storage = static_cast<const MetalStorage*>(lhs.storage().get());
        auto* rhs_storage = static_cast<const MetalStorage*>(rhs.storage().get());
        auto* res_storage = static_cast<MetalStorage*>(result.storage().get());
        
        id<MTLCommandQueue> command_queue = (__bridge id<MTLCommandQueue>)MetalContext::instance().command_queue();
        id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
        id<MTLComputeCommandEncoder> command_encoder = [command_buffer computeCommandEncoder];

        [command_encoder setComputePipelineState:pipeline_state];
        
        // Prepare parameters for the kernel
        KernelParams params;
        params.op_type = to_metal_op_type(op_type_);
        params.rank = result_shape.size();

        if (params.rank > kMaxDims) {
            throw std::runtime_error("Tensor rank exceeds Metal kernel's max dimensions.");
        }
        
        auto lhs_strides_vec = calculate_broadcast_strides(lhs.shape(), result_shape, lhs.itemsize());
        auto rhs_strides_vec = calculate_broadcast_strides(rhs.shape(), result_shape, rhs.itemsize());
        auto res_strides_vec = ShapeUtils::get_contiguous_strides(result_shape, result.itemsize());
        
        // Zero-initialize arrays
        std::fill_n(params.lhs_shape, kMaxDims, 0);
        std::fill_n(params.rhs_shape, kMaxDims, 0);
        std::fill_n(params.result_shape, kMaxDims, 0);
        std::fill_n(params.lhs_strides, kMaxDims, 0);
        std::fill_n(params.rhs_strides, kMaxDims, 0);
        std::fill_n(params.result_strides, kMaxDims, 0);
        
        // Copy data into fixed-size C-style arrays for the kernel
        int rank_diff_lhs = result_shape.size() - lhs.shape().size();
        for(size_t i = 0; i < lhs.shape().size(); ++i) params.lhs_shape[i + rank_diff_lhs] = lhs.shape()[i];
        
        int rank_diff_rhs = result_shape.size() - rhs.shape().size();
        for(size_t i = 0; i < rhs.shape().size(); ++i) params.rhs_shape[i + rank_diff_rhs] = rhs.shape()[i];

        for(size_t i = 0; i < result_shape.size(); ++i) params.result_shape[i] = result_shape[i];
        
        std::copy(lhs_strides_vec.begin(), lhs_strides_vec.end(), params.lhs_strides);
        std::copy(rhs_strides_vec.begin(), rhs_strides_vec.end(), params.rhs_strides);
        std::copy(res_strides_vec.begin(), res_strides_vec.end(), params.result_strides);


        [command_encoder setBuffer:(__bridge id<MTLBuffer>)lhs_storage->buffer() offset:lhs_storage->offset() atIndex:0];
        [command_encoder setBuffer:(__bridge id<MTLBuffer>)rhs_storage->buffer() offset:rhs_storage->offset() atIndex:1];
        [command_encoder setBuffer:(__bridge id<MTLBuffer>)res_storage->buffer() offset:res_storage->offset() atIndex:2];
        [command_encoder setBytes:&params length:sizeof(params) atIndex:3];

        NSUInteger width = result.size();
        NSUInteger max_threads = [pipeline_state maxTotalThreadsPerThreadgroup];
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

void register_metal_operations() {
    if (!is_metal_available()) return;
    
    ops::OperationRegistry::register_operation(
        ops::OpType::Add, Device::GPU,
        std::make_unique<MetalBinaryOperation>(ops::OpType::Add, "add"));

    ops::OperationRegistry::register_operation(
        ops::OpType::Subtract, Device::GPU,
        std::make_unique<MetalBinaryOperation>(ops::OpType::Subtract, "subtract"));
        
    ops::OperationRegistry::register_operation(
        ops::OpType::Multiply, Device::GPU,
        std::make_unique<MetalBinaryOperation>(ops::OpType::Multiply, "multiply"));

    ops::OperationRegistry::register_operation(
        ops::OpType::Divide, Device::GPU,
        std::make_unique<MetalBinaryOperation>(ops::OpType::Divide, "divide"));
}

} // namespace metal
} // namespace backends
} // namespace axiom 