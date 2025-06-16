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
#import <string>

namespace axiom {
namespace backends {
namespace metal {

constexpr int kMaxDims = 8;

// Must match the order in kernels.metal
enum class MetalOpType {
    // Binary
    Add,
    Subtract,
    Multiply,
    Divide,
    // Unary
    Negate,
    Abs,
    Sqrt,
    Exp,
    Log,
    Sin,
    Cos,
    Tan,
    // Reductions
    Sum,
    Mean,
    Max,
    Min
};

// Must match layout in binary_kernels.metal
struct BinaryKernelParams {
    MetalOpType op_type;
    uint rank;
    uint lhs_shape[kMaxDims];
    uint lhs_strides[kMaxDims];
    uint rhs_shape[kMaxDims];
    uint rhs_strides[kMaxDims];
    uint result_shape[kMaxDims];
    uint result_strides[kMaxDims];
};

// Must match layout in unary_kernels.metal
struct UnaryKernelParams {
    MetalOpType op_type;
    uint rank;
    uint input_shape[kMaxDims];
    uint input_strides[kMaxDims];
};

// Must match layout in kernels.metal
struct ReductionKernelParams {
    MetalOpType op_type;
    uint rank;
    uint output_rank;
    uint input_shape[kMaxDims];
    uint input_strides[kMaxDims];
    uint output_shape[kMaxDims];
    uint output_strides[kMaxDims];
    uint reduction_axes[kMaxDims];
    uint num_reduction_axes;
    uint reduction_size;
};

MetalOpType to_metal_op_type(ops::OpType op_type) {
    switch(op_type) {
        // Binary
        case ops::OpType::Add: return MetalOpType::Add;
        case ops::OpType::Subtract: return MetalOpType::Subtract;
        case ops::OpType::Multiply: return MetalOpType::Multiply;
        case ops::OpType::Divide: return MetalOpType::Divide;
        // Unary
        case ops::OpType::Negate: return MetalOpType::Negate;
        case ops::OpType::Abs:    return MetalOpType::Abs;
        case ops::OpType::Sqrt:   return MetalOpType::Sqrt;
        case ops::OpType::Exp:    return MetalOpType::Exp;
        case ops::OpType::Log:    return MetalOpType::Log;
        case ops::OpType::Sin:    return MetalOpType::Sin;
        case ops::OpType::Cos:    return MetalOpType::Cos;
        case ops::OpType::Tan:    return MetalOpType::Tan;
        // Reductions
        case ops::OpType::Sum:    return MetalOpType::Sum;
        case ops::OpType::Mean:   return MetalOpType::Mean;
        case ops::OpType::Max:    return MetalOpType::Max;
        case ops::OpType::Min:    return MetalOpType::Min;
        default: throw std::runtime_error("Unsupported operation for Metal.");
    }
}

// Helper functions for logging
#ifdef DEBUG
template<typename T>
NSString* vectorToString(const std::vector<T>& vec) {
    NSMutableString* str = [NSMutableString stringWithString:@"["];
    for (size_t i = 0; i < vec.size(); ++i) {
        [str appendFormat:@"%llu%@", (unsigned long long)vec[i], (i < vec.size() - 1) ? @", " : @""];
    }
    [str appendString:@"]"];
    return str;
}
#endif

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
        BinaryKernelParams params;
        params.op_type = to_metal_op_type(op_type_);
        params.rank = result_shape.size();

        if (params.rank > kMaxDims) {
            throw std::runtime_error("Tensor rank exceeds Metal kernel's max dimensions.");
        }
        
        // The result tensor is always contiguous
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
        for(size_t i = 0; i < lhs.shape().size(); ++i) {
            params.lhs_shape[i + rank_diff_lhs] = lhs.shape()[i];
            // Convert byte stride to element stride for the kernel
            if (lhs.shape()[i] != 1) {
                 params.lhs_strides[i + rank_diff_lhs] = lhs.strides()[i] / lhs.itemsize();
            } else {
                 params.lhs_strides[i + rank_diff_lhs] = 0;
            }
        }
        
        int rank_diff_rhs = result_shape.size() - rhs.shape().size();
        for(size_t i = 0; i < rhs.shape().size(); ++i) {
            params.rhs_shape[i + rank_diff_rhs] = rhs.shape()[i];
            // Convert byte stride to element stride for the kernel
            if (rhs.shape()[i] != 1) {
                params.rhs_strides[i + rank_diff_rhs] = rhs.strides()[i] / rhs.itemsize();
            } else {
                params.rhs_strides[i + rank_diff_rhs] = 0;
            }
        }

        for(size_t i = 0; i < result_shape.size(); ++i) {
            params.result_shape[i] = result_shape[i];
            params.result_strides[i] = res_strides_vec[i] / result.itemsize();
        }

        [command_encoder setBuffer:(__bridge id<MTLBuffer>)lhs_storage->buffer() offset:lhs.offset() atIndex:0];
        [command_encoder setBuffer:(__bridge id<MTLBuffer>)rhs_storage->buffer() offset:rhs.offset() atIndex:1];
        [command_encoder setBuffer:(__bridge id<MTLBuffer>)res_storage->buffer() offset:result.offset() atIndex:2];
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

    Tensor execute_unary(const Tensor& input) const override {
        (void)input;
        throw std::runtime_error("Not a unary operation");
    }

    Tensor execute_reduction(const Tensor& input, const std::vector<int>& axis, bool keep_dims) const override {
        (void)input; (void)axis; (void)keep_dims;
        throw std::runtime_error("Not a reduction operation");
    }

    void execute_binary_inplace(Tensor& lhs, const Tensor& rhs) const override {
        (void)lhs; (void)rhs;
        throw std::runtime_error("In-place not supported for Metal binary operations yet.");
    }
};

class MetalUnaryOperation : public ops::Operation {
private:
    ops::OpType op_type_;
    std::string op_name_;
    mutable std::map<DType, id<MTLComputePipelineState>> pipeline_states_;

public:
    MetalUnaryOperation(ops::OpType op_type, std::string op_name)
        : op_type_(op_type), op_name_(std::move(op_name)) {}

    ops::OpType type() const override { return op_type_; }
    std::string name() const override { return op_name_; }
    Device device() const override { return Device::GPU; }

    id<MTLComputePipelineState> get_pipeline_state(DType dtype) const {
        if (pipeline_states_.count(dtype)) {
            return pipeline_states_.at(dtype);
        }

        std::string type_suffix;
        switch (dtype) {
            case DType::Float32: type_suffix = "float"; break;
            case DType::Float16: type_suffix = "half"; break;
            default:
                throw std::runtime_error("Unsupported data type for Metal unary operation: " + dtype_name(dtype));
        }

        std::string kernel_name = "unary_kernel_" + type_suffix;

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

    Tensor execute_binary(const Tensor& lhs, const Tensor& rhs) const override {
        (void)lhs; (void)rhs;
        throw std::runtime_error("Not a binary operation");
    }

    Tensor execute_unary(const Tensor& input) const override {
        if (input.device() != Device::GPU) {
            throw std::runtime_error("Metal unary op input must be on GPU.");
        }
        
        DType dtype = input.dtype();
        id<MTLComputePipelineState> pipeline_state = get_pipeline_state(dtype);

        Tensor result = Tensor(input.shape(), dtype, Device::GPU);

        auto* input_storage = static_cast<const MetalStorage*>(input.storage().get());
        auto* res_storage = static_cast<MetalStorage*>(result.storage().get());
        
        id<MTLCommandQueue> command_queue = (__bridge id<MTLCommandQueue>)MetalContext::instance().command_queue();
        id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
        id<MTLComputeCommandEncoder> command_encoder = [command_buffer computeCommandEncoder];

        [command_encoder setComputePipelineState:pipeline_state];
        
        UnaryKernelParams params;
        params.op_type = to_metal_op_type(op_type_);
        params.rank = input.ndim();

        if (params.rank > kMaxDims) {
            throw std::runtime_error("Tensor rank exceeds Metal kernel's max dimensions.");
        }
        
        std::fill_n(params.input_shape, kMaxDims, 0);
        std::fill_n(params.input_strides, kMaxDims, 0);
        
        for(size_t i = 0; i < input.ndim(); ++i) {
            params.input_shape[i] = input.shape()[i];
            params.input_strides[i] = input.strides()[i] / input.itemsize();
        }

        [command_encoder setBuffer:(__bridge id<MTLBuffer>)input_storage->buffer() offset:input.offset() atIndex:0];
        [command_encoder setBuffer:(__bridge id<MTLBuffer>)res_storage->buffer() offset:result.offset() atIndex:1];
        [command_encoder setBytes:&params length:sizeof(params) atIndex:2];

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

    Tensor execute_reduction(const Tensor& input, const std::vector<int>& axis, bool keep_dims) const override {
        (void)input; (void)axis; (void)keep_dims;
        throw std::runtime_error("Not a reduction operation");
    }

    void execute_binary_inplace(Tensor& lhs, const Tensor& rhs) const override {
        (void)lhs; (void)rhs;
        throw std::runtime_error("Not a binary operation");
    }
};

// ============================================================================
// Metal Reduction Operation
// ============================================================================

class MetalReductionOperation : public ops::Operation {
private:
    ops::OpType op_type_;
    std::string op_name_;
    
    // Cache for pipeline states
    mutable std::map<DType, id<MTLComputePipelineState>> pipeline_states_;

public:
    MetalReductionOperation(ops::OpType op_type, std::string op_name)
        : op_type_(op_type), op_name_(std::move(op_name)) {}

    ops::OpType type() const override { return op_type_; }
    std::string name() const override { return op_name_; }
    Device device() const override { return Device::GPU; }

    Tensor execute_binary(const Tensor& lhs, const Tensor& rhs) const override {
        (void)lhs; (void)rhs;
        throw std::runtime_error("Not a binary operation");
    }

    Tensor execute_unary(const Tensor& input) const override {
        (void)input;
        throw std::runtime_error("Not a unary operation");
    }

    id<MTLComputePipelineState> get_pipeline_state(DType dtype) const {
        if (pipeline_states_.count(dtype)) {
            return pipeline_states_.at(dtype);
        }

        std::string type_suffix;
        switch (dtype) {
            case DType::Float32: type_suffix = "float"; break;
            case DType::Float16: type_suffix = "half"; break;
            // TODO: Add support for other types like int
            default:
                throw std::runtime_error("Unsupported data type for Metal reduction: " + dtype_name(dtype));
        }

        std::string kernel_name = "reduction_kernel_" + type_suffix;

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

    Tensor execute_reduction(const Tensor& input, const std::vector<int>& raw_axes, bool keep_dims) const override {
        if (input.device() != Device::GPU) {
            throw std::runtime_error("Metal reduction op input must be on GPU.");
        }

        // --- Prepare shapes and axes ---
        auto axes = raw_axes;
        if (axes.empty()) { // Reduce over all axes
            axes.resize(input.shape().size());
            std::iota(axes.begin(), axes.end(), 0);
        }

        Shape output_shape;
        size_t reduction_size = 1;
        std::vector<bool> is_reduction_axis(input.shape().size(), false);
        for (int axis : axes) {
            is_reduction_axis[axis] = true;
            reduction_size *= input.shape()[axis];
        }

        for (size_t i = 0; i < input.shape().size(); ++i) {
            if (!is_reduction_axis[i]) {
                output_shape.push_back(input.shape()[i]);
            } else if (keep_dims) {
                output_shape.push_back(1);
            }
        }
        if (output_shape.empty()) output_shape.push_back(1);
        
        // --- Get pipeline and create result tensor ---
        DType dtype = input.dtype();
        id<MTLComputePipelineState> pipeline_state = get_pipeline_state(dtype);
        Tensor result = Tensor(output_shape, dtype, Device::GPU);

        // --- Get command encoder ---
        id<MTLCommandQueue> command_queue = (__bridge id<MTLCommandQueue>)MetalContext::instance().command_queue();
        id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
        id<MTLComputeCommandEncoder> command_encoder = [command_buffer computeCommandEncoder];

        [command_encoder setComputePipelineState:pipeline_state];

        // --- Prepare kernel parameters ---
        ReductionKernelParams params;
        params.op_type = to_metal_op_type(op_type_);
        params.rank = input.shape().size();
        params.output_rank = result.shape().size();
        if (params.rank > kMaxDims) {
            throw std::runtime_error("Tensor rank exceeds Metal kernel's max dimensions.");
        }

        std::fill_n(params.input_shape, kMaxDims, 0);
        std::fill_n(params.input_strides, kMaxDims, 0);
        std::fill_n(params.output_shape, kMaxDims, 0);
        std::fill_n(params.output_strides, kMaxDims, 0);
        std::fill_n(params.reduction_axes, kMaxDims, 0);

        for(size_t i = 0; i < input.shape().size(); ++i) {
            params.input_shape[i] = input.shape()[i];
            params.input_strides[i] = input.strides()[i] / input.itemsize();
        }

        auto result_strides_vec = ShapeUtils::get_contiguous_strides(result.shape(), result.itemsize());
        for(size_t i = 0; i < result.shape().size(); ++i) {
            params.output_shape[i] = result.shape()[i];
            params.output_strides[i] = result_strides_vec[i] / result.itemsize();
        }

        params.num_reduction_axes = axes.size();
        for(size_t i = 0; i < axes.size(); ++i) {
            params.reduction_axes[i] = axes[i];
        }
        params.reduction_size = reduction_size;

        auto* input_storage = static_cast<const MetalStorage*>(input.storage().get());
        auto* res_storage = static_cast<MetalStorage*>(result.storage().get());

        [command_encoder setBuffer:(__bridge id<MTLBuffer>)input_storage->buffer() offset:input.offset() atIndex:0];
        [command_encoder setBuffer:(__bridge id<MTLBuffer>)res_storage->buffer() offset:result.offset() atIndex:1];
        [command_encoder setBytes:&params length:sizeof(params) atIndex:2];

        // --- Dispatch threads ---
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
    
    // Binary Ops
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

    // Unary Ops
    ops::OperationRegistry::register_operation(
        ops::OpType::Negate, Device::GPU,
        std::make_unique<MetalUnaryOperation>(ops::OpType::Negate, "negate"));
    ops::OperationRegistry::register_operation(
        ops::OpType::Abs, Device::GPU,
        std::make_unique<MetalUnaryOperation>(ops::OpType::Abs, "abs"));
    ops::OperationRegistry::register_operation(
        ops::OpType::Sqrt, Device::GPU,
        std::make_unique<MetalUnaryOperation>(ops::OpType::Sqrt, "sqrt"));
    ops::OperationRegistry::register_operation(
        ops::OpType::Exp, Device::GPU,
        std::make_unique<MetalUnaryOperation>(ops::OpType::Exp, "exp"));
    ops::OperationRegistry::register_operation(
        ops::OpType::Log, Device::GPU,
        std::make_unique<MetalUnaryOperation>(ops::OpType::Log, "log"));
    ops::OperationRegistry::register_operation(
        ops::OpType::Sin, Device::GPU,
        std::make_unique<MetalUnaryOperation>(ops::OpType::Sin, "sin"));
    ops::OperationRegistry::register_operation(
        ops::OpType::Cos, Device::GPU,
        std::make_unique<MetalUnaryOperation>(ops::OpType::Cos, "cos"));
    ops::OperationRegistry::register_operation(
        ops::OpType::Tan, Device::GPU,
        std::make_unique<MetalUnaryOperation>(ops::OpType::Tan, "tan"));

    // Reduction Ops
    ops::OperationRegistry::register_operation(ops::OpType::Sum, Device::GPU, std::make_unique<MetalReductionOperation>(ops::OpType::Sum, "sum"));
    ops::OperationRegistry::register_operation(ops::OpType::Mean, Device::GPU, std::make_unique<MetalReductionOperation>(ops::OpType::Mean, "mean"));
    ops::OperationRegistry::register_operation(ops::OpType::Max, Device::GPU, std::make_unique<MetalReductionOperation>(ops::OpType::Max, "max"));
    ops::OperationRegistry::register_operation(ops::OpType::Min, Device::GPU, std::make_unique<MetalReductionOperation>(ops::OpType::Min, "min"));
}

} // namespace metal
} // namespace backends
} // namespace axiom 