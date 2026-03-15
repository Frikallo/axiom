#import "axiom/nn/ane_compiled_model.hpp"

#ifdef AXIOM_HAS_ANE

#import <Accelerate/Accelerate.h>
#import <IOSurface/IOSurface.h>

#include "axiom/error.hpp"
#include "axiom/nn/activation.hpp"
#include "axiom/nn/attention.hpp"
#include "axiom/nn/container.hpp"
#include "axiom/nn/conv.hpp"
#include "axiom/nn/linear.hpp"
#include "axiom/nn/normalization.hpp"
#include "axiom/nn/pooling.hpp"
#include "backends/ane/ane_bridge.h"
#include "backends/ane/ane_iosurface.h"
#include "backends/ane/mil_generator.hpp"

namespace axiom {
namespace backends {
namespace ane {

// ============================================================================
// Implementation detail
// ============================================================================

struct ANECompiledModel::Impl {
    ANEModelHandle *handle = nullptr;
    Shape input_shape;                 // Axiom layout (e.g., [batch, features])
    Shape output_shape;                // Axiom layout
    std::vector<int64_t> ane_in_shape; // ANE [1, C, 1, S] layout
    std::vector<int64_t> ane_out_shape;
    size_t input_elements = 0;
    size_t output_elements = 0;
    size_t surface_bytes = 0; // Page-aligned buffer size
    std::string module_type;
    int compile_count = 0;
    size_t weight_bytes = 0;

    ~Impl() {
        if (handle) {
            ane_release(handle);
            handle = nullptr;
        }
    }
};

// ============================================================================
// Layout conversion helpers
// ============================================================================

// Convert Axiom shape to ANE [1, C, 1, S] shape.
// [features] → [1, features, 1, 1]
// [batch, features] → [1, features, 1, batch]
// [batch, seq, features] → [1, features, 1, batch*seq]
static std::vector<int64_t> to_ane_shape(const Shape &shape) {
    if (shape.size() == 1) {
        return {1, static_cast<int64_t>(shape[0]), 1, 1};
    }
    if (shape.size() == 2) {
        return {1, static_cast<int64_t>(shape[1]), 1,
                static_cast<int64_t>(shape[0])};
    }
    if (shape.size() == 3) {
        int64_t spatial = static_cast<int64_t>(shape[0] * shape[1]);
        return {1, static_cast<int64_t>(shape[2]), 1, spatial};
    }
    // 4D: assume already in NCHW-ish format
    return {static_cast<int64_t>(shape[0]), static_cast<int64_t>(shape[1]),
            static_cast<int64_t>(shape[2]), static_cast<int64_t>(shape[3])};
}

// Compute output shape in ANE layout after processing through a module.
static std::vector<int64_t>
compute_ane_output_shape(const std::vector<int64_t> &ane_in,
                          const nn::Module &module) {
    if (auto *linear = dynamic_cast<const nn::Linear *>(&module)) {
        int64_t out_f = static_cast<int64_t>(linear->weight().shape()[0]);
        return {ane_in[0], out_f, ane_in[2], ane_in[3]};
    }
    // Shape-preserving modules (activations, norms, dropout)
    return ane_in;
}

// Convert Axiom output ANE shape back to Axiom shape.
static Shape to_axiom_shape(const std::vector<int64_t> &ane_shape,
                             const Shape &original_input_shape) {
    if (original_input_shape.size() == 1) {
        return {static_cast<size_t>(ane_shape[1])};
    }
    if (original_input_shape.size() == 2) {
        return {static_cast<size_t>(ane_shape[3]),
                static_cast<size_t>(ane_shape[1])};
    }
    if (original_input_shape.size() == 3) {
        return {original_input_shape[0], original_input_shape[1],
                static_cast<size_t>(ane_shape[1])};
    }
    Shape result;
    for (auto d : ane_shape)
        result.push_back(static_cast<size_t>(d));
    return result;
}

// Write FP32 tensor data into a flat IOSurface as FP16 in channel-first order.
// Axiom tensor: row-major [batch, ..., features]
// ANE layout:   [1, features, 1, spatial] contiguous FP16
static void write_tensor_to_surface(IOSurfaceRef surface, const Tensor &t,
                                     const std::vector<int64_t> &ane_shape) {
    Tensor cpu = t.cpu().ascontiguousarray();
    if (cpu.dtype() != DType::Float32) {
        cpu = cpu.astype(DType::Float32);
    }

    size_t num_elements = cpu.size();
    const float *src = cpu.typed_data<float>();

    int channels = static_cast<int>(ane_shape[1]);
    int spatial = static_cast<int>(ane_shape[3]);

    // Transpose from row-major [spatial, channels] to channel-first
    // [channels, spatial].
    // When spatial==1, data is already contiguous per-channel (no transpose).
    std::vector<float> transposed(num_elements);
    if (spatial == 1) {
        // [1, C, 1, 1]: channels are contiguous, just copy
        std::memcpy(transposed.data(), src, num_elements * sizeof(float));
    } else {
        for (int s = 0; s < spatial; s++) {
            for (int c = 0; c < channels; c++) {
                transposed[c * spatial + s] = src[s * channels + c];
            }
        }
    }

    // Convert FP32 → FP16
    std::vector<uint16_t> fp16(num_elements);
    vImage_Buffer src_buf = {
        .data = transposed.data(),
        .height = 1,
        .width = static_cast<vImagePixelCount>(num_elements),
        .rowBytes = num_elements * sizeof(float),
    };
    vImage_Buffer dst_buf = {
        .data = fp16.data(),
        .height = 1,
        .width = static_cast<vImagePixelCount>(num_elements),
        .rowBytes = num_elements * sizeof(uint16_t),
    };
    vImageConvert_PlanarFtoPlanar16F(&src_buf, &dst_buf, 0);

    // Write to IOSurface respecting row alignment
    IOSurfaceLock(surface, 0, NULL);
    auto *base = static_cast<uint8_t *>(IOSurfaceGetBaseAddress(surface));
    size_t alloc = IOSurfaceGetAllocSize(surface);
    std::memset(base, 0, alloc); // Clear
    size_t row_bytes = IOSurfaceGetBytesPerRow(surface);
    size_t data_per_row = static_cast<size_t>(spatial) * sizeof(uint16_t);
    for (int c = 0; c < channels; c++) {
        std::memcpy(base + c * row_bytes,
                     fp16.data() + c * spatial,
                     data_per_row);
    }
    IOSurfaceUnlock(surface, 0, NULL);
}

// Read FP16 channel-first data from IOSurface back to Axiom tensor.
static Tensor read_tensor_from_surface(IOSurfaceRef surface,
                                        const std::vector<int64_t> &ane_shape,
                                        const Shape &axiom_shape) {
    int channels = static_cast<int>(ane_shape[1]);
    int spatial = static_cast<int>(ane_shape[3]);
    size_t num_elements = static_cast<size_t>(channels * spatial);

    // Read FP16 from IOSurface respecting row alignment
    IOSurfaceLock(surface, kIOSurfaceLockReadOnly, NULL);
    auto *base =
        static_cast<const uint8_t *>(IOSurfaceGetBaseAddress(surface));
    size_t row_bytes = IOSurfaceGetBytesPerRow(surface);
    size_t data_per_row = static_cast<size_t>(spatial) * sizeof(uint16_t);

    std::vector<uint16_t> fp16(num_elements);
    for (int c = 0; c < channels; c++) {
        std::memcpy(fp16.data() + c * spatial,
                     base + c * row_bytes,
                     data_per_row);
    }
    IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, NULL);

    // Convert FP16 → FP32
    std::vector<float> f32(num_elements);
    vImage_Buffer src_buf = {
        .data = fp16.data(),
        .height = 1,
        .width = static_cast<vImagePixelCount>(num_elements),
        .rowBytes = num_elements * sizeof(uint16_t),
    };
    vImage_Buffer dst_buf = {
        .data = f32.data(),
        .height = 1,
        .width = static_cast<vImagePixelCount>(num_elements),
        .rowBytes = num_elements * sizeof(float),
    };
    vImageConvert_Planar16FtoPlanarF(&src_buf, &dst_buf, 0);

    // Transpose from channel-first [channels, spatial] to row-major
    // [spatial, channels].
    // When spatial==1, data is already in the right order (no transpose).
    std::vector<float> transposed(num_elements);
    if (spatial == 1) {
        std::memcpy(transposed.data(), f32.data(),
                     num_elements * sizeof(float));
    } else {
        for (int s = 0; s < spatial; s++) {
            for (int c = 0; c < channels; c++) {
                transposed[s * channels + c] = f32[c * spatial + s];
            }
        }
    }

    // Create Axiom tensor
    auto result = Tensor(axiom_shape, DType::Float32);
    std::memcpy(result.typed_data<float>(), transposed.data(),
                num_elements * sizeof(float));
    return result;
}

// ============================================================================
// Module → MIL compilation
// ============================================================================

// Walk a module and emit MIL operations. Returns the output variable name.
static std::string walk_module(MILGenerator &gen, const nn::Module &module,
                                const std::string &input_var,
                                const std::string &prefix) {
    // Linear
    if (auto *linear = dynamic_cast<const nn::Linear *>(&module)) {
        const Tensor *bias =
            linear->has_bias() ? &linear->bias() : nullptr;
        return gen.add_linear(input_var, linear->weight(), bias, prefix);
    }

    // Activations (stateless)
    if (dynamic_cast<const nn::ReLU *>(&module)) {
        return gen.add_relu(input_var, prefix);
    }
    if (dynamic_cast<const nn::SiLU *>(&module)) {
        return gen.add_silu(input_var, prefix);
    }
    if (dynamic_cast<const nn::GELU *>(&module)) {
        return gen.add_gelu(input_var, prefix);
    }
    if (dynamic_cast<const nn::Sigmoid *>(&module)) {
        return gen.add_sigmoid(input_var, prefix);
    }

    // Normalization
    if (auto *ln = dynamic_cast<const nn::LayerNorm *>(&module)) {
        return gen.add_layer_norm(input_var, ln->weight(), ln->bias(),
                                  ln->eps(), prefix);
    }
    if (auto *rn = dynamic_cast<const nn::RMSNorm *>(&module)) {
        return gen.add_rms_norm(input_var, rn->weight(), rn->eps(), prefix);
    }

    // Dropout (identity in inference)
    if (dynamic_cast<const nn::Dropout *>(&module)) {
        return input_var; // No-op
    }

    // Sequential: chain all children
    if (auto *seq = dynamic_cast<const nn::Sequential *>(&module)) {
        std::string var = input_var;
        int idx = 0;
        // Walk via named_parameters isn't enough — we need to walk
        // submodules in order. Use the forward pattern: each child
        // processes the output of the previous.
        // Since Sequential doesn't expose iteration, we compile it
        // via its forward() on CPU instead.
        // TODO: Add submodule iteration to Sequential for ANE compilation
        throw RuntimeError("Sequential ANE compilation requires submodule "
                           "iteration (not yet implemented)");
    }

    throw RuntimeError("Unsupported module type for ANE compilation: " +
                       std::string(typeid(module).name()));
}

// Track the output shape through a module chain.
static std::vector<int64_t>
infer_output_shape(const nn::Module &module,
                   const std::vector<int64_t> &ane_in) {
    if (auto *linear = dynamic_cast<const nn::Linear *>(&module)) {
        int64_t out_f = static_cast<int64_t>(linear->weight().shape()[0]);
        return {ane_in[0], out_f, ane_in[2], ane_in[3]};
    }
    // Shape-preserving
    return ane_in;
}

// ============================================================================
// Public API
// ============================================================================

ANECompiledModel::ANECompiledModel() : impl_(std::make_unique<Impl>()) {}
ANECompiledModel::~ANECompiledModel() = default;
ANECompiledModel::ANECompiledModel(ANECompiledModel &&) noexcept = default;
ANECompiledModel &
ANECompiledModel::operator=(ANECompiledModel &&) noexcept = default;

bool ANECompiledModel::is_supported(const nn::Module &module) {
    if (dynamic_cast<const nn::Linear *>(&module))
        return true;
    if (dynamic_cast<const nn::ReLU *>(&module))
        return true;
    if (dynamic_cast<const nn::SiLU *>(&module))
        return true;
    if (dynamic_cast<const nn::GELU *>(&module))
        return true;
    if (dynamic_cast<const nn::Sigmoid *>(&module))
        return true;
    if (dynamic_cast<const nn::LayerNorm *>(&module))
        return true;
    if (dynamic_cast<const nn::RMSNorm *>(&module))
        return true;
    if (dynamic_cast<const nn::Dropout *>(&module))
        return true;
    return false;
}

ANECompiledModel ANECompiledModel::compile(const nn::Module &module,
                                            const Shape &input_shape) {
    if (!ane_is_available()) {
        throw DeviceError::not_available("ANE");
    }

    ANECompiledModel result;
    auto &impl = *result.impl_;

    impl.input_shape = input_shape;
    impl.ane_in_shape = to_ane_shape(input_shape);

    // Compute element counts
    impl.input_elements = 1;
    for (auto d : input_shape)
        impl.input_elements *= d;

    // Generate MIL
    MILGenerator gen;
    gen.begin_program();
    auto x = gen.add_input("x", impl.ane_in_shape);

    std::string out_var;
    try {
        out_var = walk_module(gen, module, x, "m");
    } catch (const std::exception &e) {
        throw RuntimeError("ANE compilation failed: " + std::string(e.what()));
    }

    gen.set_output(out_var);
    std::string mil = gen.finalize();

    // Compute output shape
    impl.ane_out_shape = gen.shape_of(out_var);
    impl.output_shape = to_axiom_shape(impl.ane_out_shape, input_shape);

    impl.output_elements = 1;
    for (auto d : impl.output_shape)
        impl.output_elements *= d;

    // Prepare weight entries
    auto &blobs = gen.weight_blobs();
    std::vector<ANEWeightEntry> entries;
    entries.reserve(blobs.size());
    for (auto &b : blobs) {
        entries.push_back(
            {b.name.c_str(), b.blob_data.data(), b.blob_data.size()});
        impl.weight_bytes += b.blob_data.size();
    }

    // Compile on ANE
    impl.handle = ane_compile_with_weights(
        mil.c_str(), entries.data(), static_cast<int>(entries.size()));
    if (!impl.handle) {
        throw RuntimeError("ANE MIL compilation failed");
    }
    impl.compile_count = 1;

    // Load onto hardware
    int rc = ane_load(impl.handle);
    if (rc != 0) {
        ane_release(impl.handle);
        impl.handle = nullptr;
        throw RuntimeError("ANE model loading failed");
    }

    // Compute surface buffer size.
    // ANE surfaces need to be large enough for the tensor data.
    // Use page-aligned size but ensure minimum for small tensors.
    size_t in_bytes = impl.input_elements * sizeof(uint16_t);
    size_t out_bytes = impl.output_elements * sizeof(uint16_t);
    impl.surface_bytes =
        std::max({in_bytes, out_bytes, size_t(16384)});
    // Round up to page boundary
    impl.surface_bytes = (impl.surface_bytes + 16383) & ~size_t(16383);

    impl.module_type = typeid(module).name();

    return result;
}

Tensor ANECompiledModel::forward(const Tensor &input) const {
    if (!impl_ || !impl_->handle) {
        throw RuntimeError("ANECompiledModel not compiled");
    }

    // Validate input shape
    if (input.shape() != impl_->input_shape) {
        std::ostringstream oss;
        oss << "ANE input shape mismatch: expected [";
        for (size_t i = 0; i < impl_->input_shape.size(); i++) {
            if (i > 0)
                oss << ", ";
            oss << impl_->input_shape[i];
        }
        oss << "] but got [";
        for (size_t i = 0; i < input.shape().size(); i++) {
            if (i > 0)
                oss << ", ";
            oss << input.shape()[i];
        }
        oss << "]";
        throw ShapeError(oss.str());
    }

    // Create IOSurfaces using the 2D format matching ANE tensor layout.
    // channels = ane_shape[1], spatial = ane_shape[3].
    int in_c = static_cast<int>(impl_->ane_in_shape[1]);
    int in_s = static_cast<int>(impl_->ane_in_shape[3]);
    int out_c = static_cast<int>(impl_->ane_out_shape[1]);
    int out_s = static_cast<int>(impl_->ane_out_shape[3]);
    IOSurfaceRef in_surface = ane_create_surface(in_c, in_s);
    IOSurfaceRef out_surface = ane_create_surface(out_c, out_s);
    if (!in_surface || !out_surface) {
        if (in_surface)
            CFRelease(in_surface);
        if (out_surface)
            CFRelease(out_surface);
        throw MemoryError::allocation_failed(impl_->surface_bytes);
    }

    // Write input tensor to IOSurface
    write_tensor_to_surface(in_surface, input, impl_->ane_in_shape);

    // Evaluate on ANE
    int rc = ane_eval(impl_->handle, in_surface, out_surface);
    if (rc != 0) {
        CFRelease(in_surface);
        CFRelease(out_surface);
        throw RuntimeError("ANE evaluation failed");
    }

    // Read output tensor from IOSurface
    Tensor result = read_tensor_from_surface(out_surface, impl_->ane_out_shape,
                                              impl_->output_shape);

    CFRelease(in_surface);
    CFRelease(out_surface);

    return result;
}

ANEPlanInfo ANECompiledModel::plan_info() const {
    ANEPlanInfo info;
    info.ane_steps = 1;
    info.cpu_steps = 0;
    info.compile_count = impl_ ? impl_->compile_count : 0;
    info.weight_bytes = impl_ ? impl_->weight_bytes : 0;

    std::ostringstream oss;
    oss << "ANE compiled model";
    if (impl_) {
        oss << " (input: [";
        for (size_t i = 0; i < impl_->input_shape.size(); i++) {
            if (i > 0)
                oss << ", ";
            oss << impl_->input_shape[i];
        }
        oss << "] → output: [";
        for (size_t i = 0; i < impl_->output_shape.size(); i++) {
            if (i > 0)
                oss << ", ";
            oss << impl_->output_shape[i];
        }
        oss << "])";
    }
    info.summary = oss.str();
    return info;
}

} // namespace ane
} // namespace backends
} // namespace axiom

#endif // AXIOM_HAS_ANE
