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

#include <cmath>

namespace axiom {
namespace backends {
namespace ane {

// ============================================================================
// Tiling constants
// ============================================================================

// ANE SRAM is ~32MB; stay under 24MB to avoid DRAM spill (30% perf cliff)
static constexpr size_t ANE_SRAM_BUDGET = 24 * 1024 * 1024;

// ============================================================================
// Implementation detail
// ============================================================================

struct ANECompiledModel::Impl {
    ANEModelHandle *handle = nullptr;
    Shape input_shape;
    Shape output_shape;
    std::vector<int64_t> ane_in_shape;
    std::vector<int64_t> ane_out_shape;
    size_t input_elements = 0;
    size_t output_elements = 0;
    std::string module_type;
    int compile_count = 0;
    size_t weight_bytes = 0;

    // Pre-allocated IOSurfaces (reused across forward() calls)
    IOSurfaceRef in_surface = nullptr;
    IOSurfaceRef out_surface = nullptr;

    // Tiling state
    int tile_count = 1;          // Number of tiles (1 = no tiling)
    int tile_spatial = 0;        // Spatial size per tile
    size_t working_set_bytes = 0;

    // Pre-allocated FP16 buffers to avoid repeated allocation
    mutable std::vector<uint16_t> fp16_in_buf;
    mutable std::vector<uint16_t> fp16_out_buf;
    mutable std::vector<float> transpose_buf;

    ~Impl() {
        if (handle) {
            ane_release(handle);
            handle = nullptr;
        }
        if (in_surface) {
            CFRelease(in_surface);
            in_surface = nullptr;
        }
        if (out_surface) {
            CFRelease(out_surface);
            out_surface = nullptr;
        }
    }
};

// ============================================================================
// Layout conversion helpers (vDSP-accelerated)
// ============================================================================

static int compute_spatial(const std::vector<int64_t> &shape) {
    int sp = 1;
    for (size_t d = 0; d < shape.size(); d++) {
        if (d != 1)
            sp *= static_cast<int>(shape[d]);
    }
    return sp;
}

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
    return {static_cast<int64_t>(shape[0]), static_cast<int64_t>(shape[1]),
            static_cast<int64_t>(shape[2]), static_cast<int64_t>(shape[3])};
}

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

// Write FP32 tensor → IOSurface (FP16 channel-first) using vDSP
static void write_tensor_to_surface(IOSurfaceRef surface, const Tensor &t,
                                     const std::vector<int64_t> &ane_shape,
                                     std::vector<float> &transpose_buf,
                                     std::vector<uint16_t> &fp16_buf) {
    Tensor cpu = t.cpu().ascontiguousarray();
    if (cpu.dtype() != DType::Float32) {
        cpu = cpu.astype(DType::Float32);
    }

    size_t num_elements = cpu.size();
    const float *src = cpu.typed_data<float>();

    int channels = static_cast<int>(ane_shape[1]);
    int spatial = compute_spatial(ane_shape);

    // Transpose [S, C] row-major → [C, S] channel-first using vDSP
    transpose_buf.resize(num_elements);
    bool needs_transpose =
        (spatial > 1) && !(ane_shape.size() == 4 && ane_shape[2] > 1);

    if (needs_transpose) {
        // vDSP_mtrans: transpose MxN matrix (S rows, C cols) → (C rows, S cols)
        vDSP_mtrans(src, 1, transpose_buf.data(), 1,
                     static_cast<vDSP_Length>(channels),
                     static_cast<vDSP_Length>(spatial));
    } else {
        std::memcpy(transpose_buf.data(), src, num_elements * sizeof(float));
    }

    // FP32 → FP16 using vImage
    fp16_buf.resize(num_elements);
    vImage_Buffer src_vimg = {transpose_buf.data(), 1,
                               static_cast<vImagePixelCount>(num_elements),
                               num_elements * sizeof(float)};
    vImage_Buffer dst_vimg = {fp16_buf.data(), 1,
                               static_cast<vImagePixelCount>(num_elements),
                               num_elements * sizeof(uint16_t)};
    vImageConvert_PlanarFtoPlanar16F(&src_vimg, &dst_vimg, 0);

    // Write to IOSurface with row alignment
    IOSurfaceLock(surface, 0, NULL);
    auto *base = static_cast<uint8_t *>(IOSurfaceGetBaseAddress(surface));
    std::memset(base, 0, IOSurfaceGetAllocSize(surface));
    size_t row_bytes = IOSurfaceGetBytesPerRow(surface);
    size_t data_per_row = static_cast<size_t>(spatial) * sizeof(uint16_t);
    for (int c = 0; c < channels; c++) {
        std::memcpy(base + c * row_bytes, fp16_buf.data() + c * spatial,
                     data_per_row);
    }
    IOSurfaceUnlock(surface, 0, NULL);
}

// Read IOSurface (FP16 channel-first) → FP32 tensor using vDSP
static Tensor read_tensor_from_surface(IOSurfaceRef surface,
                                        const std::vector<int64_t> &ane_shape,
                                        const Shape &axiom_shape,
                                        std::vector<uint16_t> &fp16_buf,
                                        std::vector<float> &transpose_buf) {
    int channels = static_cast<int>(ane_shape[1]);
    int spatial = compute_spatial(ane_shape);
    size_t num_elements = static_cast<size_t>(channels * spatial);

    // Read FP16 with row alignment
    fp16_buf.resize(num_elements);
    IOSurfaceLock(surface, kIOSurfaceLockReadOnly, NULL);
    auto *base =
        static_cast<const uint8_t *>(IOSurfaceGetBaseAddress(surface));
    size_t row_bytes = IOSurfaceGetBytesPerRow(surface);
    size_t data_per_row = static_cast<size_t>(spatial) * sizeof(uint16_t);
    for (int c = 0; c < channels; c++) {
        std::memcpy(fp16_buf.data() + c * spatial, base + c * row_bytes,
                     data_per_row);
    }
    IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, NULL);

    // FP16 → FP32
    std::vector<float> f32(num_elements);
    vImage_Buffer src_vimg = {fp16_buf.data(), 1,
                               static_cast<vImagePixelCount>(num_elements),
                               num_elements * sizeof(uint16_t)};
    vImage_Buffer dst_vimg = {f32.data(), 1,
                               static_cast<vImagePixelCount>(num_elements),
                               num_elements * sizeof(float)};
    vImageConvert_Planar16FtoPlanarF(&src_vimg, &dst_vimg, 0);

    // Transpose [C, S] → [S, C] using vDSP
    bool needs_transpose =
        (spatial > 1) && !(ane_shape.size() == 4 && ane_shape[2] > 1);

    transpose_buf.resize(num_elements);
    if (needs_transpose) {
        vDSP_mtrans(f32.data(), 1, transpose_buf.data(), 1,
                     static_cast<vDSP_Length>(spatial),
                     static_cast<vDSP_Length>(channels));
    } else {
        std::memcpy(transpose_buf.data(), f32.data(),
                     num_elements * sizeof(float));
    }

    auto result = Tensor(axiom_shape, DType::Float32);
    std::memcpy(result.typed_data<float>(), transpose_buf.data(),
                num_elements * sizeof(float));
    return result;
}

// ============================================================================
// INT8 Weight Quantization (defined here for use by walk_module)
// ============================================================================

struct QuantizedWeight {
    std::vector<int8_t> data;
    std::vector<float> scale;
    std::vector<int8_t> zero_point;
    size_t out_channels;
    size_t elements_per_channel;
};

static QuantizedWeight quantize_weight(const Tensor &weight) {
    Tensor cpu = weight.cpu().ascontiguousarray().astype(DType::Float32);
    const float *src = cpu.typed_data<float>();

    QuantizedWeight qw;
    qw.out_channels = cpu.shape()[0];
    qw.elements_per_channel = cpu.size() / qw.out_channels;
    qw.data.resize(cpu.size());
    qw.scale.resize(qw.out_channels);
    qw.zero_point.resize(qw.out_channels, 0);

    for (size_t c = 0; c < qw.out_channels; c++) {
        const float *ch_data = src + c * qw.elements_per_channel;
        float max_abs = 0.0f;
        for (size_t i = 0; i < qw.elements_per_channel; i++)
            max_abs = std::max(max_abs, std::abs(ch_data[i]));
        float s = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
        qw.scale[c] = s;
        int8_t *dst = qw.data.data() + c * qw.elements_per_channel;
        for (size_t i = 0; i < qw.elements_per_channel; i++) {
            float val = std::max(-127.0f, std::min(127.0f,
                                  std::round(ch_data[i] / s)));
            dst[i] = static_cast<int8_t>(val);
        }
    }
    return qw;
}

// ============================================================================
// Module → MIL compilation
// ============================================================================

static bool g_quantize_weights = false;

static std::string walk_module(MILGenerator &gen, const nn::Module &module,
                                const std::string &input_var,
                                const std::string &prefix) {
    if (auto *linear = dynamic_cast<const nn::Linear *>(&module)) {
        const Tensor *bias =
            linear->has_bias() ? &linear->bias() : nullptr;

        if (g_quantize_weights && linear->weight().storage()) {
            auto qw = quantize_weight(linear->weight());
            return gen.add_linear_int8(
                input_var, qw.data, qw.scale, qw.zero_point,
                static_cast<int64_t>(linear->weight().shape()[0]),
                static_cast<int64_t>(linear->weight().shape()[1]),
                bias, prefix);
        }
        return gen.add_linear(input_var, linear->weight(), bias, prefix);
    }

    if (auto *conv = dynamic_cast<const nn::Conv2d *>(&module)) {
        const Tensor *bias = conv->has_bias() ? &conv->bias() : nullptr;
        return gen.add_conv2d(input_var, conv->weight(), bias, conv->stride(),
                              conv->padding(), conv->dilation(), conv->groups(),
                              prefix);
    }

    if (auto *mha = dynamic_cast<const nn::MultiHeadAttention *>(&module)) {
        const Tensor *qb =
            mha->q_proj().has_bias() ? &mha->q_proj().bias() : nullptr;
        const Tensor *kb =
            mha->k_proj().has_bias() ? &mha->k_proj().bias() : nullptr;
        const Tensor *vb =
            mha->v_proj().has_bias() ? &mha->v_proj().bias() : nullptr;
        const Tensor *ob =
            mha->out_proj().has_bias() ? &mha->out_proj().bias() : nullptr;
        return gen.add_multihead_attention(
            input_var, mha->q_proj().weight(), mha->k_proj().weight(),
            mha->v_proj().weight(), mha->out_proj().weight(), qb, kb, vb, ob,
            mha->num_heads(), prefix);
    }

    if (dynamic_cast<const nn::ReLU *>(&module))
        return gen.add_relu(input_var, prefix);
    if (dynamic_cast<const nn::SiLU *>(&module))
        return gen.add_silu(input_var, prefix);
    if (dynamic_cast<const nn::GELU *>(&module))
        return gen.add_gelu(input_var, prefix);
    if (dynamic_cast<const nn::Sigmoid *>(&module))
        return gen.add_sigmoid(input_var, prefix);

    if (auto *ln = dynamic_cast<const nn::LayerNorm *>(&module))
        return gen.add_layer_norm(input_var, ln->weight(), ln->bias(),
                                  ln->eps(), prefix);
    if (auto *rn = dynamic_cast<const nn::RMSNorm *>(&module))
        return gen.add_rms_norm(input_var, rn->weight(), rn->eps(), prefix);

    if (dynamic_cast<const nn::Dropout *>(&module))
        return input_var;

    // Sequential / containers: chain children
    if (dynamic_cast<const nn::Sequential *>(&module)) {
        auto &kids = module.children();
        std::string var = input_var;
        for (size_t i = 0; i < kids.size(); i++)
            var = walk_module(gen, *kids[i].second, var,
                              prefix + "_" + std::to_string(i));
        return var;
    }

    // Generic: walk children in order
    auto &kids = module.children();
    if (!kids.empty()) {
        std::string var = input_var;
        for (size_t i = 0; i < kids.size(); i++)
            var = walk_module(gen, *kids[i].second, var,
                              prefix + "_" + kids[i].first);
        return var;
    }

    throw RuntimeError("Unsupported module type for ANE compilation: " +
                       std::string(typeid(module).name()));
}

// ============================================================================
// Tiling
// ============================================================================

// Estimate working set for a given input shape and weight bytes.
static size_t estimate_working_set(const std::vector<int64_t> &ane_in,
                                    const std::vector<int64_t> &ane_out,
                                    size_t weight_bytes) {
    size_t in_bytes = sizeof(uint16_t);
    for (auto d : ane_in) in_bytes *= static_cast<size_t>(d);
    size_t out_bytes = sizeof(uint16_t);
    for (auto d : ane_out) out_bytes *= static_cast<size_t>(d);
    // Intermediates ≈ 2x output (conservative estimate for conv/matmul scratch)
    return in_bytes + out_bytes + out_bytes * 2 + weight_bytes;
}

// Compute optimal tile count to stay within SRAM budget.
static int compute_tile_count(const std::vector<int64_t> &ane_in,
                               const std::vector<int64_t> &ane_out,
                               size_t weight_bytes) {
    size_t full_working_set =
        estimate_working_set(ane_in, ane_out, weight_bytes);
    if (full_working_set <= ANE_SRAM_BUDGET)
        return 1; // No tiling needed

    // Tile along spatial dimension (last dim for [1,C,1,S])
    int spatial = compute_spatial(ane_in);
    int tiles = static_cast<int>(
        std::ceil(static_cast<double>(full_working_set) / ANE_SRAM_BUDGET));
    // Ensure tile size divides spatial evenly (round up tiles)
    while (spatial % tiles != 0 && tiles < spatial)
        tiles++;
    return std::min(tiles, spatial);
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
    if (dynamic_cast<const nn::Linear *>(&module)) return true;
    if (dynamic_cast<const nn::Conv2d *>(&module)) return true;
    if (dynamic_cast<const nn::MultiHeadAttention *>(&module)) return true;
    if (dynamic_cast<const nn::ReLU *>(&module)) return true;
    if (dynamic_cast<const nn::SiLU *>(&module)) return true;
    if (dynamic_cast<const nn::GELU *>(&module)) return true;
    if (dynamic_cast<const nn::Sigmoid *>(&module)) return true;
    if (dynamic_cast<const nn::LayerNorm *>(&module)) return true;
    if (dynamic_cast<const nn::RMSNorm *>(&module)) return true;
    if (dynamic_cast<const nn::Dropout *>(&module)) return true;

    if (dynamic_cast<const nn::Sequential *>(&module)) {
        for (auto &[name, child] : module.children())
            if (!is_supported(*child)) return false;
        return !module.children().empty();
    }

    auto &kids = module.children();
    if (!kids.empty()) {
        for (auto &[name, child] : kids)
            if (!is_supported(*child)) return false;
        return true;
    }
    return false;
}

ANECompiledModel ANECompiledModel::compile(const nn::Module &module,
                                            const Shape &input_shape,
                                            bool quantize_weights) {
    if (!ane_is_available())
        throw DeviceError::not_available("ANE");

    ANECompiledModel result;
    auto &impl = *result.impl_;

    impl.input_shape = input_shape;
    impl.ane_in_shape = to_ane_shape(input_shape);

    impl.input_elements = 1;
    for (auto d : input_shape)
        impl.input_elements *= d;

    // Generate MIL
    g_quantize_weights = quantize_weights;
    MILGenerator gen;
    gen.begin_program();
    auto x = gen.add_input("x", impl.ane_in_shape);
    std::string out_var;
    try {
        out_var = walk_module(gen, module, x, "m");
        g_quantize_weights = false;
    } catch (const std::exception &e) {
        g_quantize_weights = false;
        throw RuntimeError("ANE compilation failed: " + std::string(e.what()));
    }
    gen.set_output(out_var);
    std::string mil = gen.finalize();

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
    if (!impl.handle)
        throw RuntimeError("ANE MIL compilation failed");
    impl.compile_count = 1;

    int rc = ane_load(impl.handle);
    if (rc != 0) {
        ane_release(impl.handle);
        impl.handle = nullptr;
        throw RuntimeError("ANE model loading failed");
    }

    // Compute tiling
    impl.working_set_bytes =
        estimate_working_set(impl.ane_in_shape, impl.ane_out_shape,
                              impl.weight_bytes);
    impl.tile_count =
        compute_tile_count(impl.ane_in_shape, impl.ane_out_shape,
                            impl.weight_bytes);
    int full_spatial = compute_spatial(impl.ane_in_shape);
    impl.tile_spatial = full_spatial / impl.tile_count;

    // Pre-allocate IOSurfaces
    int in_c = static_cast<int>(impl.ane_in_shape[1]);
    int in_s = (impl.tile_count > 1) ? impl.tile_spatial : full_spatial;
    int out_c = static_cast<int>(impl.ane_out_shape[1]);
    int out_s = in_s; // Tile spatial preserved for shape-preserving ops

    // For ops that change spatial dims (unlikely for [1,C,1,S] layout),
    // compute the actual output spatial per tile
    if (impl.tile_count <= 1) {
        out_s = compute_spatial(impl.ane_out_shape);
    }

    impl.in_surface = ane_create_surface(in_c, in_s);
    impl.out_surface = ane_create_surface(out_c, out_s);
    if (!impl.in_surface || !impl.out_surface)
        throw MemoryError::allocation_failed(
            impl.input_elements * sizeof(uint16_t));

    // Pre-allocate conversion buffers
    size_t max_elems = std::max(impl.input_elements, impl.output_elements);
    impl.fp16_in_buf.resize(max_elems);
    impl.fp16_out_buf.resize(max_elems);
    impl.transpose_buf.resize(max_elems);

    impl.module_type = typeid(module).name();
    return result;
}

Tensor ANECompiledModel::forward(const Tensor &input) const {
    if (!impl_ || !impl_->handle)
        throw RuntimeError("ANECompiledModel not compiled");

    if (input.shape() != impl_->input_shape) {
        std::ostringstream oss;
        oss << "ANE input shape mismatch: expected [";
        for (size_t i = 0; i < impl_->input_shape.size(); i++) {
            if (i > 0) oss << ", ";
            oss << impl_->input_shape[i];
        }
        oss << "] but got [";
        for (size_t i = 0; i < input.shape().size(); i++) {
            if (i > 0) oss << ", ";
            oss << input.shape()[i];
        }
        oss << "]";
        throw ShapeError(oss.str());
    }

    if (impl_->tile_count <= 1) {
        // No tiling — single dispatch
        write_tensor_to_surface(impl_->in_surface, input,
                                 impl_->ane_in_shape, impl_->transpose_buf,
                                 impl_->fp16_in_buf);
        int rc = ane_eval(impl_->handle, impl_->in_surface,
                           impl_->out_surface);
        if (rc != 0)
            throw RuntimeError("ANE evaluation failed");
        return read_tensor_from_surface(impl_->out_surface,
                                         impl_->ane_out_shape,
                                         impl_->output_shape,
                                         impl_->fp16_out_buf,
                                         impl_->transpose_buf);
    }

    // Tiled execution: split along spatial (batch) dimension
    Tensor cpu_input = input.cpu().ascontiguousarray();
    if (cpu_input.dtype() != DType::Float32)
        cpu_input = cpu_input.astype(DType::Float32);

    int channels = static_cast<int>(impl_->ane_in_shape[1]);
    int full_spatial = compute_spatial(impl_->ane_in_shape);
    int tile_sp = impl_->tile_spatial;

    auto result = Tensor(impl_->output_shape, DType::Float32);
    float *result_ptr = result.typed_data<float>();
    const float *input_ptr = cpu_input.typed_data<float>();

    int out_channels = static_cast<int>(impl_->ane_out_shape[1]);

    for (int t = 0; t < impl_->tile_count; t++) {
        int sp_start = t * tile_sp;
        int sp_end = std::min(sp_start + tile_sp, full_spatial);
        int cur_sp = sp_end - sp_start;

        // Extract tile from input (contiguous slice along spatial/batch dim)
        size_t tile_elems = static_cast<size_t>(channels * cur_sp);
        std::vector<float> tile_data(tile_elems);

        // Transpose tile: [sp_start:sp_end, :] → [C, cur_sp]
        for (int s = 0; s < cur_sp; s++) {
            for (int c = 0; c < channels; c++) {
                tile_data[c * cur_sp + s] =
                    input_ptr[(sp_start + s) * channels + c];
            }
        }

        // Convert to FP16 and write to IOSurface
        std::vector<uint16_t> tile_fp16(tile_elems);
        vImage_Buffer src_v = {tile_data.data(), 1,
                                static_cast<vImagePixelCount>(tile_elems),
                                tile_elems * sizeof(float)};
        vImage_Buffer dst_v = {tile_fp16.data(), 1,
                                static_cast<vImagePixelCount>(tile_elems),
                                tile_elems * sizeof(uint16_t)};
        vImageConvert_PlanarFtoPlanar16F(&src_v, &dst_v, 0);

        IOSurfaceLock(impl_->in_surface, 0, NULL);
        auto *base = static_cast<uint8_t *>(
            IOSurfaceGetBaseAddress(impl_->in_surface));
        std::memset(base, 0, IOSurfaceGetAllocSize(impl_->in_surface));
        size_t row_bytes = IOSurfaceGetBytesPerRow(impl_->in_surface);
        for (int c = 0; c < channels; c++) {
            std::memcpy(base + c * row_bytes,
                         tile_fp16.data() + c * cur_sp,
                         cur_sp * sizeof(uint16_t));
        }
        IOSurfaceUnlock(impl_->in_surface, 0, NULL);

        // Eval tile
        int rc = ane_eval(impl_->handle, impl_->in_surface,
                           impl_->out_surface);
        if (rc != 0)
            throw RuntimeError("ANE tiled evaluation failed on tile " +
                               std::to_string(t));

        // Read output tile
        size_t out_tile_elems = static_cast<size_t>(out_channels * cur_sp);
        std::vector<uint16_t> out_fp16(out_tile_elems);
        IOSurfaceLock(impl_->out_surface, kIOSurfaceLockReadOnly, NULL);
        auto *out_base = static_cast<const uint8_t *>(
            IOSurfaceGetBaseAddress(impl_->out_surface));
        size_t out_row = IOSurfaceGetBytesPerRow(impl_->out_surface);
        for (int c = 0; c < out_channels; c++) {
            std::memcpy(out_fp16.data() + c * cur_sp,
                         out_base + c * out_row,
                         cur_sp * sizeof(uint16_t));
        }
        IOSurfaceUnlock(impl_->out_surface, kIOSurfaceLockReadOnly, NULL);

        // FP16 → FP32
        std::vector<float> out_f32(out_tile_elems);
        vImage_Buffer s2 = {out_fp16.data(), 1,
                             static_cast<vImagePixelCount>(out_tile_elems),
                             out_tile_elems * sizeof(uint16_t)};
        vImage_Buffer d2 = {out_f32.data(), 1,
                             static_cast<vImagePixelCount>(out_tile_elems),
                             out_tile_elems * sizeof(float)};
        vImageConvert_Planar16FtoPlanarF(&s2, &d2, 0);

        // Transpose back [C, cur_sp] → [cur_sp, C] and copy to result
        for (int s = 0; s < cur_sp; s++) {
            for (int c = 0; c < out_channels; c++) {
                result_ptr[(sp_start + s) * out_channels + c] =
                    out_f32[c * cur_sp + s];
            }
        }
    }

    return result;
}

ANEPlanInfo ANECompiledModel::plan_info() const {
    ANEPlanInfo info;
    info.ane_steps = impl_ ? impl_->tile_count : 0;
    info.cpu_steps = 0;
    info.compile_count = impl_ ? impl_->compile_count : 0;
    info.weight_bytes = impl_ ? impl_->weight_bytes : 0;

    std::ostringstream oss;
    oss << "ANE compiled model";
    if (impl_) {
        oss << " (input: [";
        for (size_t i = 0; i < impl_->input_shape.size(); i++) {
            if (i > 0) oss << ", ";
            oss << impl_->input_shape[i];
        }
        oss << "] → output: [";
        for (size_t i = 0; i < impl_->output_shape.size(); i++) {
            if (i > 0) oss << ", ";
            oss << impl_->output_shape[i];
        }
        oss << "]";
        if (impl_->tile_count > 1) {
            oss << ", " << impl_->tile_count << " tiles"
                << " (working set " << impl_->working_set_bytes / 1024 / 1024
                << "MB > " << ANE_SRAM_BUDGET / 1024 / 1024 << "MB budget)";
        }
    }
    info.summary = oss.str();
    return info;
}

} // namespace ane
} // namespace backends
} // namespace axiom

#endif // AXIOM_HAS_ANE
