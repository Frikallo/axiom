#import "mil_generator.hpp"
#import "ane_bridge.h"

#import <Accelerate/Accelerate.h>

#include <cmath>
#include <sstream>
#include <stdexcept>

namespace axiom {
namespace backends {
namespace ane {

// ============================================================================
// Formatting helpers
// ============================================================================

std::string MILGenerator::mil_type(const std::vector<int64_t> &shape) {
    std::ostringstream oss;
    oss << "tensor<fp16, [";
    for (size_t i = 0; i < shape.size(); i++) {
        if (i > 0)
            oss << ", ";
        oss << shape[i];
    }
    oss << "]>";
    return oss.str();
}

std::string MILGenerator::mil_shape(const std::vector<int64_t> &shape) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < shape.size(); i++) {
        if (i > 0)
            oss << ", ";
        oss << shape[i];
    }
    oss << "]";
    return oss.str();
}

std::string MILGenerator::next_var(const std::string &prefix) {
    return prefix + std::to_string(var_counter_++);
}

void MILGenerator::track(const std::string &var,
                          const std::vector<int64_t> &shape) {
    shapes_[var] = shape;
}

const std::vector<int64_t> &
MILGenerator::shape_of(const std::string &var) const {
    auto it = shapes_.find(var);
    if (it == shapes_.end()) {
        throw std::runtime_error("MILGenerator: unknown variable '" + var +
                                 "'");
    }
    return it->second;
}

// ============================================================================
// Weight packing
// ============================================================================

std::string MILGenerator::pack_weight(const Tensor &t,
                                       const std::vector<int64_t> &ane_shape,
                                       const std::string &name) {
    Tensor cpu_t = t.cpu().ascontiguousarray();

    size_t num_elements = 1;
    for (auto d : ane_shape)
        num_elements *= static_cast<size_t>(d);

    // Convert to FP16
    std::vector<uint16_t> fp16_data(num_elements);

    Tensor f32 =
        (cpu_t.dtype() == DType::Float32) ? cpu_t : cpu_t.astype(DType::Float32);
    const float *src = f32.typed_data<float>();

    vImage_Buffer src_buf = {
        .data = const_cast<float *>(src),
        .height = 1,
        .width = static_cast<vImagePixelCount>(num_elements),
        .rowBytes = num_elements * sizeof(float),
    };
    vImage_Buffer dst_buf = {
        .data = fp16_data.data(),
        .height = 1,
        .width = static_cast<vImagePixelCount>(num_elements),
        .rowBytes = num_elements * sizeof(uint16_t),
    };
    vImageConvert_PlanarFtoPlanar16F(&src_buf, &dst_buf, 0);

    // Build blob (128-byte header + data)
    size_t data_bytes = num_elements * sizeof(uint16_t);
    size_t total_size = 0;
    void *blob =
        ane_build_weight_blob(fp16_data.data(), data_bytes, &total_size);

    WeightBlob wb;
    wb.name = name;
    wb.blob_data.resize(total_size);
    std::memcpy(wb.blob_data.data(), blob, total_size);
    free(blob);

    weight_blobs_.push_back(std::move(wb));
    return name;
}

// ============================================================================
// Constant emission
// ============================================================================

std::string MILGenerator::emit_scalar_const(const std::string &name,
                                             const std::string &type,
                                             float value) {
    std::ostringstream oss;
    oss << "        " << type << " " << name << " = const()[name=string(\""
        << name << "\"), val=" << type << "(" << value << ")];\n";
    body_ += oss.str();
    return name;
}

std::string MILGenerator::emit_int_const(const std::string &name, int value) {
    std::ostringstream oss;
    oss << "        int32 " << name << " = const()[name=string(\"" << name
        << "\"), val=int32(" << value << ")];\n";
    body_ += oss.str();
    return name;
}

std::string MILGenerator::emit_bool_const(const std::string &name,
                                           bool value) {
    std::ostringstream oss;
    oss << "        bool " << name << " = const()[name=string(\"" << name
        << "\"), val=bool(" << (value ? "true" : "false") << ")];\n";
    body_ += oss.str();
    return name;
}

std::string
MILGenerator::emit_int_tensor_const(const std::string &name,
                                     const std::vector<int> &values) {
    std::ostringstream oss;
    oss << "        tensor<int32, [" << values.size() << "]> " << name
        << " = const()[name=string(\"" << name << "\"), val=tensor<int32, ["
        << values.size() << "]>([";
    for (size_t i = 0; i < values.size(); i++) {
        if (i > 0)
            oss << ", ";
        oss << values[i];
    }
    oss << "])];\n";
    body_ += oss.str();
    return name;
}

// ============================================================================
// Conv boilerplate (emitted once per program)
// ============================================================================

void MILGenerator::ensure_conv_consts() {
    if (conv_consts_emitted_)
        return;
    conv_consts_emitted_ = true;

    emit_int_tensor_const("_st", {1, 1});
    emit_int_tensor_const("_pd", {0, 0, 0, 0});
    emit_int_tensor_const("_dl", {1, 1});
    emit_int_const("_gr", 1);
    body_ += "        string _pt = const()[name=string(\"_pt\"), "
             "val=string(\"valid\")];\n";
}

// ============================================================================
// Program structure
// ============================================================================

void MILGenerator::begin_program() {
    body_.clear();
    input_decl_.clear();
    output_var_.clear();
    weight_blobs_.clear();
    shapes_.clear();
    var_counter_ = 0;
    conv_consts_emitted_ = false;
}

std::string MILGenerator::add_input(const std::string &name,
                                     const std::vector<int64_t> &shape) {
    input_decl_ = mil_type(shape) + " " + name;
    track(name, shape);
    return name;
}

void MILGenerator::set_output(const std::string &output_var) {
    output_var_ = output_var;
}

std::string MILGenerator::finalize() {
    std::ostringstream oss;
    oss << "program(1.3)\n";
    oss << "[buildInfo = dict<string, string>({"
           "{\"coremlc-component-MIL\", \"3510.2.1\"}, "
           "{\"coremlc-version\", \"3505.4.1\"}, "
           "{\"coremltools-component-milinternal\", \"\"}, "
           "{\"coremltools-version\", \"9.0\"}"
           "})]\n";
    oss << "{\n";
    oss << "    func main<ios18>(" << input_decl_ << ") {\n";
    oss << body_;
    oss << "    } -> (" << output_var_ << ");\n";
    oss << "}\n";
    return oss.str();
}

// ============================================================================
// Linear (as 1x1 conv)
// ============================================================================

std::string MILGenerator::add_linear(const std::string &input_var,
                                      const Tensor &weight, const Tensor *bias,
                                      const std::string &name) {
    ensure_conv_consts();

    auto &in_shape = shape_of(input_var);
    int64_t out_features = weight.shape()[0];
    int64_t in_features = weight.shape()[1];
    std::vector<int64_t> out_shape = {in_shape[0], out_features, in_shape[2],
                                       in_shape[3]};

    // Weight: [out, in, 1, 1]
    std::vector<int64_t> w_shape = {out_features, in_features, 1, 1};
    std::string w_name = name + "_w";
    pack_weight(weight, w_shape, w_name);

    // Emit weight BLOBFILE const
    body_ += "        " + mil_type(w_shape) + " " + w_name +
             " = const()[name=string(\"" + w_name + "\"), val=" +
             mil_type(w_shape) + "(BLOBFILE(path=string(" +
             "\"@model_path/weights/" + w_name +
             ".bin\"), offset=uint64(64)))];\n";

    std::string out_var = next_var("l");

    if (bias) {
        // Bias: [out_features] (1D for conv)
        std::string b_name = name + "_b";
        pack_weight(*bias, {out_features}, b_name);

        body_ += "        tensor<fp16, [" + std::to_string(out_features) +
                 "]> " + b_name + " = const()[name=string(\"" + b_name +
                 "\"), val=tensor<fp16, [" + std::to_string(out_features) +
                 "]>(BLOBFILE(path=string(\"@model_path/weights/" + b_name +
                 ".bin\"), offset=uint64(64)))];\n";

        body_ += "        " + mil_type(out_shape) + " " + out_var +
                 " = conv(bias=" + b_name +
                 ", dilations=_dl, groups=_gr, pad=_pd, pad_type=_pt, "
                 "strides=_st, weight=" +
                 w_name + ", x=" + input_var + ")[name=string(\"" + name +
                 "\")];\n";
    } else {
        body_ += "        " + mil_type(out_shape) + " " + out_var +
                 " = conv(dilations=_dl, groups=_gr, pad=_pd, pad_type=_pt, "
                 "strides=_st, weight=" +
                 w_name + ", x=" + input_var + ")[name=string(\"" + name +
                 "\")];\n";
    }

    track(out_var, out_shape);
    return out_var;
}

// ============================================================================
// Element-wise operations (shape-preserving for matching shapes)
// ============================================================================

std::string MILGenerator::add_add(const std::string &a, const std::string &b,
                                   const std::string &name) {
    // Use shape of a (assumes broadcastable)
    auto it = shapes_.find(a);
    auto &out_shape = (it != shapes_.end()) ? it->second : shapes_.begin()->second;

    std::string out = next_var("a");
    body_ += "        " + mil_type(out_shape) + " " + out + " = add(x=" + a +
             ", y=" + b + ")[name=string(\"" + name + "\")];\n";
    track(out, out_shape);
    return out;
}

std::string MILGenerator::add_mul(const std::string &a, const std::string &b,
                                   const std::string &name) {
    auto it = shapes_.find(a);
    auto &out_shape = (it != shapes_.end()) ? it->second : shapes_.begin()->second;

    std::string out = next_var("m");
    body_ += "        " + mil_type(out_shape) + " " + out + " = mul(x=" + a +
             ", y=" + b + ")[name=string(\"" + name + "\")];\n";
    track(out, out_shape);
    return out;
}

std::string MILGenerator::add_sub(const std::string &a, const std::string &b,
                                   const std::string &name) {
    auto it = shapes_.find(a);
    auto &out_shape = (it != shapes_.end()) ? it->second : shapes_.begin()->second;

    std::string out = next_var("s");
    body_ += "        " + mil_type(out_shape) + " " + out + " = sub(x=" + a +
             ", y=" + b + ")[name=string(\"" + name + "\")];\n";
    track(out, out_shape);
    return out;
}

// ============================================================================
// Activations (shape-preserving)
// ============================================================================

std::string MILGenerator::add_relu(const std::string &input_var,
                                    const std::string &name) {
    // Use relu op (standard MIL op)
    auto &out_shape = shape_of(input_var);
    std::string out = next_var("r");
    body_ += "        " + mil_type(out_shape) + " " + out +
             " = relu(x=" + input_var + ")[name=string(\"" + name + "\")];\n";
    track(out, out_shape);
    return out;
}

std::string MILGenerator::add_sigmoid(const std::string &input_var,
                                       const std::string &name) {
    auto &out_shape = shape_of(input_var);
    std::string out = next_var("sg");
    body_ += "        " + mil_type(out_shape) + " " + out +
             " = sigmoid(x=" + input_var + ")[name=string(\"" + name +
             "\")];\n";
    track(out, out_shape);
    return out;
}

std::string MILGenerator::add_silu(const std::string &input_var,
                                    const std::string &name) {
    // SiLU = x * sigmoid(x)
    std::string sig = add_sigmoid(input_var, name + "_sig");
    return add_mul(input_var, sig, name);
}

std::string MILGenerator::add_gelu(const std::string &input_var,
                                    const std::string &name) {
    // GELU ≈ x * sigmoid(1.702 * x)
    auto &in_shape = shape_of(input_var);
    std::string scale = next_var("gs");
    emit_scalar_const(scale, "fp16", 1.702f);

    std::string scaled = add_mul(input_var, scale, name + "_sc");
    std::string sig = add_sigmoid(scaled, name + "_sig");
    return add_mul(input_var, sig, name);
}

std::string MILGenerator::add_softmax(const std::string &input_var, int axis,
                                       const std::string &name) {
    auto &out_shape = shape_of(input_var);
    std::string ax = next_var("sax");
    emit_int_const(ax, axis);

    std::string out = next_var("sm");
    body_ += "        " + mil_type(out_shape) + " " + out +
             " = softmax(axis=" + ax + ", x=" + input_var +
             ")[name=string(\"" + name + "\")];\n";
    track(out, out_shape);
    return out;
}

// ============================================================================
// Normalization
// ============================================================================

std::string MILGenerator::add_rms_norm(const std::string &input_var,
                                        const Tensor &weight, float eps,
                                        const std::string &name) {
    auto &in_shape = shape_of(input_var);
    int64_t dim = weight.shape()[0];

    // Pack weight as [1, dim, 1, 1]
    std::string rw_name = name + "_rw";
    pack_weight(weight, {1, dim, 1, 1}, rw_name);

    body_ += "        tensor<fp16, [1, " + std::to_string(dim) +
             ", 1, 1]> " + rw_name + " = const()[name=string(\"" + rw_name +
             "\"), val=tensor<fp16, [1, " + std::to_string(dim) +
             ", 1, 1]>(BLOBFILE(path=string(\"@model_path/weights/" +
             rw_name + ".bin\"), offset=uint64(64)))];\n";

    // Reduction output: [1, 1, 1, S]
    std::vector<int64_t> reduced_shape = {1, 1, in_shape[2], in_shape[3]};

    std::string eps_var = next_var("eps");
    emit_scalar_const(eps_var, "fp16", eps);

    std::string invd_var = next_var("invd");
    emit_scalar_const(invd_var, "fp16", 1.0f / static_cast<float>(dim));

    std::string nhalf_var = next_var("nh");
    emit_scalar_const(nhalf_var, "fp16", -0.5f);

    std::string axes_var = next_var("rax");
    emit_int_tensor_const(axes_var, {1});

    std::string kd_var = next_var("kd");
    emit_bool_const(kd_var, true);

    // sq = x * x
    std::string sq = add_mul(input_var, input_var, name + "_sq");

    // ss = reduce_sum(sq, axis=1, keep_dims=true)
    std::string ss = next_var("ss");
    body_ += "        " + mil_type(reduced_shape) + " " + ss +
             " = reduce_sum(axes=" + axes_var + ", keep_dims=" + kd_var +
             ", x=" + sq + ")[name=string(\"" + name + "_ss\")];\n";
    track(ss, reduced_shape);

    // ss2 = ss * invd
    std::string ss2 = add_mul(ss, invd_var, name + "_ss2");

    // ss3 = ss2 + eps
    std::string ss3 = add_add(ss2, eps_var, name + "_ss3");

    // rrms = ss3 ^ (-0.5)
    std::string rrms = next_var("rrms");
    body_ += "        " + mil_type(reduced_shape) + " " + rrms +
             " = pow(x=" + ss3 + ", y=" + nhalf_var + ")[name=string(\"" +
             name + "_rrms\")];\n";
    track(rrms, reduced_shape);

    // xr = x * rrms (broadcast)
    std::string xr = add_mul(input_var, rrms, name + "_xr");

    // xn = xr * rw (broadcast)
    return add_mul(xr, rw_name, name + "_xn");
}

std::string MILGenerator::add_layer_norm(const std::string &input_var,
                                          const Tensor &weight,
                                          const Tensor &bias, float eps,
                                          const std::string &name) {
    auto &in_shape = shape_of(input_var);
    int64_t dim = weight.shape()[0];

    // Weight and bias as [1, dim, 1, 1]
    std::string w_name = name + "_lnw";
    std::string b_name = name + "_lnb";
    pack_weight(weight, {1, dim, 1, 1}, w_name);
    pack_weight(bias, {1, dim, 1, 1}, b_name);

    for (auto &n : {w_name, b_name}) {
        body_ += "        tensor<fp16, [1, " + std::to_string(dim) +
                 ", 1, 1]> " + n + " = const()[name=string(\"" + n +
                 "\"), val=tensor<fp16, [1, " + std::to_string(dim) +
                 ", 1, 1]>(BLOBFILE(path=string(\"@model_path/weights/" + n +
                 ".bin\"), offset=uint64(64)))];\n";
    }

    std::vector<int64_t> reduced_shape = {1, 1, in_shape[2], in_shape[3]};

    std::string eps_var = next_var("eps");
    emit_scalar_const(eps_var, "fp16", eps);

    std::string invd_var = next_var("invd");
    emit_scalar_const(invd_var, "fp16", 1.0f / static_cast<float>(dim));

    std::string axes_var = next_var("rax");
    emit_int_tensor_const(axes_var, {1});

    std::string kd_var = next_var("kd");
    emit_bool_const(kd_var, true);

    // mean = reduce_sum(x, axis=1) * invd
    std::string sum_x = next_var("sx");
    body_ += "        " + mil_type(reduced_shape) + " " + sum_x +
             " = reduce_sum(axes=" + axes_var + ", keep_dims=" + kd_var +
             ", x=" + input_var + ")[name=string(\"" + name + "_sx\")];\n";
    track(sum_x, reduced_shape);

    std::string mean = add_mul(sum_x, invd_var, name + "_mean");

    // xc = x - mean
    std::string xc = add_sub(input_var, mean, name + "_xc");

    // var = reduce_sum(xc*xc) * invd
    std::string sq = add_mul(xc, xc, name + "_sq");
    std::string sum_sq = next_var("ssq");
    body_ += "        " + mil_type(reduced_shape) + " " + sum_sq +
             " = reduce_sum(axes=" + axes_var + ", keep_dims=" + kd_var +
             ", x=" + sq + ")[name=string(\"" + name + "_ssq\")];\n";
    track(sum_sq, reduced_shape);

    std::string var = add_mul(sum_sq, invd_var, name + "_var");

    // rstd = (var + eps) ^ (-0.5)
    std::string nhalf_var = next_var("nh");
    emit_scalar_const(nhalf_var, "fp16", -0.5f);

    std::string var_eps = add_add(var, eps_var, name + "_ve");

    std::string rstd = next_var("rstd");
    body_ += "        " + mil_type(reduced_shape) + " " + rstd +
             " = pow(x=" + var_eps + ", y=" + nhalf_var + ")[name=string(\"" +
             name + "_rstd\")];\n";
    track(rstd, reduced_shape);

    // out = (xc * rstd) * weight + bias
    std::string xn = add_mul(xc, rstd, name + "_xn");
    std::string scaled = add_mul(xn, w_name, name + "_sc");
    return add_add(scaled, b_name, name + "_out");
}

// ============================================================================
// Shape operations
// ============================================================================

std::string MILGenerator::add_reshape(const std::string &input_var,
                                       const std::vector<int64_t> &shape,
                                       const std::string &name) {
    std::string shape_var = next_var("sh");
    std::vector<int> shape_ints(shape.begin(), shape.end());
    emit_int_tensor_const(shape_var, shape_ints);

    std::string out = next_var("rs");
    body_ += "        " + mil_type(shape) + " " + out +
             " = reshape(shape=" + shape_var + ", x=" + input_var +
             ")[name=string(\"" + name + "\")];\n";
    track(out, shape);
    return out;
}

std::string MILGenerator::add_transpose(const std::string &input_var,
                                          const std::vector<int> &perm,
                                          const std::string &name) {
    auto &in_shape = shape_of(input_var);
    std::vector<int64_t> out_shape(perm.size());
    for (size_t i = 0; i < perm.size(); i++) {
        out_shape[i] = in_shape[static_cast<size_t>(perm[i])];
    }

    std::string perm_var = next_var("pm");
    emit_int_tensor_const(perm_var, perm);

    std::string out = next_var("tp");
    body_ += "        " + mil_type(out_shape) + " " + out +
             " = transpose(perm=" + perm_var + ", x=" + input_var +
             ")[name=string(\"" + name + "\")];\n";
    track(out, out_shape);
    return out;
}

// ============================================================================
// Matmul
// ============================================================================

std::string MILGenerator::add_matmul(const std::string &a,
                                      const std::string &b, bool transpose_a,
                                      bool transpose_b,
                                      const std::string &name) {
    auto &a_shape = shape_of(a);
    auto &b_shape = shape_of(b);

    // Compute output shape for matmul
    // [..., M, K] @ [..., K, N] → [..., M, N]
    std::vector<int64_t> out_shape = a_shape;
    int64_t m = transpose_a ? a_shape.back() : a_shape[a_shape.size() - 2];
    int64_t n = transpose_b ? b_shape[b_shape.size() - 2] : b_shape.back();
    out_shape[out_shape.size() - 2] = m;
    out_shape[out_shape.size() - 1] = n;

    std::string ta = next_var("ta");
    std::string tb = next_var("tb");
    emit_bool_const(ta, transpose_a);
    emit_bool_const(tb, transpose_b);

    std::string out = next_var("mm");
    body_ += "        " + mil_type(out_shape) + " " + out +
             " = matmul(transpose_x=" + ta + ", transpose_y=" + tb +
             ", x=" + a + ", y=" + b + ")[name=string(\"" + name + "\")];\n";
    track(out, out_shape);
    return out;
}

} // namespace ane
} // namespace backends
} // namespace axiom
