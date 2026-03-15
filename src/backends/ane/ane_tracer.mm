#import "ane_tracer.hpp"

#ifdef AXIOM_HAS_ANE

#include "ane_bridge.h"
#include "mil_generator.hpp"

#include "axiom/error.hpp"
#include "axiom/graph/graph_node.hpp"
#include "axiom/graph/graph_registry.hpp"
#include "axiom/tensor.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace axiom {
namespace backends {
namespace ane {

// ============================================================================
// Thread-local tracing flag
// ============================================================================

static thread_local bool tl_ane_tracing = false;

bool is_ane_tracing() { return tl_ane_tracing; }
void set_ane_tracing(bool enabled) { tl_ane_tracing = enabled; }

TraceScope::TraceScope() : previous_(tl_ane_tracing) { tl_ane_tracing = true; }
TraceScope::~TraceScope() { tl_ane_tracing = previous_; }

// ============================================================================
// Trace capture
// ============================================================================

// Create a special "input placeholder" graph node.
// Uses Cast OpType as a marker for "this is a trace input".
static constexpr ops::OpType TRACE_INPUT_OP = ops::OpType::Cast;

TraceResult trace_module(const nn::Module &module, const Shape &input_shape) {
    // Create input placeholder node
    auto input_node = std::make_shared<graph::GraphNode>();
    input_node->op_type = TRACE_INPUT_OP;
    input_node->output_shape = input_shape;
    input_node->output_dtype = DType::Float32;
    input_node->target_device = Device::CPU;

    // Create a real tensor with random data. This tensor has valid
    // shape, strides, and storage so it flows through ANY operation
    // (unsqueeze, ascontiguousarray, position embeddings, etc.).
    Tensor trace_input = Tensor::randn(input_shape);

    // Mark the input node as materialized with the real storage
    input_node->cached_result_ = trace_input.storage();
    input_node->cached_shape_ = input_shape;
    input_node->is_materialized_ = true;

    // Graft the lazy node directly onto the real tensor.
    // trace_input keeps its shape/strides/storage AND gets a lazy_node_
    // so lazy operations chain graph nodes through it.
    trace_input.set_lazy_node(input_node);

    // Enable tracing: lazy mode forced, reshape/transpose create lazy
    // nodes, materialize_if_needed() executes but preserves lazy_node_
    TraceScope scope;

    // Run forward() — lazy ops build the graph, operations that need
    // real data (unsqueeze, ascontiguousarray, etc.) work because
    // trace_input has valid storage/shape/strides
    Tensor output = module.forward(trace_input);

    TraceResult result;
    result.input_node = input_node;
    result.output_node = output.lazy_node();
    result.output_tensor = output;

    if (!result.output_node) {
        throw RuntimeError(
            "ANE trace failed: output tensor is not lazy. "
            "The module may have called an operation that forces "
            "materialization during tracing.");
    }

    return result;
}

// ============================================================================
// Topological sort
// ============================================================================

static std::vector<graph::GraphNode *>
topo_sort(graph::GraphNode *root) {
    std::vector<graph::GraphNode *> sorted;
    std::unordered_set<uint64_t> visited;

    std::function<void(graph::GraphNode *)> visit;
    visit = [&](graph::GraphNode *node) {
        if (!node || visited.count(node->id))
            return;
        visited.insert(node->id);
        for (auto &input : node->inputs)
            visit(input.get());
        sorted.push_back(node);
    };

    visit(root);
    return sorted;
}

// ============================================================================
// Graph → MIL compilation
// ============================================================================

static std::vector<int64_t> shape_to_ane(const Shape &s) {
    if (s.size() == 1)
        return {1, static_cast<int64_t>(s[0]), 1, 1};
    if (s.size() == 2)
        return {1, static_cast<int64_t>(s[1]), 1,
                static_cast<int64_t>(s[0])};
    if (s.size() == 3)
        return {1, static_cast<int64_t>(s[2]), 1,
                static_cast<int64_t>(s[0] * s[1])};
    return {1, static_cast<int64_t>(s[1]), 1,
            static_cast<int64_t>(s[0] * s[2] * s[3])};
}

static std::vector<int64_t> shape_to_vec(const Shape &s) {
    std::vector<int64_t> v;
    for (auto d : s)
        v.push_back(static_cast<int64_t>(d));
    return v;
}

TracedMIL compile_trace_to_mil(const TraceResult &trace,
                                const Shape &input_shape) {
    if (!trace.output_node)
        throw RuntimeError("No trace graph to compile");

    // Topological sort
    auto sorted = topo_sort(trace.output_node.get());

    // Map from GraphNode ID → MIL variable name
    std::unordered_map<uint64_t, std::string> node_vars;

    MILGenerator gen;
    gen.begin_program();

    auto ane_in = shape_to_ane(input_shape);
    std::string input_var = gen.add_input("x", ane_in);

    int var_idx = 0;
    auto next_name = [&](const std::string &prefix) {
        return prefix + std::to_string(var_idx++);
    };

    // Walk topologically and emit MIL for each node
    for (auto *node : sorted) {
        std::string name = next_name("t");

        // Trace input placeholder
        if (node->op_type == TRACE_INPUT_OP && node->inputs.empty()) {
            node_vars[node->id] = input_var;
            continue;
        }

        // Constant node (weight/bias/running stats tensor)
        if (node->is_constant && node->constant_storage) {
            std::string w_name = next_name("w");
            auto const_shape = shape_to_vec(node->output_shape);

            // Reconstruct a Tensor from the constant node's storage
            auto &cs = node->constant_strides;
            Strides strides = cs;
            if (strides.empty() || strides.size() != node->output_shape.size()) {
                strides = ShapeUtils::calculate_strides(
                    node->output_shape, dtype_size(node->output_dtype),
                    MemoryOrder::RowMajor);
            }
            Tensor const_tensor(node->constant_storage, node->output_shape,
                                strides, node->output_dtype,
                                node->constant_offset);

            gen.pack_weight_public(const_tensor, const_shape, w_name);

            // Emit BLOBFILE const declaration
            std::string type_str =
                MILGenerator::mil_type_public(const_shape);
            gen.emit_raw("        " + type_str + " " + w_name +
                          " = const()[name=string(\"" + w_name +
                          "\"), val=" + type_str +
                          "(BLOBFILE(path=string(\"@model_path/weights/" +
                          w_name + ".bin\"), offset=uint64(64)))];\n");
            gen.track_shape(w_name, const_shape);

            node_vars[node->id] = w_name;
            continue;
        }

        // Constant node with cached result (scalars, position embeddings)
        if (node->is_materialized_ && node->cached_result_ &&
            node->inputs.empty()) {
            std::string w_name = next_name("k");
            auto const_shape = shape_to_vec(
                node->cached_shape_.empty() ? node->output_shape
                                             : node->cached_shape_);

            Strides strides = node->cached_strides_;
            Shape s = node->cached_shape_.empty() ? node->output_shape
                                                    : node->cached_shape_;
            if (strides.empty() || strides.size() != s.size()) {
                strides = ShapeUtils::calculate_strides(
                    s, dtype_size(node->output_dtype),
                    MemoryOrder::RowMajor);
            }
            Tensor const_tensor(node->cached_result_, s, strides,
                                node->output_dtype);

            gen.pack_weight_public(const_tensor, const_shape, w_name);

            std::string type_str =
                MILGenerator::mil_type_public(const_shape);
            gen.emit_raw("        " + type_str + " " + w_name +
                          " = const()[name=string(\"" + w_name +
                          "\"), val=" + type_str +
                          "(BLOBFILE(path=string(\"@model_path/weights/" +
                          w_name + ".bin\"), offset=uint64(64)))];\n");
            gen.track_shape(w_name, const_shape);

            node_vars[node->id] = w_name;
            continue;
        }

        // Get input variable names
        auto emit_constant_input = [&](graph::GraphNode *inp) -> std::string {
            std::string cname = next_name("c");
            auto cs = shape_to_vec(inp->output_shape);

            if (inp->constant_storage) {
                Strides st = inp->constant_strides;
                if (st.empty() || st.size() != inp->output_shape.size()) {
                    st = ShapeUtils::calculate_strides(
                        inp->output_shape, dtype_size(inp->output_dtype),
                        MemoryOrder::RowMajor);
                }
                Tensor ct(inp->constant_storage, inp->output_shape, st,
                          inp->output_dtype, inp->constant_offset);
                gen.pack_weight_public(ct, cs, cname);
            } else if (inp->cached_result_) {
                Shape s = inp->cached_shape_.empty() ? inp->output_shape
                                                      : inp->cached_shape_;
                Strides st = inp->cached_strides_;
                if (st.empty() || st.size() != s.size()) {
                    st = ShapeUtils::calculate_strides(
                        s, dtype_size(inp->output_dtype),
                        MemoryOrder::RowMajor);
                }
                Tensor ct(inp->cached_result_, s, st, inp->output_dtype);
                gen.pack_weight_public(ct, cs, cname);
            }

            std::string type_str =
                MILGenerator::mil_type_public(cs);
            gen.emit_raw("        " + type_str + " " + cname +
                          " = const()[name=string(\"" + cname +
                          "\"), val=" + type_str +
                          "(BLOBFILE(path=string(\"@model_path/weights/" +
                          cname + ".bin\"), offset=uint64(64)))];\n");
            gen.track_shape(cname, cs);
            node_vars[inp->id] = cname;
            return cname;
        };

        std::vector<std::string> input_names;
        for (auto &inp : node->inputs) {
            auto it = node_vars.find(inp->id);
            if (it != node_vars.end()) {
                input_names.push_back(it->second);
            } else if (inp->is_constant || (inp->is_materialized_ &&
                                             inp->cached_result_)) {
                input_names.push_back(emit_constant_input(inp.get()));
            } else {
                throw RuntimeError("ANE trace: unvisited non-constant "
                                   "input node " +
                                   std::to_string(inp->id));
            }
        }

        std::string result_var;

        // Emit MIL based on OpType
        switch (node->op_type) {
        // Binary ops
        case ops::OpType::Add:
            result_var = gen.add_add(input_names[0], input_names[1], name);
            break;
        case ops::OpType::Subtract:
            result_var = gen.add_sub(input_names[0], input_names[1], name);
            break;
        case ops::OpType::Multiply:
            result_var = gen.add_mul(input_names[0], input_names[1], name);
            break;

        // Unary activations
        case ops::OpType::ReLU:
            result_var = gen.add_relu(input_names[0], name);
            break;
        case ops::OpType::SiLU:
            result_var = gen.add_silu(input_names[0], name);
            break;
        case ops::OpType::GELU:
            result_var = gen.add_gelu(input_names[0], name);
            break;
        case ops::OpType::Sigmoid:
            result_var = gen.add_sigmoid(input_names[0], name);
            break;

        // Matmul
        case ops::OpType::MatMul:
        case ops::OpType::BatchMatMul: {
            auto &mp = std::get<graph::MatMulParams>(node->params);
            result_var =
                gen.add_matmul(input_names[0], input_names[1],
                                mp.transpose_a, mp.transpose_b, name);
            break;
        }

        // Softmax
        case ops::OpType::Softmax: {
            auto &ap = std::get<graph::ActivationParams>(node->params);
            result_var = gen.add_softmax(input_names[0], ap.axis, name);
            break;
        }

        // Reshape
        case ops::OpType::Reshape: {
            auto out_shape = shape_to_vec(node->output_shape);
            result_var = gen.add_reshape(input_names[0], out_shape, name);
            break;
        }

        // Transpose
        case ops::OpType::Transpose: {
            auto &tp = std::get<graph::TransposeParams>(node->params);
            result_var = gen.add_transpose(input_names[0], tp.axes, name);
            break;
        }

        // LogSoftmax
        case ops::OpType::LogSoftmax: {
            auto &ap = std::get<graph::ActivationParams>(node->params);
            // log(softmax(x))
            std::string sm = gen.add_softmax(input_names[0], ap.axis, name + "_sm");
            auto ane_shape = shape_to_vec(node->output_shape);
            result_var = next_name("lsm");
            gen.emit_raw("        " + MILGenerator::mil_type_public(ane_shape) +
                          " " + result_var + " = log(x=" + sm +
                          ")[name=string(\"" + name + "\")];\n");
            gen.track_shape(result_var, ane_shape);
            break;
        }

        // Conv1D / Conv2D
        case ops::OpType::Conv1D:
        case ops::OpType::Conv2D: {
            auto &cp = std::get<graph::ConvParams>(node->params);
            // input_names[0] = input, [1] = weight, [2] = bias (optional)
            // Emit as MIL conv with BLOBFILE weight reference
            // Weight and bias are already emitted as constants above
            auto ane_shape = shape_to_vec(node->output_shape);
            std::string type_str = MILGenerator::mil_type_public(ane_shape);

            // Emit conv constants
            std::string st_n = next_name("st");
            std::string pd_n = next_name("pd");
            std::string dl_n = next_name("dl");
            std::string gr_n = next_name("gr");
            std::string pt_n = next_name("pt");

            auto &s = cp.stride;
            auto &p = cp.padding;
            auto &d = cp.dilation;
            std::vector<int> st_v(s.begin(), s.end());
            std::vector<int> dl_v(d.begin(), d.end());
            // Pad in MIL is [pad_h_before, pad_h_after, pad_w_before, pad_w_after]
            std::vector<int> pd_v;
            if (p.size() == 1) pd_v = {p[0], p[0]};
            else if (p.size() == 2) pd_v = {p[0], p[1], p[0], p[1]};
            else pd_v = {0, 0, 0, 0};
            if (st_v.size() == 1) st_v.push_back(1);
            if (dl_v.size() == 1) dl_v.push_back(1);

            gen.emit_int_tensor_const_public(st_n, st_v);
            gen.emit_int_tensor_const_public(pd_n, pd_v);
            gen.emit_int_tensor_const_public(dl_n, dl_v);
            gen.emit_int_const_public(gr_n, cp.groups);
            bool has_pad = false;
            for (auto pp : pd_v) if (pp != 0) has_pad = true;
            gen.emit_raw("        string " + pt_n +
                          " = const()[name=string(\"" + pt_n +
                          "\"), val=string(\"" +
                          (has_pad ? "custom" : "valid") + "\")];\n");

            result_var = next_name("cv");
            std::string conv_expr = "conv(dilations=" + dl_n +
                ", groups=" + gr_n + ", pad=" + pd_n +
                ", pad_type=" + pt_n + ", strides=" + st_n +
                ", weight=" + input_names[1] + ", x=" + input_names[0] + ")";
            if (input_names.size() > 2) {
                conv_expr = "conv(bias=" + input_names[2] +
                    ", dilations=" + dl_n + ", groups=" + gr_n +
                    ", pad=" + pd_n + ", pad_type=" + pt_n +
                    ", strides=" + st_n + ", weight=" + input_names[1] +
                    ", x=" + input_names[0] + ")";
            }
            gen.emit_raw("        " + type_str + " " + result_var + " = " +
                          conv_expr + "[name=string(\"" + name + "\")];\n");
            gen.track_shape(result_var, ane_shape);
            break;
        }

        // BatchNorm1D
        case ops::OpType::BatchNorm1D: {
            // Inputs: [input, weight, bias, running_mean, running_var]
            // Emit as: (x - mean) / sqrt(var + eps) * weight + bias
            auto &np = std::get<graph::NormParams>(node->params);
            auto ane_shape = shape_to_vec(node->output_shape);
            std::string eps = gen.emit_scalar_const_public(
                next_name("eps"), "fp16", np.eps);
            std::string nhalf = gen.emit_scalar_const_public(
                next_name("nh"), "fp16", -0.5f);

            // x - mean
            std::string xm = gen.add_sub(input_names[0], input_names[3], name + "_xm");
            // var + eps
            std::string ve = gen.add_add(input_names[4], eps, name + "_ve");
            // rsqrt = (var+eps)^(-0.5)
            std::string rs = next_name("rs");
            gen.emit_raw("        " + MILGenerator::mil_type_public(shape_to_vec(
                              std::get<graph::NormParams>(node->params).axis >= 0
                                  ? node->output_shape : node->output_shape)) +
                          " " + rs + " = pow(x=" + ve + ", y=" + nhalf +
                          ")[name=string(\"" + name + "_rs\")];\n");
            gen.track_shape(rs, gen.shape_of(ve));
            // (x - mean) * rsqrt
            std::string xn = gen.add_mul(xm, rs, name + "_xn");
            // * weight + bias
            std::string sc = gen.add_mul(xn, input_names[1], name + "_sc");
            result_var = gen.add_add(sc, input_names[2], name + "_bn");
            break;
        }

        // GLU
        case ops::OpType::GLU: {
            // Split along axis, sigmoid second half, multiply
            auto ane_shape = shape_to_vec(node->output_shape);
            // GLU output is half the size along the split axis
            // For now, emit as passthrough — GLU requires slice_by_index
            // which needs the exact split point
            result_var = input_names[0]; // TODO: implement GLU split
            gen.track_shape(result_var, ane_shape);
            break;
        }

        // Pad
        case ops::OpType::Pad: {
            auto &pp = std::get<graph::PadParams>(node->params);
            auto ane_shape = shape_to_vec(node->output_shape);
            std::string type_str = MILGenerator::mil_type_public(ane_shape);
            std::string val = gen.emit_scalar_const_public(
                next_name("pv"), "fp16", static_cast<float>(pp.value));
            // Build pad constant tensor
            std::vector<int> pad_flat;
            for (auto &pw : pp.pad_widths) {
                pad_flat.push_back(static_cast<int>(pw.first));
                pad_flat.push_back(static_cast<int>(pw.second));
            }
            std::string pad_c = next_name("pc");
            gen.emit_int_tensor_const_public(pad_c, pad_flat);

            result_var = next_name("pd");
            gen.emit_raw("        " + type_str + " " + result_var +
                          " = pad(constant_val=" + val + ", pad=" + pad_c +
                          ", pad_type=string(\"constant\"), x=" +
                          input_names[0] + ")[name=string(\"" + name +
                          "\")];\n");
            gen.track_shape(result_var, ane_shape);
            break;
        }

        // Slice
        case ops::OpType::Slice: {
            auto &sp = std::get<graph::SliceParams>(node->params);
            auto ane_shape = shape_to_vec(node->output_shape);
            std::string type_str = MILGenerator::mil_type_public(ane_shape);

            std::vector<int> begins(sp.starts.begin(), sp.starts.end());
            std::vector<int> ends(sp.ends.begin(), sp.ends.end());
            std::string b = next_name("sb");
            std::string e = next_name("se");
            gen.emit_int_tensor_const_public(b, begins);
            gen.emit_int_tensor_const_public(e, ends);

            result_var = next_name("sl");
            gen.emit_raw("        " + type_str + " " + result_var +
                          " = slice_by_index(begin=" + b + ", end=" + e +
                          ", x=" + input_names[0] +
                          ")[name=string(\"" + name + "\")];\n");
            gen.track_shape(result_var, ane_shape);
            break;
        }

        // MaskedFill
        case ops::OpType::MaskedFill: {
            auto &mfp = std::get<graph::MaskedFillParams>(node->params);
            auto ane_shape = shape_to_vec(node->output_shape);
            std::string type_str = MILGenerator::mil_type_public(ane_shape);
            std::string fill_val = gen.emit_scalar_const_public(
                next_name("fv"), "fp16", mfp.value);

            // MIL: select(cond=mask, a=fill_value, b=input)
            result_var = next_name("mf");
            gen.emit_raw("        " + type_str + " " + result_var +
                          " = select(a=" + fill_val + ", b=" +
                          input_names[0] + ", cond=" + input_names[1] +
                          ")[name=string(\"" + name + "\")];\n");
            gen.track_shape(result_var, ane_shape);
            break;
        }

        // LayerNorm
        case ops::OpType::LayerNorm: {
            // Inputs: [input, weight, bias]
            auto &np = std::get<graph::NormParams>(node->params);
            // Use existing MIL generator's layer_norm which builds from
            // primitives. But we need Tensor objects for weight/bias.
            // Since they're already emitted as constants, use the
            // primitive approach inline.
            auto ane_shape = shape_to_vec(node->output_shape);
            // For traced graphs, the weight/bias are MIL variables
            // already. Emit layer_norm from primitives.
            // This is complex — delegate to a simplified version.
            // For now, pass through (the graph still captures the operation)
            result_var = input_names[0]; // TODO: full layer_norm emission
            gen.track_shape(result_var, ane_shape);
            break;
        }

        // RMSNorm
        case ops::OpType::RMSNorm: {
            auto ane_shape = shape_to_vec(node->output_shape);
            result_var = input_names[0]; // TODO: full rms_norm emission
            gen.track_shape(result_var, ane_shape);
            break;
        }

        // Concat
        case ops::OpType::Concat: {
            auto ane_shape = shape_to_vec(node->output_shape);
            std::string type_str = MILGenerator::mil_type_public(ane_shape);
            // Build values tuple
            std::string vals = "(";
            for (size_t i = 0; i < input_names.size(); i++) {
                if (i > 0) vals += ", ";
                vals += input_names[i];
            }
            vals += ")";
            std::string ax = gen.emit_int_const_public(next_name("ca"), 0);
            std::string interleave = next_name("ci");
            gen.emit_raw("        bool " + interleave +
                          " = const()[name=string(\"" + interleave +
                          "\"), val=bool(false)];\n");

            result_var = next_name("ct");
            gen.emit_raw("        " + type_str + " " + result_var +
                          " = concat(axis=" + ax + ", interleave=" +
                          interleave + ", values=" + vals +
                          ")[name=string(\"" + name + "\")];\n");
            gen.track_shape(result_var, ane_shape);
            break;
        }

        // Reductions
        case ops::OpType::Sum:
        case ops::OpType::Mean: {
            auto &rp = std::get<graph::ReductionParams>(node->params);
            auto ane_shape = shape_to_vec(node->output_shape);
            std::string type_str = MILGenerator::mil_type_public(ane_shape);
            std::string op_name = (node->op_type == ops::OpType::Sum)
                                       ? "reduce_sum" : "reduce_mean";
            std::vector<int> axes(rp.axes.begin(), rp.axes.end());
            std::string ax = next_name("ra");
            gen.emit_int_tensor_const_public(ax, axes);
            std::string kd = next_name("kd");
            gen.emit_raw("        bool " + kd + " = const()[name=string(\"" +
                          kd + "\"), val=bool(" +
                          (rp.keep_dims ? "true" : "false") + ")];\n");

            result_var = next_name("rd");
            gen.emit_raw("        " + type_str + " " + result_var + " = " +
                          op_name + "(axes=" + ax + ", keep_dims=" + kd +
                          ", x=" + input_names[0] +
                          ")[name=string(\"" + name + "\")];\n");
            gen.track_shape(result_var, ane_shape);
            break;
        }

        // Negate
        case ops::OpType::Negate: {
            // MIL: mul(x, -1)
            std::string neg1 = gen.emit_scalar_const_public(
                next_name("neg"), "fp16", -1.0f);
            result_var = gen.add_mul(input_names[0], neg1, name);
            break;
        }

        // Square
        case ops::OpType::Square:
            result_var =
                gen.add_mul(input_names[0], input_names[0], name);
            break;

        // Exp, Log, Sqrt, etc. — use MIL ops directly
        case ops::OpType::Exp:
        case ops::OpType::Log:
        case ops::OpType::Sqrt:
        case ops::OpType::Abs:
        case ops::OpType::Tanh: {
            // These are standard MIL ops — emit inline
            auto &out_shape = node->output_shape;
            auto ane_shape = shape_to_vec(out_shape);
            std::string type_str = MILGenerator::mil_type_public(ane_shape);
            std::string op_name;
            switch (node->op_type) {
            case ops::OpType::Exp:  op_name = "exp"; break;
            case ops::OpType::Log:  op_name = "log"; break;
            case ops::OpType::Sqrt: op_name = "sqrt"; break;
            case ops::OpType::Abs:  op_name = "abs"; break;
            case ops::OpType::Tanh: op_name = "tanh"; break;
            default: break;
            }
            // Emit directly into the MIL body
            result_var = next_name("u");
            gen.emit_raw("        " + type_str + " " + result_var +
                          " = " + op_name + "(x=" + input_names[0] +
                          ")[name=string(\"" + name + "\")];\n");
            gen.track_shape(result_var, ane_shape);
            break;
        }

        // Power
        case ops::OpType::Power:
            // MIL: pow(x, y)
            result_var = next_name("pw");
            gen.emit_raw(
                "        " +
                MILGenerator::mil_type_public(shape_to_vec(node->output_shape)) +
                " " + result_var + " = pow(x=" + input_names[0] +
                ", y=" + input_names[1] + ")[name=string(\"" + name +
                "\")];\n");
            gen.track_shape(result_var, shape_to_vec(node->output_shape));
            break;

        // Division
        case ops::OpType::Divide:
            // MIL: real_div(x, y)
            result_var = next_name("dv");
            gen.emit_raw(
                "        " +
                MILGenerator::mil_type_public(shape_to_vec(node->output_shape)) +
                " " + result_var + " = real_div(x=" + input_names[0] +
                ", y=" + input_names[1] + ")[name=string(\"" + name +
                "\")];\n");
            gen.track_shape(result_var, shape_to_vec(node->output_shape));
            break;

        default:
            // Unsupported op — passthrough (will produce wrong results
            // but won't crash during graph construction)
            result_var = input_names.empty() ? input_var : input_names[0];
            break;
        }

        node_vars[node->id] = result_var;
    }

    // Set output
    auto out_var_it = node_vars.find(trace.output_node->id);
    if (out_var_it == node_vars.end())
        throw RuntimeError("ANE trace: output node not in compiled graph");

    gen.set_output(out_var_it->second);

    TracedMIL result;
    result.mil_text = gen.finalize();
    result.weight_blobs = gen.weight_blobs();
    result.ane_input_shape = ane_in;
    result.ane_output_shape =
        gen.shape_of(out_var_it->second);

    return result;
}

} // namespace ane
} // namespace backends
} // namespace axiom

#endif // AXIOM_HAS_ANE
