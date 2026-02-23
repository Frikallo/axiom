#include "nvrtc_codegen.hpp"

#ifdef AXIOM_CUDA_SUPPORT

#include <functional>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace axiom {
namespace backends {
namespace cuda {

// ============================================================================
// Supported op/dtype queries
// ============================================================================

bool is_nvrtc_supported_op(ops::OpType op) {
    using Op = ops::OpType;
    switch (op) {
    // Binary arithmetic
    case Op::Add:
    case Op::Subtract:
    case Op::Multiply:
    case Op::Divide:
    case Op::Power:
    case Op::Modulo:
    // Comparison
    case Op::Equal:
    case Op::NotEqual:
    case Op::Less:
    case Op::LessEqual:
    case Op::Greater:
    case Op::GreaterEqual:
    // Logical
    case Op::LogicalAnd:
    case Op::LogicalOr:
    case Op::LogicalXor:
    case Op::LogicalNot:
    // Min/Max
    case Op::Maximum:
    case Op::Minimum:
    case Op::Atan2:
    case Op::Hypot:
    // Unary math
    case Op::Negate:
    case Op::Abs:
    case Op::Sqrt:
    case Op::Exp:
    case Op::Log:
    case Op::Sin:
    case Op::Cos:
    case Op::Tan:
    case Op::Erf:
    case Op::Sign:
    case Op::Floor:
    case Op::Ceil:
    case Op::Trunc:
    case Op::Round:
    case Op::Reciprocal:
    case Op::Square:
    case Op::Cbrt:
    // Testing
    case Op::IsNaN:
    case Op::IsInf:
    case Op::IsFinite:
    // Activations
    case Op::ReLU:
    case Op::LeakyReLU:
    case Op::SiLU:
    case Op::Sigmoid:
    case Op::Tanh:
    case Op::GELU:
        return true;
    default:
        return false;
    }
}

bool is_nvrtc_supported_dtype(DType dtype) {
    switch (dtype) {
    case DType::Float32:
    case DType::Float64:
    case DType::Int32:
    case DType::Int64:
    case DType::Int16:
    case DType::Int8:
        return true;
    default:
        return false;
    }
}

// ============================================================================
// Internal helpers
// ============================================================================

namespace {

// Returns the CUDA C type name for a dtype.
const char *cuda_type_name(DType dtype) {
    switch (dtype) {
    case DType::Float32:
        return "float";
    case DType::Float64:
        return "double";
    case DType::Int32:
        return "int";
    case DType::Int64:
        return "long long";
    case DType::Int16:
        return "short";
    case DType::Int8:
        return "signed char";
    default:
        return "float";
    }
}

bool is_float_dtype(DType dtype) {
    return dtype == DType::Float32 || dtype == DType::Float64;
}

bool is_double(DType dtype) { return dtype == DType::Float64; }

bool is_binary_op(ops::OpType op) {
    using Op = ops::OpType;
    switch (op) {
    case Op::Add:
    case Op::Subtract:
    case Op::Multiply:
    case Op::Divide:
    case Op::Power:
    case Op::Modulo:
    case Op::Equal:
    case Op::NotEqual:
    case Op::Less:
    case Op::LessEqual:
    case Op::Greater:
    case Op::GreaterEqual:
    case Op::LogicalAnd:
    case Op::LogicalOr:
    case Op::LogicalXor:
    case Op::Maximum:
    case Op::Minimum:
    case Op::Atan2:
    case Op::Hypot:
        return true;
    default:
        return false;
    }
}

// Emit the C expression for a binary op.
// `a` and `b` are the operand expressions, `T` is the type name.
std::string binary_expr(ops::OpType op, const std::string &a,
                        const std::string &b, const char *T, DType dtype) {
    using Op = ops::OpType;
    bool dbl = is_double(dtype);
    switch (op) {
    case Op::Add:
        return "(" + a + " + " + b + ")";
    case Op::Subtract:
        return "(" + a + " - " + b + ")";
    case Op::Multiply:
        return "(" + a + " * " + b + ")";
    case Op::Divide:
        return "(" + a + " / " + b + ")";
    case Op::Power:
        return std::string(dbl ? "pow" : "powf") + "(" + a + ", " + b + ")";
    case Op::Modulo:
        if (is_float_dtype(dtype))
            return std::string(dbl ? "fmod" : "fmodf") + "(" + a + ", " + b +
                   ")";
        return "(" + a + " % " + b + ")";
    case Op::Equal:
        return "((" + std::string(T) + ")((" + a + " == " + b + ") ? 1 : 0))";
    case Op::NotEqual:
        return "((" + std::string(T) + ")((" + a + " != " + b + ") ? 1 : 0))";
    case Op::Less:
        return "((" + std::string(T) + ")((" + a + " < " + b + ") ? 1 : 0))";
    case Op::LessEqual:
        return "((" + std::string(T) + ")((" + a + " <= " + b + ") ? 1 : 0))";
    case Op::Greater:
        return "((" + std::string(T) + ")((" + a + " > " + b + ") ? 1 : 0))";
    case Op::GreaterEqual:
        return "((" + std::string(T) + ")((" + a + " >= " + b + ") ? 1 : 0))";
    case Op::LogicalAnd:
        return "((" + std::string(T) + ")(((" + a + " != (" + T + ")0) && (" +
               b + " != (" + T + ")0)) ? 1 : 0))";
    case Op::LogicalOr:
        return "((" + std::string(T) + ")(((" + a + " != (" + T + ")0) || (" +
               b + " != (" + T + ")0)) ? 1 : 0))";
    case Op::LogicalXor:
        return "((" + std::string(T) + ")(((" + a + " != (" + T + ")0) != (" +
               b + " != (" + T + ")0)) ? 1 : 0))";
    case Op::Maximum:
        if (is_float_dtype(dtype))
            return std::string(dbl ? "fmax" : "fmaxf") + "(" + a + ", " + b +
                   ")";
        return "((" + a + " > " + b + ") ? " + a + " : " + b + ")";
    case Op::Minimum:
        if (is_float_dtype(dtype))
            return std::string(dbl ? "fmin" : "fminf") + "(" + a + ", " + b +
                   ")";
        return "((" + a + " < " + b + ") ? " + a + " : " + b + ")";
    case Op::Atan2:
        return std::string(dbl ? "atan2" : "atan2f") + "(" + a + ", " + b +
               ")";
    case Op::Hypot:
        return std::string(dbl ? "hypot" : "hypotf") + "(" + a + ", " + b +
               ")";
    default:
        return a;
    }
}

// Emit the C expression for a unary op.
std::string unary_expr(ops::OpType op, const std::string &a, const char *T,
                       DType dtype) {
    using Op = ops::OpType;
    bool dbl = is_double(dtype);
    switch (op) {
    case Op::Negate:
        return "(-" + a + ")";
    case Op::Abs:
        if (is_float_dtype(dtype))
            return std::string(dbl ? "fabs" : "fabsf") + "(" + a + ")";
        return "((" + a + " < 0) ? -" + a + " : " + a + ")";
    case Op::Sqrt:
        return std::string(dbl ? "sqrt" : "sqrtf") + "(" + a + ")";
    case Op::Exp:
        return std::string(dbl ? "exp" : "__expf") + "(" + a + ")";
    case Op::Log:
        return std::string(dbl ? "log" : "__logf") + "(" + a + ")";
    case Op::Sin:
        return std::string(dbl ? "sin" : "__sinf") + "(" + a + ")";
    case Op::Cos:
        return std::string(dbl ? "cos" : "__cosf") + "(" + a + ")";
    case Op::Tan:
        return std::string(dbl ? "tan" : "__tanf") + "(" + a + ")";
    case Op::Erf:
        return std::string(dbl ? "erf" : "erff") + "(" + a + ")";
    case Op::Tanh:
        return std::string(dbl ? "tanh" : "tanhf") + "(" + a + ")";
    case Op::Sign:
        return "((" + std::string(T) + ")((" + a + " > (" + T + ")0) ? 1 : ((" +
               a + " < (" + T + ")0) ? -1 : 0)))";
    case Op::Floor:
        return std::string(dbl ? "floor" : "floorf") + "(" + a + ")";
    case Op::Ceil:
        return std::string(dbl ? "ceil" : "ceilf") + "(" + a + ")";
    case Op::Trunc:
        return std::string(dbl ? "trunc" : "truncf") + "(" + a + ")";
    case Op::Round:
        return std::string(dbl ? "rint" : "rintf") + "(" + a + ")";
    case Op::Reciprocal:
        return "((" + std::string(T) + ")1 / " + a + ")";
    case Op::Square:
        return "(" + a + " * " + a + ")";
    case Op::Cbrt:
        return std::string(dbl ? "cbrt" : "cbrtf") + "(" + a + ")";
    case Op::IsNaN:
        return "((" + std::string(T) + ")(isnan(" + a + ") ? 1 : 0))";
    case Op::IsInf:
        return "((" + std::string(T) + ")(isinf(" + a + ") ? 1 : 0))";
    case Op::IsFinite:
        return "((" + std::string(T) + ")(isfinite(" + a + ") ? 1 : 0))";
    case Op::LogicalNot:
        return "((" + std::string(T) + ")((" + a + " == (" + T + ")0) ? 1 : 0))";
    case Op::ReLU:
        return "(" + a + " > (" + std::string(T) + ")0 ? " + a + " : (" + T +
               ")0)";
    case Op::LeakyReLU:
        return "(" + a + " > (" + std::string(T) + ")0 ? " + a + " : " + a +
               " * (" + T + ")0.01)";
    case Op::Sigmoid:
        return "((" + std::string(T) + ")1 / ((" + T + ")1 + " +
               std::string(dbl ? "exp" : "expf") + "(-" + a + ")))";
    case Op::SiLU:
        return "(" + a + " / ((" + std::string(T) + ")1 + " +
               std::string(dbl ? "exp" : "expf") + "(-" + a + ")))";
    case Op::GELU: {
        // Tanh approximation: 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
        std::string tf = dbl ? "tanh" : "tanhf";
        std::string sq = dbl ? "sqrt" : "sqrtf";
        return "((" + std::string(T) + ")0.5 * " + a + " * ((" + T +
               ")1 + " + tf + "(" + sq + "((" + T + ")0.6366197723675814) * (" +
               a + " + (" + T + ")0.044715 * " + a + " * " + a + " * " + a +
               "))))";
    }
    default:
        return a;
    }
}

// Hash the op chain + dtype into a short hex string for the kernel name.
size_t spec_hash(const FusedKernelSpec &spec) {
    size_t h = std::hash<int>{}(static_cast<int>(spec.compute_dtype));
    for (auto op : spec.op_chain)
        h ^= std::hash<int>{}(static_cast<int>(op)) + 0x9e3779b9 + (h << 6) +
             (h >> 2);
    if (spec.needs_broadcast)
        h ^= 0xdeadbeef;
    return h;
}

} // anonymous namespace

// ============================================================================
// Code generation
// ============================================================================

GeneratedKernel generate_fused_kernel(const FusedKernelSpec &spec) {
    const char *T = cuda_type_name(spec.compute_dtype);
    std::string entry =
        "fused_" + std::to_string(spec_hash(spec));

    std::ostringstream src;

    // CUDA device code has math builtins (sqrtf, isnan, etc.) — no include needed.

    src << "extern \"C\" __global__ void " << entry << "(\n";

    // Input pointers
    for (size_t i = 0; i < spec.num_external_inputs; ++i) {
        src << "    const " << T << "* __restrict__ in" << i << ",\n";
    }
    src << "    " << T << "* __restrict__ out,\n";
    src << "    unsigned long long n";

    if (spec.needs_broadcast) {
        src << ",\n    const long long* __restrict__ strides,\n";
        src << "    const long long* __restrict__ out_shape,\n";
        src << "    int ndim";
    }

    src << ") {\n";
    src << "    unsigned long long gid = "
           "(unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;\n";
    src << "    if (gid >= n) return;\n\n";

    // Compute coordinates for broadcast indexing
    if (spec.needs_broadcast) {
        src << "    // Decompose flat index into N-D coordinates\n";
        src << "    long long coord[8];\n";
        src << "    unsigned long long tmp = gid;\n";
        src << "    for (int d = ndim - 1; d >= 0; --d) {\n";
        src << "        coord[d] = (long long)(tmp % "
               "(unsigned long long)out_shape[d]);\n";
        src << "        tmp /= (unsigned long long)out_shape[d];\n";
        src << "    }\n\n";

        // Compute per-input linear index from strides
        for (size_t i = 0; i < spec.num_external_inputs; ++i) {
            src << "    long long idx_" << i << " = 0;\n";
            src << "    for (int d = 0; d < ndim; ++d) idx_" << i
                << " += coord[d] * strides[" << i << " * ndim + d];\n";
            src << "    " << T << " v" << i << " = in" << i << "[idx_" << i
                << "];\n";
        }
    } else {
        // Flat indexing — all inputs use gid
        for (size_t i = 0; i < spec.num_external_inputs; ++i) {
            src << "    " << T << " v" << i << " = in" << i << "[gid];\n";
        }
    }
    src << "\n";

    // Emit chain operations
    std::string prev_var;
    for (size_t i = 0; i < spec.op_chain.size(); ++i) {
        ops::OpType op = spec.op_chain[i];
        std::string result_var = "t" + std::to_string(i);
        const auto &slots = spec.input_slot_indices[i];

        if (is_binary_op(op)) {
            // Resolve the two operands
            std::string operand_a =
                (slots.size() > 0 && slots[0] >= 0)
                    ? "v" + std::to_string(slots[0])
                    : prev_var;
            std::string operand_b =
                (slots.size() > 1 && slots[1] >= 0)
                    ? "v" + std::to_string(slots[1])
                    : prev_var;
            src << "    " << T << " " << result_var << " = "
                << binary_expr(op, operand_a, operand_b, T, spec.compute_dtype)
                << ";\n";
        } else {
            // Unary — single operand
            std::string operand =
                (slots.size() > 0 && slots[0] >= 0)
                    ? "v" + std::to_string(slots[0])
                    : prev_var;
            src << "    " << T << " " << result_var << " = "
                << unary_expr(op, operand, T, spec.compute_dtype) << ";\n";
        }
        prev_var = result_var;
    }

    src << "    out[gid] = " << prev_var << ";\n";
    src << "}\n";

    return {src.str(), entry};
}

} // namespace cuda
} // namespace backends
} // namespace axiom

#endif
