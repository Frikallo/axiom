#include "axiom/graph/fused_patterns.hpp"
#include "axiom/graph/op_traits.hpp"
#include "backends/cpu/simd/simd_dispatch.hpp"

namespace axiom {
namespace graph {

// ============================================================================
// Pattern Detection
// ============================================================================

FusedPattern detect_pattern(const FusedOpChain &chain) {
    // 2-op patterns (Binary + Unary)
    if (chain.ops.size() == 2) {
        auto op0 = chain.ops[0];
        auto op1 = chain.ops[1];

        // Add + activation
        if (op0 == ops::OpType::Add && op1 == ops::OpType::ReLU)
            return FusedPattern::AddReLU;
        if (op0 == ops::OpType::Add && op1 == ops::OpType::Square)
            return FusedPattern::AddSquare;
        if (op0 == ops::OpType::Add && op1 == ops::OpType::Sigmoid)
            return FusedPattern::AddSigmoid;

        // Sub + activation
        if (op0 == ops::OpType::Subtract && op1 == ops::OpType::Abs)
            return FusedPattern::SubAbs;
        if (op0 == ops::OpType::Subtract && op1 == ops::OpType::Square)
            return FusedPattern::SubSquare;

        // Mul + activation
        if (op0 == ops::OpType::Multiply && op1 == ops::OpType::ReLU)
            return FusedPattern::MulReLU;
        if (op0 == ops::OpType::Multiply && op1 == ops::OpType::Sigmoid)
            return FusedPattern::MulSigmoid;
    }

    // 3-op patterns (Ternary)
    if (chain.ops.size() == 3) {
        auto op0 = chain.ops[0];
        auto op1 = chain.ops[1];
        auto op2 = chain.ops[2];

        // Scale-shift-activation: mul -> add -> activation
        if (op0 == ops::OpType::Multiply && op1 == ops::OpType::Add) {
            if (op2 == ops::OpType::ReLU)
                return FusedPattern::ScaleShiftReLU;
        }

        // Add/Sub -> Mul -> activation
        if (op0 == ops::OpType::Add && op1 == ops::OpType::Multiply &&
            op2 == ops::OpType::ReLU)
            return FusedPattern::AddMulReLU;
        if (op0 == ops::OpType::Subtract && op1 == ops::OpType::Multiply &&
            op2 == ops::OpType::Abs)
            return FusedPattern::SubMulAbs;
    }

    // Check for MulAdd/MulSub (2 ops but ternary inputs)
    if (chain.ops.size() == 2 && chain.input_nodes.size() >= 3) {
        if (chain.ops[0] == ops::OpType::Multiply &&
            chain.ops[1] == ops::OpType::Add)
            return FusedPattern::MulAdd;
        if (chain.ops[0] == ops::OpType::Multiply &&
            chain.ops[1] == ops::OpType::Subtract)
            return FusedPattern::MulSub;
    }

    return FusedPattern::None;
}

// ============================================================================
// SIMD Dtype Check
// ============================================================================

bool is_fused_simd_dtype(DType dtype) {
    return dtype == DType::Float32 || dtype == DType::Float64 ||
           dtype == DType::Int32 || dtype == DType::Int64;
}

// ============================================================================
// Integer Pattern Support
// ============================================================================

bool pattern_supports_integer(FusedPattern pattern) {
    switch (pattern) {
    case FusedPattern::AddReLU:
    case FusedPattern::SubAbs:
    case FusedPattern::AddSquare:
    case FusedPattern::SubSquare:
    case FusedPattern::MulAdd:
        return true;
    default:
        return false;
    }
}

// ============================================================================
// SIMD Dispatch
// ============================================================================

bool dispatch_fused_pattern(FusedPattern pattern,
                            const std::vector<Tensor> &inputs, Tensor &result,
                            size_t off, size_t cnt) {
    if (inputs.empty())
        return false;

    DType dtype = inputs[0].dtype();
    if (cnt == 0)
        cnt = inputs[0].size();

    bool is_integer = (dtype == DType::Int32 || dtype == DType::Int64);
    if (is_integer && !pattern_supports_integer(pattern))
        return false;

#define DISPATCH_BINARY(PATTERN, DISPATCH_FN)                                  \
    case FusedPattern::PATTERN:                                                \
        if (inputs.size() >= 2) {                                              \
            if (dtype == DType::Float32) {                                     \
                simd::DISPATCH_FN<float>(inputs[0].typed_data<float>() + off,  \
                                         inputs[1].typed_data<float>() + off,  \
                                         result.typed_data<float>() + off,     \
                                         cnt);                                 \
                return true;                                                   \
            } else if (dtype == DType::Float64) {                              \
                simd::DISPATCH_FN<double>(                                     \
                    inputs[0].typed_data<double>() + off,                      \
                    inputs[1].typed_data<double>() + off,                      \
                    result.typed_data<double>() + off, cnt);                   \
                return true;                                                   \
            }                                                                  \
        }                                                                      \
        break

#define DISPATCH_BINARY_WITH_INT(PATTERN, DISPATCH_FN)                         \
    case FusedPattern::PATTERN:                                                \
        if (inputs.size() >= 2) {                                              \
            if (dtype == DType::Float32) {                                     \
                simd::DISPATCH_FN<float>(inputs[0].typed_data<float>() + off,  \
                                         inputs[1].typed_data<float>() + off,  \
                                         result.typed_data<float>() + off,     \
                                         cnt);                                 \
                return true;                                                   \
            } else if (dtype == DType::Float64) {                              \
                simd::DISPATCH_FN<double>(                                     \
                    inputs[0].typed_data<double>() + off,                      \
                    inputs[1].typed_data<double>() + off,                      \
                    result.typed_data<double>() + off, cnt);                   \
                return true;                                                   \
            } else if (dtype == DType::Int32) {                                \
                simd::DISPATCH_FN<int32_t>(                                    \
                    inputs[0].typed_data<int32_t>() + off,                     \
                    inputs[1].typed_data<int32_t>() + off,                     \
                    result.typed_data<int32_t>() + off, cnt);                  \
                return true;                                                   \
            } else if (dtype == DType::Int64) {                                \
                simd::DISPATCH_FN<int64_t>(                                    \
                    inputs[0].typed_data<int64_t>() + off,                     \
                    inputs[1].typed_data<int64_t>() + off,                     \
                    result.typed_data<int64_t>() + off, cnt);                  \
                return true;                                                   \
            }                                                                  \
        }                                                                      \
        break

#define DISPATCH_TERNARY(PATTERN, DISPATCH_FN)                                 \
    case FusedPattern::PATTERN:                                                \
        if (inputs.size() >= 3) {                                              \
            if (dtype == DType::Float32) {                                     \
                simd::DISPATCH_FN<float>(inputs[0].typed_data<float>() + off,  \
                                         inputs[1].typed_data<float>() + off,  \
                                         inputs[2].typed_data<float>() + off,  \
                                         result.typed_data<float>() + off,     \
                                         cnt);                                 \
                return true;                                                   \
            } else if (dtype == DType::Float64) {                              \
                simd::DISPATCH_FN<double>(                                     \
                    inputs[0].typed_data<double>() + off,                      \
                    inputs[1].typed_data<double>() + off,                      \
                    inputs[2].typed_data<double>() + off,                      \
                    result.typed_data<double>() + off, cnt);                   \
                return true;                                                   \
            }                                                                  \
        }                                                                      \
        break

#define DISPATCH_TERNARY_WITH_INT(PATTERN, DISPATCH_FN)                        \
    case FusedPattern::PATTERN:                                                \
        if (inputs.size() >= 3) {                                              \
            if (dtype == DType::Float32) {                                     \
                simd::DISPATCH_FN<float>(inputs[0].typed_data<float>() + off,  \
                                         inputs[1].typed_data<float>() + off,  \
                                         inputs[2].typed_data<float>() + off,  \
                                         result.typed_data<float>() + off,     \
                                         cnt);                                 \
                return true;                                                   \
            } else if (dtype == DType::Float64) {                              \
                simd::DISPATCH_FN<double>(                                     \
                    inputs[0].typed_data<double>() + off,                      \
                    inputs[1].typed_data<double>() + off,                      \
                    inputs[2].typed_data<double>() + off,                      \
                    result.typed_data<double>() + off, cnt);                   \
                return true;                                                   \
            } else if (dtype == DType::Int32) {                                \
                simd::DISPATCH_FN<int32_t>(                                    \
                    inputs[0].typed_data<int32_t>() + off,                     \
                    inputs[1].typed_data<int32_t>() + off,                     \
                    inputs[2].typed_data<int32_t>() + off,                     \
                    result.typed_data<int32_t>() + off, cnt);                  \
                return true;                                                   \
            } else if (dtype == DType::Int64) {                                \
                simd::DISPATCH_FN<int64_t>(                                    \
                    inputs[0].typed_data<int64_t>() + off,                     \
                    inputs[1].typed_data<int64_t>() + off,                     \
                    inputs[2].typed_data<int64_t>() + off,                     \
                    result.typed_data<int64_t>() + off, cnt);                  \
                return true;                                                   \
            }                                                                  \
        }                                                                      \
        break

    switch (pattern) {
        DISPATCH_BINARY_WITH_INT(AddReLU, dispatch_fused_add_relu);
        DISPATCH_BINARY_WITH_INT(SubAbs, dispatch_fused_sub_abs);
        DISPATCH_BINARY_WITH_INT(AddSquare, dispatch_fused_add_square);
        DISPATCH_BINARY_WITH_INT(SubSquare, dispatch_fused_sub_square);
        DISPATCH_BINARY(MulReLU, dispatch_fused_mul_relu);
        DISPATCH_BINARY(AddSigmoid, dispatch_fused_add_sigmoid);
        DISPATCH_BINARY(MulSigmoid, dispatch_fused_mul_sigmoid);
        DISPATCH_TERNARY_WITH_INT(MulAdd, dispatch_fused_mul_add);
        DISPATCH_TERNARY(MulSub, dispatch_fused_mul_sub);
        DISPATCH_TERNARY(ScaleShiftReLU, dispatch_fused_scale_shift_relu);
        DISPATCH_TERNARY(AddMulReLU, dispatch_fused_add_mul_relu);
        DISPATCH_TERNARY(SubMulAbs, dispatch_fused_sub_mul_abs);
    default:
        break;
    }

#undef DISPATCH_BINARY
#undef DISPATCH_BINARY_WITH_INT
#undef DISPATCH_TERNARY
#undef DISPATCH_TERNARY_WITH_INT

    return false;
}

} // namespace graph
} // namespace axiom
