#pragma once

#include <cmath>
#include <type_traits>
#include <vector>

#include "axiom/dtype.hpp"
#include "axiom/shape.hpp"
#include "axiom/tensor.hpp"
#include "base.hpp"
#include "binary.hpp"
#include "traits.hpp"
#include "unary.hpp"

// ARM NEON SIMD support
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

namespace axiom {
namespace expr {

// ============================================================================
// SIMD Configuration
// ============================================================================

#ifdef __ARM_NEON
inline constexpr bool kHasNEON = true;
inline constexpr size_t kSimdWidth = 4;  // float32x4_t
#else
inline constexpr bool kHasNEON = false;
inline constexpr size_t kSimdWidth = 1;
#endif

// ============================================================================
// Scalar Operation Application (fallback)
// ============================================================================

template <typename Op, typename T>
inline T apply_binary_op_scalar(T lhs, T rhs) {
    if constexpr (std::is_same_v<Op, AddOp>) {
        return lhs + rhs;
    } else if constexpr (std::is_same_v<Op, SubOp>) {
        return lhs - rhs;
    } else if constexpr (std::is_same_v<Op, MulOp>) {
        return lhs * rhs;
    } else if constexpr (std::is_same_v<Op, DivOp>) {
        return lhs / rhs;
    } else if constexpr (std::is_same_v<Op, MaxOp>) {
        return lhs > rhs ? lhs : rhs;
    } else if constexpr (std::is_same_v<Op, MinOp>) {
        return lhs < rhs ? lhs : rhs;
    } else {
        static_assert(std::is_same_v<Op, AddOp>, "Unsupported binary op");
        return T{};
    }
}

template <typename Op, typename T>
inline T apply_unary_op_scalar(T val) {
    if constexpr (std::is_same_v<Op, NegOp>) {
        return -val;
    } else if constexpr (std::is_same_v<Op, AbsOp>) {
        return std::abs(val);
    } else if constexpr (std::is_same_v<Op, SqrtOp>) {
        return std::sqrt(val);
    } else if constexpr (std::is_same_v<Op, ExpOp>) {
        return std::exp(val);
    } else if constexpr (std::is_same_v<Op, LogOp>) {
        return std::log(val);
    } else if constexpr (std::is_same_v<Op, ReciprocalOp>) {
        return T(1) / val;
    } else if constexpr (std::is_same_v<Op, SquareOp>) {
        return val * val;
    } else if constexpr (std::is_same_v<Op, ReluOp>) {
        return val > T(0) ? val : T(0);
    } else if constexpr (std::is_same_v<Op, SigmoidOp>) {
        return T(1) / (T(1) + std::exp(-val));
    } else if constexpr (std::is_same_v<Op, TanhActivationOp>) {
        return std::tanh(val);
    } else if constexpr (std::is_same_v<Op, GeluOp>) {
        // GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
        return T(0.5) * val * (T(1) + std::erf(val / std::sqrt(T(2))));
    } else if constexpr (std::is_same_v<Op, SiluOp>) {
        // SiLU(x) = x * sigmoid(x)
        return val / (T(1) + std::exp(-val));
    } else if constexpr (std::is_same_v<Op, SinOp>) {
        return std::sin(val);
    } else if constexpr (std::is_same_v<Op, CosOp>) {
        return std::cos(val);
    } else if constexpr (std::is_same_v<Op, TanOp>) {
        return std::tan(val);
    } else if constexpr (std::is_same_v<Op, ErfOp>) {
        return std::erf(val);
    } else if constexpr (std::is_same_v<Op, SignOp>) {
        return (T(0) < val) - (val < T(0));
    } else if constexpr (std::is_same_v<Op, FloorOp>) {
        return std::floor(val);
    } else if constexpr (std::is_same_v<Op, CeilOp>) {
        return std::ceil(val);
    } else if constexpr (std::is_same_v<Op, TruncOp>) {
        return std::trunc(val);
    } else if constexpr (std::is_same_v<Op, RoundOp>) {
        return std::round(val);
    } else if constexpr (std::is_same_v<Op, CbrtOp>) {
        return std::cbrt(val);
    } else {
        static_assert(std::is_same_v<Op, NegOp>, "Unsupported unary op");
        return T{};
    }
}

// ============================================================================
// SIMD Operation Application (NEON)
// ============================================================================

#ifdef __ARM_NEON

template <typename Op>
inline float32x4_t apply_binary_op_simd(float32x4_t lhs, float32x4_t rhs) {
    if constexpr (std::is_same_v<Op, AddOp>) {
        return vaddq_f32(lhs, rhs);
    } else if constexpr (std::is_same_v<Op, SubOp>) {
        return vsubq_f32(lhs, rhs);
    } else if constexpr (std::is_same_v<Op, MulOp>) {
        return vmulq_f32(lhs, rhs);
    } else if constexpr (std::is_same_v<Op, DivOp>) {
        return vdivq_f32(lhs, rhs);
    } else if constexpr (std::is_same_v<Op, MaxOp>) {
        return vmaxq_f32(lhs, rhs);
    } else if constexpr (std::is_same_v<Op, MinOp>) {
        return vminq_f32(lhs, rhs);
    } else {
        // Fallback: process element-by-element
        float l[4], r[4], out[4];
        vst1q_f32(l, lhs);
        vst1q_f32(r, rhs);
        for (int i = 0; i < 4; ++i) {
            out[i] = apply_binary_op_scalar<Op>(l[i], r[i]);
        }
        return vld1q_f32(out);
    }
}

template <typename Op>
inline float32x4_t apply_unary_op_simd(float32x4_t val) {
    if constexpr (std::is_same_v<Op, NegOp>) {
        return vnegq_f32(val);
    } else if constexpr (std::is_same_v<Op, AbsOp>) {
        return vabsq_f32(val);
    } else if constexpr (std::is_same_v<Op, SquareOp>) {
        return vmulq_f32(val, val);
    } else if constexpr (std::is_same_v<Op, ReluOp>) {
        return vmaxq_f32(val, vdupq_n_f32(0.0f));
    } else if constexpr (std::is_same_v<Op, ReciprocalOp>) {
        // Use NEON reciprocal estimate + Newton-Raphson
        float32x4_t est = vrecpeq_f32(val);
        est = vmulq_f32(vrecpsq_f32(val, est), est);
        return est;
    } else if constexpr (std::is_same_v<Op, SqrtOp>) {
        return vsqrtq_f32(val);
    } else {
        // Fallback: process element-by-element for transcendentals
        float v[4], out[4];
        vst1q_f32(v, val);
        for (int i = 0; i < 4; ++i) {
            out[i] = apply_unary_op_scalar<Op>(v[i]);
        }
        return vld1q_f32(out);
    }
}

#endif // __ARM_NEON

// ============================================================================
// SIMD Expression Evaluation
// Evaluates expression at index i, returning SIMD vector
// ============================================================================

#ifdef __ARM_NEON

// Forward declaration
template <typename Expr>
inline float32x4_t eval_simd(const Expr& expr, size_t i);

// TensorRef: load 4 elements
template <>
inline float32x4_t eval_simd<TensorRef>(const TensorRef& ref, size_t i) {
    return vld1q_f32(ref.tensor().typed_data<float>() + i);
}

// ScalarExpr<float>: broadcast scalar to vector
template <>
inline float32x4_t eval_simd<ScalarExpr<float>>(const ScalarExpr<float>& scalar, size_t) {
    return vdupq_n_f32(scalar.value());
}

// ScalarExpr<double>: broadcast scalar to vector (cast to float)
template <>
inline float32x4_t eval_simd<ScalarExpr<double>>(const ScalarExpr<double>& scalar, size_t) {
    return vdupq_n_f32(static_cast<float>(scalar.value()));
}

// ScalarExpr<int>: broadcast scalar to vector (cast to float)
template <>
inline float32x4_t eval_simd<ScalarExpr<int>>(const ScalarExpr<int>& scalar, size_t) {
    return vdupq_n_f32(static_cast<float>(scalar.value()));
}

// BinaryExpr: evaluate both sides and apply SIMD op
template <typename Op, typename LHS, typename RHS>
inline float32x4_t eval_simd(const BinaryExpr<Op, LHS, RHS>& expr, size_t i) {
    float32x4_t lhs = eval_simd(expr.lhs(), i);
    float32x4_t rhs = eval_simd(expr.rhs(), i);
    return apply_binary_op_simd<Op>(lhs, rhs);
}

// UnaryExpr: evaluate operand and apply SIMD op
template <typename Op, typename Operand>
inline float32x4_t eval_simd(const UnaryExpr<Op, Operand>& expr, size_t i) {
    float32x4_t operand = eval_simd(expr.operand(), i);
    return apply_unary_op_simd<Op>(operand);
}

#endif // __ARM_NEON

// ============================================================================
// Scalar Expression Evaluation (fallback and tail handling)
// ============================================================================

template <typename Expr>
inline float eval_scalar(const Expr& expr, size_t i);

template <>
inline float eval_scalar<TensorRef>(const TensorRef& ref, size_t i) {
    return ref.tensor().typed_data<float>()[i];
}

template <>
inline float eval_scalar<ScalarExpr<float>>(const ScalarExpr<float>& scalar, size_t) {
    return scalar.value();
}

template <>
inline float eval_scalar<ScalarExpr<double>>(const ScalarExpr<double>& scalar, size_t) {
    return static_cast<float>(scalar.value());
}

template <>
inline float eval_scalar<ScalarExpr<int>>(const ScalarExpr<int>& scalar, size_t) {
    return static_cast<float>(scalar.value());
}

template <typename Op, typename LHS, typename RHS>
inline float eval_scalar(const BinaryExpr<Op, LHS, RHS>& expr, size_t i) {
    float lhs = eval_scalar(expr.lhs(), i);
    float rhs = eval_scalar(expr.rhs(), i);
    return apply_binary_op_scalar<Op>(lhs, rhs);
}

template <typename Op, typename Operand>
inline float eval_scalar(const UnaryExpr<Op, Operand>& expr, size_t i) {
    float operand = eval_scalar(expr.operand(), i);
    return apply_unary_op_scalar<Op>(operand);
}

// ============================================================================
// Contiguity Checking
// ============================================================================

template <typename Expr>
void collectTensorRefsForContiguityCheck(const Expr& expr,
                                         std::vector<const Tensor*>& refs) {
    if constexpr (is_tensor_ref_v<Expr>) {
        refs.push_back(expr.tensor_ptr());
    } else if constexpr (is_scalar_expr_v<Expr>) {
        // No tensor refs
    } else if constexpr (is_binary_expr_v<Expr>) {
        collectTensorRefsForContiguityCheck(expr.lhs(), refs);
        collectTensorRefsForContiguityCheck(expr.rhs(), refs);
    } else if constexpr (is_unary_expr_v<Expr>) {
        collectTensorRefsForContiguityCheck(expr.operand(), refs);
    } else if constexpr (is_matmul_expr_v<Expr>) {
        collectTensorRefsForContiguityCheck(expr.lhs(), refs);
        collectTensorRefsForContiguityCheck(expr.rhs(), refs);
    }
}

template <typename Expr>
bool allTensorsContiguousForFusion(const Expr& expr) {
    std::vector<const Tensor*> refs;
    collectTensorRefsForContiguityCheck(expr, refs);
    for (const Tensor* t : refs) {
        if (!t->is_contiguous()) {
            return false;
        }
    }
    return true;
}

// ============================================================================
// CPU Fused Evaluation with SIMD
// ============================================================================

template <typename Expr>
Tensor cpu_eval_fused(const Expr& expr) {
    Shape result_shape = expr.shape();
    DType result_dtype = expr.dtype();

    // Only float32 fusion is currently supported with SIMD
    if (result_dtype != DType::Float32) {
        // Fall back to eager evaluation for other types
        return expr.eval_eager();
    }

    Tensor result(result_shape, result_dtype, Device::CPU);
    float* out = result.typed_data<float>();
    const size_t n = result.size();

#ifdef __ARM_NEON
    // SIMD vectorized loop - process 4 elements at a time
    constexpr size_t width = 4;
    size_t i = 0;

    // Main SIMD loop
    for (; i + width <= n; i += width) {
        float32x4_t v = eval_simd(expr, i);
        vst1q_f32(out + i, v);
    }

    // Scalar tail for remaining elements
    for (; i < n; ++i) {
        out[i] = eval_scalar(expr, i);
    }
#else
    // Scalar fallback when NEON not available
    for (size_t i = 0; i < n; ++i) {
        out[i] = eval_scalar(expr, i);
    }
#endif

    return result;
}

// ============================================================================
// Fusion Decision Logic
//
// IMPORTANT: CPU fusion is currently DISABLED because Apple's Accelerate
// framework (vDSP) outperforms our SIMD fusion by ~7x.
//
// Benchmark results (1M float32 elements, expression: ((a+b)*c)-d):
// - Our SIMD fusion: ~2.8ms/operation
// - Accelerate step-by-step: ~0.4ms/operation
//
// Why Accelerate is faster despite doing 3 separate passes:
// 1. vDSP routines are hand-tuned for Apple Silicon cache hierarchy
// 2. They use optimal register allocation and loop unrolling
// 3. They may use multiple SIMD units in parallel
// 4. Memory prefetching is optimized for sequential access patterns
//
// Our fusion, while reducing memory traffic (20MB vs 36MB for the above),
// cannot overcome the per-element overhead from:
// - Template recursion at each SIMD chunk
// - Suboptimal register usage from recursive evaluation
// - Less aggressive compiler optimizations on template code
//
// To compete with Accelerate, we would need:
// 1. JIT compilation of expression trees (like TensorFlow XLA)
// 2. Or: compile-time code generation using heavy template metaprogramming
//    to produce a flat loop (like Eigen's sophisticated evaluators)
//
// For now, all CPU expressions use the eager evaluation path which
// benefits from Apple Accelerate's industry-leading implementations.
// ============================================================================

template <typename Expr>
bool shouldFuseCPU(const Expr&) {
    // CPU fusion disabled - Accelerate is faster (see above)
    return false;
}

} // namespace expr
} // namespace axiom
