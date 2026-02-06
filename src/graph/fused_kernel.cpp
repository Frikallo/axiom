#include "axiom/graph/fused_kernel.hpp"
#include "backends/cpu/simd/simd_dispatch.hpp"

namespace axiom {
namespace graph {

// ============================================================================
// Typed wrapper templates: cast void* to typed pointers and call SIMD
// ============================================================================

template <typename T, void (*Fn)(const T *, T *, size_t)>
static void unary_wrap(const void *in, void *out, size_t n) {
    Fn(static_cast<const T *>(in), static_cast<T *>(out), n);
}

template <typename T, void (*Fn)(const T *, const T *, T *, size_t)>
static void binary_wrap(const void *a, const void *b, void *out, size_t n) {
    Fn(static_cast<const T *>(a), static_cast<const T *>(b),
       static_cast<T *>(out), n);
}

// ============================================================================
// Unary dispatch table
// ============================================================================

UnaryFn get_unary_fn(ops::OpType op, DType dtype) {
    using ops::OpType;

#define UNARY_CASE(OP, DISPATCH)                                               \
    case OpType::OP:                                                           \
        if (dtype == DType::Float32)                                           \
            return unary_wrap<float, simd::DISPATCH<float>>;                   \
        if (dtype == DType::Float64)                                           \
            return unary_wrap<double, simd::DISPATCH<double>>;                 \
        return nullptr

#define UNARY_CASE_INT(OP, DISPATCH)                                           \
    case OpType::OP:                                                           \
        if (dtype == DType::Float32)                                           \
            return unary_wrap<float, simd::DISPATCH<float>>;                   \
        if (dtype == DType::Float64)                                           \
            return unary_wrap<double, simd::DISPATCH<double>>;                 \
        if (dtype == DType::Int32)                                             \
            return unary_wrap<int32_t, simd::DISPATCH<int32_t>>;               \
        if (dtype == DType::Int64)                                             \
            return unary_wrap<int64_t, simd::DISPATCH<int64_t>>;               \
        return nullptr

    switch (op) {
        UNARY_CASE_INT(Negate, dispatch_unary_neg);
        UNARY_CASE_INT(Abs, dispatch_unary_abs);
        UNARY_CASE(Sqrt, dispatch_unary_sqrt);
        UNARY_CASE(Exp, dispatch_unary_exp);
        UNARY_CASE(Log, dispatch_unary_log);
        UNARY_CASE(Sin, dispatch_unary_sin);
        UNARY_CASE(Cos, dispatch_unary_cos);
        UNARY_CASE(Tan, dispatch_unary_tan);
        UNARY_CASE(Tanh, dispatch_unary_tanh);
        UNARY_CASE(Erf, dispatch_unary_erf);
        UNARY_CASE(Cbrt, dispatch_unary_cbrt);
        UNARY_CASE_INT(Square, dispatch_unary_square);
        UNARY_CASE(Reciprocal, dispatch_unary_reciprocal);
        UNARY_CASE_INT(Sign, dispatch_unary_sign);
        UNARY_CASE(Floor, dispatch_unary_floor);
        UNARY_CASE(Ceil, dispatch_unary_ceil);
        UNARY_CASE(Round, dispatch_unary_round);
        UNARY_CASE(Trunc, dispatch_unary_trunc);
        UNARY_CASE(ReLU, dispatch_activation_relu);
        UNARY_CASE(Sigmoid, dispatch_activation_sigmoid);
        UNARY_CASE(GELU, dispatch_activation_gelu);
        UNARY_CASE(SiLU, dispatch_activation_silu);
    default:
        return nullptr;
    }

#undef UNARY_CASE
#undef UNARY_CASE_INT
}

// ============================================================================
// Binary dispatch table
// ============================================================================

BinaryFn get_binary_fn(ops::OpType op, DType dtype) {
    using ops::OpType;

#define BINARY_CASE(OP, DISPATCH)                                              \
    case OpType::OP:                                                           \
        if (dtype == DType::Float32)                                           \
            return binary_wrap<float, simd::DISPATCH<float>>;                  \
        if (dtype == DType::Float64)                                           \
            return binary_wrap<double, simd::DISPATCH<double>>;                \
        return nullptr

#define BINARY_CASE_INT(OP, DISPATCH)                                          \
    case OpType::OP:                                                           \
        if (dtype == DType::Float32)                                           \
            return binary_wrap<float, simd::DISPATCH<float>>;                  \
        if (dtype == DType::Float64)                                           \
            return binary_wrap<double, simd::DISPATCH<double>>;                \
        if (dtype == DType::Int32)                                             \
            return binary_wrap<int32_t, simd::DISPATCH<int32_t>>;              \
        if (dtype == DType::Int64)                                             \
            return binary_wrap<int64_t, simd::DISPATCH<int64_t>>;              \
        return nullptr

    switch (op) {
        BINARY_CASE_INT(Add, dispatch_binary_add);
        BINARY_CASE_INT(Subtract, dispatch_binary_sub);
        BINARY_CASE_INT(Multiply, dispatch_binary_mul);
        BINARY_CASE_INT(Divide, dispatch_binary_div);
        BINARY_CASE_INT(Maximum, dispatch_binary_max);
        BINARY_CASE_INT(Minimum, dispatch_binary_min);
        BINARY_CASE(Power, dispatch_binary_pow);
        BINARY_CASE(Atan2, dispatch_binary_atan2);
        BINARY_CASE(Hypot, dispatch_binary_hypot);
        BINARY_CASE(Modulo, dispatch_binary_fmod);
    default:
        return nullptr;
    }

#undef BINARY_CASE
#undef BINARY_CASE_INT
}

} // namespace graph
} // namespace axiom
