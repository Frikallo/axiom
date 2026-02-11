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
// DType dispatch helper — single template replaces 6 macros
// ============================================================================

// Float-only dispatch — only instantiates Wrap for float/double.
// This avoids linker errors for ops with no integer SIMD implementations.
template <template <typename> class Wrap, typename FnType>
static FnType dispatch_float_only(DType dtype) {
    if (dtype == DType::Float32)
        return Wrap<float>::fn;
    if (dtype == DType::Float64)
        return Wrap<double>::fn;
    return nullptr;
}

// All-type dispatch — instantiates Wrap for every supported type.
template <template <typename> class Wrap, typename FnType>
static FnType dispatch_all_types(DType dtype) {
    if (dtype == DType::Float32)
        return Wrap<float>::fn;
    if (dtype == DType::Float64)
        return Wrap<double>::fn;
    if (dtype == DType::Int32)
        return Wrap<int32_t>::fn;
    if (dtype == DType::Int64)
        return Wrap<int64_t>::fn;
    if (dtype == DType::Int8)
        return Wrap<int8_t>::fn;
    if (dtype == DType::Int16)
        return Wrap<int16_t>::fn;
    if (dtype == DType::UInt8)
        return Wrap<uint8_t>::fn;
    if (dtype == DType::UInt16)
        return Wrap<uint16_t>::fn;
    if (dtype == DType::UInt32)
        return Wrap<uint32_t>::fn;
    if (dtype == DType::UInt64)
        return Wrap<uint64_t>::fn;
    return nullptr;
}

// ============================================================================
// Unary dispatch table
// ============================================================================

// Helper: wraps a simd dispatch function into a UnaryFn
template <typename T, void (*Fn)(const T *, T *, size_t)> struct UnaryWrap {
    static constexpr UnaryFn fn = unary_wrap<T, Fn>;
};

#define UNARY_DISPATCH(DISPATCH)                                               \
    template <typename T> struct UnaryWrap_##DISPATCH {                        \
        static constexpr UnaryFn fn = unary_wrap<T, simd::DISPATCH<T>>;        \
    }

// All-type unary ops (have integer SIMD implementations)
UNARY_DISPATCH(dispatch_unary_neg);
UNARY_DISPATCH(dispatch_unary_abs);
UNARY_DISPATCH(dispatch_unary_square);
UNARY_DISPATCH(dispatch_unary_sign);
UNARY_DISPATCH(dispatch_unary_floor);
UNARY_DISPATCH(dispatch_unary_ceil);
UNARY_DISPATCH(dispatch_unary_round);
UNARY_DISPATCH(dispatch_unary_trunc);
UNARY_DISPATCH(dispatch_activation_relu);

// Float-only unary ops (no integer SIMD — only declare for float/double)
UNARY_DISPATCH(dispatch_unary_sqrt);
UNARY_DISPATCH(dispatch_unary_exp);
UNARY_DISPATCH(dispatch_unary_log);
UNARY_DISPATCH(dispatch_unary_sin);
UNARY_DISPATCH(dispatch_unary_cos);
UNARY_DISPATCH(dispatch_unary_tan);
UNARY_DISPATCH(dispatch_unary_tanh);
UNARY_DISPATCH(dispatch_unary_erf);
UNARY_DISPATCH(dispatch_unary_cbrt);
UNARY_DISPATCH(dispatch_unary_reciprocal);
UNARY_DISPATCH(dispatch_activation_sigmoid);
UNARY_DISPATCH(dispatch_activation_gelu);
UNARY_DISPATCH(dispatch_activation_silu);

#undef UNARY_DISPATCH

UnaryFn get_unary_fn(ops::OpType op, DType dtype) {
    using ops::OpType;
    switch (op) {
    // All-type ops
    case OpType::Negate:
        return dispatch_all_types<UnaryWrap_dispatch_unary_neg, UnaryFn>(dtype);
    case OpType::Abs:
        return dispatch_all_types<UnaryWrap_dispatch_unary_abs, UnaryFn>(dtype);
    case OpType::Square:
        return dispatch_all_types<UnaryWrap_dispatch_unary_square, UnaryFn>(
            dtype);
    case OpType::Sign:
        return dispatch_all_types<UnaryWrap_dispatch_unary_sign, UnaryFn>(
            dtype);
    case OpType::Floor:
        return dispatch_all_types<UnaryWrap_dispatch_unary_floor, UnaryFn>(
            dtype);
    case OpType::Ceil:
        return dispatch_all_types<UnaryWrap_dispatch_unary_ceil, UnaryFn>(
            dtype);
    case OpType::Round:
        return dispatch_all_types<UnaryWrap_dispatch_unary_round, UnaryFn>(
            dtype);
    case OpType::Trunc:
        return dispatch_all_types<UnaryWrap_dispatch_unary_trunc, UnaryFn>(
            dtype);
    case OpType::ReLU:
        return dispatch_all_types<UnaryWrap_dispatch_activation_relu, UnaryFn>(
            dtype);
    // Float-only ops (no integer SIMD implementations exist)
    case OpType::Sqrt:
        return dispatch_float_only<UnaryWrap_dispatch_unary_sqrt, UnaryFn>(
            dtype);
    case OpType::Exp:
        return dispatch_float_only<UnaryWrap_dispatch_unary_exp, UnaryFn>(
            dtype);
    case OpType::Log:
        return dispatch_float_only<UnaryWrap_dispatch_unary_log, UnaryFn>(
            dtype);
    case OpType::Sin:
        return dispatch_float_only<UnaryWrap_dispatch_unary_sin, UnaryFn>(
            dtype);
    case OpType::Cos:
        return dispatch_float_only<UnaryWrap_dispatch_unary_cos, UnaryFn>(
            dtype);
    case OpType::Tan:
        return dispatch_float_only<UnaryWrap_dispatch_unary_tan, UnaryFn>(
            dtype);
    case OpType::Tanh:
        return dispatch_float_only<UnaryWrap_dispatch_unary_tanh, UnaryFn>(
            dtype);
    case OpType::Erf:
        return dispatch_float_only<UnaryWrap_dispatch_unary_erf, UnaryFn>(
            dtype);
    case OpType::Cbrt:
        return dispatch_float_only<UnaryWrap_dispatch_unary_cbrt, UnaryFn>(
            dtype);
    case OpType::Reciprocal:
        return dispatch_float_only<UnaryWrap_dispatch_unary_reciprocal,
                                   UnaryFn>(dtype);
    case OpType::Sigmoid:
        return dispatch_float_only<UnaryWrap_dispatch_activation_sigmoid,
                                   UnaryFn>(dtype);
    case OpType::GELU:
        return dispatch_float_only<UnaryWrap_dispatch_activation_gelu, UnaryFn>(
            dtype);
    case OpType::SiLU:
        return dispatch_float_only<UnaryWrap_dispatch_activation_silu, UnaryFn>(
            dtype);
    default:
        return nullptr;
    }
}

// ============================================================================
// Binary dispatch table
// ============================================================================

#define BINARY_DISPATCH(DISPATCH)                                              \
    template <typename T> struct BinaryWrap_##DISPATCH {                       \
        static constexpr BinaryFn fn = binary_wrap<T, simd::DISPATCH<T>>;      \
    }

// All-type binary ops
BINARY_DISPATCH(dispatch_binary_add);
BINARY_DISPATCH(dispatch_binary_sub);
BINARY_DISPATCH(dispatch_binary_mul);
BINARY_DISPATCH(dispatch_binary_div);
BINARY_DISPATCH(dispatch_binary_max);
BINARY_DISPATCH(dispatch_binary_min);

// Float-only binary ops
BINARY_DISPATCH(dispatch_binary_pow);
BINARY_DISPATCH(dispatch_binary_atan2);
BINARY_DISPATCH(dispatch_binary_hypot);
BINARY_DISPATCH(dispatch_binary_fmod);

#undef BINARY_DISPATCH

BinaryFn get_binary_fn(ops::OpType op, DType dtype) {
    using ops::OpType;
    switch (op) {
    // All-type ops
    case OpType::Add:
        return dispatch_all_types<BinaryWrap_dispatch_binary_add, BinaryFn>(
            dtype);
    case OpType::Subtract:
        return dispatch_all_types<BinaryWrap_dispatch_binary_sub, BinaryFn>(
            dtype);
    case OpType::Multiply:
        return dispatch_all_types<BinaryWrap_dispatch_binary_mul, BinaryFn>(
            dtype);
    case OpType::Divide:
        return dispatch_all_types<BinaryWrap_dispatch_binary_div, BinaryFn>(
            dtype);
    case OpType::Maximum:
        return dispatch_all_types<BinaryWrap_dispatch_binary_max, BinaryFn>(
            dtype);
    case OpType::Minimum:
        return dispatch_all_types<BinaryWrap_dispatch_binary_min, BinaryFn>(
            dtype);
    // Float-only ops
    case OpType::Power:
        return dispatch_float_only<BinaryWrap_dispatch_binary_pow, BinaryFn>(
            dtype);
    case OpType::Atan2:
        return dispatch_float_only<BinaryWrap_dispatch_binary_atan2, BinaryFn>(
            dtype);
    case OpType::Hypot:
        return dispatch_float_only<BinaryWrap_dispatch_binary_hypot, BinaryFn>(
            dtype);
    case OpType::Modulo:
        return dispatch_float_only<BinaryWrap_dispatch_binary_fmod, BinaryFn>(
            dtype);
    default:
        return nullptr;
    }
}

} // namespace graph
} // namespace axiom
