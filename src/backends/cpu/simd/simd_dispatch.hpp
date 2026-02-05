#pragma once

// Runtime SIMD Dispatch Interface
//
// This header provides dispatch wrappers that select the optimal SIMD
// implementation at runtime based on CPU capabilities.
//
// Usage:
//   simd::dispatch_binary_add(a, b, result, n);  // Runtime dispatch
//   simd::BinaryAdd{}(xsimd::avx2{}, a, b, result, n);  // Direct call
//
// When AXIOM_SIMD_MULTI_ARCH is defined, xsimd::dispatch() selects the
// best available implementation. Otherwise, uses compile-time default.

#include "simd_arch_list.hpp"
#include "simd_kernels.hpp"

namespace axiom {
namespace backends {
namespace cpu {
namespace simd {

// ============================================================================
// Dispatch Wrappers
// ============================================================================
// These wrap the kernel functors with xsimd::dispatch() for runtime selection.

#ifdef AXIOM_SIMD_MULTI_ARCH

// Platform-specific dispatch arch lists - FULL coverage for maximum performance
// Runtime dispatch selects the best available architecture
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) ||             \
    defined(_M_IX86)
// Full x86-64 chain: AVX512 -> FMA3+AVX2 -> AVX2 -> FMA3+AVX -> AVX -> SSE4.2
// -> ... -> SSE2
using dispatch_arch_list = x86_64_full_arch_list;
#elif defined(__aarch64__) || defined(_M_ARM64)
using dispatch_arch_list = arm64_arch_list;
#elif defined(__arm__) || defined(_M_ARM)
using dispatch_arch_list = arm32_arch_list;
#elif defined(__PPC64__) || defined(__powerpc64__)
using dispatch_arch_list = ppc_arch_list;
#elif defined(__EMSCRIPTEN__) || defined(__wasm__)
using dispatch_arch_list = wasm_arch_list;
#else
using dispatch_arch_list = xsimd::arch_list<xsimd::default_arch>;
#endif

// --- Binary Operations ---

template <typename T>
inline void dispatch_binary_add(const T *a, const T *b, T *result, size_t n) {
    xsimd::dispatch<dispatch_arch_list>(BinaryAdd{})(a, b, result, n);
}

template <typename T>
inline void dispatch_binary_sub(const T *a, const T *b, T *result, size_t n) {
    xsimd::dispatch<dispatch_arch_list>(BinarySub{})(a, b, result, n);
}

template <typename T>
inline void dispatch_binary_mul(const T *a, const T *b, T *result, size_t n) {
    xsimd::dispatch<dispatch_arch_list>(BinaryMul{})(a, b, result, n);
}

template <typename T>
inline void dispatch_binary_div(const T *a, const T *b, T *result, size_t n) {
    xsimd::dispatch<dispatch_arch_list>(BinaryDiv{})(a, b, result, n);
}

template <typename T>
inline void dispatch_binary_max(const T *a, const T *b, T *result, size_t n) {
    xsimd::dispatch<dispatch_arch_list>(BinaryMax{})(a, b, result, n);
}

template <typename T>
inline void dispatch_binary_min(const T *a, const T *b, T *result, size_t n) {
    xsimd::dispatch<dispatch_arch_list>(BinaryMin{})(a, b, result, n);
}

template <typename T>
inline void dispatch_binary_pow(const T *a, const T *b, T *result, size_t n) {
    xsimd::dispatch<dispatch_arch_list>(BinaryPow{})(a, b, result, n);
}

template <typename T>
inline void dispatch_binary_atan2(const T *a, const T *b, T *result, size_t n) {
    xsimd::dispatch<dispatch_arch_list>(BinaryAtan2{})(a, b, result, n);
}

template <typename T>
inline void dispatch_binary_hypot(const T *a, const T *b, T *result, size_t n) {
    xsimd::dispatch<dispatch_arch_list>(BinaryHypot{})(a, b, result, n);
}

template <typename T>
inline void dispatch_binary_fmod(const T *a, const T *b, T *result, size_t n) {
    xsimd::dispatch<dispatch_arch_list>(BinaryFmod{})(a, b, result, n);
}

// --- Unary Operations ---

template <typename T>
inline void dispatch_unary_neg(const T *input, T *output, size_t n) {
    xsimd::dispatch<dispatch_arch_list>(UnaryNeg{})(input, output, n);
}

template <typename T>
inline void dispatch_unary_abs(const T *input, T *output, size_t n) {
    xsimd::dispatch<dispatch_arch_list>(UnaryAbs{})(input, output, n);
}

template <typename T>
inline void dispatch_unary_sqrt(const T *input, T *output, size_t n) {
    xsimd::dispatch<dispatch_arch_list>(UnarySqrt{})(input, output, n);
}

template <typename T>
inline void dispatch_unary_exp(const T *input, T *output, size_t n) {
    xsimd::dispatch<dispatch_arch_list>(UnaryExp{})(input, output, n);
}

template <typename T>
inline void dispatch_unary_log(const T *input, T *output, size_t n) {
    xsimd::dispatch<dispatch_arch_list>(UnaryLog{})(input, output, n);
}

template <typename T>
inline void dispatch_unary_sin(const T *input, T *output, size_t n) {
    xsimd::dispatch<dispatch_arch_list>(UnarySin{})(input, output, n);
}

template <typename T>
inline void dispatch_unary_cos(const T *input, T *output, size_t n) {
    xsimd::dispatch<dispatch_arch_list>(UnaryCos{})(input, output, n);
}

template <typename T>
inline void dispatch_unary_tanh(const T *input, T *output, size_t n) {
    xsimd::dispatch<dispatch_arch_list>(UnaryTanh{})(input, output, n);
}

template <typename T>
inline void dispatch_unary_tan(const T *input, T *output, size_t n) {
    xsimd::dispatch<dispatch_arch_list>(UnaryTan{})(input, output, n);
}

template <typename T>
inline void dispatch_unary_erf(const T *input, T *output, size_t n) {
    xsimd::dispatch<dispatch_arch_list>(UnaryErf{})(input, output, n);
}

template <typename T>
inline void dispatch_unary_cbrt(const T *input, T *output, size_t n) {
    xsimd::dispatch<dispatch_arch_list>(UnaryCbrt{})(input, output, n);
}

template <typename T>
inline void dispatch_unary_square(const T *input, T *output, size_t n) {
    xsimd::dispatch<dispatch_arch_list>(UnarySquare{})(input, output, n);
}

template <typename T>
inline void dispatch_unary_reciprocal(const T *input, T *output, size_t n) {
    xsimd::dispatch<dispatch_arch_list>(UnaryReciprocal{})(input, output, n);
}

template <typename T>
inline void dispatch_unary_sign(const T *input, T *output, size_t n) {
    xsimd::dispatch<dispatch_arch_list>(UnarySign{})(input, output, n);
}

template <typename T>
inline void dispatch_unary_floor(const T *input, T *output, size_t n) {
    xsimd::dispatch<dispatch_arch_list>(UnaryFloor{})(input, output, n);
}

template <typename T>
inline void dispatch_unary_ceil(const T *input, T *output, size_t n) {
    xsimd::dispatch<dispatch_arch_list>(UnaryCeil{})(input, output, n);
}

template <typename T>
inline void dispatch_unary_round(const T *input, T *output, size_t n) {
    xsimd::dispatch<dispatch_arch_list>(UnaryRound{})(input, output, n);
}

template <typename T>
inline void dispatch_unary_trunc(const T *input, T *output, size_t n) {
    xsimd::dispatch<dispatch_arch_list>(UnaryTrunc{})(input, output, n);
}

// --- Reductions ---

template <typename T> inline T dispatch_reduce_sum(const T *data, size_t n) {
    return xsimd::dispatch<dispatch_arch_list>(ReduceSum{})(data, n);
}

template <typename T> inline T dispatch_reduce_max(const T *data, size_t n) {
    return xsimd::dispatch<dispatch_arch_list>(ReduceMax{})(data, n);
}

template <typename T> inline T dispatch_reduce_min(const T *data, size_t n) {
    return xsimd::dispatch<dispatch_arch_list>(ReduceMin{})(data, n);
}

template <typename T> inline T dispatch_reduce_prod(const T *data, size_t n) {
    return xsimd::dispatch<dispatch_arch_list>(ReduceProd{})(data, n);
}

// --- Activations ---

template <typename T>
inline void dispatch_activation_relu(const T *input, T *output, size_t n) {
    xsimd::dispatch<dispatch_arch_list>(ActivationReLU{})(input, output, n);
}

template <typename T>
inline void dispatch_activation_sigmoid(const T *input, T *output, size_t n) {
    xsimd::dispatch<dispatch_arch_list>(ActivationSigmoid{})(input, output, n);
}

template <typename T>
inline void dispatch_activation_gelu(const T *input, T *output, size_t n) {
    xsimd::dispatch<dispatch_arch_list>(ActivationGELU{})(input, output, n);
}

template <typename T>
inline void dispatch_activation_silu(const T *input, T *output, size_t n) {
    xsimd::dispatch<dispatch_arch_list>(ActivationSiLU{})(input, output, n);
}

template <typename T>
inline void dispatch_activation_leaky_relu(const T *input, T *output, size_t n,
                                           double alpha = 0.01) {
    xsimd::dispatch<dispatch_arch_list>(ActivationLeakyReLU{alpha})(input,
                                                                    output, n);
}

#else // !AXIOM_SIMD_MULTI_ARCH

// Compile-time dispatch (development builds, or when -march=native is used)
// Uses xsimd::default_arch which is determined at compile time

// --- Binary Operations ---

template <typename T>
inline void dispatch_binary_add(const T *a, const T *b, T *result, size_t n) {
    BinaryAdd{}(xsimd::default_arch{}, a, b, result, n);
}

template <typename T>
inline void dispatch_binary_sub(const T *a, const T *b, T *result, size_t n) {
    BinarySub{}(xsimd::default_arch{}, a, b, result, n);
}

template <typename T>
inline void dispatch_binary_mul(const T *a, const T *b, T *result, size_t n) {
    BinaryMul{}(xsimd::default_arch{}, a, b, result, n);
}

template <typename T>
inline void dispatch_binary_div(const T *a, const T *b, T *result, size_t n) {
    BinaryDiv{}(xsimd::default_arch{}, a, b, result, n);
}

template <typename T>
inline void dispatch_binary_max(const T *a, const T *b, T *result, size_t n) {
    BinaryMax{}(xsimd::default_arch{}, a, b, result, n);
}

template <typename T>
inline void dispatch_binary_min(const T *a, const T *b, T *result, size_t n) {
    BinaryMin{}(xsimd::default_arch{}, a, b, result, n);
}

template <typename T>
inline void dispatch_binary_pow(const T *a, const T *b, T *result, size_t n) {
    BinaryPow{}(xsimd::default_arch{}, a, b, result, n);
}

template <typename T>
inline void dispatch_binary_atan2(const T *a, const T *b, T *result, size_t n) {
    BinaryAtan2{}(xsimd::default_arch{}, a, b, result, n);
}

template <typename T>
inline void dispatch_binary_hypot(const T *a, const T *b, T *result, size_t n) {
    BinaryHypot{}(xsimd::default_arch{}, a, b, result, n);
}

template <typename T>
inline void dispatch_binary_fmod(const T *a, const T *b, T *result, size_t n) {
    BinaryFmod{}(xsimd::default_arch{}, a, b, result, n);
}

// --- Unary Operations ---

template <typename T>
inline void dispatch_unary_neg(const T *input, T *output, size_t n) {
    UnaryNeg{}(xsimd::default_arch{}, input, output, n);
}

template <typename T>
inline void dispatch_unary_abs(const T *input, T *output, size_t n) {
    UnaryAbs{}(xsimd::default_arch{}, input, output, n);
}

template <typename T>
inline void dispatch_unary_sqrt(const T *input, T *output, size_t n) {
    UnarySqrt{}(xsimd::default_arch{}, input, output, n);
}

template <typename T>
inline void dispatch_unary_exp(const T *input, T *output, size_t n) {
    UnaryExp{}(xsimd::default_arch{}, input, output, n);
}

template <typename T>
inline void dispatch_unary_log(const T *input, T *output, size_t n) {
    UnaryLog{}(xsimd::default_arch{}, input, output, n);
}

template <typename T>
inline void dispatch_unary_sin(const T *input, T *output, size_t n) {
    UnarySin{}(xsimd::default_arch{}, input, output, n);
}

template <typename T>
inline void dispatch_unary_cos(const T *input, T *output, size_t n) {
    UnaryCos{}(xsimd::default_arch{}, input, output, n);
}

template <typename T>
inline void dispatch_unary_tanh(const T *input, T *output, size_t n) {
    UnaryTanh{}(xsimd::default_arch{}, input, output, n);
}

template <typename T>
inline void dispatch_unary_tan(const T *input, T *output, size_t n) {
    UnaryTan{}(xsimd::default_arch{}, input, output, n);
}

template <typename T>
inline void dispatch_unary_erf(const T *input, T *output, size_t n) {
    UnaryErf{}(xsimd::default_arch{}, input, output, n);
}

template <typename T>
inline void dispatch_unary_cbrt(const T *input, T *output, size_t n) {
    UnaryCbrt{}(xsimd::default_arch{}, input, output, n);
}

template <typename T>
inline void dispatch_unary_square(const T *input, T *output, size_t n) {
    UnarySquare{}(xsimd::default_arch{}, input, output, n);
}

template <typename T>
inline void dispatch_unary_reciprocal(const T *input, T *output, size_t n) {
    UnaryReciprocal{}(xsimd::default_arch{}, input, output, n);
}

template <typename T>
inline void dispatch_unary_sign(const T *input, T *output, size_t n) {
    UnarySign{}(xsimd::default_arch{}, input, output, n);
}

template <typename T>
inline void dispatch_unary_floor(const T *input, T *output, size_t n) {
    UnaryFloor{}(xsimd::default_arch{}, input, output, n);
}

template <typename T>
inline void dispatch_unary_ceil(const T *input, T *output, size_t n) {
    UnaryCeil{}(xsimd::default_arch{}, input, output, n);
}

template <typename T>
inline void dispatch_unary_round(const T *input, T *output, size_t n) {
    UnaryRound{}(xsimd::default_arch{}, input, output, n);
}

template <typename T>
inline void dispatch_unary_trunc(const T *input, T *output, size_t n) {
    UnaryTrunc{}(xsimd::default_arch{}, input, output, n);
}

// --- Reductions ---

template <typename T> inline T dispatch_reduce_sum(const T *data, size_t n) {
    return ReduceSum{}(xsimd::default_arch{}, data, n);
}

template <typename T> inline T dispatch_reduce_max(const T *data, size_t n) {
    return ReduceMax{}(xsimd::default_arch{}, data, n);
}

template <typename T> inline T dispatch_reduce_min(const T *data, size_t n) {
    return ReduceMin{}(xsimd::default_arch{}, data, n);
}

template <typename T> inline T dispatch_reduce_prod(const T *data, size_t n) {
    return ReduceProd{}(xsimd::default_arch{}, data, n);
}

// --- Activations ---

template <typename T>
inline void dispatch_activation_relu(const T *input, T *output, size_t n) {
    ActivationReLU{}(xsimd::default_arch{}, input, output, n);
}

template <typename T>
inline void dispatch_activation_sigmoid(const T *input, T *output, size_t n) {
    ActivationSigmoid{}(xsimd::default_arch{}, input, output, n);
}

template <typename T>
inline void dispatch_activation_gelu(const T *input, T *output, size_t n) {
    ActivationGELU{}(xsimd::default_arch{}, input, output, n);
}

template <typename T>
inline void dispatch_activation_silu(const T *input, T *output, size_t n) {
    ActivationSiLU{}(xsimd::default_arch{}, input, output, n);
}

template <typename T>
inline void dispatch_activation_leaky_relu(const T *input, T *output, size_t n,
                                           double alpha = 0.01) {
    ActivationLeakyReLU{alpha}(xsimd::default_arch{}, input, output, n);
}

#endif // AXIOM_SIMD_MULTI_ARCH

} // namespace simd
} // namespace cpu
} // namespace backends
} // namespace axiom
