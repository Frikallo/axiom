#pragma once

// Runtime SIMD Dispatch Interface (Highway Backend)
//
// This header provides dispatch wrappers that select the optimal SIMD
// implementation at runtime based on CPU capabilities.
//
// Highway automatically handles:
// - x86: SSE2, SSSE3, SSE4, AVX, AVX2, AVX3, AVX3_DL, AVX3_ZEN4
// - ARM: NEON, SVE, SVE2
// - RISC-V: RVV
// - WebAssembly: WASM, WASM_EMU256
// - PowerPC: PPC8, PPC9, PPC10
//
// Usage:
//   simd::dispatch_binary_add(a, b, result, n);  // Runtime dispatch

#include <cstddef>
#include <cstdint>
#include <string>

#include "hwy/highway.h"

namespace axiom {
namespace backends {
namespace cpu {
namespace simd {

// ============================================================================
// SIMD Architecture Info
// ============================================================================

struct SimdInfo {
    const char *arch_name; // Architecture name (e.g., "NEON", "AVX2")
    size_t alignment;      // Required alignment in bytes
    size_t float32_width;  // Vector width for float (elements)
    size_t float64_width;  // Vector width for double (elements)
    size_t int32_width;    // Vector width for int32_t (elements)
    size_t int64_width;    // Vector width for int64_t (elements)
};

// Get runtime SIMD architecture info
inline SimdInfo get_simd_info() {
    // Get the supported targets bitfield
    int64_t targets = hwy::SupportedTargets();

    // Find the best (highest priority) supported target
    // Highway orders targets so that lower bit positions = better targets
    // The lowest set bit represents the best available target
    int64_t best_target = targets & -targets; // Isolate lowest set bit
    const char *name = hwy::TargetName(best_target);

    // For Highway, alignment is typically 64 bytes (AVX-512) or less
    constexpr size_t alignment = HWY_MAX_BYTES;

    // Report typical vector widths - these are compile-time estimates
    // Actual widths depend on runtime target selection
    return SimdInfo{
        name,
        alignment,
        HWY_MAX_BYTES / sizeof(float),   // float32 width
        HWY_MAX_BYTES / sizeof(double),  // float64 width
        HWY_MAX_BYTES / sizeof(int32_t), // int32 width
        HWY_MAX_BYTES / sizeof(int64_t), // int64 width
    };
}

// Print SIMD architecture info to stdout
inline void print_simd_info() {
    auto info = get_simd_info();
    std::printf("SIMD Architecture: %s (Highway)\n", info.arch_name);
    std::printf("  Alignment: %zu bytes\n", info.alignment);
    std::printf("  Vector widths:\n");
    std::printf("    float32: %zu elements (%zu bytes)\n", info.float32_width,
                info.float32_width * sizeof(float));
    std::printf("    float64: %zu elements (%zu bytes)\n", info.float64_width,
                info.float64_width * sizeof(double));
    std::printf("    int32:   %zu elements (%zu bytes)\n", info.int32_width,
                info.int32_width * sizeof(int32_t));
    std::printf("    int64:   %zu elements (%zu bytes)\n", info.int64_width,
                info.int64_width * sizeof(int64_t));
}

// Get SIMD info as a formatted string
inline std::string simd_info_string() {
    auto info = get_simd_info();
    char buf[512];
    std::snprintf(buf, sizeof(buf),
                  "SIMD: %s (align=%zu, f32x%zu, f64x%zu, i32x%zu, i64x%zu)",
                  info.arch_name, info.alignment, info.float32_width,
                  info.float64_width, info.int32_width, info.int64_width);
    return std::string(buf);
}

// ============================================================================
// SIMD Support Detection
// ============================================================================

// Highway always provides SIMD support for float/double
template <typename T>
inline constexpr bool has_support =
    std::is_same_v<T, float> || std::is_same_v<T, double> ||
    std::is_same_v<T, int32_t> || std::is_same_v<T, int64_t>;

// ============================================================================
// Dispatch Function Declarations (defined in hwy_kernels.cc)
// ============================================================================

} // namespace simd
} // namespace cpu
} // namespace backends

// Forward declarations in axiom::simd namespace
namespace simd {

// Binary operations
template <typename T>
void dispatch_binary_add(const T *a, const T *b, T *result, size_t n);
template <typename T>
void dispatch_binary_sub(const T *a, const T *b, T *result, size_t n);
template <typename T>
void dispatch_binary_mul(const T *a, const T *b, T *result, size_t n);
template <typename T>
void dispatch_binary_div(const T *a, const T *b, T *result, size_t n);
template <typename T>
void dispatch_binary_max(const T *a, const T *b, T *result, size_t n);
template <typename T>
void dispatch_binary_min(const T *a, const T *b, T *result, size_t n);
template <typename T>
void dispatch_binary_pow(const T *a, const T *b, T *result, size_t n);
template <typename T>
void dispatch_binary_atan2(const T *a, const T *b, T *result, size_t n);
template <typename T>
void dispatch_binary_hypot(const T *a, const T *b, T *result, size_t n);
template <typename T>
void dispatch_binary_fmod(const T *a, const T *b, T *result, size_t n);

// Unary operations
template <typename T>
void dispatch_unary_neg(const T *input, T *output, size_t n);
template <typename T>
void dispatch_unary_abs(const T *input, T *output, size_t n);
template <typename T>
void dispatch_unary_sqrt(const T *input, T *output, size_t n);
template <typename T>
void dispatch_unary_exp(const T *input, T *output, size_t n);
template <typename T>
void dispatch_unary_log(const T *input, T *output, size_t n);
template <typename T>
void dispatch_unary_sin(const T *input, T *output, size_t n);
template <typename T>
void dispatch_unary_cos(const T *input, T *output, size_t n);
template <typename T>
void dispatch_unary_tanh(const T *input, T *output, size_t n);
template <typename T>
void dispatch_unary_tan(const T *input, T *output, size_t n);
template <typename T>
void dispatch_unary_erf(const T *input, T *output, size_t n);
template <typename T>
void dispatch_unary_cbrt(const T *input, T *output, size_t n);
template <typename T>
void dispatch_unary_square(const T *input, T *output, size_t n);
template <typename T>
void dispatch_unary_reciprocal(const T *input, T *output, size_t n);
template <typename T>
void dispatch_unary_sign(const T *input, T *output, size_t n);
template <typename T>
void dispatch_unary_floor(const T *input, T *output, size_t n);
template <typename T>
void dispatch_unary_ceil(const T *input, T *output, size_t n);
template <typename T>
void dispatch_unary_round(const T *input, T *output, size_t n);
template <typename T>
void dispatch_unary_trunc(const T *input, T *output, size_t n);

// Reductions
template <typename T> T dispatch_reduce_sum(const T *data, size_t n);
template <typename T> T dispatch_reduce_max(const T *data, size_t n);
template <typename T> T dispatch_reduce_min(const T *data, size_t n);
template <typename T> T dispatch_reduce_prod(const T *data, size_t n);

// Activations
template <typename T>
void dispatch_activation_relu(const T *input, T *output, size_t n);
template <typename T>
void dispatch_activation_sigmoid(const T *input, T *output, size_t n);
template <typename T>
void dispatch_activation_gelu(const T *input, T *output, size_t n);
template <typename T>
void dispatch_activation_silu(const T *input, T *output, size_t n);
template <typename T>
void dispatch_activation_leaky_relu(const T *input, T *output, size_t n,
                                    double alpha = 0.01);

} // namespace simd

// Re-export in the old namespace for compatibility
namespace backends {
namespace cpu {
namespace simd {

using axiom::simd::dispatch_binary_add;
using axiom::simd::dispatch_binary_atan2;
using axiom::simd::dispatch_binary_div;
using axiom::simd::dispatch_binary_fmod;
using axiom::simd::dispatch_binary_hypot;
using axiom::simd::dispatch_binary_max;
using axiom::simd::dispatch_binary_min;
using axiom::simd::dispatch_binary_mul;
using axiom::simd::dispatch_binary_pow;
using axiom::simd::dispatch_binary_sub;

using axiom::simd::dispatch_unary_abs;
using axiom::simd::dispatch_unary_cbrt;
using axiom::simd::dispatch_unary_ceil;
using axiom::simd::dispatch_unary_cos;
using axiom::simd::dispatch_unary_erf;
using axiom::simd::dispatch_unary_exp;
using axiom::simd::dispatch_unary_floor;
using axiom::simd::dispatch_unary_log;
using axiom::simd::dispatch_unary_neg;
using axiom::simd::dispatch_unary_reciprocal;
using axiom::simd::dispatch_unary_round;
using axiom::simd::dispatch_unary_sign;
using axiom::simd::dispatch_unary_sin;
using axiom::simd::dispatch_unary_sqrt;
using axiom::simd::dispatch_unary_square;
using axiom::simd::dispatch_unary_tan;
using axiom::simd::dispatch_unary_tanh;
using axiom::simd::dispatch_unary_trunc;

using axiom::simd::dispatch_reduce_max;
using axiom::simd::dispatch_reduce_min;
using axiom::simd::dispatch_reduce_prod;
using axiom::simd::dispatch_reduce_sum;

using axiom::simd::dispatch_activation_gelu;
using axiom::simd::dispatch_activation_leaky_relu;
using axiom::simd::dispatch_activation_relu;
using axiom::simd::dispatch_activation_sigmoid;
using axiom::simd::dispatch_activation_silu;

} // namespace simd
} // namespace cpu
} // namespace backends

} // namespace axiom
