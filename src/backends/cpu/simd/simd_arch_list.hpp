#pragma once

// Runtime SIMD dispatch architecture definitions
// Full coverage of all xsimd-supported architectures for maximum performance
//
// x86/x86-64: SSE2, SSE3, SSSE3, SSE4.1, SSE4.2, AVX, FMA3+AVX, AVX2,
// FMA3+AVX2, AVX512 x86 AMD:    All above + FMA4 ARM:        ARMv7 (NEON),
// ARMv8 (NEON64) WebAssembly: WASM SIMD128 RISC-V:     RVV (Vector ISA)
// PowerPC:    VSX

#include <xsimd/xsimd.hpp>

namespace axiom {
namespace backends {
namespace cpu {
namespace simd {

// ============================================================================
// x86/x86-64 Architecture Lists (Intel/AMD)
// ============================================================================

// Full x86-64 dispatch chain - ordered newest to oldest for optimal dispatch
// Dispatch tries architectures in order and uses the first available one
using x86_64_full_arch_list = xsimd::arch_list<
    // AVX-512 family (Skylake-X 2017+, Ice Lake 2019+)
    xsimd::avx512bw, // AVX-512 + Byte/Word operations
    xsimd::avx512dq, // AVX-512 + DoubleWord/QuadWord
    xsimd::avx512cd, // AVX-512 + Conflict Detection
    xsimd::avx512f,  // AVX-512 Foundation

    // AVX2 family (Haswell 2013+) - most common modern architecture
    xsimd::fma3<xsimd::avx2>, // AVX2 + FMA3 (fused multiply-add)
    xsimd::avx2,              // AVX2 without FMA

    // AVX family (Sandy Bridge 2011+)
    xsimd::fma3<xsimd::avx>, // AVX + FMA3
    xsimd::avx,              // AVX without FMA

    // SSE4 family (Nehalem 2008+)
    xsimd::fma3<xsimd::sse4_2>, // SSE4.2 + FMA3 (rare but possible)
    xsimd::sse4_2,              // SSE4.2
    xsimd::sse4_1,              // SSE4.1

    // SSE3 family (Prescott 2004+)
    xsimd::ssse3, // Supplemental SSE3
    xsimd::sse3,  // SSE3

    // SSE2 baseline (always available on x86-64)
    xsimd::sse2>;

// AMD-specific list (includes FMA4 for Bulldozer family)
using x86_amd_arch_list =
    xsimd::arch_list<xsimd::avx512bw, xsimd::avx512dq, xsimd::avx512cd,
                     xsimd::avx512f, xsimd::fma3<xsimd::avx2>, xsimd::avx2,
                     xsimd::fma4, // AMD Bulldozer/Piledriver FMA4
                     xsimd::fma3<xsimd::avx>, xsimd::avx, xsimd::sse4_2,
                     xsimd::sse4_1, xsimd::ssse3, xsimd::sse3, xsimd::sse2>;

// ============================================================================
// ARM Architecture Lists
// ============================================================================

// ARM 64-bit (ARMv8 - Apple Silicon, AWS Graviton, Ampere, etc.)
using arm64_arch_list =
    xsimd::arch_list<xsimd::neon64 // NEON64 always available on AArch64
                     >;

// ARM 32-bit (ARMv7 with NEON)
using arm32_arch_list = xsimd::arch_list<xsimd::neon // NEON on ARMv7
                                         >;

// ============================================================================
// Other Architectures
// ============================================================================

// PowerPC (POWER7+)
using ppc_arch_list = xsimd::arch_list<xsimd::vsx // Vector Scalar Extension
                                       >;

// WebAssembly SIMD
using wasm_arch_list = xsimd::arch_list<xsimd::wasm // WASM SIMD128
                                        >;

// RISC-V Vector Extension (fixed width variants)
// Note: RVV is templated on vector width, common sizes are 128, 256, 512
#if defined(__riscv_v_fixed_vlen)
using riscv_arch_list =
    xsimd::arch_list<xsimd::detail::rvv<__riscv_v_fixed_vlen>>;
#else
// Fallback for when vector length isn't known at compile time
using riscv_arch_list =
    xsimd::arch_list<xsimd::detail::rvv<128> // Assume minimum 128-bit vectors
                     >;
#endif

// ============================================================================
// Platform-Specific Default Architecture List
// ============================================================================

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) ||             \
    defined(_M_IX86)
// x86/x86-64: Use full dispatch chain
using default_arch_list = x86_64_full_arch_list;
#elif defined(__aarch64__) || defined(_M_ARM64)
// ARM64: NEON64 always available
using default_arch_list = arm64_arch_list;
#elif defined(__arm__) || defined(_M_ARM)
// ARM32: NEON may or may not be available
using default_arch_list = arm32_arch_list;
#elif defined(__PPC64__) || defined(__powerpc64__) || defined(__ppc64__)
// PowerPC 64-bit
using default_arch_list = ppc_arch_list;
#elif defined(__EMSCRIPTEN__) || defined(__wasm__) || defined(__wasm32__) ||   \
    defined(__wasm64__)
// WebAssembly
using default_arch_list = wasm_arch_list;
#elif defined(__riscv) || defined(__riscv__)
// RISC-V
using default_arch_list = riscv_arch_list;
#else
// Unknown platform: use compile-time default
using default_arch_list = xsimd::arch_list<xsimd::default_arch>;
#endif

// ============================================================================
// Runtime Architecture Detection Helpers
// ============================================================================

// Get the name of the best available architecture at runtime
inline const char *get_runtime_arch_name() {
    auto available = xsimd::available_architectures();

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) ||             \
    defined(_M_IX86)
    // Check in order from best to worst
    if (available.avx512bw)
        return "avx512bw";
    if (available.avx512dq)
        return "avx512dq";
    if (available.avx512cd)
        return "avx512cd";
    if (available.avx512f)
        return "avx512f";
    if (available.avx2 && available.fma3_avx2)
        return "fma3<avx2>";
    if (available.avx2)
        return "avx2";
    if (available.fma4)
        return "fma4";
    if (available.avx && available.fma3_avx)
        return "fma3<avx>";
    if (available.avx)
        return "avx";
    if (available.sse4_2 && available.fma3_sse42)
        return "fma3<sse4_2>";
    if (available.sse4_2)
        return "sse4_2";
    if (available.sse4_1)
        return "sse4_1";
    if (available.ssse3)
        return "ssse3";
    if (available.sse3)
        return "sse3";
    if (available.sse2)
        return "sse2";
#elif defined(__aarch64__) || defined(_M_ARM64)
    if (available.neon64)
        return "neon64";
#elif defined(__arm__) || defined(_M_ARM)
    if (available.neon)
        return "neon";
#elif defined(__PPC64__) || defined(__powerpc64__)
    if (available.vsx)
        return "vsx";
#elif defined(__wasm__)
    if (available.wasm)
        return "wasm";
#elif defined(__riscv)
    if (available.rvv)
        return "rvv";
#endif
    return "scalar";
}

// Get detailed SIMD info string
inline const char *get_simd_details() {
    auto available = xsimd::available_architectures();
    static char buffer[256];

#if defined(__x86_64__) || defined(_M_X64)
    snprintf(buffer, sizeof(buffer),
             "x86-64: SSE2=%d SSE3=%d SSSE3=%d SSE4.1=%d SSE4.2=%d "
             "AVX=%d FMA3=%d AVX2=%d FMA4=%d AVX512F=%d AVX512BW=%d",
             available.sse2, available.sse3, available.ssse3, available.sse4_1,
             available.sse4_2, available.avx, available.fma3_avx,
             available.avx2, available.fma4, available.avx512f,
             available.avx512bw);
#elif defined(__aarch64__)
    snprintf(buffer, sizeof(buffer), "ARM64: NEON64=%d", available.neon64);
#elif defined(__arm__)
    snprintf(buffer, sizeof(buffer), "ARM32: NEON=%d", available.neon);
#else
    snprintf(buffer, sizeof(buffer), "Platform: %s", get_runtime_arch_name());
#endif
    return buffer;
}

} // namespace simd
} // namespace cpu
} // namespace backends
} // namespace axiom
