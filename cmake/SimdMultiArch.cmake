# SimdMultiArch.cmake
# Full multi-architecture SIMD builds for maximum performance on every CPU
#
# Supported architectures:
# x86-64: SSE2, SSE3, SSSE3, SSE4.1, SSE4.2, AVX, FMA3+AVX, AVX2, FMA3+AVX2,
#         AVX512F, AVX512CD, AVX512DQ, AVX512BW, FMA4 (AMD)
# ARM:    NEON (ARMv7), NEON64 (ARMv8)
# Other:  VSX (PowerPC), WASM (WebAssembly)
#
# Usage:
#   cmake -B build -DAXIOM_SIMD_MULTI_ARCH=ON
#   cmake --build build

option(AXIOM_SIMD_MULTI_ARCH "Build with runtime SIMD dispatch for maximum performance" OFF)

set(AXIOM_SIMD_OBJECTS "")

if(AXIOM_SIMD_MULTI_ARCH)
    message(STATUS "SIMD: Multi-architecture runtime dispatch enabled")

    # Helper macro to add a SIMD object library (macros don't have scope issues)
    macro(axiom_add_simd_arch name source)
        set(_flags ${ARGN})
        add_library(axiom_simd_${name} OBJECT
            ${CMAKE_SOURCE_DIR}/src/backends/cpu/simd/${source})
        target_include_directories(axiom_simd_${name} PRIVATE
            ${CMAKE_SOURCE_DIR}/include
            ${CMAKE_SOURCE_DIR}/src
            ${CMAKE_SOURCE_DIR}/third_party/xsimd/include
            ${CMAKE_SOURCE_DIR}/third_party/FXdiv/include)
        if(_flags)
            target_compile_options(axiom_simd_${name} PRIVATE ${_flags})
        endif()
        target_compile_definitions(axiom_simd_${name} PRIVATE AXIOM_SIMD_MULTI_ARCH)
        set_target_properties(axiom_simd_${name} PROPERTIES
            CXX_STANDARD 20
            CXX_STANDARD_REQUIRED ON)
        list(APPEND AXIOM_SIMD_OBJECTS $<TARGET_OBJECTS:axiom_simd_${name}>)
    endmacro()

    # =========================================================================
    # x86-64 architectures
    # =========================================================================
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|x86|i[3-6]86")
        message(STATUS "SIMD: Building x86-64 architectures")

        # SSE family (baseline to SSE4.2)
        axiom_add_simd_arch(sse2 simd_kernels_sse2.cpp -msse2)
        axiom_add_simd_arch(sse3 simd_kernels_sse3.cpp -msse3)
        axiom_add_simd_arch(ssse3 simd_kernels_ssse3.cpp -mssse3)
        axiom_add_simd_arch(sse4_1 simd_kernels_sse4_1.cpp -msse4.1)
        axiom_add_simd_arch(sse4_2 simd_kernels_sse4_2.cpp -msse4.2)
        axiom_add_simd_arch(fma3_sse4_2 simd_kernels_fma3_sse4_2.cpp -msse4.2 -mfma)

        # AVX family
        axiom_add_simd_arch(avx simd_kernels_avx.cpp -mavx)
        axiom_add_simd_arch(fma3_avx simd_kernels_fma3_avx.cpp -mavx -mfma)

        # AVX2 family (most common modern CPU)
        axiom_add_simd_arch(avx2 simd_kernels_avx2.cpp -mavx2)
        axiom_add_simd_arch(fma3_avx2 simd_kernels_fma3_avx2.cpp -mavx2 -mfma)

        # AVX-512 family
        axiom_add_simd_arch(avx512f simd_kernels_avx512f.cpp -mavx512f)
        axiom_add_simd_arch(avx512cd simd_kernels_avx512cd.cpp -mavx512f -mavx512cd)
        axiom_add_simd_arch(avx512dq simd_kernels_avx512dq.cpp -mavx512f -mavx512cd -mavx512dq)
        axiom_add_simd_arch(avx512bw simd_kernels_avx512.cpp -mavx512f -mavx512bw -mavx512dq)

        # AMD FMA4 (Bulldozer/Piledriver era)
        axiom_add_simd_arch(fma4 simd_kernels_fma4.cpp -mfma4)

        message(STATUS "SIMD: x86-64 targets: SSE2, SSE3, SSSE3, SSE4.1, SSE4.2, AVX, AVX2, AVX512, FMA3, FMA4")

    # =========================================================================
    # ARM64 (AArch64) architectures
    # =========================================================================
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
        message(STATUS "SIMD: Building ARM64 architecture")
        axiom_add_simd_arch(neon64 simd_kernels_neon64.cpp)
        message(STATUS "SIMD: ARM64 target: NEON64")

    # =========================================================================
    # ARM32 (ARMv7) architectures
    # =========================================================================
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm|ARM")
        message(STATUS "SIMD: Building ARM32 architecture")
        axiom_add_simd_arch(neon simd_kernels_neon.cpp -mfpu=neon)
        message(STATUS "SIMD: ARM32 target: NEON")

    # =========================================================================
    # PowerPC architectures
    # =========================================================================
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "ppc64|powerpc64|ppc")
        message(STATUS "SIMD: Building PowerPC architecture")
        axiom_add_simd_arch(vsx simd_kernels_vsx.cpp -mvsx)
        message(STATUS "SIMD: PowerPC target: VSX")

    # =========================================================================
    # WebAssembly
    # =========================================================================
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "wasm" OR EMSCRIPTEN)
        message(STATUS "SIMD: Building WebAssembly architecture")
        axiom_add_simd_arch(wasm simd_kernels_wasm.cpp -msimd128)
        message(STATUS "SIMD: WebAssembly target: SIMD128")

    else()
        message(STATUS "SIMD: Unknown architecture ${CMAKE_SYSTEM_PROCESSOR}, using default")
    endif()

else()
    message(STATUS "SIMD: Compile-time architecture detection")
    message(STATUS "      Enable runtime dispatch with: -DAXIOM_SIMD_MULTI_ARCH=ON")
endif()
