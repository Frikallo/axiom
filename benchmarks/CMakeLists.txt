# ============================================================================
# Axiom Benchmarks
# ============================================================================

# Common benchmark utilities (header-only)
add_library(axiom_benchmark_common INTERFACE)
target_include_directories(axiom_benchmark_common INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/common
)

# ============================================================================
# GEMM Benchmarks
# ============================================================================

# Main GEMM benchmark (CPU + GPU if available)
add_executable(bench_gemm gemm/gemm_benchmark.cpp)
target_link_libraries(bench_gemm PRIVATE
    axiom
    axiom_benchmark_common
    benchmark::benchmark
    benchmark::benchmark_main
)
set_target_properties(bench_gemm PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

# GPU-specific overhead benchmarks (macOS with Metal only)
if(APPLE AND AXIOM_HAS_METAL)
    add_executable(bench_gemm_gpu gemm/gemm_gpu_benchmark.cpp)
    target_link_libraries(bench_gemm_gpu PRIVATE
        axiom
        axiom_benchmark_common
        benchmark::benchmark
        benchmark::benchmark_main
    )
    set_target_properties(bench_gemm_gpu PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
    )
endif()
