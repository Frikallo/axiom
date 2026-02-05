# ============================================================================
# Axiom Benchmarks
# ============================================================================

# Common benchmark utilities (header-only)
add_library(axiom_benchmark_common INTERFACE)
target_include_directories(axiom_benchmark_common INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/common
)

# ============================================================================
# Core Benchmarks (Axiom internal)
# ============================================================================

# --- GEMM Benchmarks ---

# Main GEMM benchmark (CPU + GPU if available)
add_executable(bench_gemm core/matmul/gemm_benchmark.cpp)
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
    add_executable(bench_gemm_gpu core/matmul/gemm_gpu_overhead.cpp)
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

# --- SIMD Kernel Benchmarks ---

add_executable(bench_simd_kernels core/simd/simd_kernels.cpp)
target_link_libraries(bench_simd_kernels PRIVATE
    axiom
    axiom_benchmark_common
    benchmark::benchmark
    benchmark::benchmark_main
)
set_target_properties(bench_simd_kernels PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

# --- Fusion Benchmarks ---

add_executable(bench_fusion core/fusion/fusion_benchmark.cpp)
target_link_libraries(bench_fusion PRIVATE
    axiom
)
set_target_properties(bench_fusion PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

add_executable(bench_lazy_vs_eager core/fusion/lazy_vs_eager.cpp)
target_link_libraries(bench_lazy_vs_eager PRIVATE
    axiom
)
set_target_properties(bench_lazy_vs_eager PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

# ============================================================================
# Comparison Benchmarks (External libraries)
# ============================================================================

# Create output directory for comparison binaries
set(COMPARE_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/compare)
file(MAKE_DIRECTORY ${COMPARE_OUTPUT_DIR})

# Axiom comparison benchmark (CPU)
add_executable(matmul_axiom compare/matmul_axiom.cpp)
target_link_libraries(matmul_axiom PRIVATE axiom)
set_target_properties(matmul_axiom PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    RUNTIME_OUTPUT_DIRECTORY ${COMPARE_OUTPUT_DIR}
)

# Axiom comparison benchmark (GPU)
if(APPLE AND AXIOM_HAS_METAL)
    add_executable(matmul_axiom_gpu compare/matmul_axiom_gpu.cpp)
    target_link_libraries(matmul_axiom_gpu PRIVATE axiom)
    set_target_properties(matmul_axiom_gpu PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
        RUNTIME_OUTPUT_DIRECTORY ${COMPARE_OUTPUT_DIR}
    )
endif()

# Eigen benchmark (optional)
find_package(Eigen3 QUIET)
if(Eigen3_FOUND)
    message(STATUS "Eigen3 found: Building Eigen benchmark")
    add_executable(matmul_eigen compare/matmul_eigen.cpp)
    target_link_libraries(matmul_eigen PRIVATE Eigen3::Eigen)
    if(APPLE)
        target_link_libraries(matmul_eigen PRIVATE "-framework Accelerate")
    endif()
    set_target_properties(matmul_eigen PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
        RUNTIME_OUTPUT_DIRECTORY ${COMPARE_OUTPUT_DIR}
    )
else()
    message(STATUS "Eigen3 not found: Skipping Eigen benchmark")
endif()

# Armadillo benchmark (optional)
find_package(Armadillo QUIET)
if(Armadillo_FOUND)
    message(STATUS "Armadillo found: Building Armadillo benchmark")
    add_executable(matmul_armadillo compare/matmul_armadillo.cpp)
    target_include_directories(matmul_armadillo PRIVATE ${ARMADILLO_INCLUDE_DIRS})
    target_link_libraries(matmul_armadillo PRIVATE ${ARMADILLO_LIBRARIES})
    if(APPLE)
        target_link_libraries(matmul_armadillo PRIVATE "-framework Accelerate")
    endif()
    set_target_properties(matmul_armadillo PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
        RUNTIME_OUTPUT_DIRECTORY ${COMPARE_OUTPUT_DIR}
    )
else()
    message(STATUS "Armadillo not found: Skipping Armadillo benchmark")
endif()

# ============================================================================
# Install targets for benchmark binaries (optional)
# ============================================================================

# install(TARGETS bench_gemm bench_simd_kernels bench_fusion
#     RUNTIME DESTINATION bin/benchmarks
# )
