# Tests for Axiom tensor library

# Basic tensor functionality test
add_executable(test_tensor_basic
    test_tensor_basic.cpp
)

target_link_libraries(test_tensor_basic
    PRIVATE
        axiom
)

target_include_directories(test_tensor_basic
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
)

# Add the test to CTest
add_test(
    NAME tensor_basic
    COMMAND test_tensor_basic
)

# Set test properties
set_tests_properties(tensor_basic
    PROPERTIES
        TIMEOUT 30
)

# Comprehensive tensor operations test
add_executable(test_tensor_operations
    test_tensor_operations.cpp
)

target_link_libraries(test_tensor_operations
    PRIVATE
        axiom
)

target_include_directories(test_tensor_operations
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
)

add_test(
    NAME tensor_operations
    COMMAND test_tensor_operations
)

set_tests_properties(tensor_operations
    PROPERTIES
        TIMEOUT 30
)

# Slicing and view tests
add_executable(test_tensor_views
    test_tensor_views.cpp
)

target_link_libraries(test_tensor_views
    PRIVATE
        axiom
)

target_include_directories(test_tensor_views
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
)

add_test(
    NAME tensor_views
    COMMAND test_tensor_views
)

set_tests_properties(tensor_views
    PROPERTIES
        TIMEOUT 30
)

# Indexing tests
add_executable(test_tensor_indexing
    test_tensor_indexing.cpp
)

target_link_libraries(test_tensor_indexing
    PRIVATE
        axiom
)

target_include_directories(test_tensor_indexing
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
)

add_test(
    NAME tensor_indexing
    COMMAND test_tensor_indexing
)

set_tests_properties(tensor_indexing
    PROPERTIES
        TIMEOUT 30
)

# Unary operations tests
add_executable(test_tensor_unary_operations
    test_tensor_unary_operations.cpp
)

target_link_libraries(test_tensor_unary_operations
    PRIVATE
        axiom
)

target_include_directories(test_tensor_unary_operations
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
)

if(AXIOM_METAL_SUPPORTED)
    target_compile_definitions(test_tensor_unary_operations PRIVATE AXIOM_TEST_GPU)
endif()

add_test(
    NAME tensor_unary_operations
    COMMAND test_tensor_unary_operations
)

set_tests_properties(tensor_unary_operations
    PROPERTIES
        TIMEOUT 30
)

# Reduction operations tests
add_executable(test_tensor_reductions
    test_tensor_reductions.cpp
)

target_link_libraries(test_tensor_reductions
    PRIVATE
        axiom
)

target_include_directories(test_tensor_reductions
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
)

add_test(
    NAME tensor_reductions
    COMMAND test_tensor_reductions
)

set_tests_properties(tensor_reductions
    PROPERTIES
        TIMEOUT 30
)

# Matrix multiplication tests
add_executable(test_tensor_matmul
    test_tensor_matmul.cpp
)

target_link_libraries(test_tensor_matmul
    PRIVATE
        axiom
)

target_include_directories(test_tensor_matmul
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
)

add_test(
    NAME tensor_matmul
    COMMAND test_tensor_matmul
)

set_tests_properties(tensor_matmul
    PROPERTIES
        TIMEOUT 30
)

# Random number generation tests
add_executable(test_tensor_random
    test_tensor_random.cpp
)

target_link_libraries(test_tensor_random
    PRIVATE
        axiom
)

target_include_directories(test_tensor_random
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
)

add_test(
    NAME tensor_random
    COMMAND test_tensor_random
)

set_tests_properties(tensor_random
    PROPERTIES
        TIMEOUT 30
)

# Shape operations tests
add_executable(test_tensor_shape_ops
    test_tensor_shape_ops.cpp
)

target_link_libraries(test_tensor_shape_ops
    PRIVATE
        axiom
)

target_include_directories(test_tensor_shape_ops
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
)

add_test(
    NAME tensor_shape_ops
    COMMAND test_tensor_shape_ops
)

set_tests_properties(tensor_shape_ops
    PROPERTIES
        TIMEOUT 30
)

# Comparison operations tests
add_executable(test_tensor_comparisons
    test_tensor_comparisons.cpp
)

target_link_libraries(test_tensor_comparisons
    PRIVATE
        axiom
)

target_include_directories(test_tensor_comparisons
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
)

add_test(
    NAME tensor_comparisons
    COMMAND test_tensor_comparisons
)

set_tests_properties(tensor_comparisons
    PROPERTIES
        TIMEOUT 30
)

# Argmax/Argmin tests
add_executable(test_tensor_argmax_argmin
    test_tensor_argmax_argmin.cpp
)

target_link_libraries(test_tensor_argmax_argmin
    PRIVATE
        axiom
)

target_include_directories(test_tensor_argmax_argmin
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
)

add_test(
    NAME tensor_argmax_argmin
    COMMAND test_tensor_argmax_argmin
)

set_tests_properties(tensor_argmax_argmin
    PROPERTIES
        TIMEOUT 30
)

# Where operation tests
add_executable(test_tensor_where
    test_tensor_where.cpp
)

target_link_libraries(test_tensor_where
    PRIVATE
        axiom
)

target_include_directories(test_tensor_where
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
)

add_test(
    NAME tensor_where
    COMMAND test_tensor_where
)

set_tests_properties(tensor_where
    PROPERTIES
        TIMEOUT 30
)

# Activation functions tests
add_executable(test_tensor_activations
    test_tensor_activations.cpp
)

target_link_libraries(test_tensor_activations
    PRIVATE
        axiom
)

target_include_directories(test_tensor_activations
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
)

add_test(
    NAME tensor_activations
    COMMAND test_tensor_activations
)

set_tests_properties(tensor_activations
    PROPERTIES
        TIMEOUT 30
)

# Complex operations tests
add_executable(test_tensor_complex
    test_tensor_complex.cpp
)

target_link_libraries(test_tensor_complex
    PRIVATE
        axiom
)

target_include_directories(test_tensor_complex
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
)

add_test(
    NAME tensor_complex
    COMMAND test_tensor_complex
)

set_tests_properties(tensor_complex
    PROPERTIES
        TIMEOUT 30
)

# Normalization operations tests
add_executable(test_tensor_normalization
    test_tensor_normalization.cpp
)

target_link_libraries(test_tensor_normalization
    PRIVATE
        axiom
)

target_include_directories(test_tensor_normalization
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
)

add_test(
    NAME tensor_normalization
    COMMAND test_tensor_normalization
)

set_tests_properties(tensor_normalization
    PROPERTIES
        TIMEOUT 30
)

# Gather/Scatter/IndexSelect tests
add_executable(test_tensor_gather_scatter
    test_tensor_gather_scatter.cpp
)

target_link_libraries(test_tensor_gather_scatter
    PRIVATE
        axiom
)

target_include_directories(test_tensor_gather_scatter
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
)

add_test(
    NAME tensor_gather_scatter
    COMMAND test_tensor_gather_scatter
)

set_tests_properties(tensor_gather_scatter
    PROPERTIES
        TIMEOUT 30
)

# Einops reduce tests
add_executable(test_einops_reduce
    test_einops_reduce.cpp
)

target_link_libraries(test_einops_reduce
    PRIVATE
        axiom
)

target_include_directories(test_einops_reduce
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
)

add_test(
    NAME einops_reduce
    COMMAND test_einops_reduce
)

set_tests_properties(einops_reduce
    PROPERTIES
        TIMEOUT 30
)

# CPU/GPU parity tests
add_executable(test_cpu_gpu_parity
    test_cpu_gpu_parity.cpp
)

target_link_libraries(test_cpu_gpu_parity
    PRIVATE
        axiom
)

target_include_directories(test_cpu_gpu_parity
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
)

add_test(
    NAME cpu_gpu_parity
    COMMAND test_cpu_gpu_parity
)

set_tests_properties(cpu_gpu_parity
    PROPERTIES
        TIMEOUT 60
)

# Transformer operations tests
add_executable(test_transformer_ops
    test_transformer_ops.cpp
)

target_link_libraries(test_transformer_ops
    PRIVATE
        axiom
)

target_include_directories(test_transformer_ops
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
)

add_test(
    NAME transformer_ops
    COMMAND test_transformer_ops
)

set_tests_properties(transformer_ops
    PROPERTIES
        TIMEOUT 60
)

# Logical/Bitwise/Math operations tests
add_executable(test_tensor_logical_bitwise
    test_tensor_logical_bitwise.cpp
)

target_link_libraries(test_tensor_logical_bitwise
    PRIVATE
        axiom
)

target_include_directories(test_tensor_logical_bitwise
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
)

add_test(
    NAME tensor_logical_bitwise
    COMMAND test_tensor_logical_bitwise
)

set_tests_properties(tensor_logical_bitwise
    PROPERTIES
        TIMEOUT 30
)

# NumPy-like operations tests
add_executable(test_tensor_numpy_ops
    test_tensor_numpy_ops.cpp
)

target_link_libraries(test_tensor_numpy_ops
    PRIVATE
        axiom
)

target_include_directories(test_tensor_numpy_ops
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
)

add_test(
    NAME tensor_numpy_ops
    COMMAND test_tensor_numpy_ops
)

set_tests_properties(tensor_numpy_ops
    PROPERTIES
        TIMEOUT 30
)

# FlatBuffers I/O tests
add_executable(test_io_flatbuffers
    test_io_flatbuffers.cpp
)

target_link_libraries(test_io_flatbuffers
    PRIVATE
        axiom
)

target_include_directories(test_io_flatbuffers
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
)

add_test(
    NAME io_flatbuffers
    COMMAND test_io_flatbuffers
)

set_tests_properties(io_flatbuffers
    PROPERTIES
        TIMEOUT 60
)

# NumPy I/O tests
add_executable(test_io_numpy
    test_io_numpy.cpp
)

target_link_libraries(test_io_numpy
    PRIVATE
        axiom
)

target_include_directories(test_io_numpy
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
)

# Generate NumPy test fixtures
find_package(Python3 COMPONENTS Interpreter)
if(Python3_FOUND)
    add_custom_target(generate_npy_fixtures
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/generate_npy_fixtures.py
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/fixtures
        COMMENT "Generating NumPy test fixtures"
    )
    add_dependencies(test_io_numpy generate_npy_fixtures)
else()
    message(WARNING "Python3 not found - NumPy test fixtures must be generated manually")
endif()

add_test(
    NAME io_numpy
    COMMAND test_io_numpy
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

set_tests_properties(io_numpy
    PROPERTIES
        TIMEOUT 30
)

# Linear algebra tests
add_executable(test_tensor_linalg
    test_tensor_linalg.cpp
)

target_link_libraries(test_tensor_linalg
    PRIVATE
        axiom
)

target_include_directories(test_tensor_linalg
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
)

add_test(
    NAME tensor_linalg
    COMMAND test_tensor_linalg
)

set_tests_properties(tensor_linalg
    PROPERTIES
        TIMEOUT 60
)

# Parallel operations tests
add_executable(test_parallel
    test_parallel.cpp
)

target_link_libraries(test_parallel
    PRIVATE
        axiom
)

target_include_directories(test_parallel
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
)

add_test(
    NAME parallel
    COMMAND test_parallel
)

set_tests_properties(parallel
    PROPERTIES
        TIMEOUT 60
)

# SIMD runtime dispatch tests
add_executable(test_simd_dispatch
    test_simd_dispatch.cpp
)

target_link_libraries(test_simd_dispatch
    PRIVATE
        axiom
)

target_include_directories(test_simd_dispatch
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/third_party/xsimd/include
        ${CMAKE_SOURCE_DIR}/third_party/FXdiv/include
)

# Pass through SIMD multi-arch flag if enabled
if(AXIOM_SIMD_MULTI_ARCH)
    target_compile_definitions(test_simd_dispatch PRIVATE AXIOM_SIMD_MULTI_ARCH)
endif()

add_test(
    NAME simd_dispatch
    COMMAND test_simd_dispatch
)

set_tests_properties(simd_dispatch
    PROPERTIES
        TIMEOUT 30
)