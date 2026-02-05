# Tests for Axiom tensor library

# Keep assertions enabled in tests even for Release builds
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    add_compile_options(-UNDEBUG)
endif()

# ============================================================================
# Google Test
# ============================================================================

include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.15.2
    GIT_SHALLOW    TRUE
)
# Prevent gtest from overriding compiler/linker settings on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

# ============================================================================
# Shared test utilities library
# ============================================================================

add_library(axiom_test_utils STATIC axiom_test_main.cpp)

target_link_libraries(axiom_test_utils
    PUBLIC
        axiom
        GTest::gtest
        GTest::gtest_main
)

target_include_directories(axiom_test_utils
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src
)

# ============================================================================
# Helper function to reduce per-test boilerplate
# ============================================================================

function(axiom_add_test TEST_NAME)
    cmake_parse_arguments(ARG "" "TIMEOUT;WORKING_DIRECTORY" "SOURCES;EXTRA_LIBS;EXTRA_INCLUDES" ${ARGN})

    if(NOT ARG_SOURCES)
        set(ARG_SOURCES ${TEST_NAME}.cpp)
    endif()
    if(NOT ARG_TIMEOUT)
        set(ARG_TIMEOUT 30)
    endif()

    add_executable(${TEST_NAME} ${ARG_SOURCES})

    target_link_libraries(${TEST_NAME}
        PRIVATE
            axiom_test_utils
            ${ARG_EXTRA_LIBS}
    )

    if(ARG_EXTRA_INCLUDES)
        target_include_directories(${TEST_NAME}
            PRIVATE
                ${ARG_EXTRA_INCLUDES}
        )
    endif()

    if(ARG_WORKING_DIRECTORY)
        gtest_discover_tests(${TEST_NAME}
            TIMEOUT ${ARG_TIMEOUT}
            WORKING_DIRECTORY ${ARG_WORKING_DIRECTORY}
            DISCOVERY_MODE PRE_TEST
        )
    else()
        gtest_discover_tests(${TEST_NAME}
            TIMEOUT ${ARG_TIMEOUT}
            DISCOVERY_MODE PRE_TEST
        )
    endif()
endfunction()

# ============================================================================
# Standard tests (30s timeout)
# ============================================================================

axiom_add_test(test_tensor_basic)
axiom_add_test(test_tensor_operations)
axiom_add_test(test_tensor_views)
axiom_add_test(test_tensor_indexing)
axiom_add_test(test_tensor_unary_operations)
axiom_add_test(test_tensor_reductions)
axiom_add_test(test_tensor_matmul)
axiom_add_test(test_tensor_random)
axiom_add_test(test_tensor_shape_ops)
axiom_add_test(test_tensor_comparisons)
axiom_add_test(test_tensor_argmax_argmin)
axiom_add_test(test_tensor_where)
axiom_add_test(test_tensor_activations)
axiom_add_test(test_tensor_complex)
axiom_add_test(test_tensor_normalization)
axiom_add_test(test_tensor_gather_scatter)
axiom_add_test(test_tensor_logical_bitwise)
axiom_add_test(test_tensor_numpy_ops)
axiom_add_test(test_broadcast)
axiom_add_test(test_random_extended)
axiom_add_test(test_shape_ops_extended)
axiom_add_test(test_advanced_indexing)
axiom_add_test(test_einsum)
axiom_add_test(test_fft)
axiom_add_test(test_pooling)
axiom_add_test(test_einops_reduce)
axiom_add_test(test_custom_functors)
axiom_add_test(test_shape_utils)

# ============================================================================
# Tests with longer timeouts (60s)
# ============================================================================

axiom_add_test(test_cpu_gpu_parity TIMEOUT 60)
axiom_add_test(test_transformer_ops TIMEOUT 60)
axiom_add_test(test_tensor_linalg TIMEOUT 60)
axiom_add_test(test_parallel TIMEOUT 60)
axiom_add_test(test_io_flatbuffers TIMEOUT 60)

# ============================================================================
# Special: SIMD dispatch test (extra libs/includes)
# ============================================================================

axiom_add_test(test_simd_dispatch
    EXTRA_LIBS hwy
    EXTRA_INCLUDES
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/third_party/FXdiv/include
        ${highway_SOURCE_DIR}
)

# ============================================================================
# Special: NumPy I/O test (fixtures + working directory)
# ============================================================================

axiom_add_test(test_io_numpy
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
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
        hwy
)

target_include_directories(test_simd_dispatch
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/third_party/FXdiv/include
        ${highway_SOURCE_DIR}
)

add_test(
    NAME simd_dispatch
    COMMAND test_simd_dispatch
)

set_tests_properties(simd_dispatch
    PROPERTIES
        TIMEOUT 30
)

# Broadcast utilities tests
add_executable(test_broadcast
    test_broadcast.cpp
)

target_link_libraries(test_broadcast
    PRIVATE
        axiom
)

target_include_directories(test_broadcast
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
)

add_test(
    NAME broadcast
    COMMAND test_broadcast
)

set_tests_properties(broadcast
    PROPERTIES
        TIMEOUT 30
)

# Extended random module tests
add_executable(test_random_extended
    test_random_extended.cpp
)

target_link_libraries(test_random_extended
    PRIVATE
        axiom
)

target_include_directories(test_random_extended
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
)

add_test(
    NAME random_extended
    COMMAND test_random_extended
)

set_tests_properties(random_extended
    PROPERTIES
        TIMEOUT 30
)

# Extended shape operations tests
add_executable(test_shape_ops_extended
    test_shape_ops_extended.cpp
)

target_link_libraries(test_shape_ops_extended
    PRIVATE
        axiom
)

target_include_directories(test_shape_ops_extended
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
)

add_test(
    NAME shape_ops_extended
    COMMAND test_shape_ops_extended
)

set_tests_properties(shape_ops_extended
    PROPERTIES
        TIMEOUT 30
)

# Advanced indexing tests
add_executable(test_advanced_indexing
    test_advanced_indexing.cpp
)

target_link_libraries(test_advanced_indexing
    PRIVATE
        axiom
)

target_include_directories(test_advanced_indexing
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
)

add_test(
    NAME advanced_indexing
    COMMAND test_advanced_indexing
)

set_tests_properties(advanced_indexing
    PROPERTIES
        TIMEOUT 30
)

# Einsum tests
add_executable(test_einsum
    test_einsum.cpp
)

target_link_libraries(test_einsum
    PRIVATE
        axiom
)

target_include_directories(test_einsum
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
)

add_test(
    NAME einsum
    COMMAND test_einsum
)

set_tests_properties(einsum
    PROPERTIES
        TIMEOUT 30
)

# FFT tests
add_executable(test_fft
    test_fft.cpp
)

target_link_libraries(test_fft
    PRIVATE
        axiom
)

target_include_directories(test_fft
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
)

add_test(
    NAME fft
    COMMAND test_fft
)

set_tests_properties(fft
    PROPERTIES
        TIMEOUT 30
)

# Pooling operations tests
add_executable(test_pooling
    test_pooling.cpp
)

target_link_libraries(test_pooling
    PRIVATE
        axiom
)

target_include_directories(test_pooling
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
)

add_test(
    NAME pooling
    COMMAND test_pooling
)

set_tests_properties(pooling
    PROPERTIES
        TIMEOUT 30
)

# Lazy evaluation tests
add_executable(test_lazy_evaluation
    test_lazy_evaluation.cpp
)

target_link_libraries(test_lazy_evaluation
    PRIVATE
        axiom
)

target_include_directories(test_lazy_evaluation
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
)

add_test(
    NAME lazy_evaluation
    COMMAND test_lazy_evaluation
)

set_tests_properties(lazy_evaluation
    PROPERTIES
        TIMEOUT 30
)

# Operator fusion tests
add_executable(test_fusion
    test_fusion.cpp
)

target_link_libraries(test_fusion
    PRIVATE
        axiom
)

target_include_directories(test_fusion
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
)

add_test(
    NAME fusion
    COMMAND test_fusion
)

set_tests_properties(fusion
    PROPERTIES
        TIMEOUT 30
)
