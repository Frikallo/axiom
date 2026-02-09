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
