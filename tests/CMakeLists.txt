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