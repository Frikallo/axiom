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