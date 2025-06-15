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

# If Metal is enabled, ensure the library is available for tests
if(APPLE AND AXIOM_HAS_METAL)
    add_custom_command(
        TARGET test_tensor_operations PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_BINARY_DIR}/default.metallib
                ${CMAKE_CURRENT_BINARY_DIR}/default.metallib
        COMMENT "Copying Metal library for tests"
    )
endif()

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