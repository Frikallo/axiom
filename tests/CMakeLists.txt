# Tests for Axiom tensor library

# Basic tensor functionality test
add_executable(test_tensor_basic
    test_tensor_basic.cpp
)

target_link_libraries(test_tensor_basic
    PRIVATE
        axiom
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