# Examples for Axiom tensor library

# Basic operations example
add_executable(basic_operations basic_operations.cpp)
target_link_libraries(basic_operations axiom)

# Set C++20 standard for examples
set_target_properties(basic_operations PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)