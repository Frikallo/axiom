# Examples for Axiom tensor library

# Basic operations example
add_executable(basic_operations basic_operations.cpp)
target_link_libraries(basic_operations axiom)

# Whisper inference POC
add_executable(whisper_poc whisper_poc.cpp)
target_link_libraries(whisper_poc axiom)

# Set C++20 standard for examples
set_target_properties(basic_operations whisper_poc PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)