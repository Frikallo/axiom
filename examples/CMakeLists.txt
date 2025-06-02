# Examples for Axiom tensor library

# Basic usage example
add_executable(basic_usage
    basic_usage.cpp
)

target_link_libraries(basic_usage
    PRIVATE
        axiom
)