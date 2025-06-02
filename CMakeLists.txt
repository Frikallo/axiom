cmake_minimum_required(VERSION 3.20)

project(Axiom
    VERSION 0.1.0
    DESCRIPTION "High-Performance Tensor Library for Apple Silicon"
    LANGUAGES CXX
)

# C++20 is required
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build options
option(AXIOM_BUILD_TESTS "Build unit tests" ON)
option(AXIOM_BUILD_EXAMPLES "Build examples" ON)
option(AXIOM_BUILD_BENCHMARKS "Build benchmarks" OFF)

# Compiler-specific options
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    add_compile_options(-Wall -Wextra -Wpedantic)
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(-O3 -DNDEBUG)
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
            # Apple Silicon optimizations
            add_compile_options(-mcpu=apple-m1)
        endif()
    endif()
endif()

# Platform-specific configurations
if(APPLE)
    find_library(METAL_FRAMEWORK Metal)
    find_library(FOUNDATION_FRAMEWORK Foundation)
    
    if(METAL_FRAMEWORK AND FOUNDATION_FRAMEWORK)
        # Enable Objective-C++ for Metal support
        enable_language(OBJCXX)
        set(CMAKE_OBJCXX_STANDARD 20)
        set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
        set(AXIOM_HAS_METAL ON)
    else()
        message(WARNING "Metal or Foundation framework not found. GPU support will be disabled.")
        set(AXIOM_HAS_METAL OFF)
    endif()
endif()

# Create the main library
set(AXIOM_SOURCES
    # Tensor core
    src/tensor/tensor.cpp
    src/tensor/shape.cpp
    src/tensor/storage.cpp
    src/tensor/dtype.cpp
    src/tensor/io.cpp
    
    # Backend registry
    src/backends/backend_registry.cpp
    
    # CPU backend
    src/backends/cpu/cpu_storage.cpp
)

# Add Metal backend if available
if(APPLE AND AXIOM_HAS_METAL)
    list(APPEND AXIOM_SOURCES
        src/backends/metal/metal_storage.mm
    )
    
    # Set Objective-C++ properties for Metal files
    set_source_files_properties(
        src/backends/metal/metal_storage.mm
        PROPERTIES
            COMPILE_FLAGS "-fobjc-arc"  # Enable ARC for Metal files
    )
endif()

add_library(axiom ${AXIOM_SOURCES})

# Set target properties
set_target_properties(axiom PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    EXPORT_NAME axiom
)

# Include directories
target_include_directories(axiom
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/backends
)

# Platform-specific linking
if(APPLE AND AXIOM_HAS_METAL)
    target_link_libraries(axiom
        PUBLIC
            ${METAL_FRAMEWORK}
            ${FOUNDATION_FRAMEWORK}
    )
    
    # Enable Metal support
    target_compile_definitions(axiom PUBLIC AXIOM_METAL_SUPPORT)
endif()

# Compiler feature requirements
target_compile_features(axiom
    PUBLIC
        cxx_std_20
)

# Add alias for easier integration
add_library(Axiom::axiom ALIAS axiom)

# Tests
if(AXIOM_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests")
    enable_testing()
    add_subdirectory(tests)
endif()

# Examples
if(AXIOM_BUILD_EXAMPLES AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples")
    add_subdirectory(examples)
endif()

# Benchmarks
if(AXIOM_BUILD_BENCHMARKS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks")
    add_subdirectory(benchmarks)
endif()

# Installation
include(GNUInstallDirs)

# Install headers
install(
    DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp"
)

# Install library
install(
    TARGETS axiom
    EXPORT AxiomTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Install CMake config files
install(
    EXPORT AxiomTargets
    FILE AxiomTargets.cmake
    NAMESPACE Axiom::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Axiom
)

# Create and install config file
include(CMakePackageConfigHelpers)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/AxiomConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/AxiomConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Axiom
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/AxiomConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

install(
    FILES
        "${CMAKE_CURRENT_BINARY_DIR}/AxiomConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/AxiomConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Axiom
)

# Print configuration summary
message(STATUS "")
message(STATUS "Axiom Configuration Summary:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "  Build tests: ${AXIOM_BUILD_TESTS}")
message(STATUS "  Build examples: ${AXIOM_BUILD_EXAMPLES}")
message(STATUS "  Build benchmarks: ${AXIOM_BUILD_BENCHMARKS}")
if(APPLE)
    if(AXIOM_HAS_METAL)
        message(STATUS "  Metal support: YES")
    else()
        message(STATUS "  Metal support: NO (frameworks not found)")
    endif()
    message(STATUS "  Target architecture: ${CMAKE_SYSTEM_PROCESSOR}")
else()
    message(STATUS "  Metal support: NO (not on Apple platform)")
endif()
message(STATUS "")