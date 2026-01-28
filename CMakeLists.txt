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
            add_compile_options(-mcpu=apple-m1)
        endif()
    endif()
endif()

# Platform-specific configurations
if(APPLE)
    option(AXIOM_EMBED_METAL_LIBRARY "Embed Metal library into the main library" ON)

    find_library(METAL_FRAMEWORK Metal)
    find_library(FOUNDATION_FRAMEWORK Foundation)
    find_library(MPS_FRAMEWORK MetalPerformanceShaders)
    find_library(MPSGRAPH_FRAMEWORK MetalPerformanceShadersGraph)

    if(METAL_FRAMEWORK AND FOUNDATION_FRAMEWORK AND MPS_FRAMEWORK AND MPSGRAPH_FRAMEWORK)
        enable_language(OBJCXX)
        set(CMAKE_OBJCXX_STANDARD 20)
        set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
        set(AXIOM_HAS_METAL ON)
    else()
        message(WARNING "Metal, Foundation, MPS, or MPSGraph framework not found. GPU support will be disabled.")
        set(AXIOM_HAS_METAL OFF)
    endif()

    # Apple Accelerate framework for BLAS/vDSP optimizations
    find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
    if(ACCELERATE_FRAMEWORK)
        set(AXIOM_HAS_ACCELERATE ON)
        message(STATUS "Accelerate framework: found")
    else()
        set(AXIOM_HAS_ACCELERATE OFF)
        message(WARNING "Accelerate framework not found. CPU optimizations will be limited.")
    endif()
endif()

# External dependencies
include(FetchContent)

FetchContent_Declare(
    half
    GIT_REPOSITORY https://github.com/suruoxi/half.git
)

FetchContent_MakeAvailable(half)

# Create the main library
set(AXIOM_SOURCES
    # Tensor core
    src/tensor/tensor.cpp
    src/tensor/shape.cpp
    src/tensor/storage.cpp
    src/tensor/dtype.cpp
    src/tensor/debug.cpp
    src/tensor/io.cpp
    src/tensor/type_conversion.cpp
    src/tensor/einops.cpp
    src/tensor/operations.cpp
    src/tensor/random.cpp
    src/tensor/system.cpp
    src/tensor/indexing.cpp
    
    # CPU backend
    src/backends/cpu/cpu_storage.cpp
    src/backends/cpu/cpu_operations.cpp
    src/backends/cpu/cpu_accelerate.cpp
)

# Add Metal backend if available
if(APPLE AND AXIOM_HAS_METAL)
    list(APPEND AXIOM_SOURCES
        src/backends/metal/metal_storage.mm
        src/backends/metal/metal_operations.mm
        src/backends/metal/metal_common.mm
        src/backends/metal/mpsgraph_operations.mm
        src/backends/metal/kernels.metal
    )
    
    set_source_files_properties(
        src/backends/metal/metal_storage.mm
        src/backends/metal/metal_operations.mm
        src/backends/metal/metal_common.mm
        src/backends/metal/mpsgraph_operations.mm
        src/backends/metal/kernels.metal
        PROPERTIES
            COMPILE_FLAGS "-fobjc-arc"
    )
endif()

add_library(axiom SHARED ${AXIOM_SOURCES})

if(APPLE AND AXIOM_HAS_METAL)
    if(AXIOM_EMBED_METAL_LIBRARY)
        enable_language(ASM)
        target_compile_definitions(axiom PUBLIC AXIOM_METAL_EMBED_LIBRARY)

        set(METAL_KERNELS_FILE ${CMAKE_CURRENT_SOURCE_DIR}/src/backends/metal/kernels.metal)
        set(METAL_LIBRARY_TEMP ${CMAKE_CURRENT_BINARY_DIR}/default.metallib.tmp)
        set(METAL_EMBED_ASM ${CMAKE_CURRENT_BINARY_DIR}/axiom-metal-embed.s)

        # 1. Compile .metal source to a temporary .metallib file
        add_custom_command(
            OUTPUT ${METAL_LIBRARY_TEMP}
            COMMAND xcrun -sdk macosx metal -c ${METAL_KERNELS_FILE} -o ${CMAKE_CURRENT_BINARY_DIR}/kernels.air
            COMMAND xcrun -sdk macosx metallib ${CMAKE_CURRENT_BINARY_DIR}/kernels.air -o ${METAL_LIBRARY_TEMP}
            DEPENDS ${METAL_KERNELS_FILE}
            COMMENT "Compiling Metal kernels to temporary library"
        )

        # 2. Embed the temporary .metallib file into an assembly file
        add_custom_command(
            OUTPUT ${METAL_EMBED_ASM}
            COMMAND echo ".section __TEXT,__const" > ${METAL_EMBED_ASM}
            COMMAND echo ".globl _axiom_metal_kernels_start" >> ${METAL_EMBED_ASM}
            COMMAND echo "_axiom_metal_kernels_start:" >> ${METAL_EMBED_ASM}
            COMMAND echo ".incbin \\\"${METAL_LIBRARY_TEMP}\\\"" >> ${METAL_EMBED_ASM}
            COMMAND echo ".globl _axiom_metal_kernels_end" >> ${METAL_EMBED_ASM}
            COMMAND echo "_axiom_metal_kernels_end:" >> ${METAL_EMBED_ASM}
            DEPENDS ${METAL_LIBRARY_TEMP}
            COMMENT "Embedding Metal library into assembly"
        )
        target_sources(axiom PRIVATE ${METAL_EMBED_ASM})
    else()
        # Compile Metal kernels to a separate file (for development)
        set(METAL_KERNELS_FILE ${CMAKE_CURRENT_SOURCE_DIR}/src/backends/metal/kernels.metal)
        set(METAL_LIBRARY_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/default.metallib)

        add_custom_command(
            OUTPUT ${METAL_LIBRARY_OUTPUT}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND xcrun -sdk macosx metal -c ${METAL_KERNELS_FILE} -o ${CMAKE_CURRENT_BINARY_DIR}/kernels.air
            COMMAND xcrun -sdk macosx metallib ${CMAKE_CURRENT_BINARY_DIR}/kernels.air -o ${METAL_LIBRARY_OUTPUT}
            DEPENDS ${METAL_KERNELS_FILE}
            COMMENT "Compiling Metal kernels"
        )

        add_custom_target(
            axiom_metal_kernels ALL
            DEPENDS ${METAL_LIBRARY_OUTPUT}
        )
        add_dependencies(axiom axiom_metal_kernels)
    endif()
endif()

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
        $<BUILD_INTERFACE:${half_SOURCE_DIR}/include>
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
            ${MPS_FRAMEWORK}
            ${MPSGRAPH_FRAMEWORK}
    )

    # Enable Metal support
    target_compile_definitions(axiom PUBLIC AXIOM_METAL_SUPPORT)
endif()

# Apple Accelerate framework linkage
if(APPLE AND AXIOM_HAS_ACCELERATE)
    target_link_libraries(axiom PUBLIC ${ACCELERATE_FRAMEWORK})
    target_compile_definitions(axiom PUBLIC AXIOM_USE_ACCELERATE)
endif()

# Fast math for aggressive floating-point optimization in Release builds
# Use -fno-finite-math-only to allow proper NaN/Inf detection (for isnan, isinf, isfinite)
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        target_compile_options(axiom PRIVATE
            -ffast-math
            -fno-finite-math-only  # Allow proper NaN/Inf handling
        )
    endif()
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
)

if(APPLE AND AXIOM_HAS_METAL AND NOT AXIOM_EMBED_METAL_LIBRARY)
    install(
        FILES ${METAL_LIBRARY_OUTPUT}
        DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
endif()

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
message(STATUS "  Half library: ${half_SOURCE_DIR}")
if(APPLE)
    if(AXIOM_HAS_METAL)
        message(STATUS "  Metal support: YES")
    else()
        message(STATUS "  Metal support: NO (frameworks not found)")
    endif()
    if(AXIOM_HAS_ACCELERATE)
        message(STATUS "  Accelerate support: YES (BLAS/vDSP)")
    else()
        message(STATUS "  Accelerate support: NO")
    endif()
    message(STATUS "  Target architecture: ${CMAKE_SYSTEM_PROCESSOR}")
else()
    message(STATUS "  Metal support: NO (not on Apple platform)")
    message(STATUS "  Accelerate support: NO (not on Apple platform)")
endif()
message(STATUS "")