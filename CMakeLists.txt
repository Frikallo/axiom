cmake_minimum_required(VERSION 3.20)

project(Axiom
    VERSION 0.1.0
    DESCRIPTION "High-Performance Cross-Platform Tensor Library"
    LANGUAGES CXX
)

# C++20 is required
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Add cmake modules path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# ============================================================================
# Platform Detection
# ============================================================================
if(WIN32)
    set(AXIOM_PLATFORM "windows")
elseif(APPLE)
    set(AXIOM_PLATFORM "macos")
else()
    set(AXIOM_PLATFORM "linux")
endif()

# ============================================================================
# Build Options
# ============================================================================
option(AXIOM_BUILD_TESTS "Build unit tests" ON)
option(AXIOM_BUILD_EXAMPLES "Build examples" ON)
option(AXIOM_BUILD_BENCHMARKS "Build benchmarks" OFF)
option(AXIOM_NATIVE_ARCH "Use -march=native for local builds (not portable)" OFF)
option(AXIOM_USE_OPENMP "Enable OpenMP parallelization" ON)
option(AXIOM_DIST_BUILD "Bundle dependencies for distribution" OFF)

# BLAS backend selection
set(AXIOM_BLAS_BACKEND "auto" CACHE STRING "BLAS backend: auto, accelerate, openblas, native")
set_property(CACHE AXIOM_BLAS_BACKEND PROPERTY STRINGS auto accelerate openblas native)

# LAPACK backend selection (follows BLAS backend - Accelerate and OpenBLAS include LAPACK)
set(AXIOM_LAPACK_BACKEND "auto" CACHE STRING "LAPACK backend: auto, accelerate, openblas, native")
set_property(CACHE AXIOM_LAPACK_BACKEND PROPERTY STRINGS auto accelerate openblas native)

# ============================================================================
# Compiler Options
# ============================================================================
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    add_compile_options(-Wall -Wextra -Wpedantic)
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(-O3 -DNDEBUG)
        # Architecture-specific optimizations
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
            if(APPLE)
                add_compile_options(-mcpu=apple-m1)
            else()
                add_compile_options(-march=armv8-a)
            endif()
        elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
            # Use native arch only if requested (not portable)
            if(AXIOM_NATIVE_ARCH)
                add_compile_options(-march=native)
            endif()
        endif()
    endif()
elseif(MSVC)
    add_compile_options(/W4 /EHsc /utf-8)
    # Disable some noisy warnings
    add_compile_options(/wd4244 /wd4267 /wd4996)
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(/O2 /DNDEBUG)
    endif()
endif()

# Windows-specific settings
if(WIN32)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
    add_compile_definitions(_USE_MATH_DEFINES)
    add_compile_definitions(NOMINMAX)
endif()

# Linux-specific settings
if(UNIX AND NOT APPLE)
    find_package(Threads REQUIRED)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

# ============================================================================
# OpenMP Detection
# ============================================================================
set(AXIOM_HAS_OPENMP OFF)
if(AXIOM_USE_OPENMP)
    # On macOS with Apple Clang, we need to use Homebrew's libomp
    if(APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        # Check for Homebrew libomp
        execute_process(
            COMMAND brew --prefix libomp
            OUTPUT_VARIABLE LIBOMP_PREFIX
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE LIBOMP_RESULT
            ERROR_QUIET
        )
        if(LIBOMP_RESULT EQUAL 0 AND EXISTS "${LIBOMP_PREFIX}")
            set(AXIOM_HAS_OPENMP ON)
            set(OpenMP_omp_LIBRARY "${LIBOMP_PREFIX}/lib/libomp.dylib")
            set(OpenMP_CXX_INCLUDE_DIR "${LIBOMP_PREFIX}/include")
            message(STATUS "OpenMP: enabled (Homebrew libomp at ${LIBOMP_PREFIX})")
        else()
            message(STATUS "OpenMP: disabled (Homebrew libomp not found, install with: brew install libomp)")
        endif()
    else()
        # Standard OpenMP detection for other platforms
        find_package(OpenMP)
        if(OpenMP_CXX_FOUND)
            set(AXIOM_HAS_OPENMP ON)
            message(STATUS "OpenMP: enabled (${OpenMP_CXX_VERSION})")
        else()
            message(STATUS "OpenMP: disabled (not found)")
        endif()
    endif()
endif()

# ============================================================================
# BLAS Backend Selection
# ============================================================================
set(AXIOM_HAS_ACCELERATE OFF)
set(AXIOM_HAS_OPENBLAS OFF)
set(AXIOM_BLAS_IMPL "native")

if(AXIOM_BLAS_BACKEND STREQUAL "auto")
    # Auto-detect best available backend
    if(APPLE)
        find_library(ACCELERATE_FRAMEWORK Accelerate)
        if(ACCELERATE_FRAMEWORK)
            set(AXIOM_HAS_ACCELERATE ON)
            set(AXIOM_BLAS_IMPL "accelerate")
            message(STATUS "BLAS backend: Accelerate (auto-detected)")
        endif()
    else()
        # Linux/Windows: Try OpenBLAS
        find_package(OpenBLAS QUIET)
        if(OpenBLAS_FOUND)
            set(AXIOM_HAS_OPENBLAS ON)
            set(AXIOM_BLAS_IMPL "openblas")
            message(STATUS "BLAS backend: OpenBLAS (auto-detected)")
        else()
            message(STATUS "BLAS backend: Native (OpenBLAS not found)")
        endif()
    endif()
elseif(AXIOM_BLAS_BACKEND STREQUAL "accelerate")
    if(APPLE)
        find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
        set(AXIOM_HAS_ACCELERATE ON)
        set(AXIOM_BLAS_IMPL "accelerate")
        message(STATUS "BLAS backend: Accelerate (requested)")
    else()
        message(FATAL_ERROR "Accelerate framework is only available on macOS")
    endif()
elseif(AXIOM_BLAS_BACKEND STREQUAL "openblas")
    find_package(OpenBLAS REQUIRED)
    set(AXIOM_HAS_OPENBLAS ON)
    set(AXIOM_BLAS_IMPL "openblas")
    message(STATUS "BLAS backend: OpenBLAS (requested)")
elseif(AXIOM_BLAS_BACKEND STREQUAL "native")
    message(STATUS "BLAS backend: Native (requested)")
endif()

# ============================================================================
# Platform-specific configurations (Metal on macOS)
# ============================================================================
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
endif()

# External dependencies
include(FetchContent)

# ============================================================================
# Google Benchmark (for benchmarks only)
# ============================================================================
if(AXIOM_BUILD_BENCHMARKS)
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_DOWNLOAD_DEPENDENCIES OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)

    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/benchmark/CMakeLists.txt")
        add_subdirectory(third_party/benchmark)
    else()
        message(FATAL_ERROR "Google Benchmark not found. Run: git submodule add https://github.com/google/benchmark.git third_party/benchmark")
    endif()
endif()

# ============================================================================
# Source Files
# ============================================================================
set(AXIOM_SOURCES
    # Tensor core
    src/tensor/tensor.cpp
    src/tensor/shape.cpp
    src/tensor/storage.cpp
    src/tensor/dtype.cpp
    src/tensor/debug.cpp
    src/tensor/type_conversion.cpp

    # IO
    src/io/io.cpp
    src/io/io_flatbuffers.cpp
    src/io/io_numpy.cpp
    src/tensor/einops.cpp
    src/tensor/operations.cpp
    src/tensor/random.cpp
    src/tensor/system.cpp
    src/tensor/indexing.cpp

    # CPU backend
    src/backends/cpu/cpu_storage.cpp
    src/backends/cpu/cpu_operations.cpp

    # BLAS backend abstraction (always included)
    src/backends/cpu/blas/blas_backend.cpp
    src/backends/cpu/blas/blas_native.cpp

    # LAPACK backend abstraction (always included)
    src/backends/cpu/lapack/lapack_backend.cpp
    src/backends/cpu/lapack/lapack_native.cpp

    # Linear algebra operations
    src/tensor/linalg.cpp

    # Parallel utilities
    src/parallel/parallel.cpp
)

# Add platform-specific BLAS backends
if(AXIOM_HAS_ACCELERATE)
    list(APPEND AXIOM_SOURCES
        src/backends/cpu/cpu_accelerate.cpp
        src/backends/cpu/blas/blas_accelerate.cpp
        src/backends/cpu/lapack/lapack_accelerate.cpp
    )
endif()

if(AXIOM_HAS_OPENBLAS)
    list(APPEND AXIOM_SOURCES
        src/backends/cpu/blas/blas_openblas.cpp
        src/backends/cpu/lapack/lapack_openblas.cpp
    )
endif()

# Add Metal backend if available
if(APPLE AND AXIOM_HAS_METAL)
    list(APPEND AXIOM_SOURCES
        src/backends/metal/metal_storage.mm
        src/backends/metal/metal_operations.mm
        src/backends/metal/metal_common.mm
        src/backends/metal/mpsgraph_operations.mm
        src/backends/metal/graph_cache.mm
        src/backends/metal/kernels.metal
    )

    set_source_files_properties(
        src/backends/metal/metal_storage.mm
        src/backends/metal/metal_operations.mm
        src/backends/metal/metal_common.mm
        src/backends/metal/mpsgraph_operations.mm
        src/backends/metal/graph_cache.mm
        src/backends/metal/kernels.metal
        PROPERTIES
            COMPILE_FLAGS "-fobjc-arc"
    )
endif()

# Build as shared library on all platforms
# On Windows, auto-export all symbols to generate .lib import library
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
add_library(axiom SHARED ${AXIOM_SOURCES})

# ============================================================================
# Metal Library Embedding (macOS only)
# ============================================================================
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

# ============================================================================
# Target Properties
# ============================================================================
set_target_properties(axiom PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    EXPORT_NAME axiom
)

# Include directories
target_include_directories(axiom
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party/xsimd/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party/FXdiv/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party/FP16/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party/flatbuffers/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/backends
        ${CMAKE_CURRENT_SOURCE_DIR}/src/backends/cpu
        ${CMAKE_CURRENT_SOURCE_DIR}/src/backends/cpu/lapack
)

# ============================================================================
# Linking
# ============================================================================

# Metal framework (macOS only)
if(APPLE AND AXIOM_HAS_METAL)
    target_link_libraries(axiom
        PUBLIC
            ${METAL_FRAMEWORK}
            ${FOUNDATION_FRAMEWORK}
            ${MPS_FRAMEWORK}
            ${MPSGRAPH_FRAMEWORK}
    )
    target_compile_definitions(axiom PUBLIC AXIOM_METAL_SUPPORT)
endif()

# BLAS backend linking
if(AXIOM_HAS_ACCELERATE)
    target_link_libraries(axiom PUBLIC ${ACCELERATE_FRAMEWORK})
    target_compile_definitions(axiom PUBLIC AXIOM_USE_ACCELERATE)
endif()

if(AXIOM_HAS_OPENBLAS)
    target_link_libraries(axiom PUBLIC OpenBLAS::OpenBLAS)
    target_compile_definitions(axiom PUBLIC AXIOM_USE_OPENBLAS)
    # On Linux, LAPACKE (C interface) is a separate library from OpenBLAS
    if(UNIX AND NOT APPLE)
        find_library(LAPACKE_LIBRARY lapacke)
        if(LAPACKE_LIBRARY)
            target_link_libraries(axiom PUBLIC ${LAPACKE_LIBRARY})
            message(STATUS "LAPACKE: found at ${LAPACKE_LIBRARY}")
        else()
            message(WARNING "LAPACKE library not found - linear algebra operations may fail")
        endif()
    endif()
endif()

# Linux threading
if(UNIX AND NOT APPLE)
    target_link_libraries(axiom PUBLIC Threads::Threads)
endif()

# OpenMP linking
if(AXIOM_HAS_OPENMP)
    if(APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        # Manual linking for Homebrew libomp on macOS
        # Apple Clang requires -Xpreprocessor -fopenmp (as separate flags)
        target_compile_options(axiom PUBLIC -Xpreprocessor -fopenmp)
        target_include_directories(axiom PUBLIC ${OpenMP_CXX_INCLUDE_DIR})
        target_link_libraries(axiom PUBLIC ${OpenMP_omp_LIBRARY})
    else()
        # Standard OpenMP target for other platforms
        target_link_libraries(axiom PUBLIC OpenMP::OpenMP_CXX)
    endif()
    target_compile_definitions(axiom PUBLIC AXIOM_USE_OPENMP)
endif()

# XSIMD for cross-platform SIMD operations (always enabled)
target_compile_definitions(axiom PUBLIC AXIOM_USE_XSIMD)

# ============================================================================
# Floating-point optimization flags
# ============================================================================
# Note: -ffast-math and -fassociative-math break XSIMD transcendental functions
# (exp, log, sin, etc.) on ARM NEON. Use safer flags that preserve math correctness.
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        target_compile_options(axiom PRIVATE
            -fno-math-errno        # Don't set errno for math functions
            -fno-trapping-math     # Assume no FP exceptions
            -fno-signed-zeros      # Allow optimizations ignoring sign of zero
            -freciprocal-math      # Allow x/y -> x * (1/y)
            # Note: -fassociative-math breaks XSIMD exp/log on ARM, do not use
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

# ============================================================================
# Tests, Examples, Benchmarks
# ============================================================================
if(AXIOM_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests")
    enable_testing()
    add_subdirectory(tests)
endif()

if(AXIOM_BUILD_EXAMPLES AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples")
    add_subdirectory(examples)
endif()

if(AXIOM_BUILD_BENCHMARKS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks")
    add_subdirectory(benchmarks)
endif()

# ============================================================================
# Installation
# ============================================================================
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

# ============================================================================
# Configuration Summary
# ============================================================================
message(STATUS "")
message(STATUS "Axiom Configuration Summary:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  Platform: ${AXIOM_PLATFORM}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "  Build tests: ${AXIOM_BUILD_TESTS}")
message(STATUS "  Build examples: ${AXIOM_BUILD_EXAMPLES}")
message(STATUS "  Build benchmarks: ${AXIOM_BUILD_BENCHMARKS}")
message(STATUS "")
message(STATUS "  BLAS backend: ${AXIOM_BLAS_IMPL}")
message(STATUS "  LAPACK backend: ${AXIOM_BLAS_IMPL}")
if(AXIOM_HAS_OPENMP)
    message(STATUS "  OpenMP: YES (${OpenMP_CXX_VERSION})")
else()
    message(STATUS "  OpenMP: NO")
endif()
if(APPLE)
    if(AXIOM_HAS_METAL)
        message(STATUS "  Metal support: YES")
    else()
        message(STATUS "  Metal support: NO (frameworks not found)")
    endif()
else()
    message(STATUS "  Metal support: NO (not on Apple platform)")
endif()
message(STATUS "  Target architecture: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "  SIMD: Use axiom::cpu_info::print_simd_info() at runtime for details")
if(AXIOM_DIST_BUILD)
    message(STATUS "  Distribution build: YES")
endif()
message(STATUS "")

# ============================================================================
# Distribution Build - Bundle Dependencies
# ============================================================================
if(AXIOM_DIST_BUILD)
    # Bundle OpenMP library
    if(AXIOM_HAS_OPENMP)
        if(APPLE)
            # Get the libomp path and install it
            if(EXISTS "${OpenMP_omp_LIBRARY}")
                # Copy libomp.dylib to lib directory
                install(FILES "${OpenMP_omp_LIBRARY}"
                    DESTINATION ${CMAKE_INSTALL_LIBDIR}
                    COMPONENT runtime
                )
                # Fix the install name in axiom library to use @rpath
                set_target_properties(axiom PROPERTIES
                    INSTALL_RPATH "@loader_path"
                    BUILD_WITH_INSTALL_RPATH OFF
                )
            endif()
        elseif(UNIX AND NOT WIN32)
            # Linux: Bundle libgomp or libomp
            find_library(OPENMP_RUNTIME_LIB NAMES gomp omp)
            if(OPENMP_RUNTIME_LIB)
                get_filename_component(OPENMP_RUNTIME_REALPATH "${OPENMP_RUNTIME_LIB}" REALPATH)
                install(FILES "${OPENMP_RUNTIME_REALPATH}"
                    DESTINATION ${CMAKE_INSTALL_LIBDIR}
                    COMPONENT runtime
                )
            endif()
        elseif(WIN32)
            # Windows: Bundle OpenMP runtime DLL
            # For MSVC, this is vcomp140.dll; for Clang/LLVM it's libomp.dll
            if(MSVC)
                # MSVC OpenMP runtime is part of Visual C++ Redistributable
                # Users need VC++ Redist installed, or we bundle vcomp140.dll
                get_filename_component(MSVC_DIR "${CMAKE_CXX_COMPILER}" DIRECTORY)
                find_file(VCOMP_DLL
                    NAMES vcomp140.dll
                    PATHS
                        "${MSVC_DIR}"
                        "${MSVC_DIR}/../../../redist/x64/Microsoft.VC143.OPENMP"
                        "${MSVC_DIR}/../../../redist/x64/Microsoft.VC142.OPENMP"
                        "$ENV{VCToolsRedistDir}/x64/Microsoft.VC143.OPENMP"
                        "$ENV{VCToolsRedistDir}/x64/Microsoft.VC142.OPENMP"
                    NO_DEFAULT_PATH
                )
                if(VCOMP_DLL)
                    install(FILES "${VCOMP_DLL}"
                        DESTINATION ${CMAKE_INSTALL_BINDIR}
                        COMPONENT runtime
                    )
                endif()
            else()
                # Clang/LLVM or MinGW
                find_library(OPENMP_RUNTIME_DLL NAMES libomp libgomp-1)
                if(OPENMP_RUNTIME_DLL)
                    get_filename_component(OMP_DIR "${OPENMP_RUNTIME_DLL}" DIRECTORY)
                    file(GLOB OMP_DLLS "${OMP_DIR}/libomp*.dll" "${OMP_DIR}/libgomp*.dll")
                    foreach(DLL ${OMP_DLLS})
                        install(FILES "${DLL}"
                            DESTINATION ${CMAKE_INSTALL_BINDIR}
                            COMPONENT runtime
                        )
                    endforeach()
                endif()
            endif()
        endif()
    endif()

    # Bundle OpenBLAS for non-macOS platforms
    if(AXIOM_HAS_OPENBLAS AND NOT APPLE)
        get_target_property(OPENBLAS_LIB_LOCATION OpenBLAS::OpenBLAS IMPORTED_LOCATION)
        if(OPENBLAS_LIB_LOCATION)
            get_filename_component(OPENBLAS_REALPATH "${OPENBLAS_LIB_LOCATION}" REALPATH)
            if(WIN32)
                # Windows: Bundle OpenBLAS DLL
                get_filename_component(OPENBLAS_DIR "${OPENBLAS_REALPATH}" DIRECTORY)
                # Look for DLL in bin/ or same directory
                find_file(OPENBLAS_DLL
                    NAMES libopenblas.dll openblas.dll
                    PATHS
                        "${OPENBLAS_DIR}"
                        "${OPENBLAS_DIR}/../bin"
                    NO_DEFAULT_PATH
                )
                if(OPENBLAS_DLL)
                    install(FILES "${OPENBLAS_DLL}"
                        DESTINATION ${CMAKE_INSTALL_BINDIR}
                        COMPONENT runtime
                    )
                endif()
            else()
                # Linux: Bundle .so files
                install(FILES "${OPENBLAS_REALPATH}"
                    DESTINATION ${CMAKE_INSTALL_LIBDIR}
                    COMPONENT runtime
                )
                # Also copy any versioned symlinks
                get_filename_component(OPENBLAS_DIR "${OPENBLAS_REALPATH}" DIRECTORY)
                file(GLOB OPENBLAS_LIBS "${OPENBLAS_DIR}/libopenblas*.so*")
                foreach(LIB ${OPENBLAS_LIBS})
                    install(FILES "${LIB}"
                        DESTINATION ${CMAKE_INSTALL_LIBDIR}
                        COMPONENT runtime
                    )
                endforeach()
            endif()
        endif()
    endif()

    # Set RPATH for the installed library (Unix only)
    if(UNIX AND NOT APPLE)
        set_target_properties(axiom PROPERTIES
            INSTALL_RPATH "$ORIGIN"
            BUILD_WITH_INSTALL_RPATH OFF
        )
    elseif(APPLE)
        set_target_properties(axiom PROPERTIES
            INSTALL_RPATH "@loader_path"
            BUILD_WITH_INSTALL_RPATH OFF
        )
    endif()
endif()

# ============================================================================
# CPack Configuration for Distribution
# ============================================================================
if(AXIOM_DIST_BUILD)
    set(CPACK_PACKAGE_NAME "axiom")
    set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
    set(CPACK_PACKAGE_VENDOR "Axiom")
    set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "High-Performance Tensor Library")

    if(APPLE)
        set(CPACK_GENERATOR "TGZ")
        set(CPACK_SYSTEM_NAME "macos-${CMAKE_SYSTEM_PROCESSOR}")
    elseif(WIN32)
        set(CPACK_GENERATOR "ZIP")
        set(CPACK_SYSTEM_NAME "windows-${CMAKE_SYSTEM_PROCESSOR}")
    else()
        set(CPACK_GENERATOR "TGZ")
        set(CPACK_SYSTEM_NAME "linux-${CMAKE_SYSTEM_PROCESSOR}")
    endif()

    include(CPack)
endif()
