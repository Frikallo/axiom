# ============================================================================
# AxiomCUDA.cmake — CUDA backend detection, linking, and distribution bundling
# ============================================================================
#
# Inputs (cache options set in top-level CMakeLists.txt):
#   AXIOM_CUDA_BACKEND  — ON to attempt CUDA detection
#   AXIOM_DIST_BUILD    — ON for distribution builds (broad arch list)
#   AXIOM_CUDA_DIST     — ON to bundle CUDA shared libraries
#
# Outputs:
#   AXIOM_HAS_CUDA      — TRUE if CUDA backend is available
#
# Requires CUDA Toolkit >= 12.8 (for Blackwell sm_100/120 support and
# nvJitLink availability).
#
# NOTE: This file is include()'d, so all variables are set in the caller's
# scope directly. return() exits only this file, not the caller.

# -- Functions (defined first so they exist even when CUDA is disabled) -------

# Link all required CUDA libraries to a target, including transitive
# dependencies (cublasLt, cusparse, nvJitLink) that cusolver/cublas/nvrtc
# need at runtime.
function(axiom_cuda_link_libraries target)
    target_link_libraries(${target} PUBLIC
        CUDA::cudart
        CUDA::cublas
        CUDA::cublasLt
        CUDA::cusolver
        CUDA::cusparse
        CUDA::cufft
        CUDA::nvrtc
        CUDA::nvJitLink
        CUDA::cuda_driver
    )
    target_compile_definitions(${target} PUBLIC AXIOM_CUDA_SUPPORT)
endfunction()

# Bundle CUDA shared libraries into the install tree for distribution.
# NOT bundled: libcuda.so (user's NVIDIA driver provides this).
# Requires axiom_bundle_shared_lib() from the caller.
function(axiom_cuda_bundle_libraries)
    if(NOT AXIOM_CUDA_DIST OR NOT AXIOM_HAS_CUDA)
        return()
    endif()

    set(_cuda_libs cudart cublas cublasLt cusolver cusparse cufft nvrtc nvJitLink)
    foreach(_lib ${_cuda_libs})
        find_library(_cuda_lib_${_lib}
            NAMES ${_lib}
            HINTS ${CUDAToolkit_LIBRARY_DIR}
            NO_DEFAULT_PATH
        )
        if(_cuda_lib_${_lib})
            axiom_bundle_shared_lib("${_cuda_lib_${_lib}}" "CUDA")
        else()
            message(WARNING "CUDA bundle: ${_lib} not found in ${CUDAToolkit_LIBRARY_DIR}")
        endif()
    endforeach()
endfunction()

# -- Detection ----------------------------------------------------------------

set(AXIOM_HAS_CUDA OFF)
set(AXIOM_CUDA_MIN_VERSION "12.8")

if(NOT AXIOM_CUDA_BACKEND)
    return()
endif()

find_package(CUDAToolkit QUIET)
if(NOT CUDAToolkit_FOUND)
    message(STATUS "CUDA backend: disabled (CUDA toolkit not found)")
    return()
endif()

if(CUDAToolkit_VERSION VERSION_LESS "${AXIOM_CUDA_MIN_VERSION}")
    message(STATUS "CUDA backend: disabled (found ${CUDAToolkit_VERSION}, need >= ${AXIOM_CUDA_MIN_VERSION})")
    return()
endif()

# -- Enable CUDA language -----------------------------------------------------
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# -- Architecture selection ---------------------------------------------------
if(AXIOM_DIST_BUILD)
    # Broad set for distribution: Volta through Blackwell
    set(CMAKE_CUDA_ARCHITECTURES "70;80;86;89;90;100;120")
else()
    # Dev builds: detect native GPU, cap to toolkit max
    set(_gpu_arch 120)
    find_program(_NVIDIA_SMI nvidia-smi)
    if(_NVIDIA_SMI)
        execute_process(
            COMMAND ${_NVIDIA_SMI}
                --query-gpu=compute_cap --format=csv,noheader,nounits
            OUTPUT_VARIABLE _smi_out
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
            RESULT_VARIABLE _smi_rc)
        if(_smi_rc EQUAL 0 AND _smi_out)
            string(REGEX MATCH "[0-9]+\\.[0-9]+" _cap "${_smi_out}")
            string(REPLACE "." "" _gpu_arch "${_cap}")
            if(_gpu_arch GREATER 120)
                message(STATUS "CUDA: GPU arch ${_gpu_arch} capped to 120 (toolkit limit)")
                set(_gpu_arch 120)
            endif()
        endif()
    endif()
    set(CMAKE_CUDA_ARCHITECTURES "${_gpu_arch}")
endif()

set(AXIOM_HAS_CUDA ON)
message(STATUS "CUDA backend: enabled (toolkit ${CUDAToolkit_VERSION}, architectures: ${CMAKE_CUDA_ARCHITECTURES})")
