# FindOpenBLAS.cmake
# -------------------
# Find the OpenBLAS library
#
# This module defines the following variables:
#   OpenBLAS_FOUND        - True if OpenBLAS was found
#   OpenBLAS_INCLUDE_DIRS - Include directories for OpenBLAS
#   OpenBLAS_LIBRARIES    - Libraries to link against
#   OpenBLAS_VERSION      - Version of OpenBLAS (if available)
#
# This module accepts the following variables:
#   OPENBLAS_ROOT         - Hint for the installation prefix
#   OpenBLAS_ROOT         - Alternative hint for the installation prefix

# Check for environment variable hints
if(DEFINED ENV{OPENBLAS_HOME})
    set(_openblas_hints $ENV{OPENBLAS_HOME})
elseif(DEFINED ENV{OpenBLAS_HOME})
    set(_openblas_hints $ENV{OpenBLAS_HOME})
elseif(DEFINED OPENBLAS_ROOT)
    set(_openblas_hints ${OPENBLAS_ROOT})
elseif(DEFINED OpenBLAS_ROOT)
    set(_openblas_hints ${OpenBLAS_ROOT})
endif()

# Common search paths
set(_openblas_search_paths
    ${_openblas_hints}
    /usr
    /usr/local
    /opt/OpenBLAS
    /opt/local  # MacPorts
    /opt/homebrew  # Homebrew on Apple Silicon
    /usr/local/opt/openblas  # Homebrew on Intel Mac
)

# Find the include directory
find_path(OpenBLAS_INCLUDE_DIR
    NAMES cblas.h openblas/cblas.h
    HINTS ${_openblas_search_paths}
    PATH_SUFFIXES include include/openblas
    DOC "OpenBLAS include directory"
)

# Find the library
find_library(OpenBLAS_LIBRARY
    NAMES openblas libopenblas
    HINTS ${_openblas_search_paths}
    PATH_SUFFIXES lib lib64 lib/openblas
    DOC "OpenBLAS library"
)

# Handle cblas.h being in openblas subdirectory
if(OpenBLAS_INCLUDE_DIR AND EXISTS "${OpenBLAS_INCLUDE_DIR}/openblas/cblas.h")
    set(OpenBLAS_INCLUDE_DIR "${OpenBLAS_INCLUDE_DIR}/openblas")
endif()

# Try to extract version from openblas_config.h
if(OpenBLAS_INCLUDE_DIR AND EXISTS "${OpenBLAS_INCLUDE_DIR}/openblas_config.h")
    file(STRINGS "${OpenBLAS_INCLUDE_DIR}/openblas_config.h" _openblas_version_line
         REGEX "^#define[ \t]+OPENBLAS_VERSION[ \t]+\"[^\"]*\"")
    if(_openblas_version_line)
        string(REGEX REPLACE "^#define[ \t]+OPENBLAS_VERSION[ \t]+\"([^\"]*)\".*" "\\1"
               OpenBLAS_VERSION "${_openblas_version_line}")
    endif()
endif()

# Standard CMake find_package handling
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OpenBLAS
    REQUIRED_VARS OpenBLAS_LIBRARY OpenBLAS_INCLUDE_DIR
    VERSION_VAR OpenBLAS_VERSION
)

# Set output variables
if(OpenBLAS_FOUND)
    set(OpenBLAS_LIBRARIES ${OpenBLAS_LIBRARY})
    set(OpenBLAS_INCLUDE_DIRS ${OpenBLAS_INCLUDE_DIR})

    # Create an imported target for modern CMake usage
    if(NOT TARGET OpenBLAS::OpenBLAS)
        add_library(OpenBLAS::OpenBLAS UNKNOWN IMPORTED)
        set_target_properties(OpenBLAS::OpenBLAS PROPERTIES
            IMPORTED_LOCATION "${OpenBLAS_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${OpenBLAS_INCLUDE_DIR}"
        )
    endif()
endif()

# Hide internal variables
mark_as_advanced(OpenBLAS_INCLUDE_DIR OpenBLAS_LIBRARY)
