# FindLibOMP.cmake
# Find and optionally bundle libomp for distribution
#
# Sets:
#   LibOMP_FOUND
#   LibOMP_INCLUDE_DIR
#   LibOMP_LIBRARY
#   LibOMP_LIBRARY_PATH (full path to dylib/so for bundling)

if(APPLE)
    # Check for Homebrew libomp
    execute_process(
        COMMAND brew --prefix libomp
        OUTPUT_VARIABLE LIBOMP_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE LIBOMP_RESULT
        ERROR_QUIET
    )

    if(LIBOMP_RESULT EQUAL 0 AND EXISTS "${LIBOMP_PREFIX}")
        set(LibOMP_INCLUDE_DIR "${LIBOMP_PREFIX}/include")
        set(LibOMP_LIBRARY "${LIBOMP_PREFIX}/lib/libomp.dylib")
        set(LibOMP_LIBRARY_PATH "${LIBOMP_PREFIX}/lib/libomp.dylib")
        set(LibOMP_FOUND TRUE)
    endif()
else()
    # Linux: Find system libomp or libgomp
    find_library(LibOMP_LIBRARY
        NAMES omp gomp
        PATHS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu
    )
    find_path(LibOMP_INCLUDE_DIR
        NAMES omp.h
        PATHS /usr/include /usr/local/include
    )
    if(LibOMP_LIBRARY AND LibOMP_INCLUDE_DIR)
        set(LibOMP_FOUND TRUE)
        get_filename_component(LibOMP_LIBRARY_PATH "${LibOMP_LIBRARY}" REALPATH)
    endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LibOMP
    REQUIRED_VARS LibOMP_LIBRARY LibOMP_INCLUDE_DIR
)
