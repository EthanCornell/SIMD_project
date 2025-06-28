@PACKAGE_INIT@

# SIMD Library Configuration File
# This file sets up the SIMD library for use with find_package()

include(CMakeFindDependencyMacro)

# Find required dependencies
if(@SIMD_ENABLE_OPENMP@)
    find_dependency(OpenMP REQUIRED)
endif()

# Include the targets file
include("${CMAKE_CURRENT_LIST_DIR}/simd_libraryTargets.cmake")

# Set up imported targets
if(NOT TARGET simd::simd_library)
    add_library(simd::simd_library ALIAS simd_library)
endif()

# Provide information about the package
set(SIMD_LIBRARY_VERSION "@PROJECT_VERSION@")
set(SIMD_LIBRARY_VERSION_MAJOR "@PROJECT_VERSION_MAJOR@")
set(SIMD_LIBRARY_VERSION_MINOR "@PROJECT_VERSION_MINOR@")
set(SIMD_LIBRARY_VERSION_PATCH "@PROJECT_VERSION_PATCH@")

# Feature flags
set(SIMD_LIBRARY_OPENMP_ENABLED @SIMD_ENABLE_OPENMP@)
set(SIMD_LIBRARY_V2_ENABLED @SIMD_USE_V2@)

check_required_components(simd_library)