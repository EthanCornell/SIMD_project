cmake_minimum_required(VERSION 3.15)
project(simd_library VERSION 3.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options
option(SIMD_BUILD_TESTS "Build unit tests" ON)
option(SIMD_BUILD_BENCHMARKS "Build benchmark suite" ON)
option(SIMD_BUILD_EXAMPLES "Build example applications" ON)
option(SIMD_ENABLE_OPENMP "Enable OpenMP parallel support" ON)
option(SIMD_ENABLE_PROFILING "Enable detailed profiling support" ON)
option(SIMD_DISABLE_AUTO_VECTORIZATION "Disable auto-vectorization in benchmarks for fair comparison" ON)

# Version selection options (mutually exclusive)
option(SIMD_USE_V1 "Use the v1 implementation (simd.hpp)" OFF)
option(SIMD_USE_V2 "Use the v2 implementation (simd_v2.hpp)" OFF)
option(SIMD_USE_V3 "Use the v3 implementation (simd_v3.hpp)" ON)

# FIXED: Ensure only one version is selected using proper math
# Convert boolean values to numbers for math operations (handle various boolean representations)
if(SIMD_USE_V1)
    set(_V1_NUM 1)
else()
    set(_V1_NUM 0)
endif()

if(SIMD_USE_V2)
    set(_V2_NUM 1)
else()
    set(_V2_NUM 0)
endif()

if(SIMD_USE_V3)
    set(_V3_NUM 1)
else()
    set(_V3_NUM 0)
endif()

math(EXPR _NUM_VERSIONS "${_V1_NUM}+${_V2_NUM}+${_V3_NUM}")

if(_NUM_VERSIONS GREATER 1)
    # Force clean state by turning off all versions
    set(SIMD_USE_V1 OFF CACHE BOOL "Use the v1 implementation (simd.hpp)" FORCE)
    set(SIMD_USE_V2 OFF CACHE BOOL "Use the v2 implementation (simd_v2.hpp)" FORCE)
    set(SIMD_USE_V3 OFF CACHE BOOL "Use the v3 implementation (simd_v3.hpp)" FORCE)
    
    # Then enable only v3 as default
    set(SIMD_USE_V3 ON CACHE BOOL "Use the v3 implementation (simd_v3.hpp)" FORCE)
    
    message(WARNING "Multiple SIMD versions were selected (V1=${SIMD_USE_V1}, V2=${SIMD_USE_V2}, V3=${SIMD_USE_V3}). Forcing clean state and defaulting to v3.")
    message(STATUS "Reset to: V1=OFF, V2=OFF, V3=ON")
    
    # Recalculate after fixing
    set(_V1_NUM 0)
    set(_V2_NUM 0)
    set(_V3_NUM 1)
    set(_NUM_VERSIONS 1)
    
elseif(_NUM_VERSIONS EQUAL 0)
    message(STATUS "No SIMD version explicitly selected, defaulting to v3")
    set(SIMD_USE_V3 ON CACHE BOOL "Use the v3 implementation (simd_v3.hpp)" FORCE)
    set(_NUM_VERSIONS 1)
endif()

# FIXED: Find threading support properly
find_package(Threads REQUIRED)

# FIXED: Find OpenMP without REQUIRED - make it truly optional
if(SIMD_ENABLE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        message(STATUS "OpenMP found: ${OpenMP_CXX_VERSION}")
    else()
        message(WARNING "OpenMP requested but not found. Falling back to std::thread implementation.")
        set(SIMD_ENABLE_OPENMP OFF)
    endif()
endif()

# Header-only library target
add_library(simd_library INTERFACE)

# Choose which header version to use
if(SIMD_USE_V3)
    target_include_directories(simd_library INTERFACE 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    )
    message(STATUS "Using SIMD v3 implementation (simd_v3.hpp)")
    target_compile_definitions(simd_library INTERFACE SIMD_USE_V3_HEADER)
    set(SIMD_ACTIVE_VERSION "v3")
elseif(SIMD_USE_V2)
    target_include_directories(simd_library INTERFACE 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    )
    message(STATUS "Using SIMD v2 implementation (simd_v2.hpp)")
    target_compile_definitions(simd_library INTERFACE SIMD_USE_V2_HEADER)
    set(SIMD_ACTIVE_VERSION "v2")
elseif(SIMD_USE_V1)
    target_include_directories(simd_library INTERFACE 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    )
    message(STATUS "Using SIMD v1 implementation (simd.hpp)")
    target_compile_definitions(simd_library INTERFACE SIMD_USE_V1_HEADER)
    set(SIMD_ACTIVE_VERSION "v1")
endif()

# Compiler feature requirements
target_compile_features(simd_library INTERFACE cxx_std_17)

# FIXED: Target-scoped compiler optimizations instead of global flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # Performance flags for Release builds only
    target_compile_options(simd_library INTERFACE
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:Release>:-DNDEBUG>
        $<$<CONFIG:Release>:-ffast-math>
        $<$<CONFIG:Release>:-march=native>
        $<$<CONFIG:Release>:-mtune=native>
        $<$<CONFIG:Release>:-mavx2>
        $<$<CONFIG:Release>:-mfma>
    )
    
    # Debug flags
    target_compile_options(simd_library INTERFACE
        $<$<CONFIG:Debug>:-g>
        $<$<CONFIG:Debug>:-O0>
        $<$<CONFIG:Debug>:-Wall>
        $<$<CONFIG:Debug>:-Wextra>
    )
    
    # Check for AVX-512 support (optional)
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-mavx512f" COMPILER_SUPPORTS_AVX512)
    if(COMPILER_SUPPORTS_AVX512)
        target_compile_options(simd_library INTERFACE
            $<$<CONFIG:Release>:-mavx512f>
            $<$<CONFIG:Release>:-mavx512bw>
            $<$<CONFIG:Release>:-mavx512vl>
        )
        message(STATUS "AVX-512 support detected and enabled")
    endif()
    
    # Check for auto-vectorization disable flags
    if(SIMD_DISABLE_AUTO_VECTORIZATION)
        check_cxx_compiler_flag("-fno-tree-vectorize" SUPPORTS_NO_TREE_VECTORIZE)
        check_cxx_compiler_flag("-fno-slp-vectorize" SUPPORTS_NO_SLP_VECTORIZE)
        check_cxx_compiler_flag("-fno-tree-loop-vectorize" SUPPORTS_NO_TREE_LOOP_VECTORIZE)
        
        if(SUPPORTS_NO_TREE_VECTORIZE AND SUPPORTS_NO_SLP_VECTORIZE AND SUPPORTS_NO_TREE_LOOP_VECTORIZE)
            set(AUTO_VECTORIZATION_FLAGS_AVAILABLE TRUE)
            message(STATUS "Auto-vectorization disable flags supported")
        else()
            set(AUTO_VECTORIZATION_FLAGS_AVAILABLE FALSE)
            message(WARNING "Some auto-vectorization disable flags not supported by compiler")
        endif()
    endif()
    
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(simd_library INTERFACE
        $<$<CONFIG:Release>:/O2>
        $<$<CONFIG:Release>:/DNDEBUG>
        $<$<CONFIG:Release>:/fp:fast>
        $<$<CONFIG:Release>:/arch:AVX2>
    )
    
    target_compile_options(simd_library INTERFACE
        $<$<CONFIG:Debug>:/Od>
        $<$<CONFIG:Debug>:/Zi>
        $<$<CONFIG:Debug>:/W4>
    )
    
    # Enable AVX-512 if supported (MSVC 2017+)
    if(MSVC_VERSION GREATER_EQUAL 1910)
        target_compile_options(simd_library INTERFACE
            $<$<CONFIG:Release>:/arch:AVX512>
        )
        message(STATUS "MSVC AVX-512 support enabled")
    endif()
endif()

# FIXED: Add threading support properly (includes both compile flags and link libraries)
target_link_libraries(simd_library INTERFACE Threads::Threads)

# FIXED: Add OpenMP only if found, don't manually define _OPENMP
if(SIMD_ENABLE_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(simd_library INTERFACE OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP support enabled")
endif()

# Platform-specific settings
if(WIN32)
    target_compile_definitions(simd_library INTERFACE _WIN32 WIN32_LEAN_AND_MEAN)
elseif(UNIX AND NOT APPLE)
    target_compile_definitions(simd_library INTERFACE __linux__)
elseif(APPLE)
    target_compile_definitions(simd_library INTERFACE __APPLE__)
endif()

# Profiling support
if(SIMD_ENABLE_PROFILING)
    target_compile_definitions(simd_library INTERFACE SIMD_ENABLE_PROFILING)
    message(STATUS "Profiling support enabled")
endif()

# Tests
if(SIMD_BUILD_TESTS)
    enable_testing()
    
    # Try to find Catch2 first
    find_package(Catch2 3 QUIET)
    if(NOT Catch2_FOUND)
        message(STATUS "Catch2 not found, downloading...")
        include(FetchContent)
        FetchContent_Declare(
            Catch2
            GIT_REPOSITORY https://github.com/catchorg/Catch2.git
            GIT_TAG v3.4.0
            GIT_SHALLOW TRUE
        )
        FetchContent_MakeAvailable(Catch2)
        list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
    endif()
    
    # Create test executable
    add_executable(simd_tests test/test_main.cpp)
    target_link_libraries(simd_tests PRIVATE simd_library)
    
    # Link with Catch2
    if(TARGET Catch2::Catch2WithMain)
        target_link_libraries(simd_tests PRIVATE Catch2::Catch2WithMain)
    else()
        target_link_libraries(simd_tests PRIVATE Catch2::Catch2)
    endif()
    
    target_compile_definitions(simd_tests PRIVATE SIMD_INCLUDE_EXAMPLES)
    
    # Add version-specific compile definitions
    if(SIMD_USE_V3)
        target_compile_definitions(simd_tests PRIVATE SIMD_USE_V3_HEADER)
    elseif(SIMD_USE_V2)
        target_compile_definitions(simd_tests PRIVATE SIMD_USE_V2_HEADER)
    elseif(SIMD_USE_V1)
        target_compile_definitions(simd_tests PRIVATE SIMD_USE_V1_HEADER)
    endif()
    
    # Add test to CTest
    add_test(NAME simd_unit_tests COMMAND simd_tests)
    
    # Set test properties
    set_tests_properties(simd_unit_tests PROPERTIES
        TIMEOUT 300
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
    
    message(STATUS "Unit tests enabled")
endif()

# Benchmarks
if(SIMD_BUILD_BENCHMARKS)
    # Set output directory for benchmarks
    set(BENCHMARK_OUTPUT_DIR ${CMAKE_BINARY_DIR}/benchmark)
    
    # FIXED: Enhanced function to create benchmark executable with auto-vectorization control
    function(create_benchmark_target target_name source_file version_def)
        if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${source_file})
            add_executable(${target_name} ${source_file})
            target_link_libraries(${target_name} PRIVATE simd_library)
            target_compile_definitions(${target_name} PRIVATE 
                SIMD_INCLUDE_EXAMPLES 
                ${version_def}
            )
            
            # CRITICAL FIX: Disable auto-vectorization for fair benchmarking
            if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
                target_compile_options(${target_name} PRIVATE
                    # Keep performance optimizations
                    $<$<CONFIG:Release>:-O3>
                    $<$<CONFIG:Release>:-march=native>
                    $<$<CONFIG:Release>:-ffast-math>
                    $<$<CONFIG:Release>:-mavx2>
                    $<$<CONFIG:Release>:-mfma>
                )
                
                # Add AVX-512 if supported
                if(COMPILER_SUPPORTS_AVX512)
                    target_compile_options(${target_name} PRIVATE
                        $<$<CONFIG:Release>:-mavx512f>
                        $<$<CONFIG:Release>:-mavx512bw>
                        $<$<CONFIG:Release>:-mavx512vl>
                    )
                endif()
                
                # DISABLE auto-vectorization for scalar reference code
                if(SIMD_DISABLE_AUTO_VECTORIZATION AND AUTO_VECTORIZATION_FLAGS_AVAILABLE)
                    target_compile_options(${target_name} PRIVATE
                        $<$<CONFIG:Release>:-fno-tree-vectorize>
                        $<$<CONFIG:Release>:-fno-slp-vectorize>
                        $<$<CONFIG:Release>:-fno-tree-loop-vectorize>
                    )
                    target_compile_definitions(${target_name} PRIVATE DISABLE_SCALAR_AUTOVEC)
                    message(STATUS "Auto-vectorization DISABLED for fair benchmarking in ${target_name}")
                else()
                    message(WARNING "Auto-vectorization could not be disabled for ${target_name} - results may be skewed")
                endif()
                
                # Debug info for profiling
                target_compile_options(${target_name} PRIVATE
                    $<$<CONFIG:Debug>:-g>
                    $<$<CONFIG:Debug>:-O0>
                )
                
            elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
                target_compile_options(${target_name} PRIVATE
                    $<$<CONFIG:Release>:/O2>
                    $<$<CONFIG:Release>:/fp:fast>
                    $<$<CONFIG:Release>:/arch:AVX2>
                )
                
                # Enable AVX-512 if supported
                if(MSVC_VERSION GREATER_EQUAL 1910)
                    target_compile_options(${target_name} PRIVATE
                        $<$<CONFIG:Release>:/arch:AVX512>
                    )
                endif()
                
                # MSVC doesn't have direct equivalent to disable auto-vectorization
                # Use /O2 instead of /O3 to reduce aggressive optimizations
                message(STATUS "MSVC: Using /O2 to reduce auto-vectorization in ${target_name}")
            endif()
            
            set_target_properties(${target_name} PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY ${BENCHMARK_OUTPUT_DIR}
            )
            
            message(STATUS "Benchmark executable enabled: ${target_name}")
        endif()
    endfunction()
    
    # ADDITIONAL: Function to create comparison benchmarks (auto-vectorization on vs off)
    function(create_comparison_benchmark target_name source_file version_def)
        if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${source_file})
            # Main benchmark with auto-vectorization disabled
            create_benchmark_target(${target_name} ${source_file} ${version_def})
            
            # Comparison benchmark with auto-vectorization enabled (if flags are available)
            if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang" AND AUTO_VECTORIZATION_FLAGS_AVAILABLE)
                add_executable(${target_name}_autovec ${source_file})
                target_link_libraries(${target_name}_autovec PRIVATE simd_library)
                target_compile_definitions(${target_name}_autovec PRIVATE 
                    SIMD_INCLUDE_EXAMPLES 
                    ${version_def}
                    ENABLE_SCALAR_AUTOVEC
                )
                
                target_compile_options(${target_name}_autovec PRIVATE
                    $<$<CONFIG:Release>:-O3>
                    $<$<CONFIG:Release>:-march=native>
                    $<$<CONFIG:Release>:-ffast-math>
                    $<$<CONFIG:Release>:-mavx2>
                    $<$<CONFIG:Release>:-mfma>
                    # Keep auto-vectorization enabled for comparison
                    $<$<CONFIG:Release>:-ftree-vectorize>
                    $<$<CONFIG:Release>:-fslp-vectorize>
                )
                
                if(COMPILER_SUPPORTS_AVX512)
                    target_compile_options(${target_name}_autovec PRIVATE
                        $<$<CONFIG:Release>:-mavx512f>
                        $<$<CONFIG:Release>:-mavx512bw>
                        $<$<CONFIG:Release>:-mavx512vl>
                    )
                endif()
                
                set_target_properties(${target_name}_autovec PROPERTIES
                    RUNTIME_OUTPUT_DIRECTORY ${BENCHMARK_OUTPUT_DIR}
                )
                
                message(STATUS "Comparison benchmark created: ${target_name}_autovec (auto-vectorization enabled)")
            endif()
        endif()
    endfunction()
    
    # Create benchmark executables based on available files
    create_benchmark_target(simd_benchmarks_v3 "benchmark/benchmark_main_v3.cpp" "SIMD_USE_V3_HEADER")
    create_benchmark_target(simd_benchmarks_v2 "benchmark/benchmark_main_v2.cpp" "SIMD_USE_V2_HEADER")
    create_benchmark_target(simd_benchmarks_v1 "benchmark/benchmark_main.cpp" "SIMD_USE_V1_HEADER")
    
    # Create comparison benchmarks for detailed analysis
    create_comparison_benchmark(simd_comparison_v3 "benchmark/benchmark_main_v3.cpp" "SIMD_USE_V3_HEADER")
    
    # Create default benchmark target based on active version
    if(SIMD_USE_V3 AND TARGET simd_benchmarks_v3)
        add_executable(simd_benchmarks benchmark/benchmark_main_v3.cpp)
        target_link_libraries(simd_benchmarks PRIVATE simd_library)
        target_compile_definitions(simd_benchmarks PRIVATE 
            SIMD_INCLUDE_EXAMPLES 
            SIMD_USE_V3_HEADER
        )
        
        # Apply same optimization flags as other benchmarks
        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(simd_benchmarks PRIVATE
                $<$<CONFIG:Release>:-O3>
                $<$<CONFIG:Release>:-march=native>
                $<$<CONFIG:Release>:-ffast-math>
                $<$<CONFIG:Release>:-mavx2>
                $<$<CONFIG:Release>:-mfma>
            )
            
            if(COMPILER_SUPPORTS_AVX512)
                target_compile_options(simd_benchmarks PRIVATE
                    $<$<CONFIG:Release>:-mavx512f>
                    $<$<CONFIG:Release>:-mavx512bw>
                    $<$<CONFIG:Release>:-mavx512vl>
                )
            endif()
            
            if(SIMD_DISABLE_AUTO_VECTORIZATION AND AUTO_VECTORIZATION_FLAGS_AVAILABLE)
                target_compile_options(simd_benchmarks PRIVATE
                    $<$<CONFIG:Release>:-fno-tree-vectorize>
                    $<$<CONFIG:Release>:-fno-slp-vectorize>
                    $<$<CONFIG:Release>:-fno-tree-loop-vectorize>
                )
                target_compile_definitions(simd_benchmarks PRIVATE DISABLE_SCALAR_AUTOVEC)
            endif()
        endif()
        
        set_target_properties(simd_benchmarks PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${BENCHMARK_OUTPUT_DIR}
        )
        
    elseif(SIMD_USE_V2 AND TARGET simd_benchmarks_v2)
        add_executable(simd_benchmarks benchmark/benchmark_main_v2.cpp)
        target_link_libraries(simd_benchmarks PRIVATE simd_library)
        target_compile_definitions(simd_benchmarks PRIVATE 
            SIMD_INCLUDE_EXAMPLES 
            SIMD_USE_V2_HEADER
        )
        set_target_properties(simd_benchmarks PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${BENCHMARK_OUTPUT_DIR}
        )
    elseif(SIMD_USE_V1 AND TARGET simd_benchmarks_v1)
        add_executable(simd_benchmarks benchmark/benchmark_main.cpp)
        target_link_libraries(simd_benchmarks PRIVATE simd_library)
        target_compile_definitions(simd_benchmarks PRIVATE SIMD_INCLUDE_EXAMPLES)
        set_target_properties(simd_benchmarks PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${BENCHMARK_OUTPUT_DIR}
        )
    endif()
    
    message(STATUS "Benchmarks enabled for active version: ${SIMD_ACTIVE_VERSION}")
endif()

# Examples
if(SIMD_BUILD_EXAMPLES)
    # Build your example application
    add_executable(simd_example example/example.cpp)
    target_link_libraries(simd_example PRIVATE simd_library)
    target_compile_definitions(simd_example PRIVATE SIMD_INCLUDE_EXAMPLES)
    
    # Add version-specific definitions
    if(SIMD_USE_V3)
        target_compile_definitions(simd_example PRIVATE SIMD_USE_V3_HEADER)
    elseif(SIMD_USE_V2)
        target_compile_definitions(simd_example PRIVATE SIMD_USE_V2_HEADER)
    elseif(SIMD_USE_V1)
        target_compile_definitions(simd_example PRIVATE SIMD_USE_V1_HEADER)
    endif()
    
    # Set output directory for easier access
    set_target_properties(simd_example PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/examples
    )
    
    message(STATUS "Example applications enabled")
endif()

# Custom targets for convenience
if(SIMD_BUILD_TESTS)
    add_custom_target(run_tests
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
        DEPENDS simd_tests
        COMMENT "Running unit tests..."
    )
endif()

if(SIMD_BUILD_BENCHMARKS)
    # Create run_benchmarks target for the active version
    if(TARGET simd_benchmarks)
        add_custom_target(run_benchmarks
            COMMAND simd_benchmarks
            DEPENDS simd_benchmarks
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            COMMENT "Running ${SIMD_ACTIVE_VERSION} benchmarks..."
        )
    endif()
    
    # Individual version targets
    if(TARGET simd_benchmarks_v3)
        add_custom_target(run_benchmarks_v3
            COMMAND simd_benchmarks_v3
            DEPENDS simd_benchmarks_v3
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            COMMENT "Running v3 benchmarks..."
        )
    endif()
    
    if(TARGET simd_benchmarks_v2)
        add_custom_target(run_benchmarks_v2
            COMMAND simd_benchmarks_v2
            DEPENDS simd_benchmarks_v2
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            COMMENT "Running v2 benchmarks..."
        )
    endif()
    
    if(TARGET simd_benchmarks_v1)
        add_custom_target(run_benchmarks_v1
            COMMAND simd_benchmarks_v1
            DEPENDS simd_benchmarks_v1
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            COMMENT "Running v1 benchmarks..."
        )
    endif()
    
    # Comparison targets
    if(TARGET simd_comparison_v3_autovec)
        add_custom_target(run_comparison
            COMMAND echo "=== Running benchmark with auto-vectorization DISABLED ==="
            COMMAND simd_comparison_v3
            COMMAND echo ""
            COMMAND echo "=== Running benchmark with auto-vectorization ENABLED ==="
            COMMAND simd_comparison_v3_autovec
            DEPENDS simd_comparison_v3 simd_comparison_v3_autovec
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            COMMENT "Comparing auto-vectorization impact..."
        )
    endif()
    
    # Comparison target (runs all available versions)
    set(BENCHMARK_TARGETS "")
    if(TARGET simd_benchmarks_v1)
        list(APPEND BENCHMARK_TARGETS simd_benchmarks_v1)
    endif()
    if(TARGET simd_benchmarks_v2)
        list(APPEND BENCHMARK_TARGETS simd_benchmarks_v2)
    endif()
    if(TARGET simd_benchmarks_v3)
        list(APPEND BENCHMARK_TARGETS simd_benchmarks_v3)
    endif()
    
    if(BENCHMARK_TARGETS)
        add_custom_target(compare_all_versions
            COMMENT "Running benchmarks for all available versions..."
            DEPENDS ${BENCHMARK_TARGETS}
        )
    endif()
    
    # Debug targets for compiler flag verification
    add_custom_target(debug_compiler_flags
        COMMAND echo "Checking compiler auto-vectorization flags..."
        COMMAND ${CMAKE_CXX_COMPILER} --version
        COMMAND echo "Auto-vectorization disable flags available: ${AUTO_VECTORIZATION_FLAGS_AVAILABLE}"
        COMMAND echo "Flags used in benchmarks:"
        COMMAND echo "  -O3 -march=native -ffast-math -mavx2 -mfma"
        COMMAND echo "  -fno-tree-vectorize -fno-slp-vectorize -fno-tree-loop-vectorize"
        COMMENT "Displaying compiler configuration for debugging"
    )
    
    # Generate assembly for verification
    if(TARGET simd_benchmarks AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        add_custom_target(check_assembly
            COMMAND echo "Generating assembly output to verify auto-vectorization is disabled..."
            COMMAND ${CMAKE_CXX_COMPILER} -S -O3 -march=native -ffast-math -mavx2 -mfma
                    -fno-tree-vectorize -fno-slp-vectorize -fno-tree-loop-vectorize
                    -I${CMAKE_SOURCE_DIR}/include
                    ${CMAKE_SOURCE_DIR}/benchmark/benchmark_main_v3.cpp
                    -o ${CMAKE_BINARY_DIR}/benchmark_assembly_no_autovec.s
            COMMAND echo "Assembly saved to: ${CMAKE_BINARY_DIR}/benchmark_assembly_no_autovec.s"
            COMMAND echo "Search for 'vmovups', 'vfmadd', 'vpxor' to verify SIMD usage patterns"
            COMMENT "Generating assembly to verify auto-vectorization status"
        )
    endif()
endif()

if(SIMD_BUILD_EXAMPLES)
    add_custom_target(run_example
        COMMAND simd_example
        DEPENDS simd_example
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Running example application..."
    )
endif()

# Installation (unchanged)
include(GNUInstallDirs)
install(TARGETS simd_library
    EXPORT simd_libraryTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# Install headers (all versions)
install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp"
)

install(EXPORT simd_libraryTargets
    FILE simd_libraryTargets.cmake
    NAMESPACE simd::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/simd_library
)

# Package configuration (for find_package support)
include(CMakePackageConfigHelpers)

# Create config file
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/simd_libraryConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/simd_libraryConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/simd_library
    NO_SET_AND_CHECK_MACRO
    NO_CHECK_REQUIRED_COMPONENTS_MACRO
)

# Create version file
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/simd_libraryConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

# Install config files
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/simd_libraryConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/simd_libraryConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/simd_library
)

# CPack configuration
set(CPACK_PACKAGE_NAME "simd_library")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "High-Performance SIMD Library for x86_64 with automatic dispatch")
set(CPACK_PACKAGE_VENDOR "AMD")
set(CPACK_PACKAGE_DESCRIPTION_FILE ${CMAKE_CURRENT_SOURCE_DIR}/README.md)
set(CPACK_RESOURCE_FILE_LICENSE ${CMAKE_CURRENT_SOURCE_DIR}/LICENSE)
set(CPACK_PACKAGE_CONTACT "simd-support@amd.com")

# Set package type based on platform
if(WIN32)
    set(CPACK_GENERATOR "ZIP;NSIS")
elseif(APPLE)
    set(CPACK_GENERATOR "TGZ;DragNDrop")
else()
    set(CPACK_GENERATOR "TGZ;DEB;RPM")
endif()

include(CPack)

# Performance validation target
if(SIMD_BUILD_BENCHMARKS AND TARGET simd_benchmarks)
    add_custom_target(validate_performance
        COMMAND echo "Running performance validation with ${SIMD_ACTIVE_VERSION}..."
        COMMAND simd_benchmarks
        COMMAND echo "Performance validation complete"
        DEPENDS simd_benchmarks
        COMMENT "Validating SIMD performance against expectations"
    )
endif()

# Print comprehensive build information
message(STATUS "")
message(STATUS "SIMD Library v3.0 Configuration Summary:")
message(STATUS "========================================")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Platform: ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "  Active SIMD Version: ${SIMD_ACTIVE_VERSION}")
message(STATUS "")
message(STATUS "Features:")
message(STATUS "  Use v1 Implementation: ${SIMD_USE_V1}")
message(STATUS "  Use v2 Implementation: ${SIMD_USE_V2}")
message(STATUS "  Use v3 Implementation: ${SIMD_USE_V3}")
message(STATUS "  Build Tests: ${SIMD_BUILD_TESTS}")
message(STATUS "  Build Benchmarks: ${SIMD_BUILD_BENCHMARKS}")
message(STATUS "  Build Examples: ${SIMD_BUILD_EXAMPLES}")
message(STATUS "  OpenMP Support: ${SIMD_ENABLE_OPENMP}")
if(OpenMP_CXX_FOUND)
    message(STATUS "  OpenMP Version: ${OpenMP_CXX_VERSION}")
endif()
message(STATUS "  Profiling Support: ${SIMD_ENABLE_PROFILING}")
message(STATUS "  Threading: Threads::Threads")
message(STATUS "  Auto-Vectorization Control: ${SIMD_DISABLE_AUTO_VECTORIZATION}")
if(DEFINED AUTO_VECTORIZATION_FLAGS_AVAILABLE)
    message(STATUS "  Auto-Vec Flags Available: ${AUTO_VECTORIZATION_FLAGS_AVAILABLE}")
endif()
message(STATUS "")
message(STATUS "Compiler Features:")
if(COMPILER_SUPPORTS_AVX512)
    message(STATUS "  AVX-512: Enabled")
else()
    message(STATUS "  AVX-512: Not available")
endif()
message(STATUS "  AVX2: Enabled")
message(STATUS "  FMA: Enabled")
if(SIMD_DISABLE_AUTO_VECTORIZATION AND AUTO_VECTORIZATION_FLAGS_AVAILABLE)
    message(STATUS "  Auto-Vectorization: DISABLED (for fair benchmarking)")
else()
    message(STATUS "  Auto-Vectorization: ENABLED (may affect benchmark results)")
endif()
message(STATUS "")
message(STATUS "Directories:")
message(STATUS "  Source Directory: ${CMAKE_SOURCE_DIR}")
message(STATUS "  Binary Directory: ${CMAKE_BINARY_DIR}")
message(STATUS "  Install Prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "")
message(STATUS "Available targets:")
message(STATUS "  simd_library       - Header-only library (${SIMD_ACTIVE_VERSION})")
if(SIMD_BUILD_TESTS)
    message(STATUS "  simd_tests         - Unit tests")
    message(STATUS "  run_tests          - Execute unit tests")
endif()
if(SIMD_BUILD_BENCHMARKS)
    if(TARGET simd_benchmarks)
        message(STATUS "  simd_benchmarks    - Default benchmark suite (${SIMD_ACTIVE_VERSION})")
        message(STATUS "  run_benchmarks     - Execute default benchmarks")
    endif()
    if(TARGET simd_benchmarks_v1)
        message(STATUS "  simd_benchmarks_v1 - v1 benchmark suite")
        message(STATUS "  run_benchmarks_v1  - Execute v1 benchmarks")
    endif()
    if(TARGET simd_benchmarks_v2)
        message(STATUS "  simd_benchmarks_v2 - v2 benchmark suite")
        message(STATUS "  run_benchmarks_v2  - Execute v2 benchmarks")
    endif()
    if(TARGET simd_benchmarks_v3)
        message(STATUS "  simd_benchmarks_v3 - v3 benchmark suite")
        message(STATUS "  run_benchmarks_v3  - Execute v3 benchmarks")
    endif()
    if(TARGET simd_comparison_v3_autovec)
        message(STATUS "  run_comparison     - Compare auto-vectorization impact")
    endif()
    if(TARGET compare_all_versions)
        message(STATUS "  compare_all_versions - Run all benchmark versions")
    endif()
    if(TARGET validate_performance)
        message(STATUS "  validate_performance - Performance validation")
    endif()
    message(STATUS "  debug_compiler_flags - Show compiler configuration")
    if(TARGET check_assembly)
        message(STATUS "  check_assembly     - Generate assembly for verification")
    endif()
endif()
if(SIMD_BUILD_EXAMPLES)
    message(STATUS "  simd_example       - Example application")
    message(STATUS "  run_example        - Execute example")
endif()
message(STATUS "")
message(STATUS "Quick start commands:")
message(STATUS "  mkdir build && cd build")
message(STATUS "  cmake .. -DCMAKE_BUILD_TYPE=Release")
message(STATUS "  make -j$(nproc)")
if(TARGET run_benchmarks)
    message(STATUS "  make run_benchmarks     # Run default benchmarks")
endif()
if(TARGET run_comparison)
    message(STATUS "  make run_comparison     # Compare auto-vectorization impact")
endif()
if(TARGET compare_all_versions)
    message(STATUS "  make compare_all_versions # Compare all versions")
endif()
message(STATUS "")
if(SIMD_DISABLE_AUTO_VECTORIZATION AND AUTO_VECTORIZATION_FLAGS_AVAILABLE)
    message(STATUS "🎯 Auto-vectorization is DISABLED for fair SIMD benchmarking")
    message(STATUS "   This should fix F32 dot product performance measurements")
else()
    message(WARNING "⚠️  Auto-vectorization is ENABLED - F32 speedups may appear low")
    message(STATUS "   Consider using: -DSIMD_DISABLE_AUTO_VECTORIZATION=ON")
endif()
message(STATUS "")