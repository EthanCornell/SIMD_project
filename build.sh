#!/bin/bash
set -e

# Build script for SIMD Library v2
echo "=== SIMD Library v2 Build Script ==="

# Default values
BUILD_TYPE="Release"
BUILD_DIR="build"
CLEAN_BUILD=false
RUN_TESTS=false
RUN_BENCHMARKS=false
RUN_EXAMPLES=false
ENABLE_OPENMP=false
ENABLE_V2=true

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        -c|--clean)
            CLEAN_BUILD=true
            shift
            ;;
        -t|--test)
            RUN_TESTS=true
            shift
            ;;
        -b|--benchmarks)
            RUN_BENCHMARKS=true
            shift
            ;;
        -e|--examples)
            RUN_EXAMPLES=true
            shift
            ;;
        --openmp)
            ENABLE_OPENMP=true
            shift
            ;;
        --v1)
            ENABLE_V2=false
            shift
            ;;
        --all)
            RUN_TESTS=true
            RUN_BENCHMARKS=true
            RUN_EXAMPLES=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  -d, --debug      Build in Debug mode (default: Release)"
            echo "  -c, --clean      Clean build directory before building"
            echo "  -t, --test       Run tests after building"
            echo "  -b, --benchmarks Run benchmarks after building"
            echo "  -e, --examples   Run examples after building"
            echo "  --all            Run tests, benchmarks, and examples"
            echo "  --openmp         Enable OpenMP support"
            echo "  --v1             Use v1 implementation instead of v2"
            echo "  -h, --help       Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 -c --all      Clean build and run everything"
            echo "  $0 --openmp -b   Build with OpenMP and run benchmarks"
            echo "  $0 --v1 -t       Build v1 implementation and run tests"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Function to create missing files
create_missing_files() {
    echo "Creating missing configuration files..."
    
    # Create cmake directory
    mkdir -p cmake
    
    # Create CMake config template if it doesn't exist
    if [ ! -f "cmake/simd_libraryConfig.cmake.in" ]; then
        cat > cmake/simd_libraryConfig.cmake.in << 'EOF'
@PACKAGE_INIT@

# SIMD Library Configuration File
include(CMakeFindDependencyMacro)

if(@SIMD_ENABLE_OPENMP@)
    find_dependency(OpenMP REQUIRED)
endif()

include("${CMAKE_CURRENT_LIST_DIR}/simd_libraryTargets.cmake")

if(NOT TARGET simd::simd_library)
    add_library(simd::simd_library ALIAS simd_library)
endif()

set(SIMD_LIBRARY_VERSION "@PROJECT_VERSION@")
set(SIMD_LIBRARY_OPENMP_ENABLED @SIMD_ENABLE_OPENMP@)
set(SIMD_LIBRARY_V2_ENABLED @SIMD_USE_V2@)

check_required_components(simd_library)
EOF
        echo "  Created cmake/simd_libraryConfig.cmake.in"
    fi
    
    # Create LICENSE file if it doesn't exist
    if [ ! -f "LICENSE" ]; then
        cat > LICENSE << 'EOF'
MIT License

Copyright (c) 2024 SIMD Library

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
EOF
        echo "  Created LICENSE file"
    fi
}

# Clean build directory if requested
if [ "$CLEAN_BUILD" = true ]; then
    echo "Cleaning build directory..."
    rm -rf "$BUILD_DIR"
fi

# Create missing files before building
create_missing_files

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure
echo "Configuring with CMake..."
echo "  Build Type: $BUILD_TYPE"
echo "  V2 Implementation: $ENABLE_V2"
echo "  OpenMP Support: $ENABLE_OPENMP"

cmake -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
      -DSIMD_BUILD_TESTS=ON \
      -DSIMD_BUILD_BENCHMARKS=ON \
      -DSIMD_BUILD_EXAMPLES=ON \
      -DSIMD_USE_V2="$ENABLE_V2" \
      -DSIMD_ENABLE_OPENMP="$ENABLE_OPENMP" \
      ..

# Build
echo "Building..."
make -j$(nproc 2>/dev/null || echo 4)

echo "Build completed successfully!"

# Helper function to find and run executable
run_executable() {
    local name=$1
    local description=$2
    
    echo ""
    echo "=== Running $description ==="
    
    # Check possible locations
    local exe_paths=(
        "./$name"
        "./examples/$name"
        "./benchmark/$name"
        "./test/$name"
    )
    
    local found=false
    for path in "${exe_paths[@]}"; do
        if [ -f "$path" ]; then
            echo "Executing: $path"
            "$path"
            found=true
            break
        fi
    done
    
    if [ "$found" = false ]; then
        echo "ERROR: $name executable not found!"
        echo "Searched in: ${exe_paths[*]}"
        return 1
    fi
}

# Run tests if requested
if [ "$RUN_TESTS" = true ]; then
    run_executable "simd_tests" "Unit Tests"
fi

# Run benchmarks if requested
if [ "$RUN_BENCHMARKS" = true ]; then
    if [ "$ENABLE_V2" = true ]; then
        run_executable "simd_benchmarks_v2" "V2 Comprehensive Benchmarks"
    else
        run_executable "simd_benchmarks_v1" "V1 Basic Benchmarks"
    fi
fi

# Run examples if requested
if [ "$RUN_EXAMPLES" = true ]; then
    run_executable "simd_example" "Example Application"
fi

echo ""
echo "=== Build Summary ==="
echo "Build Type: $BUILD_TYPE"
echo "Implementation: $([ "$ENABLE_V2" = true ] && echo "v2 (Complete)" || echo "v1 (Basic)")"
echo "OpenMP: $([ "$ENABLE_OPENMP" = true ] && echo "Enabled" || echo "Disabled")"
echo "Build Directory: $(pwd)"

echo ""
echo "Available executables:"
[ -f "./simd_tests" ] && echo "  ✓ simd_tests (unit tests)"
[ -f "./simd_benchmarks_v3" ] && echo "  ✓ simd_benchmarks_v3 (AXV512 benchmarks)"
[ -f "./simd_benchmarks_v2" ] && echo "  ✓ simd_benchmarks_v2 (comprehensive benchmarks)"
[ -f "./simd_benchmarks_v1" ] && echo "  ✓ simd_benchmarks_v1 (basic benchmarks)"
[ -f "./simd_example" ] && echo "  ✓ simd_example (example application)"
[ -f "./examples/simd_example" ] && echo "  ✓ examples/simd_example (example application)"

echo ""
echo "Quick commands:"
echo "  Run tests:      make run_tests"
echo "  Run benchmarks: make run_benchmarks"
echo "  Run examples:   make run_example"
echo "  Validate perf:  make validate_performance"

echo ""
echo "Manual execution:"
[ -f "./simd_tests" ] && echo "  Tests:      ./build/simd_tests"
[ -f "./simd_benchmarks_v3" ] && echo "  Benchmarks: ./build/benchmark/simd_benchmarks_v3"
[ -f "./simd_benchmarks_v2" ] && echo "  Benchmarks: ./build/benchmark/simd_benchmarks_v2"
[ -f "./simd_benchmarks_v1" ] && echo "  Benchmarks: ./build/benchmark/simd_benchmarks_v1"
[ -f "./simd_example" ] && echo "  Example:    ./build/simd_example"
[ -f "./examples/simd_example" ] && echo "  Example:    ./build/examples/simd_example"

echo ""
echo "🚀 Build complete! Use './build.sh --all' to run everything."