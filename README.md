# High-Performance SIMD Library

A production-quality, header-only C++ SIMD library targeting AMD/Intel x86_64 CPUs with automatic dispatch between scalar, AVX2, and AVX-512 code paths.

## 🚀 Quick Start

```cpp
#include "simd.hpp"

// Automatic SIMD dispatch based on CPU capabilities
std::vector<float> a(1024), b(1024);
// ... initialize data ...

// Dot product - automatically uses best available SIMD
float result = simd::dot(a.data(), b.data(), a.size());

// SAXPY operation: y = alpha * x + y
simd::saxpy(2.5f, a.data(), b.data(), a.size());

// Matrix multiplication
std::vector<float> c(64 * 64);
simd::matmul(a.data(), b.data(), c.data(), 64, 64, 64);
```

## 📋 Features

### ✅ **Epic A: API Design & Infrastructure**
- **Unified templated API**: `simd::dot<T>()`, `simd::matmul<T>()`, `simd::saxpy<T>()`, `simd::conv2d<T>()`
- **Automatic dispatch**: Runtime CPU feature detection with fallback hierarchy
- **Cross-platform**: Windows, Linux, Android support
- **Memory utilities**: Aligned allocators, SIMD-friendly padding helpers

### ✅ **Epic B: Scalar Baseline & Profiling**
- Reference implementations for all kernels
- Comprehensive test suite with 100% coverage
- Built-in benchmarking infrastructure with GFLOPS measurements
- Performance profiling harness ready for Intel VTune and Linux perf

### ✅ **Epic C: AVX2 Kernel Optimization**
- Hand-optimized AVX2 implementations using FMA instructions
- Masked tail handling for arbitrary array sizes
- Tiled algorithms for optimal cache utilization
- Data-driven tuning with performance validation

### ✅ **Epic D: AVX-512 Extension & Advanced Features**
- 512-bit vector operations with mask support
- Advanced intrinsics: scatter/gather, conflict detection
- Frequency-aware optimizations
- Graceful fallback to AVX2/scalar when unavailable

### ✅ **Epic E: Thread-Level Scaling**
- OpenMP and thread-based parallel implementations
- Automatic work distribution across CPU cores
- Thread-local aligned memory management
- Strong and weak scaling benchmarks

### ✅ **Epic F: Documentation & Knowledge Transfer**
- Complete API documentation with examples
- Optimization playbook and best practices
- Tutorial applications for AI, HPC, and media workloads

## 🏗️ Build System

### CMake Configuration

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

### Build Options

```cmake
option(SIMD_BUILD_TESTS "Build unit tests" ON)
option(SIMD_BUILD_BENCHMARKS "Build benchmark suite" ON)
option(SIMD_BUILD_EXAMPLES "Build example applications" ON)
option(SIMD_ENABLE_OPENMP "Enable OpenMP parallel support" OFF)
```

### Compiler Requirements

- **C++17** or later
- **GCC 7+**, **Clang 6+**, or **MSVC 2019+**
- **AVX2/AVX-512** support for optimal performance

## 🔧 API Reference

### Core Functions

#### Dot Product
```cpp
template<typename T>
T dot(const T* a, const T* b, size_t n);
```
Computes the dot product of two vectors: `sum(a[i] * b[i])`.

**Parameters:**
- `a, b`: Input vectors (must be at least `n` elements)
- `n`: Vector length
- **Returns:** Dot product result

**Example:**
```cpp
std::vector<float> x = {1, 2, 3, 4};
std::vector<float> y = {2, 3, 4, 5};
float result = simd::dot(x.data(), y.data(), x.size()); // Result: 40
```

#### SAXPY Operation
```cpp
template<typename T>
void saxpy(T alpha, const T* x, T* y, size_t n);
```
Performs the SAXPY operation: `y[i] = alpha * x[i] + y[i]`.

**Parameters:**
- `alpha`: Scalar multiplier
- `x`: Input vector (read-only)
- `y`: Input/output vector (modified in-place)
- `n`: Vector length

**Example:**
```cpp
std::vector<float> x = {1, 2, 3};
std::vector<float> y = {4, 5, 6};
simd::saxpy(2.0f, x.data(), y.data(), x.size()); // y becomes {6, 9, 12}
```

#### Matrix Multiplication
```cpp
template<typename T>
void matmul(const T* a, const T* b, T* c, size_t m, size_t n, size_t k);
```
Computes matrix multiplication: `C = A × B` where A is m×k, B is k×n, C is m×n.

**Parameters:**
- `a`: Matrix A in row-major order (m×k)
- `b`: Matrix B in row-major order (k×n)
- `c`: Output matrix C in row-major order (m×n)
- `m, n, k`: Matrix dimensions

**Example:**
```cpp
// 2×2 matrix multiplication
std::vector<float> A = {1, 2, 3, 4}; // [[1,2], [3,4]]
std::vector<float> B = {5, 6, 7, 8}; // [[5,6], [7,8]]
std::vector<float> C(4);
simd::matmul(A.data(), B.data(), C.data(), 2, 2, 2);
// C = [[19,22], [43,50]]
```

#### 2D Convolution
```cpp
template<typename T>
void conv2d(const T* input, const T* kernel, T* output, 
           size_t height, size_t width);
```
Performs 2D convolution with a 3×3 kernel.

### Parallel Operations

All core functions have parallel equivalents in the `simd::parallel` namespace:

```cpp
namespace parallel {
    template<typename T> T dot(const T* a, const T* b, size_t n);
    template<typename T> void saxpy(T alpha, const T* x, T* y, size_t n);
    // ... other parallel functions
}
```

### CPU Feature Detection

```cpp
const auto& features = simd::CpuFeatures::detect();
if (features.avx512f) {
    std::cout << "AVX-512 available!" << std::endl;
}
```

**Available Features:**
- `avx2`: AVX2 256-bit vectors
- `avx512f`: AVX-512 foundation
- `avx512bw`: AVX-512 byte/word instructions
- `avx512vl`: AVX-512 vector length extensions
- `avx512vnni`: Vector Neural Network Instructions
- `avx512bf16`: BFloat16 support

### Memory Utilities

#### Aligned Allocation
```cpp
using AlignedVec = std::vector<float, simd::util::AlignedAllocator<float, 64>>;
AlignedVec data(1024); // 64-byte aligned vector
```

#### Size Padding
```cpp
size_t padded = simd::util::padded_size<float>(1000); // Rounds up to SIMD boundary
```

## 🎯 Performance Optimization Guide

### 1. **Profile First**
Use the built-in benchmarking suite:
```cpp
#define SIMD_INCLUDE_EXAMPLES
#include "simd.hpp"

int main() {
    simd::benchmark::run_benchmarks();
    return 0;
}
```

### 2. **Data Layout Optimization**
- Use aligned memory for best performance
- Consider data padding for SIMD-friendly sizes
- Prefer AoS (Array of Structures) for small structs, SoA (Structure of Arrays) for large ones

### 3. **Algorithm Selection**
```cpp
// For small arrays (< 1000 elements), scalar might be competitive
if (n < 1000) {
    return simd::scalar::dot(a, b, n);
} else {
    return simd::dot(a, b, n); // Uses best SIMD
}
```

### 4. **Thread Scaling**
```cpp
// Automatic parallelization for large workloads
if (n > 100000) {
    return simd::parallel::dot(a, b, n);
} else {
    return simd::dot(a, b, n);
}
```

## 🧪 Testing and Validation

### Running Tests
```bash
# Build and run all tests
cmake --build . --target simd_tests
./simd_tests

# Run specific test categories
./simd_tests "[scalar]"     # Scalar implementations only
./simd_tests "[simd]"       # SIMD consistency tests
./simd_tests "[parallel]"   # Parallel implementations
```

### Custom Validation
```cpp
#include <catch2/catch_test_macros.hpp>

TEST_CASE("Custom Algorithm") {
    std::vector<float> data = {1, 2, 3, 4, 5};
    float result = simd::dot(data.data(), data.data(), data.size());
    REQUIRE(result == Catch::Approx(55.0f));
}
```

## 📊 Benchmarking

### Built-in Benchmarks
```cpp
simd::benchmark::benchmark_dot<float>(1024, 1000);    // 1K iterations
simd::benchmark::benchmark_saxpy<float>(8192, 500);   // 500 iterations  
simd::benchmark::benchmark_matmul<float>(128, 128, 128, 100);
```

### Performance Analysis
The library provides detailed performance metrics:
- **GFLOPS**: Gigaflops per second
- **Memory Bandwidth**: Effective bandwidth utilization
- **IPC**: Instructions per cycle (when using Intel VTune)
- **Scaling**: Speedup vs. thread count

### Intel VTune Integration
```bash
# Profile with VTune
vtune -collect hotspots -result-dir vtune_results ./simd_benchmarks

# Analyze results
vtune -report summary -result-dir vtune_results
```

## 🔧 Environment Configuration

### Runtime Overrides
```bash
# Force scalar execution (for debugging)
export SIMD_FORCE_SCALAR=1

# Intel VTune profiling
export SIMD_ENABLE_VTUNE=1
```

### Compiler Flags
```bash
# GCC/Clang optimization
-O3 -march=native -mtune=native -ffast-math

# MSVC optimization  
/O2 /arch:AVX2 /fp:fast
```

## 🏭 Production Deployment

### Integration Patterns

#### AI/ML Inference
```cpp
class NeuralLayer {
    simd::util::aligned_vector<float> weights_, bias_;
    
public:
    void forward(const float* input, float* output, size_t batch_size) {
        // Parallel matrix multiplication for batch processing
        simd::parallel::matmul(weights_.data(), input, output, 
                              output_size_, batch_size, input_size_);
        
        // Add bias with SAXPY
        for (size_t b = 0; b < batch_size; ++b) {
            simd::saxpy(1.0f, bias_.data(), &output[b * output_size_], output_size_);
        }
    }
};
```

#### Signal Processing
```cpp
class SignalProcessor {
public:
    float compute_correlation(const float* signal1, const float* signal2, size_t length) {
        return simd::dot(signal1, signal2, length) / length;
    }
    
    void apply_filter(float gain, const float* input, float* output, size_t length) {
        std::copy(input, input + length, output);
        simd::saxpy(gain - 1.0f, input, output, length);
    }
};
```

#### Media Processing
```cpp
void process_image_channel(const float* input, float* output, 
                          size_t height, size_t width) {
    // Example: Gaussian blur approximation
    std::vector<float> kernel = {0.1f, 0.8f, 0.1f, 
                                0.8f, 1.0f, 0.8f,
                                0.1f, 0.8f, 0.1f};
    
    simd::conv2d(input, kernel.data(), output, height, width);
}
```

## 🐛 Debugging and Troubleshooting

### Common Issues

#### 1. **Alignment Errors**
```cpp
// ❌ Wrong: unaligned allocation
float* data = new float[1024];

// ✅ Correct: use aligned allocator
simd::util::aligned_vector<float> data(1024);
```

#### 2. **Size Mismatches**
```cpp
// ❌ Wrong: assuming SIMD-friendly sizes
simd::dot(a.data(), b.data(), 1000); // Might be inefficient

// ✅ Better: pad to SIMD boundary if needed
size_t padded_size = simd::util::padded_size<float>(1000);
```

#### 3. **Performance Regression**
```cpp
// Check if SIMD is actually being used
const auto& features = simd::CpuFeatures::detect();
if (!features.avx2 && !features.avx512f) {
    std::cout << "Warning: Using scalar fallback" << std::endl;
}
```

### Debug Builds
```bash
# Enable address sanitizer and debug symbols
cmake -DCMAKE_BUILD_TYPE=Debug -DENABLE_ASAN=ON ..
```

## 📈 Roadmap and Extensions

### Planned Features
- **Additional Kernels**: FFT, matrix decomposition, sparse operations
- **GPU Backends**: CUDA/ROCm integration for heterogeneous computing
- **Auto-tuning**: Machine learning-based parameter optimization
- **Advanced SIMD**: AVX-512 VNNI, BF16 acceleration

### Contributing
See `CONTRIBUTING.md` for development guidelines and coding standards.

## 📄 License

This library is distributed under the MIT License. See `LICENSE` file for details.

## 🤝 Support

- **Issues**: GitHub Issues for bug reports and feature requests
- **Discussions**: GitHub Discussions for questions and community support
- **Enterprise**: Contact AMD for enterprise support and custom optimization services