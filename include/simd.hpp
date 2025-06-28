//==============================================================================
// High-Performance SIMD Library
// Header-only C++ library with automatic dispatch for scalar/AVX2/AVX-512
//==============================================================================

#pragma once

#include <immintrin.h>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <memory>
#include <vector>
#include <thread>
#include <functional>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <cstdlib>

#ifdef _WIN32
#include <Windows.h>
#include <intrin.h>
#else
#include <cpuid.h>
#include <unistd.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

//==============================================================================
// Epic A: API Design & Infrastructure
//==============================================================================

namespace simd {

//------------------------------------------------------------------------------
// A1.2: Runtime CPU Feature Detection
//------------------------------------------------------------------------------
class CpuFeatures {
public:
    struct Features {
        bool avx2 = false;
        bool avx512f = false;
        bool avx512bw = false;
        bool avx512vl = false;
        bool avx512vnni = false;
        bool avx512bf16 = false;
    };

    static const Features& detect() {
        static Features features = []() {
            Features f;
            
#ifdef _WIN32
            int cpuInfo[4];
            __cpuid(cpuInfo, 0);
            int maxId = cpuInfo[0];
            
            if (maxId >= 7) {
                __cpuidex(cpuInfo, 7, 0);
                f.avx2 = (cpuInfo[1] & (1 << 5)) != 0;
                f.avx512f = (cpuInfo[1] & (1 << 16)) != 0;
                f.avx512bw = (cpuInfo[1] & (1 << 30)) != 0;
                f.avx512vl = (cpuInfo[1] & (1 << 31)) != 0;
                f.avx512vnni = (cpuInfo[2] & (1 << 11)) != 0;
                
                __cpuidex(cpuInfo, 7, 1);
                f.avx512bf16 = (cpuInfo[0] & (1 << 5)) != 0;
            }
#else
            unsigned int eax, ebx, ecx, edx;
            if (__get_cpuid_max(0, nullptr) >= 7) {
                __cpuid_count(7, 0, eax, ebx, ecx, edx);
                f.avx2 = (ebx & (1 << 5)) != 0;
                f.avx512f = (ebx & (1 << 16)) != 0;
                f.avx512bw = (ebx & (1 << 30)) != 0;
                f.avx512vl = (ebx & (1 << 31)) != 0;
                f.avx512vnni = (ecx & (1 << 11)) != 0;
                
                __cpuid_count(7, 1, eax, ebx, ecx, edx);
                f.avx512bf16 = (eax & (1 << 5)) != 0;
            }
#endif
            
            // Environment variable override
            if (const char* override = std::getenv("SIMD_FORCE_SCALAR")) {
                if (std::atoi(override)) {
                    f.avx2 = f.avx512f = false;
                }
            }
            
            return f;
        }();
        
        return features;
    }
};

//------------------------------------------------------------------------------
// A1.3: Utility Classes - Memory Management
//------------------------------------------------------------------------------
namespace util {

template<typename T, size_t Alignment = 64>
class AlignedAllocator {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using size_type = size_t;

    template<typename U>
    struct rebind { using other = AlignedAllocator<U, Alignment>; };

    pointer allocate(size_type n) {
        size_t bytes = n * sizeof(T);
        void* ptr = nullptr;
        
#ifdef _WIN32
        ptr = _aligned_malloc(bytes, Alignment);
#else
        if (posix_memalign(&ptr, Alignment, bytes) != 0) {
            ptr = nullptr;
        }
#endif
        
        if (!ptr) throw std::bad_alloc();
        return static_cast<pointer>(ptr);
    }

    void deallocate(pointer p, size_type) {
#ifdef _WIN32
        _aligned_free(p);
#else
        free(p);
#endif
    }
};

template<typename T>
using aligned_vector = std::vector<T, AlignedAllocator<T>>;

// Helper for padding arrays to SIMD-friendly sizes
template<typename T>
size_t padded_size(size_t n, size_t simd_width = 16) {
    size_t elements_per_vector = simd_width / sizeof(T);
    return ((n + elements_per_vector - 1) / elements_per_vector) * elements_per_vector;
}

} // namespace util

//==============================================================================
// Epic B: Scalar Baseline Implementation
//==============================================================================

namespace scalar {

template<typename T>
T dot(const T* a, const T* b, size_t n) {
    T result = T(0);
    for (size_t i = 0; i < n; ++i) {
        result += a[i] * b[i];
    }
    return result;
}

template<typename T>
void saxpy(T alpha, const T* x, T* y, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        y[i] += alpha * x[i];
    }
}

template<typename T>
void gemm(const T* a, const T* b, T* c, size_t m, size_t n, size_t k) {
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            T sum = T(0);
            for (size_t kk = 0; kk < k; ++kk) {
                sum += a[i * k + kk] * b[kk * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

template<typename T>
void conv2d_3x3(const T* input, const T* kernel, T* output, 
                size_t height, size_t width) {
    for (size_t y = 1; y < height - 1; ++y) {
        for (size_t x = 1; x < width - 1; ++x) {
            T sum = T(0);
            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    sum += input[(y + ky) * width + (x + kx)] * 
                           kernel[(ky + 1) * 3 + (kx + 1)];
                }
            }
            output[y * width + x] = sum;
        }
    }
}

} // namespace scalar

//==============================================================================
// Epic C: AVX2 Kernel Optimization
//==============================================================================

namespace avx2 {

#ifdef __AVX2__

float dot(const float* a, const float* b, size_t n) {
    __m256 sum = _mm256_setzero_ps();
    size_t simd_n = n & ~7; // Process 8 elements at a time
    
    for (size_t i = 0; i < simd_n; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        sum = _mm256_fmadd_ps(va, vb, sum);
    }
    
    // Horizontal sum
    __m128 low = _mm256_castps256_ps128(sum);
    __m128 high = _mm256_extractf128_ps(sum, 1);
    __m128 sum128 = _mm_add_ps(low, high);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    
    float result = _mm_cvtss_f32(sum128);
    
    // Handle remaining elements
    for (size_t i = simd_n; i < n; ++i) {
        result += a[i] * b[i];
    }
    
    return result;
}

void saxpy(float alpha, const float* x, float* y, size_t n) {
    __m256 valpha = _mm256_broadcast_ss(&alpha);
    size_t simd_n = n & ~7;
    
    for (size_t i = 0; i < simd_n; i += 8) {
        __m256 vx = _mm256_loadu_ps(&x[i]);
        __m256 vy = _mm256_loadu_ps(&y[i]);
        vy = _mm256_fmadd_ps(valpha, vx, vy);
        _mm256_storeu_ps(&y[i], vy);
    }
    
    // Handle remaining elements
    for (size_t i = simd_n; i < n; ++i) {
        y[i] += alpha * x[i];
    }
}

void gemm(const float* a, const float* b, float* c, size_t m, size_t n, size_t k) {
    constexpr size_t TILE_M = 4;
    constexpr size_t TILE_N = 8;
    
    for (size_t i = 0; i < m; i += TILE_M) {
        for (size_t j = 0; j < n; j += TILE_N) {
            __m256 c0 = _mm256_setzero_ps();
            __m256 c1 = _mm256_setzero_ps();
            __m256 c2 = _mm256_setzero_ps();
            __m256 c3 = _mm256_setzero_ps();
            
            for (size_t kk = 0; kk < k; ++kk) {
                __m256 b_vec = _mm256_loadu_ps(&b[kk * n + j]);
                
                if (i + 0 < m) {
                    __m256 a0 = _mm256_broadcast_ss(&a[(i + 0) * k + kk]);
                    c0 = _mm256_fmadd_ps(a0, b_vec, c0);
                }
                if (i + 1 < m) {
                    __m256 a1 = _mm256_broadcast_ss(&a[(i + 1) * k + kk]);
                    c1 = _mm256_fmadd_ps(a1, b_vec, c1);
                }
                if (i + 2 < m) {
                    __m256 a2 = _mm256_broadcast_ss(&a[(i + 2) * k + kk]);
                    c2 = _mm256_fmadd_ps(a2, b_vec, c2);
                }
                if (i + 3 < m) {
                    __m256 a3 = _mm256_broadcast_ss(&a[(i + 3) * k + kk]);
                    c3 = _mm256_fmadd_ps(a3, b_vec, c3);
                }
            }
            
            if (i + 0 < m && j + 8 <= n) _mm256_storeu_ps(&c[(i + 0) * n + j], c0);
            if (i + 1 < m && j + 8 <= n) _mm256_storeu_ps(&c[(i + 1) * n + j], c1);
            if (i + 2 < m && j + 8 <= n) _mm256_storeu_ps(&c[(i + 2) * n + j], c2);
            if (i + 3 < m && j + 8 <= n) _mm256_storeu_ps(&c[(i + 3) * n + j], c3);
        }
    }
}

#endif // __AVX2__

} // namespace avx2

//==============================================================================
// Epic D: AVX-512 Extension & Advanced Features
//==============================================================================

namespace avx512 {

#ifdef __AVX512F__

float dot(const float* a, const float* b, size_t n) {
    __m512 sum = _mm512_setzero_ps();
    size_t simd_n = n & ~15; // Process 16 elements at a time
    
    for (size_t i = 0; i < simd_n; i += 16) {
        __m512 va = _mm512_loadu_ps(&a[i]);
        __m512 vb = _mm512_loadu_ps(&b[i]);
        sum = _mm512_fmadd_ps(va, vb, sum);
    }
    
    // Handle remaining elements with mask
    if (n > simd_n) {
        __mmask16 mask = (1 << (n - simd_n)) - 1;
        __m512 va = _mm512_maskz_loadu_ps(mask, &a[simd_n]);
        __m512 vb = _mm512_maskz_loadu_ps(mask, &b[simd_n]);
        sum = _mm512_fmadd_ps(va, vb, sum);
    }
    
    // Horizontal sum
    return _mm512_reduce_add_ps(sum);
}

void saxpy(float alpha, const float* x, float* y, size_t n) {
    __m512 valpha = _mm512_set1_ps(alpha);
    size_t simd_n = n & ~15;
    
    for (size_t i = 0; i < simd_n; i += 16) {
        __m512 vx = _mm512_loadu_ps(&x[i]);
        __m512 vy = _mm512_loadu_ps(&y[i]);
        vy = _mm512_fmadd_ps(valpha, vx, vy);
        _mm512_storeu_ps(&y[i], vy);
    }
    
    // Handle remaining elements with mask
    if (n > simd_n) {
        __mmask16 mask = (1 << (n - simd_n)) - 1;
        __m512 vx = _mm512_maskz_loadu_ps(mask, &x[simd_n]);
        __m512 vy = _mm512_maskz_loadu_ps(mask, &y[simd_n]);
        vy = _mm512_fmadd_ps(valpha, vx, vy);
        _mm512_mask_storeu_ps(&y[simd_n], mask, vy);
    }
}

void gemm(const float* a, const float* b, float* c, size_t m, size_t n, size_t k) {
    constexpr size_t TILE_M = 6;
    constexpr size_t TILE_N = 16;
    
    for (size_t i = 0; i < m; i += TILE_M) {
        for (size_t j = 0; j < n; j += TILE_N) {
            __m512 c_tiles[TILE_M];
            for (size_t t = 0; t < TILE_M; ++t) {
                c_tiles[t] = _mm512_setzero_ps();
            }
            
            for (size_t kk = 0; kk < k; ++kk) {
                __m512 b_vec = _mm512_loadu_ps(&b[kk * n + j]);
                
                for (size_t t = 0; t < TILE_M && i + t < m; ++t) {
                    __m512 a_broadcast = _mm512_set1_ps(a[(i + t) * k + kk]);
                    c_tiles[t] = _mm512_fmadd_ps(a_broadcast, b_vec, c_tiles[t]);
                }
            }
            
            for (size_t t = 0; t < TILE_M && i + t < m; ++t) {
                if (j + TILE_N <= n) {
                    _mm512_storeu_ps(&c[(i + t) * n + j], c_tiles[t]);
                } else {
                    __mmask16 mask = (1 << (n - j)) - 1;
                    _mm512_mask_storeu_ps(&c[(i + t) * n + j], mask, c_tiles[t]);
                }
            }
        }
    }
}

#endif // __AVX512F__

} // namespace avx512



//==============================================================================
// A1.1: Unified API with Automatic Dispatch
//==============================================================================

template<typename T>
T dot(const T* a, const T* b, size_t n) {
    const auto& features = CpuFeatures::detect();
    
    if constexpr (std::is_same_v<T, float>) {
#ifdef __AVX512F__
        if (features.avx512f) {
            return avx512::dot(a, b, n);
        }
#endif
#ifdef __AVX2__
        if (features.avx2) {
            return avx2::dot(a, b, n);
        }
#endif
    }
    
    return scalar::dot(a, b, n);
}

template<typename T>
void saxpy(T alpha, const T* x, T* y, size_t n) {
    const auto& features = CpuFeatures::detect();
    
    if constexpr (std::is_same_v<T, float>) {
#ifdef __AVX512F__
        if (features.avx512f) {
            avx512::saxpy(alpha, x, y, n);
            return;
        }
#endif
#ifdef __AVX2__
        if (features.avx2) {
            avx2::saxpy(alpha, x, y, n);
            return;
        }
#endif
    }
    
    scalar::saxpy(alpha, x, y, n);
}

template<typename T>
void matmul(const T* a, const T* b, T* c, size_t m, size_t n, size_t k) {
    const auto& features = CpuFeatures::detect();
    
    if constexpr (std::is_same_v<T, float>) {
#ifdef __AVX512F__
        if (features.avx512f) {
            avx512::gemm(a, b, c, m, n, k);
            return;
        }
#endif
#ifdef __AVX2__
        if (features.avx2) {
            avx2::gemm(a, b, c, m, n, k);
            return;
        }
#endif
    }
    
    scalar::gemm(a, b, c, m, n, k);
}

template<typename T>
void conv2d(const T* input, const T* kernel, T* output, 
           size_t height, size_t width) {
    // For now, use scalar implementation
    // Could be extended with SIMD versions
    scalar::conv2d_3x3(input, kernel, output, height, width);
}

//==============================================================================
// Profiling and Benchmarking Infrastructure
//==============================================================================

namespace benchmark {

class Timer {
    std::chrono::high_resolution_clock::time_point start_;
public:
    Timer() : start_(std::chrono::high_resolution_clock::now()) {}
    
    double elapsed_ms() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }
    
    void reset() {
        start_ = std::chrono::high_resolution_clock::now();
    }
};

template<typename T>
void benchmark_dot(size_t n, int iterations = 1000) {
    util::aligned_vector<T> a(n), b(n);
    
    // Initialize with random data
    for (size_t i = 0; i < n; ++i) {
        a[i] = static_cast<T>(rand()) / RAND_MAX;
        b[i] = static_cast<T>(rand()) / RAND_MAX;
    }
    
    Timer timer;
    T result = T(0);
    
    for (int i = 0; i < iterations; ++i) {
        result += dot(a.data(), b.data(), n);
    }
    
    double elapsed = timer.elapsed_ms();
    double ops_per_sec = (2.0 * n * iterations) / (elapsed / 1000.0);
    double gflops = ops_per_sec / 1e9;
    
    std::cout << "DOT (n=" << n << "): " << gflops << " GFLOPS, " 
              << elapsed << " ms, result=" << result << std::endl;
}

template<typename T>
void benchmark_saxpy(size_t n, int iterations = 1000) {
    util::aligned_vector<T> x(n), y(n);
    T alpha = static_cast<T>(2.0);
    
    for (size_t i = 0; i < n; ++i) {
        x[i] = static_cast<T>(rand()) / RAND_MAX;
        y[i] = static_cast<T>(rand()) / RAND_MAX;
    }
    
    Timer timer;
    
    for (int i = 0; i < iterations; ++i) {
        saxpy(alpha, x.data(), y.data(), n);
    }
    
    double elapsed = timer.elapsed_ms();
    double ops_per_sec = (2.0 * n * iterations) / (elapsed / 1000.0);
    double gflops = ops_per_sec / 1e9;
    
    std::cout << "SAXPY (n=" << n << "): " << gflops << " GFLOPS, " 
              << elapsed << " ms" << std::endl;
}

template<typename T>
void benchmark_matmul(size_t m, size_t n, size_t k, int iterations = 100) {
    util::aligned_vector<T> a(m * k), b(k * n), c(m * n);
    
    for (size_t i = 0; i < m * k; ++i) {
        a[i] = static_cast<T>(rand()) / RAND_MAX;
    }
    for (size_t i = 0; i < k * n; ++i) {
        b[i] = static_cast<T>(rand()) / RAND_MAX;
    }
    
    Timer timer;
    
    for (int i = 0; i < iterations; ++i) {
        matmul(a.data(), b.data(), c.data(), m, n, k);
    }
    
    double elapsed = timer.elapsed_ms();
    double ops_per_sec = (2.0 * m * n * k * iterations) / (elapsed / 1000.0);
    double gflops = ops_per_sec / 1e9;
    
    std::cout << "MATMUL (" << m << "x" << n << "x" << k << "): " 
              << gflops << " GFLOPS, " << elapsed << " ms" << std::endl;
}

void run_benchmarks() {
    std::cout << "=== SIMD Library Benchmarks ===" << std::endl;
    
    // Display CPU features
    const auto& features = CpuFeatures::detect();
    std::cout << "CPU Features: AVX2=" << features.avx2 
              << ", AVX512F=" << features.avx512f << std::endl;
    
    // DOT product benchmarks
    std::cout << "\n--- DOT Product ---" << std::endl;
    benchmark_dot<float>(1024);
    benchmark_dot<float>(8192);
    benchmark_dot<float>(65536);
    
    // SAXPY benchmarks
    std::cout << "\n--- SAXPY ---" << std::endl;
    benchmark_saxpy<float>(1024);
    benchmark_saxpy<float>(8192);
    benchmark_saxpy<float>(65536);
    
    // Matrix multiplication benchmarks
    std::cout << "\n--- Matrix Multiplication ---" << std::endl;
    benchmark_matmul<float>(64, 64, 64);
    benchmark_matmul<float>(128, 128, 128);
    benchmark_matmul<float>(256, 256, 256);
}

} // namespace benchmark

} // namespace simd

