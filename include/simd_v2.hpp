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
#include <fstream>
#include <map>
#include <complex>
#include <cmath>
#include <random>
#include <ctime>

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

// Add missing M_PI if not defined
#ifndef M_PI
#define M_PI 3.14159265358979323846
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
    
    // IMPROVED: More precise horizontal sum using manual extraction
    alignas(32) float sum_array[8];
    _mm256_store_ps(sum_array, sum);
    
    float result = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3] +
                   sum_array[4] + sum_array[5] + sum_array[6] + sum_array[7];
    
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



//==============================================================================
// IMPROVED: Optimized GEMM implementations (replaces poor performance)
//==============================================================================

void optimized_gemm_small(const float* A, const float* B, float* C, 
                          size_t M, size_t N, size_t K) {
    // Zero out result matrix
    std::memset(C, 0, M * N * sizeof(float));
    
    // For small matrices, use straightforward SIMD without blocking
    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; ++k) {
            __m256 a_vec = _mm256_broadcast_ss(&A[i * K + k]);
            
            size_t j = 0;
            // Process 8 elements at a time
            for (; j + 8 <= N; j += 8) {
                __m256 b_vec = _mm256_loadu_ps(&B[k * N + j]);
                __m256 c_vec = _mm256_loadu_ps(&C[i * N + j]);
                c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
                _mm256_storeu_ps(&C[i * N + j], c_vec);
            }
            
            // Handle remaining elements
            for (; j < N; ++j) {
                C[i * N + j] += A[i * K + k] * B[k * N + j];
            }
        }
    }
}

void optimized_gemm_blocked(const float* A, const float* B, float* C, 
                            size_t M, size_t N, size_t K) {
    // Block sizes optimized for L1 cache (32KB)
    constexpr size_t BLOCK_M = 64;
    constexpr size_t BLOCK_N = 64; 
    constexpr size_t BLOCK_K = 64;
    
    // Zero out result matrix
    std::memset(C, 0, M * N * sizeof(float));
    
    // Process in blocks to improve cache locality
    for (size_t mm = 0; mm < M; mm += BLOCK_M) {
        size_t m_end = std::min(mm + BLOCK_M, M);
        
        for (size_t nn = 0; nn < N; nn += BLOCK_N) {
            size_t n_end = std::min(nn + BLOCK_N, N);
            
            for (size_t kk = 0; kk < K; kk += BLOCK_K) {
                size_t k_end = std::min(kk + BLOCK_K, K);
                
                // Process the block with optimized inner kernels
                for (size_t i = mm; i < m_end; ++i) {
                    for (size_t k = kk; k < k_end; ++k) {
                        __m256 a_vec = _mm256_broadcast_ss(&A[i * K + k]);
                        
                        size_t j = nn;
                        for (; j + 8 <= n_end; j += 8) {
                            __m256 b_vec = _mm256_loadu_ps(&B[k * N + j]);
                            __m256 c_vec = _mm256_loadu_ps(&C[i * N + j]);
                            c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
                            _mm256_storeu_ps(&C[i * N + j], c_vec);
                        }
                        
                        // Handle remaining elements in this block
                        for (; j < n_end; ++j) {
                            C[i * N + j] += A[i * K + k] * B[k * N + j];
                        }
                    }
                }
            }
        }
    }
}


void gemm(const float* a, const float* b, float* c, size_t m, size_t n, size_t k) {
    // Adaptive strategy based on problem size
    if (m * n * k < 8192) {
        optimized_gemm_small(a, b, c, m, n, k);
    } else {
        optimized_gemm_blocked(a, b, c, m, n, k);
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

// Advanced AVX-512 Features
namespace advanced {

#ifdef __AVX512BF16__
// BF16 dot product using VPDPBF16
float dot_bf16(const uint16_t* a, const uint16_t* b, size_t n) {
    __m512 sum = _mm512_setzero_ps();
    size_t simd_n = n & ~31; // Process 32 BF16 elements at a time (16 pairs)
    
    for (size_t i = 0; i < simd_n; i += 32) {
        __m512bh va = _mm512_loadu_pbh(&a[i]);
        __m512bh vb = _mm512_loadu_pbh(&b[i]);
        sum = _mm512_dpbf16_ps(sum, va, vb);
    }
    
    // Handle remaining elements (fallback to scalar)
    float scalar_sum = _mm512_reduce_add_ps(sum);
    for (size_t i = simd_n; i < n; ++i) {
        // Convert BF16 to float and multiply
        float fa = *reinterpret_cast<const float*>(&a[i]) * (1.0f / 65536.0f);
        float fb = *reinterpret_cast<const float*>(&b[i]) * (1.0f / 65536.0f);
        scalar_sum += fa * fb;
    }
    
    return scalar_sum;
}
#endif

#ifdef __AVX512CD__
// Gather-scatter operations
void gather_scatter_operation(const float* source, float* dest, 
                             const int* indices, size_t count) {
    size_t simd_count = count & ~15;
    
    for (size_t i = 0; i < simd_count; i += 16) {
        __m512i idx = _mm512_loadu_si512(&indices[i]);
        __m512 gathered = _mm512_i32gather_ps(idx, source, 4);
        
        // Process gathered data (example: multiply by 2)
        gathered = _mm512_mul_ps(gathered, _mm512_set1_ps(2.0f));
        
        // Scatter back
        _mm512_i32scatter_ps(dest, idx, gathered, 4);
    }
    
    // Handle remaining elements
    for (size_t i = simd_count; i < count; ++i) {
        dest[indices[i]] = source[indices[i]] * 2.0f;
    }
}
#endif

} // namespace advanced

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
            // NOW USES THE IMPROVED VERSION
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
    scalar::conv2d_3x3(input, kernel, output, height, width);
}

//==============================================================================
// Epic E: Thread-Level Scaling & Parallel Wrappers
//==============================================================================

namespace parallel {

// OpenMP-based implementations
#ifdef _OPENMP

template<typename T>
T dot(const T* a, const T* b, size_t n) {
    T result = T(0);
    
    #pragma omp parallel for reduction(+:result)
    for (size_t i = 0; i < n; ++i) {
        result += a[i] * b[i];
    }
    
    return result;
}

template<typename T>
void saxpy(T alpha, const T* x, T* y, size_t n) {
    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        y[i] += alpha * x[i];
    }
}

template<typename T>
void matmul(const T* a, const T* b, T* c, size_t m, size_t n, size_t k) {
    #pragma omp parallel for collapse(2)
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

#else

// Std::thread fallback implementations
template<typename T>
T dot(const T* a, const T* b, size_t n) {
    const size_t num_threads = std::thread::hardware_concurrency();
    const size_t chunk_size = n / num_threads;
    
    std::vector<T> partial_sums(num_threads, T(0));
    std::vector<std::thread> threads;
    
    for (size_t t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = (t == num_threads - 1) ? n : start + chunk_size;
        
        threads.emplace_back([&, t, start, end]() {
            partial_sums[t] = simd::dot(&a[start], &b[start], end - start);
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    T result = T(0);
    for (const T& sum : partial_sums) {
        result += sum;
    }
    
    return result;
}

template<typename T>
void saxpy(T alpha, const T* x, T* y, size_t n) {
    const size_t num_threads = std::thread::hardware_concurrency();
    const size_t chunk_size = n / num_threads;
    
    std::vector<std::thread> threads;
    
    for (size_t t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = (t == num_threads - 1) ? n : start + chunk_size;
        
        threads.emplace_back([&, start, end]() {
            simd::saxpy(alpha, &x[start], &y[start], end - start);
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
}

template<typename T>
void matmul(const T* a, const T* b, T* c, size_t m, size_t n, size_t k) {
    const size_t num_threads = std::thread::hardware_concurrency();
    const size_t rows_per_thread = m / num_threads;
    
    std::vector<std::thread> threads;
    
    for (size_t t = 0; t < num_threads; ++t) {
        size_t start_row = t * rows_per_thread;
        size_t end_row = (t == num_threads - 1) ? m : start_row + rows_per_thread;
        
        threads.emplace_back([&, start_row, end_row]() {
            simd::matmul(&a[start_row * k], b, &c[start_row * n], 
                        end_row - start_row, n, k);
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
}

#endif

} // namespace parallel

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
    std::srand(static_cast<unsigned int>(std::time(nullptr)));
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
    
    std::cout << "DOT (n=" << n << "): " << std::fixed << std::setprecision(2) 
              << gflops << " GFLOPS, " << std::setprecision(3) << elapsed 
              << " ms, result=" << result << std::endl;
}

template<typename T>
void benchmark_saxpy(size_t n, int iterations = 1000) {
    util::aligned_vector<T> x(n), y(n);
    T alpha = static_cast<T>(2.0);
    
    std::srand(static_cast<unsigned int>(std::time(nullptr)));
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
    
    std::cout << "SAXPY (n=" << n << "): " << std::fixed << std::setprecision(2) 
              << gflops << " GFLOPS, " << std::setprecision(3) << elapsed 
              << " ms" << std::endl;
}

template<typename T>
void benchmark_matmul(size_t m, size_t n, size_t k, int iterations = 100) {
    util::aligned_vector<T> a(m * k), b(k * n), c(m * n);
    
    // Initialize matrices with random data
    std::srand(static_cast<unsigned int>(std::time(nullptr)));
    for (size_t i = 0; i < m * k; ++i) {
        a[i] = static_cast<T>(rand()) / RAND_MAX;
    }
    for (size_t i = 0; i < k * n; ++i) {
        b[i] = static_cast<T>(rand()) / RAND_MAX;
    }
    
    // Zero-initialize result matrix
    std::fill(c.begin(), c.end(), T(0));
    
    // Warmup run
    matmul(a.data(), b.data(), c.data(), m, n, k);
    
    Timer timer;
    
    // Benchmark runs
    for (int i = 0; i < iterations; ++i) {
        // Clear result matrix for each iteration
        std::fill(c.begin(), c.end(), T(0));
        matmul(a.data(), b.data(), c.data(), m, n, k);
    }
    
    double elapsed = timer.elapsed_ms();
    double ops_per_sec = (2.0 * m * n * k * iterations) / (elapsed / 1000.0);
    double gflops = ops_per_sec / 1e9;
    
    // Calculate memory bandwidth (rough estimate)
    double bytes_accessed = static_cast<double>(m * k + k * n + m * n) * sizeof(T) * iterations;
    double bandwidth_gb_s = bytes_accessed / (elapsed / 1000.0) / 1e9;
    
    std::cout << "MATMUL (" << m << "x" << n << "x" << k << "): " 
              << std::fixed << std::setprecision(2) << gflops << " GFLOPS, "
              << std::setprecision(1) << bandwidth_gb_s << " GB/s, "
              << std::setprecision(3) << elapsed << " ms" << std::endl;
    
    // Validate result (simple sanity check)
    T checksum = T(0);
    for (size_t i = 0; i < std::min(static_cast<size_t>(10), m * n); ++i) {
        checksum += c[i];
    }
    
    if (std::isnan(static_cast<double>(checksum)) || std::isinf(static_cast<double>(checksum))) {
        std::cout << "⚠️  WARNING: Invalid result detected in matrix multiplication!" << std::endl;
    }
}

void run_benchmarks() {
    std::cout << "=== SIMD Library Comprehensive Benchmarks ===" << std::endl;
    
    // Display CPU features
    const auto& features = CpuFeatures::detect();
    std::cout << "CPU Features: AVX2=" << features.avx2 
              << ", AVX512F=" << features.avx512f 
              << ", AVX512BW=" << features.avx512bw
              << ", AVX512VNNI=" << features.avx512vnni
              << ", AVX512BF16=" << features.avx512bf16 << std::endl;
    
    // Core SIMD benchmarks
    std::cout << "\n--- Core SIMD Operations ---" << std::endl;
    benchmark_dot<float>(1024);
    benchmark_dot<float>(8192);
    benchmark_dot<float>(65536);
    
    benchmark_saxpy<float>(1024);
    benchmark_saxpy<float>(8192);
    benchmark_saxpy<float>(65536);
    
    benchmark_matmul<float>(64, 64, 64);
    benchmark_matmul<float>(128, 128, 128);
    benchmark_matmul<float>(256, 256, 256);
    
    // Advanced AVX-512 benchmarks if available
#ifdef __AVX512F__
    if (features.avx512f) {
        std::cout << "\n--- Advanced AVX-512 Features ---" << std::endl;
        
#ifdef __AVX512CD__
        std::cout << "Testing gather/scatter operations..." << std::endl;
        std::vector<float> source(1000), dest(1000);
        std::vector<int> indices(100);
        
        for (size_t i = 0; i < 1000; ++i) source[i] = static_cast<float>(i);
        for (size_t i = 0; i < 100; ++i) indices[i] = static_cast<int>(i * 10);
        
        auto start = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < 1000; ++iter) {
            avx512::advanced::gather_scatter_operation(source.data(), dest.data(), 
                                                      indices.data(), indices.size());
        }
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration<double, std::milli>(end - start);
        std::cout << "Gather/Scatter (100 elements, 1000 iterations): " << duration.count() << " ms" << std::endl;
#endif
    }
#endif
}

} // namespace benchmark

//==============================================================================
// Hardware Bring-up & Performance Counter Integration
//==============================================================================

namespace hardware {

// Performance counter wrapper for different platforms
class PerformanceCounters {
public:
    struct Metrics {
        uint64_t cycles = 0;
        uint64_t instructions = 0;
        uint64_t cache_misses = 0;
        uint64_t cache_references = 0;
        double ipc = 0.0; // Instructions per cycle
        double cache_miss_rate = 0.0;
        double frequency_mhz = 0.0;
    };

private:
    bool counters_available_ = false;

public:
    PerformanceCounters() {
        initialize_counters();
    }
    
    void initialize_counters() {
#ifdef __linux__
        counters_available_ = true; // Simplified - real implementation would check
#endif
    }
    
    Metrics measure_kernel(std::function<void()> kernel) {
        Metrics metrics;
        
        if (!counters_available_) {
            // Fallback to time-based measurements
            auto start = std::chrono::high_resolution_clock::now();
            kernel();
            auto end = std::chrono::high_resolution_clock::now();
            
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
            metrics.cycles = duration.count() * 3; // Assume 3GHz for estimation
            return metrics;
        }
        
        // Run kernel
        kernel();
        
        // Mock metrics for demonstration
        metrics.cycles = 1000000;
        metrics.instructions = 800000;
        metrics.cache_misses = 1000;
        metrics.cache_references = 10000;
        metrics.ipc = static_cast<double>(metrics.instructions) / metrics.cycles;
        metrics.cache_miss_rate = static_cast<double>(metrics.cache_misses) / metrics.cache_references;
        metrics.frequency_mhz = 3000.0;
        
        return metrics;
    }
};

// Hardware bring-up utilities
namespace bringup {

struct HardwareInfo {
    std::string cpu_model = "Unknown CPU";
    std::string cpu_vendor = "Unknown";
    size_t cache_l1_size = 32 * 1024;
    size_t cache_l2_size = 256 * 1024;
    size_t cache_l3_size = 8 * 1024 * 1024;
    size_t num_cores = 1;
    size_t num_threads = 1;
    bool has_avx2 = false;
    bool has_avx512 = false;
    std::vector<std::string> supported_features;
};

HardwareInfo detect_hardware() {
    HardwareInfo info;
    
    const auto& features = simd::CpuFeatures::detect();
    info.has_avx2 = features.avx2;
    info.has_avx512 = features.avx512f;
    
    info.num_threads = std::thread::hardware_concurrency();
    info.num_cores = info.num_threads; // Simplified
    
    // Platform-specific hardware detection
#ifdef __linux__
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    while (std::getline(cpuinfo, line)) {
        if (line.find("model name") != std::string::npos) {
            size_t pos = line.find(":");
            if (pos != std::string::npos) {
                info.cpu_model = line.substr(pos + 2);
            }
            break;
        }
    }
#elif defined(_WIN32)
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    info.num_cores = sysinfo.dwNumberOfProcessors;
#endif
    
    // Feature list
    if (info.has_avx2) info.supported_features.push_back("AVX2");
    if (info.has_avx512) info.supported_features.push_back("AVX-512F");
    if (features.avx512bw) info.supported_features.push_back("AVX-512BW");
    if (features.avx512vl) info.supported_features.push_back("AVX-512VL");
    if (features.avx512vnni) info.supported_features.push_back("AVX-512VNNI");
    if (features.avx512bf16) info.supported_features.push_back("AVX-512BF16");
    
    return info;
}

void validate_expected_throughput(const std::string& operation, double measured_gflops, double expected_gflops) {
    double ratio = measured_gflops / expected_gflops;
    
    std::cout << "=== Hardware Validation: " << operation << " ===" << std::endl;
    std::cout << "Expected GFLOPS: " << expected_gflops << std::endl;
    std::cout << "Measured GFLOPS: " << measured_gflops << std::endl;
    std::cout << "Performance ratio: " << std::fixed << std::setprecision(2) << ratio << std::endl;
    
    if (ratio >= 0.8) {
        std::cout << "✅ PASS: Performance within expected range" << std::endl;
    } else if (ratio >= 0.5) {
        std::cout << "⚠️  WARN: Performance below expected but acceptable" << std::endl;
    } else {
        std::cout << "❌ FAIL: Performance significantly below expected" << std::endl;
    }
}

// ASIC/FPGA prototype support (framework)
class PrototypeInterface {
public:
    virtual ~PrototypeInterface() = default;
    virtual bool is_available() const = 0;
    virtual void upload_kernel(const std::string& kernel_name, const void* data, size_t size) = 0;
    virtual void execute_kernel(const std::string& kernel_name, void* inputs[], void* outputs[], size_t count) = 0;
    virtual std::vector<uint64_t> read_performance_counters() = 0;
};

// Mock FPGA interface for demonstration
class MockFPGAInterface : public PrototypeInterface {
private:
    bool available_ = false;
    std::map<std::string, std::vector<uint8_t>> uploaded_kernels_;
    
public:
    MockFPGAInterface() {
        available_ = std::getenv("SIMD_FPGA_AVAILABLE") != nullptr;
    }
    
    bool is_available() const override {
        return available_;
    }
    
    void upload_kernel(const std::string& kernel_name, const void* data, size_t size) override {
        if (!available_) throw std::runtime_error("FPGA not available");
        
        std::vector<uint8_t> kernel_data(static_cast<const uint8_t*>(data), 
                                        static_cast<const uint8_t*>(data) + size);
        uploaded_kernels_[kernel_name] = kernel_data;
        
        std::cout << "Uploaded kernel '" << kernel_name << "' (" << size << " bytes) to FPGA" << std::endl;
    }
    
    void execute_kernel(const std::string& kernel_name, void* inputs[], void* outputs[], size_t count) override {
        if (!available_) throw std::runtime_error("FPGA not available");
        if (uploaded_kernels_.find(kernel_name) == uploaded_kernels_.end()) {
            throw std::runtime_error("Kernel not uploaded: " + kernel_name);
        }
        
        std::cout << "Executing kernel '" << kernel_name << "' with " << count << " buffers" << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    std::vector<uint64_t> read_performance_counters() override {
        return {1000000, 800000, 50000}; // Mock cycles, instructions, cache misses
    }
};

void run_hardware_bringup_tests() {
    std::cout << "\n=== Hardware Bring-up Tests ===" << std::endl;
    
    // Detect hardware
    auto hw_info = detect_hardware();
    std::cout << "CPU: " << hw_info.cpu_model << std::endl;
    std::cout << "Cores: " << hw_info.num_cores << ", Threads: " << hw_info.num_threads << std::endl;
    std::cout << "L1 Cache: " << hw_info.cache_l1_size / 1024 << "KB" << std::endl;
    std::cout << "L2 Cache: " << hw_info.cache_l2_size / 1024 << "KB" << std::endl;
    std::cout << "L3 Cache: " << hw_info.cache_l3_size / 1024 / 1024 << "MB" << std::endl;
    
    std::cout << "Supported features: ";
    for (const auto& feature : hw_info.supported_features) {
        std::cout << feature << " ";
    }
    std::cout << std::endl;
    
    // Performance validation
    const size_t test_size = 1024 * 1024;
    simd::util::aligned_vector<float> a(test_size), b(test_size);
    for (size_t i = 0; i < test_size; ++i) {
        a[i] = static_cast<float>(i);
        b[i] = 2.0f;
    }
    
    // Measure DOT product performance
    auto start = std::chrono::high_resolution_clock::now();
    float result = simd::dot(a.data(), b.data(), test_size);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration<double>(end - start).count();
    double gflops = (2.0 * test_size) / duration / 1e9;
    
    // Expected performance based on hardware (rough estimates)
    double expected_gflops = 1.0; // Base scalar performance
    if (hw_info.has_avx2) expected_gflops = 8.0;
    if (hw_info.has_avx512) expected_gflops = 16.0;
    
    validate_expected_throughput("DOT Product", gflops, expected_gflops);
    
    // Test prototype interface if available
    MockFPGAInterface fpga;
    if (fpga.is_available()) {
        std::cout << "\n=== FPGA Prototype Testing ===" << std::endl;
        
        std::vector<uint8_t> kernel_binary = {0x48, 0x65, 0x6C, 0x6C, 0x6F}; // "Hello"
        fpga.upload_kernel("dot_product", kernel_binary.data(), kernel_binary.size());
        
        void* inputs[] = {a.data(), b.data()};
        void* outputs[] = {&result};
        fpga.execute_kernel("dot_product", inputs, outputs, 2);
        
        auto counters = fpga.read_performance_counters();
        std::cout << "FPGA Performance counters: ";
        for (auto counter : counters) {
            std::cout << counter << " ";
        }
        std::cout << std::endl;
    }
}

} // namespace bringup

} // namespace hardware

//==============================================================================
// Documentation & API Guide Functions
//==============================================================================

namespace documentation {

void print_api_guide() {
    std::cout << R"(
=== SIMD Library API Guide ===

BASIC USAGE:
  #include "simd_v2.hpp"
  
  // Automatic dispatch based on CPU features
  float result = simd::dot(a, b, n);           // DOT product
  simd::saxpy(alpha, x, y, n);                 // y = alpha*x + y
  simd::matmul(A, B, C, m, n, k);             // C = A*B (matrix multiply)
  simd::conv2d(input, kernel, output, h, w);  // 2D convolution

PARALLEL OPERATIONS:
  // Multi-threaded versions (OpenMP or std::thread)
  float result = simd::parallel::dot(a, b, n);
  simd::parallel::saxpy(alpha, x, y, n);
  simd::parallel::matmul(A, B, C, m, n, k);

MEMORY MANAGEMENT:
  // Aligned allocators for optimal SIMD performance
  simd::util::aligned_vector<float> data(size);
  size_t padded = simd::util::padded_size<float>(n, 64);

FEATURE DETECTION:
  auto features = simd::CpuFeatures::detect();
  if (features.avx512f) { /* use AVX-512 specific code */ }

ENVIRONMENT CONTROLS:
  export SIMD_FORCE_SCALAR=1  # Force scalar implementation
  export SIMD_FPGA_AVAILABLE=1  # Enable FPGA prototype interface

PERFORMANCE TIPS:
  ✅ Use aligned memory (simd::util::aligned_vector)
  ✅ Process large arrays (>1024 elements)
  ✅ Avoid frequent small operations
  ✅ Consider parallel versions for multi-core
  ✅ Use appropriate data types (float for best SIMD support)

ADVANCED FEATURES:
  - Hardware performance counters
  - FPGA/ASIC prototype interface
  - BF16 operations (AVX-512BF16)
  - Gather/scatter operations
)";
}

void print_optimization_playbook() {
    std::cout << R"(
=== SIMD Optimization Playbook ===

DATA-DRIVEN WORKFLOW:
  1. PROFILE  → Identify hotspots with tools or built-in Timer
  2. PORT     → Replace scalar with SIMD operations
  3. TUNE     → Adjust tile sizes, unroll factors
  4. VALIDATE → Compare results with scalar baseline

PROFILING CHECKLIST:
  □ Measure baseline scalar performance
  □ Identify memory vs compute bound operations
  □ Check cache hit rates and memory bandwidth
  □ Profile with realistic data sizes
  □ Test on target hardware architecture

PORTING STRATEGIES:
  □ Start with highest-impact operations (DOT, SAXPY, GEMM)
  □ Use automatic dispatch API for portability
  □ Handle edge cases (non-aligned, odd sizes)
  □ Validate numerical accuracy vs scalar

TUNING PARAMETERS:
  □ Tile sizes: balance cache usage vs parallelism
  □ Unroll factors: optimize for instruction pipeline
  □ Memory access patterns: prefer sequential over random
  □ Thread count: match hardware capabilities

VALIDATION METHODS:
  □ Bit-exact comparison for integer operations
  □ Relative error < 1e-5 for floating point
  □ Test edge cases (zero, negative, NaN, infinity)
  □ Stress test with random data

COMMON PITFALLS:
  ❌ Ignoring memory alignment requirements
  ❌ Not handling array size edge cases
  ❌ Assuming SIMD is always faster
  ❌ Neglecting numerical stability
  ❌ Over-optimizing memory-bound operations
)";
}

} // namespace documentation

//==============================================================================
// Test Correctness Function
//==============================================================================

#ifdef SIMD_INCLUDE_EXAMPLES

#include <cassert>

inline void test_correctness() {
    const size_t n = 1000;
    simd::util::aligned_vector<float> a(n), b(n), y1(n), y2(n);
    
    // Initialize test data
    for (size_t i = 0; i < n; ++i) {
        a[i] = static_cast<float>(i + 1);
        b[i] = static_cast<float>(2 * i + 1);
        y1[i] = y2[i] = static_cast<float>(i);
    }
    
    // Test DOT product
    float dot_scalar = simd::scalar::dot(a.data(), b.data(), n);
    float dot_simd = simd::dot(a.data(), b.data(), n);
    assert(std::abs(dot_scalar - dot_simd) < 1e-5f);
    
    // Test SAXPY
    float alpha = 2.5f;
    simd::scalar::saxpy(alpha, a.data(), y1.data(), n);
    simd::saxpy(alpha, a.data(), y2.data(), n);
    
    for (size_t i = 0; i < n; ++i) {
        assert(std::abs(y1[i] - y2[i]) < 1e-5f);
    }
    
    std::cout << "All correctness tests passed!" << std::endl;
}

#endif // SIMD_INCLUDE_EXAMPLES

} // namespace simd


