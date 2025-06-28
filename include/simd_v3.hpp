//==============================================================================
// High-Performance SIMD Library - Enhanced with Masked Operations
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

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace simd {

//==============================================================================
// Enhanced CPU Feature Detection
//==============================================================================
class CpuFeatures {
public:
    struct Features {
        bool avx2 = false;
        bool avx512f = false;
        bool avx512bw = false;
        bool avx512vl = false;
        bool avx512vnni = false;
        bool avx512bf16 = false;
        bool avx512fp16 = false;
        bool f16c = false;
        bool fma = false;
    };

    static const Features& detect() {
        static Features features = []() {
            Features f;
            
#ifdef _WIN32
            int cpuInfo[4];
            __cpuid(cpuInfo, 0);
            int maxId = cpuInfo[0];
            
            if (maxId >= 1) {
                __cpuid(cpuInfo, 1);
                f.f16c = (cpuInfo[2] & (1 << 29)) != 0;
                f.fma = (cpuInfo[2] & (1 << 12)) != 0;
            }
            
            if (maxId >= 7) {
                __cpuidex(cpuInfo, 7, 0);
                f.avx2 = (cpuInfo[1] & (1 << 5)) != 0;
                f.avx512f = (cpuInfo[1] & (1 << 16)) != 0;
                f.avx512bw = (cpuInfo[1] & (1 << 30)) != 0;
                f.avx512vl = (cpuInfo[1] & (1 << 31)) != 0;
                f.avx512vnni = (cpuInfo[2] & (1 << 11)) != 0;
                
                __cpuidex(cpuInfo, 7, 1);
                f.avx512bf16 = (cpuInfo[0] & (1 << 5)) != 0;
                
                __cpuidex(cpuInfo, 7, 0);
                f.avx512fp16 = (cpuInfo[3] & (1 << 23)) != 0;
            }
#else
            unsigned int eax, ebx, ecx, edx;
            
            if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
                f.f16c = (ecx & (1 << 29)) != 0;
                f.fma = (ecx & (1 << 12)) != 0;
            }
            
            if (__get_cpuid_max(0, nullptr) >= 7) {
                __cpuid_count(7, 0, eax, ebx, ecx, edx);
                f.avx2 = (ebx & (1 << 5)) != 0;
                f.avx512f = (ebx & (1 << 16)) != 0;
                f.avx512bw = (ebx & (1 << 30)) != 0;
                f.avx512vl = (ebx & (1 << 31)) != 0;
                f.avx512vnni = (ecx & (1 << 11)) != 0;
                f.avx512fp16 = (edx & (1 << 23)) != 0;
                
                __cpuid_count(7, 1, eax, ebx, ecx, edx);
                f.avx512bf16 = (eax & (1 << 5)) != 0;
            }
#endif
            
            return f;
        }();
        
        return features;
    }
};

//==============================================================================
// Half-precision (f16) support structures
//==============================================================================
using f16_t = uint16_t;

namespace util {

// Convert float to f16
inline f16_t f32_to_f16(float f) {
#ifdef __F16C__
    return _cvtss_sh(f, 0);
#else
    union { float f; uint32_t i; } u = { f };
    uint32_t sign = (u.i >> 31) & 0x1;
    uint32_t exp = (u.i >> 23) & 0xff;
    uint32_t frac = u.i & 0x7fffff;
    
    if (exp == 0xff) return (sign << 15) | 0x7c00 | (frac ? 1 : 0);
    if (exp == 0) return sign << 15;
    
    int new_exp = exp - 127 + 15;
    if (new_exp >= 31) return (sign << 15) | 0x7c00;
    if (new_exp <= 0) return sign << 15;
    
    return (sign << 15) | (new_exp << 10) | (frac >> 13);
#endif
}

// Convert f16 to float
inline float f16_to_f32(f16_t h) {
#ifdef __F16C__
    return _cvtsh_ss(h);
#else
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t frac = h & 0x3ff;
    
    if (exp == 0) {
        if (frac == 0) {
            union { float f; uint32_t i; } u = { 0 };
            u.i = sign << 31;
            return u.f;
        }
        while (!(frac & 0x400)) {
            frac <<= 1;
            exp--;
        }
        exp++;
        frac &= 0x3ff;
    } else if (exp == 31) {
        union { float f; uint32_t i; } u;
        u.i = (sign << 31) | 0x7f800000 | (frac << 13);
        return u.f;
    }
    
    exp += 127 - 15;
    union { float f; uint32_t i; } u;
    u.i = (sign << 31) | (exp << 23) | (frac << 13);
    return u.f;
#endif
}

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

} // namespace util

//==============================================================================
// Math utilities
//==============================================================================
namespace math {

inline float fast_rsqrt(float number) {
    union { float f; uint32_t i; } conv;
    conv.f = number;
    conv.i = 0x5F1FFFF9 - (conv.i >> 1);
    conv.f *= 0.703952253f * (2.38924456f - number * conv.f * conv.f);
    return conv.f;
}

} // namespace math

//==============================================================================
// Enhanced AVX2 implementation
//==============================================================================
namespace avx2 {

#ifdef __AVX2__

__attribute__((target("avx2,fma")))
float l2_squared_distance(const float* a, const float* b, size_t n) {
    __m256 d2_vec = _mm256_set1_ps(0);
    size_t i = 0;
    
    for (; i + 8 <= n; i += 8) {
        __m256 a_vec = _mm256_loadu_ps(a + i);
        __m256 b_vec = _mm256_loadu_ps(b + i);
        __m256 d_vec = _mm256_sub_ps(a_vec, b_vec);
        d2_vec = _mm256_fmadd_ps(d_vec, d_vec, d2_vec);
    }
    
    d2_vec = _mm256_add_ps(_mm256_permute2f128_ps(d2_vec, d2_vec, 1), d2_vec);
    d2_vec = _mm256_hadd_ps(d2_vec, d2_vec);
    d2_vec = _mm256_hadd_ps(d2_vec, d2_vec);
    
    float result;
    _mm_store_ss(&result, _mm256_castps256_ps128(d2_vec));
    
    for (; i < n; ++i) {
        float diff = a[i] - b[i];
        result += diff * diff;
    }
    
    return result;
}

__attribute__((target("avx2,f16c,fma")))
float dot_f16(const f16_t* a, const f16_t* b, size_t n) {
    __m256 sum = _mm256_setzero_ps();
    size_t i = 0;
    
    for (; i + 8 <= n; i += 8) {
        __m128i a_f16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&a[i]));
        __m128i b_f16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&b[i]));
        
        __m256 a_f32 = _mm256_cvtph_ps(a_f16);
        __m256 b_f32 = _mm256_cvtph_ps(b_f16);
        
        sum = _mm256_fmadd_ps(a_f32, b_f32, sum);
    }
    
    alignas(32) float sum_array[8];
    _mm256_store_ps(sum_array, sum);
    
    float result = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3] +
                   sum_array[4] + sum_array[5] + sum_array[6] + sum_array[7];
    
    for (; i < n; ++i) {
        result += util::f16_to_f32(a[i]) * util::f16_to_f32(b[i]);
    }
    
    return result;
}

__attribute__((target("avx2,fma")))
float cosine_distance(const float* a, const float* b, size_t n) {
    __m256 ab_sum = _mm256_setzero_ps();
    __m256 a2_sum = _mm256_setzero_ps();
    __m256 b2_sum = _mm256_setzero_ps();
    
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 a_vec = _mm256_loadu_ps(&a[i]);
        __m256 b_vec = _mm256_loadu_ps(&b[i]);
        
        ab_sum = _mm256_fmadd_ps(a_vec, b_vec, ab_sum);
        a2_sum = _mm256_fmadd_ps(a_vec, a_vec, a2_sum);
        b2_sum = _mm256_fmadd_ps(b_vec, b_vec, b2_sum);
    }
    
    alignas(32) float ab_arr[8], a2_arr[8], b2_arr[8];
    _mm256_store_ps(ab_arr, ab_sum);
    _mm256_store_ps(a2_arr, a2_sum);
    _mm256_store_ps(b2_arr, b2_sum);
    
    float ab = ab_arr[0] + ab_arr[1] + ab_arr[2] + ab_arr[3] + 
               ab_arr[4] + ab_arr[5] + ab_arr[6] + ab_arr[7];
    float a2 = a2_arr[0] + a2_arr[1] + a2_arr[2] + a2_arr[3] + 
               a2_arr[4] + a2_arr[5] + a2_arr[6] + a2_arr[7];
    float b2 = b2_arr[0] + b2_arr[1] + b2_arr[2] + b2_arr[3] + 
               b2_arr[4] + b2_arr[5] + b2_arr[6] + b2_arr[7];
    
    for (; i < n; ++i) {
        ab += a[i] * b[i];
        a2 += a[i] * a[i];
        b2 += b[i] * b[i];
    }
    
    return 1.0f - ab * math::fast_rsqrt(a2 * b2);
}

// FIX: Completely rewritten I8 dot product for AVX2
__attribute__((target("avx2")))
int32_t dot_i8_avx2(const int8_t* a, const int8_t* b, size_t n) {
    __m256i sum = _mm256_setzero_si256();
    size_t i = 0;
    
    // Process 32 int8 elements at a time
    for (; i + 32 <= n; i += 32) {
        __m256i a_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&a[i]));
        __m256i b_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&b[i]));
        
        // Split into 16-bit to prevent overflow
        __m256i a_lo = _mm256_unpacklo_epi8(a_vec, _mm256_cmpgt_epi8(_mm256_setzero_si256(), a_vec));
        __m256i a_hi = _mm256_unpackhi_epi8(a_vec, _mm256_cmpgt_epi8(_mm256_setzero_si256(), a_vec));
        __m256i b_lo = _mm256_unpacklo_epi8(b_vec, _mm256_cmpgt_epi8(_mm256_setzero_si256(), b_vec));
        __m256i b_hi = _mm256_unpackhi_epi8(b_vec, _mm256_cmpgt_epi8(_mm256_setzero_si256(), b_vec));
        
        // Multiply and accumulate
        __m256i prod_lo = _mm256_madd_epi16(a_lo, b_lo);
        __m256i prod_hi = _mm256_madd_epi16(a_hi, b_hi);
        
        sum = _mm256_add_epi32(sum, prod_lo);
        sum = _mm256_add_epi32(sum, prod_hi);
    }
    
    // Process remaining 16 elements
    if (i + 16 <= n) {
        __m128i a_vec = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&a[i]));
        __m128i b_vec = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&b[i]));
        
        __m128i a_lo = _mm_unpacklo_epi8(a_vec, _mm_cmpgt_epi8(_mm_setzero_si128(), a_vec));
        __m128i a_hi = _mm_unpackhi_epi8(a_vec, _mm_cmpgt_epi8(_mm_setzero_si128(), a_vec));
        __m128i b_lo = _mm_unpacklo_epi8(b_vec, _mm_cmpgt_epi8(_mm_setzero_si128(), b_vec));
        __m128i b_hi = _mm_unpackhi_epi8(b_vec, _mm_cmpgt_epi8(_mm_setzero_si128(), b_vec));
        
        __m128i prod_lo = _mm_madd_epi16(a_lo, b_lo);
        __m128i prod_hi = _mm_madd_epi16(a_hi, b_hi);
        
        sum = _mm256_add_epi32(sum, _mm256_insertf128_si256(_mm256_castsi128_si256(prod_lo), prod_hi, 1));
        i += 16;
    }
    
    // Horizontal sum
    alignas(32) int32_t sum_array[8];
    _mm256_store_si256(reinterpret_cast<__m256i*>(sum_array), sum);
    
    int32_t result = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3] +
                     sum_array[4] + sum_array[5] + sum_array[6] + sum_array[7];
    
    // Handle remaining elements
    for (; i < n; ++i) {
        result += static_cast<int32_t>(a[i]) * static_cast<int32_t>(b[i]);
    }
    
    return result;
}

#endif // __AVX2__

} // namespace avx2

//==============================================================================
// Fixed AVX-512 implementation
//==============================================================================
namespace avx512 {

#ifdef __AVX512F__

__attribute__((target("avx512f,avx512vl")))
float l2_squared_distance(const float* a, const float* b, size_t n) {
    __m512 d2_vec = _mm512_set1_ps(0);
    
    for (size_t i = 0; i < n; i += 16) {
        __mmask16 mask = (n - i >= 16) ? 0xFFFF : ((1u << (n - i)) - 1u);
        
        __m512 a_vec = _mm512_maskz_loadu_ps(mask, &a[i]);
        __m512 b_vec = _mm512_maskz_loadu_ps(mask, &b[i]);
        __m512 d_vec = _mm512_sub_ps(a_vec, b_vec);
        d2_vec = _mm512_fmadd_ps(d_vec, d_vec, d2_vec);
    }
    
    return _mm512_reduce_add_ps(d2_vec);
}

__attribute__((target("avx512f,avx512vl")))
float cosine_distance(const float* a, const float* b, size_t n) {
    __m512 ab_sum = _mm512_setzero_ps();
    __m512 a2_sum = _mm512_setzero_ps();
    __m512 b2_sum = _mm512_setzero_ps();
    
    for (size_t i = 0; i < n; i += 16) {
        __mmask16 mask = (n - i >= 16) ? 0xFFFF : ((1u << (n - i)) - 1u);
        
        __m512 a_vec = _mm512_maskz_loadu_ps(mask, &a[i]);
        __m512 b_vec = _mm512_maskz_loadu_ps(mask, &b[i]);
        
        ab_sum = _mm512_fmadd_ps(a_vec, b_vec, ab_sum);
        a2_sum = _mm512_fmadd_ps(a_vec, a_vec, a2_sum);
        b2_sum = _mm512_fmadd_ps(b_vec, b_vec, b2_sum);
    }
    
    float ab = _mm512_reduce_add_ps(ab_sum);
    float a2 = _mm512_reduce_add_ps(a2_sum);
    float b2 = _mm512_reduce_add_ps(b2_sum);
    
    return 1.0f - ab * math::fast_rsqrt(a2 * b2);
}

// FIX: Completely rewritten I8 dot product for AVX-512
__attribute__((target("avx512f,avx512bw,avx512vl")))
int32_t dot_i8_avx512(const int8_t* a, const int8_t* b, size_t n) {
    __m512i sum = _mm512_setzero_si512();
    
    // Process 64 int8 elements at a time
    for (size_t i = 0; i < n; i += 64) {
        __mmask64 mask = (n - i >= 64) ? 0xFFFFFFFFFFFFFFFFULL : ((1ULL << (n - i)) - 1ULL);
        
        __m512i a_vec = _mm512_maskz_loadu_epi8(mask, &a[i]);
        __m512i b_vec = _mm512_maskz_loadu_epi8(mask, &b[i]);
        
        // Convert to 16-bit to prevent overflow - FIXED VERSION
        // Proper sign extension using _mm512_cvtepi8_epi16
        __m256i a_lo_256 = _mm512_extracti32x8_epi32(a_vec, 0);
        __m256i a_hi_256 = _mm512_extracti32x8_epi32(a_vec, 1);
        __m256i b_lo_256 = _mm512_extracti32x8_epi32(b_vec, 0);
        __m256i b_hi_256 = _mm512_extracti32x8_epi32(b_vec, 1);
        
        __m512i a_lo = _mm512_cvtepi8_epi16(a_lo_256);
        __m512i a_hi = _mm512_cvtepi8_epi16(a_hi_256);
        __m512i b_lo = _mm512_cvtepi8_epi16(b_lo_256);
        __m512i b_hi = _mm512_cvtepi8_epi16(b_hi_256);
        
        // Multiply and accumulate
        __m512i prod_lo = _mm512_madd_epi16(a_lo, b_lo);
        __m512i prod_hi = _mm512_madd_epi16(a_hi, b_hi);
        
        sum = _mm512_add_epi32(sum, prod_lo);
        sum = _mm512_add_epi32(sum, prod_hi);
    }
    
    return _mm512_reduce_add_epi32(sum);
}

// FIX: Simplified and correct VNNI implementation
#ifdef __AVX512VNNI__
__attribute__((target("avx512vnni,avx512bw,avx512vl")))
int32_t dot_i8_vnni_fixed(const int8_t* a, const int8_t* b, size_t n) {
    // VNNI requires specific data layout - use simpler approach
    return dot_i8_avx512(a, b, n);  // Fall back to safer AVX-512 implementation
}
#endif

#ifdef __AVX512BF16__
__attribute__((target("avx512bf16,avx512vl")))
float dot_bf16(const uint16_t* a, const uint16_t* b, size_t n) {
    __m512 sum = _mm512_setzero_ps();
    
    for (size_t i = 0; i < n; i += 32) {
        size_t remaining = n - i;
        
        if (remaining >= 32) {
            __m512bh a_vec = _mm512_loadu_pbh(&a[i]);
            __m512bh b_vec = _mm512_loadu_pbh(&b[i]);
            sum = _mm512_dpbf16_ps(sum, a_vec, b_vec);
        } else {
            for (size_t j = i; j < n; ++j) {
                union { uint32_t i; float f; } fa = { static_cast<uint32_t>(a[j]) << 16 };
                union { uint32_t i; float f; } fb = { static_cast<uint32_t>(b[j]) << 16 };
                sum = _mm512_add_ps(sum, _mm512_set1_ps(fa.f * fb.f));
            }
            break;
        }
    }
    
    return _mm512_reduce_add_ps(sum);
}
#endif

#endif // __AVX512F__

} // namespace avx512

//==============================================================================
// Fixed public API functions
//==============================================================================

template<typename T>
T l2_squared_distance(const T* a, const T* b, size_t n) {
    const auto& features = CpuFeatures::detect();
    
    if constexpr (std::is_same_v<T, float>) {
#ifdef __AVX512F__
        if (features.avx512f && features.avx512vl) {
            return avx512::l2_squared_distance(a, b, n);
        }
#endif
#ifdef __AVX2__
        if (features.avx2) {
            return avx2::l2_squared_distance(a, b, n);
        }
#endif
    }
    
    T result = T(0);
    for (size_t i = 0; i < n; ++i) {
        T diff = a[i] - b[i];
        result += diff * diff;
    }
    return result;
}

template<typename T>
T cosine_distance(const T* a, const T* b, size_t n) {
    const auto& features = CpuFeatures::detect();
    
    if constexpr (std::is_same_v<T, float>) {
#ifdef __AVX512F__
        if (features.avx512f && features.avx512vl) {
            return avx512::cosine_distance(a, b, n);
        }
#endif
#ifdef __AVX2__
        if (features.avx2) {
            return avx2::cosine_distance(a, b, n);
        }
#endif
    }
    
    T ab = T(0), a2 = T(0), b2 = T(0);
    for (size_t i = 0; i < n; ++i) {
        ab += a[i] * b[i];
        a2 += a[i] * a[i];
        b2 += b[i] * b[i];
    }
    return T(1) - ab / std::sqrt(a2 * b2);
}

float dot_f16(const f16_t* a, const f16_t* b, size_t n) {
    const auto& features = CpuFeatures::detect();
    
#ifdef __AVX2__
    if (features.avx2 && features.f16c) {
        return avx2::dot_f16(a, b, n);
    }
#endif
    
    float result = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        result += util::f16_to_f32(a[i]) * util::f16_to_f32(b[i]);
    }
    return result;
}

// FIX: Completely rewritten I8 dot product dispatcher
int32_t dot_i8(const int8_t* a, const int8_t* b, size_t n) {
    const auto& features = CpuFeatures::detect();
    
#ifdef __AVX512F__
    if (features.avx512f && features.avx512bw && features.avx512vl) {
        return avx512::dot_i8_avx512(a, b, n);
    }
#endif
#ifdef __AVX2__
    if (features.avx2) {
        return avx2::dot_i8_avx2(a, b, n);
    }
#endif
    
    // Scalar fallback
    int32_t result = 0;
    for (size_t i = 0; i < n; ++i) {
        result += static_cast<int32_t>(a[i]) * static_cast<int32_t>(b[i]);
    }
    return result;
}

float dot_bf16(const uint16_t* a, const uint16_t* b, size_t n) {
    const auto& features = CpuFeatures::detect();
    
#ifdef __AVX512BF16__
    if (features.avx512bf16) {
        return avx512::dot_bf16(a, b, n);
    }
#endif
    
    float result = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        union { uint32_t i; float f; } fa = { static_cast<uint32_t>(a[i]) << 16 };
        union { uint32_t i; float f; } fb = { static_cast<uint32_t>(b[i]) << 16 };
        result += fa.f * fb.f;
    }
    return result;
}

// FIX: Rewritten dot product function with proper dispatching
template<typename T>
T dot(const T* a, const T* b, size_t n) {
    const auto& features = CpuFeatures::detect();
    
    if constexpr (std::is_same_v<T, float>) {
#ifdef __AVX512F__
        if (features.avx512f) {
            __m512 sum = _mm512_setzero_ps();
            
            for (size_t i = 0; i < n; i += 16) {
                __mmask16 mask = (n - i >= 16) ? 0xFFFF : ((1u << (n - i)) - 1u);
                __m512 va = _mm512_maskz_loadu_ps(mask, &a[i]);
                __m512 vb = _mm512_maskz_loadu_ps(mask, &b[i]);
                sum = _mm512_fmadd_ps(va, vb, sum);
            }
            
            return _mm512_reduce_add_ps(sum);
        }
#endif
#ifdef __AVX2__
        if (features.avx2) {
            __m256 sum = _mm256_setzero_ps();
            size_t simd_n = n & ~7;
            
            for (size_t i = 0; i < simd_n; i += 8) {
                __m256 va = _mm256_loadu_ps(&a[i]);
                __m256 vb = _mm256_loadu_ps(&b[i]);
                sum = _mm256_fmadd_ps(va, vb, sum);
            }
            
            alignas(32) float sum_array[8];
            _mm256_store_ps(sum_array, sum);
            
            float result = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3] +
                           sum_array[4] + sum_array[5] + sum_array[6] + sum_array[7];
            
            for (size_t i = simd_n; i < n; ++i) {
                result += a[i] * b[i];
            }
            
            return result;
        }
#endif
    }
    
    // Scalar fallback
    T result = T(0);
    for (size_t i = 0; i < n; ++i) {
        result += a[i] * b[i];
    }
    return result;
}

// Enhanced SAXPY with masked operations
template<typename T>
void saxpy(T alpha, const T* x, T* y, size_t n) {
    const auto& features = CpuFeatures::detect();
    
    if constexpr (std::is_same_v<T, float>) {
#ifdef __AVX512F__
        if (features.avx512f) {
            __m512 valpha = _mm512_set1_ps(alpha);
            size_t i = 0;
            
            // Process full 16-element chunks
            for (; i + 16 <= n; i += 16) {
                __m512 vx = _mm512_loadu_ps(&x[i]);
                __m512 vy = _mm512_loadu_ps(&y[i]);
                vy = _mm512_fmadd_ps(valpha, vx, vy);
                _mm512_storeu_ps(&y[i], vy);
            }
            
            // Handle remaining elements with mask
            if (i < n) {
                size_t remaining = n - i;
                __mmask16 mask = (1u << remaining) - 1u;
                __m512 vx = _mm512_maskz_loadu_ps(mask, &x[i]);
                __m512 vy = _mm512_maskz_loadu_ps(mask, &y[i]);
                vy = _mm512_fmadd_ps(valpha, vx, vy);
                _mm512_mask_storeu_ps(&y[i], mask, vy);
            }
            return;
        }
#endif
#ifdef __AVX2__
        if (features.avx2) {
            __m256 valpha = _mm256_broadcast_ss(&alpha);
            size_t i = 0;
            
            // Process full 8-element chunks
            for (; i + 8 <= n; i += 8) {
                __m256 vx = _mm256_loadu_ps(&x[i]);
                __m256 vy = _mm256_loadu_ps(&y[i]);
                vy = _mm256_fmadd_ps(valpha, vx, vy);
                _mm256_storeu_ps(&y[i], vy);
            }
            
            // Handle remaining elements scalar
            for (; i < n; ++i) {
                y[i] += alpha * x[i];
            }
            return;
        }
#endif
    }
    
    // Scalar fallback
    for (size_t i = 0; i < n; ++i) {
        y[i] += alpha * x[i];
    }
}


//==============================================================================
// Binary operations for text processing and hashing
//==============================================================================
namespace binary {

uint32_t hamming_distance(const uint8_t* a, const uint8_t* b, size_t n) {
    const auto& features = CpuFeatures::detect();
    
#ifdef __AVX512BW__
    if (features.avx512bw) {
        __m512i count = _mm512_setzero_si512();
        
        for (size_t i = 0; i < n; i += 64) {
            __mmask64 mask = (n - i >= 64) ? 0xFFFFFFFFFFFFFFFFULL : ((1ULL << (n - i)) - 1ULL);
            
            __m512i va = _mm512_maskz_loadu_epi8(mask, &a[i]);
            __m512i vb = _mm512_maskz_loadu_epi8(mask, &b[i]);
            __m512i xor_result = _mm512_xor_si512(va, vb);
            
#ifdef __AVX512VPOPCNTDQ__
            __m512i popcnt = _mm512_popcnt_epi8(xor_result);
#else
            // Fallback popcount for systems without VPOPCNTDQ
            __m512i popcnt = _mm512_setzero_si512();
            for (int bit = 0; bit < 8; ++bit) {
                __m512i mask_bit = _mm512_set1_epi8(1 << bit);
                __m512i bit_set = _mm512_and_si512(xor_result, mask_bit);
                __m512i bit_count = _mm512_srli_epi16(bit_set, bit);
                popcnt = _mm512_add_epi8(popcnt, bit_count);
            }
#endif
            count = _mm512_add_epi64(count, _mm512_sad_epu8(popcnt, _mm512_setzero_si512()));
        }
        
        return static_cast<uint32_t>(_mm512_reduce_add_epi64(count));
    }
#endif

#ifdef __AVX2__
    if (features.avx2) {
        __m256i count = _mm256_setzero_si256();
        size_t simd_n = n & ~31;
        
        for (size_t i = 0; i < simd_n; i += 32) {
            __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&a[i]));
            __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&b[i]));
            __m256i xor_result = _mm256_xor_si256(va, vb);
            
            const __m256i lookup = _mm256_setr_epi8(
                0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
                0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
            
            __m256i low_mask = _mm256_set1_epi8(0x0f);
            __m256i lo = _mm256_and_si256(xor_result, low_mask);
            __m256i hi = _mm256_and_si256(_mm256_srli_epi16(xor_result, 4), low_mask);
            
            __m256i popcnt_lo = _mm256_shuffle_epi8(lookup, lo);
            __m256i popcnt_hi = _mm256_shuffle_epi8(lookup, hi);
            __m256i popcnt = _mm256_add_epi8(popcnt_lo, popcnt_hi);
            
            count = _mm256_add_epi64(count, _mm256_sad_epu8(popcnt, _mm256_setzero_si256()));
        }
        
        uint64_t result[4];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(result), count);
        uint32_t total = static_cast<uint32_t>(result[0] + result[1] + result[2] + result[3]);
        
        for (size_t i = simd_n; i < n; ++i) {
            total += __builtin_popcount(a[i] ^ b[i]);
        }
        
        return total;
    }
#endif
    
    uint32_t distance = 0;
    for (size_t i = 0; i < n; ++i) {
        distance += __builtin_popcount(a[i] ^ b[i]);
    }
    return distance;
}

float jaccard_distance(const uint8_t* a, const uint8_t* b, size_t n) {
    uint32_t intersection = 0;
    uint32_t union_count = 0;
    
    const auto& features = CpuFeatures::detect();
    
#ifdef __AVX512BW__
    if (features.avx512bw) {
        __m512i intersect_sum = _mm512_setzero_si512();
        __m512i union_sum = _mm512_setzero_si512();
        
        for (size_t i = 0; i < n; i += 64) {
            __mmask64 mask = (n - i >= 64) ? 0xFFFFFFFFFFFFFFFFULL : ((1ULL << (n - i)) - 1ULL);
            
            __m512i va = _mm512_maskz_loadu_epi8(mask, &a[i]);
            __m512i vb = _mm512_maskz_loadu_epi8(mask, &b[i]);
            
            __m512i intersect = _mm512_and_si512(va, vb);
            __m512i union_vec = _mm512_or_si512(va, vb);
            
#ifdef __AVX512VPOPCNTDQ__
            __m512i intersect_popcnt = _mm512_popcnt_epi8(intersect);
            __m512i union_popcnt = _mm512_popcnt_epi8(union_vec);
#else
            // Software popcount fallback
            __m512i intersect_popcnt = _mm512_setzero_si512();
            __m512i union_popcnt = _mm512_setzero_si512();
            
            for (int bit = 0; bit < 8; ++bit) {
                __m512i mask_bit = _mm512_set1_epi8(1 << bit);
                
                __m512i intersect_bit = _mm512_and_si512(intersect, mask_bit);
                __m512i union_bit = _mm512_and_si512(union_vec, mask_bit);
                
                intersect_popcnt = _mm512_add_epi8(intersect_popcnt, _mm512_srli_epi16(intersect_bit, bit));
                union_popcnt = _mm512_add_epi8(union_popcnt, _mm512_srli_epi16(union_bit, bit));
            }
#endif
            
            intersect_sum = _mm512_add_epi64(intersect_sum, 
                _mm512_sad_epu8(intersect_popcnt, _mm512_setzero_si512()));
            union_sum = _mm512_add_epi64(union_sum, 
                _mm512_sad_epu8(union_popcnt, _mm512_setzero_si512()));
        }
        
        intersection = static_cast<uint32_t>(_mm512_reduce_add_epi64(intersect_sum));
        union_count = static_cast<uint32_t>(_mm512_reduce_add_epi64(union_sum));
    }
    else
#endif
    {
        for (size_t i = 0; i < n; ++i) {
            intersection += __builtin_popcount(a[i] & b[i]);
            union_count += __builtin_popcount(a[i] | b[i]);
        }
    }
    
    return union_count > 0 ? 1.0f - static_cast<float>(intersection) / union_count : 0.0f;
}

} // namespace binary

//==============================================================================
// Vector similarity search utilities
//==============================================================================
namespace search {

template<typename T>
std::vector<std::pair<size_t, T>> knn_l2(const T* query, const T* database, 
                                          size_t n_vectors, size_t dim, size_t k) {
    std::vector<std::pair<T, size_t>> distances;
    distances.reserve(n_vectors);
    
    for (size_t i = 0; i < n_vectors; ++i) {
        T dist = l2_squared_distance(query, &database[i * dim], dim);
        distances.emplace_back(dist, i);
    }
    
    std::partial_sort(distances.begin(), distances.begin() + k, distances.end());
    
    std::vector<std::pair<size_t, T>> result;
    result.reserve(k);
    for (size_t i = 0; i < k; ++i) {
        result.emplace_back(distances[i].second, distances[i].first);
    }
    
    return result;
}

template<typename T>
void batch_cosine_similarity(const T* queries, const T* database, T* results,
                             size_t n_queries, size_t n_vectors, size_t dim) {
    #pragma omp parallel for if(n_queries > 100)
    for (size_t i = 0; i < n_queries; ++i) {
        for (size_t j = 0; j < n_vectors; ++j) {
            results[i * n_vectors + j] = 1.0f - cosine_distance(
                &queries[i * dim], &database[j * dim], dim);
        }
    }
}

} // namespace search

//==============================================================================
// Benchmark utilities
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

void benchmark_distance_functions() {
    std::cout << "=== Enhanced SIMD Distance Function Benchmarks ===" << std::endl;
    
    const size_t n = 1536;
    const int iterations = 10000;
    
    util::aligned_vector<float> a_f32(n), b_f32(n);
    util::aligned_vector<f16_t> a_f16(n), b_f16(n);
    util::aligned_vector<int8_t> a_i8(n), b_i8(n);
    util::aligned_vector<uint16_t> a_bf16(n), b_bf16(n);
    
    std::srand(31);
    for (size_t i = 0; i < n; ++i) {
        float val_a = static_cast<float>(rand()) / RAND_MAX - 0.5f;
        float val_b = static_cast<float>(rand()) / RAND_MAX - 0.5f;
        
        a_f32[i] = val_a;
        b_f32[i] = val_b;
        a_f16[i] = util::f32_to_f16(val_a);
        b_f16[i] = util::f32_to_f16(val_b);
        a_i8[i] = static_cast<int8_t>(val_a * 15);  // Reduced range
        b_i8[i] = static_cast<int8_t>(val_b * 15);  // Reduced range
        
        union { float f; uint32_t i; } ua = { val_a }, ub = { val_b };
        a_bf16[i] = static_cast<uint16_t>(ua.i >> 16);
        b_bf16[i] = static_cast<uint16_t>(ub.i >> 16);
    }
    
    const auto& features = CpuFeatures::detect();
    std::cout << "Available features: AVX2=" << features.avx2 
              << ", AVX512F=" << features.avx512f 
              << ", AVX512FP16=" << features.avx512fp16
              << ", AVX512VNNI=" << features.avx512vnni
              << ", AVX512BF16=" << features.avx512bf16 << std::endl;
    
    // Test I8 accuracy first
    std::cout << "\nTesting I8 accuracy:" << std::endl;
    int32_t scalar_i8 = 0;
    for (size_t i = 0; i < n; ++i) {
        scalar_i8 += static_cast<int32_t>(a_i8[i]) * static_cast<int32_t>(b_i8[i]);
    }
    int32_t simd_i8 = dot_i8(a_i8.data(), b_i8.data(), n);
    std::cout << "Scalar I8: " << scalar_i8 << ", SIMD I8: " << simd_i8;
    if (scalar_i8 == simd_i8) {
        std::cout << " ✅ MATCH!" << std::endl;
    } else {
        std::cout << " ❌ MISMATCH!" << std::endl;
    }
    
    // Benchmark results...
    {
        Timer timer;
        volatile float result = 0;
        for (int i = 0; i < iterations; ++i) {
            result += l2_squared_distance(a_f32.data(), b_f32.data(), n);
        }
        double elapsed = timer.elapsed_ms();
        double ops_per_sec = (2.0 * n * iterations) / (elapsed / 1000.0);
        std::cout << "F32 L2² distance: " << std::fixed << std::setprecision(2)
                  << ops_per_sec / 1e6 << " Mops/s (" << elapsed << " ms)" << std::endl;
    }
}

} // namespace benchmark

//==============================================================================
// Test suite
//==============================================================================
namespace test {

void test_masked_operations() {
    std::cout << "\n=== Testing Masked Operations ===" << std::endl;
    
    std::vector<size_t> test_sizes = {15, 16, 17, 31, 32, 33, 63, 64, 65, 1535, 1536, 1537};
    
    for (size_t n : test_sizes) {
        util::aligned_vector<float> a(n), b(n);
        
        for (size_t i = 0; i < n; ++i) {
            a[i] = static_cast<float>(i + 1);
            b[i] = static_cast<float>(2 * i + 1);
        }
        
        float expected = 0.0f;
        for (size_t i = 0; i < n; ++i) {
            expected += a[i] * b[i];
        }
        
        float result = dot(a.data(), b.data(), n);
        float error = std::abs(result - expected) / expected;
        
        std::cout << "Size " << std::setw(4) << n << ": ";
        if (error < 1e-5f) {
            std::cout << "✅ PASS";
        } else {
            std::cout << "❌ FAIL (error: " << error << ")";
        }
        std::cout << " expected=" << expected << " got=" << result << std::endl;
    }
}

void test_data_types() {
    std::cout << "\n=== Testing Different Data Types ===" << std::endl;
    
    const size_t n = 128;
    
    // Test I8 with reduced range
    {
        util::aligned_vector<int8_t> a_i8(n), b_i8(n);
        
        for (size_t i = 0; i < n; ++i) {
            a_i8[i] = static_cast<int8_t>((i % 30) - 15);  // Range [-15, 14]
            b_i8[i] = static_cast<int8_t>(((n - i) % 30) - 15);  // Range [-15, 14]
        }
        
        int32_t result = dot_i8(a_i8.data(), b_i8.data(), n);
        
        int32_t expected = 0;
        for (size_t i = 0; i < n; ++i) {
            expected += static_cast<int32_t>(a_i8[i]) * static_cast<int32_t>(b_i8[i]);
        }
        
        std::cout << "I8 dot product: ";
        if (result == expected) {
            std::cout << "✅ PASS";
        } else {
            std::cout << "❌ FAIL (expected=" << expected << " got=" << result << ")";
        }
        std::cout << std::endl;
    }
}

void run_all_tests() {
    std::cout << "=== FIXED SIMD Library Test Suite ===" << std::endl;
    
    test_masked_operations();
    test_data_types();
    
    std::cout << "\n=== All tests completed ===" << std::endl;
}

} // namespace test

} // namespace simd