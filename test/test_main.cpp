#include "../include/simd.hpp"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

void test_cpu_features() {
    std::cout << "Testing CPU feature detection..." << std::endl;
    const auto& features = simd::CpuFeatures::detect();
    
    // Just verify it doesn't crash
    std::cout << "  AVX2: " << (features.avx2 ? "Yes" : "No") << std::endl;
    std::cout << "  AVX-512F: " << (features.avx512f ? "Yes" : "No") << std::endl;
    std::cout << "PASS: CPU feature detection works" << std::endl;
}

void test_dot_product() {
    std::cout << "Testing DOT product..." << std::endl;
    
    const size_t n = 100;
    std::vector<float> a(n), b(n);
    
    // Simple test case
    for (size_t i = 0; i < n; ++i) {
        a[i] = static_cast<float>(i + 1);
        b[i] = 2.0f;
    }
    
    float scalar_result = simd::scalar::dot(a.data(), b.data(), n);
    float simd_result = simd::dot(a.data(), b.data(), n);
    
    // Expected: sum(i+1) * 2 for i from 0 to 99 = sum(1 to 100) * 2 = 5050 * 2 = 10100
    float expected = 10100.0f;
    
    assert(std::abs(scalar_result - expected) < 1e-5f);
    assert(std::abs(simd_result - expected) < 1e-5f);
    assert(std::abs(scalar_result - simd_result) < 1e-5f);
    
    std::cout << "  Expected: " << expected << std::endl;
    std::cout << "  Scalar: " << scalar_result << std::endl;
    std::cout << "  SIMD: " << simd_result << std::endl;
    std::cout << "PASS: DOT product test" << std::endl;
}

void test_saxpy() {
    std::cout << "Testing SAXPY..." << std::endl;
    
    const size_t n = 50;
    float alpha = 3.0f;
    std::vector<float> x(n), y_scalar(n), y_simd(n);
    
    // Initialize test data
    for (size_t i = 0; i < n; ++i) {
        x[i] = static_cast<float>(i + 1);
        y_scalar[i] = y_simd[i] = static_cast<float>(i * 2);
    }
    
    // Test both implementations
    simd::scalar::saxpy(alpha, x.data(), y_scalar.data(), n);
    simd::saxpy(alpha, x.data(), y_simd.data(), n);
    
    // Verify results match
    for (size_t i = 0; i < n; ++i) {
        assert(std::abs(y_scalar[i] - y_simd[i]) < 1e-5f);
        
        // Verify calculation: y[i] = alpha * x[i] + y_original[i]
        float expected = alpha * (i + 1) + (i * 2);
        assert(std::abs(y_simd[i] - expected) < 1e-5f);
    }
    
    std::cout << "PASS: SAXPY test" << std::endl;
}

void test_matrix_multiplication() {
    std::cout << "Testing Matrix Multiplication..." << std::endl;
    
    // Small 2x2 test
    const size_t m = 2, n = 2, k = 2;
    std::vector<float> a = {1, 2, 3, 4}; // [[1,2], [3,4]]
    std::vector<float> b = {5, 6, 7, 8}; // [[5,6], [7,8]]
    std::vector<float> c_scalar(4, 0), c_simd(4, 0);
    
    simd::scalar::gemm(a.data(), b.data(), c_scalar.data(), m, n, k);
    simd::matmul(a.data(), b.data(), c_simd.data(), m, n, k);
    
    // Expected result: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19,22], [43,50]]
    std::vector<float> expected = {19, 22, 43, 50};
    
    for (size_t i = 0; i < 4; ++i) {
        assert(std::abs(c_scalar[i] - expected[i]) < 1e-5f);
        assert(std::abs(c_simd[i] - expected[i]) < 1e-5f);
        assert(std::abs(c_scalar[i] - c_simd[i]) < 1e-5f);
    }
    
    std::cout << "PASS: Matrix multiplication test" << std::endl;
}

void test_memory_utilities() {
    std::cout << "Testing memory utilities..." << std::endl;
    
    // Test aligned vector
    simd::util::aligned_vector<float> vec(100);
    assert(vec.size() == 100);
    
    // Check alignment
    uintptr_t addr = reinterpret_cast<uintptr_t>(vec.data());
    assert(addr % 64 == 0); // Should be 64-byte aligned
    
    // Test padded size
    assert(simd::util::padded_size<float>(15, 16) == 16);
    assert(simd::util::padded_size<float>(16, 16) == 16);
    assert(simd::util::padded_size<float>(17, 16) == 20);
    
    std::cout << "PASS: Memory utilities test" << std::endl;
}

void test_edge_cases() {
    std::cout << "Testing edge cases..." << std::endl;
    
    // Zero-length arrays
    std::vector<float> empty;
    float result = simd::dot(empty.data(), empty.data(), 0);
    assert(result == 0.0f);
    
    // Single element
    std::vector<float> single_a = {3.0f};
    std::vector<float> single_b = {4.0f};
    result = simd::dot(single_a.data(), single_b.data(), 1);
    assert(std::abs(result - 12.0f) < 1e-5f);
    
    // Odd sizes (test SIMD tail handling)
    const size_t odd_size = 37;
    std::vector<float> a(odd_size), b(odd_size);
    for (size_t i = 0; i < odd_size; ++i) {
        a[i] = b[i] = 1.0f;
    }
    
    float scalar_result = simd::scalar::dot(a.data(), b.data(), odd_size);
    float simd_result = simd::dot(a.data(), b.data(), odd_size);
    
    assert(std::abs(scalar_result - odd_size) < 1e-5f);
    assert(std::abs(simd_result - odd_size) < 1e-5f);
    
    std::cout << "PASS: Edge cases test" << std::endl;
}

int main() {
    std::cout << "=== SIMD Library Unit Tests ===" << std::endl;
    
    try {
        test_cpu_features();
        test_dot_product();
        test_saxpy();
        test_matrix_multiplication();
        test_memory_utilities();
        test_edge_cases();
        
        std::cout << "\n=== ALL TESTS PASSED ===" << std::endl;
        std::cout << "The SIMD library is working correctly!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}