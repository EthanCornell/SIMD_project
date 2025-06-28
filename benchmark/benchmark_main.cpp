#include "../include/simd.hpp"
#include <iostream>
#include <random>
#include <chrono>
#include <cmath>

// Add missing M_PI if not defined
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

//==============================================================================
// Simple AI Inference Example
//==============================================================================

class DenseLayer {
private:
    simd::util::aligned_vector<float> weights_;
    simd::util::aligned_vector<float> bias_;
    size_t input_size_;
    size_t output_size_;

public:
    DenseLayer(size_t input_size, size_t output_size) 
        : input_size_(input_size), output_size_(output_size) {
        
        // Initialize weights and bias
        weights_.resize(output_size * input_size);
        bias_.resize(output_size);
        
        // Simple random initialization
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
        
        for (auto& w : weights_) w = dist(gen);
        for (auto& b : bias_) b = dist(gen);
    }
    
    void forward(const float* input, float* output) const {
        // Matrix-vector multiplication: output = weights * input
        simd::matmul(weights_.data(), input, output, output_size_, 1, input_size_);
        
        // Add bias: output += bias
        simd::saxpy(1.0f, bias_.data(), output, output_size_);
        
        // ReLU activation: output = max(0, output)
        for (size_t i = 0; i < output_size_; ++i) {
            output[i] = std::max(0.0f, output[i]);
        }
    }
    
    size_t get_output_size() const { return output_size_; }
};

void run_ai_example() {
    std::cout << "\n=== AI Inference Example ===" << std::endl;
    
    // Create simple layers: 784 -> 128 -> 10
    DenseLayer layer1(784, 128);
    DenseLayer layer2(128, 10);
    
    // Generate random input
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    std::vector<float> input(784);
    for (auto& x : input) x = dist(gen);
    
    // Benchmark inference
    const int num_inferences = 1000;
    std::vector<float> hidden(128);
    std::vector<float> output(10);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_inferences; ++i) {
        layer1.forward(input.data(), hidden.data());
        layer2.forward(hidden.data(), output.data());
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "Performed " << num_inferences << " inferences in " 
              << duration.count() << " μs" << std::endl;
    std::cout << "Average inference time: " 
              << duration.count() / float(num_inferences) << " μs" << std::endl;
    
    // Display sample output
    std::cout << "Sample output: ";
    for (size_t i = 0; i < 5; ++i) {
        std::cout << output[i] << " ";
    }
    std::cout << "..." << std::endl;
}

//==============================================================================
// Signal Processing Example
//==============================================================================

void generate_test_signal(float* signal, size_t length, float frequency, float sample_rate) {
    for (size_t i = 0; i < length; ++i) {
        float t = i / sample_rate;
        signal[i] = std::sin(2.0f * M_PI * frequency * t);
    }
}

void run_signal_example() {
    std::cout << "\n=== Signal Processing Example ===" << std::endl;
    
    const float sample_rate = 44100.0f;
    const size_t signal_length = 4410; // 0.1 second
    const float test_frequency = 1000.0f;
    
    // Generate test signals
    simd::util::aligned_vector<float> signal1(signal_length);
    simd::util::aligned_vector<float> signal2(signal_length);
    simd::util::aligned_vector<float> output(signal_length);
    
    generate_test_signal(signal1.data(), signal_length, test_frequency, sample_rate);
    generate_test_signal(signal2.data(), signal_length, test_frequency * 1.1f, sample_rate);
    
    // Benchmark signal operations
    auto start = std::chrono::high_resolution_clock::now();
    
    // Cross-correlation using dot product
    float correlation = simd::dot(signal1.data(), signal2.data(), signal_length);
    
    // Apply gain
    simd::saxpy(1.5f - 1.0f, signal1.data(), output.data(), signal_length);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "Processed " << signal_length << " samples in " << duration.count() << " μs" << std::endl;
    std::cout << "Processing rate: " << (signal_length * 1e6f) / duration.count() << " samples/sec" << std::endl;
    std::cout << "Cross-correlation: " << correlation << std::endl;
}

//==============================================================================
// Matrix Operations Example
//==============================================================================

void run_matrix_example() {
    std::cout << "\n=== Matrix Operations Example ===" << std::endl;
    
    const size_t n = 128;
    simd::util::aligned_vector<float> A(n * n);
    simd::util::aligned_vector<float> B(n * n);
    simd::util::aligned_vector<float> C(n * n);
    
    // Initialize with random data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    for (auto& val : A) val = dist(gen);
    for (auto& val : B) val = dist(gen);
    
    // Benchmark matrix multiplication
    auto start = std::chrono::high_resolution_clock::now();
    simd::matmul(A.data(), B.data(), C.data(), n, n, n);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Calculate GFLOPS
    double ops = 2.0 * n * n * n; // multiply-add operations
    double gflops = ops / (duration.count() / 1e6) / 1e9;
    
    std::cout << "Matrix multiplication (" << n << "x" << n << "): " 
              << duration.count() << " μs" << std::endl;
    std::cout << "Performance: " << gflops << " GFLOPS" << std::endl;
    
    // Compute Frobenius norm
    float norm = std::sqrt(simd::dot(C.data(), C.data(), n * n));
    std::cout << "Result Frobenius norm: " << norm << std::endl;
}

//==============================================================================
// Main Benchmark Application
//==============================================================================

int main() {
    std::cout << "=== SIMD Library Benchmark Suite ===" << std::endl;
    
    // Display system information
    const auto& features = simd::CpuFeatures::detect();
    std::cout << "\nCPU Features:" << std::endl;
    std::cout << "  AVX2: " << (features.avx2 ? "Yes" : "No") << std::endl;
    std::cout << "  AVX-512F: " << (features.avx512f ? "Yes" : "No") << std::endl;
    std::cout << "  AVX-512BW: " << (features.avx512bw ? "Yes" : "No") << std::endl;
    std::cout << "  AVX-512VL: " << (features.avx512vl ? "Yes" : "No") << std::endl;
    std::cout << "  Hardware Threads: " << std::thread::hardware_concurrency() << std::endl;
    
    try {
        // Run example applications
        run_ai_example();
        run_signal_example();
        run_matrix_example();
        
        // Run built-in benchmarks
        std::cout << "\n=== Built-in Benchmarks ===" << std::endl;
        simd::benchmark::run_benchmarks();
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "\n=== Benchmark Suite Complete ===" << std::endl;
    
    return 0;
}