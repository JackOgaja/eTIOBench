#include "simd_utils.hpp"
#include <algorithm>
#include <numeric>
#include <stdexcept>

#ifdef __AVX2__
#include <immintrin.h>
#endif

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

namespace etiobench {
namespace simd {

bool is_avx2_available() {
#ifdef __AVX2__
    return true;
#elif defined(__ARM_NEON)
    return true; // NEON is available
#else
    return false;
#endif
}

double simd_mean(const double* data, size_t size) {
    if (size == 0) return 0.0;
    
#ifdef __AVX2__
    if (size >= 4) {
        const size_t simd_size = size - (size % 4);
        __m256d sum_vec = _mm256_setzero_pd();
        
        for (size_t i = 0; i < simd_size; i += 4) {
            __m256d data_vec = _mm256_loadu_pd(&data[i]);
            sum_vec = _mm256_add_pd(sum_vec, data_vec);
        }
        
        // Extract sum from vector
        double sum_array[4];
        _mm256_storeu_pd(sum_array, sum_vec);
        double sum = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3];
        
        // Add remaining elements
        for (size_t i = simd_size; i < size; ++i) {
            sum += data[i];
        }
        
        return sum / static_cast<double>(size);
    }
#endif
    
    // Fallback implementation
    return std::accumulate(data, data + size, 0.0) / static_cast<double>(size);
}

double simd_variance(const double* data, size_t size, double mean) {
    if (size <= 1) return 0.0;
    
#ifdef __AVX2__
    if (size >= 4) {
        const size_t simd_size = size - (size % 4);
        __m256d mean_vec = _mm256_set1_pd(mean);
        __m256d sum_sq_vec = _mm256_setzero_pd();
        
        for (size_t i = 0; i < simd_size; i += 4) {
            __m256d data_vec = _mm256_loadu_pd(&data[i]);
            __m256d diff_vec = _mm256_sub_pd(data_vec, mean_vec);
            __m256d sq_vec = _mm256_mul_pd(diff_vec, diff_vec);
            sum_sq_vec = _mm256_add_pd(sum_sq_vec, sq_vec);
        }
        
        // Extract sum from vector
        double sum_sq_array[4];
        _mm256_storeu_pd(sum_sq_array, sum_sq_vec);
        double sum_sq = sum_sq_array[0] + sum_sq_array[1] + sum_sq_array[2] + sum_sq_array[3];
        
        // Add remaining elements
        for (size_t i = simd_size; i < size; ++i) {
            double diff = data[i] - mean;
            sum_sq += diff * diff;
        }
        
        return sum_sq / static_cast<double>(size - 1);
    }
#endif
    
    // Fallback implementation
    double sum_sq = 0.0;
    for (size_t i = 0; i < size; ++i) {
        double diff = data[i] - mean;
        sum_sq += diff * diff;
    }
    return sum_sq / static_cast<double>(size - 1);
}

double simd_sum(const double* data, size_t size) {
    if (size == 0) return 0.0;
    
#ifdef __AVX2__
    if (size >= 4) {
        const size_t simd_size = size - (size % 4);
        __m256d sum_vec = _mm256_setzero_pd();
        
        for (size_t i = 0; i < simd_size; i += 4) {
            __m256d data_vec = _mm256_loadu_pd(&data[i]);
            sum_vec = _mm256_add_pd(sum_vec, data_vec);
        }
        
        // Extract sum from vector
        double sum_array[4];
        _mm256_storeu_pd(sum_array, sum_vec);
        double sum = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3];
        
        // Add remaining elements
        for (size_t i = simd_size; i < size; ++i) {
            sum += data[i];
        }
        
        return sum;
    }
#endif
    
    // Fallback implementation
    return std::accumulate(data, data + size, 0.0);
}

double simd_dot_product(const double* a, const double* b, size_t size) {
    if (size == 0) return 0.0;
    
#ifdef __AVX2__
    if (size >= 4) {
        const size_t simd_size = size - (size % 4);
        __m256d sum_vec = _mm256_setzero_pd();
        
        for (size_t i = 0; i < simd_size; i += 4) {
            __m256d a_vec = _mm256_loadu_pd(&a[i]);
            __m256d b_vec = _mm256_loadu_pd(&b[i]);
            __m256d prod_vec = _mm256_mul_pd(a_vec, b_vec);
            sum_vec = _mm256_add_pd(sum_vec, prod_vec);
        }
        
        // Extract sum from vector
        double sum_array[4];
        _mm256_storeu_pd(sum_array, sum_vec);
        double sum = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3];
        
        // Add remaining elements
        for (size_t i = simd_size; i < size; ++i) {
            sum += a[i] * b[i];
        }
        
        return sum;
    }
#endif
    
    // Fallback implementation
    double sum = 0.0;
    for (size_t i = 0; i < size; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

void simd_add(const double* a, const double* b, double* result, size_t size) {
#ifdef __AVX2__
    if (size >= 4) {
        const size_t simd_size = size - (size % 4);
        
        for (size_t i = 0; i < simd_size; i += 4) {
            __m256d a_vec = _mm256_loadu_pd(&a[i]);
            __m256d b_vec = _mm256_loadu_pd(&b[i]);
            __m256d result_vec = _mm256_add_pd(a_vec, b_vec);
            _mm256_storeu_pd(&result[i], result_vec);
        }
        
        // Handle remaining elements
        for (size_t i = simd_size; i < size; ++i) {
            result[i] = a[i] + b[i];
        }
        return;
    }
#endif
    
    // Fallback implementation
    for (size_t i = 0; i < size; ++i) {
        result[i] = a[i] + b[i];
    }
}

void simd_multiply(const double* a, const double* b, double* result, size_t size) {
#ifdef __AVX2__
    if (size >= 4) {
        const size_t simd_size = size - (size % 4);
        
        for (size_t i = 0; i < simd_size; i += 4) {
            __m256d a_vec = _mm256_loadu_pd(&a[i]);
            __m256d b_vec = _mm256_loadu_pd(&b[i]);
            __m256d result_vec = _mm256_mul_pd(a_vec, b_vec);
            _mm256_storeu_pd(&result[i], result_vec);
        }
        
        // Handle remaining elements
        for (size_t i = simd_size; i < size; ++i) {
            result[i] = a[i] * b[i];
        }
        return;
    }
#endif
    
    // Fallback implementation
    for (size_t i = 0; i < size; ++i) {
        result[i] = a[i] * b[i];
    }
}

void simd_scale(const double* data, double scale, double* result, size_t size) {
#ifdef __AVX2__
    if (size >= 4) {
        const size_t simd_size = size - (size % 4);
        __m256d scale_vec = _mm256_set1_pd(scale);
        
        for (size_t i = 0; i < simd_size; i += 4) {
            __m256d data_vec = _mm256_loadu_pd(&data[i]);
            __m256d result_vec = _mm256_mul_pd(data_vec, scale_vec);
            _mm256_storeu_pd(&result[i], result_vec);
        }
        
        // Handle remaining elements
        for (size_t i = simd_size; i < size; ++i) {
            result[i] = data[i] * scale;
        }
        return;
    }
#endif
    
    // Fallback implementation
    for (size_t i = 0; i < size; ++i) {
        result[i] = data[i] * scale;
    }
}

} // namespace simd
} // namespace etiobench
