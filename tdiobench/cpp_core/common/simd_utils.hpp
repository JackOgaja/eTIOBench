#pragma once

#include <vector>
#include <cstddef>

namespace etiobench {
namespace simd {

/**
 * SIMD-optimized utilities for mathematical operations.
 * Provides AVX2 acceleration when available with automatic fallback.
 */

// Check SIMD availability at runtime
bool is_avx2_available();

// SIMD-accelerated mathematical operations
double simd_mean(const double* data, size_t size);
double simd_variance(const double* data, size_t size, double mean);
double simd_sum(const double* data, size_t size);
double simd_dot_product(const double* a, const double* b, size_t size);

// Vector operations
void simd_add(const double* a, const double* b, double* result, size_t size);
void simd_multiply(const double* a, const double* b, double* result, size_t size);
void simd_scale(const double* data, double scale, double* result, size_t size);

// Convenience wrappers for std::vector
inline double simd_mean(const std::vector<double>& data) {
    return simd_mean(data.data(), data.size());
}

inline double simd_variance(const std::vector<double>& data, double mean) {
    return simd_variance(data.data(), data.size(), mean);
}

inline double simd_sum(const std::vector<double>& data) {
    return simd_sum(data.data(), data.size());
}

inline double simd_dot_product(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vector sizes must match for dot product");
    }
    return simd_dot_product(a.data(), b.data(), a.size());
}

} // namespace simd
} // namespace etiobench
