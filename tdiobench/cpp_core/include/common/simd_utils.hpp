#pragma once

#include <cstddef>

namespace etiobench {
namespace core {
namespace simd {

/**
 * @brief SIMD-optimized mathematical operations
 * 
 * This module provides vectorized implementations of common mathematical
 * operations using AVX2 instructions when available.
 */

/**
 * @brief Check if SIMD instructions are available
 * @return True if AVX2 is supported
 */
bool is_avx2_supported();

/**
 * @brief SIMD-optimized mean calculation
 * @param data Pointer to data array (must be aligned to 32 bytes)
 * @param size Number of elements
 * @return Mean value
 */
double calculate_mean(const double* data, size_t size);

/**
 * @brief SIMD-optimized variance calculation
 * @param data Pointer to data array
 * @param size Number of elements
 * @param mean Pre-calculated mean value
 * @return Variance value
 */
double calculate_variance(const double* data, size_t size, double mean);

/**
 * @brief SIMD-optimized sum calculation
 * @param data Pointer to data array
 * @param size Number of elements
 * @return Sum value
 */
double calculate_sum(const double* data, size_t size);

/**
 * @brief SIMD-optimized dot product
 * @param a First vector
 * @param b Second vector
 * @param size Number of elements
 * @return Dot product
 */
double dot_product(const double* a, const double* b, size_t size);

/**
 * @brief SIMD-optimized element-wise addition
 * @param a First vector
 * @param b Second vector
 * @param result Output vector
 * @param size Number of elements
 */
void vector_add(const double* a, const double* b, double* result, size_t size);

/**
 * @brief SIMD-optimized element-wise multiplication
 * @param a First vector
 * @param b Second vector
 * @param result Output vector
 * @param size Number of elements
 */
void vector_multiply(const double* a, const double* b, double* result, size_t size);

/**
 * @brief SIMD-optimized scalar multiplication
 * @param data Input vector
 * @param scalar Scalar value
 * @param result Output vector
 * @param size Number of elements
 */
void scalar_multiply(const double* data, double scalar, double* result, size_t size);

} // namespace simd
} // namespace core
} // namespace etiobench
