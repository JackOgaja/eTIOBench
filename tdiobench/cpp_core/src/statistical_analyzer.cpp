#include "statistical_analyzer.hpp"
#include "simd_utils.hpp"
#include "threading_utils.hpp"

#include <algorithm>
#include <numeric>
#include <cmath>
#include <execution>
#include <thread>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace etiobench {
namespace core {

StatisticalAnalyzer::StatisticalAnalyzer(const Config& config)
    : config_(config) {
    initialize_threading();
}

StatisticalAnalyzer::StatisticalAnalyzer()
    : config_() {
    initialize_threading();
}

void StatisticalAnalyzer::initialize_threading() {
    if (config_.num_threads == 0) {
        optimal_threads_ = std::thread::hardware_concurrency();
    } else {
        optimal_threads_ = config_.num_threads;
    }
    
#ifdef _OPENMP
    omp_set_num_threads(static_cast<int>(optimal_threads_));
#endif
}

StatisticalAnalyzer::StatisticalResult 
StatisticalAnalyzer::analyze_test_data(const BenchmarkMetrics& metrics) {
    
    // Analyze throughput data (primary metric)
    if (!metrics.throughput_mbps.empty()) {
        auto result = calculate_basic_statistics(metrics.throughput_mbps);
        
        // Calculate confidence interval
        auto ci = calculate_confidence_interval(metrics.throughput_mbps, config_.confidence_level);
        result.confidence_interval_lower = ci.first;
        result.confidence_interval_upper = ci.second;
        
        // Detect outliers if enabled
        if (config_.enable_outlier_detection) {
            result.outlier_indices = detect_outliers(metrics.throughput_mbps, config_.outlier_threshold);
        }
        
        return result;
    }
    
    // Fallback to IOPS if throughput not available
    if (!metrics.iops.empty()) {
        return calculate_basic_statistics(metrics.iops);
    }
    
    // Return empty result if no data
    return StatisticalResult{};
}

StatisticalAnalyzer::StatisticalResult 
StatisticalAnalyzer::calculate_basic_statistics(const std::vector<double>& data) {
    
    if (data.empty()) {
        return StatisticalResult{};
    }
    
    StatisticalResult result;
    result.sample_count = data.size();
    
    // SIMD-optimized mean calculation
    if (config_.enable_simd) {
        result.mean = simd_mean(data.data(), data.size());
    } else {
        result.mean = std::accumulate(data.begin(), data.end(), 0.0) / data.size();
    }
    
    // SIMD-optimized variance calculation
    if (config_.enable_simd) {
        result.variance = simd_variance(data.data(), data.size(), result.mean);
    } else {
        double sum_sq_diff = 0.0;
        for (double value : data) {
            double diff = value - result.mean;
            sum_sq_diff += diff * diff;
        }
        result.variance = sum_sq_diff / (data.size() - 1);
    }
    
    result.std_deviation = std::sqrt(result.variance);
    
    // Min/Max using standard algorithms
    auto [min_iter, max_iter] = std::minmax_element(data.begin(), data.end());
    result.min_value = *min_iter;
    result.max_value = *max_iter;
    
    // Calculate percentiles
    result.percentiles = calculate_percentiles(data, config_.percentiles);
    if (!result.percentiles.empty()) {
        result.median = result.percentiles[0];  // First percentile is 50th
    }
    
    // Higher-order moments
    result.skewness = calculate_skewness(data, result.mean, result.std_deviation);
    result.kurtosis = calculate_kurtosis(data, result.mean, result.std_deviation);
    
    return result;
}

double StatisticalAnalyzer::simd_mean(const double* data, size_t size) const {
    if (!config_.enable_simd || size < 4) {
        return std::accumulate(data, data + size, 0.0) / size;
    }
    
    return simd::simd_mean(data, size);
}

double StatisticalAnalyzer::simd_variance(const double* data, size_t size, double mean) const {
    if (!config_.enable_simd || size < 4) {
        double sum_sq_diff = 0.0;
        for (size_t i = 0; i < size; ++i) {
            double diff = data[i] - mean;
            sum_sq_diff += diff * diff;
        }
        return sum_sq_diff / (size - 1);
    }
    
    return simd::simd_variance(data, size, mean);
}

std::vector<size_t> StatisticalAnalyzer::detect_outliers(
    const std::vector<double>& data, double threshold) {
    
    std::vector<size_t> outlier_indices;
    
    if (data.size() < 3) {
        return outlier_indices;
    }
    
    // Calculate mean and standard deviation
    double mean = simd_mean(data.data(), data.size());
    double std_dev = std::sqrt(simd_variance(data.data(), data.size(), mean));
    
    double lower_bound = mean - threshold * std_dev;
    double upper_bound = mean + threshold * std_dev;
    
    // Parallel outlier detection
#ifdef _OPENMP
    std::vector<bool> is_outlier(data.size(), false);
    
    #pragma omp parallel for
    for (size_t i = 0; i < data.size(); ++i) {
        if (data[i] < lower_bound || data[i] > upper_bound) {
            is_outlier[i] = true;
        }
    }
    
    // Collect outlier indices
    for (size_t i = 0; i < data.size(); ++i) {
        if (is_outlier[i]) {
            outlier_indices.push_back(i);
        }
    }
#else
    for (size_t i = 0; i < data.size(); ++i) {
        if (data[i] < lower_bound || data[i] > upper_bound) {
            outlier_indices.push_back(i);
        }
    }
#endif
    
    return outlier_indices;
}

std::vector<double> StatisticalAnalyzer::calculate_percentiles(
    const std::vector<double>& data, const std::vector<double>& percentiles) {
    
    if (data.empty()) {
        return std::vector<double>(percentiles.size(), 0.0);
    }
    
    // Prepare sorted copy using work buffer
    ensure_work_buffer_size(data.size());
    prepare_sorted_copy(data, work_buffer_);
    
    std::vector<double> results;
    results.reserve(percentiles.size());
    
    for (double p : percentiles) {
        double index = (p / 100.0) * (work_buffer_.size() - 1);
        size_t lower_idx = static_cast<size_t>(std::floor(index));
        size_t upper_idx = static_cast<size_t>(std::ceil(index));
        
        if (lower_idx == upper_idx || lower_idx >= work_buffer_.size()) {
            results.push_back(work_buffer_[std::min(lower_idx, work_buffer_.size() - 1)]);
        } else {
            double weight = index - lower_idx;
            double interpolated = work_buffer_[lower_idx] * (1.0 - weight) + 
                                work_buffer_[upper_idx] * weight;
            results.push_back(interpolated);
        }
    }
    
    return results;
}

std::pair<double, double> StatisticalAnalyzer::calculate_confidence_interval(
    const std::vector<double>& data, double confidence_level) {
    
    if (data.size() < 2) {
        return {0.0, 0.0};
    }
    
    double mean = simd_mean(data.data(), data.size());
    double std_dev = std::sqrt(simd_variance(data.data(), data.size(), mean));
    double standard_error = std_dev / std::sqrt(data.size());
    
    double t_critical = get_t_critical_value(data.size(), confidence_level);
    double margin_error = t_critical * standard_error;
    
    return {mean - margin_error, mean + margin_error};
}

std::unordered_map<std::string, StatisticalAnalyzer::StatisticalResult> 
StatisticalAnalyzer::compare_tiers(
    const std::unordered_map<std::string, BenchmarkMetrics>& tier_data) {
    
    std::unordered_map<std::string, StatisticalResult> results;
    
    // Parallel analysis of multiple tiers
    std::vector<std::string> tier_names;
    std::vector<BenchmarkMetrics> tier_metrics;
    
    tier_names.reserve(tier_data.size());
    tier_metrics.reserve(tier_data.size());
    
    for (const auto& [name, metrics] : tier_data) {
        tier_names.push_back(name);
        tier_metrics.push_back(metrics);
    }
    
    std::vector<StatisticalResult> tier_results(tier_names.size());
    
#ifdef _OPENMP
    #pragma omp parallel for
    for (size_t i = 0; i < tier_names.size(); ++i) {
        tier_results[i] = analyze_test_data(tier_metrics[i]);
    }
#else
    for (size_t i = 0; i < tier_names.size(); ++i) {
        tier_results[i] = analyze_test_data(tier_metrics[i]);
    }
#endif
    
    // Combine results
    for (size_t i = 0; i < tier_names.size(); ++i) {
        results[tier_names[i]] = std::move(tier_results[i]);
    }
    
    return results;
}

void StatisticalAnalyzer::set_config(const Config& new_config) {
    config_ = new_config;
    initialize_threading();
}

double StatisticalAnalyzer::calculate_skewness(
    const std::vector<double>& data, double mean, double std_dev) const {
    
    if (data.size() < 3 || std_dev == 0.0) {
        return 0.0;
    }
    
    double sum_cubed_z = 0.0;
    for (double value : data) {
        double z = (value - mean) / std_dev;
        sum_cubed_z += z * z * z;
    }
    
    return sum_cubed_z / data.size();
}

double StatisticalAnalyzer::calculate_kurtosis(
    const std::vector<double>& data, double mean, double std_dev) const {
    
    if (data.size() < 4 || std_dev == 0.0) {
        return 0.0;
    }
    
    double sum_fourth_z = 0.0;
    for (double value : data) {
        double z = (value - mean) / std_dev;
        double z_squared = z * z;
        sum_fourth_z += z_squared * z_squared;
    }
    
    return (sum_fourth_z / data.size()) - 3.0;  // Excess kurtosis
}

double StatisticalAnalyzer::get_t_critical_value(size_t sample_size, double confidence_level) const {
    // Simplified t-critical values for common confidence levels
    // For production, would use proper t-distribution lookup table
    
    if (sample_size >= 30) {
        // Use normal approximation for large samples
        if (confidence_level >= 0.99) return 2.576;
        if (confidence_level >= 0.95) return 1.96;
        if (confidence_level >= 0.90) return 1.645;
    }
    
    // Conservative estimate for small samples
    if (confidence_level >= 0.99) return 3.0;
    if (confidence_level >= 0.95) return 2.5;
    return 2.0;
}

void StatisticalAnalyzer::ensure_work_buffer_size(size_t required_size) const {
    if (work_buffer_.size() < required_size) {
        work_buffer_.resize(required_size);
    }
}

void StatisticalAnalyzer::prepare_sorted_copy(
    const std::vector<double>& source, std::vector<double>& dest) const {
    
    dest.assign(source.begin(), source.end());
    
    // Use standard sort for all datasets
    std::sort(dest.begin(), dest.end());
}

} // namespace core
} // namespace etiobench
