#pragma once

#include <vector>
#include <unordered_map>
#include <string>
#include <memory>
#include <optional>
#include <cstddef>

namespace etiobench {
namespace core {

/**
 * @brief High-performance statistical analysis for benchmark data
 * 
 * This class provides SIMD-optimized statistical computations with API
 * compatibility to the Python StatisticalAnalyzer class.
 * 
 * Expected performance improvement: 20-50x over Python implementation
 */
class StatisticalAnalyzer {
public:
    /**
     * @brief Statistical result structure matching Python API
     */
    struct StatisticalResult {
        double mean = 0.0;
        double std_deviation = 0.0;
        double min_value = 0.0;
        double max_value = 0.0;
        double median = 0.0;
        std::vector<double> percentiles;
        std::vector<size_t> outlier_indices;
        double confidence_interval_lower = 0.0;
        double confidence_interval_upper = 0.0;
        size_t sample_count = 0;
        
        // Additional statistics
        double variance = 0.0;
        double skewness = 0.0;
        double kurtosis = 0.0;
    };
    
    /**
     * @brief Benchmark metrics structure matching Python BenchmarkData
     */
    struct BenchmarkMetrics {
        std::vector<double> throughput_mbps;
        std::vector<double> iops;
        std::vector<double> latency_ms;
        std::vector<double> timestamps;
        std::string tier_name;
        std::string test_name;
    };
    
    /**
     * @brief Configuration matching Python config structure
     */
    struct Config {
        double confidence_level = 0.95;
        std::vector<double> percentiles = {50.0, 95.0, 99.0, 99.9};
        bool enable_outlier_detection = true;
        double outlier_threshold = 2.0;
        bool enable_simd = true;
        size_t num_threads = 0;  // 0 = auto-detect
    };

private:
    Config config_;
    
    // Performance optimization members
    mutable std::vector<double> work_buffer_;  // Reusable buffer for computations
    size_t optimal_threads_;

public:
    /**
     * @brief Constructor matching Python StatisticalAnalyzer.__init__
     * @param config Configuration object
     */
    explicit StatisticalAnalyzer(const Config& config);
    StatisticalAnalyzer(); // Default constructor
    
    /**
     * @brief Destructor
     */
    ~StatisticalAnalyzer() = default;
    
    // Copy and move semantics
    StatisticalAnalyzer(const StatisticalAnalyzer&) = delete;
    StatisticalAnalyzer& operator=(const StatisticalAnalyzer&) = delete;
    StatisticalAnalyzer(StatisticalAnalyzer&&) = default;
    StatisticalAnalyzer& operator=(StatisticalAnalyzer&&) = default;
    
    /**
     * @brief Analyze test data - main entry point matching Python API
     * @param metrics Benchmark metrics to analyze
     * @return Statistical analysis results
     */
    StatisticalResult analyze_test_data(const BenchmarkMetrics& metrics);
    
    /**
     * @brief Calculate basic statistics for a dataset
     * @param data Input data vector
     * @return Statistical result
     */
    StatisticalResult calculate_basic_statistics(const std::vector<double>& data);
    
    /**
     * @brief Detect outliers using Z-score method
     * @param data Input data vector
     * @param threshold Z-score threshold (default: 2.0)
     * @return Indices of outlier points
     */
    std::vector<size_t> detect_outliers(const std::vector<double>& data, 
                                       double threshold = 2.0);
    
    /**
     * @brief Calculate percentiles for given data
     * @param data Input data vector
     * @param percentiles Percentile values to calculate
     * @return Calculated percentile values
     */
    std::vector<double> calculate_percentiles(const std::vector<double>& data,
                                            const std::vector<double>& percentiles);
    
    /**
     * @brief Calculate confidence interval
     * @param data Input data vector
     * @param confidence_level Confidence level (0.0-1.0)
     * @return Lower and upper bounds of confidence interval
     */
    std::pair<double, double> calculate_confidence_interval(
        const std::vector<double>& data, 
        double confidence_level);
    
    /**
     * @brief Compare multiple tiers - matching Python compare_tiers
     * @param tier_data Map of tier name to benchmark metrics
     * @return Map of tier name to statistical results
     */
    std::unordered_map<std::string, StatisticalResult> compare_tiers(
        const std::unordered_map<std::string, BenchmarkMetrics>& tier_data);
    
    /**
     * @brief Get current configuration
     * @return Current configuration
     */
    const Config& get_config() const { return config_; }
    
    /**
     * @brief Update configuration
     * @param new_config New configuration
     */
    void set_config(const Config& new_config);

private:
    // SIMD-optimized computational kernels
    double simd_mean(const double* data, size_t size) const;
    double simd_variance(const double* data, size_t size, double mean) const;
    double simd_sum(const double* data, size_t size) const;
    void simd_sort_partial(std::vector<double>& data, size_t k) const;
    
    // Statistical computation helpers
    double calculate_skewness(const std::vector<double>& data, double mean, double std_dev) const;
    double calculate_kurtosis(const std::vector<double>& data, double mean, double std_dev) const;
    double get_t_critical_value(size_t sample_size, double confidence_level) const;
    
    // Memory management helpers
    void ensure_work_buffer_size(size_t required_size) const;
    void prepare_sorted_copy(const std::vector<double>& source, 
                           std::vector<double>& dest) const;
    
    // Threading utilities
    void initialize_threading();
    template<typename Func>
    void parallel_for(size_t start, size_t end, Func&& func) const;
};

} // namespace core
} // namespace etiobench
