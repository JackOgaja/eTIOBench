#pragma once

#include <vector>
#include <unordered_map>
#include <string>
#include <memory>
#include <functional>
#include <optional>

namespace etiobench {
namespace core {

/**
 * @brief High-performance data processing module
 * 
 * Provides zero-copy data transformations and SIMD-optimized aggregations
 * with API compatibility to Python data_processor module.
 * 
 * Expected performance improvement: 15-30x over Python implementation
 */
class DataProcessor {
public:
    /**
     * @brief Time series data structure matching Python API
     */
    struct TimeSeriesData {
        std::vector<double> timestamps;
        std::unordered_map<std::string, std::vector<double>> metrics;
        double interval = 1.0;
        std::string tier;
        std::string test_id;
        
        // Convenience methods
        bool empty() const { return timestamps.empty(); }
        size_t size() const { return timestamps.size(); }
        void clear();
        void reserve(size_t capacity);
    };
    
    /**
     * @brief Aggregation result structure
     */
    struct AggregationResult {
        std::vector<double> aggregated_timestamps;
        std::unordered_map<std::string, std::vector<double>> aggregated_metrics;
        std::string aggregation_function;
        double new_interval;
        size_t original_points;
        size_t aggregated_points;
    };
    
    /**
     * @brief Transformation types matching Python enum
     */
    enum class TransformationType {
        LOG,
        SQRT,
        DIFFERENCE,
        PERCENT_CHANGE,
        Z_SCORE,
        MIN_MAX,
        STANDARDIZE
    };
    
    /**
     * @brief Aggregation functions matching Python enum
     */
    enum class AggregationFunction {
        MEAN,
        SUM,
        MIN,
        MAX,
        MEDIAN,
        STD_DEV,
        FIRST,
        LAST,
        COUNT
    };
    
    /**
     * @brief Configuration structure
     */
    struct Config {
        size_t max_chunk_size = 10000;
        size_t num_threads = 0;  // 0 = auto-detect
        bool enable_simd = true;
        bool enable_parallel_processing = true;
        double memory_limit_gb = 2.0;  // Memory usage limit
    };

private:
    Config config_;
    
    // Memory management
    mutable std::vector<double> work_buffer_1_;
    mutable std::vector<double> work_buffer_2_;
    mutable std::vector<size_t> index_buffer_;

public:
    /**
     * @brief Constructor matching Python DataProcessor.__init__
     * @param config Configuration object
     */
    explicit DataProcessor(const Config& config);
    DataProcessor(); // Default constructor
    
    /**
     * @brief Destructor
     */
    ~DataProcessor() = default;
    
    // Copy and move semantics
    DataProcessor(const DataProcessor&) = delete;
    DataProcessor& operator=(const DataProcessor&) = delete;
    DataProcessor(DataProcessor&&) = default;
    DataProcessor& operator=(DataProcessor&&) = default;
    
    /**
     * @brief High-performance time series aggregation
     * @param data Input time series data
     * @param target_interval Target aggregation interval in seconds
     * @param func Aggregation function to use
     * @return Aggregated time series data
     */
    AggregationResult aggregate_time_series(
        const TimeSeriesData& data,
        double target_interval,
        AggregationFunction func = AggregationFunction::MEAN
    );
    
    /**
     * @brief Zero-copy data transformation
     * @param data Time series data (modified in-place for efficiency)
     * @param transformation Type of transformation
     * @return Reference to transformed data
     */
    TimeSeriesData& transform_data_inplace(
        TimeSeriesData& data,
        TransformationType transformation
    );
    
    /**
     * @brief Data transformation with copy
     * @param data Input time series data
     * @param transformation Type of transformation
     * @return New transformed time series data
     */
    TimeSeriesData transform_data(
        const TimeSeriesData& data,
        TransformationType transformation
    );
    
    /**
     * @brief Normalize specific metric using Z-score or Min-Max
     * @param data Time series data (modified in-place)
     * @param metric_name Name of metric to normalize
     * @param normalization Type of normalization
     */
    void normalize_metric(
        TimeSeriesData& data,
        const std::string& metric_name,
        TransformationType normalization = TransformationType::Z_SCORE
    );
    
    /**
     * @brief Convert to matrix format for ML processing
     * @param data Input time series data
     * @param metric_order Order of metrics in matrix columns
     * @return Matrix representation (rows = time points, cols = metrics)
     */
    std::vector<std::vector<double>> to_matrix(
        const TimeSeriesData& data,
        const std::vector<std::string>& metric_order = {}
    );
    
    /**
     * @brief Create summary statistics for time series
     * @param data Input time series data
     * @return Summary statistics per metric
     */
    std::unordered_map<std::string, std::unordered_map<std::string, double>> 
    create_summary_statistics(const TimeSeriesData& data);
    
    /**
     * @brief Resample time series to new interval
     * @param data Input time series data
     * @param new_interval Target interval in seconds
     * @param method Resampling method ("interpolate", "nearest", "zero_hold")
     * @return Resampled time series data
     */
    TimeSeriesData resample(
        const TimeSeriesData& data,
        double new_interval,
        const std::string& method = "interpolate"
    );
    
    /**
     * @brief Merge multiple time series with timestamp alignment
     * @param datasets Vector of time series to merge
     * @param alignment_method Method for timestamp alignment
     * @return Merged time series data
     */
    TimeSeriesData merge_time_series(
        const std::vector<TimeSeriesData>& datasets,
        const std::string& alignment_method = "interpolate"
    );
    
    /**
     * @brief Get current configuration
     * @return Current configuration
     */
    const Config& get_config() const { return config_; }
    
    /**
     * @brief Update configuration
     * @param new_config New configuration
     */
    void set_config(const Config& new_config) { config_ = new_config; }

private:
    // Aggregation implementations
    std::vector<double> aggregate_mean(
        const std::vector<double>& data,
        const std::vector<std::pair<size_t, size_t>>& groups
    );
    
    std::vector<double> aggregate_sum(
        const std::vector<double>& data,
        const std::vector<std::pair<size_t, size_t>>& groups
    );
    
    std::vector<double> aggregate_minmax(
        const std::vector<double>& data,
        const std::vector<std::pair<size_t, size_t>>& groups,
        bool find_max
    );
    
    std::vector<double> aggregate_median(
        const std::vector<double>& data,
        const std::vector<std::pair<size_t, size_t>>& groups
    );
    
    // Transformation implementations
    void apply_log_transform(std::vector<double>& data);
    void apply_sqrt_transform(std::vector<double>& data);
    void apply_difference_transform(std::vector<double>& data);
    void apply_percent_change_transform(std::vector<double>& data);
    void apply_zscore_transform(std::vector<double>& data);
    void apply_minmax_transform(std::vector<double>& data);
    
    // Utility functions
    std::vector<std::pair<size_t, size_t>> create_time_groups(
        const std::vector<double>& timestamps,
        double target_interval
    );
    
    std::vector<double> generate_target_timestamps(
        double start_time,
        double end_time,
        double interval
    );
    
    void ensure_buffer_capacity(size_t required_size) const;
    
    // SIMD-optimized operations
    double simd_group_mean(const double* data, size_t start, size_t end);
    double simd_group_sum(const double* data, size_t start, size_t end);
    std::pair<double, double> simd_group_minmax(const double* data, size_t start, size_t end);
    
    // Parallel processing helpers
    template<typename Func>
    void parallel_transform_metric(std::vector<double>& data, Func&& func);
    
    template<typename Func>
    std::vector<double> parallel_aggregate_groups(
        const std::vector<double>& data,
        const std::vector<std::pair<size_t, size_t>>& groups,
        Func&& func
    );
};

} // namespace core
} // namespace etiobench
