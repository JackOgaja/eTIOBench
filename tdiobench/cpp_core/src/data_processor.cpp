#include "data_processor.hpp"
#include "simd_utils.hpp"
#include "threading_utils.hpp"

#include <algorithm>
#include <numeric>
#include <cmath>
#include <execution>
#include <cassert>

namespace etiobench {
namespace core {

void DataProcessor::TimeSeriesData::clear() {
    timestamps.clear();
    metrics.clear();
    interval = 1.0;
    tier.clear();
    test_id.clear();
}

void DataProcessor::TimeSeriesData::reserve(size_t capacity) {
    timestamps.reserve(capacity);
    for (auto& [name, values] : metrics) {
        values.reserve(capacity);
    }
}

DataProcessor::DataProcessor(const Config& config)
    : config_(config) {
    // Pre-allocate work buffers
    work_buffer_1_.reserve(config_.max_chunk_size);
    work_buffer_2_.reserve(config_.max_chunk_size);
    index_buffer_.reserve(config_.max_chunk_size);
}

DataProcessor::DataProcessor()
    : config_() {
    // Pre-allocate work buffers with default config
    work_buffer_1_.reserve(config_.max_chunk_size);
    work_buffer_2_.reserve(config_.max_chunk_size);
    index_buffer_.reserve(config_.max_chunk_size);
}

DataProcessor::AggregationResult DataProcessor::aggregate_time_series(
    const TimeSeriesData& data,
    double target_interval,
    AggregationFunction func) {
    
    if (data.empty() || target_interval <= 0.0) {
        return AggregationResult{};
    }
    
    AggregationResult result;
    result.aggregation_function = [func]() {
        switch (func) {
            case AggregationFunction::MEAN: return "mean";
            case AggregationFunction::SUM: return "sum";
            case AggregationFunction::MIN: return "min";
            case AggregationFunction::MAX: return "max";
            case AggregationFunction::MEDIAN: return "median";
            case AggregationFunction::STD_DEV: return "std_dev";
            default: return "mean";
        }
    }();
    
    result.new_interval = target_interval;
    result.original_points = data.size();
    
    // Create time groups for aggregation
    auto time_groups = create_time_groups(data.timestamps, target_interval);
    
    if (time_groups.empty()) {
        return result;
    }
    
    result.aggregated_points = time_groups.size();
    
    // Generate aggregated timestamps
    result.aggregated_timestamps.reserve(time_groups.size());
    for (const auto& [start_idx, end_idx] : time_groups) {
        if (start_idx < data.timestamps.size()) {
            // Use the middle timestamp of the group
            size_t mid_idx = start_idx + (end_idx - start_idx) / 2;
            result.aggregated_timestamps.push_back(data.timestamps[mid_idx]);
        }
    }
    
    // Aggregate each metric
    for (const auto& [metric_name, metric_values] : data.metrics) {
        if (metric_values.size() != data.timestamps.size()) {
            continue;  // Skip mismatched data
        }
        
        std::vector<double> aggregated_values;
        
        switch (func) {
            case AggregationFunction::MEAN:
                aggregated_values = aggregate_mean(metric_values, time_groups);
                break;
            case AggregationFunction::SUM:
                aggregated_values = aggregate_sum(metric_values, time_groups);
                break;
            case AggregationFunction::MIN:
                aggregated_values = aggregate_minmax(metric_values, time_groups, false);
                break;
            case AggregationFunction::MAX:
                aggregated_values = aggregate_minmax(metric_values, time_groups, true);
                break;
            case AggregationFunction::MEDIAN:
                aggregated_values = aggregate_median(metric_values, time_groups);
                break;
            default:
                aggregated_values = aggregate_mean(metric_values, time_groups);
                break;
        }
        
        result.aggregated_metrics[metric_name] = std::move(aggregated_values);
    }
    
    return result;
}

DataProcessor::TimeSeriesData& DataProcessor::transform_data_inplace(
    TimeSeriesData& data,
    TransformationType transformation) {
    
    if (data.empty()) {
        return data;
    }
    
    // Apply transformation to each metric
    for (auto& [metric_name, metric_values] : data.metrics) {
        switch (transformation) {
            case TransformationType::LOG:
                apply_log_transform(metric_values);
                break;
            case TransformationType::SQRT:
                apply_sqrt_transform(metric_values);
                break;
            case TransformationType::DIFFERENCE:
                apply_difference_transform(metric_values);
                break;
            case TransformationType::PERCENT_CHANGE:
                apply_percent_change_transform(metric_values);
                break;
            case TransformationType::Z_SCORE:
                apply_zscore_transform(metric_values);
                break;
            case TransformationType::MIN_MAX:
                apply_minmax_transform(metric_values);
                break;
            case TransformationType::STANDARDIZE:
                apply_zscore_transform(metric_values);  // Standardize is same as z-score
                break;
        }
    }
    
    return data;
}

DataProcessor::TimeSeriesData DataProcessor::transform_data(
    const TimeSeriesData& data,
    TransformationType transformation) {
    
    TimeSeriesData result = data;  // Copy
    return transform_data_inplace(result, transformation);
}

void DataProcessor::normalize_metric(
    TimeSeriesData& data,
    const std::string& metric_name,
    TransformationType normalization) {
    
    auto it = data.metrics.find(metric_name);
    if (it == data.metrics.end()) {
        return;  // Metric not found
    }
    
    auto& metric_values = it->second;
    
    switch (normalization) {
        case TransformationType::Z_SCORE:
            apply_zscore_transform(metric_values);
            break;
        case TransformationType::MIN_MAX:
            apply_minmax_transform(metric_values);
            break;
        default:
            apply_zscore_transform(metric_values);
            break;
    }
}

std::vector<std::vector<double>> DataProcessor::to_matrix(
    const TimeSeriesData& data,
    const std::vector<std::string>& metric_order) {
    
    if (data.empty()) {
        return {};
    }
    
    // Determine metric order
    std::vector<std::string> metrics;
    if (metric_order.empty()) {
        // Use all available metrics
        metrics.reserve(data.metrics.size());
        for (const auto& [name, values] : data.metrics) {
            metrics.push_back(name);
        }
        std::sort(metrics.begin(), metrics.end());  // Consistent ordering
    } else {
        // Use specified order, filtering available metrics
        for (const auto& metric : metric_order) {
            if (data.metrics.find(metric) != data.metrics.end()) {
                metrics.push_back(metric);
            }
        }
    }
    
    if (metrics.empty()) {
        return {};
    }
    
    // Create matrix
    std::vector<std::vector<double>> matrix;
    matrix.reserve(data.size());
    
    for (size_t i = 0; i < data.size(); ++i) {
        std::vector<double> row;
        row.reserve(metrics.size());
        
        for (const auto& metric : metrics) {
            const auto& values = data.metrics.at(metric);
            if (i < values.size()) {
                row.push_back(values[i]);
            } else {
                row.push_back(std::numeric_limits<double>::quiet_NaN());
            }
        }
        
        matrix.push_back(std::move(row));
    }
    
    return matrix;
}

std::unordered_map<std::string, std::unordered_map<std::string, double>> 
DataProcessor::create_summary_statistics(const TimeSeriesData& data) {
    
    std::unordered_map<std::string, std::unordered_map<std::string, double>> result;
    
    for (const auto& [metric_name, values] : data.metrics) {
        if (values.empty()) {
            continue;
        }
        
        auto& stats = result[metric_name];
        
        // Calculate basic statistics using SIMD when possible
        if (config_.enable_simd) {
            stats["mean"] = simd::simd_mean(values.data(), values.size());
            stats["variance"] = simd::simd_variance(values.data(), values.size(), stats["mean"]);
        } else {
            stats["mean"] = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
            
            double sum_sq_diff = 0.0;
            for (double val : values) {
                double diff = val - stats["mean"];
                sum_sq_diff += diff * diff;
            }
            stats["variance"] = sum_sq_diff / (values.size() - 1);
        }
        
        stats["std_dev"] = std::sqrt(stats["variance"]);
        
        auto [min_iter, max_iter] = std::minmax_element(values.begin(), values.end());
        stats["min"] = *min_iter;
        stats["max"] = *max_iter;
        stats["range"] = stats["max"] - stats["min"];
        stats["count"] = static_cast<double>(values.size());
        
        // Calculate median
        ensure_buffer_capacity(values.size());
        work_buffer_1_.assign(values.begin(), values.end());
        std::nth_element(work_buffer_1_.begin(), 
                        work_buffer_1_.begin() + work_buffer_1_.size() / 2,
                        work_buffer_1_.end());
        stats["median"] = work_buffer_1_[work_buffer_1_.size() / 2];
    }
    
    return result;
}

std::vector<std::pair<size_t, size_t>> DataProcessor::create_time_groups(
    const std::vector<double>& timestamps,
    double target_interval) {
    
    std::vector<std::pair<size_t, size_t>> groups;
    
    if (timestamps.empty() || target_interval <= 0.0) {
        return groups;
    }
    
    double start_time = timestamps.front();
    double current_boundary = start_time + target_interval;
    size_t group_start = 0;
    
    for (size_t i = 0; i < timestamps.size(); ++i) {
        if (timestamps[i] >= current_boundary) {
            // End current group
            if (i > group_start) {
                groups.emplace_back(group_start, i);
            }
            
            // Start new group
            group_start = i;
            current_boundary = timestamps[i] + target_interval;
        }
    }
    
    // Add final group
    if (group_start < timestamps.size()) {
        groups.emplace_back(group_start, timestamps.size());
    }
    
    return groups;
}

std::vector<double> DataProcessor::aggregate_mean(
    const std::vector<double>& data,
    const std::vector<std::pair<size_t, size_t>>& groups) {
    
    std::vector<double> result;
    result.reserve(groups.size());
    
    if (config_.enable_parallel_processing && groups.size() > 100) {
        return parallel_aggregate_groups(data, groups, [this](const double* ptr, size_t start, size_t end) {
            return simd_group_mean(ptr, start, end);
        });
    }
    
    // Sequential processing
    for (const auto& [start_idx, end_idx] : groups) {
        if (start_idx >= end_idx || end_idx > data.size()) {
            result.push_back(0.0);
            continue;
        }
        
        double mean = simd_group_mean(data.data(), start_idx, end_idx);
        result.push_back(mean);
    }
    
    return result;
}

std::vector<double> DataProcessor::aggregate_sum(
    const std::vector<double>& data,
    const std::vector<std::pair<size_t, size_t>>& groups) {
    
    std::vector<double> result;
    result.reserve(groups.size());
    
    for (const auto& [start_idx, end_idx] : groups) {
        if (start_idx >= end_idx || end_idx > data.size()) {
            result.push_back(0.0);
            continue;
        }
        
        double sum = simd_group_sum(data.data(), start_idx, end_idx);
        result.push_back(sum);
    }
    
    return result;
}

void DataProcessor::apply_log_transform(std::vector<double>& data) {
    if (config_.enable_parallel_processing && data.size() > 1000) {
        parallel_transform_metric(data, [](double& val) {
            val = val > 0.0 ? std::log(val) : std::numeric_limits<double>::quiet_NaN();
        });
    } else {
        for (double& val : data) {
            val = val > 0.0 ? std::log(val) : std::numeric_limits<double>::quiet_NaN();
        }
    }
}

void DataProcessor::apply_zscore_transform(std::vector<double>& data) {
    if (data.size() < 2) return;
    
    double mean = simd::simd_mean(data.data(), data.size());
    double std_dev = std::sqrt(simd::simd_variance(data.data(), data.size(), mean));
    
    if (std_dev == 0.0) {
        std::fill(data.begin(), data.end(), 0.0);
        return;
    }
    
    if (config_.enable_parallel_processing && data.size() > 1000) {
        parallel_transform_metric(data, [mean, std_dev](double& val) {
            val = (val - mean) / std_dev;
        });
    } else {
        for (double& val : data) {
            val = (val - mean) / std_dev;
        }
    }
}

void DataProcessor::apply_minmax_transform(std::vector<double>& data) {
    if (data.empty()) return;
    
    auto [min_iter, max_iter] = std::minmax_element(data.begin(), data.end());
    double min_val = *min_iter;
    double max_val = *max_iter;
    double range = max_val - min_val;
    
    if (range == 0.0) {
        std::fill(data.begin(), data.end(), 0.0);
        return;
    }
    
    if (config_.enable_parallel_processing && data.size() > 1000) {
        parallel_transform_metric(data, [min_val, range](double& val) {
            val = (val - min_val) / range;
        });
    } else {
        for (double& val : data) {
            val = (val - min_val) / range;
        }
    }
}

void DataProcessor::apply_sqrt_transform(std::vector<double>& data) {
    if (config_.enable_parallel_processing && data.size() > 1000) {
        parallel_transform_metric(data, [](double& val) {
            val = val >= 0 ? std::sqrt(val) : 0.0;
        });
    } else {
        for (double& val : data) {
            val = val >= 0 ? std::sqrt(val) : 0.0;
        }
    }
}

void DataProcessor::apply_difference_transform(std::vector<double>& data) {
    if (data.size() < 2) return;
    
    std::vector<double> result;
    result.reserve(data.size() - 1);
    
    for (size_t i = 1; i < data.size(); ++i) {
        result.push_back(data[i] - data[i-1]);
    }
    
    data = std::move(result);
}

void DataProcessor::apply_percent_change_transform(std::vector<double>& data) {
    if (data.size() < 2) return;
    
    std::vector<double> result;
    result.reserve(data.size() - 1);
    
    for (size_t i = 1; i < data.size(); ++i) {
        if (data[i-1] != 0.0) {
            result.push_back((data[i] - data[i-1]) / data[i-1] * 100.0);
        } else {
            result.push_back(0.0);
        }
    }
    
    data = std::move(result);
}

double DataProcessor::simd_group_mean(const double* data, size_t start, size_t end) {
    if (start >= end) return 0.0;
    
    size_t count = end - start;
    if (config_.enable_simd && count >= 4) {
        return simd::simd_mean(data + start, count);
    }
    
    return std::accumulate(data + start, data + end, 0.0) / count;
}

double DataProcessor::simd_group_sum(const double* data, size_t start, size_t end) {
    if (start >= end) return 0.0;
    
    size_t count = end - start;
    if (config_.enable_simd && count >= 4) {
        return simd::simd_sum(data + start, count);
    }
    
    return std::accumulate(data + start, data + end, 0.0);
}

void DataProcessor::ensure_buffer_capacity(size_t required_size) const {
    if (work_buffer_1_.capacity() < required_size) {
        work_buffer_1_.reserve(required_size);
    }
    if (work_buffer_2_.capacity() < required_size) {
        work_buffer_2_.reserve(required_size);
    }
}

template<typename Func>
void DataProcessor::parallel_transform_metric(std::vector<double>& data, Func&& func) {
    threading::parallel_for(0, data.size(), config_.num_threads, [&data, &func](size_t i) {
        func(data[i]);
    });
}

template<typename Func>
std::vector<double> DataProcessor::parallel_aggregate_groups(
    const std::vector<double>& data,
    const std::vector<std::pair<size_t, size_t>>& groups,
    Func&& func) {
    
    std::vector<double> result(groups.size());
    
    threading::parallel_for(0, groups.size(), config_.num_threads, [&](size_t i) {
        const auto& [start_idx, end_idx] = groups[i];
        if (start_idx < end_idx && end_idx <= data.size()) {
            result[i] = func(data.data(), start_idx, end_idx);
        } else {
            result[i] = 0.0;
        }
    });
    
    return result;
}

std::vector<double> DataProcessor::aggregate_minmax(
    const std::vector<double>& data,
    const std::vector<std::pair<size_t, size_t>>& groups,
    bool find_max) {
    
    if (find_max) {
        return parallel_aggregate_groups(data, groups, [](const double* data, size_t start, size_t end) {
            return *std::max_element(data + start, data + end);
        });
    } else {
        return parallel_aggregate_groups(data, groups, [](const double* data, size_t start, size_t end) {
            return *std::min_element(data + start, data + end);
        });
    }
}

std::vector<double> DataProcessor::aggregate_median(
    const std::vector<double>& data,
    const std::vector<std::pair<size_t, size_t>>& groups) {
    
    return parallel_aggregate_groups(data, groups, [](const double* data, size_t start, size_t end) {
        std::vector<double> temp(data + start, data + end);
        std::sort(temp.begin(), temp.end());
        size_t size = temp.size();
        if (size % 2 == 0) {
            return (temp[size/2 - 1] + temp[size/2]) / 2.0;
        } else {
            return temp[size/2];
        }
    });
}

} // namespace core
} // namespace etiobench
