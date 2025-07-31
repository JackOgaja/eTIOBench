#include "time_series_collector.hpp"
#include "threading_utils.hpp"

#include <algorithm>
#include <numeric>
#include <chrono>
#include <fstream>
#include <sstream>
#include <cstring>

#ifdef __linux__
#include <sys/sysinfo.h>
#include <sys/stat.h>
#include <unistd.h>
#include <proc/readproc.h>
#include <proc/sysinfo.h>
#endif

#ifdef __APPLE__
#include <sys/types.h>
#include <sys/sysctl.h>
#include <mach/mach.h>
#include <mach/host_info.h>
#include <mach/mach_host.h>
#include <mach/vm_map.h>
#endif

namespace etiobench {
namespace collection {

// CircularBuffer implementation
bool TimeSeriesCollector::CircularBuffer::push(const DataPoint& point) {
    std::lock_guard<std::mutex> lock(mutex);
    
    if (full()) {
        return false; // Buffer overflow
    }
    
    buffer[tail] = point;
    tail = (tail + 1) % capacity;
    ++size;
    return true;
}

bool TimeSeriesCollector::CircularBuffer::pop(DataPoint& point) {
    std::lock_guard<std::mutex> lock(mutex);
    
    if (empty()) {
        return false;
    }
    
    point = buffer[head];
    head = (head + 1) % capacity;
    --size;
    return true;
}

// TimeSeriesCollector implementation
TimeSeriesCollector::TimeSeriesCollector(const CollectionConfig& config)
    : config_(config) {
    buffer_ = std::make_unique<CircularBuffer>(config_.buffer_size);
    stats_.start_time = std::chrono::steady_clock::now();
}

TimeSeriesCollector::TimeSeriesCollector()
    : config_() {
    buffer_ = std::make_unique<CircularBuffer>(config_.buffer_size);
    stats_.start_time = std::chrono::steady_clock::now();
}

TimeSeriesCollector::~TimeSeriesCollector() {
    stop_collection();
}

void TimeSeriesCollector::add_metric(const MetricConfig& metric) {
    std::lock_guard<std::mutex> lock(collection_mutex_);
    metrics_[metric.name] = metric;
}

void TimeSeriesCollector::remove_metric(const std::string& metric_name) {
    std::lock_guard<std::mutex> lock(collection_mutex_);
    metrics_.erase(metric_name);
}

void TimeSeriesCollector::update_metric_config(const std::string& metric_name, const MetricConfig& config) {
    std::lock_guard<std::mutex> lock(collection_mutex_);
    auto it = metrics_.find(metric_name);
    if (it != metrics_.end()) {
        it->second = config;
    }
}

std::vector<TimeSeriesCollector::MetricConfig> TimeSeriesCollector::get_metrics() const {
    std::lock_guard<std::mutex> lock(collection_mutex_);
    std::vector<MetricConfig> result;
    result.reserve(metrics_.size());
    
    for (const auto& pair : metrics_) {
        result.push_back(pair.second);
    }
    
    return result;
}

void TimeSeriesCollector::start_collection() {
    if (collecting_.load()) {
        return;
    }
    
    collecting_.store(true);
    shutdown_.store(false);
    
    // Start worker threads
    worker_threads_.clear();
    worker_threads_.reserve(config_.num_worker_threads);
    
    for (size_t i = 0; i < config_.num_worker_threads; ++i) {
        worker_threads_.emplace_back(&TimeSeriesCollector::worker_thread_func, this);
    }
    
    stats_.start_time = std::chrono::steady_clock::now();
}

void TimeSeriesCollector::stop_collection() {
    if (!collecting_.load()) {
        return;
    }
    
    collecting_.store(false);
    shutdown_.store(true);
    collection_cv_.notify_all();
    
    // Join worker threads
    for (auto& thread : worker_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    worker_threads_.clear();
}

void TimeSeriesCollector::collect_point(const DataPoint& point) {
    if (!collecting_.load()) {
        return;
    }
    
    // Apply filtering if enabled
    if (should_filter_point(point)) {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        ++stats_.points_dropped;
        return;
    }
    
    // Add to buffer
    if (!buffer_->push(point)) {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        ++stats_.points_dropped;
        return;
    }
    
    // Update statistics
    update_stats(point);
    
    // Stream callback if enabled
    {
        std::lock_guard<std::mutex> lock(callback_mutex_);
        if (stream_callback_) {
            stream_callback_(point);
        }
    }
    
    // Auto flush if needed
    if (config_.enable_auto_flush) {
        auto_flush_if_needed();
    }
}

void TimeSeriesCollector::collect_batch(const std::vector<DataPoint>& points) {
    if (!collecting_.load()) {
        return;
    }
    
    for (const auto& point : points) {
        collect_point(point);
    }
}

void TimeSeriesCollector::register_collection_function(const std::string& name, CollectionFunction func) {
    std::lock_guard<std::mutex> lock(functions_mutex_);
    collection_functions_[name] = std::move(func);
}

void TimeSeriesCollector::unregister_collection_function(const std::string& name) {
    std::lock_guard<std::mutex> lock(functions_mutex_);
    collection_functions_.erase(name);
}

TimeSeriesCollector::CollectionResult TimeSeriesCollector::get_collected_data() {
    CollectionResult result;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Extract all data from buffer
    DataPoint point;
    while (buffer_->pop(point)) {
        result.data_points.push_back(point);
    }
    
    // Get statistics
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        result.stats = stats_;
    }
    
    // Calculate metric summaries
    std::unordered_map<std::string, std::vector<double>> metric_values;
    for (const auto& dp : result.data_points) {
        metric_values[dp.metric_name].push_back(dp.value);
    }
    
    for (const auto& pair : metric_values) {
        const auto& values = pair.second;
        if (!values.empty()) {
            double sum = std::accumulate(values.begin(), values.end(), 0.0);
            result.metric_summaries[pair.first + "_mean"] = sum / values.size();
            
            auto minmax = std::minmax_element(values.begin(), values.end());
            result.metric_summaries[pair.first + "_min"] = *minmax.first;
            result.metric_summaries[pair.first + "_max"] = *minmax.second;
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    result.collection_duration_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    
    return result;
}

TimeSeriesCollector::CollectionResult TimeSeriesCollector::get_data_for_metric(const std::string& metric_name) {
    auto all_data = get_collected_data();
    CollectionResult result;
    
    result.stats = all_data.stats;
    result.success = all_data.success;
    result.error_message = all_data.error_message;
    
    // Filter for specific metric
    for (const auto& point : all_data.data_points) {
        if (point.metric_name == metric_name) {
            result.data_points.push_back(point);
        }
    }
    
    // Update metric summaries for this metric only
    if (!result.data_points.empty()) {
        std::vector<double> values;
        for (const auto& point : result.data_points) {
            values.push_back(point.value);
        }
        
        double sum = std::accumulate(values.begin(), values.end(), 0.0);
        result.metric_summaries[metric_name + "_mean"] = sum / values.size();
        
        auto minmax = std::minmax_element(values.begin(), values.end());
        result.metric_summaries[metric_name + "_min"] = *minmax.first;
        result.metric_summaries[metric_name + "_max"] = *minmax.second;
    }
    
    return result;
}

TimeSeriesCollector::CollectionResult TimeSeriesCollector::get_data_for_tier(const std::string& tier) {
    auto all_data = get_collected_data();
    CollectionResult result;
    
    result.stats = all_data.stats;
    result.success = all_data.success;
    result.error_message = all_data.error_message;
    
    // Filter for specific tier
    for (const auto& point : all_data.data_points) {
        if (point.tier == tier) {
            result.data_points.push_back(point);
        }
    }
    
    return result;
}

TimeSeriesCollector::CollectionResult TimeSeriesCollector::get_data_in_range(double start_time, double end_time) {
    auto all_data = get_collected_data();
    CollectionResult result;
    
    result.stats = all_data.stats;
    result.success = all_data.success;
    result.error_message = all_data.error_message;
    
    // Filter for time range
    for (const auto& point : all_data.data_points) {
        if (point.timestamp >= start_time && point.timestamp <= end_time) {
            result.data_points.push_back(point);
        }
    }
    
    return result;
}

void TimeSeriesCollector::set_stream_callback(StreamCallback callback) {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    stream_callback_ = std::move(callback);
}

void TimeSeriesCollector::remove_stream_callback() {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    stream_callback_ = nullptr;
}

TimeSeriesCollector::CollectionStats TimeSeriesCollector::get_stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    CollectionStats stats = stats_;
    stats.collection_rate_hz = calculate_collection_rate();
    stats.memory_usage_mb = get_memory_usage_mb();
    return stats;
}

void TimeSeriesCollector::reset_stats() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_ = CollectionStats{};
    stats_.start_time = std::chrono::steady_clock::now();
}

double TimeSeriesCollector::get_memory_usage_mb() const {
    // Estimate memory usage
    size_t buffer_size = buffer_->available() * sizeof(DataPoint);
    size_t cache_size = 0;
    
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        for (const auto& pair : metric_cache_) {
            cache_size += pair.second.size() * sizeof(double);
        }
    }
    
    return (buffer_size + cache_size) / (1024.0 * 1024.0);
}

double TimeSeriesCollector::get_collection_rate_hz() const {
    return calculate_collection_rate();
}

void TimeSeriesCollector::flush_buffers() {
    // For now, this is handled by get_collected_data()
    // In a full implementation, this might write to disk
}

void TimeSeriesCollector::clear_buffers() {
    DataPoint point;
    while (buffer_->pop(point)) {
        // Discard all points
    }
    
    std::lock_guard<std::mutex> lock(cache_mutex_);
    metric_cache_.clear();
}

size_t TimeSeriesCollector::get_buffer_utilization() const {
    return (buffer_->available() * 100) / config_.buffer_size;
}

void TimeSeriesCollector::update_config(const CollectionConfig& config) {
    std::lock_guard<std::mutex> lock(collection_mutex_);
    config_ = config;
    
    // Recreate buffer if size changed
    if (buffer_->capacity != config_.buffer_size) {
        buffer_ = std::make_unique<CircularBuffer>(config_.buffer_size);
    }
}

// Private methods
void TimeSeriesCollector::worker_thread_func() {
    while (!shutdown_.load()) {
        // Process collection functions
        std::unordered_map<std::string, CollectionFunction> functions_copy;
        {
            std::lock_guard<std::mutex> lock(functions_mutex_);
            functions_copy = collection_functions_;
        }
        
        for (const auto& pair : functions_copy) {
            if (shutdown_.load()) break;
            process_collection_function(pair.first, pair.second);
        }
        
        // Wait for next collection interval
        std::unique_lock<std::mutex> lock(collection_mutex_);
        collection_cv_.wait_for(lock, 
            std::chrono::milliseconds(static_cast<int>(config_.default_interval_ms)),
            [this] { return shutdown_.load(); });
    }
}

void TimeSeriesCollector::process_collection_function(const std::string& name, const CollectionFunction& func) {
    try {
        auto points = func();
        collect_batch(points);
    } catch (const std::exception& e) {
        // Log error but continue collection
        std::lock_guard<std::mutex> lock(stats_mutex_);
        ++stats_.points_dropped;
    }
}

void TimeSeriesCollector::update_stats(const DataPoint& point) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    ++stats_.total_points_collected;
    stats_.last_collection = std::chrono::steady_clock::now();
}

bool TimeSeriesCollector::should_filter_point(const DataPoint& point) const {
    auto metric_it = metrics_.find(point.metric_name);
    if (metric_it == metrics_.end()) {
        return false; // No filtering config for this metric
    }
    
    const auto& config = metric_it->second;
    if (!config.enable_filtering) {
        return false;
    }
    
    // Check thresholds
    if (point.value < config.threshold_min || point.value > config.threshold_max) {
        return true;
    }
    
    return false;
}

double TimeSeriesCollector::calculate_collection_rate() const {
    auto now = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - stats_.start_time);
    
    if (duration.count() == 0) {
        return 0.0;
    }
    
    return static_cast<double>(stats_.total_points_collected) / duration.count();
}

void TimeSeriesCollector::auto_flush_if_needed() {
    if (buffer_->available() >= config_.auto_flush_threshold) {
        flush_buffers();
    }
}

// SystemMetricsCollector implementation
SystemMetricsCollector::SystemMetricsCollector(const Config& config)
    : config_(config) {
    metrics_history_.reserve(max_history_size_);
}

SystemMetricsCollector::SystemMetricsCollector()
    : config_() {
    metrics_history_.reserve(max_history_size_);
}

SystemMetricsCollector::~SystemMetricsCollector() {
    stop_collection();
}

void SystemMetricsCollector::start_collection() {
    if (collecting_.load()) {
        return;
    }
    
    collecting_.store(true);
    shutdown_.store(false);
    
    collection_thread_ = std::thread(&SystemMetricsCollector::collection_loop, this);
}

void SystemMetricsCollector::stop_collection() {
    if (!collecting_.load()) {
        return;
    }
    
    collecting_.store(false);
    shutdown_.store(true);
    
    if (collection_thread_.joinable()) {
        collection_thread_.join();
    }
}

SystemMetricsCollector::SystemMetrics SystemMetricsCollector::get_current_metrics() {
    return collect_system_metrics_impl();
}

std::vector<SystemMetricsCollector::SystemMetrics> SystemMetricsCollector::get_metrics_history() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    return metrics_history_;
}

SystemMetricsCollector::SystemMetrics SystemMetricsCollector::get_average_metrics(size_t window_size) const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    if (metrics_history_.empty()) {
        return SystemMetrics{};
    }
    
    if (window_size == 0 || window_size > metrics_history_.size()) {
        window_size = metrics_history_.size();
    }
    
    SystemMetrics avg;
    size_t start_idx = metrics_history_.size() - window_size;
    
    for (size_t i = start_idx; i < metrics_history_.size(); ++i) {
        const auto& metrics = metrics_history_[i];
        avg.cpu_usage_percent += metrics.cpu_usage_percent;
        avg.memory_used_gb += metrics.memory_used_gb;
        avg.disk_read_bps += metrics.disk_read_bps;
        avg.disk_write_bps += metrics.disk_write_bps;
        avg.net_rx_bps += metrics.net_rx_bps;
        avg.net_tx_bps += metrics.net_tx_bps;
    }
    
    double factor = 1.0 / window_size;
    avg.cpu_usage_percent *= factor;
    avg.memory_used_gb *= factor;
    avg.disk_read_bps *= factor;
    avg.disk_write_bps *= factor;
    avg.net_rx_bps *= factor;
    avg.net_tx_bps *= factor;
    
    return avg;
}

std::vector<TimeSeriesCollector::DataPoint> SystemMetricsCollector::collect_cpu_metrics() {
    auto metrics = get_current_metrics();
    std::vector<TimeSeriesCollector::DataPoint> points;
    
    points.emplace_back(metrics.timestamp, metrics.cpu_usage_percent, "cpu_usage_percent");
    points.emplace_back(metrics.timestamp, metrics.cpu_user_percent, "cpu_user_percent");
    points.emplace_back(metrics.timestamp, metrics.cpu_system_percent, "cpu_system_percent");
    points.emplace_back(metrics.timestamp, metrics.cpu_idle_percent, "cpu_idle_percent");
    points.emplace_back(metrics.timestamp, metrics.cpu_iowait_percent, "cpu_iowait_percent");
    
    if (config_.collect_per_core_cpu) {
        for (size_t i = 0; i < metrics.per_core_usage.size(); ++i) {
            points.emplace_back(metrics.timestamp, metrics.per_core_usage[i], 
                              "cpu_core_" + std::to_string(i) + "_usage");
        }
    }
    
    return points;
}

std::vector<TimeSeriesCollector::DataPoint> SystemMetricsCollector::collect_memory_metrics() {
    auto metrics = get_current_metrics();
    std::vector<TimeSeriesCollector::DataPoint> points;
    
    points.emplace_back(metrics.timestamp, metrics.memory_used_gb, "memory_used_gb");
    points.emplace_back(metrics.timestamp, metrics.memory_free_gb, "memory_free_gb");
    points.emplace_back(metrics.timestamp, metrics.memory_available_gb, "memory_available_gb");
    points.emplace_back(metrics.timestamp, metrics.memory_cached_gb, "memory_cached_gb");
    points.emplace_back(metrics.timestamp, metrics.memory_buffers_gb, "memory_buffers_gb");
    points.emplace_back(metrics.timestamp, metrics.swap_used_gb, "swap_used_gb");
    
    return points;
}

std::vector<TimeSeriesCollector::DataPoint> SystemMetricsCollector::collect_io_metrics() {
    auto metrics = get_current_metrics();
    std::vector<TimeSeriesCollector::DataPoint> points;
    
    points.emplace_back(metrics.timestamp, metrics.disk_read_bps, "disk_read_bps");
    points.emplace_back(metrics.timestamp, metrics.disk_write_bps, "disk_write_bps");
    points.emplace_back(metrics.timestamp, metrics.disk_read_iops, "disk_read_iops");
    points.emplace_back(metrics.timestamp, metrics.disk_write_iops, "disk_write_iops");
    points.emplace_back(metrics.timestamp, metrics.disk_util_percent, "disk_util_percent");
    
    if (config_.collect_per_device_io) {
        for (const auto& pair : metrics.per_device_metrics) {
            points.emplace_back(metrics.timestamp, pair.second, pair.first);
        }
    }
    
    return points;
}

std::vector<TimeSeriesCollector::DataPoint> SystemMetricsCollector::collect_network_metrics() {
    auto metrics = get_current_metrics();
    std::vector<TimeSeriesCollector::DataPoint> points;
    
    points.emplace_back(metrics.timestamp, metrics.net_rx_bps, "net_rx_bps");
    points.emplace_back(metrics.timestamp, metrics.net_tx_bps, "net_tx_bps");
    points.emplace_back(metrics.timestamp, metrics.net_rx_pps, "net_rx_pps");
    points.emplace_back(metrics.timestamp, metrics.net_tx_pps, "net_tx_pps");
    
    if (config_.collect_per_interface_net) {
        for (const auto& pair : metrics.per_interface_metrics) {
            points.emplace_back(metrics.timestamp, pair.second, pair.first);
        }
    }
    
    return points;
}

std::vector<TimeSeriesCollector::DataPoint> SystemMetricsCollector::collect_process_metrics() {
    auto metrics = get_current_metrics();
    std::vector<TimeSeriesCollector::DataPoint> points;
    
    points.emplace_back(metrics.timestamp, metrics.process_cpu_percent, "process_cpu_percent");
    points.emplace_back(metrics.timestamp, metrics.process_memory_mb, "process_memory_mb");
    points.emplace_back(metrics.timestamp, static_cast<double>(metrics.process_threads), "process_threads");
    points.emplace_back(metrics.timestamp, static_cast<double>(metrics.process_fds), "process_fds");
    
    return points;
}

void SystemMetricsCollector::update_config(const Config& config) {
    config_ = config;
}

void SystemMetricsCollector::register_with_collector(TimeSeriesCollector& collector) {
    // Register collection functions with the time series collector
    collector.register_collection_function("system_cpu", 
        [this]() { return collect_cpu_metrics(); });
    
    collector.register_collection_function("system_memory", 
        [this]() { return collect_memory_metrics(); });
    
    collector.register_collection_function("system_io", 
        [this]() { return collect_io_metrics(); });
    
    collector.register_collection_function("system_network", 
        [this]() { return collect_network_metrics(); });
    
    if (config_.collect_process_metrics) {
        collector.register_collection_function("system_process", 
            [this]() { return collect_process_metrics(); });
    }
}

// Private methods
SystemMetricsCollector::SystemMetrics SystemMetricsCollector::collect_system_metrics_impl() {
    SystemMetrics metrics;
    metrics.timestamp = get_current_timestamp();
    
    // Collect CPU metrics
    metrics.cpu_usage_percent = get_cpu_usage();
    if (config_.collect_per_core_cpu) {
        metrics.per_core_usage = get_per_core_cpu_usage();
    }
    
    // Collect other metrics
    get_memory_info(metrics);
    get_io_info(metrics);
    get_network_info(metrics);
    
    if (config_.collect_process_metrics) {
        get_process_info(metrics);
    }
    
    return metrics;
}

void SystemMetricsCollector::collection_loop() {
    while (!shutdown_.load()) {
        auto metrics = collect_system_metrics_impl();
        
        // Add to history
        {
            std::lock_guard<std::mutex> lock(metrics_mutex_);
            metrics_history_.push_back(metrics);
            trim_history_if_needed();
        }
        
        // Sleep for collection interval
        std::this_thread::sleep_for(
            std::chrono::milliseconds(static_cast<int>(config_.collection_interval_ms)));
    }
}

double SystemMetricsCollector::get_current_timestamp() const {
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration<double>(duration).count();
}

void SystemMetricsCollector::trim_history_if_needed() {
    if (metrics_history_.size() > max_history_size_) {
        metrics_history_.erase(metrics_history_.begin(), 
                              metrics_history_.begin() + (metrics_history_.size() - max_history_size_));
    }
}

// Platform-specific implementations (simplified for this example)
double SystemMetricsCollector::get_cpu_usage() {
    // This would contain platform-specific CPU usage collection
    // For now, return a placeholder
    return 50.0; // Example: 50% CPU usage
}

std::vector<double> SystemMetricsCollector::get_per_core_cpu_usage() {
    // This would contain platform-specific per-core CPU collection
    size_t num_cores = std::thread::hardware_concurrency();
    return std::vector<double>(num_cores, 50.0); // Example values
}

void SystemMetricsCollector::get_memory_info(SystemMetrics& metrics) {
    // Platform-specific memory info collection
    metrics.memory_total_gb = 16.0; // Example
    metrics.memory_used_gb = 8.0;   // Example
    metrics.memory_free_gb = 8.0;   // Example
}

void SystemMetricsCollector::get_io_info(SystemMetrics& metrics) {
    // Platform-specific I/O info collection
    metrics.disk_read_bps = 1000000.0;  // Example
    metrics.disk_write_bps = 500000.0;  // Example
}

void SystemMetricsCollector::get_network_info(SystemMetrics& metrics) {
    // Platform-specific network info collection
    metrics.net_rx_bps = 10000000.0; // Example
    metrics.net_tx_bps = 5000000.0;  // Example
}

void SystemMetricsCollector::get_process_info(SystemMetrics& metrics) {
    // Platform-specific process info collection
    metrics.process_cpu_percent = 25.0; // Example
    metrics.process_memory_mb = 512.0;  // Example
}

} // namespace collection
} // namespace etiobench
