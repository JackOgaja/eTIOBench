# eTIOBench User Guide

## Enhanced Tiered Storage I/O Benchmark Suite - Comprehensive Usage Documentation

### Table of Contents
1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Basic Usage](#basic-usage)
4. [Advanced Features](#advanced-features)
5. [Configuration](#configuration)
6. [Data Collection](#data-collection)
7. [Analysis and Reporting](#analysis-and-reporting)
8. [Safety Features](#safety-features)
9. [Troubleshooting](#troubleshooting)
10. [Best Practices](#best-practices)

---

## Quick Start

### Running Your First Benchmark

```bash
# Simple benchmark on a directory
python -m tdiobench.cli.commandline run --tiers /path/to/storage --profile quick_scan

# Production-safe benchmark with time series data
python -m tdiobench.cli.commandline run --tiers /mnt/nvme,/mnt/ssd --profile production_safe --time-series

# Comprehensive analysis with all features
python -m tdiobench.cli.commandline run --tiers /storage --profile comprehensive \
  --time-series --system-metrics --network-analysis --save
```

### Understanding Results

eTIOBench generates results in multiple formats:
- **JSON**: Machine-readable detailed results (`results/[UUID].json`)
- **HTML**: Interactive reports with charts (`results/report.html`)
- **Markdown**: Executive summaries and recommendations

---

## Installation

### Prerequisites

- **Python 3.8+**
- **FIO (Flexible I/O Tester)**: Required for benchmark execution
- **Linux/Unix System**: Primary supported platform
- **Root/Sudo Access**: Required for some advanced features

### Install Dependencies

```bash
# Install FIO
# Ubuntu/Debian:
sudo apt-get install fio

# RHEL/CentOS:
sudo yum install fio

# Or build from source:
git clone https://github.com/axboe/fio.git
cd fio && make && sudo make install
```

### Install eTIOBench

```bash
# Clone the repository
git clone https://github.com/JackOgaja/eTIOBench.git
cd eTIOBench

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -m tdiobench.cli.commandline check-env
```

---

## Basic Usage

### Command Structure

```bash
python -m tdiobench.cli.commandline [GLOBAL_OPTIONS] COMMAND [COMMAND_OPTIONS]
```

### Available Commands

| Command | Purpose | Example |
|---------|---------|---------|
| `run` | Execute benchmarks | `run --tiers /storage --profile quick_scan` |
| `analyze` | Analyze existing results | `analyze --result-id abc123 --analysis-types statistical` |
| `report` | Generate reports | `report --result-id abc123 --format html` |
| `compare` | Compare multiple results | `compare --results abc123,def456` |
| `list` | List available results | `list --last 10` |
| `profile` | Manage profiles | `profile list` |
| `tier` | Manage storage tiers | `tier validate /storage/path` |
| `check-env` | Environment validation | `check-env` |
| `cleanup` | Clean old results | `cleanup --older-than 30` |

### Global Options

- `--config PATH`: Custom configuration file (JSON/YAML)
- `--log-level LEVEL`: Set logging level (debug, info, warning, error)
- `--output-dir PATH`: Override output directory
- `--version`: Show version information

---

## Advanced Features

### 1. Benchmark Profiles

eTIOBench includes predefined profiles optimized for different use cases:

#### Quick Scan Profile
```bash
python -m tdiobench.cli.commandline run --tiers /storage --profile quick_scan
```
- **Duration**: 30 seconds
- **Block Sizes**: 4k, 1m
- **Patterns**: read, write
- **Use Case**: Fast overview, development testing

#### Production Safe Profile
```bash
python -m tdiobench.cli.commandline run --tiers /storage --profile production_safe
```
- **Duration**: 60 seconds
- **Block Sizes**: 4k, 1m
- **Patterns**: read, randread
- **Safety**: Conservative resource limits
- **Use Case**: Production environment testing

#### Comprehensive Profile
```bash
python -m tdiobench.cli.commandline run --tiers /storage --profile comprehensive
```
- **Duration**: 300 seconds
- **Block Sizes**: 512, 4k, 8k, 16k, 32k, 64k, 128k, 256k, 512k, 1m, 2m, 4m
- **Patterns**: read, write, randread, randwrite, readwrite, randrw
- **Use Case**: Detailed performance characterization

#### Latency Focused Profile
```bash
python -m tdiobench.cli.commandline run --tiers /storage --profile latency_focused
```
- **Duration**: 120 seconds
- **Block Sizes**: 512, 4k, 8k
- **Patterns**: randread, randwrite
- **Use Case**: Latency-sensitive application analysis

### 2. Custom Benchmark Parameters

Override profile defaults with command-line options:

```bash
python -m tdiobench.cli.commandline run \
  --tiers /nvme,/ssd,/hdd \
  --duration 180 \
  --block-sizes 4k,64k,1m \
  --patterns read,write,randread \
  --io-depth 64 \
  --num-jobs 4 \
  --direct \
  --time-series \
  --system-metrics
```

#### Key Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--tiers` | Storage paths to test | `/mnt/nvme,/mnt/ssd` |
| `--duration` | Test duration (≥30s) | `120` |
| `--block-sizes` | I/O block sizes | `4k,64k,1m` |
| `--patterns` | I/O patterns | `read,write,randread` |
| `--io-depth` | Queue depth | `32` |
| `--num-jobs` | Parallel jobs | `4` |
| `--rate-limit` | I/O rate limit | `50m` (50MB/s) |
| `--direct/--no-direct` | Direct I/O flag | `--direct` |

### 3. Multi-Tier Benchmarking

Test multiple storage tiers simultaneously:

```bash
python -m tdiobench.cli.commandline run \
  --tiers /mnt/nvme,/mnt/ssd,/mnt/hdd \
  --profile comprehensive \
  --baseline-tier /mnt/nvme \
  --time-series \
  --save
```

**Benefits:**
- Comparative analysis across storage types
- Tier performance ranking
- Workload placement recommendations
- Cost-performance optimization

---

## Data Collection

### Time Series Data Collection

eTIOBench provides **FIO-native time series collection** for accurate, real-time performance monitoring:

```bash
python -m tdiobench.cli.commandline run \
  --tiers /storage \
  --time-series \
  --ts-interval 1.0 \
  --profile comprehensive
```

#### Features:
- **Real-time Data**: 1-second granularity during benchmark execution
- **FIO Integration**: Direct log file monitoring for maximum accuracy
- **Three Key Metrics**:
  - **Throughput (MB/s)**: Bandwidth measurements
  - **IOPS**: I/O operations per second
  - **Latency (ms)**: Response time measurements

#### Data Accuracy:
- ✅ **Benchmark-specific**: Only captures I/O from test files, not system-wide activity
- ✅ **FIO-native**: Uses FIO's internal logging for precise measurements
- ✅ **Non-intrusive**: Background monitoring doesn't affect benchmark performance

### System Metrics Collection

Monitor system resources during benchmarks:

```bash
python -m tdiobench.cli.commandline run \
  --tiers /storage \
  --system-metrics \
  --sm-interval 2.0 \
  --profile production_safe
```

#### Collected Metrics:
- **CPU Usage**: Per-core and overall utilization
- **Memory Usage**: RAM consumption and swap activity
- **Disk I/O**: System-wide disk activity
- **Network I/O**: Network bandwidth utilization

### Network Analysis

Analyze network impact during distributed storage testing:

```bash
python -m tdiobench.cli.commandline run \
  --tiers /mnt/nfs,/mnt/lustre \
  --network-analysis \
  --monitor-network eth0,ib0 \
  --profile comprehensive
```

#### Network Metrics:
- Interface-specific bandwidth utilization
- Packet loss and error rates
- Network latency during I/O operations
- Impact correlation with storage performance

---

## Configuration

### Configuration Files

eTIOBench supports JSON and YAML configuration files for complex setups:

#### Example Configuration Structure:

```yaml
benchmark_suite:
  metadata:
    created_by: "Your Organization"
    description: "Custom benchmark configuration"
    version: "1.0.0"
  
  core:
    safety:
      enabled: true
      max_cpu_percent: 70
      max_memory_percent: 80
      monitoring_interval_seconds: 2
    
    output:
      base_directory: "./benchmark_results"
      create_timestamp_subdirectory: true
      retain_raw_data: true
  
  tiers:
    tier_definitions:
      - name: "nvme_tier"
        path: "/mnt/nvme"
        type: "nvme"
        description: "NVMe storage tier"
        expected_min_throughput_MBps: 2000
        expected_min_iops: 400000
      
      - name: "ssd_tier"
        path: "/mnt/ssd"
        type: "ssd"
        description: "SSD storage tier"
        expected_min_throughput_MBps: 500
        expected_min_iops: 100000
  
  benchmark_profiles:
    custom_profile:
      description: "Custom benchmark settings"
      duration_seconds: 180
      block_sizes: ["4k", "64k", "1m"]
      patterns: ["read", "write", "randread"]
      io_depth: 32
      direct: true
```

#### Using Configuration Files:

```bash
python -m tdiobench.cli.commandline --config /path/to/config.yaml run \
  --tiers /storage \
  --profile custom_profile
```

### Storage Tier Configuration

Define and validate storage tiers:

```bash
# Validate a storage path
python -m tdiobench.cli.commandline tier validate /mnt/storage

# List configured tiers
python -m tdiobench.cli.commandline tier list

# Test tier performance expectations
python -m tdiobench.cli.commandline tier test /mnt/nvme
```

### Profile Management

Create and manage custom benchmark profiles:

```bash
# List available profiles
python -m tdiobench.cli.commandline profile list

# Show profile details
python -m tdiobench.cli.commandline profile show comprehensive

# Create custom profile (via configuration file)
python -m tdiobench.cli.commandline --config custom_profiles.yaml profile list
```

---

## Analysis and Reporting

### Statistical Analysis

eTIOBench provides advanced statistical analysis capabilities:

```bash
python -m tdiobench.cli.commandline analyze \
  --result-id abc123-def456-789 \
  --analysis-types statistical,time_series,anomaly \
  --output-format html
```

#### Analysis Types:

1. **Statistical Analysis**:
   - Performance distributions
   - Outlier detection
   - Confidence intervals
   - Variance analysis

2. **Time Series Analysis**:
   - Trend detection
   - Seasonality analysis
   - Performance degradation identification
   - Peak/valley analysis

3. **Anomaly Detection**:
   - Z-score based detection
   - Isolation forest algorithm
   - Performance threshold violations
   - Unusual pattern identification

4. **Network Analysis**:
   - Network impact correlation
   - Bandwidth utilization patterns
   - Latency correlation analysis

### Comparative Analysis

Compare multiple benchmark results:

```bash
python -m tdiobench.cli.commandline compare \
  --results abc123,def456,ghi789 \
  --metrics throughput,iops,latency \
  --output-format html \
  --baseline abc123
```

#### Comparison Features:
- Performance regression detection
- Tier ranking and optimization
- Configuration impact analysis
- Historical trend analysis

### Report Generation

Generate comprehensive reports in multiple formats:

```bash
python -m tdiobench.cli.commandline report \
  --result-id abc123-def456-789 \
  --format html,json,markdown \
  --include-charts \
  --output-dir ./reports
```

#### Report Formats:

1. **HTML Reports**:
   - Interactive charts and visualizations
   - Detailed performance analysis
   - Executive summaries
   - Actionable recommendations

2. **JSON Reports**:
   - Machine-readable results
   - API integration support
   - Custom analysis workflows
   - Data export capabilities

3. **Markdown Reports**:
   - Documentation-ready format
   - Executive summaries
   - Performance highlights
   - Integration with documentation systems

### Visualization

eTIOBench provides rich visualization capabilities:

#### Performance Charts:
- Throughput over time
- IOPS distributions
- Latency percentile analysis
- Multi-tier comparisons

#### System Resource Charts:
- CPU utilization during benchmarks
- Memory usage patterns
- Network bandwidth correlation
- Resource contention analysis

#### Heatmaps:
- Performance across block sizes and patterns
- Tier performance matrices
- Time-based performance variations
- Resource utilization heatmaps

---

## Safety Features

### Production Safety Mode

eTIOBench includes comprehensive safety features for production environments:

```bash
python -m tdiobench.cli.commandline run \
  --tiers /production/storage \
  --profile production_safe \
  --no-safety  # Only use with extreme caution!
```

#### Safety Features:

1. **Resource Monitoring**:
   - Real-time CPU usage monitoring
   - Memory consumption tracking
   - I/O impact assessment
   - Network bandwidth monitoring

2. **Automatic Throttling**:
   - Progressive load reduction when thresholds exceeded
   - Configurable throttling steps
   - Graceful performance degradation
   - Benchmark continuation with reduced impact

3. **Emergency Abort**:
   - Immediate termination when critical thresholds reached
   - System resource protection
   - Data integrity preservation
   - Graceful cleanup procedures

4. **Configurable Limits**:
   ```yaml
   safety:
     max_cpu_percent: 70          # Maximum CPU usage
     max_memory_percent: 80       # Maximum memory usage
     throttle_step_percent: 10    # Throttling increment
     abort_threshold_cpu_percent: 90    # Emergency abort CPU
     abort_threshold_memory_percent: 95 # Emergency abort memory
     monitoring_interval_seconds: 2     # Monitoring frequency
   ```

### Safety Best Practices

1. **Always Use Production Safe Profile**:
   ```bash
   python -m tdiobench.cli.commandline run \
     --tiers /production/storage \
     --profile production_safe
   ```

2. **Monitor During Execution**:
   - Watch system resource usage
   - Monitor application performance
   - Check for user complaints
   - Have rollback procedures ready

3. **Test in Staging First**:
   ```bash
   python -m tdiobench.cli.commandline run \
     --tiers /staging/storage \
     --profile comprehensive \
     --dry-run  # Preview without execution
   ```

4. **Use Rate Limiting**:
   ```bash
   python -m tdiobench.cli.commandline run \
     --tiers /storage \
     --rate-limit 50m \  # Limit to 50MB/s
     --io-depth 8        # Reduce queue depth
   ```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. Environment Setup Issues

**Problem**: `fio: command not found`
```bash
# Solution: Install FIO
sudo apt-get install fio  # Ubuntu/Debian
sudo yum install fio      # RHEL/CentOS
```

**Problem**: Permission denied errors
```bash
# Solution: Check directory permissions
ls -la /path/to/storage
sudo chown user:group /path/to/storage
chmod 755 /path/to/storage
```

#### 2. Benchmark Execution Issues

**Problem**: "Duration must be at least 30 seconds"
```bash
# Solution: Use minimum duration
python -m tdiobench.cli.commandline run \
  --tiers /storage \
  --duration 30  # Minimum allowed
```

**Problem**: High system resource usage
```bash
# Solution: Use production safe settings
python -m tdiobench.cli.commandline run \
  --tiers /storage \
  --profile production_safe \
  --rate-limit 25m
```

#### 3. Data Collection Issues

**Problem**: No time series data in results
```bash
# Solution: Enable time series collection explicitly
python -m tdiobench.cli.commandline run \
  --tiers /storage \
  --time-series \
  --profile quick_scan
```

**Problem**: Incomplete system metrics
```bash
# Solution: Check permissions and enable system metrics
sudo python -m tdiobench.cli.commandline run \
  --tiers /storage \
  --system-metrics \
  --profile production_safe
```

#### 4. Analysis and Reporting Issues

**Problem**: Missing charts in HTML reports
```bash
# Solution: Install visualization dependencies
pip install matplotlib plotly seaborn
```

**Problem**: Analysis timeout for large datasets
```bash
# Solution: Increase timeout or reduce data scope
python -m tdiobench.cli.commandline analyze \
  --result-id abc123 \
  --analysis-types statistical \  # Reduce analysis scope
  --timeout 3600                  # Increase timeout
```

### Environment Validation

Use the built-in environment checker:

```bash
python -m tdiobench.cli.commandline check-env
```

This validates:
- Python version compatibility
- Required dependencies
- FIO installation and version
- Storage path accessibility
- Permission requirements

### Debugging Tips

1. **Enable Debug Logging**:
   ```bash
   python -m tdiobench.cli.commandline --log-level debug run \
     --tiers /storage --profile quick_scan
   ```

2. **Dry Run Mode**:
   ```bash
   python -m tdiobench.cli.commandline run \
     --tiers /storage \
     --profile quick_scan \
     --dry-run
   ```

3. **Check System Resources**:
   ```bash
   # Before benchmark
   htop
   iostat -x 1
   iotop
   
   # During benchmark (separate terminal)
   watch -n 1 'free -h && df -h'
   ```

4. **Validate Storage Paths**:
   ```bash
   python -m tdiobench.cli.commandline tier validate /path/to/storage
   ```

---

## Best Practices

### 1. Benchmark Planning

#### Pre-Benchmark Checklist:
- [ ] Validate storage paths and permissions
- [ ] Check available disk space (≥10GB recommended)
- [ ] Verify system resources availability
- [ ] Plan for benchmark duration and impact
- [ ] Notify relevant teams about testing
- [ ] Prepare monitoring and rollback procedures

#### Environment Preparation:
```bash
# 1. Check environment
python -m tdiobench.cli.commandline check-env

# 2. Validate storage tiers
python -m tdiobench.cli.commandline tier validate /mnt/nvme
python -m tdiobench.cli.commandline tier validate /mnt/ssd

# 3. Test with dry run
python -m tdiobench.cli.commandline run \
  --tiers /mnt/nvme,/mnt/ssd \
  --profile production_safe \
  --dry-run
```

### 2. Production Environment Guidelines

#### Use Conservative Settings:
```bash
python -m tdiobench.cli.commandline run \
  --tiers /production/storage \
  --profile production_safe \
  --rate-limit 25m \
  --io-depth 8 \
  --num-jobs 1 \
  --time-series
```

#### Monitor During Execution:
- System resource usage (CPU, memory, I/O)
- Application performance impact
- User experience and complaints
- Network bandwidth utilization

#### Schedule Appropriately:
- **Preferred**: Off-hours or maintenance windows
- **Acceptable**: Low-traffic periods with monitoring
- **Avoid**: Peak business hours or critical operations

### 3. Development and Testing Guidelines

#### Use Comprehensive Analysis:
```bash
python -m tdiobench.cli.commandline run \
  --tiers /dev/storage \
  --profile comprehensive \
  --time-series \
  --system-metrics \
  --network-analysis \
  --save
```

#### Iterate and Compare:
```bash
# Baseline benchmark
python -m tdiobench.cli.commandline run \
  --tiers /storage \
  --profile comprehensive \
  --tag baseline \
  --save

# After optimization
python -m tdiobench.cli.commandline run \
  --tiers /storage \
  --profile comprehensive \
  --tag optimized \
  --save

# Compare results
python -m tdiobench.cli.commandline compare \
  --results baseline,optimized \
  --output-format html
```

### 4. Data Management

#### Result Organization:
- Use meaningful tags: `--tag "pre-upgrade-nvme"`
- Save important results: `--save`
- Regular cleanup: `cleanup --older-than 30`
- Export critical data: `report --format json`

#### Backup and Archive:
```bash
# Backup critical results
cp -r results/ /backup/benchmark-results-$(date +%Y%m%d)

# Export specific results
python -m tdiobench.cli.commandline report \
  --result-id critical-benchmark-id \
  --format json \
  --output-dir /archive/benchmarks
```

### 5. Analysis and Optimization

#### Systematic Analysis Approach:
1. **Baseline Establishment**:
   ```bash
   python -m tdiobench.cli.commandline run \
     --tiers /storage \
     --profile comprehensive \
     --tag baseline \
     --save
   ```

2. **Performance Optimization**:
   - Adjust block sizes for workload
   - Optimize I/O depth and job parallelism
   - Tune filesystem parameters
   - Configure hardware settings

3. **Validation Testing**:
   ```bash
   python -m tdiobench.cli.commandline run \
     --tiers /storage \
     --profile comprehensive \
     --tag post-optimization \
     --save
   ```

4. **Comparative Analysis**:
   ```bash
   python -m tdiobench.cli.commandline compare \
     --results baseline,post-optimization \
     --output-format html
   ```

#### Performance Tuning Insights:
- **Throughput Optimization**: Larger block sizes (1m, 4m)
- **IOPS Optimization**: Smaller block sizes (4k, 8k)
- **Latency Optimization**: Lower I/O depth, direct I/O
- **Multi-tier Workloads**: Match workload to tier characteristics

### 6. Reporting and Documentation

#### Executive Reporting:
```bash
python -m tdiobench.cli.commandline report \
  --result-id benchmark-id \
  --format markdown \
  --include-executive-summary \
  --output-dir ./reports
```

#### Technical Documentation:
```bash
python -m tdiobench.cli.commandline report \
  --result-id benchmark-id \
  --format html \
  --include-charts \
  --include-raw-data \
  --output-dir ./technical-reports
```

#### Continuous Monitoring:
- Schedule regular benchmarks
- Track performance trends over time
- Set up automated alerting for degradation
- Integrate with monitoring systems

---

## Example Workflows

### Workflow 1: Production Storage Validation

```bash
# 1. Environment check
python -m tdiobench.cli.commandline check-env

# 2. Validate storage tiers
python -m tdiobench.cli.commandline tier validate /mnt/production

# 3. Conservative benchmark with monitoring
python -m tdiobench.cli.commandline run \
  --tiers /mnt/production \
  --profile production_safe \
  --time-series \
  --system-metrics \
  --tag production-validation-$(date +%Y%m%d) \
  --save

# 4. Generate report
python -m tdiobench.cli.commandline report \
  --result-id [result-id] \
  --format html,markdown \
  --output-dir ./production-reports
```

### Workflow 2: Performance Optimization

```bash
# 1. Baseline measurement
python -m tdiobench.cli.commandline run \
  --tiers /storage \
  --profile comprehensive \
  --time-series \
  --tag baseline \
  --save

# 2. Apply optimizations (filesystem tuning, hardware config, etc.)

# 3. Validation measurement
python -m tdiobench.cli.commandline run \
  --tiers /storage \
  --profile comprehensive \
  --time-series \
  --tag optimized \
  --save

# 4. Compare results
python -m tdiobench.cli.commandline compare \
  --results baseline,optimized \
  --output-format html \
  --output-dir ./optimization-analysis
```

### Workflow 3: Multi-Tier Analysis

```bash
# 1. Comprehensive multi-tier benchmark
python -m tdiobench.cli.commandline run \
  --tiers /mnt/nvme,/mnt/ssd,/mnt/hdd \
  --profile comprehensive \
  --baseline-tier /mnt/nvme \
  --time-series \
  --system-metrics \
  --network-analysis \
  --save

# 2. Advanced analysis
python -m tdiobench.cli.commandline analyze \
  --result-id [result-id] \
  --analysis-types statistical,time_series,anomaly \
  --output-format html

# 3. Generate executive report
python -m tdiobench.cli.commandline report \
  --result-id [result-id] \
  --format markdown \
  --include-executive-summary \
  --include-recommendations
```

---

## Summary

eTIOBench provides a comprehensive solution for storage performance analysis with:

- **Safety-first design** for production environments
- **FIO-native time series collection** for accurate performance monitoring
- **Advanced analysis capabilities** with statistical and anomaly detection
- **Rich visualization and reporting** for actionable insights
- **Flexible configuration** for diverse storage environments
- **Multi-tier support** for complex storage infrastructures

For additional support, consult the project documentation, example configurations, and community resources.

---

**Last Updated**: July 18, 2025  
**Version**: 2.0.0  
**Documentation Version**: 1.0  
