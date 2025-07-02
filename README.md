# extensible-Tiered Storage I/O Benchmark Suite (eTIOBench)

[![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)](https://github.com/JackOgaja/tiered-storage-benchmark)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![Platform](https://img.shields.io/badge/platform-Linux-lightgrey.svg)](https://kernel.org)
[![Status](https://img.shields.io/badge/status-Production%20Ready-brightgreen.svg)](README.md)

**Professional-grade storage performance analysis platform for multi-tier storage environments**

A comprehensive, production-safe benchmarking solution for tiered storage systems with advanced analysis capabilities.

## Overview

The Tiered Storage I/O Benchmark Suite is designed to characterize and analyze the performance of complex storage infrastructures across multiple tiers. It provides safety-first benchmarking with real-time monitoring, sophisticated analysis, and rich visualization capabilities.

Unlike traditional benchmarking tools that focus solely on raw performance numbers, this suite emphasizes safety, comprehensive analysis, and actionable **PERFORMANCE** insights—helping you understand not just how fast your storage is, but *why* it performs the way it does and how it affects production workloads. This suite also provides modules which can be integrated into monitoring systems and opportunity for intelligent job scheduling for performance improvement.

## Key Features

- **Production-Safe Benchmarking**
  - Real-time resource monitoring with automatic throttling
  - Emergency abort capabilities when thresholds are exceeded
  - Configurable safety limits for CPU, memory, and I/O

- **Multi-Tier Storage Testing**
  - Specialized configurations for different storage tiers (NVMe, SSD, HDD)
  - Support for distributed filesystems (Lustre, Ceph, BeeGFS)
  - Tier-specific benchmark parameters

- **Advanced Analysis**
  - Statistical analysis with outlier detection
  - Time series analysis with trend and seasonality detection
  - Network impact analysis
  - Anomaly detection using multiple methods (Z-score, isolation forest, etc.)
  - Correlation analysis between metrics

- **Rich Visualization**
  - Interactive charts (when configured)
  - Time series visualizations
  - Performance heatmaps
  - Distribution analysis
  - Comparative visualizations

- **Comprehensive Reporting**
  - Detailed HTML, JSON, and Markdown reports
  - Executive summaries
  - Actionable recommendations
  - Performance bottleneck identification

- **Flexible Engines**
  - FIO integration for detailed I/O testing
  - Extensible engine architecture

## Installation

### Prerequisites

- Python 3.8 or higher
- FIO 3.16 or higher
- For visualization: matplotlib, plotly, seaborn
- For analysis: numpy, pandas, scipy, scikit-learn
- Storage systems mounted and accessible

### Installation Options

#### Option 1: Install from PyPI (Recommended for Users)

```bash
# Install the basic package
pip install enhanced-tiered-storage-benchmark

# Install with optional components
pip install enhanced-tiered-storage-benchmark[network]  # For network analysis
pip install enhanced-tiered-storage-benchmark[distributed]  # For distributed benchmarking
pip install enhanced-tiered-storage-benchmark[full]  # For all optional components
```

#### Option 2: Install from Source (Recommended for Developers)

```bash
# Clone the repository
git clone https://github.com/JackOgaja/enhanced-tiered-storage-benchmark.git
cd enhanced-tiered-storage-benchmark

# Install in development mode
pip install -e .  # Basic installation
pip install -e ".[dev]"  # Installation with development tools
```

#### Option 3: System-wide Installation with Dependencies

```bash
# First, install system dependencies
bash scripts/install_dependencies.sh

# Then install the Python package
pip install enhanced-tiered-storage-benchmark
```

### Verifying Your Installation

After installation, you can verify that everything is set up correctly:

```bash
# Check if the command-line tool is available
benchmark-suite --version

# Run the environment checker
python scripts/check_environment.py
```

### Building Distribution Packages

If you want to build distribution packages for the benchmark suite:

```bash
# Make sure you have the build package
pip install build

# Build both source distribution and wheel
python -m build

# This will create .tar.gz and .whl files in the dist/ directory
```

## Quick Start

### Basic Benchmark

```bash
# Run a quick, safe benchmark on a specific storage tier
benchmark-suite run --tier /mnt/nvme --profile quick_scan
```

### With Custom Configuration

```bash
# Run with a custom configuration file
benchmark-suite run --config my_benchmark_config.json
```

### View Results

```bash
# Generate HTML report from the last benchmark
benchmark-suite report --format html --output benchmark_report.html
```

## Configuration

The benchmark suite uses JSON configuration files to define behavior. A minimal example:

```json
{
  "benchmark_suite": {
    "core": {
      "safety": {
        "enabled": true,
        "max_cpu_percent": 70,
        "max_memory_percent": 80
      }
    },
    "tiers": {
      "tier_definitions": [
        {
          "name": "nvme_tier",
          "path": "/mnt/nvme",
          "type": "nvme"
        }
      ]
    },
    "benchmark_profiles": {
      "quick_scan": {
        "description": "Fast overview of performance",
        "duration_seconds": 30,
        "block_sizes": ["4k", "1m"],
        "patterns": ["read", "write"]
      }
    }
  }
}
```

See [Configuration Guide](docs/configuration.md) for detailed options.

## Detailed Usage

### Running Benchmarks

```bash
# Run a comprehensive benchmark on multiple tiers
benchmark-suite run --tiers nvme_tier,ssd_tier --profile comprehensive

# Run with specific block sizes and IO patterns
benchmark-suite run --tier nvme_tier --block-sizes 4k,64k,1m --patterns read,write,randread

# Run with time limit
benchmark-suite run --tier ssd_tier --duration 600
```

### Analyzing Results

```bash
# Analyze the last benchmark with anomaly detection
benchmark-suite analyze --last --anomaly-detection

# Compare multiple benchmark results
benchmark-suite compare results_1.json results_2.json --output comparison.html
```

### Visualization

```bash
# Generate visualizations from benchmark data
benchmark-suite visualize --data benchmark_data.json --charts time_series,heatmap

# Create a performance dashboard
benchmark-suite dashboard --data benchmark_data.json --output dashboard.html
```

## Architecture

The Enhanced Tiered Storage I/O Benchmark Suite is organized into several key components:

- **Core**: Configuration management, safety monitoring, and benchmark orchestration
- **Engines**: Benchmark execution engines (currently FIO)
- **Collection**: Time series data collection and system metrics
- **Analysis**: Statistical analysis, anomaly detection, and correlation analysis
- **Visualization**: Chart generation and reporting
- **Results**: Result storage and management

```
benchmark_suite/
├── cli/            # Command-line interface
├── core/           # Core components 
├── engines/        # Benchmark execution engines
├── collection/     # Data collection modules
├── analysis/       # Analysis modules
├── visualization/  # Report and chart generation
├── results/        # Result storage
└── utils/          # Utility functions
```

## Safety Features

The benchmark suite prioritizes the safety of your production environment:

- **Resource Monitoring**: Continuously monitors CPU, memory, and I/O to prevent impact on production workloads
- **Adaptive Throttling**: Automatically reduces benchmark load when resource usage approaches thresholds
- **Emergency Abort**: Immediately terminates benchmarks if safety thresholds are exceeded
- **Progressive Loading**: Gradually increases I/O load to identify safety limits before full testing

## Production Impact Assessment

When considering running this benchmark in production environments, note:

- **Resource Contention**: Benchmarks will compete with production workloads for CPU, memory, and I/O resources
- **Storage Impact**: I/O-intensive benchmarks may significantly reduce available IOPS and bandwidth for production
- **Recovery Time**: Storage systems may require time to recover normal performance after intensive benchmarking
- **Tier-Specific Impacts**: HDD systems experience more prolonged impact than SSD/NVMe systems

We recommend:
1. Run comprehensive benchmarks during maintenance windows when possible
2. Use the `production_safe` profile for minimal impact during operational hours
3. Enable all safety features with conservative thresholds
4. Monitor production application performance during benchmarking

## Development and Contribution

We welcome contributions to the Enhanced Tiered Storage I/O Benchmark Suite! 

### Development Setup

```bash
# Clone the repository
git clone https://github.com/JackOgaja/enhanced-tiered-storage-benchmark.git
cd enhanced-tiered-storage-benchmark

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest
```

### Contribution Guidelines

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this benchmark suite in your research or publications, please cite:

```bibtex
@software{tiered_io_bench_2025,
  author = {Jack Ogaja},
  title = {Tiered Storage I/O Benchmark Suite},
  year = {2025},
  url = {https://github.com/JackOgaja/Tiered-I-OBench},
  license = {MIT}
}
```

## Contact and Support

- **Project Maintainer**: JackOgaja
- **GitHub Issues**: Please report bugs and feature requests via GitHub issues
- **Documentation**: Full documentation available at [docs/](docs/)

---

## 🙏 Acknowledgments

- **FIO Team**: Built using [FIO (Flexible I/O Tester)](https://github.com/axboe/fio)
- **Python Community**: Analysis tools powered by Python scientific stack

---

## 📊 Project Statistics

![GitHub stars](https://img.shields.io/github/stars/JackOgaja/tiered-storage-benchmark?style=social)
![GitHub forks](https://img.shields.io/github/forks/JackOgaja/tiered-storage-benchmark?style=social)
![GitHub issues](https://img.shields.io/github/issues/JackOgaja/tiered-storage-benchmark)
![GitHub pull requests](https://img.shields.io/github/issues-pr/JackOgaja/tiered-storage-benchmark)

---


Created for storage performance engineers and sysadmins everywhere.
