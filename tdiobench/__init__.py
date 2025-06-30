#!/usr/bin/env python3
"""
Tiered Storage I/O Benchmark Suite

A comprehensive benchmarking suite for analyzing storage performance across
different tiers of storage infrastructure with statistical rigor.

Author: Jack Ogaja
Date: 2025-06-26
Version: 2.0.0
"""

# Import core modules for easier access
from tdiobench.core.benchmark_suite import BenchmarkSuite, BenchmarkEvent
from tdiobench.core.benchmark_config import BenchmarkConfig
from tdiobench.core.benchmark_data import BenchmarkData, BenchmarkResult, TimeSeriesData
from tdiobench.core.benchmark_analysis import AnalysisResult
from tdiobench.core.benchmark_exceptions import (
    BenchmarkError, BenchmarkConfigError, BenchmarkExecutionError,
    BenchmarkResourceError, BenchmarkDataError, BenchmarkAnalysisError,
    BenchmarkReportError, BenchmarkAPIError, BenchmarkNetworkError,
    BenchmarkTimeoutError, BenchmarkAuthenticationError, BenchmarkStorageError
)

# Define version information
__version__ = "2.0.0"
__author__ = "Jack Ogaja"
__email__ = "jogaja@acm.org"
__license__ = "MIT"

__all__ = [
    'BenchmarkSuite', 'BenchmarkEvent', 'BenchmarkConfig',
    'BenchmarkData', 'BenchmarkResult', 'TimeSeriesData', 'AnalysisResult',
    'BenchmarkError', 'BenchmarkConfigError', 'BenchmarkExecutionError',
    'BenchmarkResourceError', 'BenchmarkDataError', 'BenchmarkAnalysisError',
    'BenchmarkReportError', 'BenchmarkAPIError', 'BenchmarkNetworkError',
    'BenchmarkTimeoutError', 'BenchmarkAuthenticationError', 'BenchmarkStorageError'
]
