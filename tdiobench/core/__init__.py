"""
Core components of the Tiered Storage I/O Benchmark Suite (tdiobench).

This package contains the core components of the benchmark suite, including
the main benchmark orchestrator, configuration management, data structures,
and exception hierarchy.

Author: Jack Ogaja
Date: 2025-06-26
"""

from tdiobench.core.benchmark_suite import BenchmarkSuite, BenchmarkEvent
from tdiobench.core.benchmark_config import BenchmarkConfig
from tdiobench.core.benchmark_data import BenchmarkData, BenchmarkResult, TimeSeriesData
from tdiobench.core.benchmark_analysis import AnalysisResult
from tdiobench.core.benchmark_runner import BenchmarkRunner
from tdiobench.core.benchmark_exceptions import (
    BenchmarkError, BenchmarkConfigError, BenchmarkExecutionError,
    BenchmarkResourceError, BenchmarkDataError, BenchmarkAnalysisError,
    BenchmarkReportError, BenchmarkAPIError, BenchmarkNetworkError,
    BenchmarkTimeoutError, BenchmarkAuthenticationError, BenchmarkStorageError
)

__all__ = [
    'BenchmarkSuite', 'BenchmarkEvent', 'BenchmarkConfig', 'BenchmarkRunner',
    'BenchmarkData', 'BenchmarkResult', 'TimeSeriesData', 'AnalysisResult',
    'BenchmarkError', 'BenchmarkConfigError', 'BenchmarkExecutionError',
    'BenchmarkResourceError', 'BenchmarkDataError', 'BenchmarkAnalysisError',
    'BenchmarkReportError', 'BenchmarkAPIError', 'BenchmarkNetworkError',
    'BenchmarkTimeoutError', 'BenchmarkAuthenticationError', 'BenchmarkStorageError'
]

# Module initialization
import logging
logger = logging.getLogger(__name__)
logger.debug(f"Benchmark Suite Core module initialized (version {__version__})")
