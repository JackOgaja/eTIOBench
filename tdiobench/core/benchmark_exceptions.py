#!/usr/bin/env python3
"""
Benchmark Exception Hierarchy (Tiered Storage I/O BEnchmark)

This module provides a standardized exception hierarchy for the benchmark suite,
enabling consistent error handling and reporting.

Author: Jack Ogaja
Date: 2025-06-26
"""


class BenchmarkError(Exception):
    """Base class for all benchmark exceptions."""

    def __init__(self, message: str, cause: Exception = None):
        self.message = message
        self.cause = cause
        super().__init__(self.message)


class BenchmarkConfigError(BenchmarkError):
    """Exception raised for configuration errors."""


class BenchmarkExecutionError(BenchmarkError):
    """Exception raised for benchmark execution errors."""


class BenchmarkResourceError(BenchmarkError):
    """Exception raised for resource limit violations."""


class BenchmarkDataError(BenchmarkError):
    """Exception raised for data handling errors."""


class BenchmarkDataValidationError(BenchmarkError):
    """Exception raised for data handling errors."""


class BenchmarkVisualizationError(BenchmarkError):
    """Exception raised for data handling errors."""


class BenchmarkAnalysisError(BenchmarkError):
    """Exception raised for analysis errors."""


class BenchmarkReportError(BenchmarkError):
    """Exception raised for report generation errors."""


class BenchmarkAPIError(BenchmarkError):
    """Exception raised for API errors."""


class BenchmarkNetworkError(BenchmarkError):
    """Exception raised for network-related errors."""


class BenchmarkTimeoutError(BenchmarkError):
    """Exception raised for timeout errors."""


class BenchmarkAuthenticationError(BenchmarkError):
    """Exception raised for authentication errors."""


class BenchmarkStorageError(BenchmarkError):
    """Exception raised for storage-related errors."""
