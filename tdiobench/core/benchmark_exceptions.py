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
    
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class BenchmarkConfigError(BenchmarkError):
    """Exception raised for configuration errors."""
    pass


class BenchmarkExecutionError(BenchmarkError):
    """Exception raised for benchmark execution errors."""
    pass


class BenchmarkResourceError(BenchmarkError):
    """Exception raised for resource limit violations."""
    pass


class BenchmarkDataError(BenchmarkError):
    """Exception raised for data handling errors."""
    pass


class BenchmarkDataValidationError(BenchmarkError):
    """Exception raised for data handling errors."""
    pass


class BenchmarkAnalysisError(BenchmarkError):
    """Exception raised for analysis errors."""
    pass


class BenchmarkReportError(BenchmarkError):
    """Exception raised for report generation errors."""
    pass


class BenchmarkAPIError(BenchmarkError):
    """Exception raised for API errors."""
    pass


class BenchmarkNetworkError(BenchmarkError):
    """Exception raised for network-related errors."""
    pass


class BenchmarkTimeoutError(BenchmarkError):
    """Exception raised for timeout errors."""
    pass


class BenchmarkAuthenticationError(BenchmarkError):
    """Exception raised for authentication errors."""
    pass


class BenchmarkStorageError(BenchmarkError):
    """Exception raised for storage-related errors."""
    pass
