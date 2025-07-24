#!/usr/bin/env python3
"""
Error Handling

This module provides standardized error handling utilities for the
Tiered Storage I/O Benchmark Suite.

Author: Jack Ogaja
Date: 2025-06-26
"""

import functools
import logging
import traceback
from typing import Callable, Optional, Type

from tdiobench.core.benchmark_exceptions import (
    BenchmarkAnalysisError,
    BenchmarkConfigError,
    BenchmarkDataError,
    BenchmarkError,
    BenchmarkExecutionError,
    BenchmarkNetworkError,
    BenchmarkReportError,
    BenchmarkResourceError,
    BenchmarkStorageError,
    BenchmarkTimeoutError,
)

logger = logging.getLogger("tdiobench.error_handling")

# Exception mapping for standard error handling
EXCEPTION_MAPPING = {
    ValueError: BenchmarkConfigError,
    TypeError: BenchmarkConfigError,
    KeyError: BenchmarkConfigError,
    AttributeError: BenchmarkConfigError,
    FileNotFoundError: BenchmarkResourceError,
    PermissionError: BenchmarkResourceError,
    OSError: BenchmarkResourceError,
    IOError: BenchmarkResourceError,
    ConnectionError: BenchmarkNetworkError,
    TimeoutError: BenchmarkTimeoutError,
}


def map_exception(
    exception: Exception, default_error_type: Type[BenchmarkError] = BenchmarkExecutionError
) -> BenchmarkError:
    """
    Map standard Python exception to benchmark exception.

    Args:
        exception: Original exception
        default_error_type: Default benchmark error type

    Returns:
        Mapped benchmark exception
    """
    # Check if exception is already a benchmark exception
    if isinstance(exception, BenchmarkError):
        return exception

    # Get exception type
    type(exception)

    # Map to benchmark exception
    for base_type, benchmark_type in EXCEPTION_MAPPING.items():
        if isinstance(exception, base_type):
            message = str(exception)
            return benchmark_type(message, cause=exception)

    # Use default error type
    message = str(exception)
    return default_error_type(message, cause=exception)


def standard_error_handler(
    error_type: Optional[Type[BenchmarkError]] = None,
    log_level: str = "error",
    include_traceback: bool = True,
) -> Callable:
    """
    Decorator for standardized error handling.

    Args:
        error_type: Default error type for unhandled exceptions
        log_level: Logging level for errors
        include_traceback: Include traceback in logs

    Returns:
        Decorator function
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Get function context
                module_name = func.__module__
                function_name = func.__qualname__

                # Log error
                log_message = f"Error in {module_name}.{function_name}: {str(e)}"

                if include_traceback:
                    log_message += f"\n{traceback.format_exc()}"

                # Log at appropriate level
                log_func = getattr(logger, log_level.lower())
                log_func(log_message)

                # Map exception to benchmark exception
                default_type = error_type or BenchmarkExecutionError
                mapped_exception = map_exception(e, default_type)

                # Add context to exception message if not already a benchmark exception
                if not isinstance(e, BenchmarkError):
                    context_message = f"Error in {module_name}.{function_name}: {str(e)}"
                    mapped_exception.message = context_message

                # Raise mapped exception
                raise mapped_exception from e

        return wrapper

    return decorator


def safe_operation(operation_type: str) -> Callable:
    """
    Decorator for safe operations with appropriate error mapping.

    Args:
        operation_type: Type of operation (config, execution, resource, data, analysis, report, network)

    Returns:
        Decorator function
    """
    # Map operation type to error type
    error_types = {
        "config": BenchmarkConfigError,
        "execution": BenchmarkExecutionError,
        "resource": BenchmarkResourceError,
        "data": BenchmarkDataError,
        "analysis": BenchmarkAnalysisError,
        "report": BenchmarkReportError,
        "network": BenchmarkNetworkError,
        "storage": BenchmarkStorageError,
    }

    error_type = error_types.get(operation_type, BenchmarkExecutionError)

    # Return standard error handler with appropriate error type
    return standard_error_handler(error_type=error_type)
