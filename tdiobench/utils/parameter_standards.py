#!/usr/bin/env python3
"""
Parameter Standards (Tiered I/O Benhcmark)

This module provides standard parameter names and utilities for
standardizing parameter naming across the benchmark suite.

Author: Jack Ogaja
Date: 2025-06-26
"""

import inspect
import functools
from typing import Dict, List, Any, Optional, Callable, get_type_hints

# Standard parameter names and descriptions
STANDARD_PARAMETERS = {
    "tier_path": "Path to storage tier",
    "tier_paths": "List of storage tier paths",
    "benchmark_id": "Unique benchmark identifier",
    "duration_seconds": "Benchmark duration in seconds",
    "block_size": "I/O block size (e.g., '4k', '1m')",
    "block_sizes": "List of I/O block sizes",
    "io_pattern": "I/O pattern (e.g., 'read', 'write', 'randrw')",
    "io_patterns": "List of I/O patterns",
    "io_depth": "I/O queue depth",
    "direct_io": "Use direct I/O (bypass cache)",
    "metrics": "List of metrics to analyze",
    "metric": "Metric name",
    "confidence_level": "Statistical confidence level (0-1)",
    "threshold": "Threshold value for detection",
    "output_path": "Output file path",
    "output_dir": "Output directory",
    "formats": "List of output formats",
    "report_title": "Report title",
    "benchmark_data": "Benchmark data",
    "benchmark_result": "Benchmark result",
    "time_series_data": "Time series data",
    "baseline_tier": "Baseline tier for comparison",
    "config": "Benchmark configuration",
    "aggregation_function": "Function for data aggregation",
    "window_size": "Window size for moving average",
    "interval": "Collection interval in seconds"
}

# Parameter type mappings
PARAMETER_TYPES = {
    "tier_path": str,
    "tier_paths": List[str],
    "benchmark_id": str,
    "duration_seconds": int,
    "block_size": str,
    "block_sizes": List[str],
    "io_pattern": str,
    "io_patterns": List[str],
    "io_depth": int,
    "direct_io": bool,
    "metrics": List[str],
    "metric": str,
    "confidence_level": float,
    "threshold": float,
    "output_path": str,
    "output_dir": str,
    "formats": List[str],
    "report_title": str,
    "aggregation_function": str,
    "window_size": int,
    "interval": float
}

# Parameter alias mappings
PARAMETER_ALIASES = {
    "tier": "tier_path",
    "tiers": "tier_paths",
    "duration": "duration_seconds",
    "block_size_list": "block_sizes",
    "pattern": "io_pattern",
    "patterns": "io_patterns",
    "queue_depth": "io_depth",
    "direct": "direct_io",
    "output": "output_path",
    "output_directory": "output_dir",
    "format_list": "formats",
    "title": "report_title",
    "window": "window_size"
}

def standardize_parameter_name(old_name: str) -> str:
    """
    Standardize parameter name according to conventions.
    
    Args:
        old_name: Original parameter name
        
    Returns:
        Standardized parameter name
    """
    # Check if name is already standardized
    if old_name in STANDARD_PARAMETERS:
        return old_name
    
    # Check if name is an alias
    if old_name in PARAMETER_ALIASES:
        return PARAMETER_ALIASES[old_name]
    
    # No standardization found, return original
    return old_name

def standardize_parameters(func: Callable) -> Callable:
    """
    Decorator to standardize parameter names in function signatures.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function with standardized parameter names
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Map aliases to standard names
        standardized_kwargs = {}
        for name, value in kwargs.items():
            standard_name = standardize_parameter_name(name)
            standardized_kwargs[standard_name] = value
        
        # Call function with standardized parameters
        return func(*args, **standardized_kwargs)
    
    return wrapper

def update_docstring_parameters(func: Callable) -> Callable:
    """
    Update function docstring with standardized parameter descriptions.
    
    Args:
        func: Function to update
        
    Returns:
        Function with updated docstring
    """
    # Get original docstring
    doc = func.__doc__
    if not doc:
        return func
    
    # Get parameter names
    sig = inspect.signature(func)
    
    # Update parameter descriptions
    updated_params = []
    for param_name in sig.parameters:
        # Skip self and cls parameters
        if param_name in ('self', 'cls'):
            continue
        
        # Standardize parameter name
        standard_name = standardize_parameter_name(param_name)
        
        # Get description from standard parameters
        description = STANDARD_PARAMETERS.get(standard_name, f"Description of {param_name}")
        
        # Add parameter to updated list
        updated_params.append(f"        {param_name}: {description}")
    
    # Replace Args section with updated parameters
    import re
    args_pattern = r'(\s+Args:\s+)([^\n]+\s+)+?(\s+Returns:|\s+Raises:|\s+Example:|\s+\w+:|\s*$)'
    replacement = r'\1' + '\n'.join(updated_params) + r'\3'
    updated_doc = re.sub(args_pattern, replacement, doc, flags=re.DOTALL)
    
    # Update function docstring
    func.__doc__ = updated_doc
    
    return func

def standard_parameters(func: Callable) -> Callable:
    """
    Combined decorator to standardize parameters and update docstring.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function with standardized parameters and updated docstring
    """
    # Apply both decorators
    return update_docstring_parameters(standardize_parameters(func))
