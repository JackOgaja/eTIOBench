#!/usr/bin/env python3
"""
Naming Conventions (Tiered Storage I/O Benchmark)

This module provides utilities for standardizing naming conventions
across the benchmark suite.

Author: JackOgaja
Date: 2025-06-29 21:05:04
"""

import re
import inspect
import functools
from typing import Dict, List, Any, Optional, Callable

# Method naming patterns
ANALYZE_PATTERN = r'^analyze_'
DETECT_PATTERN = r'^detect_'
GET_PATTERN = r'^get_'
TRANSFORM_PATTERN = r'^transform_'
CALCULATE_PATTERN = r'^calculate_'

def standardize_method_name(old_name: str, method_type: str) -> str:
    """
    Standardize method name according to conventions.
    
    Args:
        old_name: Original method name
        method_type: Method type (analyze, detect, get, transform, calculate)
        
    Returns:
        Standardized method name
    """
    # Remove any existing prefix
    name = re.sub(r'^(analyze|detect|get|transform|calculate)_', '', old_name)
    
    # Add appropriate prefix
    if method_type == "analyze":
        return f"analyze_{name}"
    elif method_type == "detect":
        return f"detect_{name}"
    elif method_type == "get":
        return f"get_{name}"
    elif method_type == "transform":
        return f"transform_{name}"
    elif method_type == "calculate":
        return f"calculate_{name}"
    else:
        return old_name

def rename_method(cls: type, old_name: str, new_name: str) -> None:
    """
    Rename a method in a class.
    
    Args:
        cls: Class to modify
        old_name: Original method name
        new_name: New method name
    """
    if hasattr(cls, old_name):
        setattr(cls, new_name, getattr(cls, old_name))
        delattr(cls, old_name)

def standardize_class_methods(cls: type, method_mappings: Dict[str, str]) -> None:
    """
    Standardize method names in a class.
    
    Args:
        cls: Class to modify
        method_mappings: Dictionary mapping old method names to method types
    """
    for old_name, method_type in method_mappings.items():
        if hasattr(cls, old_name):
            new_name = standardize_method_name(old_name, method_type)
            if old_name != new_name:
                rename_method(cls, old_name, new_name)

# Decorator for standardized method naming
def standard_method_name(method_type: str) -> Callable:
    """
    Decorator to standardize method name.
    
    Args:
        method_type: Method type (analyze, detect, get, transform, calculate)
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        # Rename function
        wrapper.__name__ = standardize_method_name(func.__name__, method_type)
        
        return wrapper
    
    return decorator
