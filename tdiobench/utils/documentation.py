#!/usr/bin/env python3
"""
Documentation Utilities

This module provides utilities for standardized documentation
across the benchmark suite for Tiered I/O Benchmark.

Author: Jack Ogaja
Date: 2025-06-26
"""

import re
import inspect
import functools
from typing import Dict, List, Any, Optional, Callable

def standardize_docstring(func: Callable) -> Callable:
    """
    Standardize function docstring.
    
    Args:
        func: Function to standardize
        
    Returns:
        Function with standardized docstring
    """
    # Get original docstring
    doc = func.__doc__
    if not doc:
        return func
    
    # Get function signature
    sig = inspect.signature(func)
    
    # Standardize format
    doc = re.sub(r'\n\s+', '\n    ', doc)
    
    # Check for Args section
    if "Args:" not in doc:
        # Add Args section
        param_section = "    Args:\n"
        for param_name, param in sig.parameters.items():
            # Skip self and cls parameters
            if param_name in ('self', 'cls'):
                continue
            
            # Add parameter
            param_section += f"        {param_name}: Description of {param_name}"
            
            # Add default value if available
            if param.default is not param.empty:
                param_section += f" (default: {param.default})"
            
            param_section += "\n"
        
        # Add to docstring
        doc += "\n" + param_section
    
    # Check for Returns section
    if "Returns:" not in doc and sig.return_annotation is not inspect.Signature.empty:
        # Add Returns section
        return_section = "    Returns:\n"
        
        # Get return annotation
        return_annotation = sig.return_annotation
        if return_annotation is not None and return_annotation is not inspect._empty:
            return_type = str(return_annotation).replace('typing.', '')
            return_section += f"        {return_type}\n"
        else:
            return_section += "        Description of return value\n"
        
        # Add to docstring
        doc += "\n" + return_section
    
    # Check for Raises section
    if "Raises:" not in doc:
        # Add Raises section based on error handling decorator if available
        if hasattr(func, '__wrapped__'):
            # Function has a decorator, check if it's error handling
            if func.__name__ == 'wrapper' and 'standard_error_handler' in str(func):
                raises_section = "    Raises:\n"
                raises_section += "        BenchmarkError: If operation fails\n"
                
                # Add to docstring
                doc += "\n" + raises_section
    
    # Update function docstring
    func.__doc__ = doc
    
    return func

def generate_class_documentation(
    cls: type,
    include_attributes: bool = True,
    include_private: bool = False
) -> str:
    """
    Generate standardized documentation for a class.
    
    Args:
        cls: Class to document
        include_attributes: Include class attributes
        include_private: Include private methods and attributes
        
    Returns:
        Documentation string
    """
    # Get class docstring
    doc = cls.__doc__ or ""
    
    # Clean up docstring
    doc = re.sub(r'\n\s+', '\n    ', doc)
    
    # Add class attributes
    if include_attributes:
        attributes = []
        
        # Get instance attributes from __init__ method
        if hasattr(cls, '__init__') and callable(cls.__init__):
            init_source = inspect.getsource(cls.__init__)
            
            # Find self.attr assignments
            attr_pattern = r'self\.([a-zA-Z_][a-zA-Z0-9_]*)\s*='
            for match in re.finditer(attr_pattern, init_source):
                attr_name = match.group(1)
                
                # Skip private attributes if not requested
                if not include_private and attr_name.startswith('_'):
                    continue
                
                attributes.append(attr_name)
        
        # Add class attributes section
        if attributes:
            doc += "\n    Attributes:\n"
            for attr in attributes:
                doc += f"        {attr}: Description of {attr}\n"
    
    # Add methods section
    methods = []
    
    for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
        # Skip private methods if not requested
        if not include_private and name.startswith('_'):
            continue
        
        # Skip special methods
        if name.startswith('__') and name.endswith('__'):
            continue
        
        methods.append((name, method))
    
    if methods:
        doc += "\n    Methods:\n"
        for name, method in methods:
            # Get method signature
            sig = inspect.signature(method)
            
            # Format method signature
            params = []
            for param_name, param in sig.parameters.items():
                if param_name in ('self', 'cls'):
                    continue
                
                if param.default is param.empty:
                    params.append(param_name)
                else:
                    params.append(f"{param_name}={param.default}")
            
            method_sig = f"{name}({', '.join(params)})"
            
            # Get method docstring
            method_doc = method.__doc__
            if method_doc:
                # Extract first line of docstring
                first_line = method_doc.strip().split('\n')[0]
                doc += f"        {method_sig}: {first_line}\n"
            else:
                doc += f"        {method_sig}\n"
    
    return doc

def example_docstring_template() -> str:
    """
    Return an example docstring template.
    
    Returns:
        Example docstring template
    """
    return """
    Short description of what the method does.
    
    Longer description with more details about the functionality,
    behavior, and any important notes.
    
    Args:
        param1: Description of first parameter
        param2: Description of second parameter (default: None)
        
    Returns:
        Description of return value
        
    Raises:
        BenchmarkConfigError: If configuration is invalid
        BenchmarkExecutionError: If execution fails
        
    Example:
        >>> instance.example_method("value")
        {"result": "success"}
    """
