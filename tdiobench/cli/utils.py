#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLI utility functions for the benchmark suite (Tiered Storage I/O Benchmark).

This module provides helper functions for the CLI commands.

Author: Jack Ogaja
Date: 2025-06-26
"""

import os
import sys
import logging
import yaml
import json
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    Set up logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
    """
    # Convert string level to logging level
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")
    
    # Configure logging format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # Basic configuration
    if log_file:
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            
        # Configure file and console logging
        logging.basicConfig(
            level=numeric_level,
            format=log_format,
            datefmt=date_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
    else:
        # Configure console logging only
        logging.basicConfig(
            level=numeric_level,
            format=log_format,
            datefmt=date_format
        )
    
    # Suppress excessive logging from third-party libraries
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Determine file format based on extension
    if config_path.endswith(('.yaml', '.yml')):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    elif config_path.endswith('.json'):
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        raise ValueError(f"Unsupported configuration file format: {config_path}")
    
    return config


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration dictionary.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if configuration is valid, False otherwise
    """
    # Check for required sections
    if 'benchmarks' not in config:
        logger.error("Missing 'benchmarks' section in configuration")
        return False
    
    # Validate benchmark configurations
    for benchmark in config['benchmarks']:
        if 'name' not in benchmark:
            logger.error("Benchmark missing 'name' field")
            return False
        
        if 'type' not in benchmark:
            logger.error(f"Benchmark '{benchmark['name']}' missing 'type' field")
            return False
    
    # All checks passed
    return True


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.2f} hours"


def get_default_output_path(benchmark_name: str, file_type: str = "json") -> str:
    """
    Get default output path for benchmark results.
    
    Args:
        benchmark_name: Name of the benchmark
        file_type: File type (json or pickle)
        
    Returns:
        Default output path
    """
    import datetime
    
    # Create sanitized benchmark name for filename
    safe_name = benchmark_name.replace(" ", "_").lower()
    
    # Add timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create path
    return f"results/{safe_name}_{timestamp}.{file_type}"
