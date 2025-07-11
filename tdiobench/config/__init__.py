#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration module for storage benchmark suite (Tiered Storage I/O Benchmark).

This module provides functions for loading, validating, and handling
configuration files in both JSON and YAML formats.

Author: Jack Ogaja
Date: 2025-06-26
"""

import os
import json
import logging
import yaml
from typing import Dict, Any, Optional, Union
from pathlib import Path
import jsonschema

from tdiobench.core.benchmark_exceptions import BenchmarkConfigurationError

logger = logging.getLogger(__name__)

# Default paths
DEFAULT_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_JSON = os.path.join(DEFAULT_CONFIG_DIR, "default_config.json")
DEFAULT_CONFIG_YAML = os.path.join(DEFAULT_CONFIG_DIR, "default_config.yaml")
SCHEMA_PATH = os.path.join(DEFAULT_CONFIG_DIR, "schema.json")


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from file, with fallback to default config.
    
    Args:
        config_path: Path to configuration file (JSON or YAML)
        
    Returns:
        Configuration dictionary
        
    Raises:
        ConfigurationError: If configuration file cannot be loaded or validated
    """
    # Load schema first
    schema = load_schema()
    
    # If no config path provided, use default
    if config_path is None:
        if os.path.exists(DEFAULT_CONFIG_JSON):
            config_path = DEFAULT_CONFIG_JSON
        elif os.path.exists(DEFAULT_CONFIG_YAML):
            config_path = DEFAULT_CONFIG_YAML
        else:
            raise ConfigurationError(
                f"No configuration path provided and default configurations not found at "
                f"{DEFAULT_CONFIG_JSON} or {DEFAULT_CONFIG_YAML}"
            )
    
    # Load and validate config
    try:
        config = load_config_file(config_path)
        validate_config(config, schema)
        return config
    except Exception as e:
        raise ConfigurationError(f"Failed to load configuration from {config_path}: {str(e)}")


def load_config_file(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from file, detecting format from extension.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        ConfigurationError: If file format is unsupported or file cannot be read
    """
    config_path = os.path.expanduser(config_path)
    
    if not os.path.exists(config_path):
        raise ConfigurationError(f"Configuration file not found: {config_path}")
    
    try:
        ext = os.path.splitext(config_path)[1].lower()
        
        if ext in ('.json', ''):
            with open(config_path, 'r') as f:
                return json.load(f)
        elif ext in ('.yml', '.yaml'):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            # Try to detect format from content
            with open(config_path, 'r') as f:
                content = f.read().strip()
                
            if content.startswith('{'):
                # Looks like JSON
                return json.loads(content)
            else:
                # Try YAML as fallback
                return yaml.safe_load(content)
    except json.JSONDecodeError:
        raise ConfigurationError(f"Invalid JSON in configuration file: {config_path}")
    except yaml.YAMLError:
        raise ConfigurationError(f"Invalid YAML in configuration file: {config_path}")
    except Exception as e:
        raise ConfigurationError(f"Error reading configuration file: {str(e)}")


def load_schema() -> Dict[str, Any]:
    """
    Load JSON schema for configuration validation.
    
    Returns:
        Schema dictionary
        
    Raises:
        ConfigurationError: If schema file cannot be loaded
    """
    try:
        with open(SCHEMA_PATH, 'r') as f:
            return json.load(f)
    except Exception as e:
        raise ConfigurationError(f"Failed to load schema from {SCHEMA_PATH}: {str(e)}")


def validate_config(config: Dict[str, Any], schema: Dict[str, Any]) -> None:
    """
    Validate configuration against schema.
    
    Args:
        config: Configuration dictionary
        schema: Schema dictionary
        
    Raises:
        ConfigurationError: If configuration does not match schema
    """
    try:
        jsonschema.validate(instance=config, schema=schema)
    except jsonschema.exceptions.ValidationError as e:
        raise ConfigurationError(f"Configuration validation failed: {str(e)}")


def get_config_value(config: Dict[str, Any], path: str, default: Any = None) -> Any:
    """
    Get value from configuration using dot notation path.
    
    Args:
        config: Configuration dictionary
        path: Dot notation path (e.g., "benchmark_suite.core.safety.enabled")
        default: Default value if path not found
        
    Returns:
        Configuration value or default
    """
    if not path:
        return config
    
    parts = path.split('.')
    current = config
    
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return default
    
    return current


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries, with override_config taking precedence.
    
    Args:
        base_config: Base configuration
        override_config: Override configuration
        
    Returns:
        Merged configuration dictionary
    """
    result = base_config.copy()
    
    def _merge_dicts(original: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        for key, value in override.items():
            if key in original and isinstance(original[key], dict) and isinstance(value, dict):
                original[key] = _merge_dicts(original[key], value)
            else:
                original[key] = value
        return original
    
    return _merge_dicts(result, override_config)


def save_config(config: Dict[str, Any], output_path: str, format: str = 'json') -> None:
    """
    Save configuration to file.
    
    Args:
        config: Configuration dictionary
        output_path: Path to save configuration
        format: Output format ('json' or 'yaml')
        
    Raises:
        ConfigurationError: If configuration cannot be saved
    """
    try:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        if format.lower() == 'json':
            with open(output_path, 'w') as f:
                json.dump(config, f, indent=2)
        elif format.lower() in ('yaml', 'yml'):
            with open(output_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
        else:
            raise ConfigurationError(f"Unsupported configuration format: {format}")
            
        logger.info(f"Configuration saved to {output_path}")
    except Exception as e:
        raise ConfigurationError(f"Failed to save configuration to {output_path}: {str(e)}")
