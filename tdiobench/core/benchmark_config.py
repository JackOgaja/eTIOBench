#!/usr/bin/env python3
"""
Benchmark Configuration

This module provides standardized configuration management for the
Tiered Storage I/O Benchmark Suite.

Author: Jack Ogaja
Date: 2025-06-26
"""

import os
import json
import yaml
import logging
from typing import Dict, List, Any, Optional, Union

from tdiobench.core.benchmark_exceptions import BenchmarkConfigError
from tdiobench.utils.error_handling import safe_operation

logger = logging.getLogger("tdiobench.core.config")

# Standard configuration schema
CONFIG_SCHEMA = {
    "tdiobench": {
        "core": {
            "safety": {
                "enabled": True,
                "max_cpu_percent": 90,
                "max_memory_percent": 90
            },
            "logging": {
                "level": "info",
                "file": None
            }
        },
        "collection": {
            "time_series": {
                "enabled": True,
                "interval": 1.0,
                "buffer_size": 1000,
                "db_path": None
            },
            "system_metrics": {
                "enabled": True,
                "interval": 5.0,
                "network": {
                    "enabled": True,
                    "connectivity_check": False
                }
            }
        },
        "analysis": {
            "statistics": {
                "enabled": True,
                "confidence_level": 95.0,
                "outlier_detection": True,
                "percentiles": [50, 95, 99, 99.9]
            },
            "time_series": {
                "enabled": True,
                "decomposition": {
                    "enabled": True
                },
                "trend_detection": {
                    "enabled": True
                },
                "seasonality_detection": {
                    "enabled": True
                },
                "correlation_analysis": {
                    "enabled": True
                },
                "forecasting": {
                    "enabled": False
                }
            },
            "network": {
                "enabled": True,
                "packet_capture": False,
                "detect_protocol": True,
                "interface_monitoring": True
            },
            "anomaly_detection": {
                "enabled": True,
                "method": "z_score",
                "threshold": 3.0,
                "min_data_points": 10,
                "contextual": {
                    "enabled": True
                }
            }
        },
        "visualization": {
            "reports": {
                "enabled": True,
                "formats": ["html", "json"],
                "template_dir": "./templates"
            },
            "charts": {
                "enabled": True,
                "types": ["bar", "line"],
                "output_dir": "./charts",
                "format": "html",
                "width": 800,
                "height": 400,
                "trendline": {
                    "type": "linear",
                    "forecast_horizon": 0
                }
            }
        },
        "results": {
            "type": "file",
            "base_dir": "./results",
            "db_path": None,
            "compression": False
        },
        "engines": {
            "fio": {
                "path": "fio",
                "cleanup_test_files": True
            },
            "dd": {
                "path": "dd",
                "bs_default": "1M",
                "count_default": 1000
            }
        }
    }
}

class BenchmarkConfig:
    """
    Configuration manager for benchmark suite.
    
    Provides methods for loading, validating, and accessing configuration.
    """
    
    def __init__(self, config_data: Optional[Dict[str, Any]] = None):
        """
        Initialize benchmark configuration.
        
        Args:
            config_data: Configuration data (optional)
        """
        # Initialize with default schema
        self.config = self._get_default_config()
        
        # Apply environment variables
        self._apply_environment_variables()
        
        # Apply provided config data
        if config_data:
            self._merge_config(config_data)
    
    @safe_operation("config")
    def load_from_file(self, file_path: str) -> None:
        """
        Load configuration from file.
        
        Args:
            file_path: Path to configuration file (JSON or YAML)
            
        Raises:
            BenchmarkConfigError: If loading fails
        """
        if not os.path.exists(file_path):
            raise BenchmarkConfigError(f"Configuration file not found: {file_path}")
        
        try:
            # Determine file type
            if file_path.endswith(('.yaml', '.yml')):
                with open(file_path, 'r') as f:
                    config_data = yaml.safe_load(f)
            else:
                with open(file_path, 'r') as f:
                    config_data = json.load(f)
            
            # Apply loaded config
            self._merge_config(config_data)
            
            logger.info(f"Loaded configuration from {file_path}")
            
        except (json.JSONDecodeError, yaml.YAMLError) as e:
            raise BenchmarkConfigError(f"Failed to parse configuration file: {str(e)}")
        except Exception as e:
            raise BenchmarkConfigError(f"Failed to load configuration: {str(e)}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.
        
        Args:
            key: Configuration key (dot-separated path)
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        # Split key into path components
        path = key.split('.')
        
        # Start at the root of the config
        value = self.config
        
        # Traverse path
        for component in path:
            if isinstance(value, dict) and component in value:
                value = value[component]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value.
        
        Args:
            key: Configuration key (dot-separated path)
            value: Configuration value
        """
        # Split key into path components
        path = key.split('.')
        
        # Start at the root of the config
        config = self.config
        
        # Traverse path
        for i, component in enumerate(path):
            # If we're at the last component, set the value
            if i == len(path) - 1:
                config[component] = value
            else:
                # Create nested dictionaries if they don't exist
                if component not in config or not isinstance(config[component], dict):
                    config[component] = {}
                
                config = config[component]
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of configuration
        """
        return self.config.copy()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration.
        
        Returns:
            Default configuration dictionary
        """
        return CONFIG_SCHEMA.copy()
    
    def _merge_config(self, config_data: Dict[str, Any]) -> None:
        """
        Merge configuration data.
        
        Args:
            config_data: Configuration data to merge
        """
        # Recursively merge dictionaries
        def merge_dict(target, source):
            for key, value in source.items():
                if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                    merge_dict(target[key], value)
                else:
                    target[key] = value
        
        # Clone config data to avoid modifying the input
        data = json.loads(json.dumps(config_data))
        
        # Merge into current config
        merge_dict(self.config, data)
    
    def _apply_environment_variables(self) -> None:
        """Apply configuration from environment variables."""
        # Look for environment variables with prefix BENCHMARK_
        for key, value in os.environ.items():
            if key.startswith('BENCHMARK_'):
                # Convert environment variable name to config key
                config_key = key[10:].lower().replace('__', '.')
                
                # Try to parse value as JSON
                try:
                    parsed_value = json.loads(value)
                except json.JSONDecodeError:
                    # Use string value if not valid JSON
                    parsed_value = value
                
                # Set config value
                self.set(config_key, parsed_value)
    
    def validate(self) -> List[str]:
        """
        Validate configuration.
        
        Returns:
            List of validation errors, or empty list if valid
        """
        errors = []
        
        # Check required paths
        for path in ["tdiobench.core.safety.enabled"]:
            if self.get(path) is None:
                errors.append(f"Missing required configuration: {path}")
        
        # Check value types
        type_checks = [
            ("tdiobench.core.safety.max_cpu_percent", int),
            ("tdiobench.core.safety.max_memory_percent", int),
            ("tdiobench.collection.time_series.interval", (int, float)),
            ("tdiobench.collection.system_metrics.interval", (int, float)),
            ("tdiobench.analysis.statistics.confidence_level", (int, float)),
            ("tdiobench.analysis.anomaly_detection.threshold", (int, float))
        ]
        
        for path, expected_type in type_checks:
            value = self.get(path)
            if value is not None and not isinstance(value, expected_type):
                errors.append(f"Invalid type for {path}: expected {expected_type}, got {type(value)}")
        
        # Check value ranges
        range_checks = [
            ("tdiobench.core.safety.max_cpu_percent", 0, 100),
            ("tdiobench.core.safety.max_memory_percent", 0, 100),
            ("tdiobench.collection.time_series.interval", 0.01, None),
            ("tdiobench.collection.system_metrics.interval", 0.1, None),
            ("tdiobench.analysis.statistics.confidence_level", 0, 100)
        ]
        
        for path, min_value, max_value in range_checks:
            value = self.get(path)
            if value is not None:
                if min_value is not None and value < min_value:
                    errors.append(f"Value for {path} too low: {value} < {min_value}")
                if max_value is not None and value > max_value:
                    errors.append(f"Value for {path} too high: {value} > {max_value}")
        
        return errors
