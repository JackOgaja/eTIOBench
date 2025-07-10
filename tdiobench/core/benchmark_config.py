#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark configuration module for storage benchmark suite.

This module provides the BenchmarkConfig class for storing, validating,
and managing benchmark configuration settings.

Author: Jack Ogaja
Date: 2025-06-26
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Union, Type, TYPE_CHECKING

# Import non-circular dependencies
from tdiobench.core.exceptions import ConfigurationError

# Use TYPE_CHECKING for type hints to avoid runtime imports
if TYPE_CHECKING:
    from tdiobench.core.tdiobench import BenchmarkSuite
    from tdiobench.engines.base_engine import BaseEngine

logger = logging.getLogger(__name__)


class BenchmarkConfig:
    """
    Configuration manager for benchmark execution.
    
    This class handles loading, validating, and providing access to
    benchmark configuration parameters.
    """
    
    def __init__(self, config_data: Optional[Dict[str, Any]] = None, 
                config_path: Optional[str] = None):
        """
        Initialize benchmark configuration.
        
        Args:
            config_data: Configuration dictionary (if provided directly)
            config_path: Path to configuration file (JSON or YAML)
        """
        self._config = {}
        self._validated = False
        
        if config_data:
            self._config = config_data
        elif config_path:
            self._load_from_file(config_path)
        else:
            self._load_default_config()
            
        # Validate the configuration
        self.validate()
        
        logger.debug("BenchmarkConfig initialized")
    
    def _load_from_file(self, config_path: str) -> None:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Raises:
            ConfigurationError: If file cannot be loaded
        """
        # Delayed import to avoid circular dependency
        from tdiobench.config import load_config_file
        
        try:
            self._config = load_config_file(config_path)
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration from {config_path}: {str(e)}")
    
    def _load_default_config(self) -> None:
        """
        Load default configuration.
        
        Raises:
            ConfigurationError: If default configuration cannot be loaded
        """
        # Delayed import to avoid circular dependency
        from tdiobench.config import load_config
        
        try:
            self._config = load_config(None)  # None triggers default config
            logger.info("Loaded default configuration")
        except Exception as e:
            raise ConfigurationError(f"Failed to load default configuration: {str(e)}")
    
    def validate(self) -> bool:
        """
        Validate the configuration.
        
        Returns:
            True if configuration is valid
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        # Delayed import to avoid circular dependency
        from tdiobench.config import load_schema, validate_config
        
        try:
            schema = load_schema()
            validate_config(self._config, schema)
            self._validated = True
            return True
        except Exception as e:
            raise ConfigurationError(f"Configuration validation failed: {str(e)}")
    
    def get(self, path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation path.
        
        Args:
            path: Dot notation path (e.g., "tdiobench.core.safety.enabled")
            default: Default value if path not found
            
        Returns:
            Configuration value or default
        """
        # Delayed import to avoid circular dependency
        from tdiobench.config import get_config_value
        
        return get_config_value(self._config, path, default)
    
    def get_tier_config(self, tier_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific storage tier.
        
        Args:
            tier_name: Name of the tier
            
        Returns:
            Dictionary with tier configuration
            
        Raises:
            ConfigurationError: If tier not found
        """
        tiers = self.get("benchmark_suite.tiers.tier_definitions", [])
        
        for tier in tiers:
            if tier.get("name") == tier_name:
                return tier
                
        raise ConfigurationError(f"Storage tier not found: {tier_name}")
    
    def get_profile_config(self, profile_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get configuration for a benchmark profile.
        
        Args:
            profile_name: Name of the profile (if None, use default)
            
        Returns:
            Dictionary with profile configuration
            
        Raises:
            ConfigurationError: If profile not found
        """
        if profile_name is None:
            profile_name = self.get("benchmark_suite.execution.default_profile", "quick_scan")
            
        profiles = self.get("benchmark_suite.benchmark_profiles", {})
        
        if profile_name in profiles:
            return profiles[profile_name]
                
        raise ConfigurationError(f"Benchmark profile not found: {profile_name}")
    
    def get_safety_config(self, profile_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get safety configuration, with profile-specific overrides if available.
        
        Args:
            profile_name: Name of the profile (if None, use default)
            
        Returns:
            Dictionary with safety configuration
        """
        # Get base safety config
        safety_config = self.get("benchmark_suite.core.safety", {})
        
        # Apply profile-specific overrides if available
        if profile_name is not None:
            try:
                profile_config = self.get_profile_config(profile_name)
                profile_safety = profile_config.get("safety", {})
                
                # Update with profile-specific safety settings
                if profile_safety:
                    safety_config.update(profile_safety)
                    
            except ConfigurationError:
                # If profile not found, just use base safety config
                pass
                
        return safety_config
    
    def get_all_tier_names(self) -> List[str]:
        """
        Get list of all configured tier names.
        
        Returns:
            List of tier names
        """
        tiers = self.get("benchmark_suite.tiers.tier_definitions", [])
        return [tier.get("name") for tier in tiers if "name" in tier]
    
    def get_all_profile_names(self) -> List[str]:
        """
        Get list of all configured profile names.
        
        Returns:
            List of profile names
        """
        profiles = self.get("benchmark_suite.benchmark_profiles", {})
        return list(profiles.keys())
    
    def get_engine_config(self) -> Dict[str, Any]:
        """
        Get engine configuration.
        
        Returns:
            Dictionary with engine configuration
        """
        return self.get("benchmark_suite.execution", {})
    
    def get_analysis_config(self) -> Dict[str, Any]:
        """
        Get analysis configuration.
        
        Returns:
            Dictionary with analysis configuration
        """
        return self.get("benchmark_suite.analysis", {})
    
    def get_visualization_config(self) -> Dict[str, Any]:
        """
        Get visualization configuration.
        
        Returns:
            Dictionary with visualization configuration
        """
        return self.get("benchmark_suite.visualization", {})
    
    def create_engine(self) -> 'BaseEngine':
        """
        Create and configure benchmark engine based on configuration.
        
        Returns:
            Configured engine instance
            
        Raises:
            ConfigurationError: If engine cannot be created
        """
        # Delayed import to avoid circular dependency
        from tdiobench.engines.fio_engine import FIOEngine
        
        engine_type = self.get("benchmark_suite.execution.engine", "fio")
        
        if engine_type.lower() == "fio":
            engine_path = self.get("benchmark_suite.execution.engine_path", "/usr/bin/fio")
            return FIOEngine(self, engine_path)
        else:
            raise ConfigurationError(f"Unsupported engine type: {engine_type}")
    
    def as_dict(self) -> Dict[str, Any]:
        """
        Get complete configuration as dictionary.
        
        Returns:
            Configuration dictionary
        """
        return self._config.copy()
    
    def __str__(self) -> str:
        """String representation of the configuration."""
        return f"BenchmarkConfig(validated={self._validated}, tiers={len(self.get_all_tier_names())})"
    
    def __repr__(self) -> str:
        """Detailed representation of the configuration."""
        return (f"BenchmarkConfig(validated={self._validated}, "
                f"tiers={self.get_all_tier_names()}, "
                f"profiles={self.get_all_profile_names()})")

# Functions that don't depend on class definition can go here
def create_benchmark_config(config_path: Optional[str] = None) -> BenchmarkConfig:
    """
    Create a new BenchmarkConfig instance.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        Configured BenchmarkConfig instance
    """
    return BenchmarkConfig(config_path=config_path)
