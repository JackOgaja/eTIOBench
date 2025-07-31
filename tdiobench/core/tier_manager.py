#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TierManager - Storage Tier Management for the benchmark suite.

This module provides methods for managing and validating storage tiers.

Author: Jack Ogaja  
Date: 2025-06-27
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class TierManager:
    """
    Manages storage tier configurations and validation.
    """

    def __init__(self):
        """Initialize the tier manager."""
        self.tiers = {}
        
    def validate_tier(self, tier_path: str) -> bool:
        """
        Validate that a tier path is accessible and suitable for benchmarking.
        
        Args:
            tier_path: Path to the storage tier
            
        Returns:
            True if tier is valid, False otherwise
        """
        try:
            path = Path(tier_path)
            
            # Check if path exists
            if not path.exists():
                logger.warning(f"Tier path does not exist: {tier_path}")
                return False
                
            # Check if path is a directory
            if not path.is_dir():
                logger.warning(f"Tier path is not a directory: {tier_path}")
                return False
                
            # Check if we can write to the directory
            if not os.access(tier_path, os.W_OK):
                logger.warning(f"No write permission for tier path: {tier_path}")
                return False
                
            # Check available space (warn if less than 1GB)
            try:
                import shutil
                total, used, free = shutil.disk_usage(tier_path)
                free_gb = free / (1024**3)
                if free_gb < 1.0:
                    logger.warning(f"Low disk space in tier {tier_path}: {free_gb:.2f} GB free")
                    
            except Exception as e:
                logger.warning(f"Could not check disk space for {tier_path}: {e}")
                
            logger.info(f"Tier validation passed: {tier_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error validating tier {tier_path}: {e}")
            return False
            
    def add_tier(self, name: str, path: str, config: Optional[Dict] = None) -> bool:
        """
        Add a storage tier configuration.
        
        Args:
            name: Name of the tier
            path: Path to the tier storage
            config: Optional tier-specific configuration
            
        Returns:
            True if tier was added successfully
        """
        if not self.validate_tier(path):
            return False
            
        self.tiers[name] = {
            "path": path,
            "config": config or {},
            "validated": True
        }
        
        logger.info(f"Added tier '{name}' at {path}")
        return True
        
    def remove_tier(self, name: str) -> bool:
        """
        Remove a storage tier configuration.
        
        Args:
            name: Name of the tier to remove
            
        Returns:
            True if tier was removed successfully
        """
        if name in self.tiers:
            del self.tiers[name]
            logger.info(f"Removed tier '{name}'")
            return True
        return False
        
    def list_tiers(self) -> List[str]:
        """
        List all configured tier names.
        
        Returns:
            List of tier names
        """
        return list(self.tiers.keys())
        
    def get_tier_info(self, name: str) -> Optional[Dict]:
        """
        Get information about a specific tier.
        
        Args:
            name: Name of the tier
            
        Returns:
            Tier information dictionary or None if not found
        """
        return self.tiers.get(name)
        
    def get_tier_path(self, name: str) -> Optional[str]:
        """
        Get the path for a specific tier.
        
        Args:
            name: Name of the tier
            
        Returns:
            Tier path or None if not found
        """
        tier_info = self.get_tier_info(name)
        return tier_info["path"] if tier_info else None
