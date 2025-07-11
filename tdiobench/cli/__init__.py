#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Command Line Interface for Benchmark Suite (Tiered Storage I/O Benchmark).

This module provides the CLI functionality for running benchmarks,
analyzing results, and generating visualizations.

Author: Jack Ogaja
Date: 2025-06-26
"""

from tdiobench.cli.commandline import (
    cli, run, analyze, report, compare, list_benchmarks, validate, 
    cleanup, tier, tier_list, tier_add, tier_test, profile, 
    profile_list, profile_show, profile_validate, check_env 
)
from tdiobench.cli.profile_manager import ProfileManager
from tdiobench.cli.safety_controller import SafetyController

__version__ = "0.1.0"
__all__ = [
    'cli',
    'run',
    'analyze',
    'report',
    'compare',
    'list_benhcmarks',
    'validate',
    'cleanup', 
    'tier', 
    'tier_list', 
    'tier_add', 
    'tier_test', 
    'profile', 
    'profile_list', 
    'profile_show', 
    'profile_validate', 
    'check_env' 
]

# Module initialization
import logging
logger = logging.getLogger(__name__)
logger.debug(f"Benchmark Suite CLI module initialized (version {__version__})")
