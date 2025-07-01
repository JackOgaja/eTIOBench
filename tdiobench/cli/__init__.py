#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Command Line Interface for Benchmark Suite (Tiered Storage I/O Benchmark).

This module provides the CLI functionality for running benchmarks,
analyzing results, and generating visualizations.

Author: Jack Ogaja
Date: 2025-06-26
"""

from tdiobench.cli.commands import main, run_command, analyze_command, list_command
from tdiobench.cli.utils import setup_logging, load_config, validate_config

__version__ = "0.1.0"
__all__ = [
    'main',
    'run_command',
    'analyze_command',
    'list_command',
    'setup_logging',
    'load_config',
    'validate_config'
]

# Module initialization
import logging
logger = logging.getLogger(__name__)
logger.debug(f"Benchmark Suite CLI module initialized (version {__version__})")
