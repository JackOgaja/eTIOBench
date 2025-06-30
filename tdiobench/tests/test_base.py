#!/usr/bin/env python3
"""
Base Test Case (Tiered Storage I/O Benchmark)

This module provides a base test case class for testing the
Tiered Storage I/O Benchmark Suite.

Author: Jack Ogaja
Date: 2025-06-26
"""

import unittest
import logging
from unittest.mock import MagicMock, patch

from tdiobench.tests.test_utils import TestEnvironment
from tdiobench.core.benchmark_config import BenchmarkConfig
from tdiobench.core.benchmark_suite import BenchmarkSuite

class BenchmarkTestCase(unittest.TestCase):
    """Base test case for benchmark suite tests."""
    
    def setUp(self):
        """Set up test environment."""
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        
        # Create test environment
        self.test_env = TestEnvironment()
        
        # Create test configuration
        self.config = self.test_env.create_test_config()
        
        # Create benchmark suite
        self.tdiobench = BenchmarkSuite(self.config)
    
    def tearDown(self):
        """Clean up test environment."""
        self.test_env.cleanup()
