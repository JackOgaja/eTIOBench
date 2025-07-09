"""
Results components for the Tiered Storage I/O Benchmark Suite.

This package contains components for storing and retrieving benchmark results
using various storage backends, including file system and databases.

Author: Jack Ogaja
Date: 2025-06-26
"""

from tdiobench.results.result_store import ResultStore
from tdiobench.results.result_aggregator import ResultsAggregator

__all__ = [
  'ResultStore',
  'ResultsAggregator'
]
