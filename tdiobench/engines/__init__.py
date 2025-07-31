"""
Benchmark execution engines for the Tiered Storage I/O Benchmark Suite.

This package contains engines for executing storage benchmarks using various
underlying tools like FIO, dd, etc.

Author: Jack Ogaja
Date: 2025-06-26
"""

from tdiobench.engines.fio_engine import FIOEngine

__all__ = ["FIOEngine"]
