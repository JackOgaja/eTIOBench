#!/usr/bin/env python3
"""
Test Utilities (Tiered Storage I/O Benchmark)

This module provides utility functions and classes for testing the
Tiered Storage I/O Benchmark Suite.

Author: Jack Ogaja
Date: 2025-06-26
"""

import os
import random
import shutil
import tempfile
from typing import Any, Dict, Optional

from tdiobench.core.benchmark_config import BenchmarkConfig
from tdiobench.core.benchmark_data import BenchmarkData


class BenchmarkTestEnvironment:
    """Helper class for setting up and tearing down test environments."""

    def __init__(self):
        """Initialize test environment."""
        self.temp_dirs = []
        self.test_files = []

    def create_temp_dir(self) -> str:
        """
        Create a temporary directory.

        Returns:
            Path to temporary directory
        """
        temp_dir = tempfile.mkdtemp(prefix="benchmark_test_")
        self.temp_dirs.append(temp_dir)
        return temp_dir

    def create_test_file(self, size_mb: int = 10, dir_path: Optional[str] = None) -> str:
        """
        Create a test file with random data.

        Args:
            size_mb: File size in megabytes
            dir_path: Directory to create file in (default: random temp dir)

        Returns:
            Path to test file
        """
        if dir_path is None:
            dir_path = self.create_temp_dir()

        file_path = os.path.join(dir_path, f"test_file_{size_mb}MB.dat")

        # Create file with specified size
        with open(file_path, "wb") as f:
            f.write(os.urandom(size_mb * 1024 * 1024))

        self.test_files.append(file_path)
        return file_path

    def create_test_config(self, config_dict: Optional[Dict[str, Any]] = None) -> BenchmarkConfig:
        """
        Create a test configuration.

        Args:
            config_dict: Configuration dictionary

        Returns:
            BenchmarkConfig instance
        """
        if config_dict is None:
            config_dict = {
                "benchmark_suite": {
                    "core": {
                        "safety": {"enabled": False},
                        "output": {"base_directory": "./test_results"},
                        "logging": {"level": "INFO"},
                    },
                    "tiers": {
                        "tier_definitions": [
                            {
                                "name": "test_tier",
                                "path": "/tmp/test",
                                "type": "auto",
                                "description": "Test tier",
                            }
                        ]
                    },
                    "benchmark_profiles": {
                        "test_profile": {
                            "description": "Test profile",
                            "duration_seconds": 5,
                            "block_sizes": ["4k"],
                            "patterns": ["read"],
                        }
                    },
                    "execution": {"engine": "fio"},
                    "collection": {
                        "time_series": {"enabled": True, "interval": 0.1},
                        "system_metrics": {"enabled": True, "interval": 0.5},
                    },
                    "analysis": {
                        "statistical": {"enabled": True},
                        "time_series": {"enabled": True},
                        "network": {"enabled": True},
                        "anomaly_detection": {
                            "enabled": True,
                            "method": "z_score",
                            "threshold": 3.0,
                        },
                    },
                }
            }

        return BenchmarkConfig(config_dict)

    def create_test_time_series_data(
        self,
        tier: str = "/test/tier1",
        duration: int = 10,
        interval: float = 0.1,
        trend: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create test time series data.

        Args:
            tier: Storage tier path
            duration: Duration in seconds
            interval: Collection interval in seconds
            trend: Add trend to data (None, "increasing", "decreasing")

        Returns:
            Time series data dictionary
        """
        # Calculate number of data points
        num_points = int(duration / interval)

        # Create timestamps
        start_time = 1635000000.0  # Example timestamp
        timestamps = [start_time + i * interval for i in range(num_points)]

        # Create values for each metric
        throughput_values = []
        iops_values = []
        latency_values = []

        base_throughput = 100.0
        base_iops = 1000.0
        base_latency = 5.0

        for i in range(num_points):
            # Add randomness
            throughput_random = random.uniform(-10, 10)
            iops_random = random.uniform(-100, 100)
            latency_random = random.uniform(-1, 1)

            # Add trend if specified
            trend_factor = 0.0
            if trend == "increasing":
                trend_factor = i / num_points * 50.0
            elif trend == "decreasing":
                trend_factor = -i / num_points * 50.0

            # Calculate values
            throughput = base_throughput + throughput_random + trend_factor
            iops = base_iops + iops_random + trend_factor * 10
            latency = base_latency + latency_random + (trend_factor * -0.1 if trend else 0)

            throughput_values.append(max(0, throughput))
            iops_values.append(max(0, iops))
            latency_values.append(max(0, latency))

        # Create time series data
        time_series_data = {
            "tier": tier,
            "test_id": "test_id",
            "start_time": start_time,
            "metrics": ["throughput_MBps", "iops", "latency_ms"],
            "interval": interval,
            "timestamps": timestamps,
            "data": {
                "throughput_MBps": throughput_values,
                "iops": iops_values,
                "latency_ms": latency_values,
            },
        }

        return time_series_data

    def create_test_benchmark_data(
        self, num_tiers: int = 2, duration: int = 10, include_time_series: bool = True
    ) -> BenchmarkData:
        """
        Create test benchmark data.

        Args:
            num_tiers: Number of storage tiers
            duration: Benchmark duration in seconds
            include_time_series: Include time series data

        Returns:
            BenchmarkData instance
        """
        # Create benchmark data
        tiers = [f"/test/tier{i+1}" for i in range(num_tiers)]
        benchmark_data = BenchmarkData(
            data={
                "tiers": tiers,
                "duration": duration,
                "block_sizes": ["4k", "64k", "1m"],
                "patterns": ["read", "write", "randrw"],
            },
            metadata={"benchmark_id": "test_benchmark_id"},
        )

        # Add tier results
        for i, tier in enumerate(tiers):
            tier_result = {
                "name": f"tier{i+1}",
                "path": tier,
                "tests": {},
                "summary": {
                    "avg_throughput_MBps": 100.0 + i * 50.0,
                    "avg_iops": 1000.0 + i * 500.0,
                    "avg_latency_ms": 5.0 - i * 1.0,
                },
            }

            # Add test results
            for block_size in ["4k", "64k", "1m"]:
                for pattern in ["read", "write", "randrw"]:
                    test_key = f"{pattern}_{block_size}"

                    tier_result["tests"][test_key] = {
                        "throughput_MBps": 100.0 + i * 50.0 + random.uniform(-10, 10),
                        "iops": 1000.0 + i * 500.0 + random.uniform(-100, 100),
                        "latency_ms": 5.0 - i * 1.0 + random.uniform(-1, 1),
                        "parameters": {
                            "block_size": block_size,
                            "pattern": pattern,
                            "io_depth": 32,
                            "direct": True,
                        },
                    }

            # Add time series data if requested
            if include_time_series:
                trend = None
                if i == 0:
                    trend = "decreasing"  # First tier has decreasing trend
                elif i == 1:
                    trend = "increasing"  # Second tier has increasing trend

                tier_result["time_series"] = self.create_test_time_series_data(
                    tier=tier, duration=duration, trend=trend
                )

            benchmark_data.add_tier_result(tier, tier_result)

        return benchmark_data

    def cleanup(self):
        """Clean up test environment."""
        # Remove test files
        for file_path in self.test_files:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception:
                    pass

        # Remove temporary directories
        for dir_path in self.temp_dirs:
            if os.path.exists(dir_path):
                try:
                    shutil.rmtree(dir_path)
                except Exception:
                    pass

        self.temp_dirs = []
        self.test_files = []
