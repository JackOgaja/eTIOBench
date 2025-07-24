#!/usr/bin/env python3
"""
Integration Tests (Tiered Storage I/O Benchmark)

This module provides integration tests for the Tiered Storage I/O Benchmark Suite.

Author: Jack Ogaja
Date: 2025-06-26
"""

import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from tdiobench.core.benchmark_config import BenchmarkConfig
from tdiobench.core.benchmark_suite import BenchmarkSuite
from tdiobench.tests.test_base import BenchmarkTestCase


class IntegrationTest(BenchmarkTestCase):
    """Integration tests for the benchmark suite."""

    def test_end_to_end_workflow(self):
        """Test the entire benchmark workflow from execution to reporting."""
        # Mock the FIO engine to avoid actual benchmark execution
        with patch("tdiobench.engines.fio_engine.FIOEngine.execute_benchmark") as mock_execute:
            # Configure mock to return test data
            mock_execute.return_value = {
                "throughput_MBps": 100.0,
                "iops": 1000.0,
                "latency_ms": 5.0,
                "read": {"throughput_MBps": 120.0, "iops": 1200.0, "latency_ms": 4.0},
                "write": {"throughput_MBps": 80.0, "iops": 800.0, "latency_ms": 6.0},
            }

            # Create temporary output directory
            output_dir = tempfile.mkdtemp(prefix="benchmark_integration_test_")

            try:
                # Create test tiers
                tier1 = os.path.join(output_dir, "tier1")
                tier2 = os.path.join(output_dir, "tier2")
                os.makedirs(tier1, exist_ok=True)
                os.makedirs(tier2, exist_ok=True)

                # Configure benchmark suite
                config = BenchmarkConfig(
                    {
                        "benchmark_suite": {
                            "core": {
                                "safety": {"enabled": False},
                                "output": {"base_directory": output_dir},
                                "logging": {"level": "INFO"},
                            },
                            "tiers": {
                                "tier_definitions": [
                                    {"name": "tier1", "path": tier1, "type": "auto"},
                                    {"name": "tier2", "path": tier2, "type": "auto"},
                                ]
                            },
                            "benchmark_profiles": {
                                "test_profile": {
                                    "description": "Integration test profile",
                                    "duration_seconds": 10,
                                    "block_sizes": ["4k"],
                                    "patterns": ["read", "write"],
                                }
                            },
                            "execution": {"engine": "fio"},
                            "collection": {
                                "time_series": {"enabled": True, "interval": 0.1},
                                "system_metrics": {"enabled": True},
                            },
                            "analysis": {
                                "statistical": {"enabled": True},
                                "time_series": {"enabled": True},
                                "network": {"enabled": True},
                                "anomaly_detection": {"enabled": True},
                            },
                            "visualization": {"reports": {"enabled": True, "formats": ["json"]}},
                        }
                    }
                )

                suite = BenchmarkSuite(config)

                # Mock time series and system metrics collectors
                suite.time_series_collector = MagicMock()
                suite.system_metrics_collector = MagicMock()

                # Configure mocks
                suite.time_series_collector.start_collection.return_value = None
                suite.time_series_collector.stop_collection.return_value = (
                    self.test_env.create_test_time_series_data()
                )
                suite.system_metrics_collector.start_collection.return_value = None
                suite.system_metrics_collector.stop_collection.return_value = {
                    "cpu": [],
                    "memory": [],
                }
                suite.system_metrics_collector.collect_tier_metadata.return_value = {
                    "fs_type": "ext4"
                }
                suite.system_metrics_collector.get_current_cpu_usage.return_value = 10.0
                suite.system_metrics_collector.get_current_memory_usage.return_value = 20.0

                # Run benchmark
                result = suite.run_comprehensive_analysis(
                    tiers=[tier1, tier2],
                    duration=30,  # Changed from 10 to meet minimum requirement
                    block_sizes=["4k"],
                    patterns=["read"],
                    enable_all_modules=True,
                )

                # Verify result structure
                self.assertIsNotNone(result.benchmark_id)
                self.assertEqual(len(result.tiers), 2)

                # Verify analysis results are present
                self.assertIn("statistics", result.analysis_results)
                self.assertIn("time_series", result.analysis_results)
                self.assertIn("network", result.analysis_results)
                self.assertIn("anomalies", result.analysis_results)

                # Generate reports
                report_files = suite.generate_reports(
                    result, output_dir=output_dir, formats=["json"]
                )

                # Verify reports were generated
                self.assertIn("json", report_files)
                self.assertTrue(os.path.exists(report_files["json"]))

                # Read report content
                with open(report_files["json"], "r") as f:
                    report_content = json.load(f)

                # Verify report content
                self.assertEqual(report_content.get("benchmark_id"), result.benchmark_id)

            finally:
                # Clean up
                shutil.rmtree(output_dir)


if __name__ == "__main__":
    unittest.main()
