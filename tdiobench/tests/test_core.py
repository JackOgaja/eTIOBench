#!/usr/bin/env python3
"""
Core Tests (Tiered Storage I/O Benchmark)

This module provides tests for the core components of the
Tiered Storage I/O Benchmark Suite.

Author: Jack Ogaja
Date: 2025-06-26
"""

import unittest
from unittest.mock import MagicMock, patch

from tdiobench.tests.test_base import BenchmarkTestCase
from tdiobench.core.benchmark_config import BenchmarkConfig
from tdiobench.core.benchmark_data import BenchmarkData, BenchmarkResult
from tdiobench.core.benchmark_analysis import AnalysisResult

class BenchmarkConfigTest(BenchmarkTestCase):
    """Tests for BenchmarkConfig."""
    
    def test_config_initialization(self):
        """Test configuration initialization."""
        config = BenchmarkConfig()
        self.assertIsNotNone(config)
    
    def test_config_get(self):
        """Test configuration get method."""
        config = BenchmarkConfig({
            "test": {
                "nested": {
                    "value": 123
                }
            }
        })
        
        self.assertEqual(config.get("test.nested.value"), 123)
        self.assertEqual(config.get("test.nested.nonexistent", "default"), "default")
    
    def test_config_set(self):
        """Test configuration set method."""
        config = BenchmarkConfig()
        
        config.set("test.nested.value", 123)
        self.assertEqual(config.get("test.nested.value"), 123)
        
        # Update value
        config.set("test.nested.value", 456)
        self.assertEqual(config.get("test.nested.value"), 456)

class BenchmarkDataTest(BenchmarkTestCase):
    """Tests for BenchmarkData."""
    
    def test_benchmark_data_initialization(self):
        """Test benchmark data initialization."""
        benchmark_data = BenchmarkData(
            benchmark_id="test_id",
            tiers=["/test/tier1", "/test/tier2"],
            duration=60,
            block_sizes=["4k", "64k"],
            patterns=["read", "write"]
        )
        
        self.assertEqual(benchmark_data.benchmark_id, "test_id")
        self.assertEqual(benchmark_data.tiers, ["/test/tier1", "/test/tier2"])
        self.assertEqual(benchmark_data.duration, 60)
        self.assertEqual(benchmark_data.block_sizes, ["4k", "64k"])
        self.assertEqual(benchmark_data.patterns, ["read", "write"])
    
    def test_add_tier_result(self):
        """Test adding tier result."""
        benchmark_data = self.test_env.create_test_benchmark_data(num_tiers=1)
        
        tier_result = {
            "name": "test_tier",
            "path": "/test/tier3",
            "tests": {},
            "summary": {
                "avg_throughput_MBps": 150.0,
                "avg_iops": 1500.0,
                "avg_latency_ms": 4.0
            }
        }
        
        benchmark_data.add_tier_result("/test/tier3", tier_result)
        
        self.assertEqual(benchmark_data.get_tier_result("/test/tier3"), tier_result)

class AnalysisResultTest(BenchmarkTestCase):
    """Tests for AnalysisResult."""
    
    def test_analysis_result_initialization(self):
        """Test analysis result initialization."""
        result = AnalysisResult(
            analysis_type="test",
            benchmark_id="test_id"
        )
        
        self.assertEqual(result.analysis_type, "test")
        self.assertEqual(result.benchmark_id, "test_id")
        self.assertEqual(result.tier_results, {})
        self.assertEqual(result.overall_results, {})
        self.assertEqual(result.recommendations, [])
    
    def test_add_tier_result(self):
        """Test adding tier result."""
        result = AnalysisResult("test", "test_id")
        
        tier_result = {
            "metric1": 123,
            "metric2": 456
        }
        
        result.add_tier_result("/test/tier1", tier_result)
        
        self.assertEqual(result.get_tier_result("/test/tier1"), tier_result)
    
    def test_add_overall_result(self):
        """Test adding overall result."""
        result = AnalysisResult("test", "test_id")
        
        result.add_overall_result("key1", "value1")
        result.add_overall_result("key2", 123)
        
        self.assertEqual(result.get_overall_result("key1"), "value1")
        self.assertEqual(result.get_overall_result("key2"), 123)
    
    def test_add_recommendation(self):
        """Test adding recommendation."""
        result = AnalysisResult("test", "test_id")
        
        result.add_recommendation("Recommendation 1")
        result.add_recommendation("Recommendation 2")
        
        self.assertEqual(len(result.recommendations), 2)
        self.assertIn("Recommendation 1", result.recommendations)
        self.assertIn("Recommendation 2", result.recommendations)
        
        # Test duplicate recommendation
        result.add_recommendation("Recommendation 1")
        self.assertEqual(len(result.recommendations), 2)  # Should not add duplicate
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = AnalysisResult("test", "test_id")
        
        result.add_tier_result("/test/tier1", {"metric": 123})
        result.add_overall_result("key", "value")
        result.add_recommendation("Recommendation")
        
        result_dict = result.to_dict()
        
        self.assertEqual(result_dict["analysis_type"], "test")
        self.assertEqual(result_dict["benchmark_id"], "test_id")
        self.assertEqual(result_dict["tier_results"]["/test/tier1"]["metric"], 123)
        self.assertEqual(result_dict["overall_results"]["key"], "value")
        self.assertEqual(result_dict["recommendations"][0], "Recommendation")
    
    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "analysis_type": "test",
            "benchmark_id": "test_id",
            "timestamp": "2025-06-29T21:05:04",
            "tier_results": {
                "/test/tier1": {"metric": 123}
            },
            "overall_results": {
                "key": "value"
            },
            "recommendations": ["Recommendation"],
            "metadata": {"meta": "data"},
            "severity": "medium"
        }
        
        result = AnalysisResult.from_dict(data)
        
        self.assertEqual(result.analysis_type, "test")
        self.assertEqual(result.benchmark_id, "test_id")
        self.assertEqual(result.timestamp, "2025-06-29T21:05:04")
        self.assertEqual(result.get_tier_result("/test/tier1")["metric"], 123)
        self.assertEqual(result.get_overall_result("key"), "value")
        self.assertEqual(result.recommendations[0], "Recommendation")
        self.assertEqual(result.metadata["meta"], "data")
        self.assertEqual(result.severity, "medium")

if __name__ == '__main__':
    unittest.main()
