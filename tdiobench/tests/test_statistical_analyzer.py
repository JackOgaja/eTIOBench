#!/usr/bin/env python3
"""
Statistical Analyzer Tests (Tiered Storage I/O Benchmark)

This module provides tests for the statistical analyzer component.

Author: Jack Ogaja
Date: 2025-06-26
"""

import unittest
from unittest.mock import MagicMock, patch

from tdiobench.tests.test_base import BenchmarkTestCase
from tdiobench.analysis.statistical_analyzer import StatisticalAnalyzer
from tdiobench.core.benchmark_analysis import AnalysisResult

class StatisticalAnalyzerTest(BenchmarkTestCase):
    """Tests for StatisticalAnalyzer."""
    
    def setUp(self):
        """Set up test environment."""
        super().setUp()
        self.analyzer = StatisticalAnalyzer(self.config)
        self.benchmark_data = self.test_env.create_test_benchmark_data(
            num_tiers=2, duration=10, include_time_series=True
        )
    
    def test_analyze_dataset(self):
        """Test analysis of benchmark data."""
        result = self.analyzer.analyze_dataset(self.benchmark_data)
        
        # Check result type
        self.assertIsInstance(result, AnalysisResult)
        self.assertEqual(result.analysis_type, "statistics")
        self.assertEqual(result.benchmark_id, self.benchmark_data.benchmark_id)
        
        # Check tier results
        self.assertEqual(len(result.tier_results), len(self.benchmark_data.tiers))
        for tier in self.benchmark_data.tiers:
            self.assertIn(tier, result.tier_results)
            
            # Check tier stats
            tier_stats = result.get_tier_result(tier)
            self.assertIn("aggregated", tier_stats)
            
            # Check test stats
            for test_name in self.benchmark_data.get_tier_result(tier)["tests"]:
                self.assertIn(test_name, tier_stats)
        
        # Check overall results
        self.assertIn("statistics", result.overall_results)
    
    def test_calculate_confidence_interval(self):
        """Test calculation of confidence interval."""
        values = [10.0, 12.0, 9.0, 11.0, 10.5]
        
        lower, upper = self.analyzer.calculate_confidence_interval(values)
        
        # Check confidence interval
        self.assertLess(lower, 10.5)  # Lower bound should be less than mean
        self.assertGreater(upper, 10.5)  # Upper bound should be greater than mean
    
    def test_detect_outliers(self):
        """Test detection of outliers."""
        values = [10.0, 12.0, 9.0, 11.0, 10.5, 20.0]  # 20.0 is an outlier
        
        outliers = self.analyzer.detect_outliers(values)
        
        # Check outliers
        self.assertEqual(len(outliers), 1)
        self.assertEqual(outliers[0], 5)  # Index of the outlier (20.0)
    
    def test_compare_tiers(self):
        """Test comparison of tiers."""
        result = self.analyzer.analyze_dataset(self.benchmark_data)
        benchmark_result = MagicMock()
        benchmark_result.tiers = self.benchmark_data.tiers
        benchmark_result.get_tier_result = self.benchmark_data.get_tier_result
        
        comparison = self.analyzer.compare_tiers(benchmark_result)
        
        # Check comparison result
        self.assertIn("baseline", comparison)
        self.assertIn("tier_comparisons", comparison)
        self.assertIn("tier_rankings", comparison)
        self.assertIn("assessment", comparison)

if __name__ == '__main__':
    unittest.main()
