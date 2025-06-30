#!/usr/bin/env python3
"""
Statistical Analyzer (Tiered I/O Benchmark)

This module provides statistical analysis functionality for benchmark results,
including confidence intervals, regression detection, and comparative analysis.

Author: Jack Ogaja
Date: 2025-06-26
"""

import math
import logging
import statistics
from typing import Dict, List, Any, Optional, Union, Tuple

from tdiobench.core.benchmark_config import BenchmarkConfig
from tdiobench.core.benchmark_data import BenchmarkData, BenchmarkResult
from tdiobench.core.benchmark_analysis import AnalysisResult
from tdiobench.core.benchmark_exceptions import BenchmarkAnalysisError
from tdiobench.utils.parameter_standards import standard_parameters
from tdiobench.utils.naming_conventions import standard_method_name

logger = logging.getLogger("tdiobench.analysis.statistics")

class StatisticalAnalyzer:
    """
    Statistical analyzer for benchmark results.
    
    Provides methods for statistical analysis of benchmark data, including
    confidence intervals, outlier detection, and regression analysis.
    """
    
    def __init__(self, config: BenchmarkConfig):
        """
        Initialize statistical analyzer.
        
        Args:
            config: Benchmark configuration
        """
        self.config = config
        self.confidence_level = config.get("analysis.statistics.confidence_level", 95) / 100.0
        self.outlier_detection = config.get("analysis.statistics.outlier_detection", True)
        self.percentiles = config.get("analysis.statistics.percentiles", [50, 95, 99, 99.9])
    
    @standard_method_name("analyze")
    @standard_parameters
    def analyze_dataset(self, benchmark_data: BenchmarkData) -> AnalysisResult:
        """
        Perform comprehensive statistical analysis on benchmark data.
        
        Args:
            benchmark_data: Benchmark data to analyze
            
        Returns:
            AnalysisResult containing statistical analysis results
        """
        logger.info("Performing statistical analysis of benchmark data")
        
        # Initialize analysis result
        result = AnalysisResult(
            analysis_type="statistics",
            benchmark_id=benchmark_data.benchmark_id
        )
        
        # Analyze each tier
        for tier in benchmark_data.get_tiers():
            tier_result = benchmark_data.get_tier_result(tier)
            
            if tier_result and "tests" in tier_result:
                # Analyze each test
                tier_stats = {}
                
                for test_name, test_data in tier_result["tests"].items():
                    test_stats = self._analyze_test_data(test_data)
                    tier_stats[test_name] = test_stats
                
                # Aggregate tier statistics
                tier_stats["aggregated"] = self._aggregate_tier_statistics(tier_stats)
                
                result.add_tier_result(tier, tier_stats)
        
        # Calculate overall statistics
        overall_stats = self._calculate_overall_statistics(
            {tier: result.get_tier_result(tier) for tier in benchmark_data.get_tiers() if result.get_tier_result(tier)}
        )
        
        result.add_overall_result("statistics", overall_stats)
        
        # Add recommendations based on statistics
        self._add_statistical_recommendations(result, overall_stats)
        
        return result
    
    @standard_method_name("calculate")
    @standard_parameters
    def calculate_confidence_interval(
        self,
        data: List[float],
        confidence_level: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Calculate confidence interval for a dataset.
        
        Args:
            data: List of numeric values
            confidence_level: Confidence level (0-1, default: use configured value)
            
        Returns:
            Tuple of (lower_bound, upper_bound)
            
        Raises:
            BenchmarkAnalysisError: If calculation fails
        """
        if not data:
            raise BenchmarkAnalysisError("Cannot calculate confidence interval for empty dataset")
        
        confidence = confidence_level if confidence_level is not None else self.confidence_level
        
        try:
            # Calculate mean and standard deviation
            mean = statistics.mean(data)
            stdev = statistics.stdev(data) if len(data) > 1 else 0
            
            # Calculate confidence interval
            if len(data) > 1:
                # Get z-score for confidence level (approximation)
                # For exact calculation, we would use t-distribution with (n-1) degrees of freedom
                z_scores = {
                    0.8: 1.282,
                    0.9: 1.645,
                    0.95: 1.96,
                    0.99: 2.576,
                    0.999: 3.291
                }
                
                # Get closest z-score
                z_score = min(z_scores.items(), key=lambda x: abs(x[0] - confidence))[1]
                
                # Calculate margin of error
                margin = z_score * stdev / math.sqrt(len(data))
                
                return (mean - margin, mean + margin)
            else:
                # Can't calculate interval with only one data point
                return (mean, mean)
                
        except Exception as e:
            logger.error(f"Error calculating confidence interval: {str(e)}")
            raise BenchmarkAnalysisError(f"Failed to calculate confidence interval: {str(e)}")
    
    @standard_method_name("detect")
    @standard_parameters
    def detect_outliers(
        self,
        data: List[float],
        method: str = "z_score",
        threshold: float = 3.0
    ) -> List[int]:
        """
        Detect outliers in a dataset.
        
        Args:
            data: List of numeric values
            method: Detection method ('z_score', 'iqr')
            threshold: Threshold for outlier detection
            
        Returns:
            List of indices of outliers
            
        Raises:
            BenchmarkAnalysisError: If detection fails
        """
        if not data:
            return []
        
        try:
            outliers = []
            
            if method == "z_score":
                # Z-score method
                mean = statistics.mean(data)
                stdev = statistics.stdev(data) if len(data) > 1 else 0
                
                if stdev > 0:
                    for i, value in enumerate(data):
                        z_score = abs(value - mean) / stdev
                        if z_score > threshold:
                            outliers.append(i)
            
            elif method == "iqr":
                # Interquartile range method
                sorted_data = sorted(data)
                q1 = sorted_data[int(len(sorted_data) * 0.25)]
                q3 = sorted_data[int(len(sorted_data) * 0.75)]
                iqr = q3 - q1
                
                lower_bound = q1 - threshold * iqr
                upper_bound = q3 + threshold * iqr
                
                for i, value in enumerate(data):
                    if value < lower_bound or value > upper_bound:
                        outliers.append(i)
            
            else:
                raise BenchmarkAnalysisError(f"Unsupported outlier detection method: {method}")
            
            return outliers
            
        except Exception as e:
            logger.error(f"Error detecting outliers: {str(e)}")
            raise BenchmarkAnalysisError(f"Failed to detect outliers: {str(e)}")
