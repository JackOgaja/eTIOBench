#!/usr/bin/env python3
"""
Time Series Analyzer (Tiered Storage I/O Bechmark)

This module provides analysis of time series performance data, including trend detection,
seasonality analysis, and anomaly detection.

Author: Jack Ogaja
Date: 2025-06-26
"""

import logging
import os
from typing import Any, Dict, List, Optional


from tdiobench.core.benchmark_analysis import AnalysisResult
from tdiobench.core.benchmark_config import BenchmarkConfig
from tdiobench.core.benchmark_data import BenchmarkData
from tdiobench.utils.naming_conventions import standard_method_name

logger = logging.getLogger("tdiobench.analysis.time_series")


class TimeSeriesAnalyzer:
    """
    Analyzer for time series performance data.

    Provides methods for decomposing time series data, detecting trends and seasonality,
    and identifying correlations between different metrics.
    """

    def __init__(self, config: BenchmarkConfig):
        """
        Initialize time series analyzer.

        Args:
            config: Benchmark configuration
        """
        self.config = config
        self.enable_decomposition = config.get("analysis.time_series.decomposition.enabled", True)
        self.enable_trend_detection = config.get(
            "analysis.time_series.trend_detection.enabled", True
        )
        self.enable_seasonality_detection = config.get(
            "analysis.time_series.seasonality_detection.enabled", True
        )
        self.enable_correlation_analysis = config.get(
            "analysis.time_series.correlation_analysis.enabled", True
        )
        self.enable_forecasting = config.get("analysis.time_series.forecasting.enabled", False)

    @standard_method_name("analyze")
    def analyze_time_series(self, benchmark_data: BenchmarkData) -> AnalysisResult:
        """
        Analyze time series data in benchmark results.

        Args:
            benchmark_data: Benchmark data

        Returns:
            AnalysisResult containing time series analysis
        """
        logger.info("Analyzing time series data")

        # Initialize analysis result
        result = AnalysisResult(
            analysis_type="time_series", benchmark_id=benchmark_data.benchmark_id
        )

        # Check if benchmark data has time series data
        if not benchmark_data.has_time_series_data():
            logger.warning("No time series data available for analysis")
            return result

        # Track trends
        trends = {}

        # Analyze each tier
        for tier in benchmark_data.get_tiers():
            tier_result = benchmark_data.get_tier_result(tier)

            if tier_result and "time_series" in tier_result:
                # Analyze time series data for this tier
                time_series = tier_result["time_series"]
                tier_analysis = self._analyze_tier_time_series(tier, time_series)
                result.add_tier_result(tier, tier_analysis)

                # Add trends
                if "trends" in tier_analysis:
                    trends[tier] = tier_analysis["trends"]

        # Add trends to overall results
        result.add_overall_result("trends", trends)

        # Analyze correlations between tiers
        if len(benchmark_data.get_tiers()) > 1 and self.enable_correlation_analysis:
            correlations = self._analyze_tier_correlations(benchmark_data)
            result.add_overall_result("correlations", correlations)

        # Set severity based on findings
        if any(
            trend.get("trend_detected", False)
            for tier_trends in trends.values()
            for metric, trend in tier_trends.items()
            if metric in ["throughput_MBps", "iops", "latency_ms"]
        ):
            result.set_severity("medium")

        # Add recommendations based on trends
        for tier, tier_trends in trends.items():
            tier_name = os.path.basename(tier)
            for metric, trend in tier_trends.items():
                if trend.get("trend_detected", False):
                    if metric == "throughput_MBps" and trend.get("trend_direction") == "decreasing":
                        result.add_recommendation(
                            f"Decreasing throughput trend detected for {tier_name}. Consider investigating performance degradation."
                        )
                    elif metric == "latency_ms" and trend.get("trend_direction") == "increasing":
                        result.add_recommendation(
                            f"Increasing latency trend detected for {tier_name}. Consider investigating performance degradation."
                        )

        return result

    @standard_method_name("decompose")
    def decompose_time_series(
        self, time_series_data: Dict[str, Any], metric: str
    ) -> Dict[str, Any]:
        """
        Decompose time series into trend, seasonality, and residual components.

        Args:
            time_series_data: Time series data
            metric: Metric to decompose

        Returns:
            Dictionary containing decomposition results

        Raises:
            BenchmarkAnalysisError: If decomposition fails
        """

    @standard_method_name("detect")
    def detect_trend(self, time_series_data: Dict[str, Any], metric: str) -> Dict[str, Any]:
        """
        Detect trend in time series data.

        Args:
            time_series_data: Time series data
            metric: Metric to analyze

        Returns:
            Dictionary containing trend analysis results

        Raises:
            BenchmarkAnalysisError: If trend detection fails
        """

    @standard_method_name("calculate")
    def calculate_correlations(
        self, time_series_data: Dict[str, Any], metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Calculate correlations between different metrics in time series data.

        Args:
            time_series_data: Time series data
            metrics: List of metrics to analyze (default: all metrics)

        Returns:
            Dictionary containing correlation analysis results

        Raises:
            BenchmarkAnalysisError: If correlation calculation fails
        """
