#!/usr/bin/env python3
"""
Data Processor (Tiered Storage I/O Benchmark)

This module provides data processing functionality for benchmark results,
including normalization, transformation, and aggregation.

Author: Jack Ogaja
Date: 2025-06-26
"""

import logging
import statistics
from typing import Dict, List, Any, Optional, Union, Tuple

from tdiobench.core.benchmark_config import BenchmarkConfig
from tdiobench.core.benchmark_data import BenchmarkData, TimeSeriesData
from tdiobench.core.benchmark_exceptions import BenchmarkDataError

logger = logging.getLogger("tdiobench.utils.data_processor")

class DataTransformer:
    """
    Transformer for benchmark data.
    
    Provides methods for transforming benchmark data into different formats
    and representations.
    """
    
    def __init__(self, config: Optional[BenchmarkConfig] = None):
        """
        Initialize data transformer.
        
        Args:
            config: Benchmark configuration (optional)
        """
        self.config = config
    
    def transform_time_series(
        self,
        time_series_data: Dict[str, Any],
        transformation: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Transform time series data.
        
        Args:
            time_series_data: Time series data
            transformation: Transformation type
            **kwargs: Additional parameters
            
        Returns:
            Transformed time series data
            
        Raises:
            BenchmarkDataError: If transformation fails
        """
        try:
            # Clone time series data
            transformed_data = {
                "tier": time_series_data.get("tier"),
                "test_id": time_series_data.get("test_id"),
                "start_time": time_series_data.get("start_time"),
                "metrics": time_series_data.get("metrics", []),
                "interval": time_series_data.get("interval", 1.0),
                "timestamps": time_series_data.get("timestamps", []).copy(),
                "data": {}
            }
            
            # Apply transformation to each metric
            for metric in time_series_data.get("metrics", []):
                if metric in time_series_data.get("data", {}):
                    values = time_series_data["data"][metric].copy()
                    
                    if transformation == "log":
                        transformed_data["data"][metric] = self._log_transform(values)
                    elif transformation == "sqrt":
                        transformed_data["data"][metric] = self._sqrt_transform(values)
                    elif transformation == "difference":
                        transformed_data["data"][metric] = self._difference_transform(values)
                    elif transformation == "percent_change":
                        transformed_data["data"][metric] = self._percent_change_transform(values)
                    elif transformation == "z_score":
                        transformed_data["data"][metric] = self._z_score_transform(values)
                    elif transformation == "min_max":
                        transformed_data["data"][metric] = self._min_max_transform(values)
                    else:
                        raise BenchmarkDataError(f"Unsupported transformation: {transformation}")
            
            return transformed_data
            
        except Exception as e:
            logger.error(f"Error transforming time series data: {str(e)}")
            raise BenchmarkDataError(f"Failed to transform time series data: {str(e)}")
    
    def transform_to_matrix(
        self,
        time_series_data: Dict[str, Any],
        metrics: Optional[List[str]] = None
    ) -> List[List[float]]:
        """
        Transform time series data to matrix format.
        
        Args:
            time_series_data: Time series data
            metrics: List of metrics to include (default: all metrics)
            
        Returns:
            Matrix of values (rows = timestamps, columns = metrics)
            
        Raises:
            BenchmarkDataError: If transformation fails
        """
        try:
            # Get available metrics
            available_metrics = list(time_series_data.get("data", {}).keys())
            
            if not metrics:
                metrics = available_metrics
            else:
                # Filter metrics to only include available ones
                metrics = [m for m in metrics if m in available_metrics]
            
            if not metrics:
                raise BenchmarkDataError("No valid metrics found for matrix transformation")
            
            # Get timestamps
            timestamps = time_series_data.get("timestamps", [])
            
            if not timestamps:
                raise BenchmarkDataError("No timestamps found for matrix transformation")
            
            # Create matrix
            matrix = []
            
            for i in range(len(timestamps)):
                row = []
                
                for metric in metrics:
                    if metric in time_series_data.get("data", {}):
                        metric_values = time_series_data["data"][metric]
                        
                        if i < len(metric_values):
                            row.append(metric_values[i] if metric_values[i] is not None else float('nan'))
                        else:
                            row.append(float('nan'))
                    else:
                        row.append(float('nan'))
                
                matrix.append(row)
            
            return matrix
            
        except Exception as e:
            logger.error(f"Error transforming time series to matrix: {str(e)}")
            raise BenchmarkDataError(f"Failed to transform time series to matrix: {str(e)}")
    
    def transform_to_pandas(
        self,
        time_series_data: Dict[str, Any],
        metrics: Optional[List[str]] = None
    ) -> Any:
        """
        Transform time series data to pandas DataFrame.
        
        Args:
            time_series_data: Time series data
            metrics: List of metrics to include (default: all metrics)
            
        Returns:
            pandas DataFrame
            
        Raises:
            BenchmarkDataError: If transformation fails
        """
        try:
            import pandas as pd
            import numpy as np
            
            # Get available metrics
            available_metrics = list(time_series_data.get("data", {}).keys())
            
            if not metrics:
                metrics = available_metrics
            else:
                # Filter metrics to only include available ones
                metrics = [m for m in metrics if m in available_metrics]
            
            if not metrics:
                raise BenchmarkDataError("No valid metrics found for pandas transformation")
            
            # Get timestamps
            timestamps = time_series_data.get("timestamps", [])
            
            if not timestamps:
                raise BenchmarkDataError("No timestamps found for pandas transformation")
            
            # Convert timestamps to datetime
            import datetime
            dates = [datetime.datetime.fromtimestamp(ts) for ts in timestamps]
            
            # Create data dictionary
            data = {}
            
            for metric in metrics:
                if metric in time_series_data.get("data", {}):
                    metric_values = time_series_data["data"][metric]
                    
                    # Ensure same length as timestamps
                    if len(metric_values) < len(timestamps):
                        metric_values = metric_values + [None] * (len(timestamps) - len(metric_values))
                    elif len(metric_values) > len(timestamps):
                        metric_values = metric_values[:len(timestamps)]
                    
                    # Convert None to NaN
                    data[metric] = [v if v is not None else np.nan for v in metric_values]
            
            # Create DataFrame
            df = pd.DataFrame(data, index=dates)
            
            return df
            
        except ImportError:
            logger.error("pandas not available for DataFrame transformation")
            raise BenchmarkDataError("pandas not available for DataFrame transformation")
        except Exception as e:
            logger.error(f"Error transforming time series to pandas DataFrame: {str(e)}")
            raise BenchmarkDataError(f"Failed to transform time series to pandas DataFrame: {str(e)}")
    
    def _log_transform(self, values: List[float]) -> List[float]:
        """
        Apply logarithmic transformation to values.
        
        Args:
            values: List of values
            
        Returns:
            Transformed values
        """
        import math
        
        transformed = []
        
        for value in values:
            if value is None:
                transformed.append(None)
            elif value > 0:
                transformed.append(math.log(value))
            else:
                # Cannot take log of non-positive values
                transformed.append(None)
        
        return transformed
    
    def _sqrt_transform(self, values: List[float]) -> List[float]:
        """
        Apply square root transformation to values.
        
        Args:
            values: List of values
            
        Returns:
            Transformed values
        """
        import math
        
        transformed = []
        
        for value in values:
            if value is None:
                transformed.append(None)
            elif value >= 0:
                transformed.append(math.sqrt(value))
            else:
                # Cannot take sqrt of negative values
                transformed.append(None)
        
        return transformed
    
    def _difference_transform(self, values: List[float]) -> List[float]:
        """
        Apply difference transformation to values.
        
        Args:
            values: List of values
            
        Returns:
            Transformed values
        """
        transformed = [None]  # First point has no difference
        
        for i in range(1, len(values)):
            if values[i] is None or values[i-1] is None:
                transformed.append(None)
            else:
                transformed.append(values[i] - values[i-1])
        
        return transformed
    
    def _percent_change_transform(self, values: List[float]) -> List[float]:
        """
        Apply percent change transformation to values.
        
        Args:
            values: List of values
            
        Returns:
            Transformed values
        """
        transformed = [None]  # First point has no percent change
        
        for i in range(1, len(values)):
            if values[i] is None or values[i-1] is None or values[i-1] == 0:
                transformed.append(None)
            else:
                transformed.append((values[i] - values[i-1]) / values[i-1] * 100)
        
        return transformed
    
    def _z_score_transform(self, values: List[float]) -> List[float]:
        """
        Apply z-score transformation to values.
        
        Args:
            values: List of values
            
        Returns:
            Transformed values
        """
        # Filter out None values
        filtered_values = [v for v in values if v is not None]
        
        if not filtered_values:
            return [None] * len(values)
        
        # Calculate mean and standard deviation
        mean = statistics.mean(filtered_values)
        stdev = statistics.stdev(filtered_values) if len(filtered_values) > 1 else 0
        
        transformed = []
        
        for value in values:
            if value is None:
                transformed.append(None)
            elif stdev > 0:
                transformed.append((value - mean) / stdev)
            else:
                transformed.append(0)  # If stdev is 0, all values are the same
        
        return transformed
    
    def _min_max_transform(self, values: List[float]) -> List[float]:
        """
        Apply min-max scaling to values (0-1 range).
        
        Args:
            values: List of values
            
        Returns:
            Transformed values
        """
        # Filter out None values
        filtered_values = [v for v in values if v is not None]
        
        if not filtered_values:
            return [None] * len(values)
        
        # Calculate min and max
        min_value = min(filtered_values)
        max_value = max(filtered_values)
        
        transformed = []
        
        for value in values:
            if value is None:
                transformed.append(None)
            elif max_value > min_value:
                transformed.append((value - min_value) / (max_value - min_value))
            else:
                transformed.append(0.5)  # If all values are the same
        
        return transformed


class DataNormalizer:
    """
    Normalizer for benchmark data.
    
    Provides methods for normalizing benchmark data to facilitate
    comparison across different metrics and scales.
    """
    
    def __init__(self, config: Optional[BenchmarkConfig] = None):
        """
        Initialize data normalizer.
        
        Args:
            config: Benchmark configuration (optional)
        """
        self.config = config
    
    def normalize_metrics(
        self,
        benchmark_result: Dict[str, Any],
        normalization_type: str = "z_score",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Normalize metrics across different tiers.
        
        Args:
            benchmark_result: Benchmark result
            normalization_type: Type of normalization
            **kwargs: Additional parameters
            
        Returns:
            Benchmark result with normalized metrics
            
        Raises:
            BenchmarkDataError: If normalization fails
        """
        try:
            # Clone benchmark result
            normalized_result = {
                "benchmark_id": benchmark_result.get("benchmark_id"),
                "timestamp": benchmark_result.get("timestamp"),
                "tiers": benchmark_result.get("tiers", []),
                "duration": benchmark_result.get("duration"),
                "tier_results": {},
                "analysis_results": benchmark_result.get("analysis_results", {}),
                "system_metrics": benchmark_result.get("system_metrics")
            }
            
            # Get tier results
            tier_results = benchmark_result.get("tier_results", {})
            
            # Collect metric values across all tiers
            metrics_values = {}
            
            for tier, tier_result in tier_results.items():
                if "summary" in tier_result:
                    summary = tier_result["summary"]
                    
                    for metric, value in summary.items():
                        if metric not in metrics_values:
                            metrics_values[metric] = []
                        
                        metrics_values[metric].append(value)
            
            # Normalize metrics
            normalized_metrics = {}
            
            for metric, values in metrics_values.items():
                if normalization_type == "z_score":
                    normalized_metrics[metric] = self._z_score_normalize(values)
                elif normalization_type == "min_max":
                    normalized_metrics[metric] = self._min_max_normalize(values)
                elif normalization_type == "percentile":
                    normalized_metrics[metric] = self._percentile_normalize(values)
                elif normalization_type == "relative":
                    baseline_tier = kwargs.get("baseline_tier")
                    normalized_metrics[metric] = self._relative_normalize(values, benchmark_result, metric, baseline_tier)
                else:
                    raise BenchmarkDataError(f"Unsupported normalization type: {normalization_type}")
            
            # Create normalized tier results
            for i, tier in enumerate(benchmark_result.get("tiers", [])):
                if tier in tier_results:
                    # Clone tier result
                    normalized_result["tier_results"][tier] = tier_results[tier].copy()
                    
                    # Add normalized summary
                    if "summary" in tier_results[tier]:
                        normalized_summary = {}
                        
                        for metric, value in tier_results[tier]["summary"].items():
                            if metric in normalized_metrics and i < len(normalized_metrics[metric]):
                                normalized_summary[f"normalized_{metric}"] = normalized_metrics[metric][i]
                        
                        normalized_result["tier_results"][tier]["normalized_summary"] = normalized_summary
            
            return normalized_result
            
        except Exception as e:
            logger.error(f"Error normalizing metrics: {str(e)}")
            raise BenchmarkDataError(f"Failed to normalize metrics: {str(e)}")
    
    def normalize_time_series(
        self,
        time_series_data: Dict[str, Any],
        normalization_type: str = "z_score"
    ) -> Dict[str, Any]:
        """
        Normalize time series data.
        
        Args:
            time_series_data: Time series data
            normalization_type: Type of normalization
            
        Returns:
            Normalized time series data
            
        Raises:
            BenchmarkDataError: If normalization fails
        """
        try:
            # Clone time series data
            normalized_data = {
                "tier": time_series_data.get("tier"),
                "test_id": time_series_data.get("test_id"),
                "start_time": time_series_data.get("start_time"),
                "metrics": time_series_data.get("metrics", []),
                "interval": time_series_data.get("interval", 1.0),
                "timestamps": time_series_data.get("timestamps", []).copy(),
                "data": {}
            }
            
            # Normalize each metric
            for metric in time_series_data.get("metrics", []):
                if metric in time_series_data.get("data", {}):
                    values = time_series_data["data"][metric]
                    
                    # Filter out None values
                    filtered_values = [v for v in values if v is not None]
                    
                    if filtered_values:
                        if normalization_type == "z_score":
                            normalized_values = self._z_score_normalize(filtered_values)
                        elif normalization_type == "min_max":
                            normalized_values = self._min_max_normalize(filtered_values)
                        elif normalization_type == "percentile":
                            normalized_values = self._percentile_normalize(filtered_values)
                        else:
                            raise BenchmarkDataError(f"Unsupported normalization type: {normalization_type}")
                        
                        # Map normalized values back to original positions
                        normalized_metric = []
                        normalized_idx = 0
                        
                        for value in values:
                            if value is None:
                                normalized_metric.append(None)
                            else:
                                normalized_metric.append(normalized_values[normalized_idx])
                                normalized_idx += 1
                        
                        normalized_data["data"][metric] = normalized_metric
            
            return normalized_data
            
        except Exception as e:
            logger.error(f"Error normalizing time series: {str(e)}")
            raise BenchmarkDataError(f"Failed to normalize time series: {str(e)}")
    
    def _z_score_normalize(self, values: List[float]) -> List[float]:
        """
        Normalize values using z-score (standard score) normalization.
        
        Args:
            values: List of values
            
        Returns:
            Normalized values
        """
        if not values:
            return []
        
        # Calculate mean and standard deviation
        mean = statistics.mean(values)
        stdev = statistics.stdev(values) if len(values) > 1 else 0
        
        # Normalize values
        if stdev > 0:
            return [(v - mean) / stdev for v in values]
        else:
            return [0] * len(values)  # If stdev is 0, all values are the same
    
    def _min_max_normalize(self, values: List[float]) -> List[float]:
        """
        Normalize values using min-max normalization (0-1 range).
        
        Args:
            values: List of values
            
        Returns:
            Normalized values
        """
        if not values:
            return []
        
        # Calculate min and max
        min_value = min(values)
        max_value = max(values)
        
        # Normalize values
        if max_value > min_value:
            return [(v - min_value) / (max_value - min_value) for v in values]
        else:
            return [0.5] * len(values)  # If all values are the same
    
    def _percentile_normalize(self, values: List[float]) -> List[float]:
        """
        Normalize values using percentile rank normalization.
        
        Args:
            values: List of values
            
        Returns:
            Normalized values
        """
        if not values:
            return []
        
        # Sort values
        sorted_values = sorted(values)
        
        # Calculate percentile ranks
        ranks = []
        
        for value in values:
            rank = sorted_values.index(value) / (len(sorted_values) - 1) if len(sorted_values) > 1 else 0.5
            ranks.append(rank)
        
        return ranks
    
    def _relative_normalize(
        self,
        values: List[float],
        benchmark_result: Dict[str, Any],
        metric: str,
        baseline_tier: Optional[str] = None
    ) -> List[float]:
        """
        Normalize values relative to a baseline tier.
        
        Args:
            values: List of values
            benchmark_result: Benchmark result
            metric: Metric name
            baseline_tier: Baseline tier (default: first tier)
            
        Returns:
            Normalized values
        """
        if not values:
            return []
        
        # Get baseline tier
        tiers = benchmark_result.get("tiers", [])
        
        if not tiers:
            return values
        
        if baseline_tier is None or baseline_tier not in tiers:
            baseline_tier = tiers[0]
        
        # Get baseline value
        baseline_idx = tiers.index(baseline_tier)
        
        if baseline_idx >= len(values) or values[baseline_idx] == 0:
            return values
        
        baseline_value = values[baseline_idx]
        
        # Normalize values relative to baseline
        return [v / baseline_value for v in values]


class DataAggregator:
    """
    Aggregator for benchmark data.
    
    Provides methods for aggregating benchmark data at different
    granularity levels.
    """
    
    def __init__(self, config: Optional[BenchmarkConfig] = None):
        """
        Initialize data aggregator.
        
        Args:
            config: Benchmark configuration (optional)
        """
        self.config = config
    
    def aggregate_time_series(
        self,
        time_series_data: Dict[str, Any],
        interval: str,
        aggregation_function: str = "mean"
    ) -> Dict[str, Any]:
        """
        Aggregate time series data to a coarser time interval.
        
        Args:
            time_series_data: Time series data
            interval: Target interval (e.g. "1m", "5m", "1h")
            aggregation_function: Aggregation function
            
        Returns:
            Aggregated time series data
            
        Raises:
            BenchmarkDataError: If aggregation fails
        """
        try:
            # Parse interval
            interval_seconds = self._parse_interval(interval)
            
            if interval_seconds <= 0:
                raise BenchmarkDataError(f"Invalid interval: {interval}")
            
            # Clone time series data
            aggregated_data = {
                "tier": time_series_data.get("tier"),
                "test_id": time_series_data.get("test_id"),
                "start_time": time_series_data.get("start_time"),
                "metrics": time_series_data.get("metrics", []),
                "interval": interval_seconds,
                "timestamps": [],
                "data": {}
            }
            
            # Get timestamps
            timestamps = time_series_data.get("timestamps", [])
            
            if not timestamps:
                return aggregated_data
            
            # Initialize data for each metric
            for metric in time_series_data.get("metrics", []):
                aggregated_data["data"][metric] = []
            
            # Calculate time buckets
            start_time = timestamps[0]
            end_time = timestamps[-1]
            
            buckets = []
            current_time = start_time
            
            while current_time <= end_time:
                buckets.append(current_time)
                current_time += interval_seconds
            
            # Assign data points to buckets
            bucket_data = {bucket: {metric: [] for metric in time_series_data.get("metrics", [])} for bucket in buckets}
            
            for i, ts in enumerate(timestamps):
                # Find appropriate bucket
                bucket_idx = int((ts - start_time) / interval_seconds)
                
                if bucket_idx < 0:
                    bucket_idx = 0
                elif bucket_idx >= len(buckets):
                    bucket_idx = len(buckets) - 1
                
                bucket = buckets[bucket_idx]
                
                # Add data for each metric to bucket
                for metric in time_series_data.get("metrics", []):
                    if metric in time_series_data.get("data", {}) and i < len(time_series_data["data"][metric]):
                        value = time_series_data["data"][metric][i]
                        
                        if value is not None:
                            bucket_data[bucket][metric].append(value)
            
            # Aggregate data in each bucket
            for bucket in buckets:
                aggregated_data["timestamps"].append(bucket)
                
                for metric in time_series_data.get("metrics", []):
                    values = bucket_data[bucket][metric]
                    
                    if values:
                        if aggregation_function == "mean":
                            aggregated_value = sum(values) / len(values)
                        elif aggregation_function == "median":
                            aggregated_value = statistics.median(values)
                        elif aggregation_function == "min":
                            aggregated_value = min(values)
                        elif aggregation_function == "max":
                            aggregated_value = max(values)
                        elif aggregation_function == "sum":
                            aggregated_value = sum(values)
                        elif aggregation_function == "count":
                            aggregated_value = len(values)
                        else:
                            raise BenchmarkDataError(f"Unsupported aggregation function: {aggregation_function}")
                    else:
                        aggregated_value = None
                    
                    aggregated_data["data"][metric].append(aggregated_value)
            
            return aggregated_data
            
        except Exception as e:
            logger.error(f"Error aggregating time series: {str(e)}")
            raise BenchmarkDataError(f"Failed to aggregate time series: {str(e)}")
    
    def aggregate_benchmark_results(
        self,
        benchmark_results: List[Dict[str, Any]],
        aggregation_function: str = "mean"
    ) -> Dict[str, Any]:
        """
        Aggregate multiple benchmark results.
        
        Args:
            benchmark_results: List of benchmark results
            aggregation_function: Aggregation function
            
        Returns:
            Aggregated benchmark result
            
        Raises:
            BenchmarkDataError: If aggregation fails
        """
        try:
            if not benchmark_results:
                raise BenchmarkDataError("No benchmark results to aggregate")
            
            # Use the first result as a template
            first_result = benchmark_results[0]
            
            # Create aggregated result
            aggregated_result = {
                "benchmark_id": f"aggregated_{first_result.get('benchmark_id', 'benchmark')}",
                "timestamp": first_result.get("timestamp"),
                "tiers": first_result.get("tiers", []),
                "duration": statistics.mean([r.get("duration", 0) for r in benchmark_results]),
                "tier_results": {},
                "analysis_results": {},
                "system_metrics": None
            }
            
            # Check if all results have the same tiers
            all_tiers_match = True
            
            for result in benchmark_results[1:]:
                if result.get("tiers") != first_result.get("tiers"):
                    all_tiers_match = False
                    break
            
            if not all_tiers_match:
                logger.warning("Not all benchmark results have the same tiers")
            
            # Aggregate tier results
            for tier in first_result.get("tiers", []):
                # Collect tier results from all benchmark results
                tier_results = []
                
                for result in benchmark_results:
                    if tier in result.get("tier_results", {}):
                        tier_results.append(result["tier_results"][tier])
                
                if tier_results:
                    # Aggregate tier results
                    aggregated_tier_result = self._aggregate_tier_results(tier_results, aggregation_function)
                    aggregated_result["tier_results"][tier] = aggregated_tier_result
            
            return aggregated_result
            
        except Exception as e:
            logger.error(f"Error aggregating benchmark results: {str(e)}")
            raise BenchmarkDataError(f"Failed to aggregate benchmark results: {str(e)}")
    
    def _parse_interval(self, interval: str) -> float:
        """
        Parse interval string to seconds.
        
        Args:
            interval: Interval string (e.g. "1m", "5m", "1h")
            
        Returns:
            Interval in seconds
        """
        # Check if interval is already a number
        try:
            return float(interval)
        except ValueError:
            pass
        
        # Parse interval string
        if not interval:
            return 0
        
        # Extract number and unit
        import re
        match = re.match(r"(\d+\.?\d*)([smhd])", interval.lower())
        
        if not match:
            raise BenchmarkDataError(f"Invalid interval format: {interval}")
        
        value = float(match.group(1))
        unit = match.group(2)
        
        # Convert to seconds
        if unit == "s":
            return value
        elif unit == "m":
            return value * 60
        elif unit == "h":
            return value * 3600
        elif unit == "d":
            return value * 86400
        else:
            raise BenchmarkDataError(f"Invalid interval unit: {unit}")
    
    def _aggregate_tier_results(
        self,
        tier_results: List[Dict[str, Any]],
        aggregation_function: str
    ) -> Dict[str, Any]:
        """
        Aggregate tier results.
        
        Args:
            tier_results: List of tier results
            aggregation_function: Aggregation function
            
        Returns:
            Aggregated tier result
        """
        # Use the first result as a template
        first_result = tier_results[0]
        
        # Create aggregated result
        aggregated_result = {
            "name": first_result.get("name"),
            "path": first_result.get("path"),
            "tests": {},
            "summary": {}
        }
        
        # Aggregate test results
        for test_name in first_result.get("tests", {}):
            # Collect test results from all tier results
            test_results = []
            
            for tier_result in tier_results:
                if test_name in tier_result.get("tests", {}):
                    test_results.append(tier_result["tests"][test_name])
            
            if test_results:
                # Aggregate test results
                aggregated_test = {}
                
                # Collect metric values
                metrics = {}
                
                for test_result in test_results:
                    for metric, value in test_result.items():
                        if isinstance(value, (int, float)) and metric != "parameters":
                            if metric not in metrics:
                                metrics[metric] = []
                            
                            metrics[metric].append(value)
                
                # Aggregate metric values
                for metric, values in metrics.items():
                    if values:
                        if aggregation_function == "mean":
                            aggregated_test[metric] = sum(values) / len(values)
                        elif aggregation_function == "median":
                            aggregated_test[metric] = statistics.median(values)
                        elif aggregation_function == "min":
                            aggregated_test[metric] = min(values)
                        elif aggregation_function == "max":
                            aggregated_test[metric] = max(values)
                        elif aggregation_function == "sum":
                            aggregated_test[metric] = sum(values)
                        elif aggregation_function == "count":
                            aggregated_test[metric] = len(values)
                
                # Add parameters from first test result
                if "parameters" in test_results[0]:
                    aggregated_test["parameters"] = test_results[0]["parameters"]
                
                aggregated_result["tests"][test_name] = aggregated_test
        
        # Aggregate summary
        summary_metrics = {}
        
        for tier_result in tier_results:
            if "summary" in tier_result:
                for metric, value in tier_result["summary"].items():
                    if isinstance(value, (int, float)):
                        if metric not in summary_metrics:
                            summary_metrics[metric] = []
                        
                        summary_metrics[metric].append(value)
        
        # Aggregate summary metrics
        for metric, values in summary_metrics.items():
            if values:
                if aggregation_function == "mean":
                    aggregated_result["summary"][metric] = sum(values) / len(values)
                elif aggregation_function == "median":
                    aggregated_result["summary"][metric] = statistics.median(values)
                elif aggregation_function == "min":
                    aggregated_result["summary"][metric] = min(values)
                elif aggregation_function == "max":
                    aggregated_result["summary"][metric] = max(values)
                elif aggregation_function == "sum":
                    aggregated_result["summary"][metric] = sum(values)
                elif aggregation_function == "count":
                    aggregated_result["summary"][metric] = len(values)
        
        return aggregated_result


class DataProcessor:
    """
    Processor for benchmark data.
    
    Provides high-level methods for processing benchmark data,
    including transformation, normalization, and aggregation.
    """
    
    def __init__(self, config: Optional[BenchmarkConfig] = None):
        """
        Initialize data processor.
        
        Args:
            config: Benchmark configuration (optional)
        """
        self.config = config
        self.transformer = DataTransformer(config)
        self.normalizer = DataNormalizer(config)
        self.aggregator = DataAggregator(config)
    
    def process_benchmark_data(
        self,
        benchmark_data: Dict[str, Any],
        operations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Process benchmark data with multiple operations.
        
        Args:
            benchmark_data: Benchmark data
            operations: List of operations to perform
            
        Returns:
            Processed benchmark data
            
        Raises:
            BenchmarkDataError: If processing fails
        """
        try:
            processed_data = benchmark_data
            
            for operation in operations:
                op_type = operation.get("type")
                
                if op_type == "transform":
                    transformation = operation.get("transformation")
                    processed_data = self.transformer.transform_time_series(processed_data, transformation)
                    
                elif op_type == "normalize":
                    normalization_type = operation.get("normalization_type", "z_score")
                    processed_data = self.normalizer.normalize_time_series(processed_data, normalization_type)
                    
                elif op_type == "aggregate":
                    interval = operation.get("interval")
                    aggregation_function = operation.get("aggregation_function", "mean")
                    processed_data = self.aggregator.aggregate_time_series(processed_data, interval, aggregation_function)
                    
                else:
                    raise BenchmarkDataError(f"Unsupported operation type: {op_type}")
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error processing benchmark data: {str(e)}")
            raise BenchmarkDataError(f"Failed to process benchmark data: {str(e)}")
    
    def create_summary_statistics(
        self,
        benchmark_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create summary statistics from benchmark result.
        
        Args:
            benchmark_result: Benchmark result
            
        Returns:
            Summary statistics
            
        Raises:
            BenchmarkDataError: If creation fails
        """
        try:
            summary = {
                "benchmark_id": benchmark_result.get("benchmark_id"),
                "timestamp": benchmark_result.get("timestamp"),
                "duration": benchmark_result.get("duration"),
                "tier_count": len(benchmark_result.get("tiers", [])),
                "tiers": {}
            }
            
            # Add tier summaries
            for tier in benchmark_result.get("tiers", []):
                tier_result = benchmark_result.get("tier_results", {}).get(tier)
                
                if tier_result and "summary" in tier_result:
                    summary["tiers"][tier] = tier_result["summary"]
            
            # Add cross-tier statistics
            metrics = ["throughput_MBps", "iops", "latency_ms"]
            cross_tier_stats = {}
            
            for metric in metrics:
                metric_key = f"avg_{metric}"
                values = [tier_summary.get(metric_key) for tier_summary in summary["tiers"].values() 
                          if metric_key in tier_summary]
                
                if values:
                    cross_tier_stats[metric] = {
                        "mean": statistics.mean(values),
                        "min": min(values),
                        "max": max(values),
                        "range": max(values) - min(values)
                    }
                    
                    if len(values) > 1:
                        cross_tier_stats[metric]["stddev"] = statistics.stdev(values)
                        cross_tier_stats[metric]["cv"] = cross_tier_stats[metric]["stddev"] / cross_tier_stats[metric]["mean"] if cross_tier_stats[metric]["mean"] > 0 else 0
            
            summary["cross_tier_stats"] = cross_tier_stats
            
            return summary
            
        except Exception as e:
            logger.error(f"Error creating summary statistics: {str(e)}")
            raise BenchmarkDataError(f"Failed to create summary statistics: {str(e)}")
