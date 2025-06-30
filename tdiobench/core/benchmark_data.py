#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark data module for storage benchmark suite.

This module provides classes to represent, store, and process benchmark data
collected during benchmark execution. It enables flexible data manipulation,
filtering, and aggregation for analysis.

Author: Jack Ogaja
Date: 2025-06-26
"""

import logging
import json
import os
import pickle
import uuid
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any, Iterator
import pandas as pd
import numpy as np
from collections import defaultdict

from benchmark_suite.core.exceptions import DataValidationError

logger = logging.getLogger(__name__)


class BenchmarkResult:
    """
    Container for results from a single benchmark run.
    
    This class stores metrics and metadata from an individual benchmark run,
    providing methods to access, analyze, and format the results.
    """
    
    def __init__(self, run_id: str, tier_name: str, profile_name: str, 
                 start_time: Optional[Union[str, datetime]] = None,
                 end_time: Optional[Union[str, datetime]] = None,
                 metrics: Optional[Dict[str, Any]] = None,
                 parameters: Optional[Dict[str, Any]] = None,
                 raw_data: Optional[Dict[str, Any]] = None):
        """
        Initialize a new BenchmarkResult instance.
        
        Args:
            run_id: Unique identifier for this benchmark run
            tier_name: Name of the storage tier
            profile_name: Name of the benchmark profile used
            start_time: Benchmark start time
            end_time: Benchmark end time
            metrics: Dictionary of result metrics (throughput, IOPS, etc.)
            parameters: Dictionary of benchmark parameters
            raw_data: Raw output data from the benchmark engine
        """
        self.run_id = run_id
        self.tier_name = tier_name
        self.profile_name = profile_name
        
        # Set timestamps
        self.start_time = start_time
        if self.start_time is None:
            self.start_time = datetime.utcnow().isoformat()
        elif isinstance(self.start_time, datetime):
            self.start_time = self.start_time.isoformat()
            
        self.end_time = end_time
        if self.end_time is None:
            self.end_time = datetime.utcnow().isoformat()
        elif isinstance(self.end_time, datetime):
            self.end_time = self.end_time.isoformat()
        
        # Store metrics and parameters
        self.metrics = metrics or {}
        self.parameters = parameters or {}
        self.raw_data = raw_data or {}
        
        # Calculate duration
        try:
            start_dt = pd.to_datetime(self.start_time)
            end_dt = pd.to_datetime(self.end_time)
            self.duration_seconds = (end_dt - start_dt).total_seconds()
        except:
            self.duration_seconds = None
            
        logger.debug(f"Created BenchmarkResult for run {run_id} on tier {tier_name}")
    
    @property
    def throughput(self) -> Optional[float]:
        """Get the throughput value in MB/s."""
        return self.metrics.get("throughput_MBps")
    
    @property
    def iops(self) -> Optional[float]:
        """Get the IOPS value."""
        return self.metrics.get("iops")
    
    @property
    def latency(self) -> Optional[float]:
        """Get the latency value in milliseconds."""
        return self.metrics.get("latency_ms")
    
    def get_metric(self, name: str, default: Any = None) -> Any:
        """
        Get a specific metric value.
        
        Args:
            name: Metric name
            default: Default value if metric not found
            
        Returns:
            Metric value or default
        """
        return self.metrics.get(name, default)
    
    def get_parameter(self, name: str, default: Any = None) -> Any:
        """
        Get a specific parameter value.
        
        Args:
            name: Parameter name
            default: Default value if parameter not found
            
        Returns:
            Parameter value or default
        """
        return self.parameters.get(name, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert result to a dictionary.
        
        Returns:
            Dictionary representation of the result
        """
        return {
            "run_id": self.run_id,
            "tier_name": self.tier_name,
            "profile_name": self.profile_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": self.duration_seconds,
            "metrics": self.metrics,
            "parameters": self.parameters
        }
    
    def to_series(self) -> pd.Series:
        """
        Convert result to a pandas Series.
        
        Returns:
            Series representation of the result
        """
        data = self.to_dict()
        # Flatten metrics and parameters
        for k, v in self.metrics.items():
            data[f"metric_{k}"] = v
        for k, v in self.parameters.items():
            data[f"param_{k}"] = v
        # Remove nested dictionaries
        del data["metrics"]
        del data["parameters"]
        return pd.Series(data)
    
    def to_row(self) -> Dict[str, Any]:
        """
        Convert result to a flattened row format suitable for DataFrames.
        
        Returns:
            Flattened dictionary
        """
        row = {
            "run_id": self.run_id,
            "tier_name": self.tier_name,
            "profile_name": self.profile_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": self.duration_seconds
        }
        
        # Add metrics with metric_ prefix
        for k, v in self.metrics.items():
            row[f"metric_{k}"] = v
            
        # Add parameters with param_ prefix
        for k, v in self.parameters.items():
            row[f"param_{k}"] = v
            
        return row
    
    def merge_raw_data(self, raw_data: Dict[str, Any]) -> None:
        """
        Merge additional raw data into the result.
        
        Args:
            raw_data: Raw data to merge
        """
        self.raw_data.update(raw_data)
    
    def update_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Update metrics with new values.
        
        Args:
            metrics: Metrics to update
        """
        self.metrics.update(metrics)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BenchmarkResult':
        """
        Create a BenchmarkResult from a dictionary.
        
        Args:
            data: Dictionary with result data
            
        Returns:
            New BenchmarkResult instance
        """
        return cls(
            run_id=data.get("run_id", str(uuid.uuid4())),
            tier_name=data.get("tier_name", "unknown"),
            profile_name=data.get("profile_name", "unknown"),
            start_time=data.get("start_time"),
            end_time=data.get("end_time"),
            metrics=data.get("metrics", {}),
            parameters=data.get("parameters", {}),
            raw_data=data.get("raw_data", {})
        )
    
    def __str__(self) -> str:
        """String representation of the benchmark result."""
        metrics_str = ", ".join([f"{k}={v}" for k, v in self.metrics.items() 
                                if k in ["throughput_MBps", "iops", "latency_ms"]])
        return (f"BenchmarkResult(run_id={self.run_id}, tier={self.tier_name}, "
                f"profile={self.profile_name}, {metrics_str})")


class BenchmarkData:
    """
    Container for benchmark data with processing capabilities.
    
    This class stores and manages data from benchmark runs, providing
    methods to manipulate, filter, and process the data for analysis.
    It supports various data formats and operations, including time series
    handling and metric extraction.
    """
    
    def __init__(self, data: Optional[Dict[str, Any]] = None, metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a new BenchmarkData instance.
        
        Args:
            data: Optional initial data dictionary
            metadata: Optional metadata about the benchmark
        """
        self._data = data or {}
        self._metadata = metadata or {
            "created_at": datetime.utcnow().isoformat(),
            "created_by": os.environ.get("USER", "unknown"),
            "id": str(uuid.uuid4()),
            "version": "1.0.0"
        }
        self._dataframes = {}
        self._time_series = None
        self._series_index = None
        
        logger.debug(f"Initialized BenchmarkData with ID: {self._metadata.get('id')}")
    
    @property
    def id(self) -> str:
        """Get the unique identifier for this benchmark data."""
        return self._metadata.get("id", str(uuid.uuid4()))
    
    @property
    def created_at(self) -> str:
        """Get the creation timestamp."""
        return self._metadata.get("created_at", datetime.utcnow().isoformat())
    
    @property
    def created_by(self) -> str:
        """Get the creator identifier."""
        return self._metadata.get("created_by", "unknown")
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Get the complete metadata dictionary."""
        return self._metadata.copy()
    
    @property
    def data(self) -> Dict[str, Any]:
        """Get the raw data dictionary."""
        return self._data.copy()
    
    def add_metadata(self, key: str, value: Any) -> None:
        """
        Add or update a metadata entry.
        
        Args:
            key: Metadata key
            value: Metadata value
        """
        self._metadata[key] = value
        logger.debug(f"Added metadata: {key}={value}")
    
    def add_data(self, key: str, value: Any) -> None:
        """
        Add or update a data entry.
        
        Args:
            key: Data key
            value: Data value
        """
        self._data[key] = value
        # Clear cached dataframes when data changes
        self._dataframes = {}
        logger.debug(f"Added data: {key} (type: {type(value).__name__})")
    
    def add_series_data(self, timestamp: Union[str, datetime], metrics: Dict[str, Any]) -> None:
        """
        Add time series data point to the benchmark data.
        
        Args:
            timestamp: Time when data was collected
            metrics: Dictionary of metric names and values
        """
        if isinstance(timestamp, datetime):
            timestamp = timestamp.isoformat()
            
        if "time_series" not in self._data:
            self._data["time_series"] = []
            
        data_point = {"timestamp": timestamp}
        data_point.update(metrics)
        
        self._data["time_series"].append(data_point)
        
        # Clear cached time series dataframe
        self._time_series = None
        logger.debug(f"Added time series data point at {timestamp} with {len(metrics)} metrics")
    
    def add_benchmark_run(self, run_id: str, run_data: Dict[str, Any]) -> None:
        """
        Add results from a benchmark run.
        
        Args:
            run_id: Unique identifier for the run
            run_data: Data collected during the run
        """
        if "runs" not in self._data:
            self._data["runs"] = {}
            
        self._data["runs"][run_id] = run_data
        logger.debug(f"Added benchmark run: {run_id}")
    
    def add_benchmark_result(self, result: BenchmarkResult) -> None:
        """
        Add a benchmark result to the data.
        
        Args:
            result: BenchmarkResult instance
        """
        if "runs" not in self._data:
            self._data["runs"] = {}
            
        self._data["runs"][result.run_id] = result.to_dict()
        logger.debug(f"Added benchmark result: {result.run_id}")
    
    def get_benchmark_results(self) -> List[BenchmarkResult]:
        """
        Get all benchmark results.
        
        Returns:
            List of BenchmarkResult instances
        """
        if "runs" not in self._data:
            return []
            
        results = []
        for run_id, run_data in self._data["runs"].items():
            results.append(BenchmarkResult.from_dict(run_data))
            
        return results
    
    def get_benchmark_result(self, run_id: str) -> Optional[BenchmarkResult]:
        """
        Get a specific benchmark result by run ID.
        
        Args:
            run_id: Run identifier
            
        Returns:
            BenchmarkResult instance if found, otherwise None
        """
        if "runs" not in self._data or run_id not in self._data["runs"]:
            return None
            
        return BenchmarkResult.from_dict(self._data["runs"][run_id])
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert all benchmark results to a pandas DataFrame.
        
        Returns:
            DataFrame with benchmark results
        """
        results = self.get_benchmark_results()
        if not results:
            return pd.DataFrame()
            
        rows = [result.to_row() for result in results]
        return pd.DataFrame(rows)
    
    def add_engine_result(self, engine: str, operation: str, result: Dict[str, Any]) -> None:
        """
        Add results from a specific benchmark engine.
        
        Args:
            engine: Name of the benchmark engine (e.g., 'fio')
            operation: Type of operation (e.g., 'read', 'write')
            result: Results from the engine
        """
        if "engine_results" not in self._data:
            self._data["engine_results"] = {}
            
        if engine not in self._data["engine_results"]:
            self._data["engine_results"][engine] = {}
            
        self._data["engine_results"][engine][operation] = result
        logger.debug(f"Added engine result: {engine}/{operation}")
    
    def get_time_series(self, reset_index: bool = False) -> pd.DataFrame:
        """
        Get benchmark time series data as a pandas DataFrame.
        
        Args:
            reset_index: If True, reset the DataFrame index
            
        Returns:
            DataFrame containing time series data
        """
        # Return cached dataframe if available
        if self._time_series is not None:
            df = self._time_series.copy()
            return df.reset_index() if reset_index else df
            
        if "time_series" not in self._data or not self._data["time_series"]:
            logger.warning("No time series data available")
            return pd.DataFrame()
            
        # Convert to DataFrame
        df = pd.DataFrame(self._data["time_series"])
        
        # Convert timestamp to datetime if present
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            # Set timestamp as index
            df = df.set_index("timestamp")
            self._series_index = "timestamp"
        
        # Cache for future use
        self._time_series = df
        
        return df.reset_index() if reset_index else df
    
    def get_dataframe(self, key: str, default_index: Optional[str] = None) -> pd.DataFrame:
        """
        Convert a specific data key to a pandas DataFrame.
        
        Args:
            key: Data key to convert
            default_index: Column to use as index (if applicable)
            
        Returns:
            DataFrame representation of the data
        """
        # Return cached dataframe if available
        if key in self._dataframes:
            return self._dataframes[key].copy()
            
        if key not in self._data:
            logger.warning(f"Key '{key}' not found in data")
            return pd.DataFrame()
            
        data = self._data[key]
        
        # Handle different data types
        if isinstance(data, list) and data and isinstance(data[0], dict):
            # List of dictionaries can be directly converted
            df = pd.DataFrame(data)
        elif isinstance(data, dict) and data:
            # For nested dictionaries, approach depends on structure
            if all(isinstance(v, dict) for v in data.values()):
                # If values are dicts, convert to DataFrame with keys as index
                df = pd.DataFrame.from_dict(data, orient='index')
            else:
                # Otherwise, convert to single-row DataFrame
                df = pd.DataFrame([data])
        else:
            logger.warning(f"Cannot convert data at key '{key}' to DataFrame")
            return pd.DataFrame()
        
        # Set index if specified and exists
        if default_index and default_index in df.columns:
            df = df.set_index(default_index)
        
        # Cache for future use
        self._dataframes[key] = df
        
        return df.copy()
    
    def get_engine_results(self, engine: Optional[str] = None, 
                           operation: Optional[str] = None) -> Dict[str, Any]:
        """
        Get results from specific benchmark engines.
        
        Args:
            engine: Specific engine name (if None, return all)
            operation: Specific operation (if None, return all)
            
        Returns:
            Dictionary of engine results
        """
        if "engine_results" not in self._data:
            return {}
            
        results = self._data["engine_results"]
        
        if engine is not None:
            if engine not in results:
                logger.warning(f"Engine '{engine}' not found in results")
                return {}
                
            if operation is not None:
                if operation not in results[engine]:
                    logger.warning(f"Operation '{operation}' not found for engine '{engine}'")
                    return {}
                return {operation: results[engine][operation]}
            
            return results[engine]
        
        return results
    
    def get_metric(self, metric_name: str, aggregation: Optional[str] = None) -> Union[Any, Dict[str, Any]]:
        """
        Get a specific metric from the benchmark data.
        
        Args:
            metric_name: Name of the metric to retrieve
            aggregation: Optional aggregation function ('mean', 'median', 'min', 'max', etc.)
            
        Returns:
            Metric value or dictionary of aggregated values
        """
        # Check if metric is in time series data
        df = self.get_time_series()
        
        if metric_name in df.columns:
            series = df[metric_name]
            
            if aggregation:
                if aggregation == 'mean':
                    return float(series.mean())
                elif aggregation == 'median':
                    return float(series.median())
                elif aggregation == 'min':
                    return float(series.min())
                elif aggregation == 'max':
                    return float(series.max())
                elif aggregation == 'std':
                    return float(series.std())
                elif aggregation == 'all':
                    return {
                        'mean': float(series.mean()),
                        'median': float(series.median()),
                        'min': float(series.min()),
                        'max': float(series.max()),
                        'std': float(series.std())
                    }
                else:
                    logger.warning(f"Unknown aggregation method: {aggregation}")
                    return None
            
            return series.to_list()
        
        # Check if metric is in engine results
        for engine, ops in self.get_engine_results().items():
            for op, results in ops.items():
                if isinstance(results, dict) and metric_name in results:
                    return results[metric_name]
        
        # Check if metric is a direct key in data
        if metric_name in self._data:
            return self._data[metric_name]
        
        logger.warning(f"Metric '{metric_name}' not found in benchmark data")
        return None
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """
        Get all available metrics with their aggregated values.
        
        Returns:
            Dictionary of metrics and their values
        """
        metrics = {}
        
        # Get metrics from time series
        df = self.get_time_series()
        for column in df.columns:
            metrics[column] = {
                'mean': float(df[column].mean()) if pd.api.types.is_numeric_dtype(df[column]) else None,
                'min': float(df[column].min()) if pd.api.types.is_numeric_dtype(df[column]) else None,
                'max': float(df[column].max()) if pd.api.types.is_numeric_dtype(df[column]) else None,
                'type': 'time_series'
            }
        
        # Get metrics from engine results
        for engine, ops in self.get_engine_results().items():
            for op, results in ops.items():
                if isinstance(results, dict):
                    for key, value in results.items():
                        metric_name = f"{engine}.{op}.{key}"
                        metrics[metric_name] = {
                            'value': value,
                            'type': 'engine_result'
                        }
        
        return metrics
    
    def filter_time_series(self, start_time: Optional[Union[str, datetime]] = None,
                          end_time: Optional[Union[str, datetime]] = None,
                          conditions: Optional[Dict[str, Any]] = None) -> 'BenchmarkData':
        """
        Create a new BenchmarkData with filtered time series.
        
        Args:
            start_time: Start time for filtering (inclusive)
            end_time: End time for filtering (inclusive)
            conditions: Dictionary of column-value conditions
            
        Returns:
            New BenchmarkData instance with filtered data
        """
        df = self.get_time_series()
        
        if df.empty:
            logger.warning("No time series data to filter")
            return BenchmarkData(metadata=self._metadata.copy())
        
        # Convert string times to datetime if needed
        if isinstance(start_time, str):
            start_time = pd.to_datetime(start_time)
        if isinstance(end_time, str):
            end_time = pd.to_datetime(end_time)
        
        # Apply time filters if index is timestamp
        if self._series_index == "timestamp":
            if start_time is not None:
                df = df[df.index >= start_time]
            if end_time is not None:
                df = df[df.index <= end_time]
        
        # Apply column conditions
        if conditions:
            for column, value in conditions.items():
                if column in df.columns:
                    df = df[df[column] == value]
        
        # Create new BenchmarkData with filtered results
        filtered_data = BenchmarkData(metadata=self._metadata.copy())
        
        # Reset index to access timestamp column
        df_reset = df.reset_index()
        
        # Add each time point to the new data
        for _, row in df_reset.iterrows():
            metrics = row.drop("timestamp").to_dict()
            filtered_data.add_series_data(row["timestamp"], metrics)
        
        # Copy other data (but not time_series)
        for key, value in self._data.items():
            if key != "time_series":
                filtered_data.add_data(key, value)
        
        return filtered_data
    
    def aggregate_time_series(self, freq: str = '1min', 
                             aggregation: str = 'mean') -> pd.DataFrame:
        """
        Aggregate time series data by a specified frequency.
        
        Args:
            freq: Pandas frequency string ('1min', '1h', '1d', etc.)
            aggregation: Aggregation method ('mean', 'median', 'min', 'max', etc.)
            
        Returns:
            DataFrame with aggregated time series data
        """
        df = self.get_time_series()
        
        if df.empty:
            logger.warning("No time series data to aggregate")
            return pd.DataFrame()
        
        # Check if index is timestamp
        if self._series_index != "timestamp":
            logger.warning("Cannot aggregate time series without timestamp index")
            return df
        
        # Apply resampling with specified aggregation
        if aggregation == 'mean':
            return df.resample(freq).mean()
        elif aggregation == 'median':
            return df.resample(freq).median()
        elif aggregation == 'min':
            return df.resample(freq).min()
        elif aggregation == 'max':
            return df.resample(freq).max()
        elif aggregation == 'sum':
            return df.resample(freq).sum()
        elif aggregation == 'count':
            return df.resample(freq).count()
        else:
            logger.warning(f"Unknown aggregation method: {aggregation}")
            return df
    
    def calculate_statistics(self, metrics: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
        """
        Calculate descriptive statistics for specified metrics.
        
        Args:
            metrics: List of metrics to analyze (if None, use all numeric columns)
            
        Returns:
            Dictionary of statistics for each metric
        """
        df = self.get_time_series()
        
        if df.empty:
            logger.warning("No time series data for statistics")
            return {}
        
        # If no metrics specified, use all numeric columns
        if metrics is None:
            metrics = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        else:
            # Filter to metrics that exist and are numeric
            metrics = [m for m in metrics if m in df.columns and pd.api.types.is_numeric_dtype(df[m])]
        
        if not metrics:
            logger.warning("No valid numeric metrics for statistics")
            return {}
        
        statistics = {}
        
        for metric in metrics:
            series = df[metric].dropna()
            
            if len(series) < 2:
                continue
                
            # Calculate statistics
            stats_dict = {
                "count": int(len(series)),
                "mean": float(series.mean()),
                "std": float(series.std()),
                "min": float(series.min()),
                "25%": float(series.quantile(0.25)),
                "50%": float(series.median()),
                "75%": float(series.quantile(0.75)),
                "max": float(series.max()),
                "cv": float(series.std() / series.mean()) if series.mean() != 0 else float('nan')
            }
            
            # Add more percentiles
            for p in [1, 5, 95, 99, 99.9]:
                stats_dict[f"{p}%"] = float(series.quantile(p/100))
            
            statistics[metric] = stats_dict
        
        return statistics
    
    def merge(self, other: 'BenchmarkData') -> 'BenchmarkData':
        """
        Merge this BenchmarkData with another instance.
        
        Args:
            other: Another BenchmarkData instance
            
        Returns:
            New BenchmarkData instance with merged data
        """
        if not isinstance(other, BenchmarkData):
            raise TypeError("Can only merge with another BenchmarkData instance")
        
        # Create new instance with merged metadata
        merged_metadata = self._metadata.copy()
        merged_metadata.update({
            "merged_from": [self.id, other.id],
            "merged_at": datetime.utcnow().isoformat(),
            "id": str(uuid.uuid4())
        })
        
        merged = BenchmarkData(metadata=merged_metadata)
        
        # Merge time series data
        self_ts = self.get_time_series(reset_index=True)
        other_ts = other.get_time_series(reset_index=True)
        
        if not self_ts.empty and not other_ts.empty:
            # Concatenate and sort by timestamp
            merged_ts = pd.concat([self_ts, other_ts]).sort_values("timestamp")
            
            # Add merged time series
            for _, row in merged_ts.iterrows():
                timestamp = row["timestamp"]
                metrics = row.drop("timestamp").to_dict()
                merged.add_series_data(timestamp, metrics)
        elif not self_ts.empty:
            # Copy time series from self
            for _, row in self_ts.iterrows():
                timestamp = row["timestamp"]
                metrics = row.drop("timestamp").to_dict()
                merged.add_series_data(timestamp, metrics)
        elif not other_ts.empty:
            # Copy time series from other
            for _, row in other_ts.iterrows():
                timestamp = row["timestamp"]
                metrics = row.drop("timestamp").to_dict()
                merged.add_series_data(timestamp, metrics)
        
        # Merge engine results
        self_results = self.get_engine_results()
        other_results = other.get_engine_results()
        
        for engine, ops in self_results.items():
            for op, results in ops.items():
                merged.add_engine_result(engine, op, results)
        
        for engine, ops in other_results.items():
            for op, results in ops.items():
                # Skip if already added from self
                if engine in self_results and op in self_results[engine]:
                    continue
                merged.add_engine_result(engine, op, results)
        
        # Merge benchmark results
        for result in self.get_benchmark_results():
            merged.add_benchmark_result(result)
            
        for result in other.get_benchmark_results():
            # Skip if already added from self
            if "runs" in self._data and result.run_id in self._data["runs"]:
                continue
            merged.add_benchmark_result(result)
        
        # Merge other data
        for key, value in self._data.items():
            if key not in ["time_series", "engine_results", "runs"]:
                merged.add_data(key, value)
        
        for key, value in other._data.items():
            if key not in ["time_series", "engine_results", "runs"]:
                # Skip if already added from self
                if key in self._data:
                    continue
                merged.add_data(key, value)
        
        return merged
    
    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate the benchmark data for consistency and completeness.
        
        Returns:
            Tuple of (is_valid, list_of_validation_messages)
        """
        messages = []
        is_valid = True
        
        # Check metadata
        if not self._metadata:
            messages.append("Missing metadata")
            is_valid = False
        else:
            required_metadata = ["id", "created_at", "created_by"]
            for field in required_metadata:
                if field not in self._metadata:
                    messages.append(f"Missing required metadata field: {field}")
                    is_valid = False
        
        # Check data content
        if not self._data:
            messages.append("No benchmark data available")
            is_valid = False
        
        # Validate time series data if present
        if "time_series" in self._data:
            if not isinstance(self._data["time_series"], list):
                messages.append("Time series data is not a list")
                is_valid = False
            elif not self._data["time_series"]:
                messages.append("Time series data is empty")
            else:
                # Check first time series point
                first_point = self._data["time_series"][0]
                if not isinstance(first_point, dict):
                    messages.append("Time series data points must be dictionaries")
                    is_valid = False
                elif "timestamp" not in first_point:
                    messages.append("Time series data points must have timestamps")
                    is_valid = False
        
        # Validate engine results if present
        if "engine_results" in self._data:
            if not isinstance(self._data["engine_results"], dict):
                messages.append("Engine results must be a dictionary")
                is_valid = False
            else:
                for engine, ops in self._data["engine_results"].items():
                    if not isinstance(ops, dict):
                        messages.append(f"Operations for engine {engine} must be a dictionary")
                        is_valid = False
                        break
        
        # Validate benchmark runs if present
        if "runs" in self._data:
            if not isinstance(self._data["runs"], dict):
                messages.append("Benchmark runs must be a dictionary")
                is_valid = False
            else:
                for run_id, run_data in self._data["runs"].items():
                    if not isinstance(run_data, dict):
                        messages.append(f"Run data for {run_id} must be a dictionary")
                        is_valid = False
                    elif "metrics" not in run_data:
                        messages.append(f"Run data for {run_id} missing required 'metrics' field")
                        is_valid = False
        
        return is_valid, messages
    
    def to_json(self, filepath: Optional[str] = None, indent: int = 2) -> Optional[str]:
        """
        Convert benchmark data to JSON.
        
        Args:
            filepath: Optional file path to save JSON
            indent: Indentation level for JSON formatting
            
        Returns:
            JSON string if filepath is None, otherwise None
        """
        # Prepare exportable data
        export_data = {
            "metadata": self._metadata,
            "data": self._data
        }
        
        # Convert to JSON
        json_data = json.dumps(export_data, indent=indent, default=self._json_serializer)
        
        # Save to file if path provided
        if filepath:
            with open(filepath, 'w') as f:
                f.write(json_data)
            logger.info(f"Saved benchmark data to JSON file: {filepath}")
            return None
        
        return json_data
    
    def to_pickle(self, filepath: str) -> None:
        """
        Save benchmark data to a pickle file.
        
        Args:
            filepath: File path to save pickle
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f"Saved benchmark data to pickle file: {filepath}")
    
    @staticmethod
    def _json_serializer(obj: Any) -> Any:
        """
        Custom JSON serializer for handling non-serializable objects.
        
        Args:
            obj: Object to serialize
            
        Returns:
            JSON-serializable version of the object
        """
        if isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if pd.isna(obj):
            return None
        
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    @classmethod
    def from_json(cls, json_data: Union[str, Dict], filepath: Optional[str] = None) -> 'BenchmarkData':
        """
        Create BenchmarkData from JSON.
        
        Args:
            json_data: JSON string or parsed dictionary
            filepath: Optional file path to load JSON from
            
        Returns:
            New BenchmarkData instance
        """
        if filepath:
            with open(filepath, 'r') as f:
                data = json.load(f)
            logger.info(f"Loaded benchmark data from JSON file: {filepath}")
        elif isinstance(json_data, str):
            data = json.loads(json_data)
        else:
            data = json_data
        
        metadata = data.get("metadata", {})
        benchmark_data = data.get("data", {})
        
        return cls(data=benchmark_data, metadata=metadata)
    
    @classmethod
    def from_pickle(cls, filepath: str) -> 'BenchmarkData':
        """
        Load BenchmarkData from a pickle file.
        
        Args:
            filepath: File path to load pickle from
            
        Returns:
            Loaded BenchmarkData instance
        """
        with open(filepath, 'rb') as f:
            instance = pickle.load(f)
            
        if not isinstance(instance, cls):
            raise TypeError(f"Loaded object is not a {cls.__name__} instance")
            
        logger.info(f"Loaded benchmark data from pickle file: {filepath}")
        return instance
    
    def __repr__(self) -> str:
        """String representation of the BenchmarkData instance."""
        ts_count = len(self._data.get("time_series", []))
        engines = list(self._data.get("engine_results", {}).keys())
        return (f"BenchmarkData(id={self.id}, created_at={self.created_at}, "
                f"time_series_points={ts_count}, engines={engines})")
    
    def __eq__(self, other: object) -> bool:
        """Check if two BenchmarkData instances are equal."""
        if not isinstance(other, BenchmarkData):
            return False
        
        # Compare metadata (except id which will be different)
        self_meta = self._metadata.copy()
        other_meta = other._metadata.copy()
        
        # Remove comparison-irrelevant fields
        for meta in [self_meta, other_meta]:
            for field in ["id", "created_at"]:
                if field in meta:
                    del meta[field]
        
        # Compare data content
        return self_meta == other_meta and self._data == other._data
