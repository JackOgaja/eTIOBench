#!/usr/bin/env python3
"""
Time Series Collector (Tiered Storage I/O Benchmark)

This module provides classes for collecting time series performance data
during benchmark execution.

Author: Jack Ogaja
Date: 2025-06-26
"""

import time
import threading
import logging
import sqlite3
from typing import Dict, List, Any, Optional, Union

from tdiobench.core.benchmark_config import BenchmarkConfig
from tdiobench.core.benchmark_data import TimeSeriesData
from tdiobench.core.benchmark_exceptions import BenchmarkDataError

logger = logging.getLogger("tdiobench.time_series")

class TimeSeriesConfig:
    """Configuration for time series collection."""
    
    def __init__(
        self,
        interval: float = 1.0,
        buffer_size: int = 1000,
        db_path: Optional[str] = None,
        metrics: Optional[List[str]] = None
    ):
        """
        Initialize time series configuration.
        
        Args:
            interval: Collection interval in seconds
            buffer_size: Maximum buffer size before flush
            db_path: Path to SQLite database (optional)
            metrics: List of metrics to collect
        """
        self.interval = interval
        self.buffer_size = buffer_size
        self.db_path = db_path
        self.metrics = metrics or ["throughput_MBps", "iops", "latency_ms"]
    
    @classmethod
    def from_benchmark_config(cls, config: BenchmarkConfig) -> 'TimeSeriesConfig':
        """
        Create time series configuration from benchmark configuration.
        
        Args:
            config: Benchmark configuration
            
        Returns:
            TimeSeriesConfig instance
        """
        return cls(
            interval=config.get("collection.time_series.interval", 1.0),
            buffer_size=config.get("collection.time_series.buffer_size", 1000),
            db_path=config.get("collection.time_series.db_path"),
            metrics=config.get("collection.time_series.metrics")
        )


class TimeSeriesBuffer:
    """In-memory buffer for time series data."""
    
    def __init__(self, buffer_size: int = 1000):
        """
        Initialize time series buffer.
        
        Args:
            buffer_size: Maximum buffer size before flush
        """
        self.buffer_size = buffer_size
        self.timestamps: List[float] = []
        self.values: Dict[str, List[float]] = {}
        self.count = 0
    
    def add_data_point(self, timestamp: float, data_point: Dict[str, float]) -> bool:
        """
        Add a data point to the buffer.
        
        Args:
            timestamp: Data point timestamp
            data_point: Dictionary mapping metric names to values
            
        Returns:
            True if buffer is full, False otherwise
        """
        self.timestamps.append(timestamp)
        
        for metric, value in data_point.items():
            if metric not in self.values:
                self.values[metric] = []
            self.values[metric].append(value)
        
        self.count += 1
        
        return self.count >= self.buffer_size
    
    def clear(self) -> None:
        """Clear buffer contents."""
        self.timestamps = []
        self.values = {}
        self.count = 0
    
    def is_empty(self) -> bool:
        """
        Check if buffer is empty.
        
        Returns:
            True if buffer is empty, False otherwise
        """
        return self.count == 0
    
    def get_data(self) -> tuple:
        """
        Get buffer data.
        
        Returns:
            Tuple of (timestamps, values)
        """
        return (self.timestamps, self.values)


class TimeSeriesCollector:
    """
    Collector for time series performance data.
    
    Collects performance metrics at regular intervals during benchmark execution.
    """
    
    def __init__(self, config: Union[BenchmarkConfig, TimeSeriesConfig]):
        """
        Initialize time series collector.
        
        Args:
            config: Configuration (BenchmarkConfig or TimeSeriesConfig)
        """
        # Handle different config types
        if isinstance(config, BenchmarkConfig):
            self.config = TimeSeriesConfig.from_benchmark_config(config)
        else:
            self.config = config
        
        # Initialize state
        self.buffer = TimeSeriesBuffer(self.config.buffer_size)
        self.time_series_data: Optional[TimeSeriesData] = None
        self.collection_active = False
        self.collection_thread: Optional[threading.Thread] = None
        self.db_connection: Optional[sqlite3.Connection] = None
        
        # Initialize database if path provided
        if self.config.db_path:
            self._init_database()
    
    def start_collection(self, tier: str, test_id: str) -> None:
        """
        Start time series data collection.
        
        Args:
            tier: Storage tier path
            test_id: Test identifier
            
        Raises:
            BenchmarkDataError: If collection is already active
        """
        if self.collection_active:
            raise BenchmarkDataError("Time series collection is already active")
        
        # Initialize time series data
        start_time = time.time()
        self.time_series_data = TimeSeriesData(
            tier=tier,
            test_id=test_id,
            start_time=start_time,
            metrics=self.config.metrics,
            interval=self.config.interval
        )
        
        # Start collection thread
        self.collection_active = True
        self.collection_thread = threading.Thread(
            target=self._collection_loop,
            args=(tier, test_id),
            daemon=True
        )
        self.collection_thread.start()
        
        logger.info(f"Started time series collection for {tier} (test {test_id})")
    
    def stop_collection(self) -> TimeSeriesData:
        """
        Stop time series data collection.
        
        Returns:
            Collected time series data
            
        Raises:
            BenchmarkDataError: If collection is not active
        """
        if not self.collection_active:
            raise BenchmarkDataError("Time series collection is not active")
        
        # Stop collection thread
        self.collection_active = False
        if self.collection_thread:
            self.collection_thread.join(timeout=3.0)
        
        # Flush buffer if not empty
        if not self.buffer.is_empty():
            self._flush_buffer()
        
        # Close database connection if open
        if self.db_connection:
            self.db_connection.close()
            self.db_connection = None
        
        logger.info(f"Stopped time series collection for {self.time_series_data.tier}")
        
        # Return collected data
        return self.time_series_data
    
    def get_collection_summary(self) -> Dict[str, Any]:
        """
        Get summary of collected time series data.
        
        Returns:
            Dictionary containing summary information
            
        Raises:
            BenchmarkDataError: If no data has been collected
        """
        if not self.time_series_data:
            raise BenchmarkDataError("No time series data has been collected")
        
        # Calculate summary statistics
        summary = {
            "tier": self.time_series_data.tier,
            "test_id": self.time_series_data.test_id,
            "start_time": self.time_series_data.start_time,
            "data_points": len(self.time_series_data.timestamps),
            "time_range": self.time_series_data.get_time_range(),
            "metrics": {}
        }
        
        # Calculate statistics for each metric
        for metric in self.time_series_data.metrics:
            values = self.time_series_data.get_data_points(metric)
            
            if values:
                filtered_values = [v for v in values if v is not None]
                
                if filtered_values:
                    summary["metrics"][metric] = {
                        "min": min(filtered_values),
                        "max": max(filtered_values),
                        "avg": sum(filtered_values) / len(filtered_values)
                    }
        
        return summary
    
    def export_data(self, format: str = "json", output_path: Optional[str] = None) -> Optional[str]:
        """
        Export time series data to file.
        
        Args:
            format: Export format (json, csv)
            output_path: Output file path
            
        Returns:
            Path to exported data file, or None if no output path provided
            
        Raises:
            BenchmarkDataError: If no data has been collected or format is unsupported
        """
        if not self.time_series_data:
            raise BenchmarkDataError("No time series data has been collected")
        
        if format.lower() == "json":
            import json
            
            data = self.time_series_data.to_dict()
            
            if output_path:
                with open(output_path, 'w') as f:
                    json.dump(data, f, indent=2)
                return output_path
            else:
                return None
                
        elif format.lower() == "csv":
            import csv
            
            if output_path:
                with open(output_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    
                    # Write header
                    header = ["timestamp"] + self.time_series_data.metrics
                    writer.writerow(header)
                    
                    # Write data
                    for i, timestamp in enumerate(self.time_series_data.timestamps):
                        row = [timestamp]
                        for metric in self.time_series_data.metrics:
                            row.append(self.time_series_data.data[metric][i])
                        writer.writerow(row)
                
                return output_path
            else:
                return None
        else:
            raise BenchmarkDataError(f"Unsupported export format: {format}")
    
    def _collection_loop(self, tier: str, test_id: str) -> None:
        """
        Main collection loop.
        
        Args:
            tier: Storage tier path
            test_id: Test identifier
        """
        last_collection = time.time()
        
        while self.collection_active:
            # Calculate sleep time to maintain consistent interval
            current_time = time.time()
            elapsed = current_time - last_collection
            sleep_time = max(0.0, self.config.interval - elapsed)
            
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            # Collect data point
            collection_time = time.time()
            try:
                data_point = self._collect_data_point(tier)
                
                # Add to time series data
                if self.time_series_data:
                    self.time_series_data.add_data_point(collection_time, data_point)
                
                # Add to buffer
                buffer_full = self.buffer.add_data_point(collection_time, data_point)
                
                # Flush buffer if full
                if buffer_full:
                    self._flush_buffer()
                
            except Exception as e:
                logger.error(f"Error collecting time series data: {str(e)}")
            
            last_collection = collection_time
    
    def _collect_data_point(self, tier: str) -> Dict[str, float]:
        """
        Collect a single data point.
        
        Args:
            tier: Storage tier path
            
        Returns:
            Dictionary mapping metric names to values
        """
        # In a real implementation, this would collect actual performance metrics
        # For this example, we'll use placeholder logic
        
        # Sample implementation: Collect current storage performance
        data_point = {}
        
        try:
            # Placeholder for actual metric collection
            # In a real implementation, this would use OS-specific methods
            # to get current storage performance metrics
            
            # Example placeholders
            data_point["throughput_MBps"] = 100.0  # MB/s
            data_point["iops"] = 1000.0  # IOPS
            data_point["latency_ms"] = 5.0  # ms
            
            # Add any additional configured metrics (with placeholder values)
            for metric in self.config.metrics:
                if metric not in data_point:
                    data_point[metric] = 0.0
            
        except Exception as e:
            logger.error(f"Error collecting metrics for {tier}: {str(e)}")
            
            # Fill with zeros on error
            for metric in self.config.metrics:
                data_point[metric] = 0.0
        
        return data_point
    
    def _flush_buffer(self) -> None:
        """
        Flush buffer data to persistent storage if configured.
        """
        if self.buffer.is_empty():
            return
        
        if self.db_connection and self.time_series_data:
            try:
                # Get buffer data
                timestamps, values = self.buffer.get_data()
                
                # Insert into database
                cursor = self.db_connection.cursor()
                
                for i, timestamp in enumerate(timestamps):
                    data_values = [
                        self.time_series_data.tier,
                        self.time_series_data.test_id,
                        timestamp
                    ]
                    
                    for metric in self.config.metrics:
                        if metric in values and i < len(values[metric]):
                            data_values.append(values[metric][i])
                        else:
                            data_values.append(None)
                    
                    # Construct placeholders for SQL query
                    placeholders = "?, ?, ?" + ", ?" * len(self.config.metrics)
                    
                    cursor.execute(
                        f"INSERT INTO time_series_data (tier, test_id, timestamp, {', '.join(self.config.metrics)}) "
                        f"VALUES ({placeholders})",
                        data_values
                    )
                
                self.db_connection.commit()
                
            except sqlite3.Error as e:
                logger.error(f"Database error while flushing buffer: {str(e)}")
            
            except Exception as e:
                logger.error(f"Error flushing buffer: {str(e)}")
        
        # Clear buffer
        self.buffer.clear()
    
    def _init_database(self) -> None:
        """
        Initialize SQLite database for time series storage.
        
        Raises:
            BenchmarkDataError: If database initialization fails
        """
        try:
            # Create database directory if it doesn't exist
            import os
            os.makedirs(os.path.dirname(os.path.abspath(self.config.db_path)), exist_ok=True)
            
            # Connect to database
            self.db_connection = sqlite3.connect(self.config.db_path)
            
            # Create table
            cursor = self.db_connection.cursor()
            
            # Generate column definitions for metrics
            metric_columns = ", ".join([f"{metric} REAL" for metric in self.config.metrics])
            
            cursor.execute(
                f"CREATE TABLE IF NOT EXISTS time_series_data ("
                f"id INTEGER PRIMARY KEY AUTOINCREMENT, "
                f"tier TEXT, "
                f"test_id TEXT, "
                f"timestamp REAL, "
                f"{metric_columns})"
            )
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_time_series_test_id ON time_series_data (test_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_time_series_timestamp ON time_series_data (timestamp)")
            
            self.db_connection.commit()
            
            logger.debug(f"Initialized time series database: {self.config.db_path}")
            
        except sqlite3.Error as e:
            logger.error(f"Database error during initialization: {str(e)}")
            raise BenchmarkDataError(f"Failed to initialize time series database: {str(e)}")
        
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            raise BenchmarkDataError(f"Failed to initialize time series database: {str(e)}")
