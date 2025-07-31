#!/usr/bin/env python3
"""
Time Series Collector (Tiered Storage I/O Benchmark)

This module provides classes for collecting time series performance data
during benchmark execution.

Author: Jack Ogaja
Date: 2025-06-26
"""

import logging
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional, Union

from tdiobench.core.benchmark_config import BenchmarkConfig
from tdiobench.core.benchmark_exceptions import BenchmarkDataError

# Import C++ integration for enhanced performance
try:
    from tdiobench.cpp_integration import CppIntegrationConfig, CppTimeSeriesCollector, CPP_AVAILABLE
    logger = logging.getLogger("tdiobench.time_series")
    logger.info(f"C++ integration available for time series collection: {CPP_AVAILABLE}")
except ImportError as e:
    CPP_AVAILABLE = False
    logger = logging.getLogger("tdiobench.time_series")
    logger.info("C++ integration not available for time series collection, using Python implementations")


class TimeSeriesData:
    """
    Container for time series performance data collected during benchmarks.
    """

    def __init__(
        self,
        tier: str,
        test_id: str,
        start_time: float,
        metrics: List[str],
        interval: float,
    ):
        """
        Initialize time series data container.

        Args:
            tier: Storage tier path
            test_id: Test identifier
            start_time: Collection start time
            metrics: List of metrics to collect
            interval: Collection interval in seconds
        """
        self.tier = tier
        self.test_id = test_id
        self.start_time = start_time
        self.metrics = metrics
        self.interval = interval
        
        # Initialize data storage
        self.timestamps: List[float] = []
        self.data: Dict[str, List[float]] = {}
        for metric in metrics:
            self.data[metric] = []

    def add_data_point(self, timestamp: float, data_point: Dict[str, float]) -> None:
        """
        Add a data point to the time series.

        Args:
            timestamp: Data point timestamp
            data_point: Dictionary mapping metric names to values
        """
        self.timestamps.append(timestamp)
        
        for metric in self.metrics:
            value = data_point.get(metric, 0.0)
            self.data[metric].append(value)

    def get_data_points(self, metric: str) -> List[float]:
        """
        Get data points for a specific metric.

        Args:
            metric: Metric name

        Returns:
            List of data points for the metric
        """
        return self.data.get(metric, [])

    def get_time_range(self) -> float:
        """
        Get the time range of collected data.

        Returns:
            Time range in seconds
        """
        if len(self.timestamps) < 2:
            return 0.0
        return self.timestamps[-1] - self.timestamps[0]

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert time series data to dictionary format.

        Returns:
            Dictionary representation of the time series data
        """
        return {
            "tier": self.tier,
            "test_id": self.test_id,
            "start_time": self.start_time,
            "interval": self.interval,
            "timestamps": self.timestamps,
            "metrics": self.metrics,
            "data": self.data
        }


class TimeSeriesConfig:
    """Configuration for time series collection."""

    def __init__(
        self,
        interval: float = 1.0,
        buffer_size: int = 1000,
        db_path: Optional[str] = None,
        metrics: Optional[List[str]] = None,
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
    def from_benchmark_config(cls, config: BenchmarkConfig) -> "TimeSeriesConfig":
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
            metrics=config.get("collection.time_series.metrics"),
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
        self.fio_callback_data: List[tuple] = []  # Store (timestamp, data_point) tuples

        # Initialize C++ integration if available
        if CPP_AVAILABLE:
            cpp_config = CppIntegrationConfig(
                use_cpp=True,
                min_data_size_for_cpp=100,
                enable_parallel=True,
                num_threads=4
            )
            self.cpp_collector = CppTimeSeriesCollector(cpp_config)
            logger.info("C++ time series collector initialized for enhanced performance")
        else:
            self.cpp_collector = None
            logger.info("Using Python implementation for time series collection")

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
            interval=self.config.interval,
        )

        # Start collection thread
        self.collection_active = True
        self.fio_callback_data = []  # Reset callback data
        self.collection_thread = threading.Thread(
            target=self._collection_loop, args=(tier, test_id), daemon=True
        )
        self.collection_thread.start()

        logger.info(f"Started time series collection for {tier} (test {test_id})")

    def add_fio_data_point(self, data_point: Dict[str, float]) -> None:
        """
        Add a data point from FIO real-time output.
        
        This method is called by the FIO engine during benchmark execution.
        
        Args:
            data_point: Dictionary mapping metric names to values
        """
        if self.collection_active and self.time_series_data:
            timestamp = time.time()
            
            # Use C++ acceleration for high-frequency data processing if available
            if (self.cpp_collector and 
                CPP_AVAILABLE and 
                len(self.fio_callback_data) >= 50):  # Batch process every 50 points
                
                try:
                    # Process batch with C++ acceleration
                    batch_data = {
                        'timestamp': timestamp,
                        'data_point': data_point,
                        'batch_size': len(self.fio_callback_data)
                    }
                    self.cpp_collector.process_data_batch(batch_data)
                    logger.debug("🚀 C++ acceleration used for time series data processing")
                except Exception as e:
                    logger.debug(f"C++ data processing failed, using Python fallback: {e}")
            
            self.fio_callback_data.append((timestamp, data_point))
            
            # Add to time series data
            self.time_series_data.add_data_point(timestamp, data_point)
            
            # Add to buffer
            buffer_full = self.buffer.add_data_point(timestamp, data_point)
            
            # Flush buffer if full
            if buffer_full:
                self._flush_buffer()

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
            "metrics": {},
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
                        "avg": sum(filtered_values) / len(filtered_values),
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
                with open(output_path, "w") as f:
                    json.dump(data, f, indent=2)
                return output_path
            else:
                return None

        elif format.lower() == "csv":
            import csv

            if output_path:
                with open(output_path, "w", newline="") as f:
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
        Main collection loop - now primarily serves as a fallback and data validator.
        Real FIO data comes through add_fio_data_point() callback.

        Args:
            tier: Storage tier path
            test_id: Test identifier
        """
        last_collection = time.time()
        last_fio_data_count = 0

        while self.collection_active:
            # Calculate sleep time to maintain consistent interval
            current_time = time.time()
            elapsed = current_time - last_collection
            sleep_time = max(0.0, self.config.interval - elapsed)

            if sleep_time > 0:
                time.sleep(sleep_time)

            collection_time = time.time()
            
            # Check if we're getting FIO data
            current_fio_data_count = len(self.fio_callback_data)
            
            if current_fio_data_count > last_fio_data_count:
                # We're receiving FIO data, no need to collect psutil data
                last_fio_data_count = current_fio_data_count
                logger.debug(f"Using FIO real-time data ({current_fio_data_count} points collected)")
            else:
                # Fallback to psutil-based collection if no FIO data is coming
                try:
                    data_point = self._collect_data_point(tier)
                    logger.debug("Using fallback psutil-based collection")

                    # Add to time series data
                    if self.time_series_data:
                        self.time_series_data.add_data_point(collection_time, data_point)

                    # Add to buffer
                    buffer_full = self.buffer.add_data_point(collection_time, data_point)

                    # Flush buffer if full
                    if buffer_full:
                        self._flush_buffer()

                except Exception as e:
                    logger.error(f"Error collecting fallback time series data: {str(e)}")

            last_collection = collection_time

    def _collect_data_point(self, tier: str) -> Dict[str, float]:
        """
        Collect a single data point by monitoring system I/O metrics.

        Args:
            tier: Storage tier path

        Returns:
            Dictionary mapping metric names to values
        """
        data_point = {}

        try:
            import psutil
            import os
            
            # Get disk I/O statistics for the storage tier
            disk_usage = psutil.disk_usage(tier)
            
            # Get current I/O statistics if available
            disk_io = None
            try:
                # Get all disk I/O counters
                disk_io_counters = psutil.disk_io_counters(perdisk=True)
                
                # Try to find the relevant disk for the tier path
                tier_device = None
                if hasattr(os, 'stat'):
                    try:
                        # Get device of the tier path
                        stat_info = os.stat(tier)
                        tier_device_id = stat_info.st_dev
                        
                        # This is a simplified approach - in production you'd need
                        # more sophisticated device mapping
                        if disk_io_counters:
                            # Use the first available disk counter as approximation
                            disk_io = list(disk_io_counters.values())[0]
                    except:
                        pass
                
                if not disk_io and disk_io_counters:
                    # Fallback: use system-wide disk I/O
                    disk_io = psutil.disk_io_counters()
                    
            except Exception as e:
                logger.debug(f"Could not get disk I/O counters: {e}")

            # Calculate metrics based on available data
            if hasattr(self, '_previous_disk_io') and disk_io and self._previous_disk_io:
                # Calculate throughput and IOPS based on delta from previous measurement
                time_delta = time.time() - getattr(self, '_previous_time', time.time())
                
                if time_delta > 0:
                    # Calculate bytes/sec
                    read_bytes_delta = disk_io.read_bytes - self._previous_disk_io.read_bytes
                    write_bytes_delta = disk_io.write_bytes - self._previous_disk_io.write_bytes
                    total_bytes_delta = read_bytes_delta + write_bytes_delta
                    
                    # Calculate IOPS
                    read_ops_delta = disk_io.read_count - self._previous_disk_io.read_count
                    write_ops_delta = disk_io.write_count - self._previous_disk_io.write_count
                    total_ops_delta = read_ops_delta + write_ops_delta
                    
                    # Calculate metrics
                    throughput_bps = total_bytes_delta / time_delta
                    data_point["throughput_MBps"] = throughput_bps / (1024 * 1024)  # Convert to MB/s
                    data_point["iops"] = total_ops_delta / time_delta
                    
                    # Estimate latency (simplified calculation)
                    if total_ops_delta > 0:
                        # Use read/write time if available
                        if hasattr(disk_io, 'read_time') and hasattr(disk_io, 'write_time'):
                            read_time_delta = disk_io.read_time - self._previous_disk_io.read_time
                            write_time_delta = disk_io.write_time - self._previous_disk_io.write_time
                            total_time_delta = read_time_delta + write_time_delta
                            data_point["latency_ms"] = total_time_delta / total_ops_delta
                        else:
                            # Fallback estimation
                            data_point["latency_ms"] = min(100.0, max(0.1, 1000.0 / data_point["iops"]))
                    else:
                        data_point["latency_ms"] = 0.0
                else:
                    # No time elapsed, use zeros
                    data_point["throughput_MBps"] = 0.0
                    data_point["iops"] = 0.0
                    data_point["latency_ms"] = 0.0
            else:
                # First measurement or no disk I/O available, use baseline values
                data_point["throughput_MBps"] = 0.0
                data_point["iops"] = 0.0
                data_point["latency_ms"] = 0.0
            
            # Store current values for next calculation
            if disk_io:
                self._previous_disk_io = disk_io
                self._previous_time = time.time()

            # Add any additional configured metrics with default values
            for metric in self.config.metrics:
                if metric not in data_point:
                    data_point[metric] = 0.0

        except ImportError:
            logger.warning("psutil not available, using placeholder metrics")
            # Fallback to placeholder values
            data_point = {
                "throughput_MBps": 50.0 + (time.time() % 10) * 5,  # Simulate varying performance
                "iops": 1000.0 + (time.time() % 10) * 100,
                "latency_ms": 1.0 + (time.time() % 10) * 0.5,
            }
            
            # Add any additional configured metrics
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
                        timestamp,
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
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_time_series_test_id ON time_series_data (test_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_time_series_timestamp ON time_series_data (timestamp)"
            )

            self.db_connection.commit()

            logger.debug(f"Initialized time series database: {self.config.db_path}")

        except sqlite3.Error as e:
            logger.error(f"Database error during initialization: {str(e)}")
            raise BenchmarkDataError(f"Failed to initialize time series database: {str(e)}")

        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            raise BenchmarkDataError(f"Failed to initialize time series database: {str(e)}")
