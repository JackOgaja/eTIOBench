#!/usr/bin/env python3
"""
FIO Engine

This module provides an execution engine for running I/O benchmarks using FIO.

Author: Jack Ogaja
Date: 2025-06-26
"""

import json
import logging
import os
import subprocess
import tempfile
from typing import Any, Dict, Optional

from tdiobench.core.benchmark_config import BenchmarkConfig
from tdiobench.core.benchmark_exceptions import (
    BenchmarkConfigError,
    BenchmarkDataError,
    BenchmarkExecutionError,
)
from tdiobench.utils.error_handling import safe_operation
from tdiobench.utils.parameter_standards import standard_parameters

logger = logging.getLogger("tdiobench.engines.fio")


class FIOEngine:
    """
    Execution engine for FIO benchmarks.

    Provides methods for running I/O benchmarks using the FIO tool.
    """

    def __init__(self, config: BenchmarkConfig):
        """
        Initialize FIO engine.

        Args:
            config: Benchmark configuration
        """
        self.config = config
        self.fio_path = config.get("engines.fio.path", "fio")
        self.fio_version = self._get_fio_version()

        # Check if FIO is available
        if not self.fio_version:
            logger.warning("FIO not found. Please install FIO or set correct path.")

    @safe_operation("config")
    def _get_fio_version(self) -> Optional[str]:
        """
        Get FIO version.

        Returns:
            FIO version string, or None if FIO is not available
        """
        try:
            output = subprocess.check_output([self.fio_path, "--version"], text=True)
            return output.strip()
        except (subprocess.SubprocessError, FileNotFoundError):
            return None

    @safe_operation("execution")
    @standard_parameters
    def execute_benchmark(
        self,
        tier_path: str,
        duration_seconds: int = 60,
        block_size: str = "4k",
        io_pattern: str = "randrw",
        io_depth: int = 32,
        direct_io: bool = True,
        time_series_enabled: bool = False,
        time_series_callback=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Execute a benchmark using FIO.

        Args:
            tier_path: Path to storage tier
            duration_seconds: Benchmark duration in seconds
            block_size: I/O block size (e.g., '4k', '1m')
            io_pattern: I/O pattern (e.g., 'read', 'write', 'randrw')
            io_depth: I/O queue depth
            direct_io: Use direct I/O (bypass cache)
            time_series_enabled: Enable time series data collection
            time_series_callback: Callback function for time series data points
            **kwargs: Additional parameters for FIO

        Returns:
            Dictionary containing benchmark results

        Raises:
            BenchmarkConfigError: If configuration is invalid
            BenchmarkExecutionError: If benchmark execution fails
        """
        logger.info(
            f"Executing FIO benchmark on {tier_path} with pattern {io_pattern}, block size {block_size}"
        )

        # Check if FIO is available
        if not self.fio_version:
            raise BenchmarkConfigError("FIO not available. Please install FIO or set correct path.")

        # Check if tier path exists
        if not os.path.exists(tier_path):
            raise BenchmarkConfigError(f"Tier path does not exist: {tier_path}")

        # Create test file path
        test_file = os.path.join(tier_path, "fio_test_file")

        # Create FIO job file
        job_file = self._create_job_file(
            test_file=test_file,
            duration_seconds=duration_seconds,
            block_size=block_size,
            io_pattern=io_pattern,
            io_depth=io_depth,
            direct_io=direct_io,
            time_series_enabled=time_series_enabled,
            **kwargs,
        )

        try:
            # Execute FIO benchmark with optional real-time monitoring
            if time_series_enabled and time_series_callback:
                output = self._run_fio_with_time_series(job_file, time_series_callback)
            else:
                output = self._run_fio(job_file)

            # Parse FIO output
            result = self._parse_fio_output(output)

            return result

        finally:
            # Clean up job file
            if os.path.exists(job_file):
                os.remove(job_file)

            # Clean up test file if specified in config
            if self.config.get("engines.fio.cleanup_test_files", True):
                if os.path.exists(test_file):
                    try:
                        os.remove(test_file)
                    except Exception as e:
                        logger.warning(f"Failed to remove test file {test_file}: {str(e)}")

    @safe_operation("execution")
    def _create_job_file(
        self,
        test_file: str,
        duration_seconds: int,
        block_size: str,
        io_pattern: str,
        io_depth: int,
        direct_io: bool,
        time_series_enabled: bool = False,
        **kwargs,
    ) -> str:
        """
        Create FIO job file.

        Args:
            test_file: Path to test file
            duration_seconds: Benchmark duration in seconds
            block_size: I/O block size
            io_pattern: I/O pattern
            io_depth: I/O queue depth
            direct_io: Use direct I/O
            time_series_enabled: Enable time series data collection
            **kwargs: Additional parameters for FIO

        Returns:
            Path to job file
        """
        # Create temporary job file
        fd, job_file = tempfile.mkstemp(suffix=".fio")
        os.close(fd)

        # Write job file content
        with open(job_file, "w") as f:
            f.write("[global]\n")
            f.write(f"runtime={duration_seconds}\n")
            f.write("time_based=1\n")
            f.write("group_reporting=1\n")
            f.write("thread=1\n")
            f.write(f"direct={1 if direct_io else 0}\n")
            
            # Note: status-interval and log settings are passed as command line arguments
            # when time_series_enabled is True, not in the job file

            # Add additional global parameters
            global_params = kwargs.get("global_params", {})
            for key, value in global_params.items():
                f.write(f"{key}={value}\n")

            f.write("\n[job1]\n")
            f.write(f"filename={test_file}\n")
            f.write(f"rw={io_pattern}\n")
            f.write(f"bs={block_size}\n")
            f.write(f"iodepth={io_depth}\n")
            
            # Use appropriate I/O engine for the platform
            import platform
            system = platform.system().lower()
            if system == "linux":
                f.write("ioengine=libaio\n")
            elif system == "darwin":  # macOS
                f.write("ioengine=posixaio\n")
            elif system == "windows":
                f.write("ioengine=windowsaio\n")
            else:
                f.write("ioengine=sync\n")  # Fallback for other systems
            
            # Add file size (required parameter)
            file_size = kwargs.get("file_size", "100M")  # Default to 100MB
            f.write(f"size={file_size}\n")
            
            # Add runtime duration
            f.write(f"runtime={duration_seconds}\n")
            
            # Add direct I/O setting
            if direct_io:
                f.write("direct=1\n")

            # Add additional job parameters
            job_params = kwargs.get("job_params", {})
            for key, value in job_params.items():
                f.write(f"{key}={value}\n")

        return job_file

    @safe_operation("execution")
    def _run_fio(self, job_file: str) -> str:
        """
        Run FIO benchmark.

        Args:
            job_file: Path to job file

        Returns:
            FIO output in JSON format

        Raises:
            BenchmarkExecutionError: If FIO execution fails
        """
        try:
            # Build FIO command
            cmd = [self.fio_path, "--output-format=json", job_file]

            # Execute FIO
            logger.debug(f"Executing FIO command: {' '.join(cmd)}")
            output = subprocess.check_output(cmd, text=True)

            return output

        except subprocess.SubprocessError as e:
            raise BenchmarkExecutionError(f"FIO execution failed: {str(e)}")

    @safe_operation("execution")
    def _run_fio_with_time_series(self, job_file: str, time_series_callback) -> str:
        """
        Run FIO benchmark with real-time time series data collection.
        
        Uses FIO's write logging capabilities to get per-second statistics.

        Args:
            job_file: Path to job file
            time_series_callback: Callback function for time series data points

        Returns:
            FIO output in JSON format

        Raises:
            BenchmarkExecutionError: If FIO execution fails
        """
        import threading
        import tempfile
        import os
        
        try:
            # Create temporary files for FIO logs
            _, bw_log_base = tempfile.mkstemp(suffix="_bw")
            _, iops_log_base = tempfile.mkstemp(suffix="_iops")
            _, lat_log_base = tempfile.mkstemp(suffix="_lat")
            
            # Remove the files so FIO can create them with proper naming
            os.remove(bw_log_base)
            os.remove(iops_log_base)
            os.remove(lat_log_base)
            
            # FIO will create files like: basename_bw.1.log, basename_iops.1.log, basename_lat.1.log
            bw_log_file = f"{bw_log_base}_bw.1.log"
            iops_log_file = f"{iops_log_base}_iops.1.log"
            lat_log_file = f"{lat_log_base}_lat.1.log"
            
            # Build FIO command with logging
            cmd = [
                self.fio_path, 
                "--output-format=json",
                f"--write_bw_log={bw_log_base}",
                f"--write_iops_log={iops_log_base}",
                f"--write_lat_log={lat_log_base}",
                "--log_avg_msec=1000",  # Log every 1 second
                job_file
            ]

            # Execute FIO
            logger.debug(f"Executing FIO command with logging: {' '.join(cmd)}")
            
            # Start FIO process
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Monitor log files in a separate thread
            def monitor_logs():
                import time
                time.sleep(2)  # Give FIO time to start and create log files
                
                last_processed = {bw_log_file: 0, iops_log_file: 0, lat_log_file: 0}
                
                while process.poll() is None:  # While FIO is running
                    try:
                        # Read new data from each log file
                        bw_data = self._read_fio_log_incremental(bw_log_file, last_processed[bw_log_file])
                        iops_data = self._read_fio_log_incremental(iops_log_file, last_processed[iops_log_file])
                        lat_data = self._read_fio_log_incremental(lat_log_file, last_processed[lat_log_file])
                        
                        # Update last processed positions
                        last_processed[bw_log_file] += len(bw_data)
                        last_processed[iops_log_file] += len(iops_data)
                        last_processed[lat_log_file] += len(lat_data)
                        
                        # Process the data and call callback
                        self._process_fio_log_data(bw_data, iops_data, lat_data, time_series_callback)
                        
                    except Exception as e:
                        logger.debug(f"Error reading FIO logs: {e}")
                    
                    time.sleep(1)  # Check logs every second
            
            # Start monitoring thread
            monitor_thread = threading.Thread(target=monitor_logs, daemon=True)
            monitor_thread.start()
            
            # Wait for completion and get final output
            stdout, stderr = process.communicate()
            
            # Clean up log files
            for log_file in [bw_log_file, iops_log_file, lat_log_file]:
                try:
                    if os.path.exists(log_file):
                        os.remove(log_file)
                except:
                    pass
            
            if process.returncode != 0:
                raise BenchmarkExecutionError(f"FIO execution failed: {stderr}")
                
            return stdout

        except subprocess.SubprocessError as e:
            raise BenchmarkExecutionError(f"FIO execution failed: {str(e)}")

    def _read_fio_log_incremental(self, log_file: str, last_position: int):
        """Read new lines from FIO log file since last position."""
        try:
            if not os.path.exists(log_file):
                return []
                
            with open(log_file, 'r') as f:
                lines = f.readlines()
                if len(lines) > last_position:
                    return lines[last_position:]
                return []
        except:
            return []

    def _process_fio_log_data(self, bw_data, iops_data, lat_data, callback):
        """Process FIO log data and create time series data points."""
        try:
            # FIO log format: timestamp_ms, value, direction, block_size, offset
            # We need to process all new entries, not just the last one
            for i in range(min(len(bw_data), len(iops_data), len(lat_data))):
                bw_line = bw_data[i].strip()
                iops_line = iops_data[i].strip()
                lat_line = lat_data[i].strip()
                
                if bw_line and iops_line and lat_line:
                    bw_parts = bw_line.split(', ')
                    iops_parts = iops_line.split(', ')
                    lat_parts = lat_line.split(', ')
                    
                    if len(bw_parts) >= 2 and len(iops_parts) >= 2 and len(lat_parts) >= 2:
                        # Parse values
                        timestamp_ms = int(bw_parts[0])
                        bandwidth_kbps = float(bw_parts[1])  # KB/s
                        iops = float(iops_parts[1])
                        latency_ns = float(lat_parts[1])  # nanoseconds
                        
                        # Convert to standard units
                        throughput_mbps = bandwidth_kbps / 1024.0  # Convert KB/s to MB/s
                        latency_ms = latency_ns / 1000000.0  # Convert nanoseconds to milliseconds
                        
                        data_point = {
                            "throughput_MBps": throughput_mbps,
                            "iops": iops,
                            "latency_ms": latency_ms,
                            "timestamp_ms": timestamp_ms
                        }
                        
                        callback(data_point)
                        logger.debug(f"FIO real-time data: {throughput_mbps:.1f} MB/s, {iops:.0f} IOPS, {latency_ms:.2f} ms")
                        
        except Exception as e:
            logger.debug(f"Error processing FIO log data: {e}")

    @safe_operation("data")
    def _parse_fio_output(self, output: str) -> Dict[str, Any]:
        """
        Parse FIO JSON output.

        Args:
            output: FIO output in JSON format

        Returns:
            Dictionary containing parsed results

        Raises:
            BenchmarkDataError: If parsing fails
        """
        try:
            # Parse JSON output
            data = json.loads(output)

            # Extract job results
            job_results = data["jobs"][0]

            # Extract metrics
            read_iops = job_results["read"]["iops"]
            read_bw = job_results["read"]["bw"] / 1024  # Convert to MB/s
            read_latency = job_results["read"]["lat_ns"]["mean"] / 1000000  # Convert to ms

            write_iops = job_results["write"]["iops"]
            write_bw = job_results["write"]["bw"] / 1024  # Convert to MB/s
            write_latency = job_results["write"]["lat_ns"]["mean"] / 1000000  # Convert to ms

            # Calculate total metrics
            total_iops = read_iops + write_iops
            total_bw = read_bw + write_bw

            # Calculate weighted latency
            if total_iops > 0:
                total_latency = (read_iops * read_latency + write_iops * write_latency) / total_iops
            else:
                total_latency = 0

            # Create result dictionary
            result = {
                "throughput_MBps": total_bw,
                "iops": total_iops,
                "latency_ms": total_latency,
                "read": {"throughput_MBps": read_bw, "iops": read_iops, "latency_ms": read_latency},
                "write": {
                    "throughput_MBps": write_bw,
                    "iops": write_iops,
                    "latency_ms": write_latency,
                },
                "raw_data": data,
            }

            return result

        except (json.JSONDecodeError, KeyError, IndexError) as e:
            raise BenchmarkDataError(f"Failed to parse FIO output: {str(e)}")
