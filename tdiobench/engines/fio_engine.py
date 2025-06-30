#!/usr/bin/env python3
"""
FIO Engine

This module provides an execution engine for running I/O benchmarks using FIO.

Author: Jack Ogaja
Date: 2025-06-26
"""

import os
import json
import logging
import subprocess
import tempfile
from typing import Dict, List, Any, Optional, Union

from tdiobench.core.benchmark_config import BenchmarkConfig
from tdiobench.core.benchmark_exceptions import BenchmarkExecutionError, BenchmarkConfigError
from tdiobench.utils.parameter_standards import standard_parameters
from tdiobench.utils.error_handling import safe_operation

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
        **kwargs
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
            **kwargs: Additional parameters for FIO
            
        Returns:
            Dictionary containing benchmark results
            
        Raises:
            BenchmarkConfigError: If configuration is invalid
            BenchmarkExecutionError: If benchmark execution fails
        """
        logger.info(f"Executing FIO benchmark on {tier_path} with pattern {io_pattern}, block size {block_size}")
        
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
            **kwargs
        )
        
        try:
            # Execute FIO benchmark
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
        **kwargs
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
            **kwargs: Additional parameters for FIO
            
        Returns:
            Path to job file
        """
        # Create temporary job file
        fd, job_file = tempfile.mkstemp(suffix=".fio")
        os.close(fd)
        
        # Write job file content
        with open(job_file, 'w') as f:
            f.write("[global]\n")
            f.write(f"runtime={duration_seconds}\n")
            f.write("time_based=1\n")
            f.write("group_reporting=1\n")
            f.write("thread=1\n")
            f.write(f"direct={1 if direct_io else 0}\n")
            
            # Add additional global parameters
            global_params = kwargs.get("global_params", {})
            for key, value in global_params.items():
                f.write(f"{key}={value}\n")
            
            f.write("\n[job1]\n")
            f.write(f"filename={test_file}\n")
            f.write(f"rw={io_pattern}\n")
            f.write(f"bs={block_size}\n")
            f.write(f"iodepth={io_depth}\n")
            f.write("ioengine=libaio\n")
            
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
                "read": {
                    "throughput_MBps": read_bw,
                    "iops": read_iops,
                    "latency_ms": read_latency
                },
                "write": {
                    "throughput_MBps": write_bw,
                    "iops": write_iops,
                    "latency_ms": write_latency
                },
                "raw_data": data
            }
            
            return result
            
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            raise BenchmarkDataError(f"Failed to parse FIO output: {str(e)}")
