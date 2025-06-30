#!/usr/bin/env python3
"""
Tiered Storage I/O Benchmark Suite - Core Orchestrator

This module provides the main orchestration for the benchmark suite,
coordinating the execution of benchmarks, analysis, and reporting.

Author: Jack Ogaja
Date: 2025-06-26
"""

import os
import time
import uuid
import logging
from typing import List, Dict, Any, Union, Optional

from tdiobench.core.benchmark_config import BenchmarkConfig
from tdiobench.core.benchmark_data import BenchmarkData, BenchmarkResult
from tdiobench.core.benchmark_exceptions import (
    BenchmarkConfigError,
    BenchmarkExecutionError,
    BenchmarkResourceError
)
from tdiobench.engines.fio_engine import FIOEngine
from tdiobench.collection.time_series_collector import TimeSeriesCollector
from tdiobench.collection.system_metrics_collector import SystemMetricsCollector
from tdiobench.analysis.statistical_analyzer import StatisticalAnalyzer
from tdiobench.analysis.network_analyzer import NetworkAnalyzer
from tdiobench.analysis.time_series_analyzer import TimeSeriesAnalyzer
from tdiobench.analysis.anomaly_detector import AnomalyDetector
from tdiobench.visualization.report_generator import ReportGenerator
from tdiobench.results.result_store import ResultStore
from tdiobench.utils.data_processor import DataProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("tdiobench")

class BenchmarkEvent:
    """Represents a significant event during benchmark execution."""
    
    def __init__(self, event_type: str, timestamp: float, details: Dict[str, Any]):
        """
        Initialize a benchmark event.
        
        Args:
            event_type: Type of event (start, end, error, etc.)
            timestamp: Event timestamp (epoch time)
            details: Event details
        """
        self.event_type = event_type
        self.timestamp = timestamp
        self.details = details
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary representation."""
        return {
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "details": self.details
        }

class BenchmarkSuite:
    """
    Main orchestrator for the Enhanced Tiered Storage I/O Benchmark Suite.
    
    This class coordinates the execution of benchmarks, data collection,
    analysis, and reporting across different storage tiers.
    """
    
    def __init__(self, config_path: Optional[str] = None, log_level: str = "info"):
        """
        Initialize the benchmark suite.
        
        Args:
            config_path: Path to configuration file (optional)
            log_level: Logging level (debug, info, warning, error)
        """
        # Set up logging
        numeric_level = getattr(logging, log_level.upper(), None)
        if isinstance(numeric_level, int):
            logger.setLevel(numeric_level)
        
        # Initialize configuration
        self.config = BenchmarkConfig.from_file(config_path) if config_path else BenchmarkConfig()
        
        # Initialize components
        self.engine = FIOEngine(self.config)
        self.time_series_collector = TimeSeriesCollector(self.config)
        self.system_metrics_collector = SystemMetricsCollector(self.config)
        self.statistical_analyzer = StatisticalAnalyzer(self.config)
        self.network_analyzer = NetworkAnalyzer(self.config)
        self.time_series_analyzer = TimeSeriesAnalyzer(self.config)
        self.anomaly_detector = AnomalyDetector(self.config)
        self.report_generator = ReportGenerator(self.config)
        self.result_store = ResultStore(self.config)
        self.data_processor = DataProcessor()
        
        # Initialize state
        self.events = []
        self.benchmark_id = str(uuid.uuid4())
        self.running = False
        self.production_safe_mode = self.config.get("production_safety.enabled", False)
        
        logger.info(f"Benchmark Suite initialized with ID: {self.benchmark_id}")
        if self.production_safe_mode:
            logger.info("Production safety mode ENABLED")
    
    def run_comprehensive_analysis(
        self,
        tiers: List[str],
        duration: int = 300,
        block_sizes: Optional[List[str]] = None,
        patterns: Optional[List[str]] = None,
        enable_all_modules: bool = False,
        **kwargs
    ) -> BenchmarkResult:
        """
        Run a comprehensive benchmark analysis across multiple storage tiers.
        
        Args:
            tiers: List of storage tier paths to analyze
            duration: Test duration in seconds
            block_sizes: Block sizes to test
            patterns: I/O patterns to test
            enable_all_modules: Enable all analysis modules
            **kwargs: Additional parameters
            
        Returns:
            BenchmarkResult object with comprehensive results
        """
        try:
            # Validate parameters
            self._validate_parameters(tiers, duration)
            
            # Prepare configuration
            block_sizes = block_sizes or self.config.get("block_sizes", ["4k", "64k", "1m"])
            patterns = patterns or self.config.get("patterns", ["read", "write", "randrw"])
            
            # Record start event
            self._record_event("benchmark_start", {
                "tiers": tiers,
                "duration": duration,
                "block_sizes": block_sizes,
                "patterns": patterns
            })
            
            # Set running state
            self.running = True
            
            # Initialize benchmark data
            benchmark_data = BenchmarkData(
                benchmark_id=self.benchmark_id,
                tiers=tiers,
                duration=duration,
                block_sizes=block_sizes,
                patterns=patterns
            )
            
            # Start system metrics collection
            if self.config.get("collection.system_metrics.enabled", True):
                self.system_metrics_collector.start_collection(benchmark_data)
            
            # Execute benchmarks for each tier
            for tier in tiers:
                tier_result = self._benchmark_tier(
                    tier, duration, block_sizes, patterns, **kwargs
                )
                benchmark_data.add_tier_result(tier, tier_result)
                
                # Check for resource limits if in production safe mode
                if self.production_safe_mode and self._check_resource_limits():
                    logger.warning("Resource limits exceeded in production safe mode. Stopping benchmark.")
                    self._record_event("resource_limit_exceeded", {
                        "reason": "Resource usage exceeded configured limits"
                    })
                    break
            
            # Stop system metrics collection
            if self.config.get("collection.system_metrics.enabled", True):
                system_metrics = self.system_metrics_collector.stop_collection()
                benchmark_data.set_system_metrics(system_metrics)
            
            # Perform analysis
            benchmark_result = self._analyze_benchmark_data(benchmark_data, enable_all_modules)
            
            # Store results
            if self.config.get("storage.enabled", True):
                self.result_store.store_results(benchmark_result)
            
            # Generate reports
            if self.config.get("visualization.reports.enabled", True):
                report_formats = self.config.get("visualization.reports.formats", ["html", "json"])
                output_dir = self.config.get("output.directory", "./results")
                report_title = kwargs.get("report_title") or self.config.get(
                    "visualization.reports.title", "Storage Benchmark Report"
                )
                
                self.report_generator.generate_reports(
                    benchmark_result, 
                    output_dir=output_dir,
                    formats=report_formats,
                    report_title=report_title
                )
            
            # Record completion event
            self._record_event("benchmark_complete", {
                "result_summary": benchmark_result.get_summary()
            })
            
            # Reset running state
            self.running = False
            
            return benchmark_result
            
        except BenchmarkConfigError as e:
            self._handle_error("configuration_error", str(e))
            raise
        except BenchmarkExecutionError as e:
            self._handle_error("execution_error", str(e))
            raise
        except BenchmarkResourceError as e:
            self._handle_error("resource_error", str(e))
            raise
        except Exception as e:
            self._handle_error("unexpected_error", str(e))
            raise BenchmarkExecutionError(f"Unexpected error during benchmark: {str(e)}") from e
    
    def run_benchmark(
        self,
        tier: str,
        duration: int = 60,
        block_size: str = "4k",
        pattern: str = "randrw",
        io_depth: int = 32,
        direct: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run a single benchmark on a specific storage tier.
        
        Args:
            tier: Storage tier path
            duration: Test duration in seconds
            block_size: Block size
            pattern: I/O pattern
            io_depth: I/O queue depth
            direct: Use direct I/O
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing benchmark results
        """
        # Validate parameters
        self._validate_parameters([tier], duration)
        
        # Record start event
        self._record_event("single_benchmark_start", {
            "tier": tier,
            "duration": duration,
            "block_size": block_size,
            "pattern": pattern
        })
        
        # Prepare benchmark parameters
        benchmark_params = {
            "tier_path": tier,
            "duration": duration,
            "block_size": block_size,
            "pattern": pattern,
            "io_depth": io_depth,
            "direct": direct
        }
        benchmark_params.update(kwargs)
        
        # Start time series collection if enabled
        if self.config.get("collection.time_series.enabled", False):
            self.time_series_collector.start_collection(tier, f"bench_{self.benchmark_id}")
        
        # Execute benchmark
        result = self.engine.execute_benchmark(benchmark_params)
        
        # Stop time series collection if enabled
        if self.config.get("collection.time_series.enabled", False):
            time_series_data = self.time_series_collector.stop_collection()
            result["time_series"] = time_series_data
        
        # Record completion event
        self._record_event("single_benchmark_complete", {
            "tier": tier,
            "result_summary": {
                "throughput_MBps": result.get("throughput_MBps"),
                "iops": result.get("iops"),
                "latency_ms": result.get("latency_ms")
            }
        })
        
        return result
    
    def compare_tiers(
        self,
        benchmark_result: BenchmarkResult,
        baseline_tier: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Compare performance across different storage tiers.
        
        Args:
            benchmark_result: Benchmark result data
            baseline_tier: Baseline tier for comparison (default: first tier)
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing comparison results
        """
        tiers = benchmark_result.get_tiers()
        
        if not tiers or len(tiers) < 2:
            raise BenchmarkConfigError("At least two tiers are required for comparison")
        
        baseline = baseline_tier if baseline_tier in tiers else tiers[0]
        
        # Perform comparison analysis
        comparison = self.statistical_analyzer.compare_tiers(
            benchmark_result, baseline, **kwargs
        )
        
        return comparison
    
    def generate_reports(
        self,
        benchmark_result: BenchmarkResult,
        output_dir: str = "results",
        formats: Optional[List[str]] = None,
        report_title: Optional[str] = None,
        **kwargs
    ) -> Dict[str, str]:
        """
        Generate reports from benchmark results.
        
        Args:
            benchmark_result: Benchmark result data
            output_dir: Output directory
            formats: Output formats
            report_title: Report title
            **kwargs: Additional parameters
            
        Returns:
            Dictionary mapping format to output file paths
        """
        formats = formats or ["html", "json"]
        report_title = report_title or "Storage Benchmark Report"
        
        return self.report_generator.generate_reports(
            benchmark_result,
            output_dir=output_dir,
            formats=formats,
            report_title=report_title,
            **kwargs
        )
    
    def load_results(self, result_id: str) -> BenchmarkResult:
        """
        Load benchmark results from storage.
        
        Args:
            result_id: Result identifier
            
        Returns:
            BenchmarkResult object
        """
        return self.result_store.load_results(result_id)
    
    def _benchmark_tier(
        self,
        tier: str,
        duration: int,
        block_sizes: List[str],
        patterns: List[str],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute benchmarks for a specific tier with multiple block sizes and patterns.
        
        Args:
            tier: Storage tier path
            duration: Test duration in seconds
            block_sizes: Block sizes to test
            patterns: I/O patterns to test
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing benchmark results
        """
        tier_results = {
            "name": os.path.basename(tier),
            "path": tier,
            "tests": {}
        }
        
        # Collect tier metadata
        tier_results["metadata"] = self.system_metrics_collector.collect_tier_metadata(tier)
        
        # Start time series collection if enabled
        if self.config.get("collection.time_series.enabled", False):
            self.time_series_collector.start_collection(tier, f"tier_{self.benchmark_id}")
        
        # Execute benchmarks for each block size and pattern
        for block_size in block_sizes:
            for pattern in patterns:
                test_key = f"{pattern}_{block_size}"
                logger.info(f"Running benchmark on {tier} with {pattern} pattern and {block_size} block size")
                
                # Execute benchmark
                result = self.run_benchmark(
                    tier=tier,
                    duration=duration // (len(block_sizes) * len(patterns)),  # Divide duration across tests
                    block_size=block_size,
                    pattern=pattern,
                    **kwargs
                )
                
                tier_results["tests"][test_key] = result
                
                # Check for resource limits if in production safe mode
                if self.production_safe_mode and self._check_resource_limits():
                    logger.warning("Resource limits exceeded in production safe mode. Skipping remaining tests.")
                    break
        
        # Stop time series collection if enabled
        if self.config.get("collection.time_series.enabled", False):
            time_series_data = self.time_series_collector.stop_collection()
            tier_results["time_series"] = time_series_data
        
        # Calculate tier summary
        tier_results["summary"] = self._calculate_tier_summary(tier_results["tests"])
        
        return tier_results
    
    def _analyze_benchmark_data(
        self,
        benchmark_data: BenchmarkData,
        enable_all_modules: bool
    ) -> BenchmarkResult:
        """
        Perform comprehensive analysis on benchmark data.
        
        Args:
            benchmark_data: Benchmark data to analyze
            enable_all_modules: Enable all analysis modules
            
        Returns:
            BenchmarkResult with analysis results
        """
        logger.info("Performing benchmark data analysis")
        
        # Create benchmark result
        benchmark_result = BenchmarkResult.from_benchmark_data(benchmark_data)
        
        # Statistical analysis
        if enable_all_modules or self.config.get("analysis.statistics.enabled", False):
            logger.info("Performing statistical analysis")
            stats_results = self.statistical_analyzer.analyze_dataset(benchmark_data)
            benchmark_result.add_analysis_results("statistics", stats_results)
        
        # Time series analysis
        if enable_all_modules or self.config.get("analysis.time_series.enabled", False):
            if benchmark_data.has_time_series_data():
                logger.info("Performing time series analysis")
                ts_results = self.time_series_analyzer.analyze_time_series(benchmark_data)
                benchmark_result.add_analysis_results("time_series", ts_results)
        
        # Network impact analysis
        if enable_all_modules or self.config.get("analysis.network.enabled", False):
            logger.info("Performing network impact analysis")
            network_results = self.network_analyzer.analyze_network_impact(benchmark_data)
            benchmark_result.add_analysis_results("network", network_results)
        
        # Anomaly detection
        if enable_all_modules or self.config.get("analysis.anomaly_detection.enabled", False):
            if benchmark_data.has_time_series_data():
                logger.info("Performing anomaly detection")
                anomaly_results = self.anomaly_detector.detect_anomalies(benchmark_data)
                benchmark_result.add_analysis_results("anomalies", anomaly_results)
        
        # Cross-tier analysis
        if len(benchmark_data.get_tiers()) > 1:
            logger.info("Performing cross-tier analysis")
            comparison_results = self.compare_tiers(benchmark_result)
            benchmark_result.add_analysis_results("comparison", comparison_results)
        
        return benchmark_result
    
    def _validate_parameters(self, tiers: List[str], duration: int) -> None:
        """
        Validate benchmark parameters.
        
        Args:
            tiers: List of storage tier paths
            duration: Test duration in seconds
            
        Raises:
            BenchmarkConfigError: If parameters are invalid
        """
        # Check tiers
        if not tiers:
            raise BenchmarkConfigError("At least one storage tier must be specified")
        
        for tier in tiers:
            if not os.path.exists(tier) and not tier.startswith(('/s3://', '/azure://', '/gcs://')):
                raise BenchmarkConfigError(f"Storage tier path does not exist: {tier}")
        
        # Check duration
        if duration < 30:
            raise BenchmarkConfigError("Duration must be at least 30 seconds")
        
        if duration > 86400 and self.production_safe_mode:  # 24 hours
            raise BenchmarkConfigError("Duration cannot exceed 24 hours in production safe mode")
    
    def _calculate_tier_summary(self, tests: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate summary metrics for a tier.
        
        Args:
            tests: Dictionary of test results
            
        Returns:
            Dictionary containing summary metrics
        """
        if not tests:
            return {}
        
        # Extract metrics
        throughputs = [test.get("throughput_MBps", 0) for test in tests.values()]
        iops_values = [test.get("iops", 0) for test in tests.values()]
        latencies = [test.get("latency_ms", 0) for test in tests.values()]
        
        # Calculate averages
        avg_throughput = sum(throughputs) / len(throughputs) if throughputs else 0
        avg_iops = sum(iops_values) / len(iops_values) if iops_values else 0
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        
        return {
            "avg_throughput_MBps": avg_throughput,
            "avg_iops": avg_iops,
            "avg_latency_ms": avg_latency
        }
    
    def _check_resource_limits(self) -> bool:
        """
        Check if system resource usage exceeds configured limits.
        
        Returns:
            True if resource limits exceeded, False otherwise
        """
        if not self.production_safe_mode:
            return False
        
        # Get current resource usage
        cpu_percent = self.system_metrics_collector.get_current_cpu_usage()
        memory_percent = self.system_metrics_collector.get_current_memory_usage()
        
        # Get configured limits
        max_cpu_percent = self.config.get("production_safety.max_cpu_percent", 70)
        max_memory_percent = self.config.get("production_safety.max_memory_percent", 70)
        
        # Check limits
        if cpu_percent > max_cpu_percent:
            logger.warning(f"CPU usage ({cpu_percent}%) exceeds limit ({max_cpu_percent}%)")
            return True
        
        if memory_percent > max_memory_percent:
            logger.warning(f"Memory usage ({memory_percent}%) exceeds limit ({max_memory_percent}%)")
            return True
        
        return False
    
    def _record_event(self, event_type: str, details: Dict[str, Any]) -> None:
        """
        Record a benchmark event.
        
        Args:
            event_type: Type of event
            details: Event details
        """
        event = BenchmarkEvent(event_type, time.time(), details)
        self.events.append(event)
        logger.debug(f"Event recorded: {event_type}")
    
    def _handle_error(self, error_type: str, error_message: str) -> None:
        """
        Handle a benchmark error.
        
        Args:
            error_type: Type of error
            error_message: Error message
        """
        self._record_event("error", {
            "error_type": error_type,
            "error_message": error_message
        })
        
        logger.error(f"{error_type}: {error_message}")
        
        # Reset running state
        self.running = False


if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Enhanced Tiered Storage I/O Benchmark Suite")
    parser.add_argument("--tiers", nargs="+", required=True, help="List of storage tier paths to analyze")
    parser.add_argument("--duration", type=int, default=300, help="Test duration in seconds")
    parser.add_argument("--block-sizes", default="4k,64k,1m", help="Block sizes to test (comma-separated)")
    parser.add_argument("--patterns", default="read,write,randrw", help="I/O patterns to test (comma-separated)")
    parser.add_argument("--output-dir", default="./results", help="Directory for output files")
    parser.add_argument("--formats", default="html,json", help="Output formats (comma-separated)")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--enable-all-analysis", action="store_true", help="Enable all analysis modules")
    parser.add_argument("--production-safe", action="store_true", help="Enable production safety features")
    parser.add_argument("--log-level", default="info", help="Logging level (debug, info, warning, error)")
    
    args = parser.parse_args()
    
    # Create and run benchmark suite
    suite = BenchmarkSuite(args.config, args.log_level)
    
    # Set production safe mode if requested
    if args.production_safe:
        suite.config.set("production_safety.enabled", True)
    
    # Run comprehensive analysis
    result = suite.run_comprehensive_analysis(
        tiers=args.tiers,
        duration=args.duration,
        block_sizes=args.block_sizes.split(","),
        patterns=args.patterns.split(","),
        enable_all_modules=args.enable_all_analysis,
        output_dir=args.output_dir,
        formats=args.formats.split(",")
    )
    
    print(f"Benchmark completed with ID: {result.benchmark_id}")
    print(f"Results available in: {args.output_dir}")
