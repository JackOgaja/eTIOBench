#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLI commands for the benchmark suite (Tiered Storage I/O Benchmark).

This module defines the command line interface for running benchmarks,
analyzing results, and generating reports.

Author: Jack Ogaja
Date: 2025-06-26
"""

import os
import sys
import logging
import argparse
from typing import List, Optional, Dict, Any
from datetime import datetime

from tdiobench.core.benchmark_runner import BenchmarkRunner
from tdiobench.core.benchmark_data import BenchmarkData
from tdiobench.analysis.base_analyzer import BaseAnalyzer
from tdiobench.visualization.chart_generator import ChartGenerator
from tdiobench.cli.utils import setup_logging, load_config, validate_config

logger = logging.getLogger(__name__)


def run_command(args: argparse.Namespace) -> int:
    """
    Run benchmarks according to specified configuration.
    
    Args:
        args: Command line arguments
        
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    
    # Load configuration
    try:
        config = load_config(args.config)
        if not validate_config(config):
            logger.error("Invalid configuration")
            return 1
    except Exception as e:
        logger.error(f"Failed to load configuration: {str(e)}")
        return 1
    
    # Create and run benchmarks
    try:
        runner = BenchmarkRunner(config)
        results = runner.run_benchmarks()
        
        # Save results if requested
        if args.output:
            output_path = args.output
            if not os.path.isabs(output_path):
                output_path = os.path.join(os.getcwd(), output_path)
                
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save in appropriate format
            if output_path.endswith('.json'):
                results.to_json(output_path)
            else:
                results.to_pickle(output_path)
                
            logger.info(f"Benchmark results saved to {output_path}")
            
        return 0
        
    except Exception as e:
        logger.error(f"Error running benchmarks: {str(e)}")
        return 1


def analyze_command(args: argparse.Namespace) -> int:
    """
    Analyze benchmark results.
    
    Args:
        args: Command line arguments
        
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    
    # Load benchmark data
    try:
        if args.input.endswith('.json'):
            data = BenchmarkData.from_json(filepath=args.input)
        else:
            data = BenchmarkData.from_pickle(filepath=args.input)
            
        logger.info(f"Loaded benchmark data from {args.input}")
    except Exception as e:
        logger.error(f"Failed to load benchmark data: {str(e)}")
        return 1
    
    # Run analysis
    try:
        analyzer = BasicAnalyzer(data)
        results = analyzer.analyze()
        
        # Generate report if requested
        if args.report:
            report = analyzer.generate_report(output_path=args.report)
            logger.info(f"Analysis report saved to {args.report}")
            
        # Generate charts if requested
        if args.charts:
            chart_generator = ChartGenerator()
            chart_generator.generate_chart_dashboard(
                data=data,
                chart_configs=[
                    {"type": "line", "metrics": ["throughput"], "title": "Throughput"},
                    {"type": "line", "metrics": ["latency"], "title": "Latency"},
                    {"type": "bar", "metrics": ["iops"], "title": "IOPS"}
                ],
                title="Benchmark Analysis",
                output_path=args.charts
            )
            logger.info(f"Charts saved to {args.charts}")
            
        return 0
        
    except Exception as e:
        logger.error(f"Error analyzing benchmarks: {str(e)}")
        return 1


def list_command(args: argparse.Namespace) -> int:
    """
    List available benchmarks.
    
    Args:
        args: Command line arguments
        
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Get list of available benchmarks
        runner = BenchmarkRunner(config)
        benchmarks = runner.list_available_benchmarks()
        
        # Print benchmark list
        print("\nAvailable Benchmarks:")
        print("-" * 60)
        for i, benchmark in enumerate(benchmarks, 1):
            print(f"{i}. {benchmark['name']}")
            print(f"   Description: {benchmark['description']}")
            print(f"   Category: {benchmark['category']}")
            print(f"   Parameters: {', '.join(benchmark['parameters'])}")
            print("-" * 60)
            
        return 0
        
    except Exception as e:
        logger.error(f"Error listing benchmarks: {str(e)}")
        return 1


def create_parser() -> argparse.ArgumentParser:
    """
    Create command line argument parser.
    
    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Benchmark Suite CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Global options
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set logging level"
    )
    parser.add_argument(
        "--log-file",
        help="Log file path"
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Run benchmarks")
    run_parser.add_argument(
        "--config", "-c",
        required=True,
        help="Path to benchmark configuration file"
    )
    run_parser.add_argument(
        "--output", "-o",
        help="Path to save benchmark results"
    )
    run_parser.set_defaults(func=run_command)
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze benchmark results")
    analyze_parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to benchmark results file"
    )
    analyze_parser.add_argument(
        "--report", "-r",
        help="Path to save analysis report"
    )
    analyze_parser.add_argument(
        "--charts", "-c",
        help="Path to save generated charts"
    )
    analyze_parser.set_defaults(func=analyze_command)
    
    # List command
    list_parser = subparsers.add_parser("list", help="List available benchmarks")
    list_parser.add_argument(
        "--config", "-c",
        default="config/benchmarks.yaml",
        help="Path to benchmark configuration file"
    )
    list_parser.set_defaults(func=list_command)
    
    return parser


def main() -> int:
    """
    Main CLI entry point.
    
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    parser = create_parser()
    args = parser.parse_args()
    
    if not hasattr(args, "func"):
        parser.print_help()
        return 1
        
    try:
        return args.func(args)
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
