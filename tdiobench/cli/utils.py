#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions for the CLI module.

Author: Jack Ogaja
Date: 2025-07-01
"""

import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import colorama
import yaml

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Represents the result of configuration validation."""

    is_valid: bool
    errors: List[str]
    warnings: List[str]
    passed_checks: List[str]
    details: Dict[str, Any]


def setup_logging(
    level: str = "INFO", log_file: Optional[str] = None, verbose: bool = False, quiet: bool = False
) -> None:
    """
    Set up logging configuration.

    Args:
        level: Logging level
        log_file: Optional log file path
        verbose: Enable verbose logging
        quiet: Suppress non-error output
    """
    # Determine effective log level
    if quiet:
        console_level = logging.ERROR
    elif verbose:
        console_level = logging.DEBUG
    else:
        console_level = getattr(logging, level.upper())

    # Create formatters
    console_formatter = logging.Formatter("%(levelname)s: %(message)s")
    file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler
    if not quiet:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)


def setup_colors(no_color: bool = False) -> None:
    """
    Set up colored output.

    Args:
        no_color: Disable colored output
    """
    if not no_color and sys.stdout.isatty():
        colorama.init(autoreset=True)
    else:
        # Disable colorama
        colorama.init(strip=True, convert=False)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
        json.JSONDecodeError: If JSON parsing fails
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        if config_path.suffix.lower() in [".yaml", ".yml"]:
            config = yaml.safe_load(f)
        elif config_path.suffix.lower() == ".json":
            config = json.load(f)
        else:
            # Try YAML first, then JSON
            content = f.read()
            try:
                config = yaml.safe_load(content)
            except yaml.YAMLError:
                config = json.loads(content)

    # Apply environment variable substitutions
    config = _substitute_env_vars(config)

    return config


def validate_config(config: Dict[str, Any], strict: bool = False) -> ValidationResult:
    """
    Validate benchmark configuration.

    Args:
        config: Configuration dictionary
        strict: Enable strict validation mode

    Returns:
        ValidationResult object
    """
    errors = []
    warnings = []
    passed_checks = []
    details = {}

    # Check required top-level sections
    required_sections = ["benchmarks", "storage", "global"]
    for section in required_sections:
        if section not in config:
            errors.append(f"Missing required section: {section}")
        else:
            passed_checks.append(f"Required section '{section}' present")

    # Validate global settings
    if "global" in config:
        global_config = config["global"]

        # Check required global settings
        required_global = ["iterations", "timeout"]
        for setting in required_global:
            if setting not in global_config:
                if strict:
                    errors.append(f"Missing required global setting: {setting}")
                else:
                    warnings.append(f"Missing recommended global setting: {setting}")
            else:
                passed_checks.append(f"Global setting '{setting}' present")

        # Validate setting values
        if "iterations" in global_config:
            iterations = global_config["iterations"]
            if not isinstance(iterations, int) or iterations <= 0:
                errors.append("'iterations' must be a positive integer")
            elif iterations < 3:
                warnings.append("Low iteration count may produce unreliable results")
            else:
                passed_checks.append("Iterations setting is valid")

        if "timeout" in global_config:
            timeout = global_config["timeout"]
            if not isinstance(timeout, (int, float)) or timeout <= 0:
                errors.append("'timeout' must be a positive number")
            else:
                passed_checks.append("Timeout setting is valid")

    # Validate benchmarks section
    if "benchmarks" in config:
        benchmarks = config["benchmarks"]

        if not isinstance(benchmarks, dict):
            errors.append("'benchmarks' section must be a dictionary")
        elif not benchmarks:
            warnings.append("No benchmarks defined")
        else:
            passed_checks.append(f"Found {len(benchmarks)} benchmark definitions")

            # Validate individual benchmarks
            for name, benchmark in benchmarks.items():
                if not isinstance(benchmark, dict):
                    errors.append(f"Benchmark '{name}' must be a dictionary")
                    continue

                # Check required benchmark fields
                required_fields = ["type", "parameters"]
                missing_fields = [f for f in required_fields if f not in benchmark]
                if missing_fields:
                    errors.append(f"Benchmark '{name}' missing required fields: {missing_fields}")
                else:
                    passed_checks.append(f"Benchmark '{name}' has required fields")

    # Validate storage section
    if "storage" in config:
        storage = config["storage"]

        if not isinstance(storage, dict):
            errors.append("'storage' section must be a dictionary")
        elif "tiers" not in storage:
            errors.append("'storage' section missing 'tiers' configuration")
        else:
            tiers = storage["tiers"]
            if not isinstance(tiers, list) or not tiers:
                errors.append("'storage.tiers' must be a non-empty list")
            else:
                passed_checks.append(f"Found {len(tiers)} storage tier definitions")

                # Validate tier definitions
                for i, tier in enumerate(tiers):
                    if not isinstance(tier, dict):
                        errors.append(f"Storage tier {i} must be a dictionary")
                        continue

                    if "name" not in tier:
                        errors.append(f"Storage tier {i} missing 'name' field")
                    elif "path" not in tier:
                        errors.append(f"Storage tier {i} missing 'path' field")
                    else:
                        # Check if path exists
                        tier_path = Path(tier["path"])
                        if not tier_path.exists():
                            if strict:
                                errors.append(
                                    f"Storage tier '{tier['name']}' path does not exist: {tier['path']}"
                                )
                            else:
                                warnings.append(
                                    f"Storage tier '{tier['name']}' path does not exist: {tier['path']}"
                                )
                        elif not os.access(tier_path, os.W_OK):
                            errors.append(
                                f"Storage tier '{tier['name']}' path is not writable: {tier['path']}"
                            )
                        else:
                            passed_checks.append(
                                f"Storage tier '{tier['name']}' path is accessible"
                            )

    # Build details
    details = {
        "config_sections": list(config.keys()),
        "benchmark_count": len(config.get("benchmarks", {})),
        "storage_tier_count": len(config.get("storage", {}).get("tiers", [])),
        "validation_timestamp": datetime.now().isoformat(),
    }

    is_valid = len(errors) == 0

    return ValidationResult(
        is_valid=is_valid,
        errors=errors,
        warnings=warnings,
        passed_checks=passed_checks,
        details=details,
    )


def get_version() -> Dict[str, str]:
    """
    Get version information.

    Returns:
        Dictionary containing version information
    """
    import platform

    import tdiobench

    return {
        "version": getattr(tdiobench, "__version__", "unknown"),
        "build": getattr(tdiobench, "__build__", "unknown"),
        "date": getattr(tdiobench, "__date__", "unknown"),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }


def format_table(headers: List[str], rows: List[List[str]]) -> str:
    """
    Format data as a table.

    Args:
        headers: Table headers
        rows: Table rows

    Returns:
        Formatted table string
    """
    if not rows:
        return "No data to display."

    # Calculate column widths
    col_widths = [len(header) for header in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(col_widths):
                col_widths[i] = max(col_widths[i], len(str(cell)))

    # Build table
    lines = []

    # Header
    header_line = " | ".join(header.ljust(col_widths[i]) for i, header in enumerate(headers))
    lines.append(header_line)
    lines.append("-" * len(header_line))

    # Rows
    for row in rows:
        row_line = " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row))
        lines.append(row_line)

    return "\n".join(lines)


def _substitute_env_vars(config: Union[Dict, List, str, Any]) -> Any:
    """
    Recursively substitute environment variables in configuration.

    Args:
        config: Configuration value to process

    Returns:
        Configuration with environment variables substituted
    """
    if isinstance(config, dict):
        return {key: _substitute_env_vars(value) for key, value in config.items()}
    elif isinstance(config, list):
        return [_substitute_env_vars(item) for item in config]
    elif isinstance(config, str):
        return os.path.expandvars(config)
    else:
        return config


def _confirm_delete(filepath: Path) -> bool:
    """
    Confirm deletion of a file.

    Args:
        filepath: Path to file

    Returns:
        True if user confirms deletion
    """
    response = input(f"Delete {filepath}? [y/N]: ").strip().lower()
    return response in ["y", "yes"]


def _get_analyzer_class(analyzer_name: str):
    """
    Get analyzer class by name.

    Args:
        analyzer_name: Name of analyzer

    Returns:
        Analyzer class
    """
    from tdiobench.analysis.advanced_analyzer import AdvancedAnalyzer
    from tdiobench.analysis.basic_analyzer import BasicAnalyzer
    from tdiobench.analysis.statistical_analyzer import StatisticalAnalyzer

    analyzers = {
        "basic": BasicAnalyzer,
        "advanced": AdvancedAnalyzer,
        "statistical": StatisticalAnalyzer,
    }

    return analyzers.get(analyzer_name.lower(), BasicAnalyzer)


def _apply_filters(data, filters_json: str):
    """
    Apply filters to benchmark data.

    Args:
        data: Benchmark data
        filters_json: JSON string containing filter criteria

    Returns:
        Filtered data
    """
    filters = json.loads(filters_json)

    # Implementation would depend on the BenchmarkData class structure
    # This is a placeholder for the actual filtering logic
    filtered_data = data.apply_filters(filters)

    return filtered_data


def _build_chart_configs(results, chart_types: List[str]) -> List[Dict[str, Any]]:
    """
    Build chart configurations based on analysis results.

    Args:
        results: Analysis results
        chart_types: Types of charts to generate

    Returns:
        List of chart configurations
    """
    configs = []

    for chart_type in chart_types:
        if chart_type == "line":
            configs.append(
                {
                    "type": "line",
                    "metrics": ["throughput", "latency"],
                    "title": "Performance Over Time",
                }
            )
        elif chart_type == "bar":
            configs.append({"type": "bar", "metrics": ["iops"], "title": "IOPS Comparison"})
        elif chart_type == "scatter":
            configs.append(
                {
                    "type": "scatter",
                    "x_metric": "latency",
                    "y_metric": "throughput",
                    "title": "Latency vs Throughput",
                }
            )
        elif chart_type == "heatmap":
            configs.append(
                {
                    "type": "heatmap",
                    "metrics": ["cpu_usage", "memory_usage"],
                    "title": "Resource Usage Heatmap",
                }
            )
        elif chart_type == "box":
            configs.append(
                {"type": "box", "metrics": ["response_time"], "title": "Response Time Distribution"}
            )

    return configs


def _export_analysis_results(results, output_path: str, format: str) -> None:
    """
    Export analysis results in specified format.

    Args:
        results: Analysis results
        output_path: Output file path
        format: Export format
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "json":
        with open(output_path, "w") as f:
            json.dump(results.to_dict(), f, indent=2, default=str)
    elif format == "yaml":
        with open(output_path, "w") as f:
            yaml.dump(results.to_dict(), f, default_flow_style=False)
    elif format == "csv":
        results.to_csv(output_path)
    else:
        raise ValueError(f"Unsupported export format: {format}")


def _print_comparison_summary(comparison_results) -> None:
    """
    Print comparison summary.

    Args:
        comparison_results: Results from benchmark comparison
    """
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)

    for metric, comparison in comparison_results.items():
        print(f"\n{metric.upper()}:")
        print(f"  Best: {comparison['best']['name']} ({comparison['best']['value']:.2f})")
        print(f"  Worst: {comparison['worst']['name']} ({comparison['worst']['value']:.2f})")
        print(
            f"  Difference: {comparison['difference']:.2f} ({comparison['difference_percent']:.1f}%)"
        )


# Placeholder implementations for missing functions
def _create_config(args) -> int:
    """Create new configuration file."""
    # Implementation would create a new config file from template
    logger.info(f"Creating configuration file: {args.output}")
    return 0


def _edit_config(args) -> int:
    """Edit configuration file."""
    # Implementation would open config file in editor
    logger.info(f"Editing configuration file: {args.config}")
    return 0


def _show_config(args) -> int:
    """Show configuration file contents."""
    # Implementation would display config file contents
    config = load_config(args.config)
    if args.format == "json":
        print(json.dumps(config, indent=2))
    else:
        print(yaml.dump(config, default_flow_style=False))
    return 0


def _list_results(args) -> int:
    """List stored results."""
    # Implementation would list available result files
    logger.info("Listing stored results")
    return 0


def _show_results(args) -> int:
    """Show result details."""
    # Implementation would show details of specific result
    logger.info(f"Showing results: {args.result_id}")
    return 0


def _delete_results(args) -> int:
    """Delete results."""
    # Implementation would delete specified results
    logger.info(f"Deleting results: {args.result_ids}")
    return 0


def _archive_results(args) -> int:
    """Archive old results."""
    # Implementation would archive old results
    logger.info(f"Archiving results older than {args.older_than} days")
    return 0
