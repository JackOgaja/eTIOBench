#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eTIOBench - Unified CLI commands for the benchmark suite (Tiered Storage I/O Benchmark).

This module defines a comprehensive command line interface for running benchmarks,
analyzing results, generating reports, and managing configurations.

Author: Jack Ogaja
Date: 2025-06-29
"""

import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
import psutil
import yaml

from tdiobench.cli.profile_manager import ProfileManager
from tdiobench.cli.safety_controller import SafetyController

# Import the necessary benchmark modules
from tdiobench.core.benchmark_suite import BenchmarkSuite
from tdiobench.utils.data_processor import DataAggregator

# Import visualization modules
from tdiobench.visualization.report_generator import ReportGenerator
from tdiobench.visualization.chart_generator import ChartGenerator
from tdiobench.core.benchmark_config import BenchmarkConfig

# Enhanced charts are available with the visualization module
ENHANCED_CHARTS_AVAILABLE = True

__version__ = "1.0.0"

# Environment variable prefix
ENV_PREFIX = "BENCHMARK_"

# Setup logging
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("benchmark-suite")


class BenchmarkContext:
    """Enhanced context for benchmark operations"""

    def __init__(self):
        self.config = {}
        self.results_dir = Path("./results")
        self.suite = None  # BenchmarkSuite instance


def get_env_config():
    """Load configuration from environment variables"""
    env_config = {}
    for key, value in os.environ.items():
        if key.startswith(ENV_PREFIX):
            config_key = key[len(ENV_PREFIX):].lower().replace("_", ".")
            config_key = "benchmark_suite." + config_key
            try:
                env_config[config_key] = json.loads(value)
            except BaseException:
                env_config[config_key] = value
    return env_config


def load_config(config_file: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration with proper precedence"""
    # Default configuration
    config = {
        "benchmark_suite.core.safety.enabled": True,
        "benchmark_suite.core.safety.max_cpu_percent": 90,
        "benchmark_suite.core.safety.max_memory_percent": 90,
        "benchmark_suite.core.logging.level": "info",
        "benchmark_suite.collection.time_series.enabled": True,
        "benchmark_suite.collection.time_series.interval": 1.0,
        "benchmark_suite.collection.system_metrics.enabled": True,
        "benchmark_suite.collection.system_metrics.interval": 5.0,
        "benchmark_suite.analysis.statistics.enabled": True,
        "benchmark_suite.analysis.statistics.confidence_level": 95.0,
        "benchmark_suite.visualization.reports.formats": ["html", "json"],
        "benchmark_suite.results.base_dir": "./results",
        "benchmark_suite.engines.fio.path": "fio",
        "benchmark_suite.engines.fio.cleanup_test_files": True,
    }

    # Load from file
    if config_file and Path(config_file).exists():
        with open(config_file, "r") as f:
            if config_file.endswith((".yaml", ".yml")):
                file_config = yaml.safe_load(f)
            else:
                file_config = json.load(f)
            config.update(file_config)

    # Override with environment
    config.update(get_env_config())

    return config


def validate_config(config: Dict[str, Any]) -> List[str]:
    """Validate configuration and return list of issues"""
    issues = []

    # Check safety limits
    cpu_limit = config.get("benchmark_suite.core.safety.max_cpu_percent", 90)
    if not 0 < cpu_limit <= 100:
        issues.append(f"Invalid CPU limit: {cpu_limit}")

    memory_limit = config.get("benchmark_suite.core.safety.max_memory_percent", 90)
    if not 0 < memory_limit <= 100:
        issues.append(f"Invalid memory limit: {memory_limit}")

    # Check paths
    results_dir = config.get("benchmark_suite.results.base_dir", "./results")
    if not Path(results_dir).parent.exists():
        issues.append(f"Results directory parent does not exist: {results_dir}")

    return issues


@click.group()
@click.option(
    "--config", type=click.Path(exists=True), help="Path to configuration file (JSON or YAML)"
)
@click.option(
    "--log-level",
    type=click.Choice(["debug", "info", "warning", "error", "critical"]),
    default="info",
    help="Logging level",
)
@click.option("--output-dir", type=click.Path(), help="Output directory for results and reports")
@click.version_option(version=__version__)
@click.pass_context
def cli(ctx, config, log_level, output_dir):
    """eTIOBench - Enhanced Tiered Storage I/O Benchmark Suite

    Professional-grade storage performance analysis platform for multi-tier storage environments.
    """
    # Initialize context
    ctx.ensure_object(BenchmarkContext)

    # Load configuration
    ctx.obj.config = load_config(config)

    # Override with CLI options
    if output_dir:
        ctx.obj.config["benchmark_suite.results.base_dir"] = output_dir

    # Setup logging
    logger.setLevel(getattr(logging, log_level.upper()))

    # Create results directory
    ctx.obj.results_dir = Path(ctx.obj.config.get("benchmark_suite.results.base_dir", "./results"))
    ctx.obj.results_dir.mkdir(parents=True, exist_ok=True)

    # Initialize BenchmarkSuite instance
    ctx.obj.suite = BenchmarkSuite(config_path=config)

    # Initialize ProfileManager instance
    # + ctx.obj.profile_manager = ProfileManager()

    # Initialize ProfileManager instance
    # + ctx.obj.safety_controller = SafetyController()


@cli.command()
@click.option(
    "--tiers", required=True, help="Comma-separated list of storage tier paths to benchmark"
)
@click.option("--duration", type=int, default=60, help="Duration of each benchmark test in seconds")
@click.option("--block-sizes", default="4k,64k,1m", help="Comma-separated list of block sizes")
@click.option(
    "--patterns", default="read,write,randrw", help="Comma-separated list of I/O patterns"
)
@click.option("--io-depth", type=int, default=32, help="I/O queue depth for benchmark tests")
@click.option("--num-jobs", type=int, default=1, help="Number of parallel jobs")
@click.option("--rate-limit", help="I/O rate limit (e.g., 50m for 50MB/s)")
@click.option("--direct/--no-direct", default=True, help="Use direct I/O (bypass cache)")
@click.option(
    "--time-series", is_flag=True, default=False, help="Enable time series data collection"
)
@click.option(
    "--ts-interval", type=float, default=1.0, help="Time series collection interval in seconds"
)
@click.option(
    "--system-metrics", is_flag=True, default=True, help="Enable system metrics collection"
)
@click.option(
    "--sm-interval", type=float, default=5.0, help="System metrics collection interval in seconds"
)
@click.option("--network-analysis", is_flag=True, help="Enable network impact analysis")
@click.option("--monitor-network", help="Network interfaces to monitor (comma-separated)")
@click.option("--analysis-types", help="Comma-separated list of analysis types to perform")
@click.option("--baseline-tier", help="Baseline tier for comparison")
@click.option("--save", is_flag=True, default=True, help="Save benchmark results")
@click.option("--reports", default="html,json", help="Comma-separated list of report formats")
@click.option(
    "--profile",
    type=click.Choice(["quick_scan", "production_safe", "comprehensive", "latency_focused"]),
    help="Use predefined benchmark profile",
)
@click.option("--tag", help="Tag for this benchmark run")
@click.option("--dry-run", is_flag=True, help="Show what would be run without executing")
@click.option("--no-safety", is_flag=True, help="Disable safety checks (dangerous!)")
@click.option("--quiet", is_flag=True, help="Suppress progress output")
@click.pass_context
def run(
    ctx,
    tiers,
    duration,
    block_sizes,
    patterns,
    io_depth,
    num_jobs,
    rate_limit,
    direct,
    time_series,
    ts_interval,
    system_metrics,
    sm_interval,
    network_analysis,
    monitor_network,
    analysis_types,
    baseline_tier,
    save,
    reports,
    profile,
    tag,
    dry_run,
    no_safety,
    quiet,
):
    """Execute storage benchmarks with integrated safety monitoring"""

    suite = ctx.obj.suite
    profile_manager = ProfileManager()
    safety_controller = SafetyController()

    # Parse tier list
    tier_list = [t.strip() for t in tiers.split(",")]

    # Validate tiers using TierManager - do this before dry-run so validation works in dry-run mode
    for tier_path in tier_list:
        if not suite.tier_manager.validate_tier(tier_path):
            click.echo(f"Error: Invalid tier path: {tier_path}", err=True)
            sys.exit(1)
    
    # Validate duration early for both dry-run and actual execution
    if duration < 30:
        click.echo("Error: Duration must be at least 30 seconds", err=True)
        sys.exit(1)
        
    if duration > 86400:  # 24 hours
        click.echo("Error: Duration cannot exceed 24 hours", err=True)
        sys.exit(1)

    # Get profile configuration
    if profile:
        profile_config = profile_manager.get_profile(profile)
        if not profile_config:
            click.echo(f"Error: Unknown profile: {profile}", err=True)
            sys.exit(1)
        
        # Override profile parameters with CLI parameters (CLI takes precedence)
        profile_config["duration"] = duration  # Always use CLI duration
        if block_sizes:
            profile_config["block_sizes"] = block_sizes.split(",")
        if patterns:
            profile_config["patterns"] = patterns.split(",")
        if io_depth:
            profile_config["io_depth"] = io_depth
        if num_jobs:
            profile_config["num_jobs"] = num_jobs
        if rate_limit:
            profile_config["rate_limit"] = rate_limit
        if direct is not None:
            profile_config["direct"] = direct
    else:
        # Create custom profile from CLI options
        profile_config = profile_manager.create_custom_profile(
            {
                "duration": duration,
                "block_sizes": block_sizes.split(","),
                "patterns": patterns.split(","),
                "io_depth": io_depth,
                "num_jobs": num_jobs,
                "rate_limit": rate_limit,
                "direct": direct,
            }
        )

    # Configure data collection
    collection_config = {
        "time_series": {"enabled": time_series, "interval": ts_interval},
        "system_metrics": {
            "enabled": system_metrics,
            "interval": sm_interval,
            "network_enabled": network_analysis,
            "interfaces": monitor_network.split(",") if monitor_network else None,
        },
    }

    # Safety configuration
    safety_config = {
        "enabled": not no_safety,
        "max_cpu_percent": ctx.obj.config.get("benchmark_suite.core.safety.max_cpu_percent", 90),
        "max_memory_percent": ctx.obj.config.get(
            "benchmark_suite.core.safety.max_memory_percent", 90
        ),
    }

    if dry_run:
        click.echo("DRY RUN - Benchmark configuration:")
        click.echo(
            json.dumps(
                {
                    "tiers": tier_list,
                    "profile": profile_config,
                    "collection": collection_config,
                    "safety": safety_config,
                    "tag": tag,
                },
                indent=2,
            )
        )
        return

    if not quiet:
        click.echo(f"Running benchmarks on {len(tier_list)} tiers...")

    try:
        # Apply collection configuration to the suite
        suite.config.set("collection.time_series.enabled", collection_config["time_series"]["enabled"])
        suite.config.set("collection.time_series.interval", collection_config["time_series"]["interval"])
        suite.config.set("collection.system_metrics.enabled", collection_config["system_metrics"]["enabled"])
        suite.config.set("collection.system_metrics.interval", collection_config["system_metrics"]["interval"])
        suite.config.set("collection.system_metrics.network_enabled", collection_config["system_metrics"]["network_enabled"])

        # Execute benchmark using BenchmarkSuite comprehensive analysis
        benchmark_result = suite.run_comprehensive_analysis(
            tiers=tier_list,
            duration=profile_config.get("duration", 60),
            block_sizes=profile_config.get("block_sizes", ["4k"]),
            patterns=profile_config.get("patterns", ["read"]),
            **{k: v for k, v in profile_config.items() 
               if k not in ["duration", "block_sizes", "patterns"]}
        )

        # Results are automatically saved by run_comprehensive_analysis
        if save and not quiet:
            click.echo(f"\nResults saved with ID: {benchmark_result.benchmark_id if hasattr(benchmark_result, 'benchmark_id') else benchmark_result.run_id}")

        # Run analysis if requested  
        if analysis_types:
            analysis_list = [a.strip() for a in analysis_types.split(",")]
            # Analysis is already included in comprehensive analysis
            if not quiet:
                click.echo(f"Analysis included: {', '.join(analysis_list)}")

        # Generate reports
        if reports:
            report_formats = [f.strip() for f in reports.split(",")]
            
            generated_reports = suite.generate_reports(
                benchmark_result=benchmark_result, 
                formats=report_formats
            )

            if not quiet:
                for fmt, path in generated_reports.items():
                    click.echo(f"{fmt.upper()} report: {path}")

        if not quiet:
            # Display custom summary for BenchmarkResult
            lines = []
            lines.append("\n📊 Benchmark Summary")
            lines.append("=" * 50)
            lines.append(f"Benchmark ID: {getattr(benchmark_result, 'benchmark_id', benchmark_result.run_id)}")
            lines.append(f"Profile: {benchmark_result.profile_name}")
            lines.append(f"Start Time: {benchmark_result.start_time}")
            lines.append(f"Duration: {benchmark_result.duration_seconds:.1f} seconds")
            lines.append("")
            
            # Display tier results
            if hasattr(benchmark_result, 'tiers') and benchmark_result.tiers:
                lines.append(f"Tested {len(benchmark_result.tiers)} tier(s):")
                for tier in benchmark_result.tiers:
                    if tier in benchmark_result.metrics:
                        tier_metrics = benchmark_result.metrics[tier]
                        test_count = len(tier_metrics)
                        lines.append(f"  ✅ {tier}: {test_count} test(s) completed")
                    else:
                        lines.append(f"  ⚠️  {tier}: No metrics available")
            else:
                lines.append("No tier results available")
            
            lines.append("")
            click.echo("\n".join(lines))

    except KeyboardInterrupt:
        click.echo("\nBenchmark interrupted by user")
        if safety_config["enabled"] and hasattr(suite, "safety_controller"):
            safety_controller.emergency_stop()
        sys.exit(1)
    except Exception as e:
        logger.error(f"Benchmark failed: {str(e)}")
        sys.exit(1)
    finally:
        if safety_config["enabled"] and hasattr(suite, "safety_controller"):
            safety_controller.stop_monitoring()


@cli.command()
@click.option("--benchmark-id", required=True, help="Benchmark ID to analyze")
@click.option("--analysis-types", help="Comma-separated list of analysis types")
@click.option("--confidence-level", type=float, default=95.0, help="Confidence level (0-100)")
@click.option(
    "--anomaly-threshold", type=float, default=3.0, help="Threshold for anomaly detection"
)
@click.option("--time-series-decomp", is_flag=True, help="Enable time series decomposition")
@click.option("--trend-detection", is_flag=True, help="Enable trend detection")
@click.option("--seasonality-detection", is_flag=True, help="Enable seasonality detection")
@click.option("--correlation-analysis", is_flag=True, help="Enable correlation analysis")
@click.option("--forecast", is_flag=True, help="Enable performance forecasting")
@click.option("--network-analysis", is_flag=True, help="Enable network impact analysis")
@click.option("--outlier-detection", is_flag=True, help="Enable outlier detection")
@click.option("--output-file", type=click.Path(), help="Output file for analysis results")
@click.pass_context
def analyze(
    ctx,
    benchmark_id,
    analysis_types,
    confidence_level,
    anomaly_threshold,
    time_series_decomp,
    trend_detection,
    seasonality_detection,
    correlation_analysis,
    forecast,
    network_analysis,
    outlier_detection,
    output_file,
):
    """Analyze benchmark results with advanced statistical methods"""

    suite = ctx.obj.suite

    # Determine analysis types
    if analysis_types:
        analysis_list = [a.strip() for a in analysis_types.split(",")]
    else:
        analysis_list = []
        if outlier_detection or not any(
            [time_series_decomp, trend_detection, seasonality_detection, network_analysis]
        ):
            analysis_list.append("statistical")
        if anomaly_threshold != 3.0:
            analysis_list.append("anomaly")
        if any(
            [
                time_series_decomp,
                trend_detection,
                seasonality_detection,
                correlation_analysis,
                forecast,
            ]
        ):
            analysis_list.append("time_series")
        if network_analysis:
            analysis_list.append("network")

    # Configure analysis options
    analysis_config = {
        "confidence_level": confidence_level,
        "anomaly_threshold": anomaly_threshold,
        "time_series_decomp": time_series_decomp,
        "trend_detection": trend_detection,
        "seasonality_detection": seasonality_detection,
        "correlation_analysis": correlation_analysis,
        "forecast": forecast,
        "outlier_detection": outlier_detection,
    }

    try:
        # Run analysis using BenchmarkSuite
        analysis_results = suite.analyze_results(
            benchmark_id=benchmark_id, analysis_types=analysis_list, analysis_config=analysis_config
        )

        # Save or display results
        if output_file:
            with open(output_file, "w") as f:
                json.dump(analysis_results, f, indent=2, default=str)
            click.echo(f"Analysis saved to: {output_file}")
        else:
            click.echo(json.dumps(analysis_results, indent=2, default=str))

    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        sys.exit(1)


@cli.command()
@click.option("--benchmark-id", help="Benchmark ID to generate reports for")
@click.option("--last", is_flag=True, help="Generate report for last benchmark")
@click.option("--formats", default="html,json,csv", help="Comma-separated list of report formats")
@click.option("--output-dir", type=click.Path(), help="Output directory for reports")
@click.option("--report-title", help="Title for generated reports")
@click.option("--display-label", help="Custom label for charts and report titles (defaults to benchmark ID)")
@click.option("--include-charts", is_flag=True, help="Include charts in reports")
@click.option("--chart-types", default="enhanced_dashboard,performance_heatmap,operation_comparison", 
              help="Comma-separated list of enhanced chart types: enhanced_dashboard, performance_heatmap, operation_comparison, block_size_trends, all")
@click.option("--chart-width", type=int, default=800, help="Width of charts in pixels")
@click.option("--chart-height", type=int, default=400, help="Height of charts in pixels")
@click.option("--include-raw", is_flag=True, help="Include raw benchmark data")
@click.option("--executive-summary", is_flag=True, help="Generate executive summary only")
@click.option(
    "--recommendations", is_flag=True, help="Include performance recommendations"
)
@click.option("--use-fallback", is_flag=True, help="Use fallback methods instead of visualization modules")
@click.option("--email-recipients", help="Comma-separated list of email recipients")
@click.pass_context
def report(
    ctx,
    benchmark_id,
    last,
    formats,
    output_dir,
    report_title,
    display_label,
    include_charts,
    chart_types,
    chart_width,
    chart_height,
    include_raw,
    executive_summary,
    recommendations,
    use_fallback,
    email_recipients,
):
    """Generate comprehensive benchmark reports"""

    # Get benchmark ID
    if last:
        benchmark_id = get_last_benchmark_id(ctx.obj.results_dir)
    if not benchmark_id:
        click.echo("Error: No benchmark ID specified", err=True)
        sys.exit(1)

    # Load results
    results_path = ctx.obj.results_dir / f"{benchmark_id}.json"
    if not results_path.exists():
        # Try alternative path structure: benchmark_id/result.json
        results_path = ctx.obj.results_dir / benchmark_id / "result.json"
    
    if not results_path.exists():
        click.echo(f"Error: Benchmark {benchmark_id} not found", err=True)
        sys.exit(1)

    with open(results_path, "r") as f:
        results = json.load(f)

    # Generate reports
    # Use display_label for the title if provided, otherwise use benchmark_id
    effective_label = display_label or benchmark_id
    report_config = {
        "title": report_title or f"Benchmark Report - {effective_label}",
        "display_label": effective_label,
        "benchmark_id": benchmark_id,  # Keep original ID for file naming and metadata
        "include_charts": include_charts,
        "chart_types": chart_types.split(","),
        "chart_width": chart_width,
        "chart_height": chart_height,
        "include_raw": include_raw,
        "executive_summary_only": executive_summary,
        "recommendations": recommendations,
        "use_fallback": use_fallback,
    }

    output_dir = Path(output_dir) if output_dir else ctx.obj.results_dir
    format_list = [f.strip() for f in formats.split(",")]
    
    # Show enhanced reporting status if flags are used
    if executive_summary or recommendations:
        click.echo(f"📊 Enhanced reporting enabled - generating formats: {', '.join(sorted(format_list))}")

    generate_reports(output_dir, benchmark_id, results, format_list, ctx.obj.config, report_config)

    if email_recipients:
        click.echo(f"Email functionality not implemented. Recipients: {email_recipients}")


@cli.command()
@click.option("--benchmark-ids", required=True, help="Comma-separated list of benchmark IDs")
@click.option("--tiers", help="Comma-separated list of tiers to compare")
@click.option("--baseline-tier", help="Baseline tier for comparison")
@click.option("--baseline-id", help="Baseline benchmark ID")
@click.option("--metrics", default="iops,bandwidth,latency", help="Comma-separated list of metrics")
@click.option("--output-file", type=click.Path(), help="Output file for comparison")
@click.option("--format", type=click.Choice(["json", "html", "markdown"]), default="json")
@click.option("--generate-report", is_flag=True, help="Generate comparison report")
@click.pass_context
def compare(
    ctx,
    benchmark_ids,
    tiers,
    baseline_tier,
    baseline_id,
    metrics,
    output_file,
    format,
    generate_report,
):
    """Compare multiple benchmark results"""

    suite = ctx.obj.suite
    id_list = [bid.strip() for bid in benchmark_ids.split(",")]

    try:
        # Use BenchmarkSuite's comparison functionality
        comparison_results = suite.compare_benchmarks(
            benchmark_ids=id_list,
            metrics=metrics.split(","),
            tiers=tiers.split(",") if tiers else None,
            baseline_tier=baseline_tier,
            baseline_id=baseline_id,
        )

        # Format output based on requested format
        if format == "json":
            output = json.dumps(comparison_results, indent=2, default=str)
        elif format == "markdown":
            aggregator = DataAggregator()
            output = aggregator.format_results(comparison_results, format="markdown")
        elif format == "html" and generate_report:
            output = suite.generate_reports(
                benchmark_id="comparison", formats=["html"], data=comparison_results
            )["html"]

        # Save or display
        if output_file:
            with open(output_file, "w") as f:
                f.write(output)
            click.echo(f"Comparison saved to: {output_file}")
        else:
            click.echo(output)

    except Exception as e:
        logger.error(f"Comparison failed: {str(e)}")
        sys.exit(1)


@cli.command(name="list")
@click.option("--limit", type=int, default=10, help="Maximum number of results")
@click.option("--offset", type=int, default=0, help="Offset for pagination")
@click.option("--sort-by", type=click.Choice(["timestamp", "duration"]), default="timestamp")
@click.option("--sort-order", type=click.Choice(["asc", "desc"]), default="desc")
@click.option("--filter", "filter_expr", help="Filter expression")
@click.option("--tags", help="Filter by tags (comma-separated)")
@click.pass_context
def list_benchmarks(ctx, limit, offset, sort_by, sort_order, filter_expr, tags):
    """List available benchmark results"""

    benchmarks = []

    # Get all benchmark files
    for file in sorted(ctx.obj.results_dir.glob("*.json")):
        # Skip metadata and report files
        if file.stem == "metadata" or file.stem.endswith("_report") or file.stem == "report":
            continue

        try:
            with open(file, "r") as f:
                data = json.load(f)

            # Extract metadata - handle both old and new formats
            benchmark = {
                "id": file.stem,
                "timestamp": data.get("timestamp") or data.get("start_time", "Unknown"),
                "duration": data.get("duration") or data.get("duration_seconds", 0),
                "tiers": list(data.get("results", {}).keys()) or [data.get("tier_name", "")],
                "tag": data.get("tag", ""),
                "profile": data.get("profile") or data.get("profile_name", "custom"),
            }

            # Apply filters
            if filter_expr and not eval_filter(benchmark, filter_expr):
                continue
            if tags and benchmark["tag"] not in tags.split(","):
                continue

            benchmarks.append(benchmark)

        except Exception as e:
            logger.warning(f"Error reading {file}: {e}")

    # Sort
    reverse = sort_order == "desc"
    benchmarks.sort(key=lambda x: x[sort_by], reverse=reverse)

    # Paginate
    benchmarks = benchmarks[offset: offset + limit]

    # Display
    if not benchmarks:
        click.echo("No benchmarks found")
        return

    click.echo(f"{'ID':<25} {'Timestamp':<20} {'Duration':<10} {'Profile':<15} {'Tiers'}")
    click.echo("-" * 90)

    for b in benchmarks:
        tiers_str = ", ".join(b["tiers"][:3])
        if len(b["tiers"]) > 3:
            tiers_str += f" (+{len(b['tiers']) - 3} more)"
        click.echo(
            f"{b['id']:<25} {b['timestamp']:<20} {b['duration']:<10} "
            f"{b['profile']:<15} {tiers_str}"
        )


@cli.command()
@click.option("--config", "-c", type=click.Path(exists=True), help="Configuration file to validate")
@click.option("--check-paths", is_flag=True, help="Verify storage paths exist")
@click.option("--check-tools", is_flag=True, help="Verify required tools installed")
@click.option("--check-permissions", is_flag=True, help="Verify file permissions")
@click.option("--check-resources", is_flag=True, help="Check system resources")
@click.option("--fix", is_flag=True, help="Attempt to fix issues")
@click.pass_context
def validate(ctx, config, check_paths, check_tools, check_permissions, check_resources, fix):
    """Validate configuration and environment"""

    config or ctx.obj.config.get("config_file")
    issues = []

    click.echo("Validating configuration...")

    # Validate config
    config_issues = validate_config(ctx.obj.config)
    issues.extend([("config", issue) for issue in config_issues])

    # Check tools
    if check_tools:
        click.echo("Checking required tools...")
        tools = ["fio", "dd"]
        for tool in tools:
            if not check_tool_exists(tool):
                issues.append(("tool", f"{tool} not found in PATH"))

    # Check paths
    if check_paths:
        click.echo("Checking paths...")
        results_dir = ctx.obj.config.get("benchmark_suite.results.base_dir")
        if not Path(results_dir).exists():
            if fix:
                Path(results_dir).mkdir(parents=True, exist_ok=True)
                click.echo(f"Created: {results_dir}")
            else:
                issues.append(("path", f"Results directory does not exist: {results_dir}"))

    # Check permissions
    if check_permissions:
        click.echo("Checking permissions...")
        results_dir = Path(ctx.obj.config.get("benchmark_suite.results.base_dir"))
        if results_dir.exists() and not os.access(results_dir, os.W_OK):
            issues.append(("permission", f"No write permission: {results_dir}"))

    # Check resources
    if check_resources:
        click.echo("Checking system resources...")
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()

        if cpu_percent > 80:
            issues.append(("resource", f"High CPU usage: {cpu_percent}%"))
        if memory.percent > 80:
            issues.append(("resource", f"High memory usage: {memory.percent}%"))

    # Report results
    if not issues:
        click.echo("\n✓ All validation checks passed")
    else:
        click.echo(f"\nFound {len(issues)} issues:")
        for category, issue in issues:
            click.echo(f"  [{category}] {issue}")

        if fix:
            click.echo("\nAttempted fixes where possible")
        else:
            click.echo("\nRun with --fix to attempt automatic fixes")

        sys.exit(1)


@cli.command()
@click.option("--days", "-d", type=int, help="Remove results older than N days")
@click.option("--keep-last", "-k", type=int, help="Keep last N results per tier")
@click.option("--pattern", help="File pattern to match")
@click.option("--dry-run", is_flag=True, help="Show what would be deleted")
@click.option("--force", "-f", is_flag=True, help="Don't prompt for confirmation")
@click.pass_context
def cleanup(ctx, days, keep_last, pattern, dry_run, force):
    """Clean up old benchmark results"""

    if not days and not keep_last:
        click.echo("Error: Specify --days or --keep-last", err=True)
        sys.exit(1)

    files_to_delete = []
    now = datetime.now()

    # Find files to delete
    for file in ctx.obj.results_dir.glob(pattern or "*.json"):
        if file.stem == "metadata":
            continue

        # Check age
        if days:
            file_age = now - datetime.fromtimestamp(file.stat().st_mtime)
            if file_age.days > days:
                files_to_delete.append(file)

    # Handle keep-last
    if keep_last and len(files_to_delete) > keep_last:
        files_to_delete.sort(key=lambda f: f.stat().st_mtime)
        files_to_delete = files_to_delete[:-keep_last]

    if not files_to_delete:
        click.echo("No files to clean up")
        return

    # Display files
    click.echo(f"Found {len(files_to_delete)} files to delete:")
    for file in files_to_delete[:10]:
        click.echo(f"  - {file.name}")
    if len(files_to_delete) > 10:
        click.echo(f"  ... and {len(files_to_delete) - 10} more")

    if dry_run:
        click.echo("\nDRY RUN - No files deleted")
        return

    # Confirm
    if not force:
        if not click.confirm(f"\nDelete {len(files_to_delete)} files?"):
            return

    # Delete
    deleted = 0
    for file in files_to_delete:
        try:
            file.unlink()
            deleted += 1
        except Exception as e:
            logger.error(f"Failed to delete {file}: {e}")

    click.echo(f"Deleted {deleted} files")


# Tier management group
@cli.group()
def tier():
    """Manage storage tier configurations"""


@tier.command("list")
@click.pass_context
def tier_list(ctx):
    """List configured storage tiers"""

    suite = ctx.obj.suite
    tiers = suite.tier_manager.list_tiers()

    if not tiers:
        click.echo("No storage tiers configured")
        return

    click.echo("Configured storage tiers:")
    for tier in tiers:
        # Get tier info including validation status
        info = suite.tier_manager.get_tier_info(tier["name"])
        status = "✓" if info["valid"] else "✗"
        click.echo(f"  {status} {tier['name']}: {tier['path']} ({tier['type']})")
        if info.get("expected_performance"):
            click.echo(
                f"      Expected IOPS: R={info['expected_performance'].get('iops_read', 'N/A')} "
                f"W={info['expected_performance'].get('iops_write', 'N/A')}"
            )


@tier.command("add")
@click.option("--name", "-n", required=True, help="Tier name")
@click.option("--path", "-p", required=True, help="Storage path")
@click.option(
    "--type",
    "-t",
    required=True,
    type=click.Choice(["nvme", "ssd", "hdd", "distributed"]),
    help="Storage type",
)
@click.option("--filesystem", help="Filesystem type (for distributed)")
@click.option("--expected-iops-read", type=int, help="Expected read IOPS")
@click.option("--expected-iops-write", type=int, help="Expected write IOPS")
@click.option("--expected-bw-read", type=int, help="Expected read bandwidth (MB/s)")
@click.option("--expected-bw-write", type=int, help="Expected write bandwidth (MB/s)")
@click.pass_context
def tier_add(
    ctx,
    name,
    path,
    type,
    filesystem,
    expected_iops_read,
    expected_iops_write,
    expected_bw_read,
    expected_bw_write,
):
    """Add a new storage tier"""

    suite = ctx.obj.suite

    # Build properties
    properties = {}
    if filesystem:
        properties["filesystem"] = filesystem

    if any([expected_iops_read, expected_iops_write, expected_bw_read, expected_bw_write]):
        properties["expected_performance"] = {}
        if expected_iops_read:
            properties["expected_performance"]["iops_read"] = expected_iops_read
        if expected_iops_write:
            properties["expected_performance"]["iops_write"] = expected_iops_write
        if expected_bw_read:
            properties["expected_performance"]["bandwidth_read_mb"] = expected_bw_read
        if expected_bw_write:
            properties["expected_performance"]["bandwidth_write_mb"] = expected_bw_write

    try:
        suite.tier_manager.add_tier(name, path, type, **properties)
        click.echo(f"Added storage tier: {name}")
    except Exception as e:
        click.echo(f"Failed to add tier: {str(e)}", err=True)
        sys.exit(1)


@tier.command("test")
@click.argument("tier_name")
@click.option("--quick", is_flag=True, help="Run quick connectivity test only")
@click.pass_context
def tier_test(ctx, tier_name, quick):
    """Test storage tier connectivity and performance"""

    suite = ctx.obj.suite
    profile_manager = ProfileManager()

    # Validate tier exists
    tier_info = suite.tier_manager.get_tier_info(tier_name)
    if not tier_info:
        click.echo(f"Error: Unknown tier: {tier_name}", err=True)
        sys.exit(1)

    click.echo(f"Testing storage tier: {tier_name}")

    # Run validation
    if not suite.tier_manager.validate_tier(tier_info["path"]):
        click.echo("✗ Connectivity test failed", err=True)
        sys.exit(1)

    click.echo("✓ Connectivity test passed")

    if not quick:
        # Run performance probe using BenchmarkSuite
        click.echo("Running performance probe...")

        # Use quick_scan profile for probe
        probe_results = suite.run_benchmark(
            tiers=[tier_info["path"]],
            profile=profile_manager.get_profile("quick_scan"),
            tag="_probe",
            quiet=True,
        )

        # Extract and display key metrics
        aggregator = DataAggregator()
        summary = aggregator.aggregate_metrics(probe_results)

        click.echo("\nPerformance probe results:")
        click.echo(f"  Sequential read: {summary['seq_read_mb']:.1f} MB/s")
        click.echo(f"  Sequential write: {summary['seq_write_mb']:.1f} MB/s")
        click.echo(f"  Random read IOPS: {summary['rand_read_iops']:,.0f}")
        click.echo(f"  Random write IOPS: {summary['rand_write_iops']:,.0f}")


# Profile management group
@cli.group()
def profile():
    """Manage benchmark profiles"""


@profile.command("list")
@click.pass_context
def profile_list(ctx):
    """List available benchmark profiles"""

    ctx.obj.suite
    profile_manager = ProfileManager()

    profiles = profile_manager.list_profiles()

    click.echo("Available benchmark profiles:")
    for name, config in profiles.items():
        click.echo(f"\n  {name}:")
        click.echo(f"    {config['description']}")
        click.echo(f"    Duration: {config['duration']}s")
        click.echo(f"    Block sizes: {', '.join(config['block_sizes'])}")
        click.echo(f"    Patterns: {', '.join(config['patterns'])}")


@profile.command("show")
@click.argument("name")
@click.pass_context
def profile_show(ctx, name):
    """Show detailed profile configuration"""

    ctx.obj.suite
    profile_manager = ProfileManager()

    profile = profile_manager.get_profile(name)

    if not profile:
        click.echo(f"Error: Unknown profile: {name}", err=True)
        sys.exit(1)

    click.echo(json.dumps(profile, indent=2))


@profile.command("validate")
@click.argument("name")
@click.pass_context
def profile_validate(ctx, name):
    """Validate a profile configuration"""

    ctx.obj.suite
    profile_manager = ProfileManager()

    profile = profile_manager.get_profile(name)

    if not profile:
        click.echo(f"Error: Unknown profile: {name}", err=True)
        sys.exit(1)

    issues = profile_manager.validate_profile(profile)

    if not issues:
        click.echo(f"✓ Profile '{name}' is valid")
    else:
        click.echo(f"Profile '{name}' has issues:")
        for issue in issues:
            click.echo(f"  - {issue}")


# Environment checking
@cli.command("check-env")
@click.pass_context
def check_env(ctx):
    """Check environment and dependencies"""

    suite = ctx.obj.suite

    click.echo("Checking environment...")

    # Use BenchmarkSuite's built-in checks
    env_status = suite.check_environment()

    # Display overall status
    status_icon = "✓" if env_status["status"] == "ok" else "⚠" if env_status["status"] == "warning" else "✗"
    click.echo(f"Overall status: {status_icon} {env_status['status'].upper()}")

    # Display dependencies
    click.echo("\nDependencies:")
    for dep, status in env_status["dependencies"].items():
        if "available" in status or "writable" in status:
            icon = "✓"
            message = status
        else:
            icon = "✗"
            message = status
        click.echo(f"  {dep}: {icon} {message}")

    # Display warnings
    if env_status["warnings"]:
        click.echo("\nWarnings:")
        for warning in env_status["warnings"]:
            click.echo(f"  ⚠ {warning}")

    # Display errors
    if env_status["errors"]:
        click.echo("\nErrors:")
        for error in env_status["errors"]:
            click.echo(f"  ✗ {error}")

    # Check system resources using SafetyController
    try:
        safety_controller = SafetyController()
        resources = safety_controller.check_resources()

        click.echo("\nSystem resources:")
        click.echo(f"  CPU usage: {resources['cpu_percent']:.1f}%")
        click.echo(f"  Memory usage: {resources['memory_percent']:.1f}%")
        click.echo(f"  Memory available: {resources['memory_available_gb']:.1f} GB")
        click.echo(f"  Disk free: {resources['disk_free_gb']:.1f} GB")
    except Exception as e:
        click.echo(f"\nWarning: Could not check system resources: {e}")

    # Final status
    if env_status["status"] == "ok":
        click.echo("\n✓ Environment check passed")
    elif env_status["status"] == "warning":
        click.echo("\n⚠ Environment check passed with warnings")
    else:
        click.echo("\n✗ Environment check failed")
        sys.exit(1)


# Helper functions


class SafetyMonitor:
    """Monitor system resources during benchmarks"""

    def __init__(self, config):
        self.config = config
        self.running = False
        self.max_cpu = config.get("benchmark_suite.core.safety.max_cpu_percent", 90)
        self.max_memory = config.get("benchmark_suite.core.safety.max_memory_percent", 90)

    def start(self):
        self.running = True

    def stop(self):
        self.running = False

    def check(self):
        if not self.running:
            return True

        cpu = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory().percent

        if cpu > self.max_cpu:
            logger.warning(f"CPU usage {cpu}% exceeds limit {self.max_cpu}%")
            return False
        if memory > self.max_memory:
            logger.warning(f"Memory usage {memory}% exceeds limit {self.max_memory}%")
            return False

        return True


class TimeSeriesCollector:
    """Collect time series data during benchmarks"""

    def __init__(self, interval=1.0):
        self.interval = interval
        self.data = []
        self.running = False
        self.name = "time_series"

    def start(self):
        self.running = True
        self.data = []

    def stop(self):
        self.running = False

    def get_data(self):
        return self.data


class SystemMetricsCollector:
    """Collect system metrics during benchmarks"""

    def __init__(self, interval=5.0, network_enabled=False, interfaces=None):
        self.interval = interval
        self.network_enabled = network_enabled
        self.interfaces = interfaces
        self.data = []
        self.running = False
        self.name = "system_metrics"

    def start(self):
        self.running = True
        self.data = []

    def stop(self):
        self.running = False

    def get_data(self):
        return self.data


def run_benchmarks(
    tier_list,
    duration,
    block_sizes,
    patterns,
    io_depth,
    num_jobs,
    rate_limit,
    direct,
    config,
    safety_monitor,
    quiet,
):
    """Execute FIO benchmarks"""

    results = {"timestamp": datetime.now().isoformat(), "duration": duration, "results": {}}

    fio_path = config.get("benchmark_suite.engines.fio.path", "fio")

    for tier in tier_list:
        if not quiet:
            click.echo(f"\nBenchmarking: {tier}")

        tier_results = {}

        for pattern in patterns:
            for block_size in block_sizes:
                if safety_monitor and not safety_monitor.check():
                    raise Exception("Safety limits exceeded")

                # Build FIO command
                cmd = [
                    fio_path,
                    "--name=benchmark",
                    f"--directory={tier}",
                    f"--rw={pattern}",
                    f"--bs={block_size}",
                    f"--iodepth={io_depth}",
                    f"--runtime={duration}",
                    "--time_based",
                    "--group_reporting",
                    "--output-format=json",
                ]

                if direct:
                    cmd.append("--direct=1")
                if num_jobs:
                    cmd.append(f"--numjobs={num_jobs}")
                if rate_limit:
                    cmd.append(f"--rate={rate_limit}")

                # Run FIO
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                    fio_output = json.loads(result.stdout)

                    # Extract metrics
                    job = fio_output["jobs"][0]
                    metrics = {
                        "iops_read": job["read"]["iops"],
                        "iops_write": job["write"]["iops"],
                        "bw_read_kb": job["read"]["bw"],
                        "bw_write_kb": job["write"]["bw"],
                        "lat_read_ns": job["read"]["lat_ns"]["mean"],
                        "lat_write_ns": job["write"]["lat_ns"]["mean"],
                    }

                    tier_results[f"{pattern}_{block_size}"] = metrics

                except subprocess.CalledProcessError as e:
                    logger.error(f"FIO failed: {e.stderr}")
                    raise

        results["results"][tier] = tier_results

    return results


def run_analysis(results, analysis_types, config, analysis_config=None):
    """Run analysis on benchmark results"""

    analysis_config = analysis_config or {}
    analysis_results = {}

    for analysis_type in analysis_types:
        if analysis_type == "statistical":
            analysis_results["statistical"] = {
                "summary": calculate_statistics(results),
                "confidence_level": analysis_config.get("confidence_level", 95.0),
            }

        elif analysis_type == "anomaly":
            analysis_results["anomaly"] = {
                "anomalies": detect_anomalies(
                    results, threshold=analysis_config.get("anomaly_threshold", 3.0)
                )
            }

        elif analysis_type == "time_series":
            ts_results = {}
            if analysis_config.get("time_series_decomp"):
                ts_results["decomposition"] = "Time series decomposition results"
            if analysis_config.get("trend_detection"):
                ts_results["trends"] = "Trend detection results"
            if analysis_config.get("seasonality_detection"):
                ts_results["seasonality"] = "Seasonality detection results"
            if analysis_config.get("correlation_analysis"):
                ts_results["correlations"] = "Correlation analysis results"
            if analysis_config.get("forecast"):
                ts_results["forecast"] = "Forecast results"
            analysis_results["time_series"] = ts_results

        elif analysis_type == "network":
            analysis_results["network"] = {"impact": "Network impact analysis results"}

    return analysis_results


def save_results(results_dir, benchmark_id, results):
    """Save benchmark results to file"""

    results_path = results_dir / f"{benchmark_id}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    return results_path


def generate_reports(output_dir, benchmark_id, results, formats, config, report_config=None):
    """Generate reports in specified formats using visualization modules"""

    report_config = report_config or {}
    output_dir = Path(output_dir)
    
    # Check if user explicitly requested fallback methods
    if report_config.get('use_fallback', False):
        click.echo("🔄 Using fallback report generation methods as requested")
        return _generate_reports_fallback(output_dir, benchmark_id, results, formats, config, report_config)
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Create benchmark config for visualization modules
        # Use a minimal configuration that will pass validation for report generation
        minimal_config = {
            'benchmark_suite': {
                'core': {
                    'safety': {
                        'enabled': True,
                        'max_cpu_percent': 90,
                        'max_memory_percent': 90
                    },
                    'output': {
                        'base_directory': str(output_dir)
                    },
                    'logging': {
                        'level': 'INFO'
                    }
                },
                'tiers': {
                    'tier_definitions': []
                },
                'benchmark_profiles': {
                    'default': {
                        'description': 'Default profile for report generation',
                        'duration_seconds': 60,
                        'block_sizes': ['4k'],
                        'patterns': ['read']
                    }
                },
                'execution': {
                    'engine': 'fio'
                },
                'visualization': {
                    'reports': {
                        'enabled': True,
                        'formats': ['html', 'json'],
                        'output_dir': str(output_dir)
                    },
                    'charts': {
                        'enabled': True,
                        'output_dir': str(output_dir / 'charts')
                    }
                }
            }
        }
        
        # Merge with provided config, giving precedence to provided values
        if config:
            merged_config = {**minimal_config}
            # Simple merge - could be made more sophisticated if needed
            for key, value in config.items():
                if isinstance(value, dict) and key in merged_config and isinstance(merged_config[key], dict):
                    merged_config[key].update(value)
                else:
                    merged_config[key] = value
        else:
            merged_config = minimal_config
            
        benchmark_config = BenchmarkConfig(merged_config)
        
        # Initialize report generator from visualization module
        report_generator = ReportGenerator(benchmark_config)
        
        # Convert results to BenchmarkResult format expected by visualization modules
        from tdiobench.core.benchmark_data import BenchmarkResult
        
        # Extract metrics data and populate tier information
        metrics_data = results.get('metrics', {})
        
        # Determine tiers from the metrics structure
        tiers = []
        tier_results = {}
        
        if metrics_data:
            # Get all unique tier names from metrics
            for tier_path, tier_data in metrics_data.items():
                if isinstance(tier_data, dict):
                    tier_name = tier_path.split('/')[-1] if '/' in tier_path else tier_path
                    if tier_name not in tiers:
                        tiers.append(tier_name)
                    
                    # Create tier summary
                    tier_summary = {}
                    all_throughput = []
                    all_iops = []
                    all_latency = []
                    
                    for pattern, pattern_data in tier_data.items():
                        if isinstance(pattern_data, dict):
                            if 'throughput_MBps' in pattern_data:
                                all_throughput.append(pattern_data['throughput_MBps'])
                            if 'iops' in pattern_data:
                                all_iops.append(pattern_data['iops'])
                            if 'latency_ms' in pattern_data:
                                all_latency.append(pattern_data['latency_ms'])
                    
                    # Calculate averages
                    tier_summary = {
                        'avg_throughput_MBps': sum(all_throughput) / len(all_throughput) if all_throughput else 0,
                        'avg_iops': sum(all_iops) / len(all_iops) if all_iops else 0,
                        'avg_latency_ms': sum(all_latency) / len(all_latency) if all_latency else 0,
                        'max_throughput_MBps': max(all_throughput) if all_throughput else 0,
                        'max_iops': max(all_iops) if all_iops else 0,
                        'min_latency_ms': min(all_latency) if all_latency else 0,
                        'patterns': list(tier_data.keys())
                    }
                    
                    tier_results[tier_name] = {
                        'summary': tier_summary,
                        'raw_data': tier_data
                    }
        
        # Create BenchmarkResult object from raw results data
        benchmark_result = BenchmarkResult(
            run_id=benchmark_id,
            tier_name=results.get('tier_name', tiers[0] if tiers else 'default'),
            profile_name=results.get('profile_name', 'default'),
            start_time=results.get('start_time', ''),
            end_time=results.get('end_time', ''),
            metrics=metrics_data,
            parameters=results.get('parameters', {}),
            raw_data=results
        )
        
        # Populate additional attributes for better report generation
        benchmark_result.tiers = tiers
        benchmark_result.tier_results = tier_results
        benchmark_result.benchmark_id = benchmark_id
        
        # Generate enhanced charts if explicitly requested and available
        enhanced_charts = []
        if report_config.get('include_charts', False) and ENHANCED_CHARTS_AVAILABLE:
            try:
                click.echo("🎨 Generating enhanced performance charts...")
                
                # Create charts directory
                charts_dir = output_dir / 'charts'
                charts_dir.mkdir(parents=True, exist_ok=True)
                
                # Get report title for charts
                chart_title = report_config.get('title', f'Benchmark Report - {benchmark_id}')
                
                # Chart generation using visualization module ChartGenerator
                if chart_generator:
                    # Generate individual charts
                    metrics_to_plot = ['throughput_MBps', 'iops', 'latency_ms']
                    
                    # Try to generate a time series chart if we have BenchmarkData
                    try:
                        fig = chart_generator.generate_time_series_chart(
                            benchmark_data,
                            metrics_to_plot,
                            title=f"{chart_title} - Time Series",
                            output_path=str(charts_dir / 'time_series.html')
                        )
                        if fig:
                            enhanced_charts.append('time_series.html')
                    except Exception as e:
                        logger.warning(f"Could not generate time series chart: {e}")
                    
                    # Try to generate a comparison chart
                    try:
                        fig = chart_generator.generate_comparison_chart(
                            benchmark_data,
                            metrics_to_plot,
                            title=f"{chart_title} - Comparison",
                            output_path=str(charts_dir / 'comparison.html')
                        )
                        if fig:
                            enhanced_charts.append('comparison.html')
                    except Exception as e:
                        logger.warning(f"Could not generate comparison chart: {e}")
                    
                    # Try to generate a bar chart
                    try:
                        fig = chart_generator.generate_bar_chart(
                            benchmark_data,
                            metric='throughput_MBps',
                            title=f"{chart_title} - Throughput",
                            output_path=str(charts_dir / 'throughput_bar.html')
                        )
                        if fig:
                            enhanced_charts.append('throughput_bar.html')
                    except Exception as e:
                        logger.warning(f"Could not generate bar chart: {e}")
                    
                if enhanced_charts:
                    click.echo(f"✅ Generated {len(enhanced_charts)} enhanced charts")
                else:
                    click.echo("⚠️  No charts generated - insufficient data")
                    
            except Exception as e:
                click.echo(f"⚠️  Enhanced chart generation failed: {str(e)}")
                logger.error(f"Chart generation error: {str(e)}")

        # Generate reports using visualization module
        report_title = report_config.get('title', f'Benchmark Report - {benchmark_id}')
        
        # Generate standard reports using the visualization module
        generated_files = {}
        
        for fmt in formats:
            try:
                if fmt == "html":
                    report_path = output_dir / f"{benchmark_id}_report.html"
                    output_path = report_generator.generate_html_report(
                        benchmark_result, str(report_path), report_title, **report_config
                    )
                    generated_files['html'] = output_path
                    click.echo(f"HTML report: {report_path}")

                elif fmt == "json":
                    report_path = output_dir / f"{benchmark_id}_report.json"
                    output_path = report_generator.generate_json_report(
                        benchmark_result, str(report_path), report_title, **report_config
                    )
                    generated_files['json'] = output_path
                    click.echo(f"JSON report: {report_path}")

                elif fmt == "markdown":
                    report_path = output_dir / f"{benchmark_id}_report.md"
                    output_path = report_generator.generate_markdown_report(
                        benchmark_result, str(report_path), report_title, **report_config
                    )
                    generated_files['markdown'] = output_path
                    click.echo(f"Markdown report: {report_path}")
                    
                elif fmt == "csv":
                    # CSV generates multiple files
                    csv_paths = report_generator.generate_csv_report(
                        benchmark_result, str(output_dir), report_title, **report_config
                    )
                    generated_files['csv'] = csv_paths
                    click.echo(f"CSV reports: {csv_paths}")
                    
            except Exception as e:
                click.echo(f"⚠️  Error generating {fmt} report: {str(e)}")
                logger.error(f"Report generation error for {fmt}: {str(e)}")
                if not report_config.get('use_fallback', False):
                    click.echo(f"💡 Tip: Use --use-fallback flag to try legacy {fmt} report generation")
                # Continue to next format instead of stopping
        
        # Summary message
        if enhanced_charts:
            click.echo(f"\n📈 Enhanced charts available in: {output_dir / 'charts'}")
            for chart in enhanced_charts:
                click.echo(f"   • {chart}")
        
        return enhanced_charts
        
    except Exception as e:
        click.echo(f"⚠️  Report generation failed: {str(e)}")
        logger.error(f"Report generation error: {str(e)}")
        
        # Only fallback if explicitly requested
        if report_config.get('use_fallback', False):
            click.echo("🔄 Falling back to legacy report generation methods...")
            return _generate_reports_fallback(output_dir, benchmark_id, results, formats, config, report_config)
        else:
            click.echo("💡 Tip: Use --use-fallback flag to try legacy report generation methods")
            raise


def _generate_html_with_visualization_module(benchmark_result, report_path, report_title, report_config):
    """Generate HTML using visualization module with special handling for executive summary and recommendations"""
    
    # Check for special report types
    if report_config.get('executive_summary_only', False):
        # Use the original executive summary generation for now
        return generate_executive_summary_html(benchmark_result.__dict__, report_config)
    
    # For standard HTML reports, try to use visualization module
    try:
        from tdiobench.core.benchmark_config import BenchmarkConfig
        benchmark_config = BenchmarkConfig({})
        report_generator = ReportGenerator(benchmark_config)
        
        # Generate using visualization module
        output_path = report_generator.generate_html_report(
            benchmark_result, str(report_path), report_title
        )
        return None  # File already written by the module
        
    except Exception as e:
        # Fallback to original HTML generation
        logger.warning(f"Visualization module HTML generation failed, using fallback: {str(e)}")
        return generate_html_report(benchmark_result.__dict__, report_config)


def _generate_reports_fallback(output_dir, benchmark_id, results, formats, config, report_config):
    """Fallback to original report generation methods"""
    
    click.echo("🔄 Using legacy report generation methods")
    
    report_config = report_config or {}
    output_dir = Path(output_dir)
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for fmt in formats:
        if fmt == "html":
            report_path = output_dir / f"{benchmark_id}_report.html"
            html_content = generate_html_report(results, report_config)
            with open(report_path, "w") as f:
                f.write(html_content)
            click.echo(f"HTML report: {report_path}")

        elif fmt == "json":
            report_path = output_dir / f"{benchmark_id}_report.json"
            with open(report_path, "w") as f:
                json.dump(results, f, indent=2, default=str)
            click.echo(f"JSON report: {report_path}")

        elif fmt == "markdown":
            report_path = output_dir / f"{benchmark_id}_report.md"
            md_content = generate_markdown_report(results, report_config)
            with open(report_path, "w") as f:
                f.write(md_content)
            click.echo(f"Markdown report: {report_path}")
    
    return []


def display_summary(results):
    """Display benchmark summary"""

    click.echo("\n" + "=" * 60)
    click.echo("BENCHMARK RESULTS SUMMARY")
    click.echo("=" * 60)

    for tier, tier_results in results.get("results", {}).items():
        click.echo(f"\n{tier}:")

        # Calculate aggregates
        total_iops_read = 0
        total_iops_write = 0
        total_bw_read = 0
        total_bw_write = 0
        count = 0

        for test, metrics in tier_results.items():
            total_iops_read += metrics.get("iops_read", 0)
            total_iops_write += metrics.get("iops_write", 0)
            total_bw_read += metrics.get("bw_read_kb", 0)
            total_bw_write += metrics.get("bw_write_kb", 0)
            count += 1

        if count > 0:
            click.echo(
                f"  Average IOPS - Read: {total_iops_read/count:,.0f}, "
                f"Write: {total_iops_write/count:,.0f}"
            )
            click.echo(
                f"  Average BW - Read: {total_bw_read/count/1024:.1f} MB/s, "
                f"Write: {total_bw_write/count/1024:.1f} MB/s"
            )


def _get_profile_config(profile_name):
    """Get configuration for a benchmark profile"""

    profiles = {
        "quick_scan": {
            "duration": 30,
            "block_sizes": "4k,1m",
            "patterns": "read,write",
            "io_depth": 32,
            "num_jobs": 1,
        },
        "production_safe": {
            "duration": 60,
            "block_sizes": "4k,1m",
            "patterns": "read,randread",
            "io_depth": 8,
            "num_jobs": 1,
        },
        "comprehensive": {
            "duration": 300,
            "block_sizes": "4k,8k,16k,32k,64k,128k,256k,512k,1m",
            "patterns": "read,write,randread,randwrite,randrw",
            "io_depth": 64,
            "num_jobs": 4,
        },
        "latency_focused": {
            "duration": 120,
            "block_sizes": "512,4k,8k",
            "patterns": "randread,randwrite",
            "io_depth": 1,
            "num_jobs": 1,
        },
    }

    return profiles.get(profile_name)


def check_tool_exists(tool):
    """Check if a tool exists in PATH"""
    try:
        subprocess.run([tool, "--version"], capture_output=True, check=False)
        return True
    except FileNotFoundError:
        return False


def check_fio_version():
    """Get FIO version"""
    try:
        result = subprocess.run(["fio", "--version"], capture_output=True, text=True)
        return result.stdout.strip()
    except BaseException:
        return None


def run_performance_probe(path):
    """Run quick performance probe on a path"""

    # Simple DD-based probe for demonstration
    results = {
        "seq_read": 100.0,  # MB/s
        "seq_write": 90.0,  # MB/s
        "rand_read_iops": 5000,
        "rand_write_iops": 4500,
    }

    return results


def calculate_statistics(results):
    """Calculate basic statistics from results"""

    stats = {}
    for tier, tier_results in results.get("results", {}).items():
        tier_stats = {
            "iops_read": {"min": float("inf"), "max": 0, "avg": 0},
            "iops_write": {"min": float("inf"), "max": 0, "avg": 0},
        }

        count = 0
        for test, metrics in tier_results.items():
            iops_read = metrics.get("iops_read", 0)
            iops_write = metrics.get("iops_write", 0)

            tier_stats["iops_read"]["min"] = min(tier_stats["iops_read"]["min"], iops_read)
            tier_stats["iops_read"]["max"] = max(tier_stats["iops_read"]["max"], iops_read)
            tier_stats["iops_read"]["avg"] += iops_read

            tier_stats["iops_write"]["min"] = min(tier_stats["iops_write"]["min"], iops_write)
            tier_stats["iops_write"]["max"] = max(tier_stats["iops_write"]["max"], iops_write)
            tier_stats["iops_write"]["avg"] += iops_write

            count += 1

        if count > 0:
            tier_stats["iops_read"]["avg"] /= count
            tier_stats["iops_write"]["avg"] /= count

        stats[tier] = tier_stats

    return stats


def detect_anomalies(results, threshold=3.0):
    """Detect anomalies in results using z-score"""

    anomalies = []

    # Simple anomaly detection for demonstration
    for tier, tier_results in results.get("results", {}).items():
        values = []
        for test, metrics in tier_results.items():
            values.append(metrics.get("iops_read", 0))

        if len(values) > 3:
            import statistics

            mean = statistics.mean(values)
            stdev = statistics.stdev(values)

            for i, value in enumerate(values):
                z_score = abs((value - mean) / stdev) if stdev > 0 else 0
                if z_score > threshold:
                    anomalies.append(
                        {
                            "tier": tier,
                            "test": list(tier_results.keys())[i],
                            "metric": "iops_read",
                            "value": value,
                            "z_score": z_score,
                        }
                    )

    return anomalies


def compare_benchmarks(all_results, tiers=None, baseline_tier=None, baseline_id=None, metrics=None):
    """Compare multiple benchmark results"""

    comparison = {
        "benchmarks": list(all_results.keys()),
        "metrics": metrics or ["iops", "bandwidth", "latency"],
    }

    # Extract and compare metrics
    for metric in comparison["metrics"]:
        comparison[metric] = {}

        for bid, results in all_results.items():
            for tier, tier_results in results.get("results", {}).items():
                if tiers and tier not in tiers:
                    continue

                if tier not in comparison[metric]:
                    comparison[metric][tier] = {}

                # Average metrics across all tests
                values = []
                for test, test_metrics in tier_results.items():
                    if metric == "iops":
                        values.append(test_metrics.get("iops_read", 0))
                        values.append(test_metrics.get("iops_write", 0))
                    elif metric == "bandwidth":
                        values.append(test_metrics.get("bw_read_kb", 0) / 1024)
                        values.append(test_metrics.get("bw_write_kb", 0) / 1024)
                    elif metric == "latency":
                        values.append(test_metrics.get("lat_read_ns", 0) / 1000000)
                        values.append(test_metrics.get("lat_write_ns", 0) / 1000000)

                if values:
                    comparison[metric][tier][bid] = sum(values) / len(values)

    return comparison


def get_last_benchmark_id(results_dir):
    """Get the ID of the most recent benchmark"""

    # First try the flat structure (*.json files), excluding report files
    # Exclude files ending with _report and generic report files
    files = [f for f in results_dir.glob("*.json") 
             if not f.stem.endswith("_report") and f.stem not in ["report"]]
    if files:
        files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        return files[0].stem
    
    # Then try the subdirectory structure (*/result.json)
    result_files = list(results_dir.glob("*/result.json"))
    if result_files:
        result_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        return result_files[0].parent.name
    
    return None


def eval_filter(benchmark, filter_expr):
    """Evaluate filter expression on benchmark metadata"""

    # Simple filter evaluation for demonstration
    try:
        # Create safe evaluation context
        safe_dict = {
            "duration": benchmark.get("duration", 0),
            "timestamp": benchmark.get("timestamp", ""),
            "tag": benchmark.get("tag", ""),
            "profile": benchmark.get("profile", ""),
        }

        return eval(filter_expr, {"__builtins__": {}}, safe_dict)
    except BaseException:
        return False


def generate_html_report(results, config):
    """Generate HTML report"""
    
    # Check if this should be an executive summary
    if config.get('executive_summary_only', False):
        return generate_executive_summary_html(results, config)

    html = f"""
    <html>
    <head>
        <title>{config.get('title', 'Benchmark Report')}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <h1>{config.get('title', 'Benchmark Report')}</h1>
        <h2>Benchmark ID: {results.get('run_id', 'Unknown')}</h2>
        <p>Timestamp: {results.get('start_time', 'Unknown')}</p>
        <p>Duration: {results.get('duration_seconds', 0):.1f} seconds</p>

        <h2>Results Summary</h2>
        <table>
            <tr>
                <th>Pattern</th>
                <th>Block Size</th>
                <th>Read IOPS</th>
                <th>Write IOPS</th>
                <th>Throughput (MB/s)</th>
                <th>Latency (ms)</th>
            </tr>
    """

    # Handle the actual data structure
    metrics = results.get("metrics", {})
    for tier_path, tier_data in metrics.items():
        if isinstance(tier_data, dict):
            # Handle direct pattern results under tier (actual format)
            for pattern_key, pattern_data in tier_data.items():
                if isinstance(pattern_data, dict) and 'iops' in pattern_data:
                    html += f"""
                <tr>
                    <td>{pattern_key}</td>
                    <td>4k</td>
                    <td>{pattern_data.get('iops', 0):,.0f}</td>
                    <td>{pattern_data.get('iops', 0):,.0f}</td>
                    <td>{pattern_data.get('throughput_MBps', 0):.1f}</td>
                    <td>{pattern_data.get('latency_ms', 0):.2f}</td>
                </tr>
                """

    html += """
        </table>
    """
    
    # Add raw data if requested
    if config.get('include_raw', False):
        html += """
        <h2>Raw Benchmark Data</h2>
        <pre style="background-color: #f5f5f5; padding: 10px; border-radius: 5px; overflow-x: auto;">
        """
        html += json.dumps(results, indent=2, default=str)
        html += """
        </pre>
        """
    
    # Add recommendations if requested
    if config.get('recommendations', False):
        html += generate_recommendations_html(results)

    html += """
    </body>
    </html>
    """

    return html


def generate_recommendations_html(results):
    """Generate performance recommendations based on benchmark results"""
    
    recommendations = []
    
    # Analyze results to generate recommendations
    all_iops = []
    all_throughput = []
    all_latency = []
    
    # Extract metrics from actual data structure
    metrics = results.get("metrics", {})
    for tier_path, tier_data in metrics.items():
        if isinstance(tier_data, dict):
            # Handle direct pattern results under tier (actual format)
            for pattern_key, pattern_data in tier_data.items():
                if isinstance(pattern_data, dict) and 'iops' in pattern_data:
                    all_iops.append(pattern_data.get('iops', 0))
                    all_throughput.append(pattern_data.get('throughput_MBps', 0))
                    latency = pattern_data.get('latency_ms', 0)
                    if latency > 0:
                        all_latency.append(latency)
    
    # Generate recommendations based on performance patterns
    if all_iops and max(all_iops) > 50000:
        recommendations.append("✅ High IOPS detected - suitable for transactional workloads")
    elif all_iops and max(all_iops) < 1000:
        recommendations.append("⚠️ Low IOPS detected - consider SSD upgrade for better performance")
    
    if all_throughput and max(all_throughput) > 1000:
        recommendations.append("✅ High throughput detected - excellent for large file operations")
    elif all_throughput and max(all_throughput) < 100:
        recommendations.append("⚠️ Low throughput - check storage configuration and network")
    
    if all_latency:
        avg_latency = sum(all_latency) / len(all_latency)
        if avg_latency < 1.0:
            recommendations.append("✅ Low latency detected - excellent for real-time applications")
        elif avg_latency > 10.0:
            recommendations.append("⚠️ High latency detected - consider faster storage or optimization")
    
    # Add general recommendations
    recommendations.append("💡 Consider running tests with different block sizes for optimization")
    recommendations.append("💡 Monitor latency patterns for consistent performance")
    
    if not recommendations:
        recommendations.append("ℹ️ No specific recommendations - performance appears normal")
    
    html = """
    <h2>Performance Recommendations</h2>
    <div style="background-color: #f0f8ff; padding: 15px; border-radius: 5px; border-left: 4px solid #007acc;">
        <ul>
    """
    
    for rec in recommendations:
        html += f"<li>{rec}</li>"
    
    html += """
        </ul>
    </div>
    """
    
    return html


def generate_executive_summary_html(results, config):
    """Generate executive summary HTML report"""
    
    # Calculate key metrics from actual data structure
    all_iops = []
    all_throughput = []
    all_latency = []
    
    metrics = results.get("metrics", {})
    for tier_path, tier_data in metrics.items():
        if isinstance(tier_data, dict):
            # Handle direct pattern results under tier (actual format)
            for pattern_key, pattern_data in tier_data.items():
                if isinstance(pattern_data, dict) and 'iops' in pattern_data:
                    all_iops.append(pattern_data.get('iops', 0))
                    all_throughput.append(pattern_data.get('throughput_MBps', 0))
                    latency = pattern_data.get('latency_ms', 0)
                    if latency > 0:
                        all_latency.append(latency)
    
    # Calculate summary stats
    max_iops = max(all_iops) if all_iops else 0
    max_throughput = max(all_throughput) if all_throughput else 0
    min_latency = min(all_latency) if all_latency else 0
    avg_latency = sum(all_latency) / len(all_latency) if all_latency else 0
    
    avg_iops = sum(all_iops) / len(all_iops) if all_iops else 0
    avg_throughput = sum(all_throughput) / len(all_throughput) if all_throughput else 0

    html = f"""
    <html>
    <head>
        <title>Executive Summary - {config.get('title', 'Benchmark Report')}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f8f9fa; }}
            .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            h1 {{ color: #2c3e50; text-align: center; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
            h2 {{ color: #34495e; }}
            .metric-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0; }}
            .metric-card {{ background: #ecf0f1; padding: 15px; border-radius: 8px; text-align: center; }}
            .metric-value {{ font-size: 24px; font-weight: bold; color: #2980b9; }}
            .metric-label {{ font-size: 14px; color: #7f8c8d; }}
            .summary-box {{ background: #e8f5e8; padding: 15px; border-left: 4px solid #27ae60; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Executive Summary</h1>
            <p><strong>Benchmark ID:</strong> {results.get('run_id', 'Unknown')}</p>
            <p><strong>Start Time:</strong> {results.get('start_time', 'Unknown')}</p>
            <p><strong>Duration:</strong> {results.get('duration_seconds', 0):.1f} seconds</p>
            
            <h2>Performance Highlights</h2>
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-value">{max_iops:,.0f}</div>
                    <div class="metric-label">Peak IOPS</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{max_throughput:.1f}</div>
                    <div class="metric-label">Peak Throughput (MB/s)</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{min_latency:.2f}</div>
                    <div class="metric-label">Min Latency (ms)</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{avg_latency:.2f}</div>
                    <div class="metric-label">Avg Latency (ms)</div>
                </div>
            </div>
            
            <h2>Average Performance</h2>
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-value">{avg_iops:,.0f}</div>
                    <div class="metric-label">Avg IOPS</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{avg_throughput:.1f}</div>
                    <div class="metric-label">Avg Throughput (MB/s)</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{len(all_iops)}</div>
                    <div class="metric-label">Test Patterns</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{len(metrics)}</div>
                    <div class="metric-label">Storage Tiers</div>
                </div>
            </div>
            
            <div class="summary-box">
                <h3>Key Findings</h3>
                <ul>
                    <li>Tested {len(metrics)} storage tier(s) with {len(all_iops)} patterns</li>
                    <li>Peak performance: {max_iops:,.0f} IOPS at {max_throughput:.1f} MB/s</li>
                    <li>Latency range: {min_latency:.2f} - {max(all_latency) if all_latency else 0:.2f} ms</li>
                </ul>
            </div>
        </div>
    </body>
    </html>
    """
    
    return html


def generate_markdown_report(results, config):
    """Generate Markdown report"""

    md = f"""# {config.get('title', 'Benchmark Report')}

## Benchmark ID: {results.get('run_id', 'Unknown')}
**Start Time:** {results.get('start_time', 'Unknown')}
**Duration:** {results.get('duration_seconds', 0):.1f} seconds

## Results Summary

| Pattern | Block Size | IOPS | Throughput (MB/s) | Latency (ms) |
|---------|------------|------|------------------|--------------|
"""

    # Handle the actual data structure
    metrics = results.get("metrics", {})
    for tier_path, tier_data in metrics.items():
        if isinstance(tier_data, dict):
            # Handle direct pattern results under tier (actual format)
            for pattern_key, pattern_data in tier_data.items():
                if isinstance(pattern_data, dict) and 'iops' in pattern_data:
                    md += f"| {pattern_key} | "
                    md += f"4k | "
                    md += f"{pattern_data.get('iops', 0):,.0f} | "
                    md += f"{pattern_data.get('throughput_MBps', 0):.1f} | "
                    md += f"{pattern_data.get('latency_ms', 0):.2f} |\n"

    # Add raw data if requested
    if config.get('include_raw', False):
        md += "\n## Raw Benchmark Data\n\n```json\n"
        md += json.dumps(results, indent=2, default=str)
        md += "\n```\n"
    
    # Add recommendations if requested
    if config.get('recommendations', False):
        md += generate_recommendations_markdown(results)

    return md


def generate_recommendations_markdown(results):
    """Generate performance recommendations in markdown format"""
    
    recommendations = []
    
    # Analyze results using actual data structure
    all_iops = []
    all_throughput = []
    all_latency = []
    
    metrics = results.get("metrics", {})
    for tier_path, tier_data in metrics.items():
        if isinstance(tier_data, dict):
            # Handle direct pattern results under tier (actual format)
            for pattern_key, pattern_data in tier_data.items():
                if isinstance(pattern_data, dict) and 'iops' in pattern_data:
                    all_iops.append(pattern_data.get('iops', 0))
                    all_throughput.append(pattern_data.get('throughput_MBps', 0))
                    latency = pattern_data.get('latency_ms', 0)
                    if latency > 0:
                        all_latency.append(latency)
    
    # Generate recommendations
    if all_iops and max(all_iops) > 50000:
        recommendations.append("✅ High IOPS detected - suitable for transactional workloads")
    elif all_iops and max(all_iops) < 1000:
        recommendations.append("⚠️ Low IOPS detected - consider SSD upgrade for better performance")
    
    if all_throughput and max(all_throughput) > 1000:
        recommendations.append("✅ High throughput detected - excellent for large file operations")
    elif all_throughput and max(all_throughput) < 100:
        recommendations.append("⚠️ Low throughput - check storage configuration and network")
    
    if all_latency:
        avg_latency = sum(all_latency) / len(all_latency)
        if avg_latency < 1.0:
            recommendations.append("✅ Low latency detected - excellent for real-time applications")
        elif avg_latency > 10.0:
            recommendations.append("⚠️ High latency detected - consider faster storage or optimization")
    
    recommendations.append("💡 Consider running tests with different block sizes for optimization")
    recommendations.append("💡 Monitor latency patterns for consistent performance")
    
    if not recommendations:
        recommendations.append("ℹ️ No specific recommendations - performance appears normal")
    
    md = "\n## Performance Recommendations\n\n"
    for rec in recommendations:
        md += f"- {rec}\n"
    
    return md


def generate_comparison_report(comparison, output_path):
    """Generate comparison report"""

    html = """
    <html>
    <head>
        <title>Benchmark Comparison Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            table { border-collapse: collapse; width: 100%; margin: 20px 0; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
        </style>
    </head>
    <body>
        <h1>Benchmark Comparison Report</h1>
    """

    for metric in comparison.get("metrics", []):
        html += f"<h2>{metric.upper()} Comparison</h2><table>"
        html += "<tr><th>Tier</th>"

        for bid in comparison["benchmarks"]:
            html += f"<th>{bid}</th>"
        html += "</tr>"

        for tier, values in comparison.get(metric, {}).items():
            html += f"<tr><td>{tier}</td>"
            for bid in comparison["benchmarks"]:
                value = values.get(bid, "N/A")
                if isinstance(value, (int, float)):
                    html += f"<td>{value:,.1f}</td>"
                else:
                    html += f"<td>{value}</td>"
            html += "</tr>"

        html += "</table>"

    html += "</body></html>"

    with open(output_path, "w") as f:
        f.write(html)


def format_comparison_markdown(comparison):
    """Format comparison as markdown"""

    md = "# Benchmark Comparison\n\n"

    for metric in comparison.get("metrics", []):
        md += f"## {metric.upper()}\n\n"
        md += "| Tier |"

        for bid in comparison["benchmarks"]:
            md += f" {bid} |"
        md += "\n|------|"

        for _ in comparison["benchmarks"]:
            md += "------|"
        md += "\n"

        for tier, values in comparison.get(metric, {}).items():
            md += f"| {tier} |"
            for bid in comparison["benchmarks"]:
                value = values.get(bid, "N/A")
                if isinstance(value, (int, float)):
                    md += f" {value:,.1f} |"
                else:
                    md += f" {value} |"
            md += "\n"

        md += "\n"

    return md


def main():
    cli()


if __name__ == "__main__":
    main()
