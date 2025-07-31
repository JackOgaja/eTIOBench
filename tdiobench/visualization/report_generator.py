#!/usr/bin/env python3
"""
Report Generator (Tiered Storage I/O Benchmark)

This module provides functionality for generating benchmark reports in various formats,
including HTML, JSON, CSV, and Markdown.

Author: Jack Ogaja
Date: 2025-06-26
"""

import csv
import datetime
import json
import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np

from tdiobench.core.benchmark_config import BenchmarkConfig
from tdiobench.core.benchmark_data import BenchmarkResult
from tdiobench.core.benchmark_exceptions import BenchmarkReportError

logger = logging.getLogger("tdiobench.visualization.report")


class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types."""
    
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class ReportGenerator:
    """
    Generator for benchmark reports.

    Provides methods for generating reports in various formats, including
    HTML, JSON, CSV, and Markdown.
    """

    def __init__(self, config: BenchmarkConfig):
        """
        Initialize report generator.

        Args:
            config: Benchmark configuration
        """
        self.config = config
        self.template_dir = config.get("visualization.reports.template_dir", "./templates")
        self.enable_charts = config.get("visualization.reports.charts.enabled", True)
        self.chart_types = config.get("visualization.reports.charts.types", ["bar", "line"])

        # Initialize formatters
        self.formatters = {
            "html": self.generate_html_report,
            "json": self.generate_json_report,
            "csv": self.generate_csv_report,
            "markdown": self.generate_markdown_report,
            "email": self.generate_email_report,
        }

    def generate_reports(
        self,
        benchmark_result: BenchmarkResult,
        output_dir: str = "results",
        formats: Optional[List[str]] = None,
        report_title: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, str]:
        """
        Generate reports in specified formats.

        Args:
            benchmark_result: Benchmark result
            output_dir: Output directory
            formats: Output formats
            report_title: Report title
            **kwargs: Additional parameters

        Returns:
            Dictionary mapping format to output file paths

        Raises:
            BenchmarkReportError: If report generation fails
        """
        # Default formats
        if not formats:
            formats = ["html", "json"]

        # Default title
        if not report_title:
            report_title = "Storage Benchmark Report"

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Generate reports
        report_files = {}

        for format in formats:
            if format.lower() in self.formatters:
                try:
                    # Generate report
                    formatter = self.formatters[format.lower()]
                    output_path = os.path.join(output_dir, f"report.{format.lower()}")

                    if format.lower() == "csv":
                        # CSV has multiple files
                        output_paths = formatter(
                            benchmark_result, output_dir, report_title, **kwargs
                        )
                        report_files[format] = output_paths
                    else:
                        # Generate single file
                        output_path = formatter(
                            benchmark_result, output_path, report_title, **kwargs
                        )
                        report_files[format] = output_path

                    logger.info(f"Generated {format} report: {output_path}")

                except Exception as e:
                    logger.error(f"Error generating {format} report: {str(e)}")
                    raise BenchmarkReportError(f"Failed to generate {format} report: {str(e)}")
            else:
                logger.warning(f"Unsupported report format: {format}")

        return report_files

    def generate_html_report(
        self, benchmark_result: BenchmarkResult, output_path: str, report_title: str, **kwargs
    ) -> str:
        """
        Generate HTML report.

        Args:
            benchmark_result: Benchmark result
            output_path: Output file path
            report_title: Report title
            **kwargs: Additional parameters

        Returns:
            Path to generated report

        Raises:
            BenchmarkReportError: If report generation fails
        """
        try:
            # Try to use Jinja2 for templating
            try:
                import datetime

                from jinja2 import Environment, FileSystemLoader

                # Check if template directory exists
                template_path = os.path.join(self.template_dir, "html")
                if not os.path.exists(template_path):
                    # Use default template
                    html = self._generate_default_html_report(benchmark_result, report_title, **kwargs)
                else:
                    # Load template
                    env = Environment(loader=FileSystemLoader(template_path))
                    template = env.get_template("report.html")

                    # Generate charts if enabled
                    charts = {}

                    if self.enable_charts:
                        charts = self._generate_charts(benchmark_result)

                    # Render template
                    html = template.render(
                        title=report_title,
                        benchmark=benchmark_result.to_dict(),
                        charts=charts,
                        generation_time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        user=os.getenv('USER', os.getenv('USERNAME', 'Unknown')),
                    )
            except ImportError:
                # Jinja2 not available, use default template
                logger.warning("Jinja2 not available, using default HTML template")
                html = self._generate_default_html_report(benchmark_result, report_title, **kwargs)

            # Write HTML report
            with open(output_path, "w") as f:
                f.write(html)

            return output_path

        except Exception as e:
            logger.error(f"Error generating HTML report: {str(e)}")
            raise BenchmarkReportError(f"Failed to generate HTML report: {str(e)}")

    def generate_json_report(
        self, benchmark_result: BenchmarkResult, output_path: str, report_title: str, **kwargs
    ) -> str:
        """
        Generate JSON report.

        Args:
            benchmark_result: Benchmark result
            output_path: Output file path
            report_title: Report title (not used for JSON)
            **kwargs: Additional parameters

        Returns:
            Path to generated report

        Raises:
            BenchmarkReportError: If report generation fails
        """
        try:
            # Convert benchmark result to dictionary
            result_dict = benchmark_result.to_dict()

            # Add metadata
            result_dict["metadata"] = {
                "report_title": report_title,
                "generation_time": datetime.datetime.now().isoformat(),
                "generator": "Enhanced Tiered Storage I/O Benchmark Suite",
                "user": os.getenv('USER', os.getenv('USERNAME', 'Unknown')),
            }

            # Add benchmark_id to the top level for compatibility with tests
            result_dict["benchmark_id"] = (
                benchmark_result.benchmark_id
                if hasattr(benchmark_result, "benchmark_id")
                else benchmark_result.run_id
            )
            
            # Add executive summary if requested
            if kwargs.get('executive_summary_only', False):
                result_dict["executive_summary"] = self._generate_executive_summary_data(benchmark_result)
            
            # Add performance recommendations if requested
            if kwargs.get('recommendations', False):
                try:
                    result_dict["performance_recommendations"] = self._generate_performance_recommendations_data(benchmark_result)
                except Exception as e:
                    logger.warning(f"Failed to generate recommendations: {str(e)}")
                    result_dict["performance_recommendations"] = [{"text": "Error generating recommendations", "type": "error"}]

            # Write JSON report
            with open(output_path, "w") as f:
                json.dump(result_dict, f, indent=2, cls=NumpyJSONEncoder)

            return output_path

        except Exception as e:
            logger.error(f"Error generating JSON report: {str(e)}")
            raise BenchmarkReportError(f"Failed to generate JSON report: {str(e)}")

    def generate_csv_report(
        self, benchmark_result: BenchmarkResult, output_dir: str, report_title: str, **kwargs
    ) -> Dict[str, str]:
        """
        Generate CSV reports.

        Args:
            benchmark_result: Benchmark result
            output_dir: Output directory
            report_title: Report title (not used for CSV)
            **kwargs: Additional parameters

        Returns:
            Dictionary mapping report type to file path

        Raises:
            BenchmarkReportError: If report generation fails
        """
        try:
            # Get benchmark ID for unique file naming
            benchmark_id = getattr(benchmark_result, 'benchmark_id', 
                                 getattr(benchmark_result, 'run_id', 'unknown'))
            
            # Create CSV files for different aspects of the report
            csv_files = {}

            # Main summary CSV with benchmark ID
            summary_path = os.path.join(output_dir, f"{benchmark_id}_summary.csv")
            csv_files["summary"] = self._generate_summary_csv(benchmark_result, summary_path, **kwargs)
            
            # Executive summary CSV if requested
            if kwargs.get('executive_summary_only', False):
                exec_summary_path = os.path.join(output_dir, f"{benchmark_id}_executive_summary.csv")
                csv_files["executive_summary"] = self._generate_executive_summary_csv(benchmark_result, exec_summary_path)
            
            # Performance recommendations CSV if requested
            if kwargs.get('recommendations', False):
                recommendations_path = os.path.join(output_dir, f"{benchmark_id}_recommendations.csv")
                csv_files["recommendations"] = self._generate_recommendations_csv(benchmark_result, recommendations_path)

            return csv_files

        except Exception as e:
            logger.error(f"Error generating CSV report: {str(e)}")
            raise BenchmarkReportError(f"Failed to generate CSV report: {str(e)}")

    def generate_markdown_report(
        self, benchmark_result: BenchmarkResult, output_path: str, report_title: str, **kwargs
    ) -> str:
        """
        Generate Markdown report.

        Args:
            benchmark_result: Benchmark result
            output_path: Output file path
            report_title: Report title
            **kwargs: Additional parameters

        Returns:
            Path to generated report

        Raises:
            BenchmarkReportError: If report generation fails
        """
        try:
            # Generate markdown content
            md_lines = []

            # Add title
            md_lines.append(f"# {report_title}")
            md_lines.append("")

            # Add generation metadata
            md_lines.append(
                f"**Generated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            benchmark_id = (
                benchmark_result.benchmark_id
                if hasattr(benchmark_result, "benchmark_id")
                else benchmark_result.run_id
            )
            md_lines.append(f"**Benchmark ID:** {benchmark_id}")
            md_lines.append("")

            # Add summary
            md_lines.append("## Summary")
            md_lines.append("")

            summary = benchmark_result.get_summary()

            md_lines.append(f"**Benchmark Duration:** {benchmark_result.duration} seconds")
            md_lines.append(f"**Storage Tiers:** {len(benchmark_result.tiers)}")

            if "analysis_types" in summary:
                md_lines.append(f"**Analysis Types:** {', '.join(summary['analysis_types'])}")

            md_lines.append("")

            # Add tier summaries
            md_lines.append("## Tier Summaries")
            md_lines.append("")

            md_lines.append("| Tier | Avg Throughput (MB/s) | Avg IOPS | Avg Latency (ms) |")
            md_lines.append("|------|----------------------|----------|------------------|")

            for tier, tier_summary in summary.get("tier_summaries", {}).items():
                tier_name = os.path.basename(tier)
                throughput = tier_summary.get("avg_throughput_MBps", 0)
                iops = tier_summary.get("avg_iops", 0)
                latency = tier_summary.get("avg_latency_ms", 0)

                md_lines.append(f"| {tier_name} | {throughput:.2f} | {iops:.2f} | {latency:.2f} |")

            md_lines.append("")

            # Add tier comparison if available
            if "comparison" in summary:
                md_lines.append("## Tier Comparison")
                md_lines.append("")

                comparison = summary["comparison"]
                baseline = comparison.get("baseline")

                if baseline and "tier_comparisons" in comparison:
                    baseline_name = os.path.basename(baseline)
                    md_lines.append(f"**Baseline Tier:** {baseline_name}")
                    md_lines.append("")

                    md_lines.append("| Tier | Throughput Ratio | IOPS Ratio | Latency Ratio |")
                    md_lines.append("|------|-----------------|------------|---------------|")

                    for tier, tier_comparison in comparison.get("tier_comparisons", {}).items():
                        tier_name = os.path.basename(tier)

                        if "metrics" in tier_comparison:
                            metrics = tier_comparison["metrics"]
                            throughput_ratio = metrics.get("avg_throughput_MBps", {}).get(
                                "ratio", 0
                            )
                            iops_ratio = metrics.get("avg_iops", {}).get("ratio", 0)
                            latency_ratio = metrics.get("avg_latency_ms", {}).get("ratio", 0)

                            md_lines.append(
                                f"| {tier_name} | {throughput_ratio:.2f}x | {iops_ratio:.2f}x | {latency_ratio:.2f}x |"
                            )

                    md_lines.append("")

            # Add detailed results for each tier
            md_lines.append("## Detailed Results")
            md_lines.append("")

            for tier in benchmark_result.tiers:
                tier_name = os.path.basename(tier)
                tier_result = benchmark_result.get_tier_result(tier)

                md_lines.append(f"### {tier_name}")
                md_lines.append("")

                if tier_result and "tests" in tier_result:
                    md_lines.append("| Test | Throughput (MB/s) | IOPS | Latency (ms) |")
                    md_lines.append("|------|------------------|------|--------------|")

                    for test_name, test_data in tier_result["tests"].items():
                        throughput = test_data.get("throughput_MBps", 0)
                        iops = test_data.get("iops", 0)
                        latency = test_data.get("latency_ms", 0)

                        md_lines.append(
                            f"| {test_name} | {throughput:.2f} | {iops:.2f} | {latency:.2f} |"
                        )

                    md_lines.append("")

            # Add analysis results if available
            for analysis_type in summary.get("analysis_types", []):
                analysis_results = benchmark_result.get_analysis_results(analysis_type)

                if analysis_results:
                    md_lines.append(f"## {analysis_type.capitalize()} Analysis")
                    md_lines.append("")

                    if analysis_type == "statistics":
                        md_lines.append("Statistical analysis results available in JSON report.")
                    elif analysis_type == "time_series":
                        md_lines.append("Time series analysis results available in JSON report.")
                    elif analysis_type == "network":
                        self._add_network_analysis_markdown(md_lines, analysis_results)
                    elif analysis_type == "anomalies":
                        self._add_anomaly_analysis_markdown(md_lines, analysis_results)
                    elif analysis_type == "comparison":
                        # Already added above
                        pass
                    else:
                        md_lines.append(
                            f"{analysis_type.capitalize()} analysis results available in JSON report."
                        )

                    md_lines.append("")

            # Write markdown report
            with open(output_path, "w") as f:
                f.write("\n".join(md_lines))

            return output_path

        except Exception as e:
            logger.error(f"Error generating Markdown report: {str(e)}")
            raise BenchmarkReportError(f"Failed to generate Markdown report: {str(e)}")

    def generate_email_report(
        self, benchmark_result: BenchmarkResult, output_path: str, report_title: str, **kwargs
    ) -> str:
        """
        Generate email report.

        Args:
            benchmark_result: Benchmark result
            output_path: Output file path
            report_title: Report title
            **kwargs: Additional parameters

        Returns:
            Path to generated report

        Raises:
            BenchmarkReportError: If report generation fails
        """
        try:
            # Generate markdown report first
            md_path = output_path.replace(".email", ".md")
            self.generate_markdown_report(benchmark_result, md_path, report_title, **kwargs)

            # Read markdown content
            with open(md_path, "r") as f:
                md_content = f.read()

            # Try to convert markdown to HTML
            try:
                import markdown

                html_content = markdown.markdown(md_content)

                # Add email wrapper
                email_html = f"""
                <html>
                <head>
                    <title>{report_title}</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; line-height: 1.5; }}
                        table {{ border-collapse: collapse; width: 100%; }}
                        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                        th {{ background-color: #f2f2f2; }}
                    </style>
                </head>
                <body>
                    {html_content}
                    <p>
                        <em>This report was automatically generated by the Enhanced Tiered Storage I/O Benchmark Suite.</em>
                    </p>
                </body>
                </html>
                """

                # Write email report
                with open(output_path, "w") as f:
                    f.write(email_html)

            except ImportError:
                # Markdown not available, use plain text
                logger.warning("Markdown module not available, using plain text email")

                # Write email report
                with open(output_path, "w") as f:
                    f.write(f"Subject: {report_title}\n\n")
                    f.write(md_content)
                    f.write(
                        "\n\nThis report was automatically generated by the Enhanced Tiered Storage I/O Benchmark Suite."
                    )

            return output_path

        except Exception as e:
            logger.error(f"Error generating email report: {str(e)}")
            raise BenchmarkReportError(f"Failed to generate email report: {str(e)}")

    def _generate_default_html_report(
        self, benchmark_result: BenchmarkResult, report_title: str, **kwargs
    ) -> str:
        """
        Generate default HTML report without templates.

        Args:
            benchmark_result: Benchmark result
            report_title: Report title
            **kwargs: Additional parameters including recommendations flag

        Returns:
            HTML content
        """
        # Generate HTML content
        html = []

        # Add header
        html.append("<!DOCTYPE html>")
        html.append("<html>")
        html.append("<head>")
        html.append(f"<title>{report_title}</title>")
        html.append("<meta charset='UTF-8'>")
        html.append("<meta name='viewport' content='width=device-width, initial-scale=1.0'>")
        html.append("<style>")
        
        # Professional CSS styling with consistent color scheme
        html.append("""
        :root {
            --primary-blue: #1e3a8a;
            --secondary-blue: #3b82f6;
            --light-blue: #dbeafe;
            --accent-teal: #0d9488;
            --light-teal: #ccfbf1;
            --gray-50: #f9fafb;
            --gray-100: #f3f4f6;
            --gray-200: #e5e7eb;
            --gray-300: #d1d5db;
            --gray-600: #4b5563;
            --gray-700: #374151;
            --gray-800: #1f2937;
            --gray-900: #111827;
            --success: #059669;
            --warning: #d97706;
            --error: #dc2626;
            --info: #0284c7;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            background: var(--gray-50);
            color: var(--gray-800);
            font-size: 14px;
        }
        
        .container {
            max-width: 1200px;
            margin: 2rem auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, var(--primary-blue) 0%, var(--secondary-blue) 100%);
            color: white;
            padding: 2rem;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.25rem;
            font-weight: 700;
            margin: 0;
            letter-spacing: -0.025em;
        }
        
        .content {
            padding: 2rem;
        }
        
        h2 {
            font-size: 1.5rem;
            font-weight: 600;
            color: var(--gray-900);
            margin: 2rem 0 1rem 0;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid var(--light-blue);
            position: relative;
        }
        
        h2::before {
            content: '';
            position: absolute;
            bottom: -2px;
            left: 0;
            width: 4rem;
            height: 2px;
            background: var(--secondary-blue);
        }
        
        h3 {
            font-size: 1.25rem;
            font-weight: 600;
            color: var(--gray-700);
            margin: 1.5rem 0 1rem 0;
            padding: 0.75rem 1rem;
            background: var(--gray-100);
            border-left: 4px solid var(--accent-teal);
            border-radius: 0 6px 6px 0;
        }
        
        .metadata {
            background: var(--gray-100);
            padding: 1.5rem;
            border-radius: 8px;
            margin-bottom: 2rem;
            border: 1px solid var(--gray-200);
        }
        
        .metadata p {
            margin: 0.5rem 0;
            font-size: 0.875rem;
            color: var(--gray-600);
        }
        
        .metadata strong {
            color: var(--gray-800);
            font-weight: 600;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 1.5rem 0;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
        }
        
        th {
            background: var(--primary-blue);
            color: white;
            padding: 1rem 0.75rem;
            text-align: left;
            font-weight: 600;
            font-size: 0.875rem;
            letter-spacing: 0.025em;
        }
        
        td {
            padding: 0.875rem 0.75rem;
            border-bottom: 1px solid var(--gray-200);
            font-size: 0.875rem;
        }
        
        tr:hover {
            background-color: var(--gray-50);
        }
        
        tr:last-child td {
            border-bottom: none;
        }
        
        tbody tr:nth-child(even) {
            background-color: rgba(249, 250, 251, 0.5);
        }
        
        /* Performance indicators with consistent colors */
        .perf-excellent { 
            color: var(--success); 
            font-weight: 600; 
        }
        
        .perf-good { 
            color: var(--info); 
            font-weight: 600; 
        }
        
        .perf-warning { 
            color: var(--warning); 
            font-weight: 600; 
        }
        
        .perf-poor { 
            color: var(--error); 
            font-weight: 600; 
        }
        
        .summary {
            background: linear-gradient(135deg, var(--primary-blue) 0%, var(--secondary-blue) 100%);
            color: white;
            padding: 2rem;
            border-radius: 8px;
            margin: 2rem 0;
        }
        
        .summary h2 {
            color: white;
            border-bottom: 2px solid rgba(255, 255, 255, 0.3);
            margin-top: 0;
        }
        
        .summary h2::before {
            background: white;
        }
        
        .recommendations {
            background: linear-gradient(135deg, var(--accent-teal) 0%, var(--success) 100%);
            color: white;
            padding: 2rem;
            border-radius: 8px;
            margin: 2rem 0;
        }
        
        .recommendations h2 {
            color: white;
            border-bottom: 2px solid rgba(255, 255, 255, 0.3);
            margin-top: 0;
        }
        
        .recommendations h2::before {
            background: white;
        }
        
        .recommendations ul {
            margin: 1rem 0;
            padding-left: 1.5rem;
        }
        
        .recommendations li {
            margin: 0.75rem 0;
            padding: 0.5rem 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.2);
            font-size: 0.95rem;
            line-height: 1.5;
        }
        
        .recommendations li:last-child {
            border-bottom: none;
        }
        
        .badge {
            display: inline-block;
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        .badge-read { 
            background: var(--light-blue); 
            color: var(--primary-blue); 
        }
        
        .badge-write { 
            background: #fef3c7; 
            color: var(--warning); 
        }
        
        .badge-mixed { 
            background: var(--light-teal); 
            color: var(--accent-teal); 
        }
        
        .metric-card {
            background: white;
            border-radius: 8px;
            padding: 1.5rem;
            margin: 1rem 0;
            border: 1px solid var(--gray-200);
        }
        
        .footer {
            text-align: center;
            padding: 2rem;
            background: var(--gray-100);
            color: var(--gray-600);
            border-top: 1px solid var(--gray-200);
            margin-top: 2rem;
        }
        
        .footer p {
            margin: 0.25rem 0;
            font-size: 0.875rem;
        }
        
        /* Icons using CSS */
        .icon-clock::before { content: '⏱'; margin-right: 0.5rem; }
        .icon-database::before { content: '🗃'; margin-right: 0.5rem; }
        .icon-chart::before { content: '📊'; margin-right: 0.5rem; }
        .icon-gear::before { content: '⚙'; margin-right: 0.5rem; }
        .icon-lightbulb::before { content: '💡'; margin-right: 0.5rem; }
        .icon-folder::before { content: '📁'; margin-right: 0.5rem; }
        .icon-list::before { content: '📋'; margin-right: 0.5rem; }
        
        /* Responsive design */
        @media (max-width: 768px) {
            .container { 
                margin: 1rem; 
                border-radius: 8px; 
            }
            .header { 
                padding: 1.5rem; 
            }
            .header h1 { 
                font-size: 1.875rem; 
            }
            .content { 
                padding: 1rem; 
            }
            table { 
                font-size: 0.75rem; 
            }
            th, td { 
                padding: 0.5rem 0.375rem; 
            }
        }
        
        /* Subtle animations */
        .container {
            animation: slideIn 0.3s ease-out;
        }
        
        @keyframes slideIn {
            from { 
                opacity: 0; 
                transform: translateY(1rem); 
            }
            to { 
                opacity: 1; 
                transform: translateY(0); 
            }
        }
        """)
        
        html.append("</style>")
        html.append("</head>")
        html.append("<body>")
        html.append("<div class='container'>")
        
        # Clean professional header
        html.append("<div class='header'>")
        html.append(f"<h1>Storage I/O Benchmark Report</h1>")
        html.append("</div>")
        
        html.append("<div class='content'>")

        # Add generation metadata with professional styling
        html.append("<div class='metadata'>")
        html.append(
            f"<p><strong>Generated:</strong> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>"
        )
        
        # Use display_label if provided in kwargs, otherwise fall back to benchmark_id
        display_label = kwargs.get('display_label')
        if display_label:
            html.append(f"<p><strong>Benchmark:</strong> {display_label}</p>")
            # Also show the actual ID for reference
            benchmark_id = (
                benchmark_result.benchmark_id
                if hasattr(benchmark_result, "benchmark_id")
                else benchmark_result.run_id
            )
            html.append(f"<p><strong>ID:</strong> <code>{benchmark_id}</code></p>")
        else:
            benchmark_id = (
                benchmark_result.benchmark_id
                if hasattr(benchmark_result, "benchmark_id")
                else benchmark_result.run_id
            )
            html.append(f"<p><strong>Benchmark ID:</strong> {benchmark_id}</p>")
        
        html.append("</div>")

        # Add professional summary
        html.append("<div class='summary'>")
        html.append("<h2>Performance Summary</h2>")

        summary = benchmark_result.get_summary()

        html.append("<div class='metric-card'>")
        html.append(f"<p><strong>Benchmark Duration:</strong> {benchmark_result.duration} seconds</p>")
        html.append(f"<p><strong>Storage Tiers:</strong> {len(benchmark_result.tiers)}</p>")

        if "analysis_types" in summary:
            html.append(
                f"<p><strong>Analysis Types:</strong> {', '.join(summary['analysis_types'])}</p>"
            )

        html.append("</div>")
        html.append("</div>")

        # Add tier summaries with professional styling
        html.append("<h2>Tier Performance Overview</h2>")

        html.append("<table>")
        html.append(
            "<tr><th>Tier</th><th>Avg Throughput (MB/s)</th><th>Avg IOPS</th><th>Avg Latency (ms)</th></tr>"
        )

        for tier, tier_summary in summary.get("tier_summaries", {}).items():
            tier_name = os.path.basename(tier)
            throughput = tier_summary.get("avg_throughput_MBps", 0)
            iops = tier_summary.get("avg_iops", 0)
            latency = tier_summary.get("avg_latency_ms", 0)
            
            # Add performance indicators based on values
            throughput_class = ""
            iops_class = ""
            latency_class = ""
            
            if throughput > 1000:
                throughput_class = "perf-excellent"
            elif throughput > 500:
                throughput_class = "perf-good"
            elif throughput > 100:
                throughput_class = "perf-warning"
            else:
                throughput_class = "perf-poor"
                
            if iops > 50000:
                iops_class = "perf-excellent"
            elif iops > 10000:
                iops_class = "perf-good"
            elif iops > 1000:
                iops_class = "perf-warning"
            else:
                iops_class = "perf-poor"
                
            if latency < 1:
                latency_class = "perf-excellent"
            elif latency < 5:
                latency_class = "perf-good"
            elif latency < 20:
                latency_class = "perf-warning"
            else:
                latency_class = "perf-poor"

            html.append(
                f"<tr>"
                f"<td><strong>{tier_name}</strong></td>"
                f"<td class='{throughput_class}'>{throughput:.2f}</td>"
                f"<td class='{iops_class}'>{iops:.2f}</td>"
                f"<td class='{latency_class}'>{latency:.2f}</td>"
                f"</tr>"
            )

        html.append("</table>")

        # Add tier comparison if available
        if "comparison" in summary:
            html.append("<h2>Tier Comparison</h2>")

            comparison = summary["comparison"]
            baseline = comparison.get("baseline")

            if baseline and "tier_comparisons" in comparison:
                baseline_name = os.path.basename(baseline)
                html.append(f"<p><strong>Baseline Tier:</strong> {baseline_name}</p>")

                html.append("<table>")
                html.append(
                    "<tr><th>Tier</th><th>Throughput Ratio</th><th>IOPS Ratio</th><th>Latency Ratio</th></tr>"
                )

                for tier, tier_comparison in comparison.get("tier_comparisons", {}).items():
                    tier_name = os.path.basename(tier)

                    if "metrics" in tier_comparison:
                        metrics = tier_comparison["metrics"]
                        throughput_ratio = metrics.get("avg_throughput_MBps", {}).get("ratio", 0)
                        iops_ratio = metrics.get("avg_iops", {}).get("ratio", 0)
                        latency_ratio = metrics.get("avg_latency_ms", {}).get("ratio", 0)

                        html.append(
                            f"<tr><td>{tier_name}</td><td>{throughput_ratio:.2f}x</td><td>{iops_ratio:.2f}x</td><td>{latency_ratio:.2f}x</td></tr>"
                        )

                html.append("</table>")

        # Add detailed results for each tier with professional styling
        html.append("<h2>Detailed Performance Results</h2>")

        for tier in benchmark_result.tiers:
            tier_name = os.path.basename(tier)
            html.append(f"<h3>{tier_name}</h3>")
            
            # Get tier data from metrics
            tier_data = benchmark_result.metrics.get(f"./{tier}", benchmark_result.metrics.get(tier, {}))
            
            if tier_data:
                html.append("<table>")
                html.append(
                    "<tr>"
                    "<th>Pattern</th>"
                    "<th>Block Size</th>"
                    "<th>Read IOPS</th>"
                    "<th>Write IOPS</th>"
                    "<th>Total IOPS</th>"
                    "<th>Throughput (MB/s)</th>"
                    "<th>Latency (ms)</th>"
                    "</tr>"
                )

                for pattern_key, pattern_data in tier_data.items():
                    if isinstance(pattern_data, dict) and any(key in pattern_data for key in ['throughput_MBps', 'iops', 'latency_ms']):
                        # Extract pattern type and block size from pattern_key (e.g., "read_4k" -> "read", "4k")
                        parts = pattern_key.split('_')
                        pattern_type = parts[0] if parts else pattern_key
                        block_size = parts[1] if len(parts) > 1 else 'unknown'
                        
                        # Get metrics
                        total_iops = pattern_data.get('iops', 0)
                        total_throughput = pattern_data.get('throughput_MBps', 0)
                        total_latency = pattern_data.get('latency_ms', 0)
                        
                        # Get read/write specific metrics if available
                        read_data = pattern_data.get('read', {})
                        write_data = pattern_data.get('write', {})
                        
                        read_iops = read_data.get('iops', 0)
                        write_iops = write_data.get('iops', 0)
                        
                        # Determine pattern badge style
                        pattern_badge = ""
                        if pattern_type.lower() == "read":
                            pattern_badge = f"<span class='badge badge-read'>{pattern_type.title()}</span>"
                        elif pattern_type.lower() == "write":
                            pattern_badge = f"<span class='badge badge-write'>{pattern_type.title()}</span>"
                        else:
                            pattern_badge = f"<span class='badge badge-mixed'>{pattern_type.title()}</span>"
                        
                        # Apply performance styling to metrics
                        iops_class = "perf-excellent" if total_iops > 50000 else "perf-good" if total_iops > 10000 else "perf-warning" if total_iops > 1000 else "perf-poor"
                        throughput_class = "perf-excellent" if total_throughput > 1000 else "perf-good" if total_throughput > 500 else "perf-warning" if total_throughput > 100 else "perf-poor"
                        latency_class = "perf-excellent" if total_latency < 1 else "perf-good" if total_latency < 5 else "perf-warning" if total_latency < 20 else "perf-poor"
                        
                        html.append(
                            f"<tr>"
                            f"<td>{pattern_badge}</td>"
                            f"<td><strong>{block_size}</strong></td>"
                            f"<td>{read_iops:.0f}</td>"
                            f"<td>{write_iops:.0f}</td>"
                            f"<td class='{iops_class}'><strong>{total_iops:.0f}</strong></td>"
                            f"<td class='{throughput_class}'><strong>{total_throughput:.2f}</strong></td>"
                            f"<td class='{latency_class}'><strong>{total_latency:.3f}</strong></td>"
                            f"</tr>"
                        )

                html.append("</table>")
            else:
                html.append("<p>No detailed results available for this tier.</p>")
        
        # Add performance recommendations only if requested with professional styling
        if kwargs.get('recommendations', False):
            html.append("<div class='recommendations'>")
            html.append("<h2>Performance Recommendations</h2>")
            
            # Generate recommendations based on the data
            recommendations = self._generate_performance_recommendations(benchmark_result)
            
            if recommendations:
                html.append("<ul>")
                for recommendation in recommendations:
                    html.append(f"<li>{recommendation}</li>")
                html.append("</ul>")
            else:
                html.append("<p>No specific recommendations available based on current results.</p>")
            
            html.append("</div>")

        # Close content div
        html.append("</div>")

        # Add professional footer
        html.append("<div class='footer'>")
        html.append(
            "<p>This report was automatically generated by the Enhanced Tiered Storage I/O Benchmark Suite</p>"
            "<p>Built by Jack Ogaja | Powered by eTIOBench</p>"
        )
        html.append("</div>")

        # Close container
        html.append("</div>")
        html.append("</body>")
        html.append("</html>")

        return "\n".join(html)

    def _generate_summary_csv(self, benchmark_result: BenchmarkResult, output_path: str, **kwargs) -> str:
        """
        Generate comprehensive summary CSV with detailed results.

        Args:
            benchmark_result: Benchmark result
            output_path: Output file path

        Returns:
            Path to generated CSV
        """
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)

            # Write benchmark summary header
            benchmark_id = (
                benchmark_result.benchmark_id
                if hasattr(benchmark_result, "benchmark_id")
                else benchmark_result.run_id
            )
            
            writer.writerow(["Benchmark Summary"])
            if kwargs.get('display_label'):
                # Use display_label if provided
                display_label = kwargs.get('display_label')
                writer.writerow(["Benchmark", display_label])
                writer.writerow(["ID", benchmark_id])
            else:
                # Default behavior - just show benchmark ID
                writer.writerow(["Benchmark ID", benchmark_id])
            writer.writerow(["Start Time", getattr(benchmark_result, 'start_time', '')])
            writer.writerow(["End Time", getattr(benchmark_result, 'end_time', '')])
            writer.writerow(["Duration (seconds)", getattr(benchmark_result, 'duration_seconds', getattr(benchmark_result, 'duration', 0))])
            writer.writerow(["Storage Tiers", len(getattr(benchmark_result, 'tiers', []))])
            writer.writerow([])  # Empty row for separation
            
            # Write tier summaries if available
            if hasattr(benchmark_result, 'tier_results') and benchmark_result.tier_results:
                writer.writerow(["Tier Summaries"])
                writer.writerow(["Tier", "Avg Throughput (MB/s)", "Avg IOPS", "Avg Latency (ms)", "Max Throughput (MB/s)", "Max IOPS", "Min Latency (ms)"])
                
                for tier_name, tier_data in benchmark_result.tier_results.items():
                    if 'summary' in tier_data:
                        summary_data = tier_data['summary']
                        writer.writerow([
                            tier_name,
                            round(summary_data.get('avg_throughput_MBps', 0), 2),
                            round(summary_data.get('avg_iops', 0), 2),
                            round(summary_data.get('avg_latency_ms', 0), 3),
                            round(summary_data.get('max_throughput_MBps', 0), 2),
                            round(summary_data.get('max_iops', 0), 2),
                            round(summary_data.get('min_latency_ms', 0), 3)
                        ])
                
                writer.writerow([])  # Empty row for separation
            
            # Write detailed results
            writer.writerow(["Detailed Results"])
            writer.writerow(["Tier", "Pattern", "Block Size", "Read IOPS", "Write IOPS", "Total IOPS", "Throughput (MB/s)", "Latency (ms)"])
            
            # Extract detailed results from metrics or tier_results
            if hasattr(benchmark_result, 'tier_results') and benchmark_result.tier_results:
                for tier_name, tier_data in benchmark_result.tier_results.items():
                    if 'raw_data' in tier_data:
                        raw_data = tier_data['raw_data']
                        for pattern, pattern_data in raw_data.items():
                            if isinstance(pattern_data, dict):
                                # Extract pattern and block size from key (e.g., "read_4k" -> "Read", "4k")
                                pattern_parts = pattern.lower().split('_')
                                pattern_clean = pattern_parts[0].title()
                                block_size = "unknown"
                                
                                # Try to extract block size from pattern name (check longer patterns first)
                                for size in ['64k', '4m', '8m', '1m', '4k']:
                                    if size in pattern.lower():
                                        block_size = size
                                        break
                                
                                # Clean up pattern name
                                if len(pattern_parts) > 1:
                                    # For patterns like "read_64k", "randrw_4k"
                                    if pattern_parts[-1] in ['4k', '64k', '1m', '4m', '8m']:
                                        pattern_clean = '_'.join(pattern_parts[:-1]).title()
                                    else:
                                        pattern_clean = pattern.replace('_', ' ').title()
                                
                                # Get read/write IOPS
                                read_iops = 0
                                write_iops = 0
                                if 'read' in pattern_data:
                                    read_iops = pattern_data['read'].get('iops', 0)
                                if 'write' in pattern_data:
                                    write_iops = pattern_data['write'].get('iops', 0)
                                
                                writer.writerow([
                                    tier_name,
                                    pattern_clean,
                                    block_size,
                                    round(read_iops, 0),
                                    round(write_iops, 0),
                                    round(pattern_data.get('iops', 0), 0),
                                    round(pattern_data.get('throughput_MBps', 0), 2),
                                    round(pattern_data.get('latency_ms', 0), 3)
                                ])
            elif hasattr(benchmark_result, 'metrics') and benchmark_result.metrics:
                # Fall back to direct metrics parsing
                for tier_path, tier_data in benchmark_result.metrics.items():
                    tier_name = tier_path.split('/')[-1] if '/' in tier_path else tier_path
                    if isinstance(tier_data, dict):
                        for pattern, pattern_data in tier_data.items():
                            if isinstance(pattern_data, dict):
                                # Extract pattern and block size
                                pattern_parts = pattern.lower().split('_')
                                pattern_clean = pattern_parts[0].title()
                                block_size = "unknown"
                                
                                # Try to extract block size from pattern name (check longer patterns first)
                                for size in ['64k', '4m', '8m', '1m', '4k']:
                                    if size in pattern.lower():
                                        block_size = size
                                        break
                                
                                # Clean up pattern name
                                if len(pattern_parts) > 1:
                                    # For patterns like "read_64k", "randrw_4k"
                                    if pattern_parts[-1] in ['4k', '64k', '1m', '4m', '8m']:
                                        pattern_clean = '_'.join(pattern_parts[:-1]).title()
                                    else:
                                        pattern_clean = pattern.replace('_', ' ').title()
                                
                                # Get read/write IOPS
                                read_iops = 0
                                write_iops = 0
                                if 'read' in pattern_data:
                                    read_iops = pattern_data['read'].get('iops', 0)
                                if 'write' in pattern_data:
                                    write_iops = pattern_data['write'].get('iops', 0)
                                
                                writer.writerow([
                                    tier_name,
                                    pattern_clean,
                                    block_size,
                                    round(read_iops, 0),
                                    round(write_iops, 0),
                                    round(pattern_data.get('iops', 0), 0),
                                    round(pattern_data.get('throughput_MBps', 0), 2),
                                    round(pattern_data.get('latency_ms', 0), 3)
                                ])

        return output_path

    def _generate_tiers_csv(self, benchmark_result: BenchmarkResult, output_path: str) -> str:
        """
        Generate tiers CSV.

        Args:
            benchmark_result: Benchmark result
            output_path: Output file path

        Returns:
            Path to generated CSV
        """
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)

            # Write header
            writer.writerow(["Tier", "Avg Throughput (MB/s)", "Avg IOPS", "Avg Latency (ms)"])

            # Write data
            for tier in benchmark_result.tiers:
                tier_result = benchmark_result.get_tier_result(tier)

                if tier_result and "summary" in tier_result:
                    summary = tier_result["summary"]

                    writer.writerow(
                        [
                            tier,
                            summary.get("avg_throughput_MBps", 0),
                            summary.get("avg_iops", 0),
                            summary.get("avg_latency_ms", 0),
                        ]
                    )

        return output_path

    def _generate_tests_csv(self, benchmark_result: BenchmarkResult, output_path: str) -> str:
        """
        Generate tests CSV.

        Args:
            benchmark_result: Benchmark result
            output_path: Output file path

        Returns:
            Path to generated CSV
        """
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)

            # Write header
            writer.writerow(["Tier", "Test", "Throughput (MB/s)", "IOPS", "Latency (ms)"])

            # Write data
            for tier in benchmark_result.tiers:
                tier_result = benchmark_result.get_tier_result(tier)

                if tier_result and "tests" in tier_result:
                    for test_name, test_data in tier_result["tests"].items():
                        writer.writerow(
                            [
                                tier,
                                test_name,
                                test_data.get("throughput_MBps", 0),
                                test_data.get("iops", 0),
                                test_data.get("latency_ms", 0),
                            ]
                        )

        return output_path

    def _generate_time_series_csv(self, time_series_data: Dict[str, Any], output_path: str) -> str:
        """
        Generate time series CSV.

        Args:
            time_series_data: Time series data
            output_path: Output file path

        Returns:
            Path to generated CSV
        """
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)

            # Get metrics
            metrics = list(time_series_data["data"].keys())

            # Write header
            writer.writerow(["Timestamp"] + metrics)

            # Write data
            timestamps = time_series_data["timestamps"]

            for i, timestamp in enumerate(timestamps):
                row = [timestamp]

                for metric in metrics:
                    if i < len(time_series_data["data"][metric]):
                        row.append(time_series_data["data"][metric][i])
                    else:
                        row.append(None)

                writer.writerow(row)

        return output_path

    def _generate_executive_summary_csv(self, benchmark_result: BenchmarkResult, output_path: str) -> str:
        """
        Generate executive summary CSV.

        Args:
            benchmark_result: Benchmark result
            output_path: Output file path

        Returns:
            Path to generated CSV
        """
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)

            # Get executive summary data
            exec_data = self._generate_executive_summary_data(benchmark_result)
            
            # Write basic info
            writer.writerow(["Metric", "Value"])
            writer.writerow(["Benchmark ID", exec_data.get("benchmark_id", "")])
            writer.writerow(["Duration (seconds)", exec_data.get("duration_seconds", 0)])
            writer.writerow(["Storage Tiers", exec_data.get("storage_tiers", 0)])
            writer.writerow([])  # Empty row for separation
            
            # Write tier summaries
            writer.writerow(["Tier Summaries"])
            writer.writerow(["Tier", "Avg Throughput (MB/s)", "Avg IOPS", "Avg Latency (ms)", "Max Throughput (MB/s)", "Max IOPS", "Min Latency (ms)"])
            
            for tier_name, tier_summary in exec_data.get("tier_summaries", {}).items():
                writer.writerow([
                    tier_name,
                    tier_summary.get("avg_throughput_MBps", 0),
                    tier_summary.get("avg_iops", 0),
                    tier_summary.get("avg_latency_ms", 0),
                    tier_summary.get("max_throughput_MBps", 0),
                    tier_summary.get("max_iops", 0),
                    tier_summary.get("min_latency_ms", 0)
                ])
            
            writer.writerow([])  # Empty row for separation
            
            # Write detailed results
            writer.writerow(["Detailed Results"])
            writer.writerow(["Tier", "Pattern", "Block Size", "Read IOPS", "Write IOPS", "Total IOPS", "Throughput (MB/s)", "Latency (ms)"])
            
            for result in exec_data.get("detailed_results", []):
                writer.writerow([
                    result.get("tier", ""),
                    result.get("pattern", ""),
                    result.get("block_size", ""),
                    result.get("read_iops", 0),
                    result.get("write_iops", 0),
                    result.get("total_iops", 0),
                    result.get("throughput_MBps", 0),
                    result.get("latency_ms", 0)
                ])

        return output_path

    def _generate_recommendations_csv(self, benchmark_result: BenchmarkResult, output_path: str) -> str:
        """
        Generate performance recommendations CSV.

        Args:
            benchmark_result: Benchmark result
            output_path: Output file path

        Returns:
            Path to generated CSV
        """
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)

            # Get recommendations data
            recommendations_data = self._generate_performance_recommendations_data(benchmark_result)
            
            # Write header
            writer.writerow(["Recommendation", "Type", "Severity", "Tier"])
            
            # Write recommendations
            for rec in recommendations_data:
                writer.writerow([
                    rec.get("text", ""),
                    rec.get("type", "general"),
                    rec.get("severity", "info"),
                    rec.get("tier", "")
                ])

        return output_path

    def _generate_charts(self, benchmark_result: BenchmarkResult) -> Dict[str, Any]:
        """
        Generate chart configurations for HTML and other reports.

        Args:
            benchmark_result: Benchmark result

        Returns:
            Dictionary containing chart configurations
        """
        charts = {}

        try:
            # Try to generate chart configurations

            # Tier comparison chart
            charts["tier_comparison"] = {
                "type": "bar",
                "data": {"labels": [], "datasets": []},
                "options": {
                    "title": {"display": True, "text": "Tier Performance Comparison"},
                    "scales": {"yAxes": [{"ticks": {"beginAtZero": True}}]},
                },
            }

            # Add tier data
            throughput_data = []
            iops_data = []
            latency_data = []
            tier_labels = []

            for tier in benchmark_result.tiers:
                tier_name = os.path.basename(tier)
                tier_result = benchmark_result.get_tier_result(tier)

                if tier_result and "summary" in tier_result:
                    summary = tier_result["summary"]

                    tier_labels.append(tier_name)
                    throughput_data.append(summary.get("avg_throughput_MBps", 0))
                    iops_data.append(summary.get("avg_iops", 0))
                    latency_data.append(summary.get("avg_latency_ms", 0))

            # Update chart data
            charts["tier_comparison"]["data"]["labels"] = tier_labels

            charts["tier_comparison"]["data"]["datasets"] = [
                {
                    "label": "Throughput (MB/s)",
                    "data": throughput_data,
                    "backgroundColor": "rgba(54, 162, 235, 0.5)",
                    "borderColor": "rgba(54, 162, 235, 1)",
                    "borderWidth": 1,
                },
                {
                    "label": "IOPS",
                    "data": iops_data,
                    "backgroundColor": "rgba(255, 99, 132, 0.5)",
                    "borderColor": "rgba(255, 99, 132, 1)",
                    "borderWidth": 1,
                },
                {
                    "label": "Latency (ms)",
                    "data": latency_data,
                    "backgroundColor": "rgba(75, 192, 192, 0.5)",
                    "borderColor": "rgba(75, 192, 192, 1)",
                    "borderWidth": 1,
                },
            ]

            # Time series chart if available
            if benchmark_result.has_time_series_data():
                time_series = benchmark_result.get_time_series_dataframe()

                if not time_series.empty:
                    # Create time series chart
                    charts["time_series"] = {
                        "type": "line",
                        "data": {"labels": [], "datasets": []},
                        "options": {
                            "title": {"display": True, "text": "Performance Over Time"},
                            "scales": {"xAxes": [{"type": "time", "time": {"unit": "second"}}]},
                        },
                    }

                    # Format time series data for chart
                    # This is simplified - in reality you'd need to handle the time format correctly
                    charts["time_series"]["data"]["labels"] = time_series.index.tolist()

                    # Add IOPS dataset
                    if "iops" in time_series.columns:
                        charts["time_series"]["data"]["datasets"].append(
                            {
                                "label": "IOPS",
                                "data": time_series["iops"].tolist(),
                                "borderColor": "rgba(255, 99, 132, 1)",
                                "backgroundColor": "rgba(255, 99, 132, 0.1)",
                                "fill": True,
                            }
                        )

                    # Add throughput dataset
                    if "throughput_MBps" in time_series.columns:
                        charts["time_series"]["data"]["datasets"].append(
                            {
                                "label": "Throughput (MB/s)",
                                "data": time_series["throughput_MBps"].tolist(),
                                "borderColor": "rgba(54, 162, 235, 1)",
                                "backgroundColor": "rgba(54, 162, 235, 0.1)",
                                "fill": True,
                            }
                        )

                    # Add latency dataset
                    if "latency_ms" in time_series.columns:
                        charts["time_series"]["data"]["datasets"].append(
                            {
                                "label": "Latency (ms)",
                                "data": time_series["latency_ms"].tolist(),
                                "borderColor": "rgba(75, 192, 192, 1)",
                                "backgroundColor": "rgba(75, 192, 192, 0.1)",
                                "fill": True,
                            }
                        )

        except Exception as e:
            logger.warning(f"Error generating charts: {str(e)}")
            # Return empty charts dictionary on error
            return {}

        return charts

    def _generate_performance_recommendations(self, benchmark_result: BenchmarkResult) -> List[str]:
        """
        Generate performance recommendations based on benchmark results.
        
        Args:
            benchmark_result: Benchmark result data
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        try:
            # Analyze metrics for each tier
            for tier in benchmark_result.tiers:
                tier_data = benchmark_result.metrics.get(f"./{tier}", benchmark_result.metrics.get(tier, {}))
                
                if not tier_data:
                    continue
                    
                # Collect all IOPS, throughput, and latency values
                all_iops = []
                all_throughput = []
                all_latency = []
                
                for pattern_key, pattern_data in tier_data.items():
                    if isinstance(pattern_data, dict):
                        if 'iops' in pattern_data:
                            all_iops.append(pattern_data['iops'])
                        if 'throughput_MBps' in pattern_data:
                            all_throughput.append(pattern_data['throughput_MBps'])
                        if 'latency_ms' in pattern_data:
                            all_latency.append(pattern_data['latency_ms'])
                
                if not all_iops and not all_throughput and not all_latency:
                    continue
                
                # Calculate averages
                avg_iops = sum(all_iops) / len(all_iops) if all_iops else 0
                avg_throughput = sum(all_throughput) / len(all_throughput) if all_throughput else 0
                avg_latency = sum(all_latency) / len(all_latency) if all_latency else 0
                max_iops = max(all_iops) if all_iops else 0
                max_throughput = max(all_throughput) if all_throughput else 0
                min_latency = min(all_latency) if all_latency else 0
                
                tier_name = tier.replace('./', '').replace('_', ' ').title()
                
                # IOPS-based recommendations
                if max_iops > 100000:
                    recommendations.append(f"✅ {tier_name}: Excellent IOPS performance ({max_iops:.0f}) - suitable for high-transaction workloads")
                elif max_iops > 50000:
                    recommendations.append(f"✅ {tier_name}: Good IOPS performance ({max_iops:.0f}) - suitable for database applications")
                elif max_iops > 10000:
                    recommendations.append(f"⚠️ {tier_name}: Moderate IOPS ({max_iops:.0f}) - consider SSD upgrade for better performance")
                elif max_iops > 0:
                    recommendations.append(f"🔧 {tier_name}: Low IOPS ({max_iops:.0f}) - recommend NVMe SSD for significant improvement")
                
                # Throughput-based recommendations
                if max_throughput > 2000:
                    recommendations.append(f"✅ {tier_name}: High throughput ({max_throughput:.1f} MB/s) - excellent for large file operations")
                elif max_throughput > 500:
                    recommendations.append(f"✅ {tier_name}: Good throughput ({max_throughput:.1f} MB/s) - suitable for most workloads")
                elif max_throughput > 100:
                    recommendations.append(f"⚠️ {tier_name}: Moderate throughput ({max_throughput:.1f} MB/s) - check storage configuration")
                elif max_throughput > 0:
                    recommendations.append(f"🔧 {tier_name}: Low throughput ({max_throughput:.1f} MB/s) - check network and storage configuration")
                
                # Latency-based recommendations
                if min_latency < 1.0:
                    recommendations.append(f"✅ {tier_name}: Excellent latency ({min_latency:.3f}ms) - perfect for real-time applications")
                elif min_latency < 5.0:
                    recommendations.append(f"✅ {tier_name}: Good latency ({min_latency:.3f}ms) - suitable for most applications")
                elif min_latency < 20.0:
                    recommendations.append(f"⚠️ {tier_name}: Moderate latency ({min_latency:.3f}ms) - consider faster storage for latency-sensitive apps")
                else:
                    recommendations.append(f"🔧 {tier_name}: High latency ({min_latency:.3f}ms) - investigate storage configuration and consider faster storage")
                
                # Pattern-specific recommendations
                read_patterns = [k for k in tier_data.keys() if 'read' in k.lower() and not 'write' in k.lower()]
                write_patterns = [k for k in tier_data.keys() if 'write' in k.lower() and not 'read' in k.lower()]
                randrw_patterns = [k for k in tier_data.keys() if 'randrw' in k.lower()]
                
                if read_patterns and write_patterns:
                    # Compare read vs write performance
                    read_perf = []
                    write_perf = []
                    
                    for pattern in read_patterns:
                        if tier_data[pattern].get('throughput_MBps', 0) > 0:
                            read_perf.append(tier_data[pattern]['throughput_MBps'])
                    
                    for pattern in write_patterns:
                        if tier_data[pattern].get('throughput_MBps', 0) > 0:
                            write_perf.append(tier_data[pattern]['throughput_MBps'])
                    
                    if read_perf and write_perf:
                        avg_read = sum(read_perf) / len(read_perf)
                        avg_write = sum(write_perf) / len(write_perf)
                        
                        if avg_read > avg_write * 2:
                            recommendations.append(f"📊 {tier_name}: Read-optimized storage (Read: {avg_read:.1f} MB/s vs Write: {avg_write:.1f} MB/s)")
                        elif avg_write > avg_read * 2:
                            recommendations.append(f"📊 {tier_name}: Write-optimized storage (Write: {avg_write:.1f} MB/s vs Read: {avg_read:.1f} MB/s)")
                        else:
                            recommendations.append(f"📊 {tier_name}: Balanced read/write performance (Read: {avg_read:.1f} MB/s, Write: {avg_write:.1f} MB/s)")
                
                # Block size recommendations
                small_block_patterns = [k for k in tier_data.keys() if '4k' in k.lower()]
                large_block_patterns = [k for k in tier_data.keys() if any(size in k.lower() for size in ['64k', '1m', '4m'])]
                
                if small_block_patterns and large_block_patterns:
                    small_block_throughput = []
                    large_block_throughput = []
                    
                    for pattern in small_block_patterns:
                        if tier_data[pattern].get('throughput_MBps', 0) > 0:
                            small_block_throughput.append(tier_data[pattern]['throughput_MBps'])
                    
                    for pattern in large_block_patterns:
                        if tier_data[pattern].get('throughput_MBps', 0) > 0:
                            large_block_throughput.append(tier_data[pattern]['throughput_MBps'])
                    
                    if small_block_throughput and large_block_throughput:
                        avg_small = sum(small_block_throughput) / len(small_block_throughput)
                        avg_large = sum(large_block_throughput) / len(large_block_throughput)
                        
                        if avg_large > avg_small * 3:
                            recommendations.append(f"📈 {tier_name}: Use larger block sizes for better throughput (Large blocks: {avg_large:.1f} MB/s vs Small blocks: {avg_small:.1f} MB/s)")
            
            # Add general recommendations
            if not recommendations:
                recommendations.append("ℹ️ No specific recommendations available - consider running more comprehensive tests")
            else:
                recommendations.append("💡 General: Monitor performance regularly and consider workload-specific optimization")
                recommendations.append("🔍 Tip: Use larger block sizes for sequential workloads and smaller blocks for random access patterns")
        
        except Exception as e:
            logger.warning(f"Error generating recommendations: {str(e)}")
            recommendations.append("⚠️ Unable to generate specific recommendations due to data analysis error")
        
        return recommendations

    def _generate_executive_summary_data(self, benchmark_result: BenchmarkResult) -> Dict[str, Any]:
        """Generate executive summary data for JSON/CSV reports"""
        try:
            summary = {}
            
            # Basic information
            summary["benchmark_id"] = getattr(benchmark_result, 'benchmark_id', benchmark_result.run_id)
            summary["duration_seconds"] = getattr(benchmark_result, 'duration_seconds', 0)
            summary["storage_tiers"] = len(getattr(benchmark_result, 'tiers', []))
            
            # Tier summaries
            tier_summaries = {}
            if hasattr(benchmark_result, 'tier_results') and benchmark_result.tier_results:
                for tier_name, tier_data in benchmark_result.tier_results.items():
                    if 'summary' in tier_data:
                        summary_data = tier_data['summary']
                        tier_summaries[tier_name] = {
                            "avg_throughput_MBps": round(summary_data.get('avg_throughput_MBps', 0), 2),
                            "avg_iops": round(summary_data.get('avg_iops', 0), 2),
                            "avg_latency_ms": round(summary_data.get('avg_latency_ms', 0), 3),
                            "max_throughput_MBps": round(summary_data.get('max_throughput_MBps', 0), 2),
                            "max_iops": round(summary_data.get('max_iops', 0), 2),
                            "min_latency_ms": round(summary_data.get('min_latency_ms', 0), 3)
                        }
            
            summary["tier_summaries"] = tier_summaries
            
            # Detailed results
            detailed_results = []
            if hasattr(benchmark_result, 'tier_results') and benchmark_result.tier_results:
                for tier_name, tier_data in benchmark_result.tier_results.items():
                    if 'raw_data' in tier_data:
                        raw_data = tier_data['raw_data']
                        for pattern, pattern_data in raw_data.items():
                            if isinstance(pattern_data, dict):
                                # Extract block size from pattern name
                                block_size = 'unknown'
                                # Check for 64k first to avoid false matches with 4k
                                if '64k' in pattern.lower():
                                    block_size = '64k'
                                elif '4k' in pattern.lower():
                                    block_size = '4k'
                                elif '1m' in pattern.lower():
                                    block_size = '1m'
                                elif '1mb' in pattern.lower():
                                    block_size = '1m'
                                elif 'block_size' in pattern_data:
                                    block_size = pattern_data['block_size']
                                
                                # Extract read and write IOPS from nested data
                                read_iops = 0
                                write_iops = 0
                                if 'read' in pattern_data and isinstance(pattern_data['read'], dict):
                                    read_iops = pattern_data['read'].get('iops', 0)
                                if 'write' in pattern_data and isinstance(pattern_data['write'], dict):
                                    write_iops = pattern_data['write'].get('iops', 0)
                                
                                detailed_results.append({
                                    "tier": tier_name,
                                    "pattern": pattern,
                                    "block_size": block_size,
                                    "read_iops": read_iops,
                                    "write_iops": write_iops,
                                    "total_iops": pattern_data.get('iops', 0),
                                    "throughput_MBps": pattern_data.get('throughput_MBps', 0),
                                    "latency_ms": pattern_data.get('latency_ms', 0)
                                })
            
            summary["detailed_results"] = detailed_results
            
            return summary
            
        except Exception as e:
            logger.warning(f"Error generating executive summary data: {str(e)}")
            return {"error": "Unable to generate executive summary data"}

    def _generate_performance_recommendations_data(self, benchmark_result: BenchmarkResult) -> List[Dict[str, Any]]:
        """Generate performance recommendations data for JSON/CSV reports"""
        try:
            recommendations_text = self._generate_performance_recommendations(benchmark_result)
            recommendations_data = []
            
            for rec_text in recommendations_text:
                # Parse the recommendation text to extract structured data
                recommendation = {
                    "text": rec_text,
                    "type": "general"
                }
                
                if "✅" in rec_text:
                    recommendation["type"] = "positive"
                    recommendation["severity"] = "info"
                elif "⚠️" in rec_text:
                    recommendation["type"] = "warning"
                    recommendation["severity"] = "warning"
                elif "❌" in rec_text:
                    recommendation["type"] = "issue"
                    recommendation["severity"] = "error"
                elif "💡" in rec_text:
                    recommendation["type"] = "tip"
                    recommendation["severity"] = "info"
                elif "📊" in rec_text:
                    recommendation["type"] = "analysis"
                    recommendation["severity"] = "info"
                elif "📈" in rec_text:
                    recommendation["type"] = "optimization"
                    recommendation["severity"] = "info"
                
                # Extract tier name if present
                if ":" in rec_text and rec_text.count(":") >= 1:
                    parts = rec_text.split(":", 2)
                    if len(parts) >= 2:
                        tier_part = parts[1].strip()
                        if tier_part and not tier_part.startswith(("Excellent", "High", "Good", "Moderate", "Poor")):
                            recommendation["tier"] = tier_part
                
                recommendations_data.append(recommendation)
            
            return recommendations_data
            
        except Exception as e:
            logger.warning(f"Error generating performance recommendations data: {str(e)}")
            return [{"text": "Unable to generate performance recommendations", "type": "error", "severity": "error"}]
