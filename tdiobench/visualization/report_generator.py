#!/usr/bin/env python3
"""
Report Generator (Tiered Storage I/O Benchmark)

This module provides functionality for generating benchmark reports in various formats,
including HTML, JSON, CSV, and Markdown.

Author: Jack Ogaja
Date: 2025-06-26
"""

import os
import json
import logging
import csv
import datetime
from typing import Dict, List, Any, Optional, Union

from tdiobench.core.benchmark_config import BenchmarkConfig
from tdiobench.core.benchmark_data import BenchmarkResult
from tdiobench.core.benchmark_exceptions import BenchmarkReportError

logger = logging.getLogger("tdiobench.visualization.report")

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
            "email": self.generate_email_report
        }
    
    def generate_reports(
        self,
        benchmark_result: BenchmarkResult,
        output_dir: str = "results",
        formats: Optional[List[str]] = None,
        report_title: Optional[str] = None,
        **kwargs
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
                        output_paths = formatter(benchmark_result, output_dir, report_title, **kwargs)
                        report_files[format] = output_paths
                    else:
                        # Generate single file
                        output_path = formatter(benchmark_result, output_path, report_title, **kwargs)
                        report_files[format] = output_path
                    
                    logger.info(f"Generated {format} report: {output_path}")
                    
                except Exception as e:
                    logger.error(f"Error generating {format} report: {str(e)}")
                    raise BenchmarkReportError(f"Failed to generate {format} report: {str(e)}")
            else:
                logger.warning(f"Unsupported report format: {format}")
        
        return report_files
    
    def generate_html_report(
        self,
        benchmark_result: BenchmarkResult,
        output_path: str,
        report_title: str,
        **kwargs
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
                from jinja2 import Environment, FileSystemLoader
                import datetime
                
                # Check if template directory exists
                template_path = os.path.join(self.template_dir, "html")
                if not os.path.exists(template_path):
                    # Use default template
                    html = self._generate_default_html_report(benchmark_result, report_title)
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
                        user="JackOgaja"
                    )
            except ImportError:
                # Jinja2 not available, use default template
                logger.warning("Jinja2 not available, using default HTML template")
                html = self._generate_default_html_report(benchmark_result, report_title)
            
            # Write HTML report
            with open(output_path, 'w') as f:
                f.write(html)
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating HTML report: {str(e)}")
            raise BenchmarkReportError(f"Failed to generate HTML report: {str(e)}")
    
    def generate_json_report(
        self,
        benchmark_result: BenchmarkResult,
        output_path: str,
        report_title: str,
        **kwargs
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
                "user": "JackOgaja"
            }
            
            # Write JSON report
            with open(output_path, 'w') as f:
                json.dump(result_dict, f, indent=2)
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating JSON report: {str(e)}")
            raise BenchmarkReportError(f"Failed to generate JSON report: {str(e)}")
    
    def generate_csv_report(
        self,
        benchmark_result: BenchmarkResult,
        output_dir: str,
        report_title: str,
        **kwargs
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
            # Create CSV files for different aspects of the report
            csv_files = {}
            
            # Summary CSV
            summary_path = os.path.join(output_dir, "summary.csv")
            csv_files["summary"] = self._generate_summary_csv(benchmark_result, summary_path)
            
            # Tier CSV
            tier_path = os.path.join(output_dir, "tiers.csv")
            csv_files["tiers"] = self._generate_tiers_csv(benchmark_result, tier_path)
            
            # Tests CSV
            tests_path = os.path.join(output_dir, "tests.csv")
            csv_files["tests"] = self._generate_tests_csv(benchmark_result, tests_path)
            
            # Time series CSV (if available)
            if benchmark_result.to_dict().get("tiers"):
                for tier, tier_data in benchmark_result.to_dict()["tiers"].items():
                    if "time_series" in tier_data:
                        tier_name = os.path.basename(tier)
                        ts_path = os.path.join(output_dir, f"time_series_{tier_name}.csv")
                        csv_files[f"time_series_{tier_name}"] = self._generate_time_series_csv(tier_data["time_series"], ts_path)
            
            return csv_files
            
        except Exception as e:
            logger.error(f"Error generating CSV report: {str(e)}")
            raise BenchmarkReportError(f"Failed to generate CSV report: {str(e)}")
    
    def generate_markdown_report(
        self,
        benchmark_result: BenchmarkResult,
        output_path: str,
        report_title: str,
        **kwargs
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
            md_lines.append(f"**Generated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            md_lines.append(f"**Benchmark ID:** {benchmark_result.benchmark_id}")
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
                            throughput_ratio = metrics.get("avg_throughput_MBps", {}).get("ratio", 0)
                            iops_ratio = metrics.get("avg_iops", {}).get("ratio", 0)
                            latency_ratio = metrics.get("avg_latency_ms", {}).get("ratio", 0)
                            
                            md_lines.append(f"| {tier_name} | {throughput_ratio:.2f}x | {iops_ratio:.2f}x | {latency_ratio:.2f}x |")
                    
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
                        
                        md_lines.append(f"| {test_name} | {throughput:.2f} | {iops:.2f} | {latency:.2f} |")
                    
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
                        md_lines.append(f"{analysis_type.capitalize()} analysis results available in JSON report.")
                    
                    md_lines.append("")
            
            # Write markdown report
            with open(output_path, 'w') as f:
                f.write("\n".join(md_lines))
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating Markdown report: {str(e)}")
            raise BenchmarkReportError(f"Failed to generate Markdown report: {str(e)}")
    
    def generate_email_report(
        self,
        benchmark_result: BenchmarkResult,
        output_path: str,
        report_title: str,
        **kwargs
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
            with open(md_path, 'r') as f:
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
                with open(output_path, 'w') as f:
                    f.write(email_html)
                
            except ImportError:
                # Markdown not available, use plain text
                logger.warning("Markdown module not available, using plain text email")
                
                # Write email report
                with open(output_path, 'w') as f:
                    f.write(f"Subject: {report_title}\n\n")
                    f.write(md_content)
                    f.write("\n\nThis report was automatically generated by the Enhanced Tiered Storage I/O Benchmark Suite.")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating email report: {str(e)}")
            raise BenchmarkReportError(f"Failed to generate email report: {str(e)}")
    
    def _generate_default_html_report(
        self,
        benchmark_result: BenchmarkResult,
        report_title: str
    ) -> str:
        """
        Generate default HTML report without templates.
        
        Args:
            benchmark_result: Benchmark result
            report_title: Report title
            
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
        html.append("<style>")
        html.append("body { font-family: Arial, sans-serif; line-height: 1.5; margin: 20px; }")
        html.append("h1, h2, h3 { color: #333; }")
        html.append("table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }")
        html.append("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }")
        html.append("th { background-color: #f2f2f2; }")
        html.append(".summary { background-color: #f9f9f9; padding: 15px; border-radius: 5px; }")
        html.append("</style>")
        html.append("</head>")
        html.append("<body>")
        
        # Add title
        html.append(f"<h1>{report_title}</h1>")
        
        # Add generation metadata
        html.append("<p>")
        html.append(f"<strong>Generated:</strong> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>")
        html.append(f"<strong>Benchmark ID:</strong> {benchmark_result.benchmark_id}<br>")
        html.append("</p>")
        
        # Add summary
        html.append("<div class='summary'>")
        html.append("<h2>Summary</h2>")
        
        summary = benchmark_result.get_summary()
        
        html.append("<p>")
        html.append(f"<strong>Benchmark Duration:</strong> {benchmark_result.duration} seconds<br>")
        html.append(f"<strong>Storage Tiers:</strong> {len(benchmark_result.tiers)}<br>")
        
        if "analysis_types" in summary:
            html.append(f"<strong>Analysis Types:</strong> {', '.join(summary['analysis_types'])}<br>")
        
        html.append("</p>")
        html.append("</div>")
        
        # Add tier summaries
        html.append("<h2>Tier Summaries</h2>")
        
        html.append("<table>")
        html.append("<tr><th>Tier</th><th>Avg Throughput (MB/s)</th><th>Avg IOPS</th><th>Avg Latency (ms)</th></tr>")
        
        for tier, tier_summary in summary.get("tier_summaries", {}).items():
            tier_name = os.path.basename(tier)
            throughput = tier_summary.get("avg_throughput_MBps", 0)
            iops = tier_summary.get("avg_iops", 0)
            latency = tier_summary.get("avg_latency_ms", 0)
            
            html.append(f"<tr><td>{tier_name}</td><td>{throughput:.2f}</td><td>{iops:.2f}</td><td>{latency:.2f}</td></tr>")
        
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
                html.append("<tr><th>Tier</th><th>Throughput Ratio</th><th>IOPS Ratio</th><th>Latency Ratio</th></tr>")
                
                for tier, tier_comparison in comparison.get("tier_comparisons", {}).items():
                    tier_name = os.path.basename(tier)
                    
                    if "metrics" in tier_comparison:
                        metrics = tier_comparison["metrics"]
                        throughput_ratio = metrics.get("avg_throughput_MBps", {}).get("ratio", 0)
                        iops_ratio = metrics.get("avg_iops", {}).get("ratio", 0)
                        latency_ratio = metrics.get("avg_latency_ms", {}).get("ratio", 0)
                        
                        html.append(f"<tr><td>{tier_name}</td><td>{throughput_ratio:.2f}x</td><td>{iops_ratio:.2f}x</td><td>{latency_ratio:.2f}x</td></tr>")
                
                html.append("</table>")
        
        # Add detailed results for each tier
        html.append("<h2>Detailed Results</h2>")
        
        for tier in benchmark_result.tiers:
            tier_name = os.path.basename(tier)
            tier_result = benchmark_result.get_tier_result(tier)
            
            html.append(f"<h3>{tier_name}</h3>")
            
            if tier_result and "tests" in tier_result:
                html.append("<table>")
                html.append("<tr><th>Test</th><th>Throughput (MB/s)</th><th>IOPS</th><th>Latency (ms)</th></tr>")
                
                for test_name, test_data in tier_result["tests"].items():
                    throughput = test_data.get("throughput_MBps", 0)
                    iops = test_data.get("iops", 0)
                    latency = test_data.get("latency_ms", 0)
                    
                    html.append(f"<tr><td>{test_name}</td><td>{throughput:.2f}</td><td>{iops:.2f}</td><td>{latency:.2f}</td></tr>")
                
                html.append("</table>")
        
        # Add footer
        html.append("<p><em>This report was automatically generated by the Enhanced Tiered Storage I/O Benchmark Suite.</em></p>")
        
        html.append("</body>")
        html.append("</html>")
        
        return "\n".join(html)
    
    def _generate_summary_csv(
        self,
        benchmark_result: BenchmarkResult,
        output_path: str
    ) -> str:
        """
        Generate summary CSV.
        
        Args:
            benchmark_result: Benchmark result
            output_path: Output file path
            
        Returns:
            Path to generated CSV
        """
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow(["Benchmark ID", "Timestamp", "Duration", "Tier Count"])
            
            # Write data
            writer.writerow([
                benchmark_result.benchmark_id,
                benchmark_result.timestamp,
                benchmark_result.duration,
                len(benchmark_result.tiers)
            ])
        
        return output_path
    
    def _generate_tiers_csv(
        self,
        benchmark_result: BenchmarkResult,
        output_path: str
    ) -> str:
        """
        Generate tiers CSV.
        
        Args:
            benchmark_result: Benchmark result
            output_path: Output file path
            
        Returns:
            Path to generated CSV
        """
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow(["Tier", "Avg Throughput (MB/s)", "Avg IOPS", "Avg Latency (ms)"])
            
            # Write data
            for tier in benchmark_result.tiers:
                tier_result = benchmark_result.get_tier_result(tier)
                
                if tier_result and "summary" in tier_result:
                    summary = tier_result["summary"]
                    
                    writer.writerow([
                        tier,
                        summary.get("avg_throughput_MBps", 0),
                        summary.get("avg_iops", 0),
                        summary.get("avg_latency_ms", 0)
                    ])
        
        return output_path
    
    def _generate_tests_csv(
        self,
        benchmark_result: BenchmarkResult,
        output_path: str
    ) -> str:
        """
        Generate tests CSV.
        
        Args:
            benchmark_result: Benchmark result
            output_path: Output file path
            
        Returns:
            Path to generated CSV
        """
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow(["Tier", "Test", "Throughput (MB/s)", "IOPS", "Latency (ms)"])
            
            # Write data
            for tier in benchmark_result.tiers:
                tier_result = benchmark_result.get_tier_result(tier)
                
                if tier_result and "tests" in tier_result:
                    for test_name, test_data in tier_result["tests"].items():
                        writer.writerow([
                            tier,
                            test_name,
                            test_data.get("throughput_MBps", 0),
                            test_data.get("iops", 0),
                            test_data.get("latency_ms", 0)
                        ])
        
        return output_path
    
    def _generate_time_series_csv(
        self,
        time_series_data: Dict[str, Any],
        output_path: str
    ) -> str:
        """
        Generate time series CSV.
        
        Args:
            time_series_data: Time series data
            output_path: Output file path
            
        Returns:
            Path to generated CSV
        """
        with open(output_path, 'w', newline='') as f:
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
    
    def _generate_charts(
        self,
        benchmark_result: BenchmarkResult
    ) -> Dict[str, Any]:
        """
        Generate chart configurations.
        
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
                "data": {
                    "labels": [],
                    "datasets": []
                },
                "options": {
                    "title": {
                        "display": True,
                        "text": "Tier Performance Comparison"
                    },
                    "scales": {
                        "yAxes": [{
                            "ticks": {
                                "beginAtZero": True
                            }
                        }]
                    }
                }
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
            
            charts["tier_comparison"]["data"]["labels"] = tier_labels
            
            charts["tier_comparison"]["data"]["datasets"] = [
                {
                    "label": "Throughput (MB/s)",
                    "data": throughput_data,
                    "backgroundColor": "rgba(54, 162, 235, 0.5)"
                },
                {
                    "label": "IOPS",
                    "data": iops_data,
                    "backgroundColor": "rgba(255, 99, 132, 0.5)"
                },
                {
                    "label": "Latency (ms)",
                    "data": latency_data,
                    "backgroundColor": "rgba(75, 192, 192, 0.5)"
                }
            ]
            
            # Time series charts for each tier
            for tier in benchmark_result.tiers:
                tier_name = os.path.basename(tier)
                tier_result = benchmark_result.get_tier_result(tier)
                
                if tier_result and "time_series" in tier_result:
                    time_series = tier_result["time_series"]
                    
                    # Create time series charts
                    for metric in ["throughput_MBps", "iops", "latency_ms"]:
                        if metric in time_series["data"]:
                            values = time_series["data"][metric]
                            timestamps = time_series["timestamps"]
                            
                            # Convert timestamps to labels
                            labels = []
                            for ts in timestamps:
                                dt = datetime.datetime.fromtimestamp(ts)
                                labels.append(dt.strftime("%H:%M:%S"))
                            
                            charts[f"{tier_name}_{metric}"] = {
                                "type": "line",
                                "data": {
                                    "labels": labels,
                                    "datasets": [{
                                        "label": f"{tier_name} - {metric}",
                                        "data": values,
                                        "borderColor": "rgba(54, 162, 235, 1)",
                                        "backgroundColor": "rgba(54, 162, 235, 0.1)"
                                    }]
                                },
                                "options": {
                                    "title": {
                                        "display": True,
                                        "text": f"{tier_name} - {metric} over time"
                                    },
                                    "scales": {
                                        "yAxes": [{
                                            "ticks": {
                                                "beginAtZero": True
                                            }
                                        }]
                                    }
                                }
                            }
        except Exception as e:
            logger.error(f"Error generating charts: {str(e)}")
        
        return charts
    
    def _add_network_analysis_markdown(
        self,
        md_lines: List[str],
        network_results: Dict[str, Any]
    ) -> None:
        """
        Add network analysis results to markdown report.
        
        Args:
            md_lines: List of markdown lines
            network_results: Network analysis results
        """
        # Add overall assessment
        if "assessment" in network_results:
            assessment = network_results["assessment"]
            
            md_lines.append("### Network Impact Assessment")
            md_lines.append("")
            md_lines.append(f"**Impact Level:** {assessment.get('impact_level', 'unknown')}")
            md_lines.append("")
            md_lines.append(assessment.get('description', 'No description available.'))
            md_lines.append("")
        
        # Add bottlenecks
        if "bottlenecks" in network_results and network_results["bottlenecks"]:
            md_lines.append("### Network Bottlenecks")
            md_lines.append("")
            
            for bottleneck in network_results["bottlenecks"]:
                md_lines.append(f"- {bottleneck.get('description', 'Unknown bottleneck')}")
            
            md_lines.append("")
        
        # Add recommendations
        if "recommendations" in network_results and network_results["recommendations"]:
            md_lines.append("### Network Optimization Recommendations")
            md_lines.append("")
            
            for recommendation in network_results["recommendations"]:
                md_lines.append(f"- {recommendation}")
            
            md_lines.append("")
    
    def _add_anomaly_analysis_markdown(
        self,
        md_lines: List[str],
        anomaly_results: Dict[str, Any]
    ) -> None:
        """
        Add anomaly analysis results to markdown report.
        
        Args:
            md_lines: List of markdown lines
            anomaly_results: Anomaly analysis results
        """
        # Add overall assessment
        if "assessment" in anomaly_results:
            assessment = anomaly_results["assessment"]
            
            md_lines.append("### Anomaly Detection Assessment")
            md_lines.append("")
            md_lines.append(f"**Severity:** {assessment.get('severity', 'unknown')}")
            md_lines.append("")
            md_lines.append(assessment.get('description', 'No description available.'))
            md_lines.append("")
        
        # Add anomaly summary
        md_lines.append("### Anomaly Summary")
        md_lines.append("")
        
        if anomaly_results.get("anomalies_detected", False):
            md_lines.append(f"**Total Anomalies:** {anomaly_results.get('total_anomalies', 0)}")
            md_lines.append("")
            
            # Add tier anomalies
            for tier, tier_results in anomaly_results.get("tiers", {}).items():
                if tier_results.get("anomalies_detected", False):
                    tier_name = os.path.basename(tier)
                    md_lines.append(f"**{tier_name}:** {tier_results.get('total_anomalies', 0)} anomalies")
                    
                    # Add metric anomalies
                    for metric, metric_results in tier_results.get("metric_anomalies", {}).items():
                        if metric_results.get("anomalies_detected", False):
                            md_lines.append(f"- {metric}: {metric_results.get('anomaly_count', 0)} anomalies")
            
            md_lines.append("")
            
            # Add recommendations
            if "assessment" in anomaly_results and "recommendations" in anomaly_results["assessment"]:
                md_lines.append("### Recommendations")
                md_lines.append("")
                
                for recommendation in anomaly_results["assessment"]["recommendations"]:
                    md_lines.append(f"- {recommendation}")
                
                md_lines.append("")
        else:
            md_lines.append("No anomalies detected in the benchmark data.")
            md_lines.append("")
