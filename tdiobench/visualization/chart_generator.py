#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chart generator module for storage benchmark visualization (Tiered Storage I/O Benchmark).

This module provides classes to generate various charts and visualizations
from benchmark data, supporting different chart types, customization options,
and export formats.

Author: Jack Ogaja
Date: 2025-06-26
"""

import logging
import os
import json
from typing import Dict, List, Tuple, Optional, Union, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
from pathlib import Path

from tdiobench.core.benchmark_data import BenchmarkData
from tdiobench.core.benchmark_exceptions import VisualizationError

logger = logging.getLogger(__name__)


class ChartGenerator:
    """
    Chart generation for benchmark data visualization.
    
    This class provides methods to create various types of charts and visualizations
    from benchmark data, including time series plots, bar charts, heatmaps, and
    comparative visualizations.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the chart generator with configuration.
        
        Args:
            config: Configuration dictionary with chart parameters
        """
        self.config = config
        self.chart_config = self.config.get("visualization", {}).get("charts", {})
        self.enabled = self.chart_config.get("enabled", True)
        self.chart_types = self.chart_config.get("types", ["bar", "line", "scatter", "heatmap"])
        self.output_dir = self.chart_config.get("output_dir", "./charts")
        self.format = self.chart_config.get("format", "html")
        self.width = self.chart_config.get("width", 900)
        self.height = self.chart_config.get("height", 500)
        self.interactive = self.chart_config.get("interactive", True)
        self.theme = self.chart_config.get("theme", "default")
        
        # Configure matplotlib and seaborn styles
        if not self.interactive:
            self._setup_matplotlib_style()
        
        # Create output directory if it doesn't exist
        if self.enabled and self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            
        logger.debug(f"Initialized ChartGenerator with format={self.format}, interactive={self.interactive}")
    
    def _setup_matplotlib_style(self) -> None:
        """Configure matplotlib style for static charts."""
        if self.theme == "dark":
            plt.style.use("dark_background")
        else:
            plt.style.use("seaborn-v0_8-whitegrid")
            
        # Set default figure size
        plt.rcParams["figure.figsize"] = (self.width/100, self.height/100)
        plt.rcParams["figure.dpi"] = 100
        
        # Improve font appearance
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
        plt.rcParams["axes.labelsize"] = 12
        plt.rcParams["axes.titlesize"] = 14
        plt.rcParams["xtick.labelsize"] = 10
        plt.rcParams["ytick.labelsize"] = 10
    
    def generate_time_series_chart(self, benchmark_data: BenchmarkData,
                                 metrics: List[str],
                                 title: Optional[str] = None,
                                 subtitle: Optional[str] = None,
                                 x_label: Optional[str] = None,
                                 y_label: Optional[str] = None,
                                 start_time: Optional[str] = None,
                                 end_time: Optional[str] = None,
                                 output_path: Optional[str] = None) -> Optional[Union[plt.Figure, Dict]]:
        """
        Generate a time series chart for benchmark metrics.
        
        Args:
            benchmark_data: BenchmarkData instance
            metrics: List of metrics to plot
            title: Chart title
            subtitle: Chart subtitle
            x_label: X-axis label
            y_label: Y-axis label
            start_time: Start time for filtering
            end_time: End time for filtering
            output_path: Path to save the chart
            
        Returns:
            Matplotlib figure or Plotly figure dict if output_path is None, otherwise None
        """
        if not self.enabled:
            logger.info("Chart generation is disabled in config")
            return None
        
        try:
            # Get time series data
            if start_time or end_time:
                filtered_data = benchmark_data.filter_time_series(start_time, end_time)
                df = filtered_data.get_time_series(reset_index=True)
            else:
                df = benchmark_data.get_time_series(reset_index=True)
            
            if df.empty:
                logger.warning("No time series data available for chart generation")
                return None
            
            # Filter to only include specified metrics that exist
            available_metrics = [m for m in metrics if m in df.columns]
            if not available_metrics:
                logger.warning(f"None of the specified metrics {metrics} found in data")
                return None
            
            # Default title if not provided
            if title is None:
                title = "Benchmark Time Series"
            
            # Generate chart based on interactive setting
            if self.interactive:
                return self._generate_plotly_time_series(
                    df, available_metrics, title, subtitle, x_label, y_label, output_path
                )
            else:
                return self._generate_matplotlib_time_series(
                    df, available_metrics, title, subtitle, x_label, y_label, output_path
                )
                
        except Exception as e:
            logger.exception(f"Error generating time series chart: {str(e)}")
            raise VisualizationError(f"Failed to generate time series chart: {str(e)}")
    
    def _generate_plotly_time_series(self, df: pd.DataFrame,
                                    metrics: List[str],
                                    title: str,
                                    subtitle: Optional[str],
                                    x_label: Optional[str],
                                    y_label: Optional[str],
                                    output_path: Optional[str]) -> Optional[Dict]:
        """
        Generate an interactive Plotly time series chart.
        
        Args:
            df: DataFrame with time series data
            metrics: List of metrics to plot
            title: Chart title
            subtitle: Chart subtitle
            x_label: X-axis label
            y_label: Y-axis label
            output_path: Path to save the chart
            
        Returns:
            Plotly figure dict if output_path is None, otherwise None
        """
        # Create figure
        fig = go.Figure()
        
        # Add traces for each metric
        for metric in metrics:
            fig.add_trace(go.Scatter(
                x=df["timestamp"] if "timestamp" in df.columns else df.index,
                y=df[metric],
                mode='lines',
                name=metric
            ))
        
        # Update layout
        layout_title = title
        if subtitle:
            layout_title = f"{title}<br><sup>{subtitle}</sup>"
            
        fig.update_layout(
            title=layout_title,
            xaxis_title=x_label or "Time",
            yaxis_title=y_label or "Value",
            legend_title="Metrics",
            width=self.width,
            height=self.height,
            hovermode="x unified",
            template="plotly_white" if self.theme == "default" else "plotly_dark"
        )
        
        # Add benchmark metadata as annotations
        if hasattr(df, 'metadata'):
            metadata_text = "<br>".join([f"{k}: {v}" for k, v in df.metadata.items()
                                      if k not in ['id', 'created_at', 'created_by']])
            fig.add_annotation(
                x=0,
                y=1.05,
                xref="paper",
                yref="paper",
                text=metadata_text,
                showarrow=False,
                align="left",
                bgcolor="rgba(255, 255, 255, 0.7)",
                bordercolor="lightgrey",
                borderwidth=1
            )
        
        # Save if output path provided
        if output_path:
            if not output_path.endswith(('.html', '.json', '.png', '.jpg', '.svg', '.pdf')):
                output_path += f".{self.format}"
                
            pio.write_html(fig, output_path)
            logger.info(f"Saved time series chart to {output_path}")
            return None
            
        return fig
    
    def _generate_matplotlib_time_series(self, df: pd.DataFrame,
                                        metrics: List[str],
                                        title: str,
                                        subtitle: Optional[str],
                                        x_label: Optional[str],
                                        y_label: Optional[str],
                                        output_path: Optional[str]) -> Optional[plt.Figure]:
        """
        Generate a static Matplotlib time series chart.
        
        Args:
            df: DataFrame with time series data
            metrics: List of metrics to plot
            title: Chart title
            subtitle: Chart subtitle
            x_label: X-axis label
            y_label: Y-axis label
            output_path: Path to save the chart
            
        Returns:
            Matplotlib figure if output_path is None, otherwise None
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(self.width/100, self.height/100))
        
        # Plot each metric
        for metric in metrics:
            ax.plot(
                df["timestamp"] if "timestamp" in df.columns else df.index,
                df[metric],
                label=metric
            )
        
        # Set labels and title
        ax.set_xlabel(x_label or "Time")
        ax.set_ylabel(y_label or "Value")
        ax.set_title(title, fontsize=14)
        
        if subtitle:
            ax.text(0.5, 0.97, subtitle, transform=ax.transAxes,
                   fontsize=11, ha='center', va='top')
        
        # Add legend
        ax.legend(title="Metrics")
        
        # Format timestamp labels
        if "timestamp" in df.columns:
            fig.autofmt_xdate()
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Tight layout
        fig.tight_layout()
        
        # Save if output path provided
        if output_path:
            if not output_path.endswith(('.png', '.jpg', '.svg', '.pdf')):
                output_path += ".png"
                
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            logger.info(f"Saved time series chart to {output_path}")
            return None
            
        return fig
    
    def generate_bar_chart(self, benchmark_data: BenchmarkData,
                         metric: str,
                         groupby: Optional[str] = None,
                         operation: str = 'mean',
                         title: Optional[str] = None,
                         x_label: Optional[str] = None,
                         y_label: Optional[str] = None,
                         output_path: Optional[str] = None) -> Optional[Union[plt.Figure, Dict]]:
        """
        Generate a bar chart for benchmark metrics.
        
        Args:
            benchmark_data: BenchmarkData instance
            metric: Metric to visualize
            groupby: Column to group by
            operation: Aggregation operation ('mean', 'max', 'min', 'sum')
            title: Chart title
            x_label: X-axis label
            y_label: Y-axis label
            output_path: Path to save the chart
            
        Returns:
            Matplotlib figure or Plotly figure dict if output_path is None, otherwise None
        """
        if not self.enabled:
            logger.info("Chart generation is disabled in config")
            return None
            
        if "bar" not in self.chart_types:
            logger.info("Bar chart type is not enabled in config")
            return None
        
        try:
            # Get time series data
            df = benchmark_data.get_time_series(reset_index=True)
            
            if df.empty:
                logger.warning("No time series data available for bar chart generation")
                return None
            
            # Check if metric exists
            if metric not in df.columns:
                logger.warning(f"Metric {metric} not found in data")
                return None
            
            # Apply grouping if specified
            if groupby and groupby in df.columns:
                # Group and aggregate
                if operation == 'mean':
                    grouped_df = df.groupby(groupby)[metric].mean().reset_index()
                elif operation == 'max':
                    grouped_df = df.groupby(groupby)[metric].max().reset_index()
                elif operation == 'min':
                    grouped_df = df.groupby(groupby)[metric].min().reset_index()
                elif operation == 'sum':
                    grouped_df = df.groupby(groupby)[metric].sum().reset_index()
                else:
                    logger.warning(f"Unknown operation {operation}, using mean")
                    grouped_df = df.groupby(groupby)[metric].mean().reset_index()
                
                x_values = grouped_df[groupby]
                y_values = grouped_df[metric]
                
                # Default labels
                default_x_label = groupby
                
            else:
                # If no groupby, just use engine results
                engine_results = benchmark_data.get_engine_results()
                if not engine_results:
                    logger.warning("No engine results available for bar chart")
                    return None
                
                # Extract values for each engine and operation
                x_values = []
                y_values = []
                
                for engine, ops in engine_results.items():
                    for op, results in ops.items():
                        if isinstance(results, dict) and metric in results:
                            x_values.append(f"{engine}-{op}")
                            y_values.append(results[metric])
                
                if not x_values:
                    logger.warning(f"Metric {metric} not found in engine results")
                    return None
                
                # Default labels
                default_x_label = "Engine-Operation"
            
            # Default title and labels
            if title is None:
                title = f"{metric.title()} by {default_x_label}" if groupby else f"{metric.title()} by Engine"
                
            if x_label is None:
                x_label = default_x_label
                
            if y_label is None:
                y_label = metric.replace('_', ' ').title()
            
            # Generate chart based on interactive setting
            if self.interactive:
                return self._generate_plotly_bar_chart(
                    x_values, y_values, title, x_label, y_label, output_path
                )
            else:
                return self._generate_matplotlib_bar_chart(
                    x_values, y_values, title, x_label, y_label, output_path
                )
                
        except Exception as e:
            logger.exception(f"Error generating bar chart: {str(e)}")
            raise VisualizationError(f"Failed to generate bar chart: {str(e)}")
    
    def _generate_plotly_bar_chart(self, x_values: Union[List, pd.Series],
                                  y_values: Union[List, pd.Series],
                                  title: str,
                                  x_label: str,
                                  y_label: str,
                                  output_path: Optional[str]) -> Optional[Dict]:
        """
        Generate an interactive Plotly bar chart.
        
        Args:
            x_values: X-axis values
            y_values: Y-axis values
            title: Chart title
            x_label: X-axis label
            y_label: Y-axis label
            output_path: Path to save the chart
            
        Returns:
            Plotly figure dict if output_path is None, otherwise None
        """
        # Create figure
        fig = go.Figure(data=[
            go.Bar(
                x=x_values,
                y=y_values,
                marker_color='royalblue'
            )
        ])
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title=x_label,
            yaxis_title=y_label,
            width=self.width,
            height=self.height,
            template="plotly_white" if self.theme == "default" else "plotly_dark"
        )
        
        # Add data labels
        fig.update_traces(
            texttemplate='%{y:.2f}',
            textposition='outside'
        )
        
        # Save if output path provided
        if output_path:
            if not output_path.endswith(('.html', '.json', '.png', '.jpg', '.svg', '.pdf')):
                output_path += f".{self.format}"
                
            pio.write_html(fig, output_path)
            logger.info(f"Saved bar chart to {output_path}")
            return None
            
        return fig
    
    def _generate_matplotlib_bar_chart(self, x_values: Union[List, pd.Series],
                                      y_values: Union[List, pd.Series],
                                      title: str,
                                      x_label: str,
                                      y_label: str,
                                      output_path: Optional[str]) -> Optional[plt.Figure]:
        """
        Generate a static Matplotlib bar chart.
        
        Args:
            x_values: X-axis values
            y_values: Y-axis values
            title: Chart title
            x_label: X-axis label
            y_label: Y-axis label
            output_path: Path to save the chart
            
        Returns:
            Matplotlib figure if output_path is None, otherwise None
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(self.width/100, self.height/100))
        
        # Create bar chart
        bars = ax.bar(x_values, y_values, color='royalblue')
        
        # Add data labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(y_values),
                   f'{height:.2f}', ha='center', va='bottom', fontsize=9)
        
        # Set labels and title
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title, fontsize=14)
        
        # Format x-axis labels if they're long
        if any(len(str(x)) > 10 for x in x_values):
            plt.xticks(rotation=45, ha='right')
        
        # Add grid
        ax.grid(True, axis='y', alpha=0.3)
        
        # Tight layout
        fig.tight_layout()
        
        # Save if output path provided
        if output_path:
            if not output_path.endswith(('.png', '.jpg', '.svg', '.pdf')):
                output_path += ".png"
                
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            logger.info(f"Saved bar chart to {output_path}")
            return None
            
        return fig
    
    def generate_heatmap(self, benchmark_data: BenchmarkData,
                        x_param: str,
                        y_param: str,
                        value_metric: str,
                        title: Optional[str] = None,
                        output_path: Optional[str] = None) -> Optional[Union[plt.Figure, Dict]]:
        """
        Generate a heatmap for parameter exploration results.
        
        Args:
            benchmark_data: BenchmarkData instance
            x_param: Parameter for x-axis
            y_param: Parameter for y-axis
            value_metric: Metric for cell values
            title: Chart title
            output_path: Path to save the chart
            
        Returns:
            Matplotlib figure or Plotly figure dict if output_path is None, otherwise None
        """
        if not self.enabled:
            logger.info("Chart generation is disabled in config")
            return None
            
        if "heatmap" not in self.chart_types:
            logger.info("Heatmap chart type is not enabled in config")
            return None
        
        try:
            # Try to get parameter exploration results
            param_results = None
            if "parameter_exploration" in benchmark_data.data:
                param_results = benchmark_data.data["parameter_exploration"]
            elif "runs" in benchmark_data.data:
                # Try to extract from runs
                param_results = self._extract_param_results(benchmark_data.data["runs"])
            
            if not param_results:
                logger.warning("No parameter exploration results found for heatmap")
                return None
            
            # Extract unique values for each parameter
            x_values = sorted(set(result.get(x_param) for result in param_results if x_param in result))
            y_values = sorted(set(result.get(y_param) for result in param_results if y_param in result))
            
            if not x_values or not y_values:
                logger.warning(f"Missing parameter values for {x_param} or {y_param}")
                return None
            
            # Create the heatmap matrix
            heat_data = np.zeros((len(y_values), len(x_values)))
            heat_data[:] = np.nan  # Initialize with NaN
            
            # Fill in the matrix
            for result in param_results:
                if x_param in result and y_param in result and value_metric in result:
                    x_idx = x_values.index(result[x_param])
                    y_idx = y_values.index(result[y_param])
                    heat_data[y_idx, x_idx] = result[value_metric]
            
            # Default title
            if title is None:
                title = f"{value_metric.title()} by {x_param} and {y_param}"
            
            # Generate chart based on interactive setting
            if self.interactive:
                return self._generate_plotly_heatmap(
                    heat_data, x_values, y_values, title, x_param, y_param, value_metric, output_path
                )
            else:
                return self._generate_matplotlib_heatmap(
                    heat_data, x_values, y_values, title, x_param, y_param, value_metric, output_path
                )
                
        except Exception as e:
            logger.exception(f"Error generating heatmap: {str(e)}")
            raise VisualizationError(f"Failed to generate heatmap: {str(e)}")
    
    def _extract_param_results(self, runs: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract parameter exploration results from benchmark runs.
        
        Args:
            runs: Dictionary of benchmark runs
            
        Returns:
            List of parameter-result pairs
        """
        results = []
        
        for run_id, run_data in runs.items():
            if "parameters" in run_data and "results" in run_data:
                # Combine parameters and results
                result = {**run_data["parameters"]}
                
                # Extract main metrics from results
                if isinstance(run_data["results"], dict):
                    for metric, value in run_data["results"].items():
                        if isinstance(value, (int, float)):
                            result[metric] = value
                
                results.append(result)
        
        return results
    
    def _generate_plotly_heatmap(self, heat_data: np.ndarray,
                               x_values: List,
                               y_values: List,
                               title: str,
                               x_label: str,
                               y_label: str,
                               value_label: str,
                               output_path: Optional[str]) -> Optional[Dict]:
        """
        Generate an interactive Plotly heatmap.
        
        Args:
            heat_data: 2D array of values
            x_values: X-axis values
            y_values: Y-axis values
            title: Chart title
            x_label: X-axis label
            y_label: Y-axis label
            value_label: Value label for colorbar
            output_path: Path to save the chart
            
        Returns:
            Plotly figure dict if output_path is None, otherwise None
        """
        # Create figure
        fig = go.Figure(data=go.Heatmap(
            z=heat_data,
            x=x_values,
            y=y_values,
            colorscale='Viridis',
            hoverongaps=False,
            colorbar=dict(title=value_label)
        ))
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title=x_label,
            yaxis_title=y_label,
            width=self.width,
            height=self.height,
            template="plotly_white" if self.theme == "default" else "plotly_dark"
        )
        
        # Save if output path provided
        if output_path:
            if not output_path.endswith(('.html', '.json', '.png', '.jpg', '.svg', '.pdf')):
                output_path += f".{self.format}"
                
            pio.write_html(fig, output_path)
            logger.info(f"Saved heatmap to {output_path}")
            return None
            
        return fig
    
    def _generate_matplotlib_heatmap(self, heat_data: np.ndarray,
                                   x_values: List,
                                   y_values: List,
                                   title: str,
                                   x_label: str,
                                   y_label: str,
                                   value_label: str,
                                   output_path: Optional[str]) -> Optional[plt.Figure]:
        """
        Generate a static Matplotlib heatmap.
        
        Args:
            heat_data: 2D array of values
            x_values: X-axis values
            y_values: Y-axis values
            title: Chart title
            x_label: X-axis label
            y_label: Y-axis label
            value_label: Value label for colorbar
            output_path: Path to save the chart
            
        Returns:
            Matplotlib figure if output_path is None, otherwise None
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(self.width/100, self.height/100))
        
        # Create heatmap
        im = ax.imshow(heat_data, cmap='viridis')
        
        # Set labels and title
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title, fontsize=14)
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(x_values)))
        ax.set_yticks(np.arange(len(y_values)))
        ax.set_xticklabels(x_values)
        ax.set_yticklabels(y_values)
        
        # Rotate x-axis labels if needed
        if any(len(str(x)) > 5 for x in x_values):
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel(value_label, rotation=-90, va="bottom")
        
        # Add text annotations with values
        for i in range(len(y_values)):
            for j in range(len(x_values)):
                if not np.isnan(heat_data[i, j]):
                    text = ax.text(j, i, f"{heat_data[i, j]:.2f}",
                                ha="center", va="center", color="w" if heat_data[i, j] > np.nanmax(heat_data)/2 else "black")
        
        # Tight layout
        fig.tight_layout()
        
        # Save if output path provided
        if output_path:
            if not output_path.endswith(('.png', '.jpg', '.svg', '.pdf')):
                output_path += ".png"
                
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            logger.info(f"Saved heatmap to {output_path}")
            return None
            
        return fig
    
    def generate_comparison_chart(self, benchmark_data_list: List[BenchmarkData],
                                labels: List[str],
                                metric: str,
                                chart_type: str = 'bar',
                                title: Optional[str] = None,
                                output_path: Optional[str] = None) -> Optional[Union[plt.Figure, Dict]]:
        """
        Generate a comparison chart for multiple benchmark runs.
        
        Args:
            benchmark_data_list: List of BenchmarkData instances
            labels: Labels for each benchmark
            metric: Metric to compare
            chart_type: Type of chart ('bar' or 'line')
            title: Chart title
            output_path: Path to save the chart
            
        Returns:
            Matplotlib figure or Plotly figure dict if output_path is None, otherwise None
        """
        if not self.enabled:
            logger.info("Chart generation is disabled in config")
            return None
            
        if chart_type not in self.chart_types:
            logger.info(f"{chart_type} chart type is not enabled in config")
            return None
            
        if len(benchmark_data_list) != len(labels):
            logger.warning("Number of benchmark datasets must match number of labels")
            return None
        
        try:
            # Extract metric values from each benchmark
            values = []
            
            for bdata in benchmark_data_list:
                # Try to get from engine results first (they're usually aggregated)
                engine_results = bdata.get_engine_results()
                metric_found = False
                
                for engine, ops in engine_results.items():
                    for op, results in ops.items():
                        if isinstance(results, dict) and metric in results:
                            values.append(results[metric])
                            metric_found = True
                            break
                    if metric_found:
                        break
                
                # If not found in engine results, try time series data
                if not metric_found:
                    metric_value = bdata.get_metric(metric, aggregation='mean')
                    if metric_value is not None:
                        values.append(metric_value)
                    else:
                        logger.warning(f"Metric {metric} not found in benchmark {labels[len(values)]}")
                        values.append(np.nan)
            
            # Default title
            if title is None:
                title = f"Comparison of {metric.replace('_', ' ').title()}"
            
            # Generate chart based on type and interactive setting
            if chart_type == 'bar':
                if self.interactive:
                    return self._generate_plotly_comparison_bar(
                        labels, values, title, metric, output_path
                    )
                else:
                    return self._generate_matplotlib_comparison_bar(
                        labels, values, title, metric, output_path
                    )
            else:  # line chart
                if self.interactive:
                    return self._generate_plotly_comparison_line(
                        labels, values, title, metric, output_path
                    )
                else:
                    return self._generate_matplotlib_comparison_line(
                        labels, values, title, metric, output_path
                    )
                
        except Exception as e:
            logger.exception(f"Error generating comparison chart: {str(e)}")
            raise VisualizationError(f"Failed to generate comparison chart: {str(e)}")
    
    def _generate_plotly_comparison_bar(self, labels: List[str],
                                      values: List[float],
                                      title: str,
                                      metric: str,
                                      output_path: Optional[str]) -> Optional[Dict]:
        """
        Generate an interactive Plotly comparison bar chart.
        
        Args:
            labels: Labels for each bar
            values: Values for each bar
            title: Chart title
            metric: Metric name
            output_path: Path to save the chart
            
        Returns:
            Plotly figure dict if output_path is None, otherwise None
        """
        # Create figure
        fig = go.Figure(data=[
            go.Bar(
                x=labels,
                y=values,
                marker_color='royalblue'
            )
        ])
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Benchmark",
            yaxis_title=metric.replace('_', ' ').title(),
            width=self.width,
            height=self.height,
            template="plotly_white" if self.theme == "default" else "plotly_dark"
        )
        
        # Add data labels
        fig.update_traces(
            texttemplate='%{y:.2f}',
            textposition='outside'
        )
        
        # Save if output path provided
        if output_path:
            if not output_path.endswith(('.html', '.json', '.png', '.jpg', '.svg', '.pdf')):
                output_path += f".{self.format}"
                
            pio.write_html(fig, output_path)
            logger.info(f"Saved comparison bar chart to {output_path}")
            return None
            
        return fig
    
    def _generate_matplotlib_comparison_bar(self, labels: List[str],
                                          values: List[float],
                                          title: str,
                                          metric: str,
                                          output_path: Optional[str]) -> Optional[plt.Figure]:
        """
        Generate a static Matplotlib comparison bar chart.
        
        Args:
            labels: Labels for each bar
            values: Values for each bar
            title: Chart title
            metric: Metric name
            output_path: Path to save the chart
            
        Returns:
            Matplotlib figure if output_path is None, otherwise None
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(self.width/100, self.height/100))
        
        # Create bar chart
        bars = ax.bar(labels, values, color='royalblue')
        
        # Add data labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(values),
                   f'{height:.2f}', ha='center', va='bottom', fontsize=9)
        
        # Set labels and title
        ax.set_xlabel("Benchmark")
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(title, fontsize=14)
        
        # Format x-axis labels if they're long
        if any(len(label) > 10 for label in labels):
            plt.xticks(rotation=45, ha='right')
        
        # Add grid
        ax.grid(True, axis='y', alpha=0.3)
        
        # Tight layout
        fig.tight_layout()
        
        # Save if output path provided
        if output_path:
            if not output_path.endswith(('.png', '.jpg', '.svg', '.pdf')):
                output_path += ".png"
                
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            logger.info(f"Saved comparison bar chart to {output_path}")
            return None
            
        return fig
    
    def _generate_plotly_comparison_line(self, labels: List[str],
                                       values: List[float],
                                       title: str,
                                       metric: str,
                                       output_path: Optional[str]) -> Optional[Dict]:
        """
        Generate an interactive Plotly comparison line chart.
        
        Args:
            labels: Labels for each point
            values: Values for each point
            title: Chart title
            metric: Metric name
            output_path: Path to save the chart
            
        Returns:
            Plotly figure dict if output_path is None, otherwise None
        """
        # Create figure
        fig = go.Figure(data=[
            go.Scatter(
                x=labels,
                y=values,
                mode='lines+markers',
                marker=dict(size=10),
                line=dict(width=3)
            )
        ])
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Benchmark",
            yaxis_title=metric.replace('_', ' ').title(),
            width=self.width,
            height=self.height,
            template="plotly_white" if self.theme == "default" else "plotly_dark"
        )
        
        # Add data labels
        fig.update_traces(
            texttemplate='%{y:.2f}',
            textposition='top center'
        )
        
        # Save if output path provided
        if output_path:
            if not output_path.endswith(('.html', '.json', '.png', '.jpg', '.svg', '.pdf')):
                output_path += f".{self.format}"
                
            pio.write_html(fig, output_path)
            logger.info(f"Saved comparison line chart to {output_path}")
            return None
            
        return fig
    
    def _generate_matplotlib_comparison_line(self, labels: List[str],
                                           values: List[float],
                                           title: str,
                                           metric: str,
                                           output_path: Optional[str]) -> Optional[plt.Figure]:
        """
        Generate a static Matplotlib comparison line chart.
        
        Args:
            labels: Labels for each point
            values: Values for each point
            title: Chart title
            metric: Metric name
            output_path: Path to save the chart
            
        Returns:
            Matplotlib figure if output_path is None, otherwise None
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(self.width/100, self.height/100))
        
        # Create line chart
        ax.plot(labels, values, 'o-', linewidth=2, markersize=8)
        
        # Add data labels
        for i, (x, y) in enumerate(zip(labels, values)):
            ax.text(i, y + 0.01*max(values), f'{y:.2f}', ha='center', va='bottom', fontsize=9)
        
        # Set labels and title
        ax.set_xlabel("Benchmark")
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(title, fontsize=14)
        
        # Format x-axis labels if they're long
        if any(len(label) > 10 for label in labels):
            plt.xticks(rotation=45, ha='right')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Tight layout
        fig.tight_layout()
        
        # Save if output path provided
        if output_path:
            if not output_path.endswith(('.png', '.jpg', '.svg', '.pdf')):
                output_path += ".png"
                
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            logger.info(f"Saved comparison line chart to {output_path}")
            return None
            
        return fig
    
    def generate_scatter_plot(self, benchmark_data: BenchmarkData,
                            x_metric: str,
                            y_metric: str,
                            color_by: Optional[str] = None,
                            title: Optional[str] = None,
                            add_trendline: bool = True,
                            output_path: Optional[str] = None) -> Optional[Union[plt.Figure, Dict]]:
        """
        Generate a scatter plot comparing two metrics.
        
        Args:
            benchmark_data: BenchmarkData instance
            x_metric: Metric for x-axis
            y_metric: Metric for y-axis
            color_by: Optional metric to color points by
            title: Chart title
            add_trendline: Whether to add a trendline
            output_path: Path to save the chart
            
        Returns:
            Matplotlib figure or Plotly figure dict if output_path is None, otherwise None
        """
        if not self.enabled:
            logger.info("Chart generation is disabled in config")
            return None
            
        if "scatter" not in self.chart_types:
            logger.info("Scatter chart type is not enabled in config")
            return None
        
        try:
            # Get time series data
            df = benchmark_data.get_time_series(reset_index=True)
            
            if df.empty:
                logger.warning("No time series data available for scatter plot generation")
                return None
            
            # Check if metrics exist
            if x_metric not in df.columns:
                logger.warning(f"X-axis metric {x_metric} not found in data")
                return None
                
            if y_metric not in df.columns:
                logger.warning(f"Y-axis metric {y_metric} not found in data")
                return None
            
            # Filter out rows with NaN values
            df = df.dropna(subset=[x_metric, y_metric])
            
            if df.empty:
                logger.warning("No valid data points after filtering NaNs")
                return None
            
            # Check color_by column
            if color_by and color_by not in df.columns:
                logger.warning(f"Color metric {color_by} not found, ignoring")
                color_by = None
            
            # Default title
            if title is None:
                title = f"{y_metric.replace('_', ' ').title()} vs {x_metric.replace('_', ' ').title()}"
            
            # Generate chart based on interactive setting
            if self.interactive:
                return self._generate_plotly_scatter(
                    df, x_metric, y_metric, color_by, title, add_trendline, output_path
                )
            else:
                return self._generate_matplotlib_scatter(
                    df, x_metric, y_metric, color_by, title, add_trendline, output_path
                )
                
        except Exception as e:
            logger.exception(f"Error generating scatter plot: {str(e)}")
            raise VisualizationError(f"Failed to generate scatter plot: {str(e)}")
    
    def _generate_plotly_scatter(self, df: pd.DataFrame,
                               x_metric: str,
                               y_metric: str,
                               color_by: Optional[str],
                               title: str,
                               add_trendline: bool,
                               output_path: Optional[str]) -> Optional[Dict]:
        """
        Generate an interactive Plotly scatter plot.
        
        Args:
            df: DataFrame with data
            x_metric: Metric for x-axis
            y_metric: Metric for y-axis
            color_by: Optional metric to color points by
            title: Chart title
            add_trendline: Whether to add a trendline
            output_path: Path to save the chart
            
        Returns:
            Plotly figure dict if output_path is None, otherwise None
        """
        # Create figure
        if color_by:
            fig = px.scatter(
                df, x=x_metric, y=y_metric, color=color_by,
                title=title,
                labels={
                    x_metric: x_metric.replace('_', ' ').title(),
                    y_metric: y_metric.replace('_', ' ').title(),
                    color_by: color_by.replace('_', ' ').title()
                },
                width=self.width,
                height=self.height,
                template="plotly_white" if self.theme == "default" else "plotly_dark"
            )
        else:
            fig = px.scatter(
                df, x=x_metric, y=y_metric,
                title=title,
                labels={
                    x_metric: x_metric.replace('_', ' ').title(),
                    y_metric: y_metric.replace('_', ' ').title()
                },
                width=self.width,
                height=self.height,
                template="plotly_white" if self.theme == "default" else "plotly_dark"
            )
        
        # Add trendline if requested
        if add_trendline:
            fig.update_layout(showlegend=True)
            
            # Add OLS trendline
            x = df[x_metric]
            y = df[y_metric]
            
            # Calculate trendline
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            
            # Add trendline trace
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=p(x),
                    mode='lines',
                    name=f'Trendline (y={z[0]:.2f}x + {z[1]:.2f})',
                    line=dict(color='red', dash='dash')
                )
            )
        
        # Save if output path provided
        if output_path:
            if not output_path.endswith(('.html', '.json', '.png', '.jpg', '.svg', '.pdf')):
                output_path += f".{self.format}"
                
            pio.write_html(fig, output_path)
            logger.info(f"Saved scatter plot to {output_path}")
            return None
            
        return fig
    
    def _generate_matplotlib_scatter(self, df: pd.DataFrame,
                                   x_metric: str,
                                   y_metric: str,
                                   color_by: Optional[str],
                                   title: str,
                                   add_trendline: bool,
                                   output_path: Optional[str]) -> Optional[plt.Figure]:
        """
        Generate a static Matplotlib scatter plot.
        
        Args:
            df: DataFrame with data
            x_metric: Metric for x-axis
            y_metric: Metric for y-axis
            color_by: Optional metric to color points by
            title: Chart title
            add_trendline: Whether to add a trendline
            output_path: Path to save the chart
            
        Returns:
            Matplotlib figure if output_path is None, otherwise None
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(self.width/100, self.height/100))
        
        # Create scatter plot
        if color_by:
            scatter = ax.scatter(
                df[x_metric], df[y_metric],
                c=df[color_by], cmap='viridis',
                alpha=0.7, edgecolors='w', linewidth=0.5
            )
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label(color_by.replace('_', ' ').title())
        else:
            ax.scatter(
                df[x_metric], df[y_metric],
                color='royalblue',
                alpha=0.7, edgecolors='w', linewidth=0.5
            )
        
        # Add trendline if requested
        if add_trendline:
            x = df[x_metric]
            y = df[y_metric]
            
            # Calculate trendline
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            
            # Add trendline to plot
            ax.plot(x, p(x), 'r--', label=f'Trendline (y={z[0]:.2f}x + {z[1]:.2f})')
            ax.legend()
        
        # Set labels and title
        ax.set_xlabel(x_metric.replace('_', ' ').title())
        ax.set_ylabel(y_metric.replace('_', ' ').title())
        ax.set_title(title, fontsize=14)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Tight layout
        fig.tight_layout()
        
        # Save if output path provided
        if output_path:
            if not output_path.endswith(('.png', '.jpg', '.svg', '.pdf')):
                output_path += ".png"
                
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            logger.info(f"Saved scatter plot to {output_path}")
            return None
            
        return fig
    
    def generate_distribution_plot(self, benchmark_data: BenchmarkData,
                                 metric: str,
                                 plot_type: str = 'histogram',
                                 title: Optional[str] = None,
                                 bins: int = 30,
                                 kde: bool = True,
                                 output_path: Optional[str] = None) -> Optional[Union[plt.Figure, Dict]]:
        """
        Generate a distribution plot for a metric.
        
        Args:
            benchmark_data: BenchmarkData instance
            metric: Metric to visualize
            plot_type: Type of plot ('histogram' or 'boxplot')
            title: Chart title
            bins: Number of bins for histogram
            kde: Whether to add KDE curve to histogram
            output_path: Path to save the chart
            
        Returns:
            Matplotlib figure or Plotly figure dict if output_path is None, otherwise None
        """
        if not self.enabled:
            logger.info("Chart generation is disabled in config")
            return None
        
        try:
            # Get time series data
            df = benchmark_data.get_time_series(reset_index=True)
            
            if df.empty:
                logger.warning("No time series data available for distribution plot")
                return None
            
            # Check if metric exists
            if metric not in df.columns:
                logger.warning(f"Metric {metric} not found in data")
                return None
            
            # Filter out NaN values
            series = df[metric].dropna()
            
            if len(series) < 2:
                logger.warning(f"Insufficient data points for {metric}")
                return None
            
            # Default title
            if title is None:
                title = f"Distribution of {metric.replace('_', ' ').title()}"
            
            # Generate chart based on plot type and interactive setting
            if plot_type == 'histogram':
                if self.interactive:
                    return self._generate_plotly_histogram(
                        series, title, metric, bins, kde, output_path
                    )
                else:
                    return self._generate_matplotlib_histogram(
                        series, title, metric, bins, kde, output_path
                    )
            elif plot_type == 'boxplot':
                if self.interactive:
                    return self._generate_plotly_boxplot(
                        series, title, metric, output_path
                    )
                else:
                    return self._generate_matplotlib_boxplot(
                        series, title, metric, output_path
                    )
            else:
                logger.warning(f"Unknown plot type: {plot_type}")
                return None
                
        except Exception as e:
            logger.exception(f"Error generating distribution plot: {str(e)}")
            raise VisualizationError(f"Failed to generate distribution plot: {str(e)}")
    
    def _generate_plotly_histogram(self, series: pd.Series,
                                 title: str,
                                 metric: str,
                                 bins: int,
                                 kde: bool,
                                 output_path: Optional[str]) -> Optional[Dict]:
        """
        Generate an interactive Plotly histogram.
        
        Args:
            series: Series with data
            title: Chart title
            metric: Metric name
            bins: Number of bins
            kde: Whether to add KDE curve
            output_path: Path to save the chart
            
        Returns:
            Plotly figure dict if output_path is None, otherwise None
        """
        # Create figure
        fig = go.Figure()
        
        # Add histogram
        fig.add_trace(go.Histogram(
            x=series,
            nbinsx=bins,
            opacity=0.7,
            name=metric
        ))
        
        # Add KDE if requested
        if kde:
            # Calculate KDE (using numpy for simplicity)
            from scipy import stats
            
            kde_x = np.linspace(min(series), max(series), 1000)
            kde_y = stats.gaussian_kde(series)(kde_x)
            
            # Scale to match histogram height
            max_kde = max(kde_y)
            max_histogram = max(np.histogram(series, bins=bins)[0])
            kde_y = kde_y * (max_histogram / max_kde)
            
            # Add KDE line
            fig.add_trace(go.Scatter(
                x=kde_x,
                y=kde_y,
                mode='lines',
                name='KDE',
                line=dict(color='red', width=2)
            ))
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title=metric.replace('_', ' ').title(),
            yaxis_title='Frequency',
            width=self.width,
            height=self.height,
            template="plotly_white" if self.theme == "default" else "plotly_dark",
            bargap=0.1
        )
        
        # Save if output path provided
        if output_path:
            if not output_path.endswith(('.html', '.json', '.png', '.jpg', '.svg', '.pdf')):
                output_path += f".{self.format}"
                
            pio.write_html(fig, output_path)
            logger.info(f"Saved histogram to {output_path}")
            return None
            
        return fig
    
    def _generate_matplotlib_histogram(self, series: pd.Series,
                                     title: str,
                                     metric: str,
                                     bins: int,
                                     kde: bool,
                                     output_path: Optional[str]) -> Optional[plt.Figure]:
        """
        Generate a static Matplotlib histogram.
        
        Args:
            series: Series with data
            title: Chart title
            metric: Metric name
            bins: Number of bins
            kde: Whether to add KDE curve
            output_path: Path to save the chart
            
        Returns:
            Matplotlib figure if output_path is None, otherwise None
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(self.width/100, self.height/100))
        
        # Create histogram with KDE
        sns.histplot(series, bins=bins, kde=kde, ax=ax, color='royalblue')
        
        # Set labels and title
        ax.set_xlabel(metric.replace('_', ' ').title())
        ax.set_ylabel('Frequency')
        ax.set_title(title, fontsize=14)
        
        # Add statistics
        stats_text = (f"Mean: {series.mean():.2f}\n"
                      f"Median: {series.median():.2f}\n"
                      f"Std Dev: {series.std():.2f}")
        
        ax.text(0.95, 0.95, stats_text,
               transform=ax.transAxes,
               verticalalignment='top',
               horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Tight layout
        fig.tight_layout()
        
        # Save if output path provided
        if output_path:
            if not output_path.endswith(('.png', '.jpg', '.svg', '.pdf')):
                output_path += ".png"
                
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            logger.info(f"Saved histogram to {output_path}")
            return None
            
        return fig
    
    def _generate_plotly_boxplot(self, series: pd.Series,
                               title: str,
                               metric: str,
                               output_path: Optional[str]) -> Optional[Dict]:
        """
        Generate an interactive Plotly boxplot.
        
        Args:
            series: Series with data
            title: Chart title
            metric: Metric name
            output_path: Path to save the chart
            
        Returns:
            Plotly figure dict if output_path is None, otherwise None
        """
        # Create figure
        fig = go.Figure()
        
        # Add box plot
        fig.add_trace(go.Box(
            y=series,
            name=metric,
            boxpoints='all',
            jitter=0.3,
            pointpos=-1.8,
            boxmean=True
        ))
        
        # Update layout
        fig.update_layout(
            title=title,
            yaxis_title=metric.replace('_', ' ').title(),
            width=self.width,
            height=self.height,
            template="plotly_white" if self.theme == "default" else "plotly_dark"
        )
        
        # Save if output path provided
        if output_path:
            if not output_path.endswith(('.html', '.json', '.png', '.jpg', '.svg', '.pdf')):
                output_path += f".{self.format}"
                
            pio.write_html(fig, output_path)
            logger.info(f"Saved boxplot to {output_path}")
            return None
            
        return fig
    
    def _generate_matplotlib_boxplot(self, series: pd.Series,
                                   title: str,
                                   metric: str,
                                   output_path: Optional[str]) -> Optional[plt.Figure]:
        """
        Generate a static Matplotlib boxplot.
        
        Args:
            series: Series with data
            title: Chart title
            metric: Metric name
            output_path: Path to save the chart
            
        Returns:
            Matplotlib figure if output_path is None, otherwise None
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(self.width/100, self.height/100))
        
        # Create boxplot
        boxplot = ax.boxplot(series, vert=True, patch_artist=True, showmeans=True)
        
        # Customize boxplot colors
        for box in boxplot['boxes']:
            box.set(facecolor='royalblue', alpha=0.7)
        for median in boxplot['medians']:
            median.set(color='darkred', linewidth=2)
        for flier in boxplot['fliers']:
            flier.set(marker='o', markerfacecolor='red', alpha=0.5)
        
        # Set labels and title
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(title, fontsize=14)
        
        # Add statistics
        stats_text = (f"Mean: {series.mean():.2f}\n"
                      f"Median: {series.median():.2f}\n"
                      f"Std Dev: {series.std():.2f}\n"
                      f"IQR: {series.quantile(0.75) - series.quantile(0.25):.2f}")
        
        ax.text(0.95, 0.95, stats_text,
               transform=ax.transAxes,
               verticalalignment='top',
               horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Remove x-ticks
        ax.set_xticks([])
        
        # Add grid
        ax.grid(True, axis='y', alpha=0.3)
        
        # Tight layout
        fig.tight_layout()
        
        # Save if output path provided
        if output_path:
            if not output_path.endswith(('.png', '.jpg', '.svg', '.pdf')):
                output_path += ".png"
                
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            logger.info(f"Saved boxplot to {output_path}")
            return None
            
        return fig
    
    def generate_chart_dashboard(self, data: BenchmarkData, 
                           chart_configs: List[Dict[str, Any]],
                           title: Optional[str] = "Benchmark Results Dashboard",
                           figsize: Tuple[int, int] = (16, 10),
                           layout: Optional[Tuple[int, int]] = None,
                           output_path: Optional[str] = None,
                           output_format: str = "png",
                           dpi: int = 300,
                           show: bool = False) -> plt.Figure:
        """
        Generate a dashboard with multiple charts arranged in a grid.
    
        Args:
            data: BenchmarkData instance to visualize
            chart_configs: List of chart configuration dictionaries
            title: Dashboard title
            figsize: Figure size (width, height) in inches
            layout: Grid layout (rows, cols), auto-calculated if None
            output_path: Path to save the dashboard
            output_format: Output file format (png, pdf, svg, etc.)
            dpi: Resolution for raster formats
            show: Whether to display the dashboard
        
        Returns:
            Matplotlib Figure with the dashboard
        """
        if not chart_configs:
            raise ValueError("No chart configurations provided")
    
        # Determine grid layout if not specified
        if layout is None:
            n_charts = len(chart_configs)
            # Calculate a reasonable grid layout based on number of charts
            if n_charts <= 2:
                layout = (1, n_charts)
            elif n_charts <= 4:
                layout = (2, 2)
            elif n_charts <= 6:
                layout = (2, 3)
            elif n_charts <= 9:
                layout = (3, 3)
            elif n_charts <= 12:
                layout = (3, 4)
            else:
                # For many charts, use a more compact layout
                n_cols = min(4, int(np.ceil(np.sqrt(n_charts))))
                n_rows = int(np.ceil(n_charts / n_cols))
                layout = (n_rows, n_cols)
    
        n_rows, n_cols = layout
    
        # Create figure and gridspec
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        gs = fig.add_gridspec(n_rows + 1, n_cols)  # +1 for title
    
        # Add dashboard title
        if title:
            title_ax = fig.add_subplot(gs[0, :])
            title_ax.set_title(title, fontsize=16, fontweight='bold')
            title_ax.axis('off')
    
        # Generate charts and add to grid
        for i, config in enumerate(chart_configs):
            if i >= n_rows * n_cols:
                logger.warning(f"Dashboard layout ({n_rows}x{n_cols}) cannot accommodate all {len(chart_configs)} charts")
                break
            
            # Calculate grid position (skip first row if title is present)
            row_offset = 1 if title else 0
            row = (i // n_cols) + row_offset
            col = i % n_cols
        
            # Create subplot
            ax = fig.add_subplot(gs[row, col])
        
            try:
                # Extract chart type and parameters
                chart_type = config.get("type", "line")
                chart_params = config.get("params", {})
            
                # Generate specific chart type
                if chart_type == "line":
                    self._generate_line_chart(data, ax=ax, **chart_params)
                elif chart_type == "bar":
                    self._generate_bar_chart(data, ax=ax, **chart_params)
                elif chart_type == "scatter":
                    self._generate_scatter_chart(data, ax=ax, **chart_params)
                elif chart_type == "histogram":
                    self._generate_histogram_chart(data, ax=ax, **chart_params)
                elif chart_type == "heatmap":
                    self._generate_heatmap_chart(data, ax=ax, **chart_params)
                elif chart_type == "box":
                    self._generate_box_chart(data, ax=ax, **chart_params)
                elif chart_type == "violin":
                    self._generate_violin_chart(data, ax=ax, **chart_params)
                elif chart_type == "pie":
                    self._generate_pie_chart(data, ax=ax, **chart_params)
                elif chart_type == "area":
                    self._generate_area_chart(data, ax=ax, **chart_params)
                elif chart_type == "radar":
                    self._generate_radar_chart(data, ax=ax, **chart_params)
                else:
                    logger.warning(f"Unsupported chart type: {chart_type}")
                    ax.text(0.5, 0.5, f"Unsupported chart type: {chart_type}",
                            ha='center', va='center', transform=ax.transAxes)
                    ax.axis('off')
                
                # Add chart title if specified
                if "title" in config:
                    ax.set_title(config["title"])
                
            except Exception as e:
                logger.error(f"Error generating chart: {str(e)}")
                ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center', 
                       transform=ax.transAxes, color='red')
                ax.axis('off')
    
        # Adjust layout
        fig.tight_layout()
    
        # Save dashboard if output path is provided
        if output_path:
            # Ensure the output directory exists
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
            # Ensure correct file extension
            if not output_path.lower().endswith(f".{output_format.lower()}"):
                output_path = f"{output_path}.{output_format.lower()}"
            
            # Save the figure
            fig.savefig(output_path, format=output_format, dpi=dpi, bbox_inches='tight')
            logger.info(f"Dashboard saved to {output_path}")
    
        # Show the dashboard if requested
        if show:
            plt.show()
    
        return fig
