#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Network analyzer module for benchmark results analysis (Tiered Storage I/O Benchmark).

This module provides methods to analyze network performance metrics
during storage benchmark execution, helping identify network bottlenecks
and correlations between network activity and storage performance.

Author: Jack Ogaja
Date: 2025-06-30
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

from tdiobench.core.benchmark_exceptions import BenchmarkAnalysisError
from tdiobench.analysis.base_analyzer import BaseAnalyzer
from tdiobench.core.benchmark_analysis import AnalysisResult

logger = logging.getLogger(__name__)


class NetworkAnalyzer(BaseAnalyzer):
    """
    Network performance analyzer for I/O benchmark data.
    
    This class implements methods to analyze network metrics collected during
    benchmark execution, identify bottlenecks, and correlate network performance
    with storage performance metrics.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the network analyzer with configuration.
        
        Args:
            config: Configuration dictionary with network analysis parameters
        """
        super().__init__(config)
        self.network_config = self.config.get("analysis", {}).get("network", {})
        self.enabled = self.network_config.get("enabled", True)
        self.detect_protocol = self.network_config.get("detect_protocol", True)
        self.packet_capture = self.network_config.get("packet_capture", False)
        self.interface_monitoring = self.network_config.get("interface_monitoring", True)
        
        # Network threshold settings
        self.thresholds = self.network_config.get("thresholds", {})
        self.utilization_threshold = self.thresholds.get("utilization_percent", 80)
        self.retransmit_threshold = self.thresholds.get("retransmit_percent", 2)
        self.latency_threshold = self.thresholds.get("latency_ms", 10)
        
        # Correlation settings
        self.correlation_threshold = self.network_config.get("correlation_threshold", 0.7)
        
        logger.debug(f"Initialized NetworkAnalyzer with interface_monitoring={self.interface_monitoring}")
    
    def analyze_network_metrics(self, data: pd.DataFrame,
                               storage_metrics: List[str] = None) -> AnalysisResult:
        """
        Analyze network metrics and their impact on storage performance.
        
        This is the main entry point for network analysis that runs various
        network-related analyses based on configuration.
        
        Args:
            data: DataFrame containing benchmark and network metrics time series data
            storage_metrics: List of storage metric columns to correlate with network
            
        Returns:
            AnalysisResult object containing network analysis results
        """
        if not self.enabled:
            logger.info("Network analysis disabled in config")
            return AnalysisResult(
                name="network_analysis",
                status="skipped",
                data={"reason": "Network analysis disabled in configuration"}
            )
        
        if data.empty:
            logger.warning("Cannot analyze network metrics in empty dataset")
            return AnalysisResult(
                name="network_analysis",
                status="error",
                data={"error": "Empty dataset provided"}
            )
        
        # Default storage metrics if none provided
        if storage_metrics is None:
            storage_metrics = ["throughput_MBps", "iops", "latency_ms"]
            # Filter to only include metrics that exist in the data
            storage_metrics = [m for m in storage_metrics if m in data.columns]
        
        # Identify network metrics in the data
        network_metrics = self._identify_network_metrics(data)
        
        if not network_metrics:
            logger.warning("No network metrics found in the dataset")
            return AnalysisResult(
                name="network_analysis",
                status="skipped",
                data={"reason": "No network metrics found in the dataset"}
            )
        
        logger.info(f"Analyzing network metrics: {network_metrics}")
        
        try:
            # Run multiple network analyses
            results = {}
            
            # Basic network statistics
            results["statistics"] = self._calculate_network_statistics(data, network_metrics)
            
            # Interface utilization analysis
            if self.interface_monitoring:
                results["interface_utilization"] = self._analyze_interface_utilization(data, network_metrics)
            
            # Network protocol detection
            if self.detect_protocol:
                results["protocol_analysis"] = self._analyze_protocols(data, network_metrics)
            
            # Network-storage correlation
            if storage_metrics:
                results["correlation"] = self._analyze_network_storage_correlation(
                    data, network_metrics, storage_metrics
                )
            
            # Network bottleneck detection
            results["bottlenecks"] = self._detect_network_bottlenecks(data, network_metrics)
            
            # Packet analysis if available
            if self.packet_capture and any(col.startswith('packet_') for col in data.columns):
                results["packet_analysis"] = self._analyze_packet_metrics(data)
            
            # Generate overall summary
            results["summary"] = self._generate_network_summary(results)
            
            return AnalysisResult(
                name="network_analysis",
                status="success",
                data=results
            )
            
        except Exception as e:
            logger.exception(f"Error during network analysis: {str(e)}")
            return AnalysisResult(
                name="network_analysis",
                status="error",
                data={"error": str(e)}
            )
    
    def _identify_network_metrics(self, data: pd.DataFrame) -> List[str]:
        """
        Identify network-related metrics in the dataset.
        
        Args:
            data: DataFrame containing benchmark data
            
        Returns:
            List of column names that are network metrics
        """
        # Common prefixes and patterns for network metrics
        network_prefixes = [
            'net_', 'network_', 'eth', 'ib_', 'nic_', 'tcp_', 'udp_', 
            'packet_', 'interface_', 'connection_', 'socket_'
        ]
        
        network_metrics = []
        
        for col in data.columns:
            # Check if column starts with any network prefix
            if any(col.startswith(prefix) for prefix in network_prefixes):
                network_metrics.append(col)
            # Check for common network metric names
            elif any(term in col.lower() for term in ['bandwidth', 'throughput', 'packet', 
                                                     'retransmit', 'mtu', 'collision']):
                network_metrics.append(col)
        
        return network_metrics
    
    def _calculate_network_statistics(self, data: pd.DataFrame,
                                     network_metrics: List[str]) -> Dict[str, Any]:
        """
        Calculate basic statistics for network metrics.
        
        Args:
            data: DataFrame containing benchmark data
            network_metrics: List of network metric columns
            
        Returns:
            Dictionary with network statistics
        """
        stats_result = {}
        
        for metric in network_metrics:
            if metric not in data.columns:
                continue
                
            series = data[metric].dropna()
            if len(series) < 2:
                continue
                
            # Calculate basic statistics
            stats_dict = {
                "mean": float(series.mean()),
                "median": float(series.median()),
                "std": float(series.std()),
                "min": float(series.min()),
                "max": float(series.max()),
                "q25": float(series.quantile(0.25)),
                "q75": float(series.quantile(0.75)),
                "iqr": float(series.quantile(0.75) - series.quantile(0.25)),
                "data_points": len(series)
            }
            
            # Calculate coefficient of variation (normalized standard deviation)
            if stats_dict["mean"] != 0:
                stats_dict["cv"] = stats_dict["std"] / stats_dict["mean"]
            else:
                stats_dict["cv"] = np.nan
                
            # Check for stability (low CV indicates stable metric)
            stats_dict["is_stable"] = stats_dict.get("cv", np.inf) < 0.1
                
            stats_result[metric] = stats_dict
        
        return stats_result
    
    def _analyze_interface_utilization(self, data: pd.DataFrame,
                                      network_metrics: List[str]) -> Dict[str, Any]:
        """
        Analyze network interface utilization.
        
        Args:
            data: DataFrame containing benchmark data
            network_metrics: List of network metric columns
            
        Returns:
            Dictionary with interface utilization analysis
        """
        utilization_result = {}
        interfaces = set()
        
        # Find interface utilization metrics
        util_metrics = [m for m in network_metrics if 'util' in m.lower()]
        bandwidth_metrics = [m for m in network_metrics if any(term in m.lower() 
                                                            for term in ['bandwidth', 'throughput'])]
        
        # Extract interface names (assuming format like eth0_util or net_eth0_throughput)
        for metric in network_metrics:
            parts = metric.split('_')
            for part in parts:
                if part.startswith('eth') or part.startswith('ib') or part == 'lo':
                    interfaces.add(part)
        
        # Analyze each interface
        for interface in interfaces:
            interface_metrics = [m for m in network_metrics if interface in m]
            
            if not interface_metrics:
                continue
                
            # Find utilization for this interface
            interface_util = None
            for metric in util_metrics:
                if interface in metric:
                    interface_util = metric
                    break
            
            # If no direct utilization metric, check for bandwidth metrics
            interface_bandwidth = None
            if not interface_util:
                for metric in bandwidth_metrics:
                    if interface in metric:
                        interface_bandwidth = metric
                        break
            
            # Analyze the available metrics
            if interface_util and interface_util in data.columns:
                series = data[interface_util].dropna()
                
                if len(series) > 0:
                    util_stats = {
                        "mean_utilization": float(series.mean()),
                        "max_utilization": float(series.max()),
                        "over_threshold_percent": float((series > self.utilization_threshold).mean() * 100),
                        "time_over_threshold_seconds": float((series > self.utilization_threshold).sum())
                    }
                    
                    # Detect saturation periods
                    if util_stats["over_threshold_percent"] > 5:
                        util_stats["bottleneck_detected"] = True
                        util_stats["bottleneck_severity"] = "high" if util_stats["over_threshold_percent"] > 20 else "medium"
                    else:
                        util_stats["bottleneck_detected"] = False
                        
                    utilization_result[interface] = util_stats
            
            # If only bandwidth available, use that for utilization estimation
            elif interface_bandwidth and interface_bandwidth in data.columns:
                series = data[interface_bandwidth].dropna()
                
                if len(series) > 0:
                    bandwidth_stats = {
                        "mean_bandwidth": float(series.mean()),
                        "max_bandwidth": float(series.max()),
                        "bandwidth_stability": float(series.std() / series.mean()) if series.mean() > 0 else np.nan
                    }
                    
                    # Cannot determine bottleneck without max capacity reference
                    bandwidth_stats["bottleneck_detected"] = "unknown"
                    
                    utilization_result[interface] = bandwidth_stats
        
        return utilization_result
    
    def _analyze_protocols(self, data: pd.DataFrame,
                          network_metrics: List[str]) -> Dict[str, Any]:
        """
        Analyze network protocols used during the benchmark.
        
        Args:
            data: DataFrame containing benchmark data
            network_metrics: List of network metric columns
            
        Returns:
            Dictionary with protocol analysis information
        """
        protocol_result = {}
        
        # Identify protocol-specific metrics
        tcp_metrics = [m for m in network_metrics if 'tcp' in m.lower()]
        udp_metrics = [m for m in network_metrics if 'udp' in m.lower()]
        ip_metrics = [m for m in network_metrics if '_ip_' in m.lower()]
        
        # Analyze TCP metrics if available
        if tcp_metrics:
            tcp_result = {
                "metrics_found": tcp_metrics,
                "protocol_active": True
            }
            
            # Look for retransmits
            retransmit_metrics = [m for m in tcp_metrics if 'retrans' in m.lower()]
            if retransmit_metrics and retransmit_metrics[0] in data.columns:
                retrans_series = data[retransmit_metrics[0]].dropna()
                if len(retrans_series) > 0:
                    tcp_result["retransmit_rate"] = float(retrans_series.mean())
                    tcp_result["max_retransmit"] = float(retrans_series.max())
                    tcp_result["retransmit_detected"] = float(retrans_series.max()) > self.retransmit_threshold
            
            protocol_result["tcp"] = tcp_result
        
        # Analyze UDP metrics if available
        if udp_metrics:
            udp_result = {
                "metrics_found": udp_metrics,
                "protocol_active": True
            }
            
            # Look for packet loss
            loss_metrics = [m for m in udp_metrics if 'loss' in m.lower()]
            if loss_metrics and loss_metrics[0] in data.columns:
                loss_series = data[loss_metrics[0]].dropna()
                if len(loss_series) > 0:
                    udp_result["packet_loss_rate"] = float(loss_series.mean())
                    udp_result["max_packet_loss"] = float(loss_series.max())
                    udp_result["packet_loss_detected"] = float(loss_series.max()) > 0
            
            protocol_result["udp"] = udp_result
        
        # Determine dominant protocol
        if tcp_metrics and udp_metrics:
            protocol_result["dominant_protocol"] = "tcp+udp"
        elif tcp_metrics:
            protocol_result["dominant_protocol"] = "tcp"
        elif udp_metrics:
            protocol_result["dominant_protocol"] = "udp"
        else:
            protocol_result["dominant_protocol"] = "unknown"
        
        return protocol_result
    
    def _analyze_network_storage_correlation(self, data: pd.DataFrame,
                                           network_metrics: List[str],
                                           storage_metrics: List[str]) -> Dict[str, Any]:
        """
        Analyze correlation between network and storage performance metrics.
        
        Args:
            data: DataFrame containing benchmark data
            network_metrics: List of network metric columns
            storage_metrics: List of storage metric columns
            
        Returns:
            Dictionary with correlation analysis results
        """
        correlation_result = {
            "strong_correlations": [],
            "correlation_matrix": {}
        }
        
        # Filter to metrics that exist in the data
        network_metrics = [m for m in network_metrics if m in data.columns]
        storage_metrics = [m for m in storage_metrics if m in data.columns]
        
        if not network_metrics or not storage_metrics:
            return correlation_result
        
        # Calculate correlation matrix between network and storage metrics
        for net_metric in network_metrics:
            correlation_result["correlation_matrix"][net_metric] = {}
            
            for storage_metric in storage_metrics:
                # Extract non-NaN values from both series
                valid_data = data[[net_metric, storage_metric]].dropna()
                
                if len(valid_data) < 10:  # Need sufficient data points for correlation
                    correlation_result["correlation_matrix"][net_metric][storage_metric] = None
                    continue
                
                # Calculate Pearson correlation coefficient
                corr, p_value = stats.pearsonr(
                    valid_data[net_metric], 
                    valid_data[storage_metric]
                )
                
                correlation_result["correlation_matrix"][net_metric][storage_metric] = {
                    "coefficient": float(corr),
                    "p_value": float(p_value),
                    "significant": p_value < 0.05
                }
                
                # Identify strong correlations
                if abs(corr) > self.correlation_threshold and p_value < 0.05:
                    correlation_result["strong_correlations"].append({
                        "network_metric": net_metric,
                        "storage_metric": storage_metric,
                        "coefficient": float(corr),
                        "p_value": float(p_value),
                        "relationship": "positive" if corr > 0 else "negative"
                    })
        
        # Sort strong correlations by absolute coefficient
        correlation_result["strong_correlations"].sort(
            key=lambda x: abs(x["coefficient"]), 
            reverse=True
        )
        
        return correlation_result
    
    def _detect_network_bottlenecks(self, data: pd.DataFrame,
                                   network_metrics: List[str]) -> Dict[str, Any]:
        """
        Detect potential network bottlenecks during benchmark execution.
        
        Args:
            data: DataFrame containing benchmark data
            network_metrics: List of network metric columns
            
        Returns:
            Dictionary with bottleneck analysis results
        """
        bottleneck_result = {
            "detected": False,
            "bottlenecks": []
        }
        
        # Look for utilization metrics
        util_metrics = [m for m in network_metrics if 'util' in m.lower()]
        
        # Check high utilization periods
        for metric in util_metrics:
            if metric not in data.columns:
                continue
                
            series = data[metric].dropna()
            if len(series) < 2:
                continue
                
            # Detect periods of high utilization
            high_util = series > self.utilization_threshold
            if high_util.any():
                bottleneck_result["detected"] = True
                
                # Find contiguous periods of high utilization
                high_periods = self._find_contiguous_periods(high_util, data)
                
                bottleneck_result["bottlenecks"].append({
                    "metric": metric,
                    "threshold": self.utilization_threshold,
                    "max_value": float(series.max()),
                    "high_periods": high_periods,
                    "total_high_time_seconds": float(high_util.sum()),
                    "percent_high": float(high_util.mean() * 100)
                })
        
        # Look for latency metrics
        latency_metrics = [m for m in network_metrics if 'latency' in m.lower() 
                         or 'delay' in m.lower() or 'rtt' in m.lower()]
        
        # Check high latency periods
        for metric in latency_metrics:
            if metric not in data.columns:
                continue
                
            series = data[metric].dropna()
            if len(series) < 2:
                continue
                
            # Detect periods of high latency
            high_latency = series > self.latency_threshold
            if high_latency.any():
                bottleneck_result["detected"] = True
                
                # Find contiguous periods of high latency
                high_periods = self._find_contiguous_periods(high_latency, data)
                
                bottleneck_result["bottlenecks"].append({
                    "metric": metric,
                    "threshold": self.latency_threshold,
                    "max_value": float(series.max()),
                    "high_periods": high_periods,
                    "total_high_time_seconds": float(high_latency.sum()),
                    "percent_high": float(high_latency.mean() * 100)
                })
        
        # Look for packet loss or errors
        error_metrics = [m for m in network_metrics if any(term in m.lower() 
                                                         for term in ['error', 'drop', 'loss', 'crc'])]
        
        # Check periods with errors
        for metric in error_metrics:
            if metric not in data.columns:
                continue
                
            series = data[metric].dropna()
            if len(series) < 2:
                continue
                
            # Detect periods with errors
            has_errors = series > 0
            if has_errors.any():
                bottleneck_result["detected"] = True
                
                # Find contiguous periods with errors
                error_periods = self._find_contiguous_periods(has_errors, data)
                
                bottleneck_result["bottlenecks"].append({
                    "metric": metric,
                    "threshold": 0,
                    "max_value": float(series.max()),
                    "high_periods": error_periods,
                    "total_high_time_seconds": float(has_errors.sum()),
                    "percent_high": float(has_errors.mean() * 100)
                })
        
        return bottleneck_result
    
    def _find_contiguous_periods(self, boolean_series: pd.Series, 
                                data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Find contiguous periods where a boolean series is True.
        
        Args:
            boolean_series: Series of boolean values
            data: Original DataFrame with potential timestamp column
            
        Returns:
            List of dictionaries describing contiguous periods
        """
        if not boolean_series.any():
            return []
            
        # Calculate runs of True values
        runs = boolean_series.ne(boolean_series.shift()).cumsum()
        
        periods = []
        for run_id, group in boolean_series.groupby(runs):
            if group.iloc[0] and len(group) > 0:
                start_idx = group.index[0]
                end_idx = group.index[-1]
                
                period = {
                    "start_idx": int(start_idx),
                    "end_idx": int(end_idx),
                    "duration_seconds": len(group)
                }
                
                # Add timestamps if available
                if 'timestamp' in data.columns:
                    period["start_time"] = data.loc[start_idx, 'timestamp']
                    period["end_time"] = data.loc[end_idx, 'timestamp']
                
                periods.append(period)
        
        return periods
    
    def _analyze_packet_metrics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze packet-level metrics if available.
        
        Args:
            data: DataFrame containing benchmark data
            
        Returns:
            Dictionary with packet analysis results
        """
        packet_result = {}
        
        # Identify packet-related metrics
        packet_metrics = [col for col in data.columns if col.startswith('packet_')]
        
        if not packet_metrics:
            return {"available": False}
            
        packet_result["available"] = True
        packet_result["metrics_found"] = packet_metrics
        
        # Analyze packet size distribution if available
        size_metrics = [m for m in packet_metrics if 'size' in m]
        if size_metrics and size_metrics[0] in data.columns:
            size_series = data[size_metrics[0]].dropna()
            if len(size_series) > 0:
                packet_result["size_distribution"] = {
                    "mean": float(size_series.mean()),
                    "median": float(size_series.median()),
                    "std": float(size_series.std()),
                    "min": float(size_series.min()),
                    "max": float(size_series.max()),
                    "histogram": np.histogram(size_series, bins=10)[0].tolist()
                }
        
        # Analyze packet rate if available
        rate_metrics = [m for m in packet_metrics if 'rate' in m or 'pps' in m]
        if rate_metrics and rate_metrics[0] in data.columns:
            rate_series = data[rate_metrics[0]].dropna()
            if len(rate_series) > 0:
                packet_result["rate_analysis"] = {
                    "mean": float(rate_series.mean()),
                    "max": float(rate_series.max()),
                    "variability": float(rate_series.std() / rate_series.mean()) if rate_series.mean() > 0 else np.nan
                }
        
        return packet_result
    
    def _generate_network_summary(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a summary of all network analysis results.
        
        Args:
            analysis_results: Dictionary of all network analysis results
            
        Returns:
            Dictionary with summary information
        """
        summary = {
            "bottlenecks_detected": False,
            "performance_impact": "none",
            "recommendations": []
        }
        
        # Check for bottlenecks
        if analysis_results.get("bottlenecks", {}).get("detected", False):
            summary["bottlenecks_detected"] = True
            bottlenecks = analysis_results["bottlenecks"].get("bottlenecks", [])
            
            # Classify bottleneck severity based on duration and magnitude
            if any(b.get("percent_high", 0) > 30 for b in bottlenecks):
                summary["performance_impact"] = "severe"
                summary["recommendations"].append(
                    "Network bottlenecks are significantly impacting storage performance. "
                    "Consider upgrading network infrastructure or reducing concurrent workloads."
                )
            elif any(b.get("percent_high", 0) > 10 for b in bottlenecks):
                summary["performance_impact"] = "moderate"
                summary["recommendations"].append(
                    "Network bottlenecks are moderately impacting storage performance. "
                    "Consider optimizing network configuration or benchmark parameters."
                )
            else:
                summary["performance_impact"] = "minor"
                summary["recommendations"].append(
                    "Minor network bottlenecks detected. Performance impact is limited."
                )
        
        # Check protocol issues
        if "protocol_analysis" in analysis_results:
            protocol_data = analysis_results["protocol_analysis"]
            
            # Check for TCP retransmit issues
            if protocol_data.get("tcp", {}).get("retransmit_detected", False):
                summary["tcp_issues"] = True
                summary["recommendations"].append(
                    "TCP retransmissions detected. Check for network congestion, "
                    "packet loss, or misconfigured TCP parameters."
                )
            
            # Check for UDP packet loss
            if protocol_data.get("udp", {}).get("packet_loss_detected", False):
                summary["udp_issues"] = True
                summary["recommendations"].append(
                    "UDP packet loss detected. Consider increasing buffer sizes "
                    "or reducing UDP traffic during benchmarks."
                )
        
        # Check for strong correlations
        if "correlation" in analysis_results:
            strong_correlations = analysis_results["correlation"].get("strong_correlations", [])
            
            if strong_correlations:
                summary["storage_network_correlation"] = True
                strongest = strong_correlations[0]
                
                if strongest["coefficient"] > 0.9:
                    summary["correlation_impact"] = "strong"
                    summary["recommendations"].append(
                        f"Storage performance is strongly {strongest['relationship']} "
                        f"correlated with {strongest['network_metric']}. "
                        f"Network is likely a primary performance factor."
                    )
                elif strongest["coefficient"] > 0.7:
                    summary["correlation_impact"] = "moderate"
                    summary["recommendations"].append(
                        f"Storage performance shows moderate {strongest['relationship']} "
                        f"correlation with {strongest['network_metric']}. "
                        f"Network may be influencing performance."
                    )
        
        # Overall assessment
        if summary["performance_impact"] == "severe":
            summary["overall_assessment"] = "Network is a major bottleneck for storage performance"
        elif summary["performance_impact"] == "moderate":
            summary["overall_assessment"] = "Network is affecting storage performance"
        elif summary["performance_impact"] == "minor":
            summary["overall_assessment"] = "Network has minor impact on storage performance"
        else:
            summary["overall_assessment"] = "Network is not limiting storage performance"
        
        return summary
    
    def visualize_network_metrics(self, data: pd.DataFrame, metrics: List[str] = None,
                                 storage_metric: str = None, 
                                 save_path: str = None) -> Optional[plt.Figure]:
        """
        Visualize network metrics with optional storage metric overlay.
        
        Args:
            data: DataFrame containing benchmark data
            metrics: List of network metrics to visualize (if None, auto-detects)
            storage_metric: Optional storage metric to overlay
            save_path: Path to save the visualization (if None, returns the figure)
            
        Returns:
            Matplotlib figure if save_path is None, otherwise None
        """
        if data.empty:
            logger.warning("Cannot visualize empty dataset")
            return None
        
        # Auto-detect network metrics if not provided
        if metrics is None:
            metrics = self._identify_network_metrics(data)
            
            # Limit to a reasonable number of metrics
            if len(metrics) > 5:
                metrics = metrics[:5]
        
        # Filter to metrics that exist in the data
        metrics = [m for m in metrics if m in data.columns]
        
        if not metrics:
            logger.warning("No valid network metrics for visualization")
            return None
        
        # Create the visualization
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Plot network metrics
        for i, metric in enumerate(metrics):
            color = f'C{i}'
            ax1.plot(data.index, data[metric], label=metric, color=color, alpha=0.7)
        
        ax1.set_xlabel('Time' if 'timestamp' not in data.columns else 'Timestamp')
        ax1.set_ylabel('Network Metrics')
        ax1.tick_params(axis='y')
        
        # Add storage metric on secondary axis if provided
        if storage_metric and storage_metric in data.columns:
            ax2 = ax1.twinx()
            ax2.plot(data.index, data[storage_metric], 'r--', label=storage_metric)
            ax2.set_ylabel(storage_metric, color='r')
            ax2.tick_params(axis='y', labelcolor='r')
        
        # Add combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        if storage_metric and storage_metric in data.columns:
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        else:
            ax1.legend(loc='upper left')
        
        # Add title
        plt.title('Network Metrics Time Series')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            return None
        else:
            return fig
    
    def visualize_network_correlation(self, data: pd.DataFrame,
                                     network_metrics: List[str] = None,
                                     storage_metrics: List[str] = None,
                                     save_path: str = None) -> Optional[plt.Figure]:
        """
        Visualize correlation between network and storage metrics.
        
        Args:
            data: DataFrame containing benchmark data
            network_metrics: List of network metrics (if None, auto-detects)
            storage_metrics: List of storage metrics (if None, uses defaults)
            save_path: Path to save the visualization (if None, returns the figure)
            
        Returns:
            Matplotlib figure if save_path is None, otherwise None
        """
        if data.empty:
            logger.warning("Cannot visualize empty dataset")
            return None
        
        # Auto-detect network metrics if not provided
        if network_metrics is None:
            network_metrics = self._identify_network_metrics(data)
            
            # Limit to a reasonable number of metrics
            if len(network_metrics) > 5:
                network_metrics = network_metrics[:5]
        
        # Default storage metrics if not provided
        if storage_metrics is None:
            storage_metrics = ["throughput_MBps", "iops", "latency_ms"]
        
        # Filter to metrics that exist in the data
        network_metrics = [m for m in network_metrics if m in data.columns]
        storage_metrics = [m for m in storage_metrics if m in data.columns]
        
        if not network_metrics or not storage_metrics:
            logger.warning("Insufficient metrics for correlation visualization")
            return None
        
        # Create correlation matrix
        corr_metrics = network_metrics + storage_metrics
        correlation_data = data[corr_metrics].corr()
        
        # Create the visualization
        plt.figure(figsize=(10, 8))
        
        # Create mask for the upper triangle
        mask = np.triu(np.ones_like(correlation_data, dtype=bool))
        
        # Generate heatmap
        sns.heatmap(correlation_data, mask=mask, cmap='coolwarm', 
                   vmin=-1, vmax=1, center=0, annot=True, fmt='.2f',
                   square=True, linewidths=.5)
        
        plt.title('Network-Storage Metric Correlation Matrix')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return None
        else:
            return plt.gcf()
    
    def visualize_bottlenecks(self, data: pd.DataFrame,
                             bottleneck_result: Dict[str, Any],
                             save_path: str = None) -> Optional[plt.Figure]:
        """
        Visualize detected network bottlenecks.
        
        Args:
            data: DataFrame containing benchmark data
            bottleneck_result: Result from _detect_network_bottlenecks method
            save_path: Path to save the visualization (if None, returns the figure)
            
        Returns:
            Matplotlib figure if save_path is None, otherwise None
        """
        if not bottleneck_result.get("detected", False):
            logger.info("No bottlenecks to visualize")
            return None
        
        bottlenecks = bottleneck_result.get("bottlenecks", [])
        if not bottlenecks:
            return None
        
        # Create the visualization
        fig, axes = plt.subplots(len(bottlenecks), 1, figsize=(12, 4*len(bottlenecks)))
        
        # Handle single bottleneck case
        if len(bottlenecks) == 1:
            axes = [axes]
        
        for i, bottleneck in enumerate(bottlenecks):
            metric = bottleneck["metric"]
            threshold = bottleneck["threshold"]
            
            if metric not in data.columns:
                continue
                
            ax = axes[i]
            
            # Plot the metric
            ax.plot(data.index, data[metric], label=metric)
            
            # Add threshold line
            ax.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold ({threshold})')
            
            # Highlight bottleneck periods
            for period in bottleneck.get("high_periods", []):
                start_idx = period["start_idx"]
                end_idx = period["end_idx"]
                
                ax.axvspan(start_idx, end_idx, color='red', alpha=0.2)
            
            ax.set_title(f'Bottleneck: {metric}')
            ax.set_xlabel('Time' if 'timestamp' not in data.columns else 'Timestamp')
            ax.set_ylabel(metric)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            return None
        else:
            return fig
    
    def generate_network_report(self, analysis_result: AnalysisResult) -> Dict[str, Any]:
        """
        Generate a comprehensive report of network analysis results.
        
        Args:
            analysis_result: AnalysisResult from analyze_network_metrics method
            
        Returns:
            Dictionary with report information
        """
        if analysis_result.status != "success":
            return {
                "status": analysis_result.status,
                "error": analysis_result.data.get("error", "Unknown error")
            }
        
        analysis_data = analysis_result.data
        
        # Build summary
        report = {
            "status": "success",
            "timestamp": pd.Timestamp.now().isoformat(),
            "network_metrics_analyzed": len(analysis_data.get("statistics", {})),
            "bottlenecks_detected": analysis_data.get("bottlenecks", {}).get("detected", False),
            "performance_impact": analysis_data.get("summary", {}).get("performance_impact", "none"),
            "recommendations": analysis_data.get("summary", {}).get("recommendations", [])
        }
        
        # Add interface summary if available
        if "interface_utilization" in analysis_data:
            interfaces = analysis_data["interface_utilization"]
            saturated_interfaces = []
            
            for interface, data in interfaces.items():
                if data.get("bottleneck_detected") == True:
                    saturated_interfaces.append({
                        "name": interface,
                        "mean_utilization": data.get("mean_utilization"),
                        "max_utilization": data.get("max_utilization"),
                        "time_over_threshold_seconds": data.get("time_over_threshold_seconds")
                    })
            
            report["network_interfaces"] = {
                "total": len(interfaces),
                "saturated": len(saturated_interfaces),
                "saturated_details": saturated_interfaces
            }
        
        # Add protocol summary if available
        if "protocol_analysis" in analysis_data:
            protocol_data = analysis_data["protocol_analysis"]
            
            report["protocols"] = {
                "dominant_protocol": protocol_data.get("dominant_protocol", "unknown"),
                "tcp_active": protocol_data.get("tcp", {}).get("protocol_active", False),
                "udp_active": protocol_data.get("udp", {}).get("protocol_active", False),
                "tcp_issues": protocol_data.get("tcp", {}).get("retransmit_detected", False),
                "udp_issues": protocol_data.get("udp", {}).get("packet_loss_detected", False)
            }
        
        # Add correlation summary if available
        if "correlation" in analysis_data:
            strong_correlations = analysis_data["correlation"].get("strong_correlations", [])
            
            report["correlations"] = {
                "strong_correlations_count": len(strong_correlations),
                "strong_correlations_details": strong_correlations[:3]  # Top 3 only
            }
        
        # Add bottleneck summary if available
        if "bottlenecks" in analysis_data and analysis_data["bottlenecks"].get("detected", False):
            bottlenecks = analysis_data["bottlenecks"].get("bottlenecks", [])
            
            report["bottlenecks"] = {
                "count": len(bottlenecks),
                "metrics_affected": [b["metric"] for b in bottlenecks],
                "most_severe": max(bottlenecks, key=lambda x: x.get("percent_high", 0))["metric"]
                if bottlenecks else None
            }
        
        # Add overall assessment
        report["overall_assessment"] = analysis_data.get("summary", {}).get(
            "overall_assessment", "No significant network issues detected"
        )
        
        return report
