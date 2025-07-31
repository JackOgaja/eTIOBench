#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Anomaly detection module for benchmark results analysis (Tiered Storage I/O Analysis).

This module provides algorithms to detect anomalies in benchmark time series data,
helping identify unusual performance patterns that may indicate problems.

Author: Jack Ogaja
Date: 2025-06-26
"""

import logging
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.seasonal import seasonal_decompose

from tdiobench.analysis.base_analyzer import BaseAnalyzer
from tdiobench.core.benchmark_analysis import AnalysisResult

logger = logging.getLogger(__name__)


class AnomalyDetector(BaseAnalyzer):
    """
    Anomaly detection for I/O benchmark data.

    This class implements various methods to detect anomalies in benchmark results,
    including statistical methods (z-score), moving averages, and machine learning
    approaches (Isolation Forest).
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the anomaly detector with configuration.

        Args:
            config: Configuration dictionary with anomaly detection parameters
        """
        super().__init__(config)
        self.detection_config = self.config.get("analysis", {}).get("anomaly_detection", {})
        self.enabled = self.detection_config.get("enabled", True)
        self.method = self.detection_config.get("method", "z_score")
        self.threshold = self.detection_config.get("threshold", 3.0)
        self.min_data_points = self.detection_config.get("min_data_points", 20)

        # Contextual anomaly detection settings
        self.contextual = self.detection_config.get("contextual", {}).get("enabled", False)
        self.window_size = self.detection_config.get("contextual", {}).get("window_size", 10)

        # Forecasting-based anomaly detection settings
        self.forecast_enabled = self.detection_config.get("forecasting", {}).get("enabled", False)
        self.forecast_model = self.detection_config.get("forecasting", {}).get("model", "arima")

        logger.debug(
            f"Initialized AnomalyDetector with method={self.method}, threshold={self.threshold}"
        )

    def detect_anomalies(self, data: pd.DataFrame, metrics: List[str] = None) -> AnalysisResult:
        """
        Detect anomalies in benchmark data.

        This is the main entry point for anomaly detection that delegates to the
        appropriate method based on configuration.

        Args:
            data: DataFrame containing benchmark time series data
            metrics: List of metric columns to analyze for anomalies

        Returns:
            AnalysisResult object containing detected anomalies
        """
        if not self.enabled:
            logger.info("Anomaly detection disabled in config")
            result = AnalysisResult(
                analysis_type="anomaly_detection",
                benchmark_id="unknown"
            )
            result.add_overall_result("status", "skipped")
            result.add_overall_result("reason", "Anomaly detection disabled in configuration")
            return result

        if data.empty:
            logger.warning("Cannot detect anomalies in empty dataset")
            result = AnalysisResult(
                analysis_type="anomaly_detection",
                benchmark_id="unknown"
            )
            result.add_overall_result("status", "error")
            result.add_overall_result("error", "Empty dataset provided")
            return result

        if len(data) < self.min_data_points:
            logger.warning(
                f"Insufficient data points for anomaly detection: "
                f"{len(data)} < {self.min_data_points}"
            )
            result = AnalysisResult(
                analysis_type="anomaly_detection",
                benchmark_id="unknown"
            )
            result.add_overall_result("skipped", f"Insufficient data points: {len(data)} < {self.min_data_points}")
            return result

        # Use default metrics if none provided
        if metrics is None:
            metrics = ["throughput_MBps", "iops", "latency_ms"]
            # Filter to only include metrics that exist in the data
            metrics = [m for m in metrics if m in data.columns]

        logger.info(f"Detecting anomalies using {self.method} method for metrics: {metrics}")

        try:
            # Dispatch to appropriate detection method
            if self.method == "z_score":
                result = self._detect_zscore_anomalies(data, metrics)
            elif self.method == "moving_avg":
                result = self._detect_moving_avg_anomalies(data, metrics)
            elif self.method == "isolation_forest":
                result = self._detect_isolation_forest_anomalies(data, metrics)
            elif self.method == "seasonal":
                result = self._detect_seasonal_anomalies(data, metrics)
            else:
                logger.warning(f"Unknown anomaly detection method: {self.method}, using z_score")
                result = self._detect_zscore_anomalies(data, metrics)

            # Add contextual analysis if enabled
            if self.contextual:
                contextual_result = self._detect_contextual_anomalies(data, metrics)
                result["contextual_anomalies"] = contextual_result

            analysis_result = AnalysisResult(
                analysis_type="anomaly_detection",
                benchmark_id="unknown"
            )
            analysis_result.add_overall_result("success", result)
            return analysis_result

        except Exception as e:
            logger.exception(f"Error during anomaly detection: {str(e)}")
            error_result = AnalysisResult(
                analysis_type="anomaly_detection",
                benchmark_id="unknown"
            )
            error_result.add_overall_result("error", {"error": str(e)})
            return error_result

    def _detect_zscore_anomalies(self, data: pd.DataFrame, metrics: List[str]) -> Dict[str, Any]:
        """
        Detect anomalies using Z-score method.

        Identifies values that are more than threshold standard deviations from mean.

        Args:
            data: DataFrame containing benchmark time series data
            metrics: List of metric columns to analyze

        Returns:
            Dictionary with anomaly information
        """
        results = {}
        anomaly_counts = {}

        for metric in metrics:
            if metric not in data.columns:
                logger.warning(f"Metric {metric} not found in data")
                continue

            series = data[metric].dropna()
            if len(series) < 2:  # Need at least 2 points to calculate z-score
                continue

            # Calculate z-scores
            z_scores = np.abs(stats.zscore(series))

            # Identify anomalies
            anomalies_idx = np.where(z_scores > self.threshold)[0]
            anomalies = series.iloc[anomalies_idx]

            # Store results
            anomaly_counts[metric] = len(anomalies)

            if len(anomalies) > 0:
                # Get the timestamps for anomalies if available
                if "timestamp" in data.columns:
                    timestamps = data["timestamp"].iloc[anomalies_idx].tolist()
                else:
                    timestamps = anomalies_idx.tolist()

                results[metric] = {
                    "anomaly_indices": anomalies_idx.tolist(),
                    "anomaly_timestamps": timestamps,
                    "anomaly_values": anomalies.tolist(),
                    "z_scores": z_scores[anomalies_idx].tolist(),
                    "threshold": float(self.threshold),
                    "mean": float(series.mean()),
                    "std": float(series.std()),
                }

        return {
            "method": "z_score",
            "anomaly_counts": anomaly_counts,
            "metrics_analyzed": metrics,
            "anomalies": results,
            "total_anomalies": sum(anomaly_counts.values()),
        }

    def _detect_moving_avg_anomalies(
        self, data: pd.DataFrame, metrics: List[str]
    ) -> Dict[str, Any]:
        """
        Detect anomalies using moving average method.

        Identifies values that deviate significantly from the moving average.

        Args:
            data: DataFrame containing benchmark time series data
            metrics: List of metric columns to analyze

        Returns:
            Dictionary with anomaly information
        """
        results = {}
        anomaly_counts = {}
        window = min(self.window_size, len(data) // 4)
        window = max(window, 3)  # Ensure window is at least 3

        for metric in metrics:
            if metric not in data.columns:
                logger.warning(f"Metric {metric} not found in data")
                continue

            series = data[metric].dropna()
            if len(series) <= window:
                continue

            # Calculate moving average and standard deviation
            rolling_mean = series.rolling(window=window, center=True).mean()
            rolling_std = series.rolling(window=window, center=True).std()

            # Handle NaN values for the first and last window/2 points
            rolling_mean.fillna(method="bfill", inplace=True)
            rolling_mean.fillna(method="ffill", inplace=True)
            rolling_std.fillna(method="bfill", inplace=True)
            rolling_std.fillna(method="ffill", inplace=True)

            # Replace zero std with mean std to avoid division by zero
            mean_std = rolling_std.mean()
            rolling_std = rolling_std.replace(0, mean_std)

            # Calculate deviations
            deviations = np.abs((series - rolling_mean) / rolling_std)

            # Identify anomalies
            anomalies_idx = np.where(deviations > self.threshold)[0]
            anomalies = series.iloc[anomalies_idx]

            # Store results
            anomaly_counts[metric] = len(anomalies)

            if len(anomalies) > 0:
                # Get the timestamps for anomalies if available
                if "timestamp" in data.columns:
                    timestamps = data["timestamp"].iloc[anomalies_idx].tolist()
                else:
                    timestamps = anomalies_idx.tolist()

                results[metric] = {
                    "anomaly_indices": anomalies_idx.tolist(),
                    "anomaly_timestamps": timestamps,
                    "anomaly_values": anomalies.tolist(),
                    "deviations": deviations.iloc[anomalies_idx].tolist(),
                    "threshold": self.threshold,
                    "window_size": window,
                }

        return {
            "method": "moving_avg",
            "anomaly_counts": anomaly_counts,
            "metrics_analyzed": metrics,
            "anomalies": results,
            "total_anomalies": sum(anomaly_counts.values()),
            "window_size": window,
        }

    def _detect_isolation_forest_anomalies(
        self, data: pd.DataFrame, metrics: List[str]
    ) -> Dict[str, Any]:
        """
        Detect anomalies using Isolation Forest algorithm.

        Uses machine learning to identify anomalies based on isolation in feature space.

        Args:
            data: DataFrame containing benchmark time series data
            metrics: List of metric columns to analyze

        Returns:
            Dictionary with anomaly information
        """
        results = {}
        anomaly_counts = {}

        # Only use existing metrics from the data
        available_metrics = [m for m in metrics if m in data.columns]

        if not available_metrics:
            logger.warning("No valid metrics found for isolation forest analysis")
            return {
                "method": "isolation_forest",
                "error": "No valid metrics found",
                "total_anomalies": 0,
            }

        try:
            # Prepare data
            X = data[available_metrics].dropna()
            if len(X) < self.min_data_points:
                return {
                    "method": "isolation_forest",
                    "error": f"Insufficient data points after removing NaN: {len(X)}",
                    "total_anomalies": 0,
                }

            # Initialize and fit the model
            contamination = min(0.1, 1.0 / len(X))  # Auto-adjust contamination
            clf = IsolationForest(
                n_estimators=100, max_samples="auto", contamination=contamination, random_state=42
            )
            clf.fit(X)

            # Predict anomalies
            # -1 for anomalies, 1 for normal points
            y_pred = clf.predict(X)
            anomalies_idx = np.where(y_pred == -1)[0]
            anomaly_scores = clf.decision_function(X)  # Lower score = more anomalous

            # Get overall results
            overall_count = len(anomalies_idx)

            if overall_count > 0:
                # Get the timestamps for anomalies if available
                if "timestamp" in data.columns:
                    timestamps = data.loc[X.index, "timestamp"].iloc[anomalies_idx].tolist()
                else:
                    timestamps = X.index[anomalies_idx].tolist()

                # Store overall results
                results["overall"] = {
                    "anomaly_indices": X.index[anomalies_idx].tolist(),
                    "anomaly_timestamps": timestamps,
                    "anomaly_scores": anomaly_scores[anomalies_idx].tolist(),
                }

                # For each metric, store the values at anomaly points
                for metric in available_metrics:
                    anomaly_values = X.loc[X.index[anomalies_idx], metric].tolist()
                    results[metric] = {"anomaly_values": anomaly_values}
                    anomaly_counts[metric] = overall_count

            return {
                "method": "isolation_forest",
                "anomaly_counts": anomaly_counts,
                "metrics_analyzed": available_metrics,
                "anomalies": results,
                "total_anomalies": overall_count,
                "contamination": contamination,
            }

        except Exception as e:
            logger.exception(f"Error in isolation forest anomaly detection: {str(e)}")
            return {"method": "isolation_forest", "error": str(e), "total_anomalies": 0}

    def _detect_seasonal_anomalies(self, data: pd.DataFrame, metrics: List[str]) -> Dict[str, Any]:
        """
        Detect anomalies by decomposing time series into seasonal components.

        Identifies anomalies based on residual components after removing trend and seasonality.

        Args:
            data: DataFrame containing benchmark time series data
            metrics: List of metric columns to analyze

        Returns:
            Dictionary with anomaly information
        """
        results = {}
        anomaly_counts = {}

        # Need at least 2 * period + 1 data points for seasonal decomposition
        min_points = 2 * 10 + 1  # Assuming minimum period of 10

        if len(data) < min_points:
            logger.warning(
                f"Insufficient data for seasonal decomposition: {len(data)} < {min_points}"
            )
            return {
                "method": "seasonal",
                "error": f"Insufficient data points: {len(data)} < {min_points}",
                "total_anomalies": 0,
            }

        for metric in metrics:
            if metric not in data.columns:
                logger.warning(f"Metric {metric} not found in data")
                continue

            series = data[metric].dropna()
            if len(series) < min_points:
                continue

            try:
                # Try to determine period automatically or use default
                # For simplicity, we use a fixed period here
                period = min(10, len(series) // 3)

                # Decompose the time series
                decomposition = seasonal_decompose(series, model="additive", period=period)

                # Extract residuals
                residuals = decomposition.resid.dropna()

                # Calculate z-scores of residuals
                z_scores = np.abs(stats.zscore(residuals))

                # Identify anomalies
                anomalies_idx = np.where(z_scores > self.threshold)[0]

                # Map back to original indices
                original_idx = residuals.index[anomalies_idx]
                anomalies = series.loc[original_idx]

                # Store results
                anomaly_counts[metric] = len(anomalies)

                if len(anomalies) > 0:
                    # Get the timestamps for anomalies if available
                    if "timestamp" in data.columns:
                        timestamps = data.loc[original_idx, "timestamp"].tolist()
                    else:
                        timestamps = original_idx.tolist()

                    results[metric] = {
                        "anomaly_indices": original_idx.tolist(),
                        "anomaly_timestamps": timestamps,
                        "anomaly_values": anomalies.tolist(),
                        "residuals": residuals.loc[original_idx].tolist(),
                        "z_scores": z_scores[anomalies_idx].tolist(),
                        "threshold": self.threshold,
                        "period": period,
                    }

            except Exception as e:
                logger.warning(f"Error in seasonal decomposition for {metric}: {str(e)}")
                continue

        return {
            "method": "seasonal",
            "anomaly_counts": anomaly_counts,
            "metrics_analyzed": metrics,
            "anomalies": results,
            "total_anomalies": sum(anomaly_counts.values()),
        }

    def _detect_contextual_anomalies(
        self, data: pd.DataFrame, metrics: List[str]
    ) -> Dict[str, Any]:
        """
        Detect contextual anomalies based on surrounding data points.

        This method identifies anomalies that are normal globally but anomalous in context.

        Args:
            data: DataFrame containing benchmark time series data
            metrics: List of metric columns to analyze

        Returns:
            Dictionary with contextual anomaly information
        """
        results = {}
        anomaly_counts = {}
        window = self.window_size

        for metric in metrics:
            if metric not in data.columns:
                logger.warning(f"Metric {metric} not found in data")
                continue

            series = data[metric].dropna()
            if len(series) <= window * 2:
                continue

            # Sliding window analysis
            contextual_anomalies = []
            contextual_indices = []
            contextual_scores = []

            for i in range(window, len(series) - window):
                # Get local window
                local_window = series.iloc[i - window: i + window + 1]
                center_value = local_window.iloc[window]

                # Exclude the center point from local statistics
                local_context = pd.concat(
                    [local_window.iloc[:window], local_window.iloc[window + 1:]]
                )

                # Calculate local statistics
                local_mean = local_context.mean()
                local_std = local_context.std()

                # Avoid division by zero
                if local_std == 0:
                    local_std = series.std() or 1.0

                # Calculate local z-score
                local_z = abs((center_value - local_mean) / local_std)

                # Check if anomalous in local context
                if local_z > self.threshold:
                    # Check if this would be anomalous globally
                    global_z = abs((center_value - series.mean()) / series.std())

                    # Only consider contextual anomalies (anomalous locally but not globally)
                    if global_z < self.threshold:
                        contextual_anomalies.append(center_value)
                        contextual_indices.append(i)
                        contextual_scores.append(local_z)

            # Store results
            anomaly_counts[metric] = len(contextual_anomalies)

            if contextual_anomalies:
                # Get the timestamps for anomalies if available
                if "timestamp" in data.columns:
                    timestamps = data["timestamp"].iloc[contextual_indices].tolist()
                else:
                    timestamps = contextual_indices

                results[metric] = {
                    "anomaly_indices": contextual_indices,
                    "anomaly_timestamps": timestamps,
                    "anomaly_values": contextual_anomalies,
                    "contextual_scores": contextual_scores,
                    "window_size": window,
                    "threshold": self.threshold,
                }

        return {
            "method": "contextual",
            "anomaly_counts": anomaly_counts,
            "metrics_analyzed": metrics,
            "anomalies": results,
            "total_anomalies": sum(anomaly_counts.values()),
            "window_size": window,
        }

    def visualize_anomalies(
        self,
        data: pd.DataFrame,
        analysis_result: AnalysisResult,
        metric: str = None,
        save_path: str = None,
    ) -> Optional[plt.Figure]:
        """
        Visualize detected anomalies in the time series data.

        Args:
            data: DataFrame containing benchmark time series data
            analysis_result: AnalysisResult from detect_anomalies method
            metric: Specific metric to visualize (if None, uses first available)
            save_path: Path to save the visualization (if None, returns the figure)

        Returns:
            Matplotlib figure if save_path is None, otherwise None
        """
        if analysis_result.status != "success":
            logger.warning(f"Cannot visualize unsuccessful analysis: {analysis_result.status}")
            return None

        anomaly_data = analysis_result.data

        # If no specific metric provided, use first available with anomalies
        if metric is None:
            for m in anomaly_data.get("metrics_analyzed", []):
                if m in anomaly_data.get("anomalies", {}) and anomaly_data["anomalies"][m]:
                    metric = m
                    break

        if not metric or metric not in data.columns:
            logger.warning("No valid metric for visualization")
            return None

        # Check if anomalies exist for this metric
        metric_anomalies = anomaly_data.get("anomalies", {}).get(metric, {})
        if not metric_anomalies:
            logger.info(f"No anomalies detected for {metric}")
            return None

        # Create the visualization
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot the time series
        x = data.index if "timestamp" not in data.columns else data["timestamp"]
        ax.plot(x, data[metric], label=metric, color="blue", alpha=0.7)

        # Plot anomalies
        indices = metric_anomalies.get("anomaly_indices", [])
        timestamps = metric_anomalies.get("anomaly_timestamps", indices)
        values = metric_anomalies.get("anomaly_values", [])

        ax.scatter(timestamps, values, color="red", s=50, label="Anomalies", zorder=5)

        # Add labels and title
        method = anomaly_data.get("method", "unknown")
        ax.set_title(f"Anomaly Detection ({method}) - {metric}")
        ax.set_xlabel("Time" if "timestamp" not in data.columns else "Timestamp")
        ax.set_ylabel(metric)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add threshold information
        threshold = metric_anomalies.get("threshold", self.threshold)
        ax.text(
            0.02,
            0.95,
            f"Threshold: {threshold:.2f}",
            transform=ax.transAxes,
            bbox=dict(facecolor="white", alpha=0.8),
        )

        # Add total anomalies count
        ax.text(
            0.02,
            0.90,
            f"Anomalies: {len(values)}",
            transform=ax.transAxes,
            bbox=dict(facecolor="white", alpha=0.8),
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            return None
        else:
            return fig

    def generate_anomaly_report(
        self, data: pd.DataFrame, analysis_result: AnalysisResult
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive report of detected anomalies.

        Args:
            data: DataFrame containing benchmark time series data
            analysis_result: AnalysisResult from detect_anomalies method

        Returns:
            Dictionary with report information
        """
        if analysis_result.status != "success":
            return {
                "status": analysis_result.status,
                "error": analysis_result.data.get("error", "Unknown error"),
            }

        anomaly_data = analysis_result.data
        method = anomaly_data.get("method", "unknown")
        metrics_analyzed = anomaly_data.get("metrics_analyzed", [])
        total_anomalies = anomaly_data.get("total_anomalies", 0)

        # Build summary
        summary = {
            "status": "success",
            "method": method,
            "metrics_analyzed": metrics_analyzed,
            "total_anomalies": total_anomalies,
            "threshold": self.threshold,
            "data_points": len(data),
            "anomaly_percentage": (total_anomalies / len(data) * 100) if len(data) > 0 else 0,
            "metrics_summary": {},
        }

        # Add per-metric details
        for metric in metrics_analyzed:
            metric_anomalies = anomaly_data.get("anomalies", {}).get(metric, {})
            anomaly_count = anomaly_data.get("anomaly_counts", {}).get(metric, 0)

            if metric_anomalies and anomaly_count > 0:
                # Calculate statistics for this metric
                values = data[metric].dropna()
                anomaly_values = metric_anomalies.get("anomaly_values", [])

                metric_summary = {
                    "count": anomaly_count,
                    "percentage": (anomaly_count / len(values) * 100) if len(values) > 0 else 0,
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "anomaly_min": float(np.min(anomaly_values)) if anomaly_values else None,
                    "anomaly_max": float(np.max(anomaly_values)) if anomaly_values else None,
                    "anomaly_mean": float(np.mean(anomaly_values)) if anomaly_values else None,
                }

                # Add first and last anomaly timestamps if available
                indices = metric_anomalies.get("anomaly_indices", [])
                if indices and "timestamp" in data.columns:
                    metric_summary["first_anomaly_time"] = data["timestamp"].iloc[indices[0]]
                    metric_summary["last_anomaly_time"] = data["timestamp"].iloc[indices[-1]]

                summary["metrics_summary"][metric] = metric_summary

        # Add contextual anomalies if available
        if "contextual_anomalies" in anomaly_data:
            contextual_data = anomaly_data["contextual_anomalies"]
            contextual_total = sum(contextual_data.get("anomaly_counts", {}).values())

            summary["contextual"] = {
                "total_anomalies": contextual_total,
                "window_size": contextual_data.get("window_size", self.window_size),
                "metrics_summary": {},
            }

            # Add per-metric contextual details
            for metric in metrics_analyzed:
                metric_anomalies = contextual_data.get("anomalies", {}).get(metric, {})
                anomaly_count = contextual_data.get("anomaly_counts", {}).get(metric, 0)

                if metric_anomalies and anomaly_count > 0:
                    summary["contextual"]["metrics_summary"][metric] = {
                        "count": anomaly_count,
                        "percentage": (anomaly_count / len(data) * 100) if len(data) > 0 else 0,
                    }

        return summary
