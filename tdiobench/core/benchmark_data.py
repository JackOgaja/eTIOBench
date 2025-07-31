#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark data module for storage benchmark suite.

This module provides classes to represent, store, and process benchmark data
collected during benchmark execution. It enables flexible data manipulation,
filtering, and aggregation for analysis.

Author: JackOgaja
Date: 2025-06-30 22:02:03
"""

import json
import logging
import os
import pickle
import uuid
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


class TimeSeriesData:
    """
    Specialized class for handling time series benchmark data.

    This class provides methods for storing, manipulating, and analyzing
    time-based measurement data from benchmark runs. It supports advanced
    operations like filtering, aggregation, and statistical analysis.
    """

    def __init__(
        self,
        data: Optional[Union[pd.DataFrame, List[Dict[str, Any]]]] = None,
        timestamp_column: str = "timestamp",
    ):
        """
        Initialize a new TimeSeriesData instance.

        Args:
            data: Initial time series data as DataFrame or list of dictionaries
            timestamp_column: Name of the timestamp column
        """
        self._timestamp_column = timestamp_column
        self._df = None

        if data is not None:
            if isinstance(data, pd.DataFrame):
                self._df = data.copy()
                # Ensure timestamp column is datetime type
                if self._timestamp_column in self._df.columns:
                    self._df[self._timestamp_column] = pd.to_datetime(
                        self._df[self._timestamp_column]
                    )
                    # Set as index if not already
                    if self._df.index.name != self._timestamp_column:
                        self._df = self._df.set_index(self._timestamp_column)
            elif isinstance(data, list) and all(isinstance(item, dict) for item in data):
                self._df = pd.DataFrame(data)
                # Ensure timestamp column is datetime type
                if self._timestamp_column in self._df.columns:
                    self._df[self._timestamp_column] = pd.to_datetime(
                        self._df[self._timestamp_column]
                    )
                    # Set as index if not already
                    if self._df.index.name != self._timestamp_column:
                        self._df = self._df.set_index(self._timestamp_column)
            else:
                raise TypeError("Data must be a pandas DataFrame or list of dictionaries")
        else:
            # Create empty DataFrame with timestamp index
            self._df = pd.DataFrame()

        logger.debug(f"Initialized TimeSeriesData with {len(self._df)} data points")

    def add_data_point(self, timestamp: Union[str, datetime], metrics: Dict[str, Any]) -> None:
        """
        Add a single data point to the time series.

        Args:
            timestamp: Time of measurement
            metrics: Dictionary of metric names and values
        """
        # Convert timestamp to pandas Timestamp
        if isinstance(timestamp, str):
            timestamp = pd.to_datetime(timestamp)

        # Create a Series for the new data point
        new_point = pd.Series(metrics, name=timestamp)

        # Append to DataFrame
        if self._df.empty:
            self._df = pd.DataFrame([new_point])
            self._df.index.name = self._timestamp_column
        else:
            self._df = pd.concat([self._df, pd.DataFrame([new_point])])

        # Sort by timestamp
        self._df = self._df.sort_index()

        logger.debug(f"Added data point at {timestamp} with {len(metrics)} metrics")

    def add_data_points(self, data_points: List[Dict[str, Any]]) -> None:
        """
        Add multiple data points to the time series.

        Args:
            data_points: List of dictionaries with timestamp and metrics
        """
        if not data_points:
            logger.warning("No data points provided")
            return

        # Create a temporary DataFrame
        temp_df = pd.DataFrame(data_points)

        # Ensure timestamp column exists
        if self._timestamp_column not in temp_df.columns:
            raise ValueError(f"Data points must contain '{self._timestamp_column}' column")

        # Convert timestamp to datetime and set as index
        temp_df[self._timestamp_column] = pd.to_datetime(temp_df[self._timestamp_column])
        temp_df = temp_df.set_index(self._timestamp_column)

        # Combine with existing data
        if self._df.empty:
            self._df = temp_df
        else:
            self._df = pd.concat([self._df, temp_df])
            self._df = self._df.sort_index()

        logger.debug(f"Added {len(data_points)} data points to time series")

    def get_dataframe(self, reset_index: bool = False) -> pd.DataFrame:
        """
        Get the time series data as a pandas DataFrame.

        Args:
            reset_index: If True, reset the DataFrame index

        Returns:
            DataFrame containing time series data
        """
        if self._df is None or self._df.empty:
            return pd.DataFrame()

        df = self._df.copy()
        return df.reset_index() if reset_index else df

    def filter(
        self,
        start_time: Optional[Union[str, datetime]] = None,
        end_time: Optional[Union[str, datetime]] = None,
        conditions: Optional[Dict[str, Any]] = None,
    ) -> "TimeSeriesData":
        """
        Create a new TimeSeriesData with filtered data.

        Args:
            start_time: Start time for filtering (inclusive)
            end_time: End time for filtering (inclusive)
            conditions: Dictionary of column-value conditions

        Returns:
            New TimeSeriesData instance with filtered data
        """
        if self._df is None or self._df.empty:
            logger.warning("No time series data to filter")
            return TimeSeriesData()

        # Start with a copy of the original data
        filtered_df = self._df.copy()

        # Convert string times to datetime if needed
        if isinstance(start_time, str):
            start_time = pd.to_datetime(start_time)
        if isinstance(end_time, str):
            end_time = pd.to_datetime(end_time)

        # Apply time filters
        if start_time is not None:
            filtered_df = filtered_df[filtered_df.index >= start_time]
        if end_time is not None:
            filtered_df = filtered_df[filtered_df.index <= end_time]

        # Apply column conditions
        if conditions:
            for column, value in conditions.items():
                if column in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df[column] == value]

        # Create new TimeSeriesData with filtered results
        return TimeSeriesData(filtered_df)

    def aggregate(self, freq: str = "1min", method: str = "mean") -> "TimeSeriesData":
        """
        Aggregate time series data by a specified frequency.

        Args:
            freq: Pandas frequency string ('1min', '1h', '1d', etc.)
            method: Aggregation method ('mean', 'median', 'min', 'max', 'sum', 'count')

        Returns:
            New TimeSeriesData instance with aggregated data
        """
        if self._df is None or self._df.empty:
            logger.warning("No time series data to aggregate")
            return TimeSeriesData()

        # Apply resampling with specified aggregation
        if method == "mean":
            aggregated_df = self._df.resample(freq).mean()
        elif method == "median":
            aggregated_df = self._df.resample(freq).median()
        elif method == "min":
            aggregated_df = self._df.resample(freq).min()
        elif method == "max":
            aggregated_df = self._df.resample(freq).max()
        elif method == "sum":
            aggregated_df = self._df.resample(freq).sum()
        elif method == "count":
            aggregated_df = self._df.resample(freq).count()
        else:
            logger.warning(f"Unknown aggregation method: {method}")
            return self

        return TimeSeriesData(aggregated_df)

    def smooth(self, window: int = 5, method: str = "rolling") -> "TimeSeriesData":
        """
        Apply smoothing to time series data.

        Args:
            window: Window size for smoothing
            method: Smoothing method ('rolling', 'ewm', 'savgol')

        Returns:
            New TimeSeriesData instance with smoothed data
        """
        if self._df is None or self._df.empty:
            logger.warning("No time series data to smooth")
            return TimeSeriesData()

        # Apply smoothing based on method
        if method == "rolling":
            # Simple rolling window average
            smoothed_df = self._df.rolling(window=window, center=True).mean()
        elif method == "ewm":
            # Exponentially weighted moving average
            smoothed_df = self._df.ewm(span=window).mean()
        elif method == "savgol":
            # Savitzky-Golay filter
            try:
                from scipy.signal import savgol_filter

                # Apply filter to each numeric column
                smoothed_df = self._df.copy()
                for col in self._df.select_dtypes(include=["number"]).columns:
                    # Ensure enough data points for filter
                    if len(self._df) > window:
                        # Window length must be odd and polyorder < window_length
                        if window % 2 == 0:
                            window += 1
                        polyorder = min(3, window - 1)
                        smoothed_df[col] = savgol_filter(self._df[col].values, window, polyorder)
            except ImportError:
                logger.warning("scipy not available, falling back to rolling average")
                smoothed_df = self._df.rolling(window=window, center=True).mean()
        else:
            logger.warning(f"Unknown smoothing method: {method}")
            return self

        return TimeSeriesData(smoothed_df)

    def interpolate(self, method: str = "linear", limit: Optional[int] = None) -> "TimeSeriesData":
        """
        Interpolate missing values in time series.

        Args:
            method: Interpolation method ('linear', 'time', 'cubic', etc.)
            limit: Maximum number of consecutive NaNs to fill

        Returns:
            New TimeSeriesData instance with interpolated data
        """
        if self._df is None or self._df.empty:
            logger.warning("No time series data to interpolate")
            return TimeSeriesData()

        # Apply interpolation
        interpolated_df = self._df.interpolate(method=method, limit=limit)

        return TimeSeriesData(interpolated_df)

    def resample(self, freq: str, method: str = "ffill") -> "TimeSeriesData":
        """
        Resample time series to a new frequency.

        Args:
            freq: New frequency string ('1s', '1min', '1h', etc.)
            method: Fill method for new points ('ffill', 'bfill', 'nearest', etc.)

        Returns:
            New TimeSeriesData instance with resampled data
        """
        if self._df is None or self._df.empty:
            logger.warning("No time series data to resample")
            return TimeSeriesData()

        # Resample to new frequency
        resampled_df = self._df.resample(freq).asfreq()

        # Fill missing values based on method
        if method == "ffill":
            resampled_df = resampled_df.fillna(method="ffill")
        elif method == "bfill":
            resampled_df = resampled_df.fillna(method="bfill")
        elif method == "nearest":
            resampled_df = resampled_df.fillna(method="nearest")
        # Other methods like interpolation could be handled here

        return TimeSeriesData(resampled_df)

    def detect_outliers(
        self, columns: Optional[List[str]] = None, method: str = "zscore", threshold: float = 3.0
    ) -> pd.DataFrame:
        """
        Detect outliers in time series data.

        Args:
            columns: List of columns to check (if None, use all numeric columns)
            method: Detection method ('zscore', 'iqr', 'mad')
            threshold: Threshold for outlier detection

        Returns:
            DataFrame with boolean mask indicating outliers
        """
        if self._df is None or self._df.empty:
            logger.warning("No time series data for outlier detection")
            return pd.DataFrame()

        # If no columns specified, use all numeric columns
        if columns is None:
            columns = self._df.select_dtypes(include=["number"]).columns.tolist()
        else:
            # Filter to columns that exist and are numeric
            columns = [
                c
                for c in columns
                if c in self._df.columns and pd.api.types.is_numeric_dtype(self._df[c])
            ]

        if not columns:
            logger.warning("No valid numeric columns for outlier detection")
            return pd.DataFrame()

        # Create result DataFrame with same index
        result_df = pd.DataFrame(index=self._df.index)

        # Detect outliers based on method
        if method == "zscore":
            # Z-score method
            for col in columns:
                series = self._df[col]
                mean = series.mean()
                std = series.std()

                if std == 0:  # Avoid division by zero
                    result_df[col] = False
                    continue

                z_scores = (series - mean) / std
                result_df[col] = abs(z_scores) > threshold

        elif method == "iqr":
            # Interquartile range method
            for col in columns:
                series = self._df[col]
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1

                if iqr == 0:  # Avoid using zero IQR
                    result_df[col] = False
                    continue

                lower_bound = q1 - threshold * iqr
                upper_bound = q3 + threshold * iqr

                result_df[col] = (series < lower_bound) | (series > upper_bound)

        elif method == "mad":
            # Median absolute deviation method
            for col in columns:
                series = self._df[col]
                median = series.median()
                mad = (series - median).abs().median()

                if mad == 0:  # Avoid division by zero
                    result_df[col] = False
                    continue

                result_df[col] = (series - median).abs() > threshold * mad

        else:
            logger.warning(f"Unknown outlier detection method: {method}")
            return pd.DataFrame()

        # Add a combined column that's True if any metric is an outlier
        if columns:
            result_df["any_outlier"] = result_df[columns].any(axis=1)

        return result_df

    def calculate_statistics(
        self, columns: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate descriptive statistics for specified columns.

        Args:
            columns: List of columns to analyze (if None, use all numeric columns)

        Returns:
            Dictionary of statistics for each column
        """
        if self._df is None or self._df.empty:
            logger.warning("No time series data for statistics")
            return {}

        # If no columns specified, use all numeric columns
        if columns is None:
            columns = self._df.select_dtypes(include=["number"]).columns.tolist()
        else:
            # Filter to columns that exist and are numeric
            columns = [
                c
                for c in columns
                if c in self._df.columns and pd.api.types.is_numeric_dtype(self._df[c])
            ]

        if not columns:
            logger.warning("No valid numeric columns for statistics")
            return {}

        statistics = {}

        for column in columns:
            series = self._df[column].dropna()

            if len(series) < 2:
                continue

            # Calculate statistics
            stats_dict = {
                "count": int(len(series)),
                "mean": float(series.mean()),
                "std": float(series.std()),
                "min": float(series.min()),
                "25%": float(series.quantile(0.25)),
                "50%": float(series.median()),
                "75%": float(series.quantile(0.75)),
                "max": float(series.max()),
                "cv": float(series.std() / series.mean()) if series.mean() != 0 else float("nan"),
            }

            # Add more percentiles
            for p in [1, 5, 95, 99, 99.9]:
                stats_dict[f"{p}%"] = float(series.quantile(p / 100))

            statistics[column] = stats_dict

        return statistics

    def analyze_trend(self, column: str, method: str = "linear") -> Dict[str, Any]:
        """
        Analyze trend in a time series column.

        Args:
            column: Column to analyze
            method: Trend analysis method ('linear', 'ols', 'seasonal')

        Returns:
            Dictionary with trend analysis results
        """
        if self._df is None or self._df.empty:
            logger.warning("No time series data for trend analysis")
            return {}

        if column not in self._df.columns:
            logger.warning(f"Column '{column}' not found in time series data")
            return {}

        # Ensure data is numeric
        if not pd.api.types.is_numeric_dtype(self._df[column]):
            logger.warning(f"Column '{column}' must be numeric for trend analysis")
            return {}

        # Basic result dictionary
        result = {"column": column, "method": method, "data_points": len(self._df)}

        if method == "linear" or method == "ols":
            # Simple linear regression
            try:
                # Convert index to numeric for regression
                if isinstance(self._df.index, pd.DatetimeIndex):
                    x = np.array(range(len(self._df))).reshape(-1, 1)
                else:
                    x = np.array(self._df.index).reshape(-1, 1)

                y = self._df[column].values

                # Handle NaN values
                mask = ~np.isnan(y)
                x_clean = x[mask]
                y_clean = y[mask]

                if len(x_clean) < 2:
                    logger.warning(f"Not enough valid data points for trend analysis of '{column}'")
                    return {}

                # Fit linear model
                from scipy import stats

                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    x_clean.flatten(), y_clean
                )

                result.update(
                    {
                        "slope": float(slope),
                        "intercept": float(intercept),
                        "r_squared": float(r_value**2),
                        "p_value": float(p_value),
                        "std_error": float(std_err),
                        "is_significant": p_value < 0.05,
                        "trend_direction": (
                            "increasing" if slope > 0 else "decreasing" if slope < 0 else "flat"
                        ),
                    }
                )

                # Add trend line data
                x_pred = np.array(range(len(self._df))).reshape(-1, 1)
                y_pred = intercept + slope * x_pred.flatten()

                result["trend_line"] = {"x": x_pred.flatten().tolist(), "y": y_pred.tolist()}

            except Exception as e:
                logger.warning(f"Error in trend analysis: {str(e)}")
                return {}

        elif method == "seasonal":
            # Seasonal decomposition
            try:
                from statsmodels.tsa.seasonal import seasonal_decompose

                # Ensure we have enough data points
                if len(self._df) < 4:
                    logger.warning("Not enough data points for seasonal decomposition")
                    return {}

                # Handle missing values by filling with interpolation
                series = self._df[column].interpolate()

                # Try to determine period automatically or use default
                if len(series) >= 14:
                    period = 7  # Weekly seasonality as default for longer series
                else:
                    period = 2  # Minimal period for short series

                # Perform decomposition
                decomposition = seasonal_decompose(
                    series, model="additive", period=period, extrapolate_trend="freq"
                )

                result.update(
                    {
                        "trend": decomposition.trend.tolist(),
                        "seasonal": decomposition.seasonal.tolist(),
                        "residual": decomposition.resid.tolist(),
                        "period": period,
                    }
                )

                # Calculate trend slope
                trend_series = decomposition.trend.dropna()
                if len(trend_series) >= 2:
                    x = np.array(range(len(trend_series))).reshape(-1, 1)
                    y = trend_series.values

                    from scipy import stats

                    slope, intercept, r_value, p_value, std_err = stats.linregress(x.flatten(), y)

                    result.update(
                        {
                            "trend_slope": float(slope),
                            "trend_r_squared": float(r_value**2),
                            "trend_direction": (
                                "increasing" if slope > 0 else "decreasing" if slope < 0 else "flat"
                            ),
                        }
                    )

            except Exception as e:
                logger.warning(f"Error in seasonal decomposition: {str(e)}")
                return {}
        else:
            logger.warning(f"Unknown trend analysis method: {method}")
            return {}

        return result

    def detect_anomalies(
        self, column: str, method: str = "zscore", threshold: float = 3.0
    ) -> pd.Series:
        """
        Detect anomalies in a time series column.

        Args:
            column: Column to analyze
            method: Anomaly detection method ('zscore', 'iqr', 'moving_avg')
            threshold: Threshold for anomaly detection

        Returns:
            Series with boolean values indicating anomalies
        """
        if self._df is None or self._df.empty:
            logger.warning("No time series data for anomaly detection")
            return pd.Series()

        if column not in self._df.columns:
            logger.warning(f"Column '{column}' not found in time series data")
            return pd.Series()

        # Ensure data is numeric
        if not pd.api.types.is_numeric_dtype(self._df[column]):
            logger.warning(f"Column '{column}' must be numeric for anomaly detection")
            return pd.Series()

        series = self._df[column]

        if method == "zscore":
            # Z-score method
            mean = series.mean()
            std = series.std()

            if std == 0:  # Avoid division by zero
                return pd.Series(False, index=self._df.index)

            z_scores = (series - mean) / std
            return pd.Series(abs(z_scores) > threshold, index=self._df.index)

        elif method == "iqr":
            # Interquartile range method
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1

            if iqr == 0:  # Avoid using zero IQR
                return pd.Series(False, index=self._df.index)

            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr

            return pd.Series((series < lower_bound) | (series > upper_bound), index=self._df.index)

        elif method == "moving_avg":
            # Moving average method
            window = min(10, len(series) // 3)  # Adaptive window size
            if window < 2:
                window = 2

            rolling_mean = series.rolling(window=window, center=True).mean()
            rolling_std = series.rolling(window=window, center=True).std()

            # Handle edge cases and missing values
            rolling_mean = rolling_mean.fillna(method="bfill").fillna(method="ffill")
            rolling_std = rolling_std.fillna(
                rolling_mean.std()
            )  # Use overall std for missing values

            # Avoid division by zero
            rolling_std = rolling_std.replace(0, series.std() or 1.0)

            # Calculate deviations
            deviations = abs(series - rolling_mean) / rolling_std

            return pd.Series(deviations > threshold, index=self._df.index)

        else:
            logger.warning(f"Unknown anomaly detection method: {method}")
            return pd.Series(False, index=self._df.index)

    def calculate_metric(self, column: str, metric: str) -> Optional[float]:
        """
        Calculate a specific metric for a column.

        Args:
            column: Column to analyze
            metric: Metric to calculate ('mean', 'median', 'min', 'max', 'std', etc.)

        Returns:
            Calculated metric value or None if error
        """
        if self._df is None or self._df.empty:
            logger.warning("No time series data for metric calculation")
            return None

        if column not in self._df.columns:
            logger.warning(f"Column '{column}' not found in time series data")
            return None

        # Ensure data is numeric
        if not pd.api.types.is_numeric_dtype(self._df[column]):
            logger.warning(f"Column '{column}' must be numeric for metric calculation")
            return None

        series = self._df[column].dropna()

        if series.empty:
            logger.warning(f"No valid data in column '{column}'")
            return None

        # Calculate requested metric
        if metric == "mean":
            return float(series.mean())
        elif metric == "median":
            return float(series.median())
        elif metric == "min":
            return float(series.min())
        elif metric == "max":
            return float(series.max())
        elif metric == "std":
            return float(series.std())
        elif metric == "var":
            return float(series.var())
        elif metric == "sum":
            return float(series.sum())
        elif metric == "count":
            return int(len(series))
        elif metric in ["p1", "1%", "percentile_1"]:
            return float(series.quantile(0.01))
        elif metric in ["p5", "5%", "percentile_5"]:
            return float(series.quantile(0.05))
        elif metric in ["p95", "95%", "percentile_95"]:
            return float(series.quantile(0.95))
        elif metric in ["p99", "99%", "percentile_99"]:
            return float(series.quantile(0.99))
        elif metric == "iqr":
            return float(series.quantile(0.75) - series.quantile(0.25))
        elif metric == "range":
            return float(series.max() - series.min())
        elif metric == "cv":  # Coefficient of variation
            mean = series.mean()
            if mean == 0:
                return None
            return float(series.std() / mean)
        else:
            logger.warning(f"Unknown metric: {metric}")
            return None

    def merge(self, other: "TimeSeriesData") -> "TimeSeriesData":
        """
        Merge this TimeSeriesData with another instance.

        Args:
            other: Another TimeSeriesData instance

        Returns:
            New TimeSeriesData instance with merged data
        """
        if not isinstance(other, TimeSeriesData):
            raise TypeError("Can only merge with another TimeSeriesData instance")

        # Get DataFrames with reset indices
        self_df = self.get_dataframe(reset_index=True)
        other_df = other.get_dataframe(reset_index=True)

        # Handle empty DataFrames
        if self_df.empty:
            return TimeSeriesData(other_df)
        if other_df.empty:
            return TimeSeriesData(self_df)

        # Ensure timestamp column exists in both
        if (
            self._timestamp_column not in self_df.columns
            or self._timestamp_column not in other_df.columns
        ):
            raise ValueError(f"Both DataFrames must have '{self._timestamp_column}' column")

        # Concatenate DataFrames
        merged_df = pd.concat([self_df, other_df], ignore_index=True)

        # Sort by timestamp and remove duplicates
        merged_df = merged_df.sort_values(self._timestamp_column)
        merged_df = merged_df.drop_duplicates(subset=[self._timestamp_column], keep="last")

        return TimeSeriesData(merged_df, timestamp_column=self._timestamp_column)

    def apply(self, func: Callable, columns: Optional[List[str]] = None) -> "TimeSeriesData":
        """
        Apply a function to selected columns.

        Args:
            func: Function to apply
            columns: List of columns to apply function to (if None, use all numeric columns)

        Returns:
            New TimeSeriesData instance with transformed data
        """
        if self._df is None or self._df.empty:
            logger.warning("No time series data to transform")
            return TimeSeriesData()

        # If no columns specified, use all numeric columns
        if columns is None:
            columns = self._df.select_dtypes(include=["number"]).columns.tolist()
        else:
            # Filter to columns that exist
            columns = [c for c in columns if c in self._df.columns]

        if not columns:
            logger.warning("No valid columns for transformation")
            return TimeSeriesData(self._df.copy())

        # Apply function to specified columns
        transformed_df = self._df.copy()
        for column in columns:
            transformed_df[column] = transformed_df[column].apply(func)

        return TimeSeriesData(transformed_df)

    def to_dict_list(self) -> List[Dict[str, Any]]:
        """
        Convert time series data to a list of dictionaries.

        Returns:
            List of dictionaries with timestamp and metrics
        """
        if self._df is None or self._df.empty:
            return []

        # Reset index to include timestamp column in the result
        df_reset = self._df.reset_index()

        # Convert each row to a dictionary
        return df_reset.to_dict(orient="records")

    def to_json(self, filepath: Optional[str] = None, orient: str = "records") -> Optional[str]:
        """
        Convert time series data to JSON.

        Args:
            filepath: Optional file path to save JSON
            orient: JSON orientation ('records', 'split', 'index', etc.)

        Returns:
            JSON string if filepath is None, otherwise None
        """
        if self._df is None or self._df.empty:
            json_data = "[]"
        else:
            # Reset index to include timestamp column
            df_reset = self._df.reset_index()

            # Convert to JSON
            json_data = df_reset.to_json(orient=orient, date_format="iso")

        # Save to file if path provided
        if filepath:
            with open(filepath, "w") as f:
                f.write(json_data)
            logger.info(f"Saved time series data to JSON file: {filepath}")
            return None

        return json_data

    @classmethod
    def from_json(
        cls,
        json_data: Union[str, List[Dict[str, Any]]],
        filepath: Optional[str] = None,
        timestamp_column: str = "timestamp",
    ) -> "TimeSeriesData":
        """
        Create TimeSeriesData from JSON.

        Args:
            json_data: JSON string or parsed list/dictionary
            filepath: Optional file path to load JSON from
            timestamp_column: Name of the timestamp column

        Returns:
            New TimeSeriesData instance
        """
        if filepath:
            with open(filepath, "r") as f:
                json_str = f.read()

            # Try to parse JSON data
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON data in file: {filepath}")
                return TimeSeriesData()

            logger.info(f"Loaded time series data from JSON file: {filepath}")
        elif isinstance(json_data, str):
            try:
                data = json.loads(json_data)
            except json.JSONDecodeError:
                logger.error("Invalid JSON string")
                return TimeSeriesData()
        else:
            data = json_data

        # Handle different data formats
        if isinstance(data, list):
            # List of dictionaries format
            return cls(data, timestamp_column=timestamp_column)
        elif isinstance(data, dict):
            # Other JSON formats might be handled here
            if "data" in data and isinstance(data["data"], list):
                return cls(data["data"], timestamp_column=timestamp_column)
            else:
                logger.warning("Unsupported JSON data format")
                return TimeSeriesData()
        else:
            logger.warning("Unsupported JSON data format")
            return TimeSeriesData()

    def __len__(self) -> int:
        """Get the number of data points."""
        return 0 if self._df is None else len(self._df)

    def __bool__(self) -> bool:
        """Check if time series has data."""
        return self._df is not None and not self._df.empty

    def __repr__(self) -> str:
        """String representation of the TimeSeriesData instance."""
        column_count = 0 if self._df is None else len(self._df.columns)
        return f"TimeSeriesData(points={len(self)}, columns={column_count})"


class BenchmarkResult:
    """
    Container for results from a single benchmark run.

    This class stores metrics and metadata from an individual benchmark run,
    providing methods to access, analyze, and format the results.
    """

    def __init__(
        self,
        run_id: str,
        tier_name: str,
        profile_name: str,
        start_time: Optional[Union[str, datetime]] = None,
        end_time: Optional[Union[str, datetime]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        raw_data: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a new BenchmarkResult instance.

        Args:
            run_id: Unique identifier for this benchmark run
            tier_name: Name of the storage tier
            profile_name: Name of the benchmark profile used
            start_time: Benchmark start time
            end_time: Benchmark end time
            metrics: Dictionary of result metrics (throughput, IOPS, etc.)
            parameters: Dictionary of benchmark parameters
            raw_data: Raw output data from the benchmark engine
        """
        self.run_id = run_id
        self.tier_name = tier_name
        self.profile_name = profile_name

        # Set timestamps
        self.start_time = start_time
        if self.start_time is None:
            self.start_time = datetime.utcnow().isoformat()
        elif isinstance(self.start_time, datetime):
            self.start_time = self.start_time.isoformat()

        self.end_time = end_time
        if self.end_time is None:
            self.end_time = datetime.utcnow().isoformat()
        elif isinstance(self.end_time, datetime):
            self.end_time = self.end_time.isoformat()

        # Store metrics and parameters
        self.metrics = metrics or {}
        self.parameters = parameters or {}
        self.raw_data = raw_data or {}

        # Initialize analysis_results dictionary
        self.analysis_results = {}

        # Initialize tier_results dictionary for multi-tier results
        self.tier_results = {}

        # Initialize tiers list
        self.tiers = []

        # Initialize time series and system metrics data
        self.time_series = {}
        self.system_metrics = {}

        # Calculate duration
        try:
            start_dt = pd.to_datetime(self.start_time)
            end_dt = pd.to_datetime(self.end_time)
            self.duration_seconds = (end_dt - start_dt).total_seconds()
        except BaseException:
            self.duration_seconds = None

        logger.debug(f"Created BenchmarkResult for run {run_id} on tier {tier_name}")

    @property
    def duration(self) -> Optional[float]:
        """Get the benchmark duration in seconds."""
        return self.duration_seconds

    @property
    def throughput(self) -> Optional[float]:
        """Get the throughput value in MB/s."""
        return self.metrics.get("throughput_MBps")

    @property
    def iops(self) -> Optional[float]:
        """Get the IOPS value."""
        return self.metrics.get("iops")

    @property
    def latency(self) -> Optional[float]:
        """Get the latency value in milliseconds."""
        return self.metrics.get("latency_ms")

    def get_metric(self, name: str, default: Any = None) -> Any:
        """
        Get a specific metric value.

        Args:
            name: Metric name
            default: Default value if metric not found

        Returns:
            Metric value or default
        """
        return self.metrics.get(name, default)

    def get_parameter(self, name: str, default: Any = None) -> Any:
        """
        Get a specific parameter value.

        Args:
            name: Parameter name
            default: Default value if parameter not found

        Returns:
            Parameter value or default
        """
        return self.parameters.get(name, default)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert result to a dictionary.

        Returns:
            Dictionary representation of the result
        """
        result = {
            "run_id": self.run_id,
            "tier_name": self.tier_name,
            "profile_name": self.profile_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": self.duration_seconds,
            "metrics": self.metrics,
            "parameters": self.parameters,
        }
        
        # Include analysis results if available
        if hasattr(self, 'analysis_results') and self.analysis_results:
            result["analysis_results"] = {}
            for analysis_type, analysis_result in self.analysis_results.items():
                if hasattr(analysis_result, 'to_dict'):
                    result["analysis_results"][analysis_type] = analysis_result.to_dict()
                else:
                    result["analysis_results"][analysis_type] = analysis_result
        
        # Include time series data if available
        if hasattr(self, 'time_series') and self.time_series:
            result["time_series"] = self.time_series
            
        # Include system metrics data if available
        if hasattr(self, 'system_metrics') and self.system_metrics:
            result["system_metrics"] = self.system_metrics
            
        return result

    def to_series(self) -> pd.Series:
        """
        Convert result to a pandas Series.

        Returns:
            Series representation of the result
        """
        data = self.to_dict()
        # Flatten metrics and parameters
        for k, v in self.metrics.items():
            data[f"metric_{k}"] = v
        for k, v in self.parameters.items():
            data[f"param_{k}"] = v
        # Remove nested dictionaries
        del data["metrics"]
        del data["parameters"]
        return pd.Series(data)

    def to_row(self) -> Dict[str, Any]:
        """
        Convert result to a flattened row format suitable for DataFrames.

        Returns:
            Flattened dictionary
        """
        row = {
            "run_id": self.run_id,
            "tier_name": self.tier_name,
            "profile_name": self.profile_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": self.duration_seconds,
        }

        # Add metrics with metric_ prefix
        for k, v in self.metrics.items():
            row[f"metric_{k}"] = v

        # Add parameters with param_ prefix
        for k, v in self.parameters.items():
            row[f"param_{k}"] = v

        return row

    def get_tiers(self) -> List[str]:
        """
        Get the list of tiers in this benchmark result.

        Returns:
            List of tier names/paths
        """
        return self.tiers if hasattr(self, "tiers") else []

    def get_tier_result(self, tier: str) -> Optional[Dict[str, Any]]:
        """
        Get the result data for a specific tier.

        Args:
            tier: Tier name/path

        Returns:
            Tier result data or None if not found
        """
        if hasattr(self, "tier_results") and tier in self.tier_results:
            return self.tier_results[tier]
        return None

    def merge_raw_data(self, raw_data: Dict[str, Any]) -> None:
        """
        Merge additional raw data into the result.

        Args:
            raw_data: Raw data to merge
        """
        self.raw_data.update(raw_data)

    def update_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Update metrics with new values.

        Args:
            metrics: Metrics to update
        """
        self.metrics.update(metrics)

    def add_analysis_results(self, analysis_type: str, results: Union[Dict[str, Any], "AnalysisResult"]) -> None:
        """
        Add analysis results to the benchmark result.

        Args:
            analysis_type: Type of analysis (e.g., "statistics", "anomaly")
            results: Analysis results dictionary or AnalysisResult object
        """
        # Use the pre-initialized self.analysis_results dictionary
        self.analysis_results[analysis_type] = results
        logger.debug(f"Added {analysis_type} analysis results to benchmark result")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkResult":
        """
        Create a BenchmarkResult from a dictionary.

        Args:
            data: Dictionary with result data

        Returns:
            New BenchmarkResult instance
        """
        instance = cls(
            run_id=data.get("run_id", str(uuid.uuid4())),
            tier_name=data.get("tier_name", "unknown"),
            profile_name=data.get("profile_name", "unknown"),
            start_time=data.get("start_time"),
            end_time=data.get("end_time"),
            metrics=data.get("metrics", {}),
            parameters=data.get("parameters", {}),
            raw_data=data.get("raw_data", {}),
        )
        
        # Restore analysis results if available
        if "analysis_results" in data:
            instance.analysis_results = data["analysis_results"]
            
        # Restore time series data if available
        if "time_series" in data:
            instance.time_series = data["time_series"]
            
        # Restore system metrics data if available  
        if "system_metrics" in data:
            instance.system_metrics = data["system_metrics"]
            
        return instance

    @classmethod
    def from_benchmark_data(cls, benchmark_data: "BenchmarkData") -> "BenchmarkResult":
        """
        Create a BenchmarkResult from BenchmarkData.

        Args:
            benchmark_data: BenchmarkData to convert

        Returns:
            BenchmarkResult instance
        """
        # Calculate actual benchmark duration and timing
        tiers = benchmark_data.get_tiers()
        tier_results = {}
        aggregated_metrics = {}
        actual_start_time = benchmark_data.created_at
        actual_end_time = benchmark_data.created_at
        total_duration = 0
        
        # Collect tier results and calculate timing
        for tier in tiers:
            tier_result = benchmark_data.get_tier_result(tier)
            if tier_result:
                tier_results[tier] = tier_result
                
                # Aggregate metrics from all test results
                if "tests" in tier_result:
                    for test_name, test_result in tier_result["tests"].items():
                        if test_result and isinstance(test_result, dict):
                            # Extract FIO results and add to aggregated metrics
                            # The metrics are directly in the test_result, not nested under "results"
                            if tier not in aggregated_metrics:
                                aggregated_metrics[tier] = {}
                            
                            # Store the complete test metrics
                            aggregated_metrics[tier][test_name] = {
                                "throughput_MBps": test_result.get("throughput_MBps", 0),
                                "iops": test_result.get("iops", 0),
                                "latency_ms": test_result.get("latency_ms", 0),
                                "read": test_result.get("read", {}),
                                "write": test_result.get("write", {}),
                            }
                            
                            # Update timing based on actual test durations
                            if "raw_data" in test_result and "jobs" in test_result["raw_data"]:
                                for job in test_result["raw_data"]["jobs"]:
                                    if "job_runtime" in job:
                                        total_duration += job["job_runtime"] / 1000  # Convert ms to seconds

        # Use actual duration if available, otherwise fall back to configured duration
        if total_duration > 0:
            actual_duration = total_duration
            if isinstance(actual_start_time, str):
                # Convert string to datetime for calculation
                actual_start_time = datetime.fromisoformat(actual_start_time.replace('Z', '+00:00'))
            actual_end_time = actual_start_time + timedelta(seconds=total_duration)
        else:
            actual_duration = benchmark_data.data.get("duration", 0)
            if isinstance(actual_start_time, str):
                # Convert string to datetime for calculation
                actual_start_time = datetime.fromisoformat(actual_start_time.replace('Z', '+00:00'))
            actual_end_time = actual_start_time + timedelta(seconds=actual_duration)

        result = cls(
            run_id=benchmark_data.benchmark_id,
            tier_name="multiple",  # This result contains multiple tiers
            profile_name="multiple",  # This result may contain multiple profiles
            start_time=actual_start_time,
            end_time=actual_end_time,
            metrics=aggregated_metrics,
            parameters={
                "tiers": benchmark_data.get_tiers(),
                "duration": benchmark_data.data.get("duration"),
                "block_sizes": benchmark_data.data.get("block_sizes"),
                "patterns": benchmark_data.data.get("patterns"),
            },
        )

        # Set benchmark_id for compatibility with result_store
        result.benchmark_id = benchmark_data.benchmark_id

        # Add tier results
        result.tiers = tiers
        result.tier_results = tier_results

        # Add analysis results placeholder
        result.analysis_results = {}

        # Extract time series data from tier results
        time_series_data = {}
        for tier, tier_result in tier_results.items():
            if tier_result and "time_series" in tier_result:
                ts_data = tier_result["time_series"]
                # Convert TimeSeriesData object to dictionary if needed
                if hasattr(ts_data, 'to_dict'):
                    time_series_data[tier] = ts_data.to_dict()
                else:
                    time_series_data[tier] = ts_data
        
        if time_series_data:
            result.time_series = time_series_data

        # Transfer system metrics data from benchmark_data
        if hasattr(benchmark_data, '_data') and benchmark_data._data:
            if 'system_metrics' in benchmark_data._data:
                result.system_metrics = benchmark_data._data['system_metrics']

        return result

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the benchmark result.

        Returns:
            Dictionary with result summary
        """
        # Create a basic summary with what we have
        summary = {
            "id": self.run_id,
            "benchmark_id": self.benchmark_id if hasattr(self, "benchmark_id") else self.run_id,
            "tier_name": self.tier_name,
            "profile_name": self.profile_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": self.duration_seconds,
            "metrics": self.metrics,
        }

        # Add tier results summary if available
        if hasattr(self, "tiers") and hasattr(self, "tier_results"):
            summary["tiers"] = self.tiers
            summary["tier_summaries"] = {}

            for tier in self.tiers:
                if tier in self.tier_results:
                    tier_result = self.tier_results[tier]
                    if "summary" in tier_result:
                        summary["tier_summaries"][tier] = tier_result["summary"]

        # Add analysis results summary if available
        if hasattr(self, "analysis_results") and self.analysis_results:
            summary["analysis"] = {}
            for analysis_type, results in self.analysis_results.items():
                if isinstance(results, dict) and "summary" in results:
                    summary["analysis"][analysis_type] = results["summary"]
                else:
                    # Just add a placeholder
                    summary["analysis"][analysis_type] = {"status": "completed"}

        return summary

    def __str__(self) -> str:
        """String representation of the benchmark result."""
        metrics_str = ", ".join(
            [
                f"{k}={v}"
                for k, v in self.metrics.items()
                if k in ["throughput_MBps", "iops", "latency_ms"]
            ]
        )
        return (
            f"BenchmarkResult(run_id={self.run_id}, tier={self.tier_name}, "
            f"profile={self.profile_name}, {metrics_str})"
        )


class BenchmarkData:
    """
    Container for benchmark data with processing capabilities.

    This class stores and manages data from benchmark runs, providing
    methods to manipulate, filter, and process the data for analysis.
    It supports various data formats and operations, including time series
    handling and metric extraction.
    """

    def __init__(
        self, data: Optional[Dict[str, Any]] = None, metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a new BenchmarkData instance.

        Args:
            data: Optional initial data dictionary
            metadata: Optional metadata about the benchmark
        """
        self._data = data or {}
        self._metadata = metadata or {
            "created_at": datetime.utcnow().isoformat(),
            "created_by": os.environ.get("USER", "unknown"),
            "id": str(uuid.uuid4()),
            "version": "1.0.0",
        }
        self._dataframes = {}
        self._time_series = None
        self._series_index = None

        logger.debug(f"Initialized BenchmarkData with ID: {self._metadata.get('id')}")

    @property
    def id(self) -> str:
        """Get the unique identifier for this benchmark data."""
        return self._metadata.get("benchmark_id") or self._metadata.get("id", str(uuid.uuid4()))

    @property
    def benchmark_id(self) -> str:
        """Get the benchmark identifier (alias for id property)."""
        return self.id

    @property
    def created_at(self) -> str:
        """Get the creation timestamp."""
        return self._metadata.get("created_at", datetime.utcnow().isoformat())

    @property
    def created_by(self) -> str:
        """Get the creator identifier."""
        return self._metadata.get("created_by", "unknown")

    @property
    def metadata(self) -> Dict[str, Any]:
        """Get the complete metadata dictionary."""
        return self._metadata.copy()

    @property
    def data(self) -> Dict[str, Any]:
        """Get the raw data dictionary."""
        return self._data.copy()

    @property
    def tiers(self) -> List[str]:
        """Get the list of tiers."""
        return self.get_tiers()

    def add_metadata(self, key: str, value: Any) -> None:
        """
        Add or update a metadata entry.

        Args:
            key: Metadata key
            value: Metadata value
        """
        self._metadata[key] = value
        logger.debug(f"Added metadata: {key}={value}")

    def add_data(self, key: str, value: Any) -> None:
        """
        Add or update a data entry.

        Args:
            key: Data key
            value: Data value
        """
        self._data[key] = value
        # Clear cached dataframes when data changes
        self._dataframes = {}
        logger.debug(f"Added data: {key} (type: {type(value).__name__})")

    def add_series_data(self, timestamp: Union[str, datetime], metrics: Dict[str, Any]) -> None:
        """
        Add time series data point to the benchmark data.

        Args:
            timestamp: Time when data was collected
            metrics: Dictionary of metric names and values
        """
        if isinstance(timestamp, datetime):
            timestamp = timestamp.isoformat()

        if "time_series" not in self._data:
            self._data["time_series"] = []

        data_point = {"timestamp": timestamp}
        data_point.update(metrics)

        self._data["time_series"].append(data_point)

        # Clear cached time series dataframe
        self._time_series = None
        logger.debug(f"Added time series data point at {timestamp} with {len(metrics)} metrics")

    def add_benchmark_run(self, run_id: str, run_data: Dict[str, Any]) -> None:
        """
        Add results from a benchmark run.

        Args:
            run_id: Unique identifier for the run
            run_data: Data collected during the run
        """
        if "runs" not in self._data:
            self._data["runs"] = {}

        self._data["runs"][run_id] = run_data
        logger.debug(f"Added benchmark run: {run_id}")

    def add_benchmark_result(self, result: BenchmarkResult) -> None:
        """
        Add a benchmark result to the data.

        Args:
            result: BenchmarkResult instance
        """
        if "runs" not in self._data:
            self._data["runs"] = {}

        self._data["runs"][result.run_id] = result.to_dict()
        logger.debug(f"Added benchmark result: {result.run_id}")

    def add_tier_result(self, tier: str, result: Dict[str, Any]) -> None:
        """
        Add result data for a specific storage tier.

        Args:
            tier: Storage tier path
            result: Result data for this tier
        """
        if "tier_results" not in self._data:
            self._data["tier_results"] = {}
        self._data["tier_results"][tier] = result
        
        # Consolidate time series data from tier results into main time_series
        if "time_series" in result and result["time_series"]:
            if "time_series" not in self._data:
                self._data["time_series"] = []
            
            # Add tier information to each time series data point
            tier_time_series = result["time_series"]
            if isinstance(tier_time_series, list):
                for data_point in tier_time_series:
                    if isinstance(data_point, dict):
                        data_point["tier"] = tier
                self._data["time_series"].extend(tier_time_series)
            elif isinstance(tier_time_series, dict):
                # Handle single data point
                tier_time_series["tier"] = tier
                self._data["time_series"].append(tier_time_series)
        
        # Also add tier to the tiers list if not already present
        if "tiers" not in self._data:
            self._data["tiers"] = []
        if tier not in self._data["tiers"]:
            self._data["tiers"].append(tier)

    def get_tier_result(self, tier: str) -> Optional[Dict[str, Any]]:
        """
        Get result data for a specific storage tier.

        Args:
            tier: Storage tier path

        Returns:
            Result data for the tier, or None if not found
        """
        tier_results = self._data.get("tier_results", {})
        return tier_results.get(tier)

    def get_benchmark_results(self) -> List[BenchmarkResult]:
        """
        Get all benchmark results.

        Returns:
            List of BenchmarkResult instances
        """
        if "runs" not in self._data:
            return []

        results = []
        for run_id, run_data in self._data["runs"].items():
            results.append(BenchmarkResult.from_dict(run_data))

        return results

    def get_benchmark_result(self, run_id: str) -> Optional[BenchmarkResult]:
        """
        Get a specific benchmark result by run ID.

        Args:
            run_id: Run identifier

        Returns:
            BenchmarkResult instance if found, otherwise None
        """
        if "runs" not in self._data or run_id not in self._data["runs"]:
            return None

        return BenchmarkResult.from_dict(self._data["runs"][run_id])

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert all benchmark results to a pandas DataFrame.

        Returns:
            DataFrame with benchmark results
        """
        results = self.get_benchmark_results()
        if not results:
            return pd.DataFrame()

        rows = [result.to_row() for result in results]
        return pd.DataFrame(rows)

    def add_engine_result(self, engine: str, operation: str, result: Dict[str, Any]) -> None:
        """
        Add results from a specific benchmark engine.

        Args:
            engine: Name of the benchmark engine (e.g., 'fio')
            operation: Type of operation (e.g., 'read', 'write')
            result: Results from the engine
        """
        if "engine_results" not in self._data:
            self._data["engine_results"] = {}

        if engine not in self._data["engine_results"]:
            self._data["engine_results"][engine] = {}

        self._data["engine_results"][engine][operation] = result
        logger.debug(f"Added engine result: {engine}/{operation}")

    def get_time_series(self, reset_index: bool = False) -> pd.DataFrame:
        """
        Get benchmark time series data as a pandas DataFrame.

        Args:
            reset_index: If True, reset the DataFrame index

        Returns:
            DataFrame containing time series data
        """
        # Return cached dataframe if available
        if self._time_series is not None:
            df = self._time_series.copy()
            return df.reset_index() if reset_index else df

        if "time_series" not in self._data or not self._data["time_series"]:
            logger.warning("No time series data available")
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(self._data["time_series"])

        # Convert timestamp to datetime if present
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            # Set timestamp as index
            df = df.set_index("timestamp")
            self._series_index = "timestamp"

        # Cache for future use
        self._time_series = df

        return df.reset_index() if reset_index else df

    def get_time_series_data(self) -> TimeSeriesData:
        """
        Get benchmark time series as a TimeSeriesData instance.

        Returns:
            TimeSeriesData instance with the benchmark's time series data
        """
        if "time_series" not in self._data or not self._data["time_series"]:
            logger.warning("No time series data available")
            return TimeSeriesData()

        return TimeSeriesData(self._data["time_series"])

    def has_time_series_data(self) -> bool:
        """
        Check if the benchmark data contains time series data.

        Returns:
            True if time series data exists, False otherwise
        """
        return (
            "time_series" in self._data
            and self._data["time_series"]
            and len(self._data["time_series"]) > 0
        )

    def get_time_series_points(self) -> int:
        """
        Get the number of time series data points.

        Returns:
            Number of time series data points
        """
        if "time_series" in self._data and isinstance(self._data["time_series"], list):
            return len(self._data["time_series"])
        return 0

    def set_time_series_data(self, time_series: TimeSeriesData) -> None:
        """
        Set time series data from a TimeSeriesData instance.

        Args:
            time_series: TimeSeriesData instance with new data
        """
        if not isinstance(time_series, TimeSeriesData):
            raise TypeError("Expected TimeSeriesData instance")

        # Clear cached time series dataframe
        self._time_series = None

        # Convert TimeSeriesData to list of dictionaries
        self._data["time_series"] = time_series.to_dict_list()

        logger.debug(f"Updated time series data with {len(time_series)} data points")

    def get_dataframe(self, key: str, default_index: Optional[str] = None) -> pd.DataFrame:
        """
        Convert a specific data key to a pandas DataFrame.

        Args:
            key: Data key to convert
            default_index: Column to use as index (if applicable)

        Returns:
            DataFrame representation of the data
        """
        # Return cached dataframe if available
        if key in self._dataframes:
            return self._dataframes[key].copy()

        if key not in self._data:
            logger.warning(f"Key '{key}' not found in data")
            return pd.DataFrame()

        data = self._data[key]

        # Handle different data types
        if isinstance(data, list) and data and isinstance(data[0], dict):
            # List of dictionaries can be directly converted
            df = pd.DataFrame(data)
        elif isinstance(data, dict) and data:
            # For nested dictionaries, approach depends on structure
            if all(isinstance(v, dict) for v in data.values()):
                # If values are dicts, convert to DataFrame with keys as index
                df = pd.DataFrame.from_dict(data, orient="index")
            else:
                # Otherwise, convert to single-row DataFrame
                df = pd.DataFrame([data])
        else:
            logger.warning(f"Cannot convert data at key '{key}' to DataFrame")
            return pd.DataFrame()

        # Set index if specified and exists
        if default_index and default_index in df.columns:
            df = df.set_index(default_index)

        # Cache for future use
        self._dataframes[key] = df

        return df.copy()

    def get_engine_results(
        self, engine: Optional[str] = None, operation: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get results from specific benchmark engines.

        Args:
            engine: Specific engine name (if None, return all)
            operation: Specific operation (if None, return all)

        Returns:
            Dictionary of engine results
        """
        if "engine_results" not in self._data:
            return {}

        results = self._data["engine_results"]

        if engine is not None:
            if engine not in results:
                logger.warning(f"Engine '{engine}' not found in results")
                return {}

            if operation is not None:
                if operation not in results[engine]:
                    logger.warning(f"Operation '{operation}' not found for engine '{engine}'")
                    return {}
                return {operation: results[engine][operation]}

            return results[engine]

        return results

    def get_metric(
        self, metric_name: str, aggregation: Optional[str] = None
    ) -> Union[Any, Dict[str, Any]]:
        """
        Get a specific metric from the benchmark data.

        Args:
            metric_name: Name of the metric to retrieve
            aggregation: Optional aggregation function ('mean', 'median', 'min', 'max', etc.)

        Returns:
            Metric value or dictionary of aggregated values
        """
        # Check if metric is in time series data
        df = self.get_time_series()

        if metric_name in df.columns:
            series = df[metric_name]

            if aggregation:
                if aggregation == "mean":
                    return float(series.mean())
                elif aggregation == "median":
                    return float(series.median())
                elif aggregation == "min":
                    return float(series.min())
                elif aggregation == "max":
                    return float(series.max())
                elif aggregation == "std":
                    return float(series.std())
                elif aggregation == "all":
                    return {
                        "mean": float(series.mean()),
                        "median": float(series.median()),
                        "min": float(series.min()),
                        "max": float(series.max()),
                        "std": float(series.std()),
                    }
                else:
                    logger.warning(f"Unknown aggregation method: {aggregation}")
                    return None

            return series.to_list()

        # Check if metric is in engine results
        for engine, ops in self.get_engine_results().items():
            for op, results in ops.items():
                if isinstance(results, dict) and metric_name in results:
                    return results[metric_name]

        # Check if metric is a direct key in data
        if metric_name in self._data:
            return self._data[metric_name]

        logger.warning(f"Metric '{metric_name}' not found in benchmark data")
        return None

    def get_all_metrics(self) -> Dict[str, Any]:
        """
        Get all available metrics with their aggregated values.

        Returns:
            Dictionary of metrics and their values
        """
        metrics = {}

        # Get metrics from time series
        df = self.get_time_series()
        for column in df.columns:
            metrics[column] = {
                "mean": (
                    float(df[column].mean()) if pd.api.types.is_numeric_dtype(df[column]) else None
                ),
                "min": (
                    float(df[column].min()) if pd.api.types.is_numeric_dtype(df[column]) else None
                ),
                "max": (
                    float(df[column].max()) if pd.api.types.is_numeric_dtype(df[column]) else None
                ),
                "type": "time_series",
            }

        # Get metrics from engine results
        for engine, ops in self.get_engine_results().items():
            for op, results in ops.items():
                if isinstance(results, dict):
                    for key, value in results.items():
                        metric_name = f"{engine}.{op}.{key}"
                        metrics[metric_name] = {"value": value, "type": "engine_result"}

        return metrics

    def filter_time_series(
        self,
        start_time: Optional[Union[str, datetime]] = None,
        end_time: Optional[Union[str, datetime]] = None,
        conditions: Optional[Dict[str, Any]] = None,
    ) -> "BenchmarkData":
        """
        Create a new BenchmarkData with filtered time series.

        Args:
            start_time: Start time for filtering (inclusive)
            end_time: End time for filtering (inclusive)
            conditions: Dictionary of column-value conditions

        Returns:
            New BenchmarkData instance with filtered data
        """
        # Get the time series data
        time_series_data = self.get_time_series_data()

        # Apply filtering
        filtered_time_series = time_series_data.filter(
            start_time=start_time, end_time=end_time, conditions=conditions
        )

        # Create new BenchmarkData with filtered results
        filtered_data = BenchmarkData(metadata=self._metadata.copy())

        # Set the filtered time series
        filtered_data.set_time_series_data(filtered_time_series)

        # Copy other data (but not time_series)
        for key, value in self._data.items():
            if key != "time_series":
                filtered_data.add_data(key, value)

        return filtered_data

    def aggregate_time_series(self, freq: str = "1min", aggregation: str = "mean") -> pd.DataFrame:
        """
        Aggregate time series data by a specified frequency.

        Args:
            freq: Pandas frequency string ('1min', '1h', '1d', etc.)
            aggregation: Aggregation method ('mean', 'median', 'min', 'max', etc.)

        Returns:
            DataFrame with aggregated time series data
        """
        df = self.get_time_series()

        if df.empty:
            logger.warning("No time series data to aggregate")
            return pd.DataFrame()

        # Check if index is timestamp
        if self._series_index != "timestamp":
            logger.warning("Cannot aggregate time series without timestamp index")
            return df

        # Apply resampling with specified aggregation
        if aggregation == "mean":
            return df.resample(freq).mean()
        elif aggregation == "median":
            return df.resample(freq).median()
        elif aggregation == "min":
            return df.resample(freq).min()
        elif aggregation == "max":
            return df.resample(freq).max()
        elif aggregation == "sum":
            return df.resample(freq).sum()
        elif aggregation == "count":
            return df.resample(freq).count()
        else:
            logger.warning(f"Unknown aggregation method: {aggregation}")
            return df

    def calculate_statistics(
        self, metrics: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate descriptive statistics for specified metrics.

        Args:
            metrics: List of metrics to analyze (if None, use all numeric columns)

        Returns:
            Dictionary of statistics for each metric
        """
        # Get time series data as TimeSeriesData instance
        time_series_data = self.get_time_series_data()

        # Use the TimeSeriesData method for statistics
        return time_series_data.calculate_statistics(metrics)

    def merge(self, other: "BenchmarkData") -> "BenchmarkData":
        """
        Merge this BenchmarkData with another instance.

        Args:
            other: Another BenchmarkData instance

        Returns:
            New BenchmarkData instance with merged data
        """
        if not isinstance(other, BenchmarkData):
            raise TypeError("Can only merge with another BenchmarkData instance")

        # Create new instance with merged metadata
        merged_metadata = self._metadata.copy()
        merged_metadata.update(
            {
                "merged_from": [self.id, other.id],
                "merged_at": datetime.utcnow().isoformat(),
                "id": str(uuid.uuid4()),
            }
        )

        merged = BenchmarkData(metadata=merged_metadata)

        # Merge time series data using TimeSeriesData class
        self_ts = self.get_time_series_data()
        other_ts = other.get_time_series_data()

        if self_ts and other_ts:
            merged_ts = self_ts.merge(other_ts)
            merged.set_time_series_data(merged_ts)
        elif self_ts:
            merged.set_time_series_data(self_ts)
        elif other_ts:
            merged.set_time_series_data(other_ts)

        # Merge engine results
        self.get_engine_results()
        other.get_engine_results()

        for result in other.get_benchmark_results():
            # Skip if already added from self
            if "runs" in self._data and result.run_id in self._data["runs"]:
                continue
            merged.add_benchmark_result(result)

        # Merge other data (but not time_series, engine_results, or runs)
        for key, value in self._data.items():
            if key not in ["time_series", "engine_results", "runs"]:
                merged.add_data(key, value)

        for key, value in other._data.items():
            if key not in ["time_series", "engine_results", "runs"]:
                # Skip if already added from self
                if key in self._data:
                    continue
                merged.add_data(key, value)

        return merged

    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate the benchmark data for consistency and completeness.

        Returns:
            Tuple of (is_valid, list_of_validation_messages)
        """
        messages = []
        is_valid = True

        # Check metadata
        if not self._metadata:
            messages.append("Missing metadata")
            is_valid = False
        else:
            required_metadata = ["id", "created_at", "created_by"]
            for field in required_metadata:
                if field not in self._metadata:
                    messages.append(f"Missing required metadata field: {field}")
                    is_valid = False

        # Check data content
        if not self._data:
            messages.append("No benchmark data available")
            is_valid = False

        # Validate time series data if present
        if "time_series" in self._data:
            if not isinstance(self._data["time_series"], list):
                messages.append("Time series data is not a list")
                is_valid = False
            elif not self._data["time_series"]:
                messages.append("Time series data is empty")
            else:
                # Check first time series point
                first_point = self._data["time_series"][0]
                if not isinstance(first_point, dict):
                    messages.append("Time series data points must be dictionaries")
                    is_valid = False
                elif "timestamp" not in first_point:
                    messages.append("Time series data points must have timestamps")
                    is_valid = False

        # Validate engine results if present
        if "engine_results" in self._data:
            if not isinstance(self._data["engine_results"], dict):
                messages.append("Engine results must be a dictionary")
                is_valid = False
            else:
                for engine, ops in self._data["engine_results"].items():
                    if not isinstance(ops, dict):
                        messages.append(f"Operations for engine {engine} must be a dictionary")
                        is_valid = False
                        break

        # Validate benchmark runs if present
        if "runs" in self._data:
            if not isinstance(self._data["runs"], dict):
                messages.append("Benchmark runs must be a dictionary")
                is_valid = False
            else:
                for run_id, run_data in self._data["runs"].items():
                    if not isinstance(run_data, dict):
                        messages.append(f"Run data for {run_id} must be a dictionary")
                        is_valid = False
                    elif "metrics" not in run_data:
                        messages.append(f"Run data for {run_id} missing required 'metrics' field")
                        is_valid = False

        return is_valid, messages

    def to_json(self, filepath: Optional[str] = None, indent: int = 2) -> Optional[str]:
        """
        Convert benchmark data to JSON.

        Args:
            filepath: Optional file path to save JSON
            indent: Indentation level for JSON formatting

        Returns:
            JSON string if filepath is None, otherwise None
        """
        # Prepare exportable data
        export_data = {"metadata": self._metadata, "data": self._data}

        # Convert to JSON
        json_data = json.dumps(export_data, indent=indent, default=self._json_serializer)

        # Save to file if path provided
        if filepath:
            with open(filepath, "w") as f:
                f.write(json_data)
            logger.info(f"Saved benchmark data to JSON file: {filepath}")
            return None

        return json_data

    def to_pickle(self, filepath: str) -> None:
        """
        Save benchmark data to a pickle file.

        Args:
            filepath: File path to save pickle
        """
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Saved benchmark data to pickle file: {filepath}")

    @staticmethod
    def _json_serializer(obj: Any) -> Any:
        """
        Custom JSON serializer for handling non-serializable objects.

        Args:
            obj: Object to serialize

        Returns:
            JSON-serializable version of the object
        """
        if isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if pd.isna(obj):
            return None

        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    @classmethod
    def from_json(
        cls, json_data: Union[str, Dict], filepath: Optional[str] = None
    ) -> "BenchmarkData":
        """
        Create BenchmarkData from JSON.

        Args:
            json_data: JSON string or parsed dictionary
            filepath: Optional file path to load JSON from

        Returns:
            New BenchmarkData instance
        """
        if filepath:
            with open(filepath, "r") as f:
                data = json.load(f)
            logger.info(f"Loaded benchmark data from JSON file: {filepath}")
        elif isinstance(json_data, str):
            data = json.loads(json_data)
        else:
            data = json_data

        metadata = data.get("metadata", {})
        benchmark_data = data.get("data", {})

        return cls(data=benchmark_data, metadata=metadata)

    @classmethod
    def from_pickle(cls, filepath: str) -> "BenchmarkData":
        """
        Load BenchmarkData from a pickle file.

        Args:
            filepath: File path to load pickle from

        Returns:
            Loaded BenchmarkData instance
        """
        with open(filepath, "rb") as f:
            instance = pickle.load(f)

        if not isinstance(instance, cls):
            raise TypeError(f"Loaded object is not a {cls.__name__} instance")

        logger.info(f"Loaded benchmark data from pickle file: {filepath}")
        return instance

    def __repr__(self) -> str:
        """String representation of the BenchmarkData instance."""
        ts_count = len(self._data.get("time_series", []))
        engines = list(self._data.get("engine_results", {}).keys())
        return (
            f"BenchmarkData(id={self.id}, created_at={self.created_at}, "
            f"time_series_points={ts_count}, engines={engines})"
        )

    def __eq__(self, other: object) -> bool:
        """Check if two BenchmarkData instances are equal."""
        if not isinstance(other, BenchmarkData):
            return False

        # Compare metadata (except id which will be different)
        self_meta = self._metadata.copy()
        other_meta = other._metadata.copy()

        # Remove comparison-irrelevant fields
        for meta in [self_meta, other_meta]:
            for field in ["id", "created_at"]:
                if field in meta:
                    del meta[field]

        # Compare data content
        return self_meta == other_meta and self._data == other._data

    def get_tiers(self) -> List[str]:
        """
        Get list of storage tiers in this benchmark data.

        Returns:
            List of tier paths/names
        """
        return self._data.get("tiers", [])

    def set_system_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Set system metrics data.

        Args:
            metrics: System metrics data
        """
        self._data["system_metrics"] = metrics
