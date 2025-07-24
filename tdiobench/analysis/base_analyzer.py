#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base analyzer module for benchmark suite (Tiered Storage I/O Benchmark).

This module provides the abstract base class for all analyzer implementations
in the benchmark suite. It defines common functionality and interfaces that
specific analyzers should implement.

Author: Jack Ogaja
Date: 2025-06-26
"""

import json
import logging
import os
from abc import ABC
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tdiobench.core.benchmark_data import BenchmarkData, BenchmarkResult, TimeSeriesData

logger = logging.getLogger(__name__)


class BaseAnalyzer(ABC):
    """
    Abstract base class for benchmark data analyzers.

    This class defines the common interface and functionality for all
    analyzer implementations. Specific analyzers should inherit from this
    class and implement the required abstract methods.
    """

    def __init__(
        self, data: Optional[BenchmarkData] = None, config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a new analyzer instance.

        Args:
            data: Optional BenchmarkData instance to analyze
            config: Optional configuration dictionary
        """
        self._data = data
        self._config = config or {}
        self._results = {}
        self._figures = {}
        self._reports = {}

        # Set default configuration
        self._set_default_config()

        # Apply user configuration
        if config:
            self._apply_config(config)

        logger.debug(f"Initialized {self.__class__.__name__}")

    def _set_default_config(self) -> None:
        """Set default configuration values."""
        self._config.setdefault("output_dir", "output")
        self._config.setdefault("figure_format", "png")
        self._config.setdefault("figure_dpi", 300)
        self._config.setdefault("report_format", "markdown")
        self._config.setdefault("confidence_level", 0.95)
        self._config.setdefault("decimal_places", 2)
        self._config.setdefault("include_raw_data", False)
        self._config.setdefault("time_format", "%Y-%m-%d %H:%M:%S")

    def _apply_config(self, config: Dict[str, Any]) -> None:
        """
        Apply user-provided configuration.

        Args:
            config: Configuration dictionary
        """
        for key, value in config.items():
            self._config[key] = value

    @property
    def data(self) -> Optional[BenchmarkData]:
        """Get the benchmark data."""
        return self._data

    @data.setter
    def data(self, value: BenchmarkData) -> None:
        """
        Set the benchmark data.

        Args:
            value: BenchmarkData instance
        """
        if not isinstance(value, BenchmarkData):
            raise TypeError("Data must be a BenchmarkData instance")
        self._data = value
        # Clear cached results when data changes
        self._results = {}

    @property
    def config(self) -> Dict[str, Any]:
        """Get the configuration dictionary."""
        return self._config.copy()

    def update_config(self, config: Dict[str, Any]) -> None:
        """
        Update configuration with new values.

        Args:
            config: Configuration dictionary with new values
        """
        self._apply_config(config)

    def load_data(self, data_source: Union[str, BenchmarkData, Dict[str, Any]]) -> None:
        """
        Load benchmark data from a source.

        Args:
            data_source: Source to load data from (filepath, BenchmarkData instance, or dictionary)
        """
        if isinstance(data_source, BenchmarkData):
            self._data = data_source
        elif isinstance(data_source, dict):
            self._data = BenchmarkData(data=data_source)
        elif isinstance(data_source, str):
            # Assume it's a filepath
            if data_source.endswith(".json"):
                self._data = BenchmarkData.from_json(filepath=data_source)
            elif data_source.endswith(".pkl") or data_source.endswith(".pickle"):
                self._data = BenchmarkData.from_pickle(filepath=data_source)
            else:
                raise ValueError(f"Unsupported file format: {data_source}")
        else:
            raise TypeError("Unsupported data source type")

        # Clear cached results when data changes
        self._results = {}
        logger.info(f"Loaded data with ID: {self._data.id}")

    # @abstractmethod
    # def analyze(self) -> Dict[str, Any]:
    #    """
    #    Perform analysis on the benchmark data.
    #
    #    This method must be implemented by subclasses to perform the
    #    specific analysis required for that analyzer type.
    #
    #    Returns:
    #        Dictionary of analysis results
    #    """
    #    pass

    def get_results(self, force_reanalyze: bool = False) -> Dict[str, Any]:
        """
        Get analysis results, running analysis if needed.

        Args:
            force_reanalyze: If True, rerun analysis even if results exist

        Returns:
            Dictionary of analysis results
        """
        if not self._data:
            raise ValueError("No data available for analysis")

        if force_reanalyze or not self._results:
            self._results = self.analyze()

        return self._results.copy()

    def get_result(self, key: str) -> Any:
        """
        Get a specific analysis result.

        Args:
            key: Result key to retrieve

        Returns:
            Result value
        """
        results = self.get_results()
        if key not in results:
            raise KeyError(f"Result key not found: {key}")

        return results[key]

    # @abstractmethod
    # def generate_report(self, output_path: Optional[str] = None,
    #                    format: Optional[str] = None) -> str:
    #    """
    #    Generate a report from analysis results.
    #
    #    This method must be implemented by subclasses to create a
    #    report in the desired format based on the analysis results.
    #
    #    Args:
    #        output_path: Optional path to save the report
    #        format: Report format ('markdown', 'html', 'pdf', etc.)
    #
    #    Returns:
    #        Report content as a string
    #    """
    #    pass

    def generate_figure(self, figure_type: str, **kwargs) -> plt.Figure:
        """
        Generate a figure from analysis results.

        Args:
            figure_type: Type of figure to generate
            **kwargs: Additional arguments for figure generation

        Returns:
            Matplotlib Figure object
        """
        if figure_type not in self._figure_generators:
            raise ValueError(f"Unknown figure type: {figure_type}")

        # Call the appropriate figure generator method
        fig = self._figure_generators[figure_type](**kwargs)

        # Store the figure for later retrieval
        self._figures[figure_type] = fig

        return fig

    def save_figure(
        self,
        figure: plt.Figure,
        filename: str,
        format: Optional[str] = None,
        dpi: Optional[int] = None,
    ) -> str:
        """
        Save a figure to a file.

        Args:
            figure: Matplotlib Figure to save
            filename: Output filename
            format: Output format (png, pdf, svg, etc.)
            dpi: Resolution for raster formats

        Returns:
            Path to the saved figure
        """
        format = format or self._config.get("figure_format", "png")
        dpi = dpi or self._config.get("figure_dpi", 300)

        # Create output directory if it doesn't exist
        output_dir = self._config.get("output_dir", "output")
        os.makedirs(output_dir, exist_ok=True)

        # Ensure filename has the correct extension
        if not filename.endswith(f".{format}"):
            filename = f"{filename}.{format}"

        # Create full path
        filepath = os.path.join(output_dir, filename)

        # Save the figure
        figure.savefig(filepath, format=format, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved figure to {filepath}")

        return filepath

    def save_all_figures(
        self,
        base_filename: Optional[str] = None,
        format: Optional[str] = None,
        dpi: Optional[int] = None,
    ) -> Dict[str, str]:
        """
        Save all generated figures.

        Args:
            base_filename: Base filename to use (default: analyzer class name)
            format: Output format (png, pdf, svg, etc.)
            dpi: Resolution for raster formats

        Returns:
            Dictionary mapping figure types to saved file paths
        """
        if not self._figures:
            logger.warning("No figures have been generated")
            return {}

        base_filename = base_filename or self.__class__.__name__.lower()
        saved_paths = {}

        for figure_type, fig in self._figures.items():
            filename = f"{base_filename}_{figure_type}"
            path = self.save_figure(fig, filename, format, dpi)
            saved_paths[figure_type] = path

        return saved_paths

    def save_results(self, filepath: Optional[str] = None) -> str:
        """
        Save analysis results to a JSON file.

        Args:
            filepath: Optional filepath to save results

        Returns:
            Path to the saved results file
        """
        results = self.get_results()

        # Create output directory if it doesn't exist
        output_dir = self._config.get("output_dir", "output")
        os.makedirs(output_dir, exist_ok=True)

        # Generate filepath if not provided
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.__class__.__name__.lower()}_results_{timestamp}.json"
            filepath = os.path.join(output_dir, filename)

        # Serialize results to JSON
        with open(filepath, "w") as f:
            json.dump(results, f, indent=2, default=self._json_serializer)

        logger.info(f"Saved analysis results to {filepath}")
        return filepath

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

    def extract_time_series(self, metrics: Optional[List[str]] = None) -> TimeSeriesData:
        """
        Extract time series data for specified metrics.

        Args:
            metrics: List of metrics to extract (if None, extract all)

        Returns:
            TimeSeriesData instance with extracted data
        """
        if not self._data:
            raise ValueError("No data available for extraction")

        return self._data.get_time_series_data()

    def extract_benchmark_results(self) -> List[BenchmarkResult]:
        """
        Extract all benchmark results.

        Returns:
            List of BenchmarkResult instances
        """
        if not self._data:
            raise ValueError("No data available for extraction")

        return self._data.get_benchmark_results()

    def validate_data(self) -> Tuple[bool, List[str]]:
        """
        Validate the benchmark data.

        Returns:
            Tuple of (is_valid, list_of_validation_messages)
        """
        if not self._data:
            return False, ["No data available for validation"]

        return self._data.validate()

    def __str__(self) -> str:
        """String representation of the analyzer."""
        data_id = self._data.id if self._data else "None"
        result_count = len(self._results)
        return f"{self.__class__.__name__}(data_id={data_id}, results={result_count})"
