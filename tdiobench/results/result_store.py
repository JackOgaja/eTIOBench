#!/usr/bin/env python3
"""
Result Store (Tiered Storage I/O Benchmark)

This module provides functionality for storing and retrieving benchmark results,
supporting various storage backends including file system, SQLite, and more.

Author: Jack Ogaja
Date: 2025-06-26
"""

import json
import logging
import os
import sqlite3
from typing import Any, Dict, List, Optional

import numpy as np

from tdiobench.core.benchmark_config import BenchmarkConfig
from tdiobench.core.benchmark_data import BenchmarkResult
from tdiobench.core.benchmark_exceptions import BenchmarkStorageError

logger = logging.getLogger("tdiobench.results")


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


class ResultStore:
    """
    Results manager for benchmark results.

    Provides methods for storing and retrieving benchmark results using
    different storage backends.
    """

    def __init__(self, config: BenchmarkConfig):
        """
        Initialize result store.

        Args:
            config: Benchmark configuration
        """
        self.config = config
        self.storage_type = config.get("storage.type", "file")
        self.base_dir = config.get("storage.base_dir", "./results")

        # Create base directory if it doesn't exist
        if self.storage_type == "file":
            os.makedirs(self.base_dir, exist_ok=True)

        # Initialize database if needed
        if self.storage_type == "sqlite":
            self.db_path = config.get(
                "storage.db_path", os.path.join(self.base_dir, "benchmark.db")
            )
            self._init_database()

    def store_results(self, benchmark_result: BenchmarkResult) -> str:
        """
        Store benchmark results.

        Args:
            benchmark_result: Benchmark result to store

        Returns:
            Identifier for stored results

        Raises:
            BenchmarkStorageError: If storage fails
        """
        try:
            benchmark_result.benchmark_id

            if self.storage_type == "file":
                return self._store_results_file(benchmark_result)
            elif self.storage_type == "sqlite":
                return self._store_results_sqlite(benchmark_result)
            else:
                raise BenchmarkStorageError(f"Unsupported storage type: {self.storage_type}")

        except Exception as e:
            logger.error(f"Error storing benchmark results: {str(e)}")
            raise BenchmarkStorageError(f"Failed to store benchmark results: {str(e)}")

    def load_results(self, result_id: str) -> BenchmarkResult:
        """
        Load benchmark results.

        Args:
            result_id: Result identifier

        Returns:
            Loaded benchmark result

        Raises:
            BenchmarkStorageError: If loading fails
        """
        try:
            if self.storage_type == "file":
                return self._load_results_file(result_id)
            elif self.storage_type == "sqlite":
                return self._load_results_sqlite(result_id)
            else:
                raise BenchmarkStorageError(f"Unsupported storage type: {self.storage_type}")

        except Exception as e:
            logger.error(f"Error loading benchmark results: {str(e)}")
            raise BenchmarkStorageError(f"Failed to load benchmark results: {str(e)}")

    def list_results(self, limit: Optional[int] = None, offset: int = 0) -> List[Dict[str, Any]]:
        """
        List available benchmark results.

        Args:
            limit: Maximum number of results to return
            offset: Offset for pagination

        Returns:
            List of result metadata

        Raises:
            BenchmarkStorageError: If listing fails
        """
        try:
            if self.storage_type == "file":
                return self._list_results_file(limit, offset)
            elif self.storage_type == "sqlite":
                return self._list_results_sqlite(limit, offset)
            else:
                raise BenchmarkStorageError(f"Unsupported storage type: {self.storage_type}")

        except Exception as e:
            logger.error(f"Error listing benchmark results: {str(e)}")
            raise BenchmarkStorageError(f"Failed to list benchmark results: {str(e)}")

    def delete_results(self, result_id: str) -> bool:
        """
        Delete benchmark results.

        Args:
            result_id: Result identifier

        Returns:
            True if deletion succeeded, False otherwise

        Raises:
            BenchmarkStorageError: If deletion fails
        """
        try:
            if self.storage_type == "file":
                return self._delete_results_file(result_id)
            elif self.storage_type == "sqlite":
                return self._delete_results_sqlite(result_id)
            else:
                raise BenchmarkStorageError(f"Unsupported storage type: {self.storage_type}")

        except Exception as e:
            logger.error(f"Error deleting benchmark results: {str(e)}")
            raise BenchmarkStorageError(f"Failed to delete benchmark results: {str(e)}")

    def _store_results_file(self, benchmark_result: BenchmarkResult) -> str:
        """
        Store benchmark results in file system.
        
        Args:
            benchmark_result: Benchmark result to store
            
        Returns:
            Identifier for stored results
        """
        result_id = benchmark_result.benchmark_id

        # Ensure base directory exists
        os.makedirs(self.base_dir, exist_ok=True)

        # Store result as JSON directly in base directory with benchmark_id as filename
        result_path = os.path.join(self.base_dir, f"{result_id}.json")

        with open(result_path, "w") as f:
            json.dump(benchmark_result.to_dict(), f, indent=2, cls=NumpyJSONEncoder)

        # Time series data is now embedded in the main JSON file
        # No need for separate time series files
        
        logger.info(f"Stored benchmark results in {result_path}")

        return result_id

    def _load_results_file(self, result_id: str) -> BenchmarkResult:
        """
        Load benchmark results from file system.

        Args:
            result_id: Result identifier

        Returns:
            Loaded benchmark result

        Raises:
            BenchmarkStorageError: If result not found
        """
        # Look for result file directly in base directory
        result_path = os.path.join(self.base_dir, f"{result_id}.json")

        if not os.path.exists(result_path):
            raise BenchmarkStorageError(f"Benchmark result not found: {result_id}")

        # Load result from JSON
        with open(result_path, "r") as f:
            result_dict = json.load(f)

        # Create benchmark result
        benchmark_result = BenchmarkResult.from_dict(result_dict)

        logger.info(f"Loaded benchmark results from {result_path}")

        return benchmark_result

    def _list_results_file(
        self, limit: Optional[int] = None, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        List available benchmark results from file system.

        Args:
            limit: Maximum number of results to return
            offset: Offset for pagination

        Returns:
            List of result metadata
        """
        results = []

        # List JSON files in base directory
        if os.path.exists(self.base_dir):
            items = os.listdir(self.base_dir)

            for item in items:
                # Skip files that are not JSON or are report files
                if not item.endswith('.json') or item.endswith('_report.json'):
                    continue
                    
                item_path = os.path.join(self.base_dir, item)

                if os.path.isfile(item_path):
                    try:
                        # Load metadata from JSON
                        with open(item_path, "r") as f:
                            result_dict = json.load(f)

                        # Extract benchmark ID from filename (remove .json extension)
                        benchmark_id = item[:-5]
                        
                        # Extract metadata
                        metadata = {
                            "result_id": result_dict.get("benchmark_id", benchmark_id),
                            "timestamp": result_dict.get("timestamp"),
                            "tiers": result_dict.get("tiers", []),
                            "duration": result_dict.get("duration", 0),
                        }

                        results.append(metadata)
                    except Exception as e:
                        logger.warning(f"Error loading metadata from {item_path}: {str(e)}")

        # Sort by timestamp (newest first)
        results.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

        # Apply pagination
        if offset > 0:
            results = results[offset:]

        if limit is not None:
            results = results[:limit]

        return results

    def _delete_results_file(self, result_id: str) -> bool:
        """
        Delete benchmark results from file system.

        Args:
            result_id: Result identifier

        Returns:
            True if deletion succeeded, False otherwise
        """
        # Delete the JSON file directly
        result_path = os.path.join(self.base_dir, f"{result_id}.json")

        if not os.path.exists(result_path):
            return False

        # Delete the result file
        os.remove(result_path)
        
        # Also delete corresponding report files if they exist
        html_report = os.path.join(self.base_dir, f"{result_id}_report.html")
        json_report = os.path.join(self.base_dir, f"{result_id}_report.json")
        
        if os.path.exists(html_report):
            os.remove(html_report)
        if os.path.exists(json_report):
            os.remove(json_report)

        logger.info(f"Deleted benchmark results: {result_id}")

        return True

    def _init_database(self) -> None:
        """
        Initialize SQLite database.

        Raises:
            BenchmarkStorageError: If initialization fails
        """
        try:
            # Create database directory if it doesn't exist
            db_dir = os.path.dirname(self.db_path)
            os.makedirs(db_dir, exist_ok=True)

            # Connect to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Create tables
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS benchmark_results (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    duration INTEGER,
                    tier_count INTEGER,
                    result_data TEXT
                )
            """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS time_series_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    benchmark_id TEXT,
                    tier TEXT,
                    data TEXT,
                    FOREIGN KEY (benchmark_id) REFERENCES benchmark_results (id) ON DELETE CASCADE
                )
            """
            )

            # Create indexes
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_benchmark_results_timestamp ON benchmark_results (timestamp)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_time_series_benchmark_id ON time_series_data (benchmark_id)"
            )

            conn.commit()
            conn.close()

            logger.debug(f"Initialized SQLite database: {self.db_path}")

        except Exception as e:
            logger.error(f"Error initializing SQLite database: {str(e)}")
            raise BenchmarkStorageError(f"Failed to initialize SQLite database: {str(e)}")

    def _store_results_sqlite(self, benchmark_result: BenchmarkResult) -> str:
        """
        Store benchmark results in SQLite database.

        Args:
            benchmark_result: Benchmark result to store

        Returns:
            Identifier for stored results
        """
        result_id = benchmark_result.benchmark_id
        result_dict = benchmark_result.to_dict()

        # Extract time series data
        time_series_data = {}

        for tier, tier_data in result_dict.get("tier_results", {}).items():
            if "time_series" in tier_data:
                time_series_data[tier] = tier_data.pop("time_series")

        # Connect to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Store benchmark result
            cursor.execute(
                "INSERT OR REPLACE INTO benchmark_results (id, timestamp, duration, tier_count, result_data) VALUES (?, ?, ?, ?, ?)",
                (
                    result_id,
                    benchmark_result.timestamp,
                    benchmark_result.duration,
                    len(benchmark_result.tiers),
                    json.dumps(result_dict),
                ),
            )

            # Store time series data
            if time_series_data:
                # Delete existing time series data
                cursor.execute("DELETE FROM time_series_data WHERE benchmark_id = ?", (result_id,))

                # Insert new time series data
                for tier, ts_data in time_series_data.items():
                    cursor.execute(
                        "INSERT INTO time_series_data (benchmark_id, tier, data) VALUES (?, ?, ?)",
                        (result_id, tier, json.dumps(ts_data)),
                    )

            conn.commit()

            logger.info(f"Stored benchmark results in SQLite database: {result_id}")

            return result_id

        except Exception as e:
            conn.rollback()
            raise e

        finally:
            conn.close()

    def _load_results_sqlite(self, result_id: str) -> BenchmarkResult:
        """
        Load benchmark results from SQLite database.

        Args:
            result_id: Result identifier

        Returns:
            Loaded benchmark result

        Raises:
            BenchmarkStorageError: If result not found
        """
        # Connect to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Load benchmark result
            cursor.execute("SELECT result_data FROM benchmark_results WHERE id = ?", (result_id,))
            row = cursor.fetchone()

            if not row:
                raise BenchmarkStorageError(f"Benchmark result not found: {result_id}")

            result_dict = json.loads(row[0])

            # Load time series data
            cursor.execute(
                "SELECT tier, data FROM time_series_data WHERE benchmark_id = ?", (result_id,)
            )
            time_series_rows = cursor.fetchall()

            if time_series_rows:
                if "tier_results" not in result_dict:
                    result_dict["tier_results"] = {}

                for tier, data in time_series_rows:
                    if tier in result_dict["tier_results"]:
                        result_dict["tier_results"][tier]["time_series"] = json.loads(data)

            # Create benchmark result
            benchmark_result = BenchmarkResult.from_dict(result_dict)

            logger.info(f"Loaded benchmark results from SQLite database: {result_id}")

            return benchmark_result

        finally:
            conn.close()

    def _list_results_sqlite(
        self, limit: Optional[int] = None, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        List available benchmark results from SQLite database.

        Args:
            limit: Maximum number of results to return
            offset: Offset for pagination

        Returns:
            List of result metadata
        """
        # Connect to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Build query
            query = "SELECT id, timestamp, duration, tier_count FROM benchmark_results ORDER BY timestamp DESC"

            if limit is not None:
                query += f" LIMIT {limit}"

            if offset > 0:
                query += f" OFFSET {offset}"

            # Execute query
            cursor.execute(query)
            rows = cursor.fetchall()

            # Convert to list of dictionaries
            results = []

            for row in rows:
                result_id, timestamp, duration, tier_count = row

                # Get tier information
                cursor.execute(
                    "SELECT tier FROM time_series_data WHERE benchmark_id = ?", (result_id,)
                )
                tier_rows = cursor.fetchall()
                tiers = [tier[0] for tier in tier_rows]

                results.append(
                    {
                        "result_id": result_id,
                        "timestamp": timestamp,
                        "duration": duration,
                        "tiers": tiers,
                    }
                )

            return results

        finally:
            conn.close()

    def _delete_results_sqlite(self, result_id: str) -> bool:
        """
        Delete benchmark results from SQLite database.

        Args:
            result_id: Result identifier

        Returns:
            True if deletion succeeded, False otherwise
        """
        # Connect to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Check if result exists
            cursor.execute("SELECT id FROM benchmark_results WHERE id = ?", (result_id,))
            row = cursor.fetchone()

            if not row:
                return False

            # Delete time series data
            cursor.execute("DELETE FROM time_series_data WHERE benchmark_id = ?", (result_id,))

            # Delete benchmark result
            cursor.execute("DELETE FROM benchmark_results WHERE id = ?", (result_id,))

            conn.commit()

            logger.info(f"Deleted benchmark results from SQLite database: {result_id}")

            return True

        except Exception as e:
            conn.rollback()
            raise e

        finally:
            conn.close()
