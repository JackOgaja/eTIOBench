#!/usr/bin/env python3
"""
System Metrics Collector (Tiered Storage I/O Benchmark)

This module provides functionality for collecting system metrics during
benchmark execution, including CPU, memory, and storage utilization.

Author: Jack Ogaja
Date: 2025-06-26
"""

import logging
import os
import platform
import subprocess
import threading
import time
from typing import Any, Dict, Optional

from tdiobench.core.benchmark_config import BenchmarkConfig
from tdiobench.core.benchmark_data import BenchmarkData
from tdiobench.core.benchmark_exceptions import BenchmarkDataError

logger = logging.getLogger("tdiobench.system_metrics")


class SystemMetricsCollector:
    """
    Collector for system performance metrics.

    Collects system metrics such as CPU usage, memory usage, and storage utilization
    during benchmark execution.
    """

    def __init__(self, config: BenchmarkConfig):
        """
        Initialize system metrics collector.

        Args:
            config: Benchmark configuration
        """
        self.config = config
        self.collection_interval = config.get("collection.system_metrics.interval", 5.0)
        self.collection_active = False
        self.collection_thread: Optional[threading.Thread] = None
        self.metrics_data: Dict[str, Any] = {
            "timestamps": [],
            "cpu": [],
            "memory": [],
            "disk": [],
            "network": [],
        }

    def start_collection(self, benchmark_data: BenchmarkData) -> None:
        """
        Start system metrics collection.

        Args:
            benchmark_data: Benchmark data

        Raises:
            BenchmarkDataError: If collection is already active
        """
        if self.collection_active:
            raise BenchmarkDataError("System metrics collection is already active")

        # Reset metrics data
        self.metrics_data = {"timestamps": [], "cpu": [], "memory": [], "disk": [], "network": []}

        # Start collection thread
        self.collection_active = True
        self.collection_thread = threading.Thread(
            target=self._collection_loop, args=(benchmark_data,), daemon=True
        )
        self.collection_thread.start()

        logger.info("Started system metrics collection")

    def stop_collection(self) -> Dict[str, Any]:
        """
        Stop system metrics collection.

        Returns:
            Collected system metrics data

        Raises:
            BenchmarkDataError: If collection is not active
        """
        if not self.collection_active:
            raise BenchmarkDataError("System metrics collection is not active")

        # Stop collection thread
        self.collection_active = False
        if self.collection_thread:
            self.collection_thread.join(timeout=3.0)

        logger.info("Stopped system metrics collection")

        # Return collected data
        return self.metrics_data

    def get_current_cpu_usage(self) -> float:
        """
        Get current CPU usage percentage.

        Returns:
            CPU usage percentage (0-100)
        """
        try:
            # Use platform-specific methods to get CPU usage
            if platform.system() == "Linux":
                return self._get_linux_cpu_usage()
            elif platform.system() == "Darwin":
                return self._get_macos_cpu_usage()
            elif platform.system() == "Windows":
                return self._get_windows_cpu_usage()
            else:
                logger.warning(f"Unsupported platform for CPU usage: {platform.system()}")
                return 0.0
        except Exception as e:
            logger.error(f"Error getting CPU usage: {str(e)}")
            return 0.0

    def get_current_memory_usage(self) -> float:
        """
        Get current memory usage percentage.

        Returns:
            Memory usage percentage (0-100)
        """
        try:
            # Use platform-specific methods to get memory usage
            if platform.system() == "Linux":
                return self._get_linux_memory_usage()
            elif platform.system() == "Darwin":
                return self._get_macos_memory_usage()
            elif platform.system() == "Windows":
                return self._get_windows_memory_usage()
            else:
                logger.warning(f"Unsupported platform for memory usage: {platform.system()}")
                return 0.0
        except Exception as e:
            logger.error(f"Error getting memory usage: {str(e)}")
            return 0.0

    def collect_tier_metadata(self, tier_path: str) -> Dict[str, Any]:
        """
        Collect metadata for a storage tier.

        Args:
            tier_path: Storage tier path

        Returns:
            Dictionary containing tier metadata
        """
        metadata = {"path": tier_path, "name": os.path.basename(tier_path)}

        try:
            # Get filesystem information
            if os.path.exists(tier_path):
                # Get filesystem type
                fs_type = self._get_filesystem_type(tier_path)
                metadata["fs_type"] = fs_type

                # Check if it's a network filesystem
                metadata["is_network_fs"] = fs_type in ["nfs", "cifs", "smb", "smbfs"]

                # Get disk usage
                total, used, free = self._get_disk_usage(tier_path)
                metadata["disk_usage"] = {
                    "total_bytes": total,
                    "used_bytes": used,
                    "free_bytes": free,
                    "used_percent": (used / total * 100) if total > 0 else 0,
                }

                # Get device information
                device = self._get_device_info(tier_path)
                if device:
                    metadata["device"] = device
            else:
                # Handle cloud storage or non-existent paths
                if tier_path.startswith(("/s3://", "/azure://", "/gcs://")):
                    metadata["is_cloud_storage"] = True
                    metadata["cloud_provider"] = tier_path.split("://")[0].lstrip("/")
                else:
                    metadata["exists"] = False

        except Exception as e:
            logger.error(f"Error collecting tier metadata for {tier_path}: {str(e)}")

        return metadata

    def _collection_loop(self, benchmark_data: BenchmarkData) -> None:
        """
        Main collection loop for system metrics.

        Args:
            benchmark_data: Benchmark data
        """
        tiers = benchmark_data.get_tiers()

        while self.collection_active:
            try:
                # Get current timestamp
                current_time = time.time()
                self.metrics_data["timestamps"].append(current_time)

                # Collect CPU usage
                cpu_usage = self.get_current_cpu_usage()
                self.metrics_data["cpu"].append(cpu_usage)

                # Collect memory usage
                memory_usage = self.get_current_memory_usage()
                self.metrics_data["memory"].append(memory_usage)

                # Collect disk usage for each tier
                disk_data = {}
                for tier in tiers:
                    if os.path.exists(tier):
                        total, used, free = self._get_disk_usage(tier)
                        disk_data[tier] = {
                            "total_bytes": total,
                            "used_bytes": used,
                            "free_bytes": free,
                            "used_percent": (used / total * 100) if total > 0 else 0,
                        }

                self.metrics_data["disk"].append(disk_data)

                # Collect network metrics if enabled
                if self.config.get("collection.system_metrics.network.enabled", False):
                    network_data = self._get_network_metrics()
                    self.metrics_data["network"].append(network_data)

                # Sleep until next collection
                time.sleep(self.collection_interval)

            except Exception as e:
                logger.error(f"Error in system metrics collection: {str(e)}")
                time.sleep(self.collection_interval)

    def _get_linux_cpu_usage(self) -> float:
        """
        Get CPU usage on Linux.

        Returns:
            CPU usage percentage (0-100)
        """
        try:
            # Use top command to get CPU usage
            cmd = ["top", "-bn1"]
            output = subprocess.check_output(cmd, text=True)

            # Parse CPU usage from output
            cpu_line = None
            for line in output.split("\n"):
                if line.startswith("%Cpu"):
                    cpu_line = line
                    break

            if cpu_line:
                # Extract user + system CPU usage
                parts = cpu_line.split(",")
                user_cpu = float(parts[0].split(":")[1].strip().split(" ")[0])
                system_cpu = float(parts[1].strip().split(" ")[0])
                return user_cpu + system_cpu

            return 0.0

        except Exception as e:
            logger.error(f"Error getting Linux CPU usage: {str(e)}")
            return 0.0

    def _get_macos_cpu_usage(self) -> float:
        """
        Get CPU usage on macOS.

        Returns:
            CPU usage percentage (0-100)
        """
        try:
            # Use top command to get CPU usage
            cmd = ["top", "-l", "1", "-n", "0"]
            output = subprocess.check_output(cmd, text=True)

            # Parse CPU usage from output
            cpu_line = None
            for line in output.split("\n"):
                if line.startswith("CPU usage"):
                    cpu_line = line
                    break

            if cpu_line:
                # Extract user + system CPU usage
                parts = cpu_line.split(": ")[1].split(",")
                user_cpu = float(parts[0].strip().replace("%", "").replace("user", ""))
                system_cpu = float(parts[1].strip().replace("%", "").replace("sys", ""))
                return user_cpu + system_cpu

            return 0.0

        except Exception as e:
            logger.error(f"Error getting macOS CPU usage: {str(e)}")
            return 0.0

    def _get_windows_cpu_usage(self) -> float:
        """
        Get CPU usage on Windows.

        Returns:
            CPU usage percentage (0-100)
        """
        try:
            # Use wmic command to get CPU usage
            cmd = ["wmic", "cpu", "get", "loadpercentage"]
            output = subprocess.check_output(cmd, text=True)

            # Parse CPU usage from output
            lines = output.strip().split("\n")
            if len(lines) >= 2:
                return float(lines[1].strip())

            return 0.0

        except Exception as e:
            logger.error(f"Error getting Windows CPU usage: {str(e)}")
            return 0.0

    def _get_linux_memory_usage(self) -> float:
        """
        Get memory usage on Linux.

        Returns:
            Memory usage percentage (0-100)
        """
        try:
            # Use free command to get memory usage
            cmd = ["free", "-m"]
            output = subprocess.check_output(cmd, text=True)

            # Parse memory usage from output
            mem_line = None
            for line in output.split("\n"):
                if line.startswith("Mem:"):
                    mem_line = line
                    break

            if mem_line:
                # Extract memory usage
                parts = mem_line.split()
                total = float(parts[1])
                used = float(parts[2])
                return (used / total * 100) if total > 0 else 0

            return 0.0

        except Exception as e:
            logger.error(f"Error getting Linux memory usage: {str(e)}")
            return 0.0

    def _get_macos_memory_usage(self) -> float:
        """
        Get memory usage on macOS.

        Returns:
            Memory usage percentage (0-100)
        """
        try:
            # Use vm_stat command to get memory usage
            cmd = ["vm_stat"]
            output = subprocess.check_output(cmd, text=True)

            # Parse memory usage from output
            # page_size = 4096  # Default page size - not used in current implementation
            pages_free = 0
            pages_active = 0
            pages_inactive = 0
            pages_speculative = 0
            pages_wired = 0

            for line in output.split("\n"):
                if line.startswith("Mach Virtual Memory Statistics:"):
                    continue
                elif line.startswith("Pages free:"):
                    pages_free = int(line.split(":")[1].strip().replace(".", ""))
                elif line.startswith("Pages active:"):
                    pages_active = int(line.split(":")[1].strip().replace(".", ""))
                elif line.startswith("Pages inactive:"):
                    pages_inactive = int(line.split(":")[1].strip().replace(".", ""))
                elif line.startswith("Pages speculative:"):
                    pages_speculative = int(line.split(":")[1].strip().replace(".", ""))
                elif line.startswith("Pages wired down:"):
                    pages_wired = int(line.split(":")[1].strip().replace(".", ""))

            # Calculate memory usage
            total_pages = (
                pages_free + pages_active + pages_inactive + pages_speculative + pages_wired
            )
            used_pages = pages_active + pages_wired

            return (used_pages / total_pages * 100) if total_pages > 0 else 0

        except Exception as e:
            logger.error(f"Error getting macOS memory usage: {str(e)}")
            return 0.0

    def _get_windows_memory_usage(self) -> float:
        """
        Get memory usage on Windows.

        Returns:
            Memory usage percentage (0-100)
        """
        try:
            # Use wmic command to get memory usage
            cmd = ["wmic", "OS", "get", "FreePhysicalMemory,TotalVisibleMemorySize"]
            output = subprocess.check_output(cmd, text=True)

            # Parse memory usage from output
            lines = output.strip().split("\n")
            if len(lines) >= 2:
                values = lines[1].split()
                if len(values) >= 2:
                    free_memory = float(values[0])
                    total_memory = float(values[1])
                    used_memory = total_memory - free_memory
                    return (used_memory / total_memory * 100) if total_memory > 0 else 0

            return 0.0

        except Exception as e:
            logger.error(f"Error getting Windows memory usage: {str(e)}")
            return 0.0

    def _get_disk_usage(self, path: str) -> tuple:
        """
        Get disk usage for a path.

        Args:
            path: Path to check

        Returns:
            Tuple of (total_bytes, used_bytes, free_bytes)
        """
        try:
            if os.path.exists(path):
                stat = os.statvfs(path)
                total = stat.f_blocks * stat.f_frsize
                free = stat.f_bfree * stat.f_frsize
                used = total - free
                return (total, used, free)
            else:
                return (0, 0, 0)
        except Exception as e:
            logger.error(f"Error getting disk usage for {path}: {str(e)}")
            return (0, 0, 0)

    def _get_filesystem_type(self, path: str) -> str:
        """
        Get filesystem type for a path.

        Args:
            path: Path to check

        Returns:
            Filesystem type
        """
        try:
            if platform.system() == "Linux":
                # Use df command to get filesystem type
                cmd = ["df", "-T", path]
                output = subprocess.check_output(cmd, text=True)

                # Parse filesystem type from output
                lines = output.strip().split("\n")
                if len(lines) >= 2:
                    fs_type = lines[1].split()[1]
                    return fs_type

            # Fallback for other platforms
            return "unknown"

        except Exception as e:
            logger.error(f"Error getting filesystem type for {path}: {str(e)}")
            return "unknown"

    def _get_device_info(self, path: str) -> Optional[Dict[str, Any]]:
        """
        Get device information for a path.

        Args:
            path: Path to check

        Returns:
            Dictionary containing device information, or None if not available
        """
        try:
            if platform.system() == "Linux":
                # Get device name
                cmd = ["df", path]
                output = subprocess.check_output(cmd, text=True)

                # Parse device name from output
                lines = output.strip().split("\n")
                if len(lines) >= 2:
                    device_name = lines[1].split()[0]

                    # Get additional device information if available
                    if device_name.startswith("/dev/"):
                        device_info = {"name": device_name}

                        # Try to get device model for block devices
                        try:
                            # Extract device base name (e.g., sda from /dev/sda1)
                            import re

                            base_device = re.sub(r"\d+$", "", device_name.split("/")[-1])

                            # Get device model
                            model_path = f"/sys/block/{base_device}/device/model"
                            if os.path.exists(model_path):
                                with open(model_path, "r") as f:
                                    device_info["model"] = f.read().strip()

                            # Get device type (SSD vs HDD)
                            rotational_path = f"/sys/block/{base_device}/queue/rotational"
                            if os.path.exists(rotational_path):
                                with open(rotational_path, "r") as f:
                                    rotational = f.read().strip()
                                    device_info["type"] = "HDD" if rotational == "1" else "SSD"

                        except Exception as e:
                            logger.debug(f"Error getting detailed device info: {str(e)}")

                        return device_info

            return None

        except Exception as e:
            logger.error(f"Error getting device info for {path}: {str(e)}")
            return None

    def _get_network_metrics(self) -> Dict[str, Any]:
        """
        Get network performance metrics.

        Returns:
            Dictionary containing network metrics
        """
        metrics = {}

        try:
            if platform.system() == "Linux":
                # Use /proc/net/dev to get network metrics
                with open("/proc/net/dev", "r") as f:
                    lines = f.readlines()

                # Parse network metrics
                interfaces = {}
                for line in lines[2:]:  # Skip header lines
                    parts = line.split(":")
                    if len(parts) >= 2:
                        interface = parts[0].strip()
                        if interface != "lo":  # Skip loopback
                            values = parts[1].split()
                            interfaces[interface] = {
                                "rx_bytes": int(values[0]),
                                "rx_packets": int(values[1]),
                                "tx_bytes": int(values[8]),
                                "tx_packets": int(values[9]),
                            }

                metrics["interfaces"] = interfaces

            # Add network connectivity check if configured
            if self.config.get("collection.system_metrics.network.connectivity_check", False):
                metrics["connectivity"] = self._check_network_connectivity()

        except Exception as e:
            logger.error(f"Error getting network metrics: {str(e)}")

        return metrics

    def _check_network_connectivity(self) -> Dict[str, bool]:
        """
        Check network connectivity to important hosts.

        Returns:
            Dictionary mapping hosts to connectivity status
        """
        hosts = self.config.get(
            "collection.system_metrics.network.check_hosts", ["8.8.8.8", "1.1.1.1"]
        )
        results = {}

        for host in hosts:
            try:
                # Use ping to check connectivity
                if platform.system() == "Windows":
                    cmd = ["ping", "-n", "1", "-w", "1000", host]
                else:
                    cmd = ["ping", "-c", "1", "-W", "1", host]

                subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                results[host] = True

            except subprocess.SubprocessError:
                results[host] = False

            except Exception as e:
                logger.error(f"Error checking connectivity to {host}: {str(e)}")
                results[host] = False

        return results
