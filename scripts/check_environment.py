#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Environment checker for Tiered Storage I/O Benchmark Suite.

This script checks if the environment has all necessary dependencies
and configurations for running the benchmark suite effectively.

Author: Jack Ogaja
Date: 2025-06-26
"""

import os
import sys
import shutil
import platform
import subprocess
from pathlib import Path


def check_python_version():
    """Check if Python version is at least 3.8."""
    print("Checking Python version...")
    major, minor = sys.version_info[:2]
    if major < 3 or (major == 3 and minor < 8):
        print(f"❌ Python version {major}.{minor} is not supported. Please use Python 3.8+")
        return False
    print(f"✅ Python version: {major}.{minor}")
    return True


def check_fio():
    """Check if FIO is installed and get its version."""
    print("Checking FIO installation...")
    if shutil.which("fio") is None:
        print("❌ FIO is not installed or not in PATH")
        return False
    
    try:
        result = subprocess.run(["fio", "--version"], 
                               capture_output=True, text=True, check=True)
        version = result.stdout.strip()
        print(f"✅ FIO version: {version}")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to get FIO version")
        return False


def check_python_dependencies():
    """Check if required Python packages are installed."""
    print("Checking Python dependencies...")
    required_packages = [
        "numpy", "pandas", "matplotlib", "seaborn", "plotly", 
        "scipy", "scikit-learn", "pyyaml", "jsonschema", "psutil"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} is installed")
        except ImportError:
            print(f"❌ {package} is not installed")
            missing_packages.append(package)
    
    if missing_packages:
        print("\nMissing packages:")
        print("pip install " + " ".join(missing_packages))
        return False
    
    return True


def check_storage_tiers():
    """Check if common storage paths are mounted and accessible."""
    print("Checking storage tiers...")
    
    # Common mount points to check
    mount_points = [
        ("/mnt/nvme", "NVMe"),
        ("/mnt/ssd", "SSD"),
        ("/mnt/hdd", "HDD"),
        ("/mnt/lustre", "Lustre"),
        ("/mnt/ceph", "Ceph")
    ]
    
    available_tiers = []
    
    for path, tier_type in mount_points:
        if os.path.isdir(path) and os.access(path, os.R_OK | os.W_OK):
            print(f"✅ {tier_type} tier at {path} is accessible")
            available_tiers.append((path, tier_type))
        else:
            print(f"ℹ️ {tier_type} tier at {path} is not accessible")
    
    if not available_tiers:
        print("⚠️ No storage tiers detected at common mount points")
        print("Please configure custom tier paths in your benchmark configuration")
    
    return True  # Return True even if no tiers found, as custom paths can be configured


def check_system_resources():
    """Check available system resources."""
    print("Checking system resources...")
    
    # Check CPU count
    import multiprocessing
    cpu_count = multiprocessing.cpu_count()
    print(f"✅ CPU cores: {cpu_count}")
    
    # Check memory
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024 ** 3)
        print(f"✅ Memory: {memory_gb:.1f} GB")
    except ImportError:
        print("⚠️ Could not determine memory size (psutil not installed)")
    
    # Check disk space
    for path in ["/", "/home", "/tmp"]:
        if os.path.exists(path):
            try:
                import shutil
                total, used, free = shutil.disk_usage(path)
                free_gb = free / (1024 ** 3)
                print(f"✅ Disk space on {path}: {free_gb:.1f} GB free")
            except Exception as e:
                print(f"⚠️ Could not determine disk space on {path}: {e}")
    
    return True


def main():
    """Run all environment checks."""
    print("=" * 60)
    print("Enhanced Tiered Storage I/O Benchmark Suite - Environment Check")
    print("=" * 60)
    print(f"Date: {os.environ.get('DATE', 'Unknown')}")
    print(f"User: {os.environ.get('USER', 'Unknown')}")
    print(f"System: {platform.system()} {platform.release()}")
    print("=" * 60)
    
    checks = [
        check_python_version(),
        check_fio(),
        check_python_dependencies(),
        check_storage_tiers(),
        check_system_resources()
    ]
    
    print("\n" + "=" * 60)
    if all(checks):
        print("✅ All critical checks passed! Your environment is ready for benchmarking.")
    else:
        print("⚠️ Some checks failed. Please address the issues before running benchmarks.")
    print("=" * 60)


if __name__ == "__main__":
    main()
