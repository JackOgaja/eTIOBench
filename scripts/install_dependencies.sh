#!/bin/bash
# Install required system dependencies for Enhanced Tiered Storage I/O Benchmark Suite
# Author: Jack Ogaja
# Date: 2025-06-26

# Exit on error
set -e

echo "Installing system dependencies for Enhanced Tiered Storage I/O Benchmark Suite..."

# Detect OS
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
else
    echo "Cannot detect operating system. Please install dependencies manually."
    exit 1
fi

# Install FIO and other dependencies based on OS
case $OS in
    ubuntu|debian)
        echo "Detected Debian/Ubuntu system"
        sudo apt-get update
        sudo apt-get install -y fio python3-dev python3-pip build-essential libssl-dev
        
        # Optional: Install additional dependencies for network analysis
        read -p "Install optional dependencies for network analysis? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            sudo apt-get install -y tshark
        fi
        ;;
        
    fedora|rhel|centos)
        echo "Detected Red Hat/Fedora system"
        sudo dnf install -y fio python3-devel python3-pip gcc openssl-devel
        
        # Optional: Install additional dependencies for network analysis
        read -p "Install optional dependencies for network analysis? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            sudo dnf install -y wireshark
        fi
        ;;
        
    *)
        echo "Unsupported operating system: $OS"
        echo "Please install fio manually, then run: pip install enhanced-tiered-storage-benchmark"
        exit 1
        ;;
esac

echo "System dependencies installed successfully."
echo "To install the Python package, run: pip install enhanced-tiered-storage-benchmark"
echo "For development installation, run: pip install -e .[dev]"
