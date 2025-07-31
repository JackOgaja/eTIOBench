tdiobench package
=================

The tdiobench package is the main entry point for the Enhanced Tiered Storage I/O Benchmark Suite.

.. automodule:: tdiobench
   :no-members:

Package Information
-------------------

:Version: 2.0.0
:Author: Jack Ogaja  
:Email: jogaja@acm.org
:License: MIT

Overview
--------

The Enhanced Tiered Storage I/O Benchmark Suite (eTIOBench) is a comprehensive benchmarking platform 
designed for analyzing storage performance across different tiers of storage infrastructure. It provides 
production-safe benchmarking with real-time monitoring, advanced analysis capabilities, and rich visualization.

Key Features:

* **FIO-native time series data collection** for accurate performance monitoring
* **Production-safe benchmarking** with resource monitoring and automatic throttling  
* **Multi-tier storage testing** with specialized configurations
* **Advanced statistical analysis** and anomaly detection
* **Rich visualization and reporting** capabilities

Main Components
---------------

The package is organized into several submodules:

* **core** - Core benchmarking functionality, data structures, and configuration
* **engines** - Benchmark execution engines (FIO, custom engines)  
* **analysis** - Statistical analysis and anomaly detection
* **collection** - Data collection systems (time series, system metrics)
* **cli** - Command-line interface and user interaction
* **visualization** - Charts, graphs, and reporting
* **utils** - Utility functions and helpers

Quick Start
-----------

.. code-block:: python

   from tdiobench import BenchmarkSuite
   
   # Create a benchmark suite
   suite = BenchmarkSuite()
   
   # Run a benchmark
   results = suite.run_benchmark(
       tier_path="/storage/path",
       profile="quick_scan"
   )

Command Line Usage
------------------

.. code-block:: bash

   # Run a basic benchmark
   python -m tdiobench.cli.commandline run --tiers /storage --profile quick_scan
   
   # Run with time series data collection
   python -m tdiobench.cli.commandline run --tiers /storage --time-series --profile comprehensive

Subpackages
-----------

.. toctree::
   :maxdepth: 4

   tdiobench.analysis
   tdiobench.cli
   tdiobench.collection
   tdiobench.config
   tdiobench.core
   tdiobench.engines
   tdiobench.results
   tdiobench.tests
   tdiobench.utils
   tdiobench.visualization
