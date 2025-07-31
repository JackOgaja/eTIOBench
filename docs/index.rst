eTIOBench Documentation
=======================

Welcome to eTIOBench
====================

eTIOBench is an enhanced tiered storage benchmark tool for high-performance computing environments.

Features
--------

- Comprehensive I/O benchmarking with FIO integration
- Multi-tier storage analysis and comparison
- Real-time performance monitoring with time series data collection
- Statistical analysis and anomaly detection
- Network performance analysis
- Rich visualization and reporting
- Production-safe benchmarking with resource monitoring
- Advanced safety features and automatic throttling

Installation
------------

To install eTIOBench:

.. code-block:: bash

   git clone https://github.com/JackOgaja/eTIOBench.git
   cd eTIOBench
   pip install -r requirements.txt

Quick Start
-----------

Run a basic benchmark:

.. code-block:: bash

   python -m tdiobench.cli.commandline run --tiers /storage/path --profile quick_scan

Run with time series data collection:

.. code-block:: bash

   python -m tdiobench.cli.commandline run --tiers /storage/path --profile comprehensive --time-series

API Documentation
=================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules

User Guide
==========

For comprehensive usage instructions, see the `User Guide <../USER_GUIDE.md>`_ which covers:

- Complete installation and setup instructions
- All CLI commands and options  
- FIO-native time series data collection
- Advanced analysis and reporting features
- Production safety guidelines
- Troubleshooting and best practices

   pip install enhanced-tiered-storage-benchmark

Quick Start
-----------

To run a basic benchmark:

.. code-block:: bash

   tdiobench run --config examples/config_examples/basic_config.yaml

API Reference
=============

.. automodule:: tdiobench
   :members:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
