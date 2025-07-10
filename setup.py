#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Setup script for Enhanced Tiered Storage I/O Benchmark Suite.

Author: Jack Ogaja
Date: 2025-06-26
"""

import os
import re
from setuptools import setup, find_packages

# Get the version from __version__.py
with open('tdiobench/__version__.py', 'r') as f:
    version_file = f.read()
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        version = version_match.group(1)
    else:
        raise RuntimeError("Unable to find version string in tdiobench/__version__.py")

# Get the long description from README.md
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

# Base dependencies required for the benchmark suite
install_requires = [
    'numpy>=1.19.0',
    'pandas>=1.1.0',
    'matplotlib>=3.3.0',
    'seaborn>=0.11.0',
    'plotly>=4.14.0',
    'scipy>=1.5.0',
    'scikit-learn>=0.23.0',
    'pyyaml>=5.4.0',
    'jsonschema>=3.2.0',
    'psutil>=5.8.0',
    'rich>=10.0.0',
    'click>=8.0.0',
    'tqdm>=4.60.0',
]

# Additional dependencies for specific features
extras_require = {
    'dev': [
        'pytest>=6.0.0',
        'pytest-cov>=2.12.0',
        'flake8>=3.9.0',
        'black>=21.5b2',
        'isort>=5.9.0',
        'mypy>=0.812',
        'sphinx>=4.0.0',
        'sphinx-rtd-theme>=0.5.2',
    ],
    'network': [
        'pyshark>=0.4.3',
        'netaddr>=0.8.0',
        'netifaces>=0.11.0',
    ],
    'distributed': [
        'paramiko>=2.8.0',
        'fabric>=2.6.0',
    ],
    'full': [
        'pyshark>=0.4.3',
        'netaddr>=0.8.0',
        'netifaces>=0.11.0',
        'paramiko>=2.8.0',
        'fabric>=2.6.0',
    ],
}

# Add 'full' as a combination of all extras
extras_require['full'] = list(set(sum([extras_require[k] for k in ['network', 'distributed']], [])))

setup(
    name='enhanced-tiered-storage-benchmark',
    version=version,
    description='A comprehensive, production-safe benchmarking solution for tiered storage systems',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Jack Ogaja',
    author_email='jogaja@acm.org',
    url='https://github.com/JackOgaja/TieredIOBench',
    packages=find_packages(exclude=['tests*', 'docs*', 'examples*']),
    include_package_data=True,
    install_requires=install_requires,
    extras_require=extras_require,
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'benchmark-suite=tdiobench.cli.commandline:main',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: System Administrators',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: System :: Benchmark',
        'Topic :: System :: Systems Administration',
        'Topic :: System :: Hardware',
    ],
    keywords='benchmark, storage, io, performance, tiered storage, fio',
)
