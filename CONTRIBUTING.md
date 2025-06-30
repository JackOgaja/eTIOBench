# Contributing to Tiered Storage I/O Benchmark Suite

Thank you for considering contributing to the Enhanced Tiered Storage I/O Benchmark Suite! This document provides guidelines and instructions for contribution.

## Code of Conduct

Please be respectful and considerate of others when contributing to this project. We expect all contributors to follow these basic principles:

- Be respectful of differing viewpoints and experiences
- Accept constructive criticism gracefully
- Focus on what is best for the community
- Show empathy towards other community members

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in the Issues section
2. If not, create a new issue with a clear title and description
3. Include steps to reproduce the bug
4. Include any relevant logs or error messages
5. Specify your environment (OS, Python version, etc.)

### Suggesting Features

1. Check if the feature has already been suggested in the Issues section
2. If not, create a new issue with a clear title and description
3. Explain why this feature would be useful to the broader community
4. Provide examples of how the feature might work

### Pull Requests

1. Fork the repository
2. Create a new branch for your feature or bugfix
3. Make your changes
4. Add tests for your changes
5. Ensure all tests pass with `pytest`
6. Make sure your code follows the project's code style
7. Create a Pull Request with a clear description of your changes

## Development Setup

```bash
# Clone your fork of the repository
git clone https://github.com/YOUR_USERNAME/enhanced-tiered-storage-benchmark.git
cd enhanced-tiered-storage-benchmark

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest
