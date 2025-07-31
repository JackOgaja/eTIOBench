# Documentation Build Instructions

## Prerequisites

Make sure you have the required documentation dependencies installed:

```bash
pip install sphinx>=5.0.0 sphinx-rtd-theme>=1.0.0
```

Or install all dependencies including documentation:

```bash
pip install -r requirements.txt
```

## Building the Documentation

### Using Make (Recommended)

From the `docs/` directory:

```bash
# Build HTML documentation
make html

# Clean previous build
make clean

# Clean and rebuild
make clean && make html
```

### Using Sphinx directly

From the `docs/` directory:

```bash
# Build HTML documentation
sphinx-build -b html . _build

# Build with quiet mode (show only warnings/errors)
sphinx-build -b html . _build -q

# Clean build
rm -rf _build && sphinx-build -b html . _build
```

## Viewing the Documentation

After building, open the documentation in your browser:

```bash
# On macOS
open _build/index.html

# On Linux
xdg-open _build/index.html

# Or navigate directly to the file
# file:///path/to/eTIOBench/docs/_build/index.html
```

## Documentation Structure

The documentation includes:

- **Main Package Overview** (`tdiobench.html`) - Entry point and quick start
- **Subpackage Documentation**:
  - `tdiobench.core` - Core benchmarking functionality
  - `tdiobench.engines` - Benchmark execution engines
  - `tdiobench.cli` - Command-line interface
  - `tdiobench.analysis` - Statistical analysis
  - `tdiobench.collection` - Data collection systems
  - `tdiobench.visualization` - Reporting and charts
  - `tdiobench.utils` - Utility functions

## Known Issues

The documentation currently has some warnings due to duplicate object descriptions between the main package and submodules. These warnings don't affect the functionality of the documentation but can be resolved by:

1. Further customizing the `autodoc` configuration
2. Using `:no-index:` directives for re-exported objects
3. Restructuring the package imports

## Updating Documentation

When you add new modules or modify docstrings:

1. Regenerate API documentation:
   ```bash
   sphinx-apidoc -o docs tdiobench --force --module-first
   ```

2. Rebuild the documentation:
   ```bash
   cd docs && make clean && make html
   ```

## Live Documentation (Development)

For development with auto-reload:

```bash
pip install sphinx-autobuild
cd docs
sphinx-autobuild . _build --ignore '*/.git/*'
```

This will start a local server and automatically rebuild when files change.
