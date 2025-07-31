#!/bin/bash

# eTIOBench C++ Module Builder
# Builds high-performance C++ modules with full Python integration

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
BUILD_TYPE=${BUILD_TYPE:-Release}
BUILD_DIR=${BUILD_DIR:-build}
INSTALL_PREFIX=${INSTALL_PREFIX:-install}
NUM_JOBS=${NUM_JOBS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}
PYTHON_EXECUTABLE=${PYTHON_EXECUTABLE:-python3}

# Flags
CLEAN_BUILD=false
RUN_TESTS=false
INSTALL_MODULE=false
BENCHMARK=false
VERBOSE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --clean)
            CLEAN_BUILD=true
            shift
            ;;
        --test)
            RUN_TESTS=true
            shift
            ;;
        --install)
            INSTALL_MODULE=true
            shift
            ;;
        --benchmark)
            BENCHMARK=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --build-type)
            BUILD_TYPE="$2"
            shift 2
            ;;
        --jobs)
            NUM_JOBS="$2"
            shift 2
            ;;
        --python)
            PYTHON_EXECUTABLE="$2"
            shift 2
            ;;
        --help)
            echo "eTIOBench C++ Module Builder"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --clean         Clean build directory before building"
            echo "  --test          Run tests after building"
            echo "  --install       Install Python module after building"
            echo "  --benchmark     Run performance benchmarks"
            echo "  --verbose       Enable verbose output"
            echo "  --build-type    Build type (Debug|Release|RelWithDebInfo) [default: Release]"
            echo "  --jobs          Number of parallel jobs [default: auto-detect]"
            echo "  --python        Python executable to use [default: python3]"
            echo "  --help          Show this help message"
            echo ""
            echo "Environment variables:"
            echo "  BUILD_TYPE      Build type (same as --build-type)"
            echo "  BUILD_DIR       Build directory [default: build]"
            echo "  INSTALL_PREFIX  Install prefix [default: install]"
            echo "  NUM_JOBS        Number of parallel jobs"
            echo "  PYTHON_EXECUTABLE Python executable"
            echo ""
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Print configuration
echo -e "${BLUE}eTIOBench C++ Module Builder${NC}"
echo -e "${BLUE}=============================${NC}"
echo "Build type: $BUILD_TYPE"
echo "Build directory: $BUILD_DIR"
echo "Install prefix: $INSTALL_PREFIX"
echo "Parallel jobs: $NUM_JOBS"
echo "Python executable: $PYTHON_EXECUTABLE"
echo "Clean build: $CLEAN_BUILD"
echo "Run tests: $RUN_TESTS"
echo "Install module: $INSTALL_MODULE"
echo "Run benchmark: $BENCHMARK"
echo ""

# Helper functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_command() {
    if ! command -v $1 &> /dev/null; then
        log_error "$1 is required but not installed"
        exit 1
    fi
}

# Check prerequisites
log_info "Checking prerequisites..."

check_command cmake
check_command make
check_command $PYTHON_EXECUTABLE

# Check Python version
PYTHON_VERSION=$($PYTHON_EXECUTABLE -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
log_info "Python version: $PYTHON_VERSION"

# Check for required Python packages
log_info "Checking Python dependencies..."

required_packages=("pybind11" "numpy")
for package in "${required_packages[@]}"; do
    if ! $PYTHON_EXECUTABLE -c "import $package" 2>/dev/null; then
        log_warn "$package not found, attempting to install..."
        $PYTHON_EXECUTABLE -m pip install $package
    else
        log_info "$package is available"
    fi
done

# Check compiler support
log_info "Checking compiler support..."

if command -v g++ &> /dev/null; then
    GCC_VERSION=$(g++ --version | head -n1 | sed -E 's/.*([0-9]+\.[0-9]+).*/\1/' | head -1)
    log_info "GCC version: $GCC_VERSION"
    
    # Check for C++17 support
    if g++ -std=c++17 -x c++ -c /dev/null -o /dev/null 2>/dev/null; then
        log_info "C++17 support: Available"
    else
        log_error "C++17 support is required"
        exit 1
    fi
fi

if command -v clang++ &> /dev/null; then
    CLANG_VERSION=$(clang++ --version | head -n1 | sed -E 's/.*([0-9]+\.[0-9]+).*/\1/' | head -1)
    log_info "Clang version: $CLANG_VERSION"
fi

# Check for SIMD support
log_info "Checking SIMD support..."
if (grep -q avx2 /proc/cpuinfo 2>/dev/null) || (sysctl -n machdep.cpu.features 2>/dev/null | grep -q AVX2) || (sysctl -n machdep.cpu.leaf7_features 2>/dev/null | grep -q AVX2); then
    log_info "AVX2 support: Available"
else
    log_warn "AVX2 support: Not detected (performance may be reduced)"
fi

# Check for OpenMP
if command -v pkg-config &> /dev/null && pkg-config --exists openmp; then
    log_info "OpenMP support: Available"
elif [ -f "/usr/lib/gcc/*/libgomp.so" ] || [ -f "/usr/local/lib/libomp.dylib" ]; then
    log_info "OpenMP support: Available"
else
    log_warn "OpenMP support: Not detected (parallelization may be limited)"
fi

# Change to cpp_core directory
cd "$(dirname "$0")"

# Clean build if requested
if [ "$CLEAN_BUILD" = true ]; then
    log_info "Cleaning build directory..."
    rm -rf $BUILD_DIR
fi

# Create build directory
log_info "Creating build directory..."
mkdir -p $BUILD_DIR
cd $BUILD_DIR

# Configure with CMake
log_info "Configuring with CMake..."

CMAKE_ARGS=(
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE
    -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX
    -DPYTHON_EXECUTABLE=$PYTHON_EXECUTABLE
)

if [ "$VERBOSE" = true ]; then
    CMAKE_ARGS+=(-DCMAKE_VERBOSE_MAKEFILE=ON)
fi

# Platform-specific configurations
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS specific settings
    CMAKE_ARGS+=(-DCMAKE_OSX_DEPLOYMENT_TARGET=10.15)
    log_info "Platform: macOS"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux specific settings
    log_info "Platform: Linux"
    CMAKE_ARGS+=(-DCMAKE_POSITION_INDEPENDENT_CODE=ON)
fi

cmake "${CMAKE_ARGS[@]}" ..

# Build the project
log_info "Building C++ modules..."

if [ "$VERBOSE" = true ]; then
    make -j$NUM_JOBS VERBOSE=1
else
    make -j$NUM_JOBS
fi

log_info "Build completed successfully!"

# Run tests if requested
if [ "$RUN_TESTS" = true ]; then
    log_info "Running tests..."
    if [ -f "test_runner" ]; then
        ./test_runner
    else
        log_warn "Test runner not found, skipping tests"
    fi
fi

# Install Python module if requested
if [ "$INSTALL_MODULE" = true ]; then
    log_info "Installing Python module..."
    
    # Copy the built module to Python site-packages or current directory
    MODULE_PATH=$(find . -name "etiobench_cpp*.so" -o -name "etiobench_cpp*.pyd" | head -1)
    if [ -n "$MODULE_PATH" ]; then
        # Install in development mode
        cd ..
        $PYTHON_EXECUTABLE -m pip install -e .
        log_info "Module installed in development mode"
    else
        log_error "Built module not found"
        exit 1
    fi
fi

# Run benchmark if requested
if [ "$BENCHMARK" = true ]; then
    log_info "Running performance benchmarks..."
    
    cd ..
    
    # Create benchmark script
    cat > benchmark_test.py << 'EOF'
import sys
import time
try:
    from python_integration import benchmark_implementations, get_performance_summary
    
    print("=== Performance Summary ===")
    summary = get_performance_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    print("\n=== Benchmark Results ===")
    for size in [1000, 10000, 100000]:
        print(f"\nTesting with {size} data points:")
        results = benchmark_implementations(size)
        
        if 'cpp_benchmark' in results:
            cpp_perf = results['cpp_benchmark']['performance_mops']
            print(f"  C++: {cpp_perf:.2f} MOPS")
        
        if 'python_benchmark' in results:
            py_perf = results['python_benchmark']['performance_mops']
            print(f"  Python: {py_perf:.2f} MOPS")
        
        if 'speedup' in results:
            print(f"  Speedup: {results['speedup']:.2f}x")

except ImportError as e:
    print(f"Cannot import module: {e}")
    sys.exit(1)
EOF

    $PYTHON_EXECUTABLE benchmark_test.py
    rm -f benchmark_test.py
fi

log_info "All operations completed successfully!"

# Print usage instructions
echo ""
echo -e "${GREEN}=== Usage Instructions ===${NC}"
echo "The C++ modules have been built successfully."
echo ""
echo "To use the modules in Python:"
echo "  from cpp_core.python_integration import create_statistical_analyzer"
echo "  analyzer = create_statistical_analyzer(prefer_cpp=True)"
echo ""
echo "Or import directly:"
echo "  import etiobench_cpp"
echo "  analyzer = etiobench_cpp.analysis.StatisticalAnalyzer()"
echo ""
echo "Module files:"
if [ -f "$BUILD_DIR/python_bindings/etiobench_cpp.so" ]; then
    echo "  $(realpath $BUILD_DIR/python_bindings/etiobench_cpp.so)"
elif [ -f "$BUILD_DIR/python_bindings/etiobench_cpp.pyd" ]; then
    echo "  $(realpath $BUILD_DIR/python_bindings/etiobench_cpp.pyd)"
fi
echo "  $(realpath python_integration.py)"
echo ""

if [ "$INSTALL_MODULE" = false ]; then
    echo "To install the module system-wide, run:"
    echo "  $0 --install"
    echo ""
fi

echo -e "${GREEN}Build process completed!${NC}"
