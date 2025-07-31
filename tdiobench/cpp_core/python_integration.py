"""
C++ Integration Layer for eTIOBench

This module provides seamless integration between C++ performance modules
and Python implementations, enabling gradual migration and fallback capabilities.
"""

import logging
import time
import warnings
from typing import Optional, Any, Dict, List, Union
from functools import wraps

# Try to import C++ modules
try:
    import etiobench_cpp
    CPP_AVAILABLE = True
    _cpp_modules = {
        'statistical_analyzer': etiobench_cpp.analysis,
        'data_processor': etiobench_cpp.core,
        'time_series_collector': etiobench_cpp.collection,
        'system_metrics_collector': etiobench_cpp.collection,
    }
    
    # Log successful import
    logger = logging.getLogger(__name__)
    perf_info = etiobench_cpp.performance_info
    logger.info(f"C++ modules loaded successfully - Expected speedup: {perf_info['expected_speedup']}")
    logger.info(f"SIMD enabled: {perf_info['simd_enabled']}, OpenMP enabled: {perf_info['openmp_enabled']}")
    
except ImportError as e:
    CPP_AVAILABLE = False
    _cpp_modules = {}
    warnings.warn(f"C++ modules not available, falling back to Python: {e}")

# Import Python implementations
try:
    from tdiobench.analysis.statistical_analyzer import StatisticalAnalyzer as PythonStatisticalAnalyzer
    from tdiobench.core.benchmark_data import DataProcessor as PythonDataProcessor
    from tdiobench.collection.time_series_collector import TimeSeriesCollector as PythonTimeSeriesCollector
    from tdiobench.collection.system_metrics_collector import SystemMetricsCollector as PythonSystemMetricsCollector
    PYTHON_AVAILABLE = True
except ImportError as e:
    PYTHON_AVAILABLE = False
    warnings.warn(f"Python implementations not available: {e}")


class PerformanceMonitor:
    """Monitor and compare performance between C++ and Python implementations."""
    
    def __init__(self):
        self.performance_data = {}
    
    def benchmark_call(self, func_name: str, cpp_func, python_func, *args, **kwargs):
        """Benchmark both C++ and Python implementations."""
        results = {}
        
        # Benchmark C++ implementation
        if cpp_func:
            start_time = time.perf_counter()
            try:
                cpp_result = cpp_func(*args, **kwargs)
                cpp_time = time.perf_counter() - start_time
                results['cpp'] = {
                    'time': cpp_time,
                    'result': cpp_result,
                    'success': True,
                    'error': None
                }
            except Exception as e:
                results['cpp'] = {
                    'time': 0,
                    'result': None,
                    'success': False,
                    'error': str(e)
                }
        
        # Benchmark Python implementation
        if python_func:
            start_time = time.perf_counter()
            try:
                python_result = python_func(*args, **kwargs)
                python_time = time.perf_counter() - start_time
                results['python'] = {
                    'time': python_time,
                    'result': python_result,
                    'success': True,
                    'error': None
                }
            except Exception as e:
                results['python'] = {
                    'time': 0,
                    'result': None,
                    'success': False,
                    'error': str(e)
                }
        
        # Calculate speedup
        if (results.get('cpp', {}).get('success') and 
            results.get('python', {}).get('success')):
            speedup = results['python']['time'] / results['cpp']['time']
            results['speedup'] = speedup
        
        self.performance_data[func_name] = results
        return results


class HybridImplementation:
    """Base class for hybrid C++/Python implementations with automatic fallback."""
    
    def __init__(self, prefer_cpp: bool = True, enable_fallback: bool = True):
        self.prefer_cpp = prefer_cpp and CPP_AVAILABLE
        self.enable_fallback = enable_fallback
        self.performance_monitor = PerformanceMonitor()
        self._cpp_impl = None
        self._python_impl = None
    
    def _get_implementation(self):
        """Get the preferred implementation with fallback."""
        if self.prefer_cpp and self._cpp_impl:
            return 'cpp', self._cpp_impl
        elif self.enable_fallback and self._python_impl:
            return 'python', self._python_impl
        else:
            raise RuntimeError("No suitable implementation available")
    
    def _call_with_fallback(self, method_name: str, *args, **kwargs):
        """Call method with automatic fallback on failure."""
        primary_type, primary_impl = self._get_implementation()
        
        try:
            method = getattr(primary_impl, method_name)
            return method(*args, **kwargs)
        except Exception as e:
            if not self.enable_fallback:
                raise
            
            # Try fallback implementation
            fallback_impl = self._python_impl if primary_type == 'cpp' else self._cpp_impl
            if fallback_impl:
                try:
                    fallback_method = getattr(fallback_impl, method_name)
                    warnings.warn(f"Fallback to {'Python' if primary_type == 'cpp' else 'C++'} "
                                f"implementation for {method_name}: {e}")
                    return fallback_method(*args, **kwargs)
                except Exception as fallback_error:
                    raise RuntimeError(f"Both implementations failed. "
                                     f"Primary ({primary_type}): {e}, "
                                     f"Fallback: {fallback_error}")
            else:
                raise


class StatisticalAnalyzer(HybridImplementation):
    """Hybrid Statistical Analyzer with C++ acceleration and Python fallback."""
    
    def __init__(self, config: Optional[Dict] = None, prefer_cpp: bool = True):
        super().__init__(prefer_cpp=prefer_cpp)
        
        # Initialize C++ implementation
        if CPP_AVAILABLE and prefer_cpp:
            try:
                cpp_config = etiobench_cpp.analysis.StatisticalAnalyzerConfig()
                if config:
                    # Map Python config to C++ config
                    if 'confidence_level' in config:
                        cpp_config.confidence_level = config['confidence_level']
                    if 'outlier_threshold' in config:
                        cpp_config.outlier_threshold = config['outlier_threshold']
                    if 'enable_parallel_processing' in config:
                        cpp_config.enable_parallel_processing = config['enable_parallel_processing']
                
                self._cpp_impl = etiobench_cpp.analysis.StatisticalAnalyzer(cpp_config)
            except Exception as e:
                warnings.warn(f"Failed to initialize C++ StatisticalAnalyzer: {e}")
                self._cpp_impl = None
        
        # Initialize Python implementation
        if PYTHON_AVAILABLE:
            try:
                self._python_impl = PythonStatisticalAnalyzer(config or {})
            except Exception as e:
                warnings.warn(f"Failed to initialize Python StatisticalAnalyzer: {e}")
                self._python_impl = None
    
    def analyze_benchmark_results(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze benchmark results with automatic implementation selection."""
        return self._call_with_fallback('analyze_benchmark_results', data)
    
    def calculate_percentiles(self, data: List[float], percentiles: List[float]) -> List[float]:
        """Calculate percentiles with optimal implementation."""
        return self._call_with_fallback('calculate_percentiles', data, percentiles)
    
    def detect_outliers(self, data: List[float], method: str = 'iqr') -> List[int]:
        """Detect outliers using the fastest available implementation."""
        return self._call_with_fallback('detect_outliers', data, method)
    
    def get_performance_info(self) -> Dict[str, Any]:
        """Get performance information about the current implementation."""
        impl_type, impl = self._get_implementation()
        info = {
            'implementation': impl_type,
            'cpp_available': CPP_AVAILABLE,
            'python_available': PYTHON_AVAILABLE,
        }
        
        if impl_type == 'cpp' and CPP_AVAILABLE:
            info.update(etiobench_cpp.performance_info)
        
        return info


class DataProcessor(HybridImplementation):
    """Hybrid Data Processor with C++ acceleration and Python fallback."""
    
    def __init__(self, config: Optional[Dict] = None, prefer_cpp: bool = True):
        super().__init__(prefer_cpp=prefer_cpp)
        
        # Initialize C++ implementation
        if CPP_AVAILABLE and prefer_cpp:
            try:
                cpp_config = etiobench_cpp.core.DataProcessorConfig()
                if config:
                    # Map Python config to C++ config
                    if 'max_chunk_size' in config:
                        cpp_config.max_chunk_size = config['max_chunk_size']
                    if 'num_threads' in config:
                        cpp_config.num_threads = config['num_threads']
                    if 'enable_simd' in config:
                        cpp_config.enable_simd = config['enable_simd']
                
                self._cpp_impl = etiobench_cpp.core.DataProcessor(cpp_config)
            except Exception as e:
                warnings.warn(f"Failed to initialize C++ DataProcessor: {e}")
                self._cpp_impl = None
        
        # Initialize Python implementation
        if PYTHON_AVAILABLE:
            try:
                self._python_impl = PythonDataProcessor(config or {})
            except Exception as e:
                warnings.warn(f"Failed to initialize Python DataProcessor: {e}")
                self._python_impl = None
    
    def transform_data(self, data: Dict, transformation: str) -> Dict:
        """Transform data using optimal implementation."""
        # Convert Python dict to C++ TimeSeriesData if using C++ implementation
        impl_type, impl = self._get_implementation()
        
        if impl_type == 'cpp':
            # Convert to C++ format
            ts_data = etiobench_cpp.core.dict_to_time_series_data(data)
            
            # Apply transformation
            if transformation.upper() in ['LOG', 'SQRT', 'DIFFERENCE', 'PERCENT_CHANGE', 'Z_SCORE', 'MIN_MAX', 'STANDARDIZE']:
                transform_type = getattr(etiobench_cpp.core.DataProcessor.TransformationType, transformation.upper())
                result_ts = impl.transform_data(ts_data, transform_type)
                
                # Convert back to Python dict
                return etiobench_cpp.core.time_series_data_to_dict(result_ts)
            else:
                raise ValueError(f"Unsupported transformation: {transformation}")
        else:
            return impl.transform_data(data, transformation)
    
    def aggregate_time_series(self, data: Dict, target_interval: float, 
                             aggregation_func: str = 'mean') -> Dict:
        """Aggregate time series data using optimal implementation."""
        return self._call_with_fallback('aggregate_time_series', data, target_interval, aggregation_func)


class TimeSeriesCollector(HybridImplementation):
    """Hybrid Time Series Collector with C++ acceleration and Python fallback."""
    
    def __init__(self, config: Optional[Dict] = None, prefer_cpp: bool = True):
        super().__init__(prefer_cpp=prefer_cpp)
        
        # Initialize C++ implementation
        if CPP_AVAILABLE and prefer_cpp:
            try:
                cpp_config = etiobench_cpp.collection.CollectionConfig()
                if config:
                    # Map Python config to C++ config
                    if 'buffer_size' in config:
                        cpp_config.buffer_size = config['buffer_size']
                    if 'num_worker_threads' in config:
                        cpp_config.num_worker_threads = config['num_worker_threads']
                    if 'default_interval_ms' in config:
                        cpp_config.default_interval_ms = config['default_interval_ms']
                
                self._cpp_impl = etiobench_cpp.collection.TimeSeriesCollector(cpp_config)
            except Exception as e:
                warnings.warn(f"Failed to initialize C++ TimeSeriesCollector: {e}")
                self._cpp_impl = None
        
        # Initialize Python implementation
        if PYTHON_AVAILABLE:
            try:
                self._python_impl = PythonTimeSeriesCollector(config or {})
            except Exception as e:
                warnings.warn(f"Failed to initialize Python TimeSeriesCollector: {e}")
                self._python_impl = None
    
    def start_collection(self):
        """Start data collection."""
        return self._call_with_fallback('start_collection')
    
    def stop_collection(self):
        """Stop data collection."""
        return self._call_with_fallback('stop_collection')
    
    def collect_point(self, timestamp: float, value: float, metric_name: str, tier: str = ""):
        """Collect a single data point."""
        impl_type, impl = self._get_implementation()
        
        if impl_type == 'cpp':
            data_point = etiobench_cpp.collection.DataPoint(timestamp, value, metric_name, tier)
            return impl.collect_point(data_point)
        else:
            return impl.collect_point(timestamp, value, metric_name, tier)
    
    def get_collected_data(self) -> Dict:
        """Get all collected data."""
        return self._call_with_fallback('get_collected_data')


def get_performance_summary() -> Dict[str, Any]:
    """Get a summary of C++/Python performance characteristics."""
    summary = {
        'cpp_available': CPP_AVAILABLE,
        'python_available': PYTHON_AVAILABLE,
        'modules': {}
    }
    
    if CPP_AVAILABLE:
        summary['cpp_info'] = etiobench_cpp.performance_info
        summary['modules']['cpp'] = list(_cpp_modules.keys())
    
    if PYTHON_AVAILABLE:
        summary['modules']['python'] = ['statistical_analyzer', 'data_processor', 
                                       'time_series_collector', 'system_metrics_collector']
    
    return summary


def benchmark_implementations(data_size: int = 100000) -> Dict[str, Any]:
    """Benchmark C++ vs Python implementations."""
    results = {}
    
    if CPP_AVAILABLE:
        results['cpp_benchmark'] = etiobench_cpp.benchmark_performance(data_size)
    
    # Simple Python benchmark for comparison
    if PYTHON_AVAILABLE:
        import time
        start_time = time.perf_counter()
        
        # Simulate computation
        data = list(range(data_size))
        result_sum = sum(x * x for x in data)
        
        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000
        
        results['python_benchmark'] = {
            'data_size': data_size,
            'duration_ms': duration_ms,
            'result': result_sum,
            'performance_mops': (data_size / 1000000.0) / (duration_ms / 1000.0)
        }
    
    # Calculate speedup if both available
    if 'cpp_benchmark' in results and 'python_benchmark' in results:
        cpp_time = results['cpp_benchmark']['duration_ms']
        python_time = results['python_benchmark']['duration_ms']
        results['speedup'] = python_time / cpp_time if cpp_time > 0 else float('inf')
    
    return results


# Convenience functions for direct module access
def create_statistical_analyzer(config: Optional[Dict] = None, prefer_cpp: bool = True) -> StatisticalAnalyzer:
    """Create a StatisticalAnalyzer with optimal implementation."""
    return StatisticalAnalyzer(config, prefer_cpp)

def create_data_processor(config: Optional[Dict] = None, prefer_cpp: bool = True) -> DataProcessor:
    """Create a DataProcessor with optimal implementation."""
    return DataProcessor(config, prefer_cpp)

def create_time_series_collector(config: Optional[Dict] = None, prefer_cpp: bool = True) -> TimeSeriesCollector:
    """Create a TimeSeriesCollector with optimal implementation."""
    return TimeSeriesCollector(config, prefer_cpp)


# Module-level configuration
class Config:
    """Global configuration for C++/Python integration."""
    
    prefer_cpp = True
    enable_fallback = True
    log_performance = False
    benchmark_on_startup = False
    
    @classmethod
    def set_cpp_preference(cls, prefer: bool):
        """Set global preference for C++ implementations."""
        cls.prefer_cpp = prefer and CPP_AVAILABLE
    
    @classmethod
    def set_fallback_enabled(cls, enabled: bool):
        """Enable/disable automatic fallback to alternative implementations."""
        cls.enable_fallback = enabled
    
    @classmethod
    def enable_performance_logging(cls, enabled: bool):
        """Enable/disable performance logging."""
        cls.log_performance = enabled


# Initialize logging
def setup_logging():
    """Setup logging for the integration layer."""
    logger = logging.getLogger(__name__)
    
    if CPP_AVAILABLE:
        logger.info("C++ acceleration enabled")
        if hasattr(etiobench_cpp, 'get_simd_support'):
            simd_info = etiobench_cpp.get_simd_support()
            logger.info(f"SIMD support: {simd_info}")
    
    if not CPP_AVAILABLE and not PYTHON_AVAILABLE:
        logger.error("No implementations available!")
    elif not CPP_AVAILABLE:
        logger.warning("C++ implementations not available, using Python fallback")
    elif not PYTHON_AVAILABLE:
        logger.warning("Python implementations not available, using C++ only")


# Module initialization
setup_logging()

if Config.benchmark_on_startup and CPP_AVAILABLE:
    startup_benchmark = benchmark_implementations(10000)
    logging.getLogger(__name__).info(f"Startup benchmark results: {startup_benchmark}")


__all__ = [
    'StatisticalAnalyzer',
    'DataProcessor', 
    'TimeSeriesCollector',
    'HybridImplementation',
    'PerformanceMonitor',
    'Config',
    'get_performance_summary',
    'benchmark_implementations',
    'create_statistical_analyzer',
    'create_data_processor',
    'create_time_series_collector',
    'CPP_AVAILABLE',
    'PYTHON_AVAILABLE'
]
