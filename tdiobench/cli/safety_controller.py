"""
SafetyController - System Resource Monitoring for eTIOBench
"""

import psutil
import os
import signal
import logging
import threading
import time
from typing import Dict, Optional, Callable

logger = logging.getLogger(__name__)


class SafetyController:
    """Monitors system resources during benchmarks to prevent system overload"""
    
    def __init__(self, max_cpu_percent: float = 90, max_memory_percent: float = 90,
                 check_interval: float = 1.0):
        """
        Initialize SafetyController
        
        Args:
            max_cpu_percent: Maximum allowed CPU usage percentage
            max_memory_percent: Maximum allowed memory usage percentage
            check_interval: Resource check interval in seconds
        """
        self.max_cpu_percent = max_cpu_percent
        self.max_memory_percent = max_memory_percent
        self.check_interval = check_interval
        self.monitoring = False
        self.throttled = False
        self.emergency_callback = None
        self.throttle_callback = None
        self._monitor_thread = None
        self._benchmark_pid = None
        
    def start_monitoring(self, benchmark_pid: Optional[int] = None,
                        emergency_callback: Optional[Callable] = None,
                        throttle_callback: Optional[Callable] = None):
        """Start resource monitoring"""
        self.monitoring = True
        self._benchmark_pid = benchmark_pid
        self.emergency_callback = emergency_callback
        self.throttle_callback = throttle_callback
        
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info(f"Safety monitoring started (CPU: {self.max_cpu_percent}%, Memory: {self.max_memory_percent}%)")
    
    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2)
        logger.info("Safety monitoring stopped")
    
    def check_resources(self) -> Dict[str, float]:
        """
        Check current resource usage
        
        Returns:
            Dict with cpu_percent, memory_percent, disk_free_gb
        """
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        resources = {
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_available_gb': memory.available / (1024**3),
            'disk_free_gb': disk.free / (1024**3),
            'disk_percent': disk.percent
        }
        
        # Check if we're exceeding limits
        if cpu_percent > self.max_cpu_percent:
            logger.warning(f"CPU usage {cpu_percent:.1f}% exceeds limit {self.max_cpu_percent}%")
            resources['cpu_exceeded'] = True
            
        if memory.percent > self.max_memory_percent:
            logger.warning(f"Memory usage {memory.percent:.1f}% exceeds limit {self.max_memory_percent}%")
            resources['memory_exceeded'] = True
        
        return resources
    
    def throttle_benchmark(self):
        """Reduce benchmark load by sending SIGSTOP/SIGCONT signals"""
        if not self._benchmark_pid:
            logger.warning("No benchmark PID to throttle")
            return
            
        try:
            # Pause benchmark process
            os.kill(self._benchmark_pid, signal.SIGSTOP)
            self.throttled = True
            logger.info(f"Throttled benchmark process {self._benchmark_pid}")
            
            # Wait for resources to recover
            time.sleep(2)
            
            # Resume with reduced priority
            os.kill(self._benchmark_pid, signal.SIGCONT)
            os.nice(5)  # Lower priority
            
            if self.throttle_callback:
                self.throttle_callback()
                
        except ProcessLookupError:
            logger.error("Benchmark process not found")
        except Exception as e:
            logger.error(f"Failed to throttle benchmark: {e}")
    
    def emergency_stop(self):
        """Immediately halt benchmark due to critical resource usage"""
        logger.critical("EMERGENCY STOP - Critical resource limits exceeded")
        
        if self._benchmark_pid:
            try:
                # Try graceful termination first
                os.kill(self._benchmark_pid, signal.SIGTERM)
                time.sleep(1)
                
                # Force kill if still running
                try:
                    os.kill(self._benchmark_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass  # Already terminated
                    
            except Exception as e:
                logger.error(f"Failed to stop benchmark: {e}")
        
        if self.emergency_callback:
            self.emergency_callback()
        
        self.stop_monitoring()
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        consecutive_violations = 0
        
        while self.monitoring:
            try:
                resources = self.check_resources()
                
                # Check for violations
                if resources.get('cpu_exceeded') or resources.get('memory_exceeded'):
                    consecutive_violations += 1
                    
                    if consecutive_violations >= 3:
                        # Critical - emergency stop
                        self.emergency_stop()
                        break
                    elif consecutive_violations >= 2:
                        # Throttle to reduce load
                        self.throttle_benchmark()
                else:
                    consecutive_violations = 0
                    if self.throttled:
                        logger.info("Resources recovered, resuming normal operation")
                        self.throttled = False
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                time.sleep(self.check_interval)
    
    def set_limits(self, cpu_percent: Optional[float] = None, 
                   memory_percent: Optional[float] = None):
        """Update resource limits"""
        if cpu_percent is not None:
            self.max_cpu_percent = cpu_percent
        if memory_percent is not None:
            self.max_memory_percent = memory_percent
        logger.info(f"Updated limits - CPU: {self.max_cpu_percent}%, Memory: {self.max_memory_percent}%")
