import psutil
import os
from functools import wraps
from typing import Dict, Any
import time
import logging

logger = logging.getLogger(__name__)

class MemoryProfiler:
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.baseline_memory = self.get_current_memory()
        self.reports = []
        
    def get_current_memory(self) -> float:
        """Get current memory usage in MB"""
        return self.process.memory_info().rss / 1024 / 1024
        
    def profile_function(self, func):
        """Decorator to profile function memory usage"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_memory = self.get_current_memory()
            start_time = time.time()
            
            result = func(*args, **kwargs)
            
            end_memory = self.get_current_memory()
            end_time = time.time()
            
            report = {
                'function': func.__name__,
                'memory_start_mb': start_memory,
                'memory_end_mb': end_memory,
                'memory_used_mb': end_memory - start_memory,
                'execution_time': end_time - start_time
            }
            
            self.reports.append(report)
            logger.info(f"Memory profile for {func.__name__}: {report}")
            
            return result
        return wrapper
        
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive memory usage report"""
        current_memory = self.get_current_memory()
        total_used = current_memory - self.baseline_memory
        
        report = {
            'baseline_memory_mb': self.baseline_memory,
            'current_memory_mb': current_memory,
            'total_memory_used_mb': total_used,
            'function_reports': self.reports,
            'memory_intensive_functions': sorted(
                self.reports,
                key=lambda x: x['memory_used_mb'],
                reverse=True
            )[:5]
        }
        
        return report
