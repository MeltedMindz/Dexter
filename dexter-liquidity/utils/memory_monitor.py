from memory_profiler import profile
import psutil
import logging
from typing import Dict, Any
import time
from functools import wraps
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class MemoryStats:
    peak_usage: float
    average_usage: float
    num_samples: int
    timestamp: float

class MemoryMonitor:
    def __init__(self, sampling_interval: float = 0.1):
        self.sampling_interval = sampling_interval
        self.measurements: Dict[str, List[float]] = {}
        self.stats: Dict[str, MemoryStats] = {}
        
    def monitor(self, func):
        """Decorator to monitor memory usage"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            process = psutil.Process()
            samples = []
            
            result = func(*args, **kwargs)
            
            # Record memory usage
            memory_info = process.memory_info()
            samples.append(memory_info.rss / 1024 / 1024)  # Convert to MB
            
            # Update statistics
            self.measurements[func.__name__] = samples
            self.stats[func.__name__] = MemoryStats(
                peak_usage=max(samples),
                average_usage=np.mean(samples),
                num_samples=len(samples),
                timestamp=start_time
            )
            
            if max(samples) > 1000:  # Alert if over 1GB
                logger.warning(
                    f"High memory usage detected in {func.__name__}: "
                    f"{max(samples):.2f} MB"
                )
                
            return result
            
        return wrapper
        
    def get_stats(self) -> Dict[str, MemoryStats]:
        """Get memory statistics"""
        return self.stats.copy()
        
    def clear_stats(self):
        """Clear collected statistics"""
        self.measurements.clear()
        self.stats.clear()
