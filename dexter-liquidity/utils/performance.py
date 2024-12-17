from typing import Dict, Any, List, Callable
import logging
from .memory_profiler import MemoryProfiler
from .parallel_processor import ParallelDataProcessor

logger = logging.getLogger(__name__)

class PerformanceManager:
    """Manages performance optimization across the system"""
    
    def __init__(
        self,
        enable_memory_profiling: bool = True,
        enable_parallel_processing: bool = True,
        n_processes: int = None,
        chunk_size: int = 1000
    ):
        self.memory_profiler = MemoryProfiler() if enable_memory_profiling else None
        self.parallel_processor = (
            ParallelDataProcessor(n_processes, chunk_size)
            if enable_parallel_processing else None
        )
        logger.info(
            f"Performance manager initialized with memory_profiling={enable_memory_profiling}, "
            f"parallel_processing={enable_parallel_processing}"
        )

    def profile_memory(self, func: Callable) -> Callable:
        """Decorator for memory profiling"""
        if self.memory_profiler:
            return self.memory_profiler.profile_function(func)
        return func

    async def process_parallel(
        self,
        data: List[Any],
        processing_func: Callable,
        *args,
        **kwargs
    ) -> List[Any]:
        """Process data in parallel when beneficial"""
        if self.parallel_processor and len(data) > self.parallel_processor.chunk_size:
            return await self.parallel_processor.process_large_dataset(
                data, processing_func, *args, **kwargs
            )
        return [processing_func(item, *args, **kwargs) for item in data]

    def get_memory_report(self) -> Dict[str, Any]:
        """Get memory usage report"""
        if self.memory_profiler:
            return self.memory_profiler.generate_report()
        return {}

    @staticmethod
    def should_parallelize(data_size: int, operation_complexity: str) -> bool:
        """Determine if parallelization would be beneficial"""
        if operation_complexity == "O(n)":
            return data_size > 10000
        elif operation_complexity == "O(n^2)":
            return data_size > 1000
        elif operation_complexity == "O(n^3)":
            return data_size > 100
        return False
