import multiprocessing as mp
from typing import List, Callable, Any
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import logging

logger = logging.getLogger(__name__)

class ParallelDataProcessor:
    def __init__(
        self,
        n_processes: int = None,
        chunk_size: int = 1000,
        use_threads: bool = False
    ):
        self.n_processes = n_processes or mp.cpu_count()
        self.chunk_size = chunk_size
        self.use_threads = use_threads
        logger.info(f"Initialized parallel processor with {self.n_processes} processes")
        
    def process_large_dataset(
        self,
        data: List[Any],
        processing_func: Callable,
        *args,
        **kwargs
    ) -> List[Any]:
        """Process large dataset in parallel"""
        try:
            # Split data into chunks
            chunks = self._create_chunks(data)
            
            # Choose executor based on configuration
            Executor = ThreadPoolExecutor if self.use_threads else ProcessPoolExecutor
            
            with Executor(max_workers=self.n_processes) as executor:
                # Process chunks in parallel
                futures = [
                    executor.submit(
                        processing_func,
                        chunk,
                        *args,
                        **kwargs
                    )
                    for chunk in chunks
                ]
                
                # Collect results
                results = [future.result() for future in futures]
                
            # Combine results
            combined_results = self._combine_results(results)
            
            logger.info(f"Processed {len(data)} items in parallel")
            return combined_results
            
        except Exception as e:
            logger.error(f"Parallel processing failed: {str(e)}")
            raise
            
    def _create_chunks(self, data: List[Any]) -> List[List[Any]]:
        """Split data into chunks for parallel processing"""
        return [
            data[i:i + self.chunk_size]
            for i in range(0, len(data), self.chunk_size)
        ]
        
    def _combine_results(self, results: List[Any]) -> List[Any]:
        """Combine results from parallel processing"""
        if isinstance(results[0], np.ndarray):
            return np.concatenate(results)
        elif isinstance(results[0], list):
            return [item for sublist in results for item in sublist]
        else:
            return results
