from typing import Dict, List, Optional, Any, Tuple, Set
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio
import logging
import time
import heapq
from enum import Enum
import numpy as np
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)

class EvictionReason(Enum):
    LFU = "least_frequently_used"
    LRU = "least_recently_used"
    SIZE = "size_limit"
    TTL = "time_to_live"
    ADAPTIVE = "adaptive"
    COST = "cost_based"

@dataclass
class CacheItemStats:
    frequency: int
    last_access: datetime
    size: int
    cost: float
    ttl: int
    hits: int
    misses: int
    creation_time: datetime
    last_write: datetime

class EvictionStrategy(ABC):
    @abstractmethod
    def should_evict(self, key: str, item: Any, stats: CacheItemStats) -> Tuple[bool, float]:
        """Return whether to evict and the eviction score"""
        pass

class AdaptiveLFUStrategy(EvictionStrategy):
    """Adaptive LFU with aging mechanism"""
    def __init__(self, aging_factor: float = 0.95, window_size: int = 3600):
        self.aging_factor = aging_factor
        self.window_size = window_size

    def should_evict(self, key: str, item: Any, stats: CacheItemStats) -> Tuple[bool, float]:
        age_seconds = (datetime.now() - stats.creation_time).total_seconds()
        windows = age_seconds / self.window_size
        
        # Apply aging to frequency
        aged_frequency = stats.frequency * (self.aging_factor ** windows)
        
        # Calculate hit ratio
        total_accesses = stats.hits + stats.misses
        hit_ratio = stats.hits / total_accesses if total_accesses > 0 else 0
        
        # Combine metrics for eviction score
        score = aged_frequency * hit_ratio
        return score < 0.1, score

class CostAwareStrategy(EvictionStrategy):
    """Cost-based eviction considering computation and fetch costs"""
    def should_evict(self, key: str, item: Any, stats: CacheItemStats) -> Tuple[bool, float]:
        # Calculate cost-effectiveness
        time_alive = (datetime.now() - stats.creation_time).total_seconds()
        cost_per_second = stats.cost / time_alive if time_alive > 0 else float('inf')
        
        # Factor in access frequency
        access_rate = stats.frequency / time_alive if time_alive > 0 else 0
        
        # Combine metrics
        effectiveness = access_rate / cost_per_second if cost_per_second > 0 else float('inf')
        return effectiveness < 0.05, 1/effectiveness

class SizeAwareStrategy(EvictionStrategy):
    """Size-based eviction with memory pressure consideration"""
    def __init__(self, target_size_mb: int = 100):
        self.target_size_bytes = target_size_mb * 1024 * 1024

    def should_evict(self, key: str, item: Any, stats: CacheItemStats) -> Tuple[bool, float]:
        # Calculate size efficiency
        size_efficiency = stats.frequency / stats.size if stats.size > 0 else float('inf')
        
        # Consider time-based decay
        age_hours = (datetime.now() - stats.creation_time).total_seconds() / 3600
        decay_factor = 0.9 ** age_hours
        
        score = size_efficiency * decay_factor
        return score < 0.001, score

class BatchOptimizer:
    """Intelligent batch request optimizer"""
    def __init__(
        self,
        min_batch_size: int = 10,
        max_batch_size: int = 100,
        initial_wait_time: float = 0.1
    ):
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.wait_time = initial_wait_time
        self.performance_history: List[Tuple[int, float]] = []  # (batch_size, latency)
        self.request_patterns: Counter = Counter()
        self.last_optimization = datetime.now()
        
    async def optimize_batch_parameters(self):
        """Dynamically adjust batch parameters based on performance"""
        if len(self.performance_history) < 10:
            return

        recent_history = self.performance_history[-50:]
        
        # Calculate optimal batch size based on latency
        batch_sizes, latencies = zip(*recent_history)
        
        # Use numpy for efficient calculations
        batch_sizes = np.array(batch_sizes)
        latencies = np.array(latencies)
        
        # Calculate throughput for each batch size
        throughputs = batch_sizes / latencies
        
        # Find batch size with best throughput
        optimal_index = np.argmax(throughputs)
        optimal_batch_size = batch_sizes[optimal_index]
        
        # Update batch size within bounds
        self.max_batch_size = min(
            max(
                self.min_batch_size,
                int(optimal_batch_size * 1.1)  # Allow 10% growth
            ),
            100  # Hard limit
        )
        
        # Adjust wait time based on request patterns
        self.adjust_wait_time()
        
        logger.info(f"meteora_fetcher.py: Optimized batch parameters - size: {self.max_batch_size}, wait: {self.wait_time}")

    def adjust_wait_time(self):
        """Adjust wait time based on request patterns"""
        total_requests = sum(self.request_patterns.values())
        if total_requests == 0:
            return

        # Calculate request rate per second
        request_rate = total_requests / 3600  # requests per second
        
        # Adjust wait time based on rate
        if request_rate > self.max_batch_size * 2:
            # High request rate - decrease wait time
            self.wait_time = max(0.05, self.wait_time * 0.8)
        elif request_rate < self.max_batch_size / 2:
            # Low request rate - increase wait time
            self.wait_time = min(0.5, self.wait_time * 1.2)

    async def process_batch(
        self,
        pending_requests: List[Tuple[str, asyncio.Future]]
    ) -> None:
        """Process a batch of requests with optimization"""
        batch_size = len(pending_requests)
        start_time = time.time()
        
        try:
            # Execute batch request
            results = await self._execute_batch([addr for addr, _ in pending_requests])
            
            # Record performance metrics
            latency = time.time() - start_time
            self.performance_history.append((batch_size, latency))
            self.request_patterns[datetime.now().hour] += batch_size
            
            # Trigger optimization periodically
            if (datetime.now() - self.last_optimization).total_seconds() > 300:  # 5 minutes
                await self.optimize_batch_parameters()
                self.last_optimization = datetime.now()
            
        except Exception as e:
            logger.error(f"meteora_fetcher.py: Batch processing error: {str(e)}")
            raise
        finally:
            # Clean up old history
            if len(self.performance_history) > 1000:
                self.performance_history = self.performance_history[-500:]

    async def _execute_batch(self, addresses: List[str]) -> List[Any]:
        """Execute actual batch request"""
        # Implementation depends on API capabilities
        pass

class AdaptiveCache:
    """Advanced cache with multiple eviction strategies"""
    def __init__(self, max_size_mb: int = 100):
        self.data: Dict[str, Any] = {}
        self.stats: Dict[str, CacheItemStats] = {}
        self.max_size = max_size_mb * 1024 * 1024
        
        # Initialize eviction strategies
        self.strategies = [
            AdaptiveLFUStrategy(),
            CostAwareStrategy(),
            SizeAwareStrategy(max_size_mb)
        ]
        
        # Start background optimization
        asyncio.create_task(self._run_optimization())

    async def _run_optimization(self):
        """Periodic cache optimization"""
        while True:
            try:
                await self._optimize_cache()
                await asyncio.sleep(60)  # Run every minute
            except Exception as e:
                logger.error(f"meteora_fetcher.py: Cache optimization error: {str(e)}")

    async def _optimize_cache(self):
        """Optimize cache based on usage patterns"""
        current_size = sum(stat.size for stat in self.stats.values())
        if current_size > self.max_size * 0.9:  # Over 90% capacity
            await self._evict_items()

    async def _evict_items(self):
        """Evict items based on combined strategy scores"""
        eviction_scores: List[Tuple[float, str]] = []
        
        for key, item in self.data.items():
            stats = self.stats[key]
            
            # Combine scores from all strategies
            combined_score = 0
            for strategy in self.strategies:
                should_evict, score = strategy.should_evict(key, item, stats)
                if should_evict:
                    combined_score += score
            
            if combined_score > 0:
                eviction_scores.append((combined_score, key))
        
        # Evict worst items
        if eviction_scores:
            heapq.heapify(eviction_scores)
            worst_items = heapq.nsmallest(
                len(eviction_scores) // 4,  # Evict up to 25%
                eviction_scores
            )
            
            for _, key in worst_items:
                del self.data[key]
                del self.stats[key]