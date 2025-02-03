"""
Metrics Module - Handles collection, calculation, and tracking of protocol metrics.

This module provides comprehensive metrics management including:
- Historical metric tracking
- Performance calculations
- Trend analysis
- Cache integration
- Health monitoring
"""

from datetime import datetime, timezone, timedelta
from decimal import Decimal, ROUND_DOWN, InvalidOperation
from typing import Dict, List, Optional, Tuple, Any
import asyncio
import logging
from dataclasses import dataclass
import json

from app.cache.redis_client import CacheManager
from app.core.base_protocol import TokenInfo, PoolMetrics, PoolStatus, MetricsError

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class HistoricalMetrics:
    """Container for historical metric data"""
    timestamp: datetime
    tvl_usd: Decimal
    daily_volume_usd: Decimal
    unique_users: int
    active_pools: int
    average_apy: Decimal

    def to_dict(self) -> Dict:
        """Convert to dictionary format for storage"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "tvl_usd": str(self.tvl_usd),
            "daily_volume_usd": str(self.daily_volume_usd),
            "unique_users": self.unique_users,
            "active_pools": self.active_pools,
            "average_apy": str(self.average_apy)
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'HistoricalMetrics':
        """Create instance from dictionary data"""
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            tvl_usd=Decimal(data["tvl_usd"]),
            daily_volume_usd=Decimal(data["daily_volume_usd"]),
            unique_users=int(data["unique_users"]),
            active_pools=int(data["active_pools"]),
            average_apy=Decimal(data["average_apy"])
        )

class MetricsManager:
    """Manages protocol metrics collection and analysis"""

    def __init__(self, cache_manager: Optional[CacheManager] = None):
        """Initialize metrics manager"""
        self.cache = cache_manager or CacheManager()
        self.METRICS_TTL = timedelta(days=30)  # Store metrics for 30 days
        self.CACHE_PREFIX = "metrics:"
        
    async def record_pool_metrics(self, protocol: str, metrics: PoolMetrics) -> None:
        """
        Record point-in-time pool metrics
        
        Args:
            protocol: Protocol identifier
            metrics: Pool metrics to record
        """
        try:
            # Create cache key with timestamp precision
            timestamp = metrics.last_updated.replace(microsecond=0)
            key = f"{self.CACHE_PREFIX}{protocol}:pool:{metrics.pool_id}:{timestamp.isoformat()}"
            
            # Store metrics
            await self.cache.set(
                key=key,
                value=metrics.to_dict(),
                ttl=self.METRICS_TTL
            )
            
            # Update latest metrics pointer
            latest_key = f"{self.CACHE_PREFIX}{protocol}:pool:{metrics.pool_id}:latest"
            await self.cache.set(
                key=latest_key,
                value=key,
                ttl=self.METRICS_TTL
            )
            
        except Exception as e:
            logger.error(f"Error recording pool metrics: {e}")
            raise MetricsError(f"Failed to record pool metrics: {e}")

    async def calculate_protocol_metrics(
        self,
        protocol: str,
        pools: List[PoolMetrics]
    ) -> HistoricalMetrics:
        """
        Calculate aggregate protocol metrics from pool data
        
        Args:
            protocol: Protocol identifier
            pools: List of pool metrics to aggregate
        
        Returns:
            HistoricalMetrics containing aggregated protocol data
        """
        try:
            if not pools:
                raise MetricsError("No pool data available for metrics calculation")

            # Calculate aggregates
            tvl = sum((pool.tvl_usd for pool in pools), Decimal('0'))
            volume = sum((pool.daily_volume_usd for pool in pools), Decimal('0'))
            
            # Calculate weighted average APY
            total_weight = Decimal('0')
            weighted_apy = Decimal('0')
            
            for pool in pools:
                weight = pool.tvl_usd
                total_weight += weight
                weighted_apy += pool.apy * weight
            
            avg_apy = (
                (weighted_apy / total_weight).quantize(Decimal('0.01'), ROUND_DOWN)
                if total_weight > 0 else Decimal('0')
            )

            return HistoricalMetrics(
                timestamp=datetime.now(timezone.utc),
                tvl_usd=tvl,
                daily_volume_usd=volume,
                unique_users=await self._count_unique_users(protocol),
                active_pools=len([p for p in pools if p.status == PoolStatus.HEALTHY]),
                average_apy=avg_apy
            )
            
        except InvalidOperation as e:
            logger.error(f"Decimal calculation error: {e}")
            raise MetricsError(f"Error in metrics calculation: {e}")
        except Exception as e:
            logger.error(f"Unexpected error calculating protocol metrics: {e}")
            raise MetricsError(f"Failed to calculate protocol metrics: {e}")

    async def get_pool_metrics_history(
        self,
        protocol: str,
        pool_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[PoolMetrics]:
        """
        Retrieve historical metrics for a specific pool
        
        Args:
            protocol: Protocol identifier
            pool_id: Pool identifier
            start_time: Start of time range (optional)
            end_time: End of time range (optional)
            
        Returns:
            List of historical pool metrics
        """
        try:
            # Set default time range if not specified
            end_time = end_time or datetime.now(timezone.utc)
            start_time = start_time or (end_time - timedelta(days=7))

            # Scan for matching keys
            pattern = f"{self.CACHE_PREFIX}{protocol}:pool:{pool_id}:*"
            metrics = []

            async for key in self._scan_keys(pattern):
                # Skip latest pointer key
                if key.endswith(":latest"):
                    continue
                    
                # Extract timestamp and check range
                timestamp = datetime.fromisoformat(key.split(":")[-1])
                if start_time <= timestamp <= end_time:
                    data = await self.cache.get(key)
                    if data:
                        metrics.append(PoolMetrics(**data))

            return sorted(metrics, key=lambda x: x.last_updated)
            
        except Exception as e:
            logger.error(f"Error retrieving pool metrics history: {e}")
            raise MetricsError(f"Failed to retrieve pool metrics history: {e}")

    async def calculate_growth_metrics(
        self,
        protocol: str,
        current: HistoricalMetrics,
        days: int = 7
    ) -> Dict[str, Decimal]:
        """
        Calculate protocol growth metrics over time period
        
        Args:
            protocol: Protocol identifier
            current: Current metrics
            days: Number of days to compare against
            
        Returns:
            Dictionary of growth metrics
        """
        try:
            # Get historical metrics
            historical = await self._get_historical_metrics(protocol, days)
            if not historical:
                return {}

            # Calculate growth percentages
            return {
                "tvl_growth": self._calculate_growth_percentage(
                    current.tvl_usd,
                    historical.tvl_usd
                ),
                "volume_growth": self._calculate_growth_percentage(
                    current.daily_volume_usd,
                    historical.daily_volume_usd
                ),
                "user_growth": self._calculate_growth_percentage(
                    Decimal(current.unique_users),
                    Decimal(historical.unique_users)
                ),
                "pool_growth": self._calculate_growth_percentage(
                    Decimal(current.active_pools),
                    Decimal(historical.active_pools)
                ),
                "apy_change": current.average_apy - historical.average_apy
            }
            
        except Exception as e:
            logger.error(f"Error calculating growth metrics: {e}")
            raise MetricsError(f"Failed to calculate growth metrics: {e}")

    async def _get_historical_metrics(
        self,
        protocol: str,
        days_ago: int
    ) -> Optional[HistoricalMetrics]:
        """Retrieve historical metrics from N days ago"""
        try:
            target_date = datetime.now(timezone.utc) - timedelta(days=days_ago)
            key_pattern = f"{self.CACHE_PREFIX}{protocol}:historical:{target_date.date().isoformat()}*"
            
            async for key in self._scan_keys(key_pattern):
                data = await self.cache.get(key)
                if data:
                    return HistoricalMetrics.from_dict(data)
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving historical metrics: {e}")
            return None

    async def _count_unique_users(self, protocol: str) -> int:
        """Count unique users in the last 24 hours"""
        try:
            # This would typically integrate with your user tracking system
            # Placeholder implementation
            return 1000
        except Exception as e:
            logger.error(f"Error counting unique users: {e}")
            return 0

    def _calculate_growth_percentage(
        self,
        current: Decimal,
        historical: Decimal
    ) -> Decimal:
        """Calculate percentage growth between two values"""
        if historical == 0:
            return Decimal('0')
        return ((current - historical) / historical * Decimal('100')
                .quantize(Decimal('0.01'), ROUND_DOWN))

    async def _scan_keys(self, pattern: str):
        """Helper to scan Redis keys matching pattern"""
        cursor = 0
        while True:
            cursor, keys = await self.cache.redis.scan(
                cursor,
                match=pattern
            )
            for key in keys:
                yield key.decode('utf-8')
            if cursor == 0:
                break

    async def cleanup_old_metrics(self) -> None:
        """Clean up expired metrics data"""
        try:
            cutoff = datetime.now(timezone.utc) - self.METRICS_TTL
            pattern = f"{self.CACHE_PREFIX}*"
            
            async for key in self._scan_keys(pattern):
                try:
                    timestamp_str = key.split(":")[-1]
                    if timestamp_str == "latest":
                        continue
                        
                    timestamp = datetime.fromisoformat(timestamp_str)
                    if timestamp < cutoff:
                        await self.cache.delete(key)
                except (ValueError, IndexError):
                    continue
                    
        except Exception as e:
            logger.error(f"Error cleaning up old metrics: {e}")