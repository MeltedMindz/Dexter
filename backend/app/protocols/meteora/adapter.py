"""
Meteora Protocol Adapter Module

This module implements the BaseProtocol interface for Meteora's DLMM protocol.
It handles all protocol-specific logic including:
- Pool data fetching and parsing
- Metric calculations
- Health monitoring
- Cache management
"""

import asyncio
from datetime import datetime, timezone, timedelta
from decimal import Decimal, ROUND_DOWN
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass

from solders.pubkey import Pubkey
from dlmm import DLMM_CLIENT
from dlmm.dlmm import DLMM
from dlmm.types import FeeInfo, GetPositionByUser, StrategyType, SwapQuote

from app.core.base_protocol import (
    BaseProtocol,
    PoolMetrics,
    TokenInfo,
    PoolStatus,
    PoolNotFoundError,
    MetricsError
)
from app.cache.redis_client import CacheManager

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class MeteoraPoolConfig:
    """Configuration for a Meteora pool"""
    address: str
    token_a_symbol: str
    token_b_symbol: str
    token_a_decimals: int
    token_b_decimals: int

class MeteoraAdapter(BaseProtocol):
    """Adapter for Meteora DLMM protocol integration"""

    def __init__(
        self,
        rpc_url: str,
        cache_manager: Optional[CacheManager] = None,
        price_feed_url: Optional[str] = None
    ):
        """
        Initialize Meteora adapter
        
        Args:
            rpc_url: Solana RPC endpoint
            cache_manager: Optional cache manager instance
            price_feed_url: Optional price feed API endpoint
        """
        super().__init__("Meteora")
        self.rpc_url = rpc_url
        self.cache = cache_manager or CacheManager()
        self.price_feed_url = price_feed_url
        self.pools_cache_ttl = timedelta(minutes=5)
        self._pool_configs: Dict[str, MeteoraPoolConfig] = {}
        
    async def initialize(self) -> None:
        """Initialize adapter state and connections"""
        try:
            # Load initial pool configurations
            # In production, this would come from a config service or database
            self._pool_configs = {
                "3W2HKgUa96Z69zzG3LK1g8KdcRAWzAttiLiHfYnKuPw5": MeteoraPoolConfig(
                    address="3W2HKgUa96Z69zzG3LK1g8KdcRAWzAttiLiHfYnKuPw5",
                    token_a_symbol="SOL",
                    token_b_symbol="USDC",
                    token_a_decimals=9,
                    token_b_decimals=6
                )
                # Add other pool configs here
            }
            
            # Verify RPC connection
            dlmm = self._create_dlmm_client(list(self._pool_configs.keys())[0])
            await self._verify_connection(dlmm)
            
            logger.info(f"Initialized Meteora adapter with {len(self._pool_configs)} pools")
            
        except Exception as e:
            logger.error(f"Failed to initialize Meteora adapter: {e}")
            raise

    async def get_pools(self) -> List[PoolMetrics]:
        """
        Retrieve metrics for all configured pools
        
        Returns:
            List of PoolMetrics for all active pools
        """
        try:
            # Check cache first
            cached = await self.cache.get("meteora:all_pools")
            if cached:
                return [PoolMetrics(**p) for p in cached]

            # Fetch all pools concurrently
            tasks = [
                self.get_pool(pool_id)
                for pool_id in self._pool_configs.keys()
            ]
            pools = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out errors and None results
            valid_pools = [
                p for p in pools
                if isinstance(p, PoolMetrics)
            ]
            
            # Cache results
            await self.cache.set(
                "meteora:all_pools",
                [p.to_dict() for p in valid_pools],
                ttl=self.pools_cache_ttl
            )
            
            return valid_pools
            
        except Exception as e:
            logger.error(f"Error fetching all pools: {e}")
            raise MetricsError(f"Failed to fetch pools: {e}")

    async def get_pool(self, pool_id: str) -> Optional[PoolMetrics]:
        """
        Retrieve metrics for a specific pool
        
        Args:
            pool_id: Pool address
            
        Returns:
            PoolMetrics if pool exists, None otherwise
        """
        try:
            # Validate pool exists in config
            if pool_id not in self._pool_configs:
                raise PoolNotFoundError(f"Pool {pool_id} not configured")
                
            # Create DLMM client
            dlmm = self._create_dlmm_client(pool_id)
            
            # Fetch pool data
            active_bin = dlmm.get_active_bin()
            fee_info = dlmm.get_fee_info()
            
            # Get token prices
            token_prices = await self._get_token_prices(
                dlmm.token_X.public_key,
                dlmm.token_Y.public_key
            )
            
            # Calculate metrics
            pool_data = {
                "pool_id": pool_id,
                "token_a": await self._create_token_info(
                    dlmm.token_X,
                    self._pool_configs[pool_id].token_a_symbol,
                    token_prices[0]
                ),
                "token_b": await self._create_token_info(
                    dlmm.token_Y,
                    self._pool_configs[pool_id].token_b_symbol,
                    token_prices[1]
                ),
                "tvl": await self._calculate_tvl(
                    dlmm,
                    active_bin,
                    token_prices
                ),
                "fee_rate": Decimal(str(fee_info.base_fee_rate_percentage)),
                "daily_volume": await self._get_daily_volume(pool_id),
                "active_bin": active_bin
            }
            
            return await self.calculate_metrics(pool_data)
            
        except PoolNotFoundError:
            logger.warning(f"Pool {pool_id} not found")
            return None
        except Exception as e:
            logger.error(f"Error fetching pool {pool_id}: {e}")
            raise MetricsError(f"Failed to fetch pool {pool_id}: {e}")

    async def validate_pool_health(
        self,
        pool_id: str
    ) -> Tuple[PoolStatus, List[str]]:
        """
        Check pool health status
        
        Args:
            pool_id: Pool address
            
        Returns:
            Tuple of PoolStatus and list of warning messages
        """
        try:
            warnings = []
            
            # Create DLMM client
            dlmm = self._create_dlmm_client(pool_id)
            
            # Check active bin
            active_bin = dlmm.get_active_bin()
            if float(active_bin.x_amount) == 0 or float(active_bin.y_amount) == 0:
                warnings.append("Pool has zero liquidity in active bin")
                
            # Check fee info
            fee_info = dlmm.get_fee_info()
            if fee_info.base_fee_rate_percentage == 0:
                warnings.append("Pool has zero fee rate")
                
            # Check dynamic fee
            dynamic_fee = dlmm.get_dynamic_fee()
            if dynamic_fee > fee_info.max_fee_rate_percentage:
                warnings.append("Dynamic fee exceeds maximum fee rate")
                
            # Determine status based on warnings
            if not warnings:
                return PoolStatus.HEALTHY, []
            elif len(warnings) > 2:
                return PoolStatus.UNHEALTHY, warnings
            else:
                return PoolStatus.WARNINGS, warnings
                
        except Exception as e:
            logger.error(f"Error validating pool health {pool_id}: {e}")
            return PoolStatus.UNKNOWN, [str(e)]

    def _create_dlmm_client(self, pool_id: str) -> DLMM:
        """Create DLMM client instance for pool"""
        return DLMM_CLIENT.create(
            Pubkey.from_string(pool_id),
            self.rpc_url
        )

    async def _verify_connection(self, dlmm: DLMM) -> None:
        """Verify RPC connection is working"""
        try:
            dlmm.get_active_bin()
        except Exception as e:
            raise ConnectionError(f"Failed to connect to RPC: {e}")

    async def _get_token_prices(
        self,
        token_x: Pubkey,
        token_y: Pubkey
    ) -> Tuple[Decimal, Decimal]:
        """
        Get current token prices
        
        Returns:
            Tuple of (token_x_price, token_y_price) in USD
        """
        # In production, integrate with price feed
        # This is a placeholder implementation
        return Decimal('20.50'), Decimal('1.00')  # SOL/USDC example

    async def _create_token_info(
        self,
        token: Any,
        symbol: str,
        price: Decimal
    ) -> TokenInfo:
        """Create TokenInfo instance from token data"""
        return TokenInfo(
            address=str(token.public_key),
            symbol=symbol,
            decimals=token.decimal,
            price_usd=price
        )

    async def _calculate_tvl(
        self,
        dlmm: DLMM,
        active_bin: Any,
        token_prices: Tuple[Decimal, Decimal]
    ) -> Decimal:
        """Calculate Total Value Locked in USD"""
        try:
            x_amount = Decimal(str(active_bin.x_amount))
            y_amount = Decimal(str(active_bin.y_amount))
            
            x_value = x_amount * token_prices[0]
            y_value = y_amount * token_prices[1]
            
            return (x_value + y_value).quantize(Decimal('0.01'), ROUND_DOWN)
            
        except Exception as e:
            logger.error(f"Error calculating TVL: {e}")
            return Decimal('0')

    async def _get_daily_volume(self, pool_id: str) -> Decimal:
        """Get 24h trading volume in USD"""
        # In production, implement volume tracking
        # This is a placeholder implementation
        return Decimal('50000.00')