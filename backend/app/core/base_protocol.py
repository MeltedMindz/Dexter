"""
Base Protocol Module - Core interfaces and data structures for protocol integrations.

This module defines the foundational abstractions that all protocol implementations must follow,
ensuring consistent behavior and data structures across different liquidity protocols.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import logging
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ProtocolError(Exception):
    """Base exception class for protocol-related errors"""
    pass

class PoolNotFoundError(ProtocolError):
    """Raised when a requested pool cannot be found"""
    pass

class PoolHealthError(ProtocolError):
    """Raised when a pool's health check fails"""
    pass

class MetricsError(ProtocolError):
    """Raised when there's an error calculating metrics"""
    pass

class PoolStatus(Enum):
    """Enumeration of possible pool health statuses"""
    HEALTHY = "healthy"
    WARNINGS = "warnings"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

@dataclass
class TokenInfo:
    """Information about a token in a pool"""
    address: str
    symbol: str
    decimals: int
    price_usd: Decimal
    
    def to_dict(self) -> Dict:
        """Convert token info to dictionary format"""
        return {
            "address": self.address,
            "symbol": self.symbol,
            "decimals": self.decimals,
            "price_usd": str(self.price_usd)
        }

@dataclass
class PoolMetrics:
    """Standardized metrics for a liquidity pool"""
    pool_id: str
    protocol_name: str
    token_a: TokenInfo
    token_b: TokenInfo
    tvl_usd: Decimal
    fee_rate: Decimal
    daily_volume_usd: Decimal
    daily_yield: Decimal
    apy: Decimal
    autocompound_apy: Decimal
    last_updated: datetime
    status: PoolStatus
    warnings: List[str]
    
    def to_dict(self) -> Dict:
        """Convert pool metrics to dictionary format"""
        return {
            "pool_id": self.pool_id,
            "protocol": self.protocol_name,
            "token_a": self.token_a.to_dict(),
            "token_b": self.token_b.to_dict(),
            "tvl_usd": str(self.tvl_usd),
            "fee_rate": str(self.fee_rate),
            "daily_volume_usd": str(self.daily_volume_usd),
            "daily_yield": str(self.daily_yield),
            "apy": str(self.apy),
            "autocompound_apy": str(self.autocompound_apy),
            "last_updated": self.last_updated.isoformat(),
            "status": self.status.value,
            "warnings": self.warnings
        }

class BaseProtocol(ABC):
    """Abstract base class that all protocol implementations must extend"""
    
    def __init__(self, protocol_name: str):
        """Initialize protocol instance"""
        self.protocol_name = protocol_name
        self.logger = logging.getLogger(f"protocol.{protocol_name}")

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize protocol connections and state"""
        raise NotImplementedError
    
    @abstractmethod
    async def get_pools(self) -> List[PoolMetrics]:
        """Retrieve all available pools with their current metrics"""
        raise NotImplementedError
    
    @abstractmethod
    async def get_pool(self, pool_id: str) -> Optional[PoolMetrics]:
        """Retrieve metrics for a specific pool"""
        raise NotImplementedError
    
    @abstractmethod
    async def validate_pool_health(self, pool_id: str) -> Tuple[PoolStatus, List[str]]:
        """
        Validate pool health and return status with any warnings
        
        Returns:
            Tuple containing PoolStatus and list of warning messages
        """
        raise NotImplementedError

    async def calculate_metrics(self, pool_data: Dict[str, Any]) -> PoolMetrics:
        """
        Calculate standardized metrics from protocol-specific pool data
        
        This method provides a default implementation of common metric calculations
        that can be overridden by specific protocols if needed.
        """
        try:
            # Extract required data with validation
            tvl = Decimal(str(pool_data.get("tvl", 0)))
            daily_volume = Decimal(str(pool_data.get("daily_volume", 0)))
            fee_rate = Decimal(str(pool_data.get("fee_rate", 0)))
            
            # Calculate yields and APY
            daily_yield = self._calculate_daily_yield(daily_volume, tvl, fee_rate)
            base_apy = daily_yield * Decimal('365')
            autocompound_apy = self._calculate_autocompound_apy(base_apy)
            
            # Get pool health status
            status, warnings = await self.validate_pool_health(pool_data["pool_id"])
            
            return PoolMetrics(
                pool_id=pool_data["pool_id"],
                protocol_name=self.protocol_name,
                token_a=pool_data["token_a"],
                token_b=pool_data["token_b"],
                tvl_usd=tvl,
                fee_rate=fee_rate,
                daily_volume_usd=daily_volume,
                daily_yield=daily_yield,
                apy=base_apy,
                autocompound_apy=autocompound_apy,
                last_updated=datetime.now(timezone.utc),
                status=status,
                warnings=warnings
            )
            
        except KeyError as e:
            raise MetricsError(f"Missing required pool data field: {e}")
        except (ValueError, TypeError) as e:
            raise MetricsError(f"Error calculating metrics: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error calculating metrics: {e}")
            raise MetricsError(f"Unexpected error calculating metrics: {e}")

    def _calculate_daily_yield(
        self,
        daily_volume: Decimal,
        tvl: Decimal,
        fee_rate: Decimal
    ) -> Decimal:
        """Calculate daily yield percentage"""
        if tvl == 0:
            return Decimal('0')
        
        daily_fees = daily_volume * (fee_rate / Decimal('100'))
        return (daily_fees / tvl) * Decimal('100')

    def _calculate_autocompound_apy(self, base_apy: Decimal) -> Decimal:
        """Calculate APY with daily autocompounding"""
        try:
            daily_rate = base_apy / Decimal('365')
            return (
                (Decimal('1') + daily_rate / Decimal('100')) ** Decimal('365') 
                - Decimal('1')
            ) * Decimal('100')
        except Exception as e:
            self.logger.error(f"Error calculating autocompound APY: {e}")
            return base_apy

    async def format_for_api(self, metrics: PoolMetrics) -> Dict:
        """Format metrics for API response"""
        try:
            return metrics.to_dict()
        except Exception as e:
            self.logger.error(f"Error formatting metrics for API: {e}")
            raise MetricsError(f"Error formatting metrics for API: {e}")

    def _validate_token_info(self, token: TokenInfo) -> List[str]:
        """Validate token information and return any warnings"""
        warnings = []
        
        if token.price_usd <= 0:
            warnings.append(f"Invalid price for token {token.symbol}")
        if token.decimals < 0 or token.decimals > 18:
            warnings.append(f"Suspicious decimals for token {token.symbol}")
            
        return warnings

    async def _validate_pool_metrics(self, metrics: PoolMetrics) -> List[str]:
        """Validate pool metrics and return any warnings"""
        warnings = []
        
        # Token validations
        warnings.extend(self._validate_token_info(metrics.token_a))
        warnings.extend(self._validate_token_info(metrics.token_b))
        
        # TVL validations
        if metrics.tvl_usd <= 0:
            warnings.append("Pool TVL is zero or negative")
        
        # Fee validations
        if metrics.fee_rate <= 0 or metrics.fee_rate > 100:
            warnings.append("Suspicious fee rate percentage")
            
        # Yield validations
        if metrics.daily_yield < 0:
            warnings.append("Negative daily yield detected")
        if metrics.apy < 0:
            warnings.append("Negative APY detected")
            
        return warnings