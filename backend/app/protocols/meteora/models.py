"""
Meteora Protocol Models Module

This module defines the data structures and type definitions for the Meteora protocol integration.
It includes:
- Protocol-specific data models
- Type conversions
- Validation logic
- Serialization methods

The models are designed to bridge between the DLMM SDK's raw data structures and our
standardized protocol interface.
"""

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
import json

from solders.pubkey import Pubkey
from dlmm.types import (
    FeeInfo as DLMMFeeInfo,
    ActiveBin as DLMMActiveBin,
    GetPositionByUser,
    PositionData
)

class PoolType(Enum):
    """Types of supported Meteora pools"""
    STANDARD = "standard"
    STABLE = "stable"
    VOLATILE = "volatile"
    WEIGHTED = "weighted"

class PoolStatus(Enum):
    """Pool operational status"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    PAUSED = "paused"
    DEPRECATED = "deprecated"

@dataclass
class TokenDetails:
    """Detailed token information"""
    address: Pubkey
    symbol: str
    decimals: int
    price_usd: Decimal
    reserve_amount: Decimal
    weight: Optional[Decimal] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary representation"""
        return {
            "address": str(self.address),
            "symbol": self.symbol,
            "decimals": self.decimals,
            "price_usd": str(self.price_usd),
            "reserve_amount": str(self.reserve_amount),
            "weight": str(self.weight) if self.weight else None
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'TokenDetails':
        """Create instance from dictionary data"""
        return cls(
            address=Pubkey.from_string(data["address"]),
            symbol=data["symbol"],
            decimals=data["decimals"],
            price_usd=Decimal(data["price_usd"]),
            reserve_amount=Decimal(data["reserve_amount"]),
            weight=Decimal(data["weight"]) if data.get("weight") else None
        )

@dataclass
class PoolFees:
    """Pool fee configuration and statistics"""
    base_fee_rate: Decimal
    max_fee_rate: Decimal
    protocol_fee: Decimal
    dynamic_fee: Decimal
    total_fees_24h: Decimal
    
    @classmethod
    def from_dlmm_fee_info(cls, fee_info: DLMMFeeInfo, dynamic_fee: float) -> 'PoolFees':
        """Create instance from DLMM FeeInfo"""
        return cls(
            base_fee_rate=Decimal(str(fee_info.base_fee_rate_percentage)),
            max_fee_rate=Decimal(str(fee_info.max_fee_rate_percentage)),
            protocol_fee=Decimal(str(fee_info.protocol_fee_percentage)),
            dynamic_fee=Decimal(str(dynamic_fee)),
            total_fees_24h=Decimal('0')  # To be updated with actual data
        )

    def to_dict(self) -> Dict:
        """Convert to dictionary representation"""
        return {
            "base_fee_rate": str(self.base_fee_rate),
            "max_fee_rate": str(self.max_fee_rate),
            "protocol_fee": str(self.protocol_fee),
            "dynamic_fee": str(self.dynamic_fee),
            "total_fees_24h": str(self.total_fees_24h)
        }

@dataclass
class PoolMetrics:
    """Current pool performance metrics"""
    tvl_usd: Decimal
    volume_24h: Decimal
    fees_24h: Decimal
    apy: Decimal
    price_impact: Decimal
    liquidity_depth: Dict[str, Decimal]
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict:
        """Convert to dictionary representation"""
        return {
            "tvl_usd": str(self.tvl_usd),
            "volume_24h": str(self.volume_24h),
            "fees_24h": str(self.fees_24h),
            "apy": str(self.apy),
            "price_impact": str(self.price_impact),
            "liquidity_depth": {
                k: str(v) for k, v in self.liquidity_depth.items()
            },
            "last_updated": self.last_updated.isoformat()
        }

@dataclass
class BinData:
    """Data for a specific liquidity bin"""
    bin_id: int
    price: Decimal
    x_amount: Decimal
    y_amount: Decimal
    liquidity: Decimal
    is_active: bool

    @classmethod
    def from_dlmm_bin(cls, bin_data: DLMMActiveBin, is_active: bool) -> 'BinData':
        """Create instance from DLMM bin data"""
        return cls(
            bin_id=bin_data.bin_id,
            price=Decimal(str(bin_data.price)),
            x_amount=Decimal(str(bin_data.x_amount)),
            y_amount=Decimal(str(bin_data.y_amount)),
            liquidity=Decimal(str(bin_data.supply)),
            is_active=is_active
        )

    def to_dict(self) -> Dict:
        """Convert to dictionary representation"""
        return {
            "bin_id": self.bin_id,
            "price": str(self.price),
            "x_amount": str(self.x_amount),
            "y_amount": str(self.y_amount),
            "liquidity": str(self.liquidity),
            "is_active": self.is_active
        }

@dataclass
class MeteoraPool:
    """Complete Meteora pool information"""
    address: Pubkey
    pool_type: PoolType
    status: PoolStatus
    token_a: TokenDetails
    token_b: TokenDetails
    fees: PoolFees
    metrics: PoolMetrics
    active_bin: BinData
    bins: List[BinData]
    creation_time: datetime
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict:
        """Convert to dictionary representation"""
        return {
            "address": str(self.address),
            "pool_type": self.pool_type.value,
            "status": self.status.value,
            "token_a": self.token_a.to_dict(),
            "token_b": self.token_b.to_dict(),
            "fees": self.fees.to_dict(),
            "metrics": self.metrics.to_dict(),
            "active_bin": self.active_bin.to_dict(),
            "bins": [bin.to_dict() for bin in self.bins],
            "creation_time": self.creation_time.isoformat(),
            "last_updated": self.last_updated.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'MeteoraPool':
        """Create instance from dictionary data"""
        return cls(
            address=Pubkey.from_string(data["address"]),
            pool_type=PoolType(data["pool_type"]),
            status=PoolStatus(data["status"]),
            token_a=TokenDetails.from_dict(data["token_a"]),
            token_b=TokenDetails.from_dict(data["token_b"]),
            fees=PoolFees(**{
                k: Decimal(v) for k, v in data["fees"].items()
            }),
            metrics=PoolMetrics(
                tvl_usd=Decimal(data["metrics"]["tvl_usd"]),
                volume_24h=Decimal(data["metrics"]["volume_24h"]),
                fees_24h=Decimal(data["metrics"]["fees_24h"]),
                apy=Decimal(data["metrics"]["apy"]),
                price_impact=Decimal(data["metrics"]["price_impact"]),
                liquidity_depth={
                    k: Decimal(v) for k, v in data["metrics"]["liquidity_depth"].items()
                },
                last_updated=datetime.fromisoformat(data["metrics"]["last_updated"])
            ),
            active_bin=BinData(
                bin_id=data["active_bin"]["bin_id"],
                price=Decimal(data["active_bin"]["price"]),
                x_amount=Decimal(data["active_bin"]["x_amount"]),
                y_amount=Decimal(data["active_bin"]["y_amount"]),
                liquidity=Decimal(data["active_bin"]["liquidity"]),
                is_active=data["active_bin"]["is_active"]
            ),
            bins=[
                BinData(
                    bin_id=bin["bin_id"],
                    price=Decimal(bin["price"]),
                    x_amount=Decimal(bin["x_amount"]),
                    y_amount=Decimal(bin["y_amount"]),
                    liquidity=Decimal(bin["liquidity"]),
                    is_active=bin["is_active"]
                )
                for bin in data["bins"]
            ],
            creation_time=datetime.fromisoformat(data["creation_time"]),
            last_updated=datetime.fromisoformat(data["last_updated"])
        )

class ValidationError(Exception):
    """Raised when model validation fails"""
    pass

def validate_pool_data(pool: MeteoraPool) -> List[str]:
    """
    Validate pool data and return list of warnings
    
    Returns:
        List of warning messages, empty if no warnings
    """
    warnings = []
    
    # Token validation
    if pool.token_a.reserve_amount <= 0:
        warnings.append(f"Zero or negative reserve for token {pool.token_a.symbol}")
    if pool.token_b.reserve_amount <= 0:
        warnings.append(f"Zero or negative reserve for token {pool.token_b.symbol}")
        
    # Price validation
    if pool.token_a.price_usd <= 0:
        warnings.append(f"Invalid price for token {pool.token_a.symbol}")
    if pool.token_b.price_usd <= 0:
        warnings.append(f"Invalid price for token {pool.token_b.symbol}")
        
    # Fee validation
    if pool.fees.base_fee_rate < 0 or pool.fees.base_fee_rate > 100:
        warnings.append("Invalid base fee rate")
    if pool.fees.dynamic_fee > pool.fees.max_fee_rate:
        warnings.append("Dynamic fee exceeds maximum fee rate")
        
    # Metrics validation
    if pool.metrics.tvl_usd <= 0:
        warnings.append("Zero or negative TVL")
    if pool.metrics.apy < 0:
        warnings.append("Negative APY")
        
    # Bin validation
    active_bin_count = sum(1 for bin in pool.bins if bin.is_active)
    if active_bin_count != 1:
        warnings.append(f"Expected 1 active bin, found {active_bin_count}")
        
    return warnings