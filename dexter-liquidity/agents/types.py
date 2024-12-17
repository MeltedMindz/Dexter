from dataclasses import dataclass
from enum import Enum
from typing import Dict

@dataclass
class LiquidityPair:
    token0: str
    token1: str
    address: str
    reserve0: float
    reserve1: float
    volume_24h: float
    fee_rate: float
    volatility_24h: float

@dataclass
class StrategyMetrics:
    impermanent_loss: float
    estimated_apr: float
    risk_score: float
    confidence: float

@dataclass
class HealthStatus:
    is_healthy: bool
    web3_connected: bool
    active_pairs_count: int
    success_rate: float
    error_count: int
    last_execution_time: float
    memory_usage: float
    specific_checks: Dict[str, bool]
    tvl: float
    fees_24h: float
    average_il: float

class RiskProfile(Enum):
    CONSERVATIVE = "conservative"
    AGGRESSIVE = "aggressive"
    HYPER_AGGRESSIVE = "hyper_aggressive"