from web3 import Web3
import numpy as np
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass
import logging
import math
from enum import Enum
from decimal import Decimal

logger = logging.getLogger(__name__)

class VolatilityTimeframe(Enum):
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    HOUR_24 = "24h"

@dataclass
class VolatilityWeight:
    timeframe: VolatilityTimeframe
    weight: float
    
    @staticmethod
    def get_default_weights() -> List['VolatilityWeight']:
        return [
            VolatilityWeight(VolatilityTimeframe.HOUR_1, 0.5),
            VolatilityWeight(VolatilityTimeframe.HOUR_4, 0.3),
            VolatilityWeight(VolatilityTimeframe.HOUR_24, 0.2)
        ]

class LiquidityRange(NamedTuple):
    lower_tick: int
    upper_tick: int
    optimal_liquidity: Decimal
    fee_estimate: Decimal

class UniswapV4Fetcher:
    """Enhanced Uniswap V4 data fetcher with optimal positioning"""
    
    def __init__(
        self, 
        web3: Web3,
        position_manager_address: str,
        volatility_weights: Optional[List[VolatilityWeight]] = None,
        k_factor: float = 2.5
    ):
        self.web3 = web3
        self.position_manager = self._init_position_manager(position_manager_address)
        self.volatility_weights = volatility_weights or VolatilityWeight.get_default_weights()
        self.k_factor = k_factor
        logger.info(f"uniswap_v4_fetcher.py: Initialized UniswapV4Fetcher with k_factor={k_factor}")
        
    async def calculate_optimal_position(
        self,
        pool_address: str,
        current_price: Decimal,
        volatilities: Dict[VolatilityTimeframe, float],
        liquidity_depth: Decimal
    ) -> LiquidityRange:
        """Calculate optimal liquidity position range"""
        logger.info(f"uniswap_v4_fetcher.py: Calculating optimal position for pool {pool_address}")
        
        try:
            # Calculate weighted volatility
            weighted_variance = sum(
                weight.weight * (volatilities[weight.timeframe] ** 2)
                for weight in self.volatility_weights
            )
            weighted_volatility = math.sqrt(weighted_variance)
            logger.debug(f"uniswap_v4_fetcher.py: Calculated weighted volatility: {weighted_volatility}")
            
            # Calculate range
            range_size = weighted_volatility * self.k_factor
            tick_spacing = 60
            price_lower = current_price * Decimal(1 - range_size)
            price_upper = current_price * Decimal(1 + range_size)
            
            lower_tick = self._price_to_tick(price_lower, tick_spacing)
            upper_tick = self._price_to_tick(price_upper, tick_spacing)
            
            optimal_liquidity = self._calculate_optimal_liquidity(
                current_price,
                price_lower,
                price_upper,
                liquidity_depth
            )
            
            fee_estimate = self._estimate_fee_returns(
                weighted_volatility,
                optimal_liquidity,
                current_price
            )
            
            range_data = LiquidityRange(lower_tick, upper_tick, optimal_liquidity, fee_estimate)
            logger.info(f"uniswap_v4_fetcher.py: Calculated position range: ticks={lower_tick}-{upper_tick}")
            return range_data
            
        except Exception as e:
            logger.error(f"uniswap_v4_fetcher.py: Error calculating optimal position: {str(e)}")
            raise

    # ... [Rest of the methods with similar logging enhancements]
