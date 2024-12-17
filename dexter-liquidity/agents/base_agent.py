from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
import logging
import time
import numpy as np

from .types import (
    LiquidityPair,
    StrategyMetrics,
    RiskProfile,
    HealthStatus
)
from utils.error_handler import ErrorHandler, DexterError
from utils.memory_monitor import MemoryMonitor
from utils.parallel_processor import ParallelDataProcessor

logger = logging.getLogger(__name__)
error_handler = ErrorHandler()
memory_monitor = MemoryMonitor()

class DexterAgent(ABC):
    def __init__(
        self,
        risk_profile: RiskProfile,
        web3_provider: 'Web3',
        min_liquidity: float = 100000,
        max_slippage: float = 0.02,
        parallel_chunk_size: int = 1000
    ):
        self.risk_profile = risk_profile
        self.web3 = web3_provider
        self.min_liquidity = min_liquidity
        self.max_slippage = max_slippage
        self.active_pairs: Dict[str, LiquidityPair] = {}
        self.performance_history: List[Dict] = []
        self.parallel_processor = ParallelDataProcessor(chunk_size=parallel_chunk_size)
        self.error_count = 0
        self.last_execution_time = time.time()
        
        logger.info(f"base_agent.py: Initialized {risk_profile.value} agent")

    @error_handler.with_retries(retries=3)
    async def initialize(self):
        """Initialize agent with error handling"""
        try:
            if not self.web3.is_connected():
                raise DexterError("Failed to connect to Web3 provider")
                
            await self._init_strategy_params()
            logger.info(f"base_agent.py: Initialized {self.risk_profile.value} agent")
            
        except Exception as e:
            logger.error(f"base_agent.py: Initialization failed: {str(e)}")
            raise

    @memory_monitor.monitor
    async def evaluate_pairs(self, pairs: List[LiquidityPair]) -> List[Tuple[LiquidityPair, StrategyMetrics]]:
        """Evaluate pairs with memory monitoring"""
        logger.info(f"base_agent.py: Evaluating {len(pairs)} pairs")
        
        if len(pairs) > self.parallel_processor.chunk_size:
            return await self._evaluate_pairs_parallel(pairs)
        return await self._evaluate_pairs_sequential(pairs)

    async def _evaluate_pairs_parallel(self, pairs: List[LiquidityPair]) -> List[Tuple[LiquidityPair, StrategyMetrics]]:
        """Parallel pair evaluation"""
        chunks = [pairs[i:i + self.parallel_processor.chunk_size] 
                 for i in range(0, len(pairs), self.parallel_processor.chunk_size)]
        
        results = await self.parallel_processor.process_large_dataset(
            chunks,
            self._evaluate_pairs_sequential
        )
        
        return [item for sublist in results for item in sublist]

    @error_handler.with_retries(retries=2)
    async def _evaluate_pairs_sequential(self, pairs: List[LiquidityPair]) -> List[Tuple[LiquidityPair, StrategyMetrics]]:
        """Sequential pair evaluation with error handling"""
        results = []
        for pair in pairs:
            try:
                if await self._meets_basic_criteria(pair):
                    metrics = await self.analyze_pair(pair)
                    if await self._meets_risk_criteria(metrics):
                        results.append((pair, metrics))
            except Exception as e:
                logger.error(f"Error evaluating pair {pair.address}: {str(e)}")
                self.error_count += 1
                continue
        return results

    async def _calculate_tvl(self) -> float:
        """Calculate total value locked in agent's positions"""
        logger.debug("base_agent.py: Calculating TVL")
        try:
            tvl = 0.0
            for pair in self.active_pairs.values():
                price0 = await self._get_token_price(pair.token0)
                price1 = await self._get_token_price(pair.token1)
                tvl += (pair.reserve0 * price0) + (pair.reserve1 * price1)
            
            logger.info(f"base_agent.py: Current TVL: ${tvl:,.2f}")
            return tvl
            
        except Exception as e:
            logger.error(f"base_agent.py: Error calculating TVL: {str(e)}")
            self.error_count += 1
            return 0.0

    async def _calculate_fees_24h(self) -> float:
        """Calculate fees earned in the last 24 hours"""
        logger.debug("base_agent.py: Calculating 24h fees")
        try:
            total_fees = 0.0
            for pair in self.active_pairs.values():
                pair_fees = pair.volume_24h * pair.fee_rate
                liquidity_share = self._get_liquidity_share(pair)
                earned_fees = pair_fees * liquidity_share
                total_fees += earned_fees
            
            logger.info(f"base_agent.py: 24h fees earned: ${total_fees:,.2f}")
            return total_fees
            
        except Exception as e:
            logger.error(f"base_agent.py: Error calculating fees: {str(e)}")
            self.error_count += 1
            return 0.0

    async def _calculate_impermanent_loss(self) -> float:
        """Calculate current impermanent loss across all positions"""
        logger.debug("base_agent.py: Calculating impermanent loss")
        try:
            total_il = 0.0
            total_value = 0.0
            
            for pair in self.active_pairs.values():
                initial_price = await self._get_initial_price(pair)
                current_price = await self._get_current_price(pair.address)
                
                price_ratio = current_price / initial_price
                sqrt_ratio = np.sqrt(price_ratio)
                position_il = 2 * sqrt_ratio / (1 + price_ratio) - 1
                
                position_value = pair.reserve0 + pair.reserve1
                total_il += position_il * position_value
                total_value += position_value
            
            weighted_il = total_il / total_value if total_value > 0 else 0
            logger.info(f"base_agent.py: Current impermanent loss: {weighted_il:.2%}")
            return weighted_il
            
        except Exception as e:
            logger.error(f"base_agent.py: Error calculating impermanent loss: {str(e)}")
            self.error_count += 1
            return 0.0

    async def check_health(self) -> HealthStatus:
        """Check agent's health status"""
        logger.debug(f"base_agent.py: Checking health for {self.risk_profile.value} agent")
        try:
            web3_connected = self.web3.is_connected()
            if not web3_connected:
                logger.error(f"base_agent.py: Web3 connection lost for {self.risk_profile.value} agent")

            active_pairs_count = len(self.active_pairs)
            recent_performance = self.performance_history[-10:] if self.performance_history else []
            success_rate = 0.0
            if recent_performance:
                success_rate = sum(1 for p in recent_performance if p.get('success', False)) / len(recent_performance)

            tvl = await self._calculate_tvl()
            fees_24h = await self._calculate_fees_24h()
            il = await self._calculate_impermanent_loss()

            memory_usage = memory_monitor.get_current_memory_usage()

            specific_checks = await self._check_specific_health()

            is_healthy = (
                web3_connected 
                and success_rate >= 0.5 
                and self.error_count < 10
                and all(specific_checks.values())
                and tvl > 0
                and il < 0.1
            )

            status = HealthStatus(
                is_healthy=is_healthy,
                web3_connected=web3_connected,
                active_pairs_count=active_pairs_count,
                success_rate=success_rate,
                error_count=self.error_count,
                last_execution_time=self.last_execution_time,
                memory_usage=memory_usage,
                specific_checks=specific_checks,
                tvl=tvl,
                fees_24h=fees_24h,
                average_il=il
            )

            logger.info(f"base_agent.py: Health check completed for {self.risk_profile.value}")
            return status

        except Exception as e:
            logger.error(f"base_agent.py: Health check failed: {str(e)}")
            self.error_count += 1
            return HealthStatus(
                is_healthy=False,
                web3_connected=False,
                active_pairs_count=0,
                success_rate=0.0,
                error_count=self.error_count,
                last_execution_time=self.last_execution_time,
                memory_usage=0.0,
                specific_checks={},
                tvl=0.0,
                fees_24h=0.0,
                average_il=0.0
            )

    def _get_liquidity_share(self, pair: LiquidityPair) -> float:
        """Calculate our share of the pool's liquidity"""
        try:
            total_liquidity = pair.reserve0 + pair.reserve1
            our_liquidity = self.active_pairs.get(pair.address, 0)
            return our_liquidity / total_liquidity if total_liquidity > 0 else 0
        except Exception as e:
            logger.error(f"base_agent.py: Error calculating liquidity share: {str(e)}")
            self.error_count += 1
            return 0

    # Abstract methods that must be implemented by derived classes
    @abstractmethod
    async def _check_specific_health(self) -> Dict[str, bool]:
        """Agent-specific health checks"""
        pass

    @abstractmethod
    async def analyze_pair(self, pair: LiquidityPair) -> StrategyMetrics:
        """Analyze trading pair for strategy execution"""
        pass

    @abstractmethod
    async def _meets_basic_criteria(self, pair: LiquidityPair) -> bool:
        """Check if pair meets basic criteria"""
        pass

    @abstractmethod
    async def _meets_risk_criteria(self, metrics: StrategyMetrics) -> bool:
        """Check if metrics meet risk criteria"""
        pass

    @abstractmethod
    async def _get_token_price(self, token_address: str) -> float:
        """Get token price"""
        pass

    @abstractmethod
    async def _get_current_price(self, pair_address: str) -> float:
        """Get current price for pair"""
        pass

    @abstractmethod
    async def _get_initial_price(self, pair: LiquidityPair) -> float:
        """Get initial price for pair"""
        pass

    @abstractmethod
    async def _init_strategy_params(self):
        """Initialize strategy parameters"""
        pass