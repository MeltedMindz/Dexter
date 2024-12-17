import pytest
import asyncio
from typing import List
from decimal import Decimal
from datetime import datetime, timedelta
import numpy as np

from utils.error_handler import ErrorHandler
from data.parallel_regime_detector import ParallelRegimeDetector
from agents import ConservativeAgent, AggressiveAgent, HyperAggressiveAgent

@pytest.fixture
def error_handler():
    return ErrorHandler()

@pytest.fixture
def regime_detector():
    return ParallelRegimeDetector(chunk_size=100)

@pytest.mark.asyncio
class TestErrorHandling:
    async def test_retry_mechanism(self, error_handler):
        @error_handler.with_retries(retries=3, delay=0.1)
        async def failing_function():
            raise NetworkError("Test error")
            
        with pytest.raises(NetworkError):
            await failing_function()
            
        stats = error_handler.get_error_stats()
        assert stats['error_counts']['failing_function'] == 4  # Initial + 3 retries
        
    async def test_fallback_behavior(self, error_handler):
        async def fallback():
            return "fallback"
            
        @error_handler.with_fallback(fallback)
        async def main_function():
            raise Exception("Main failed")
            
        result = await main_function()
        assert result == "fallback"

@pytest.mark.asyncio
class TestParallelProcessing:
    async def test_regime_detection_parallel(self, regime_detector):
        # Generate test data
        prices = np.random.lognormal(0, 0.1, 1000)
        volumes = np.random.lognormal(0, 0.5, 1000)
        timestamps = list(range(1000))
        
        results = await regime_detector.analyze_regimes(
            prices.tolist(),
            volumes.tolist(),
            timestamps
        )
        
        assert len(results) == 951  # 1000 - window_size + 1
        assert all(isinstance(r.confidence, float) for r in results)

@pytest.mark.asyncio
class TestAgentEdgeCases:
    async def test_extreme_volatility(self, web3_mock):
        agent = HyperAggressiveAgent(web3_mock)
        
        # Test with extreme volatility
        pair = LiquidityPair(
            token0="0x...",
            token1="0x...",
            address="0x...",
            reserve0=1000000,
            reserve1=1000000,
            volume_24h=1000000,
            fee_rate=0.003,
            volatility_24h=1.0  # 100% volatility
        )
        
        metrics = await agent.analyze_pair(pair)
        assert metrics.risk_score > 0.8  # High risk score
        
    async def test_zero_liquidity(self, web3_mock):
        agent = ConservativeAgent(web3_mock)
        
        # Test with zero liquidity
        pair = LiquidityPair(
            token0="0x...",
            token1="0x...",
            address="0x...",
            reserve0=0,
            reserve1=0,
            volume_24h=1000,
            fee_rate=0.003,
            volatility_24h=0.1
        )
        
        should_execute = await agent._meets_basic_criteria(pair)
        assert not should_execute  # Should reject zero liquidity

# Add more test cases...
