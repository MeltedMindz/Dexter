import pytest
from web3 import Web3
from unittest.mock import Mock, patch
import asyncio

from agents import (
    DexterAgent,
    ConservativeAgent,
    AggressiveAgent,
    HyperAggressiveAgent,
    RiskProfile,
    LiquidityPair,
    StrategyMetrics
)

@pytest.fixture
def web3_mock():
    web3 = Mock()
    web3.is_connected.return_value = True
    eth_mock = Mock()
    eth_mock.get_block.return_value = Mock(timestamp=1234567890)
    web3.eth = eth_mock
    return web3

@pytest.fixture
def sample_pair():
    return LiquidityPair(
        token0="0x...",
        token1="0x...",
        address="0x...",
        reserve0=1000000,
        reserve1=1000000,
        volume_24h=100000,
        fee_rate=0.003,
        volatility_24h=0.1
    )

@pytest.mark.asyncio
async def test_conservative_agent_initialization(web3_mock):
    agent = ConservativeAgent(web3_mock)
    await agent.initialize()
    assert agent.risk_profile == RiskProfile.CONSERVATIVE
    assert agent.volatility_threshold == 0.15
    assert agent.min_liquidity == 500000

@pytest.mark.asyncio
async def test_basic_criteria_check(web3_mock, sample_pair):
    agent = ConservativeAgent(web3_mock)
    await agent.initialize()
    meets_criteria = await agent._meets_basic_criteria(sample_pair)
    assert meets_criteria == True

@pytest.mark.asyncio
async def test_risk_profiles_different_thresholds(web3_mock):
    conservative = ConservativeAgent(web3_mock)
    aggressive = AggressiveAgent(web3_mock)
    hyper = HyperAggressiveAgent(web3_mock)
    
    await asyncio.gather(
        conservative.initialize(),
        aggressive.initialize(),
        hyper.initialize()
    )
    
    assert conservative.volatility_threshold < aggressive.volatility_threshold
    assert aggressive.volatility_threshold < hyper.volatility_threshold
