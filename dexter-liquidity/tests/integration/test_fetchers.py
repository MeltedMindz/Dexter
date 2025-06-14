import pytest
import asyncio
from decimal import Decimal
from datetime import datetime, timedelta
from data.fetchers.base_interface import (
    LiquidityPoolFetcher,
    PoolStats,
    PositionInfo,
    NetworkError,
    DataValidationError
)
from data.fetchers.uniswap_v4_fetcher import UniswapV4Fetcher
from data.fetchers.meteora_fetcher import MeteoraFetcher

@pytest.fixture
def mock_pool_address():
    return "0x1234567890123456789012345678901234567890"

@pytest.fixture
async def uniswap_fetcher():
    # Initialize with test configuration
    fetcher = UniswapV4Fetcher(
        web3_provider="TEST_PROVIDER",
        position_manager_address="TEST_ADDRESS",
        retry_attempts=2,
        timeout=5
    )
    yield fetcher

@pytest.fixture
async def meteora_fetcher():
    # Initialize with test configuration
    fetcher = MeteoraFetcher(
        api_url="TEST_URL",
        retry_attempts=2,
        timeout=5
    )
    yield fetcher

@pytest.mark.asyncio
async def test_get_pool_data(uniswap_fetcher, mock_pool_address):
    """Test pool data fetching"""
    pool_data = await uniswap_fetcher.get_pool_data(mock_pool_address)
    
    assert pool_data is not None
    assert isinstance(pool_data, PoolStats)
    assert pool_data.address == mock_pool_address
    assert pool_data.tvl > 0
    assert pool_data.volume_24h >= 0

@pytest.mark.asyncio
async def test_calculate_optimal_position(uniswap_fetcher, mock_pool_address):
    """Test position calculation"""
    position = await uniswap_fetcher.calculate_optimal_position(
        pool_address=mock_pool_address,
        amount=Decimal("1000"),
        max_slippage=Decimal("0.01")
    )
    
    assert position is not None
    assert isinstance(position, PositionInfo)
    assert position.lower_price < position.upper_price
    assert position.optimal_liquidity > 0

@pytest.mark.asyncio
async def test_rate_limiting(uniswap_fetcher, mock_pool_address):
    """Test rate limiting functionality"""
    start_time = asyncio.get_event_loop().time()
    
    # Make multiple requests
    requests = [
        uniswap_fetcher.get_pool_data(mock_pool_address)
        for _ in range(3)
    ]
    
    await asyncio.gather(*requests)
    elapsed = asyncio.get_event_loop().time() - start_time
    
    # Should take at least 0.4 seconds (2 intervals of 0.2s)
    assert elapsed >= 0.4

@pytest.mark.asyncio
async def test_error_handling(uniswap_fetcher):
    """Test error handling with invalid data"""
    with pytest.raises(DataValidationError):
        await uniswap_fetcher.get_pool_data("invalid_address")

@pytest.mark.asyncio
async def test_historical_data(uniswap_fetcher, mock_pool_address):
    """Test historical data fetching"""
    end_time = datetime.now()
    start_time = end_time - timedelta(days=7)
    
    data = await uniswap_fetcher.get_historical_data(
        pool_address=mock_pool_address,
        start_time=start_time,
        end_time=end_time
    )
    
    assert isinstance(data, list)
    assert len(data) > 0
    assert all('timestamp' in entry for entry in data)

@pytest.mark.asyncio
async def test_fetcher_comparison(uniswap_fetcher, meteora_fetcher, mock_pool_address):
    """Compare data between different fetchers"""
    uni_data = await uniswap_fetcher.get_pool_data(mock_pool_address)
    meteora_data = await meteora_fetcher.get_pool_data(mock_pool_address)
    
    # Basic consistency checks
    assert abs(uni_data.tvl - meteora_data.tvl) / uni_data.tvl < 0.1  # Within 10%
    assert uni_data.token0_address == meteora_data.token0_address

# Add more specific tests for each fetcher implementation...
