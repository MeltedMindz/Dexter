import pytest
import asyncio
from unittest.mock import Mock, patch
from web3 import Web3
from decimal import Decimal
from data.fetchers.base_fetcher import BaseFetcher, BasePoolData

@pytest.fixture
def mock_web3():
    return Mock(spec=Web3)

@pytest.fixture
def mock_contract():
    contract = Mock()
    contract.functions.slot0.return_value.call.return_value = [
        2**96,  # sqrt_price_x96
        0,      # tick
        0,      # observation_index
        0,      # observation_cardinality
        0,      # observation_cardinality_next
        0,      # fee_protocol
        True    # unlocked
    ]
    contract.functions.liquidity.return_value.call.return_value = 1000000
    contract.functions.token0.return_value.call.return_value = "0x123"
    contract.functions.token1.return_value.call.return_value = "0x456"
    contract.functions.fee.return_value.call.return_value = 3000
    return contract

@pytest.fixture
def fetcher(mock_web3):
    return BaseFetcher(
        alchemy_key="test_key",
        subgraph_url="http://test.url"
    )

@pytest.mark.asyncio
async def test_get_pool_data_cache(fetcher, mock_contract):
    """Test pool data caching"""
    pool_address = "0xtest"
    
    # Mock contract creation
    with patch.object(fetcher, '_get_pool_contract', return_value=mock_contract):
        # First call should fetch data
        data1 = await fetcher.get_pool_data(pool_address)
        assert data1 is not None
        assert data1.address == pool_address
        
        # Second call should use cache
        data2 = await fetcher.get_pool_data(pool_address)
        assert data2 is data1  # Should be same object from cache

@pytest.mark.asyncio
async def test_fee_tier_calculation(fetcher, mock_contract):
    """Test fee tier fetching and caching"""
    pool_address = "0xtest"
    
    with patch.object(fetcher, '_get_pool_contract', return_value=mock_contract):
        fee_tier = await fetcher._get_fee_tier(mock_contract)
        assert fee_tier == 3000  # 0.3%
        
        # Should use cached value
        fee_tier2 = await fetcher._get_fee_tier(mock_contract)
        assert fee_tier2 == fee_tier

@pytest.mark.asyncio
async def test_token_decimals(fetcher):
    """Test token decimals fetching"""
    token_address = "0xtest_token"
    
    # Mock token contract
    mock_token = Mock()
    mock_token.functions.decimals.return_value.call.return_value = 18
    
    with patch.object(fetcher.w3.eth, 'contract', return_value=mock_token):
        decimals = await fetcher._get_token_decimals(token_address)
        assert decimals == 18
        
        # Should use cached value
        decimals2 = await fetcher._get_token_decimals(token_address)
        assert decimals2 == decimals

@pytest.mark.asyncio
async def test_pool_metrics_fetching(fetcher):
    """Test subgraph metrics fetching"""
    pool_address = "0xtest"
    
    # Mock subgraph response
    mock_response = {
        "pool": {
            "totalValueLockedUSD": "1000000",
            "volumeUSD": "500000",
            "feesUSD": "1500"
        }
    }
    
    with patch('gql.Client.execute_async', return_value=mock_response):
        metrics = await fetcher._fetch_pool_metrics(pool_address)
        assert metrics['tvl'] == 1000000
        assert metrics['volume'] == 500000

@pytest.mark.asyncio
async def test_error_handling(fetcher):
    """Test error handling in data fetching"""
    pool_address = "0xtest"
    
    # Mock contract that raises an error
    mock_contract = Mock()
    mock_contract.functions.slot0.return_value.call.side_effect = Exception("Test error")
    
    with patch.object(fetcher, '_get_pool_contract', return_value=mock_contract):
        data = await fetcher.get_pool_data(pool_address)
        assert data is None  # Should return None on error
