"""
Tests for Uniswap V4 fetcher integration
"""
import pytest
from unittest.mock import Mock, AsyncMock, patch
from decimal import Decimal
from web3 import Web3

from data.fetchers.uniswap_v4_fetcher import (
    UniswapV4Fetcher,
    VolatilityTimeframe,
    VolatilityWeight,
    LiquidityRange,
    CONTRACT_ADDRESSES
)


@pytest.fixture
def mock_web3():
    """Create mock Web3 instance"""
    web3 = Mock(spec=Web3)
    web3.eth = Mock()
    web3.eth.contract = Mock()
    web3.keccak = Mock(return_value=b"test_pool_id")
    return web3


@pytest.fixture
def mock_contracts(mock_web3):
    """Create mock contracts"""
    pool_manager = Mock()
    pool_manager.functions = Mock()
    pool_manager.functions.getSlot0 = Mock()
    pool_manager.functions.updateDynamicLPFee = Mock()
    
    position_manager = Mock()
    position_manager.functions = Mock()
    position_manager.functions.modifyLiquidities = Mock()
    
    universal_router = Mock()
    universal_router.functions = Mock()
    universal_router.functions.execute = Mock()
    
    quoter = Mock()
    quoter.functions = Mock()
    quoter.functions.quoteExactInputSingle = Mock()
    
    mock_web3.eth.contract.side_effect = [
        pool_manager,
        position_manager,
        universal_router,
        quoter
    ]
    
    return {
        "pool_manager": pool_manager,
        "position_manager": position_manager,
        "universal_router": universal_router,
        "quoter": quoter
    }


@pytest.fixture
def uniswap_fetcher(mock_web3, mock_contracts):
    """Create UniswapV4Fetcher instance with mocked dependencies"""
    with patch.object(UniswapV4Fetcher, '_get_pool_manager_abi'), \
         patch.object(UniswapV4Fetcher, '_get_position_manager_abi'), \
         patch.object(UniswapV4Fetcher, '_get_universal_router_abi'), \
         patch.object(UniswapV4Fetcher, '_get_quoter_abi'):
        
        fetcher = UniswapV4Fetcher(mock_web3)
        return fetcher


class TestUniswapV4Fetcher:
    """Test cases for UniswapV4Fetcher"""
    
    def test_initialization(self, uniswap_fetcher):
        """Test fetcher initialization"""
        assert uniswap_fetcher.k_factor == 2.5
        assert len(uniswap_fetcher.volatility_weights) == 3
        assert uniswap_fetcher.pool_manager is not None
        assert uniswap_fetcher.position_manager is not None
    
    def test_contract_addresses(self):
        """Test contract addresses are correct for Base network"""
        assert CONTRACT_ADDRESSES["POOL_MANAGER"] == "0x498581ff718922c3f8e6a244956af099b2652b2b"
        assert CONTRACT_ADDRESSES["POSITION_MANAGER"] == "0x7c5f5a4bbd8fd63184577525326123b519429bdc"
        assert CONTRACT_ADDRESSES["UNIVERSAL_ROUTER"] == "0x6ff5693b99212da76ad316178a184ab56d299b43"
    
    @pytest.mark.asyncio
    async def test_get_pool_state(self, uniswap_fetcher, mock_contracts):
        """Test getting pool state from PoolManager"""
        # Mock pool state response
        mock_slot0 = [1000000000000000000, 100, 500, 3000]  # sqrtPrice, tick, protocolFee, lpFee
        mock_contracts["pool_manager"].functions.getSlot0.return_value.call = AsyncMock(
            return_value=mock_slot0
        )
        
        pool_key = {
            "currency0": "0xToken0",
            "currency1": "0xToken1", 
            "fee": 3000
        }
        
        result = await uniswap_fetcher.get_pool_state(pool_key)
        
        assert result["sqrt_price_x96"] == mock_slot0[0]
        assert result["tick"] == mock_slot0[1]
        assert result["protocol_fee"] == mock_slot0[2]
        assert result["lp_fee"] == mock_slot0[3]
    
    @pytest.mark.asyncio
    async def test_mint_position(self, uniswap_fetcher, mock_contracts):
        """Test minting new liquidity position"""
        mock_tx_hash = "0x123...abc"
        mock_contracts["position_manager"].functions.modifyLiquidities.return_value.transact = AsyncMock(
            return_value=Mock(hex=Mock(return_value=mock_tx_hash))
        )
        
        pool_key = {"currency0": "0xToken0", "currency1": "0xToken1", "fee": 3000}
        
        result = await uniswap_fetcher.mint_position(
            pool_key=pool_key,
            tick_lower=-1000,
            tick_upper=1000,
            liquidity=1000000,
            amount0_max=100000,
            amount1_max=100000,
            recipient="0xRecipient",
            deadline=1234567890
        )
        
        assert result == mock_tx_hash
        mock_contracts["position_manager"].functions.modifyLiquidities.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_increase_liquidity(self, uniswap_fetcher, mock_contracts):
        """Test increasing liquidity in existing position"""
        mock_tx_hash = "0x456...def"
        mock_contracts["position_manager"].functions.modifyLiquidities.return_value.transact = AsyncMock(
            return_value=Mock(hex=Mock(return_value=mock_tx_hash))
        )
        
        result = await uniswap_fetcher.increase_liquidity(
            token_id=1,
            liquidity=500000,
            amount0_max=50000,
            amount1_max=50000,
            deadline=1234567890
        )
        
        assert result == mock_tx_hash
    
    @pytest.mark.asyncio
    async def test_decrease_liquidity(self, uniswap_fetcher, mock_contracts):
        """Test decreasing liquidity from position"""
        mock_tx_hash = "0x789...ghi"
        mock_contracts["position_manager"].functions.modifyLiquidities.return_value.transact = AsyncMock(
            return_value=Mock(hex=Mock(return_value=mock_tx_hash))
        )
        
        result = await uniswap_fetcher.decrease_liquidity(
            token_id=1,
            liquidity=250000,
            amount0_min=10000,
            amount1_min=10000,
            recipient="0xRecipient",
            deadline=1234567890
        )
        
        assert result == mock_tx_hash
    
    @pytest.mark.asyncio
    async def test_collect_fees(self, uniswap_fetcher, mock_contracts):
        """Test collecting fees from position"""
        mock_tx_hash = "0xabc...123"
        mock_contracts["position_manager"].functions.modifyLiquidities.return_value.transact = AsyncMock(
            return_value=Mock(hex=Mock(return_value=mock_tx_hash))
        )
        
        result = await uniswap_fetcher.collect_fees(
            token_id=1,
            recipient="0xRecipient",
            deadline=1234567890
        )
        
        assert result == mock_tx_hash
    
    @pytest.mark.asyncio
    async def test_burn_position(self, uniswap_fetcher, mock_contracts):
        """Test burning entire position"""
        mock_tx_hash = "0xdef...456"
        mock_contracts["position_manager"].functions.modifyLiquidities.return_value.transact = AsyncMock(
            return_value=Mock(hex=Mock(return_value=mock_tx_hash))
        )
        
        result = await uniswap_fetcher.burn_position(
            token_id=1,
            amount0_min=5000,
            amount1_min=5000,
            recipient="0xRecipient",
            deadline=1234567890
        )
        
        assert result == mock_tx_hash
    
    @pytest.mark.asyncio
    async def test_swap_exact_input(self, uniswap_fetcher, mock_contracts):
        """Test executing swap via Universal Router"""
        mock_tx_hash = "0x111...222"
        mock_contracts["universal_router"].functions.execute.return_value.transact = AsyncMock(
            return_value=Mock(hex=Mock(return_value=mock_tx_hash))
        )
        
        pool_key = {"currency0": "0xToken0", "currency1": "0xToken1", "fee": 3000}
        
        result = await uniswap_fetcher.swap_exact_input(
            pool_key=pool_key,
            amount_in=100000,
            amount_out_minimum=95000,
            deadline=1234567890
        )
        
        assert result == mock_tx_hash
    
    @pytest.mark.asyncio
    async def test_update_dynamic_fee(self, uniswap_fetcher, mock_contracts):
        """Test updating dynamic fee"""
        mock_tx_hash = "0x333...444"
        mock_contracts["pool_manager"].functions.updateDynamicLPFee.return_value.transact = AsyncMock(
            return_value=Mock(hex=Mock(return_value=mock_tx_hash))
        )
        
        pool_key = {"currency0": "0xToken0", "currency1": "0xToken1", "fee": 3000}
        
        result = await uniswap_fetcher.update_dynamic_fee(
            pool_key=pool_key,
            new_fee=2500
        )
        
        assert result == mock_tx_hash
    
    @pytest.mark.asyncio
    async def test_batch_liquidity_operations(self, uniswap_fetcher, mock_contracts):
        """Test batch liquidity operations"""
        mock_tx_hash = "0x555...666"
        mock_contracts["position_manager"].functions.modifyLiquidities.return_value.transact = AsyncMock(
            return_value=Mock(hex=Mock(return_value=mock_tx_hash))
        )
        
        operations = [
            {
                "type": "mint",
                "pool_key": {"currency0": "0xToken0", "currency1": "0xToken1", "fee": 3000},
                "tick_lower": -1000,
                "tick_upper": 1000,
                "liquidity": 1000000,
                "amount0_max": 100000,
                "amount1_max": 100000,
                "recipient": "0xRecipient"
            },
            {
                "type": "increase",
                "token_id": 1,
                "liquidity": 500000,
                "amount0_max": 50000,
                "amount1_max": 50000
            }
        ]
        
        result = await uniswap_fetcher.batch_liquidity_operations(
            operations=operations,
            deadline=1234567890
        )
        
        assert result == mock_tx_hash
    
    @pytest.mark.asyncio
    async def test_quote_exact_input(self, uniswap_fetcher, mock_contracts):
        """Test getting quote for exact input"""
        mock_quote = [95000, 21000]  # amountOut, gasEstimate
        mock_contracts["quoter"].functions.quoteExactInputSingle.return_value.call = AsyncMock(
            return_value=mock_quote
        )
        
        pool_key = {"currency0": "0xToken0", "currency1": "0xToken1", "fee": 3000}
        
        amount_out, gas_estimate = await uniswap_fetcher.quote_exact_input(
            pool_key=pool_key,
            amount_in=100000,
            zero_for_one=True
        )
        
        assert amount_out == mock_quote[0]
        assert gas_estimate == mock_quote[1]
    
    def test_encode_actions(self, uniswap_fetcher):
        """Test action encoding for PositionManager"""
        actions = ["MINT_POSITION", "SETTLE_PAIR"]
        encoded = uniswap_fetcher._encode_actions(actions)
        
        assert encoded == b"\x00\x04"  # MINT_POSITION=0, SETTLE_PAIR=4
    
    def test_price_to_tick(self, uniswap_fetcher):
        """Test price to tick conversion"""
        price = Decimal("1.5")
        tick_spacing = 60
        
        tick = uniswap_fetcher._price_to_tick(price, tick_spacing)
        
        assert isinstance(tick, int)
        assert tick % tick_spacing == 0  # Should be aligned to tick spacing
    
    @pytest.mark.asyncio
    async def test_calculate_optimal_position(self, uniswap_fetcher):
        """Test optimal position calculation"""
        volatilities = {
            VolatilityTimeframe.HOUR_1: 0.1,
            VolatilityTimeframe.HOUR_4: 0.15,
            VolatilityTimeframe.HOUR_24: 0.2
        }
        
        result = await uniswap_fetcher.calculate_optimal_position(
            pool_address="0xPoolAddress",
            current_price=Decimal("1.0"),
            volatilities=volatilities,
            liquidity_depth=Decimal("1000000")
        )
        
        assert isinstance(result, LiquidityRange)
        assert result.lower_tick < result.upper_tick
        assert result.optimal_liquidity > 0
        assert result.fee_estimate > 0


class TestVolatilityComponents:
    """Test volatility-related components"""
    
    def test_volatility_weight_default(self):
        """Test default volatility weights"""
        weights = VolatilityWeight.get_default_weights()
        
        assert len(weights) == 3
        assert sum(w.weight for w in weights) == 1.0
        
        # Check timeframes are included
        timeframes = [w.timeframe for w in weights]
        assert VolatilityTimeframe.HOUR_1 in timeframes
        assert VolatilityTimeframe.HOUR_4 in timeframes
        assert VolatilityTimeframe.HOUR_24 in timeframes
    
    def test_liquidity_range_named_tuple(self):
        """Test LiquidityRange named tuple"""
        range_data = LiquidityRange(
            lower_tick=-1000,
            upper_tick=1000,
            optimal_liquidity=Decimal("500000"),
            fee_estimate=Decimal("0.001")
        )
        
        assert range_data.lower_tick == -1000
        assert range_data.upper_tick == 1000
        assert range_data.optimal_liquidity == Decimal("500000")
        assert range_data.fee_estimate == Decimal("0.001")


@pytest.mark.integration
class TestUniswapV4Integration:
    """Integration tests that would require actual blockchain connection"""
    
    @pytest.mark.skip(reason="Requires Base network connection")
    async def test_real_pool_state(self):
        """Test getting real pool state from Base network"""
        # This would test against actual deployed contracts
        pass
    
    @pytest.mark.skip(reason="Requires Base network connection and funded wallet")
    async def test_real_liquidity_operations(self):
        """Test real liquidity operations on testnet"""
        # This would test actual position management
        pass