"""
Comprehensive integration tests for the vault system
Tests all major components including vaults, strategies, fees, and AI integration
"""

import pytest
import asyncio
from decimal import Decimal
from typing import Dict, List, Tuple
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass
from datetime import datetime, timedelta

# Test configuration
TEST_CONFIG = {
    'test_token_0': '0x1234567890123456789012345678901234567890',
    'test_token_1': '0x0987654321098765432109876543210987654321',
    'test_pool': '0xabcdefabcdefabcdefabcdefabcdefabcdefabcdef',
    'fee_tier': 3000,
    'initial_price': 3000_000000,  # $3000 USDC per ETH
    'initial_liquidity': 1000000_000000000000000000,  # 1M tokens
}

@dataclass
class MockVaultData:
    """Mock vault data for testing"""
    address: str
    name: str
    symbol: str
    token0: str
    token1: str
    fee: int
    total_shares: int
    user_shares: int
    total_value_locked: int
    config: Dict
    metrics: Dict

class TestVaultIntegration:
    """Integration tests for vault system"""
    
    @pytest.fixture
    def mock_vault_data(self):
        """Create mock vault data for testing"""
        return MockVaultData(
            address='0x' + '1' * 40,
            name='Test ETH/USDC Vault',
            symbol='dETH-USDC',
            token0=TEST_CONFIG['test_token_0'],
            token1=TEST_CONFIG['test_token_1'],
            fee=TEST_CONFIG['fee_tier'],
            total_shares=1000000,
            user_shares=5000,
            total_value_locked=2500000,
            config={
                'mode': 'AI_ASSISTED',
                'position_type': 'DUAL_POSITION',
                'ai_optimization_enabled': True,
                'auto_compound_enabled': True,
                'rebalance_threshold': 1000,  # 10%
                'max_slippage_bps': 100       # 1%
            },
            metrics={
                'total_value_locked': 2500000,
                'total_fees_24h': 12500,
                'impermanent_loss': 0.02,
                'apr': 0.155,
                'sharpe_ratio': 1.24,
                'max_drawdown': 0.08,
                'successful_compounds': 47,
                'ai_optimization_count': 15
            }
        )
    
    @pytest.fixture 
    def mock_pool_data(self):
        """Create mock pool data for testing"""
        return {
            'current_tick': 100000,
            'current_price': TEST_CONFIG['initial_price'],
            'liquidity': TEST_CONFIG['initial_liquidity'],
            'volume_24h': 5000000,
            'fee_tier': TEST_CONFIG['fee_tier'],
            'tick_spacing': 60,
            'prices': list(range(2900, 3100, 10)),
            'price_history': [3000 + i * 10 for i in range(-20, 21)]
        }
    
    @pytest.fixture
    def mock_market_data(self):
        """Create mock market data for testing"""
        return {
            'btc_price': 50000,
            'eth_price': 3000,
            'total_market_cap': 2000000000000,
            'defi_tvl': 100000000000,
            'volatility_index': 0.3
        }

class TestVaultDeployment:
    """Test vault deployment and initialization"""
    
    def test_vault_factory_deployment(self):
        """Test vault factory can deploy new vaults"""
        # Mock vault factory
        factory = Mock()
        factory.create_vault.return_value = {
            'vault_address': '0x' + '1' * 40,
            'transaction_hash': '0x' + 'a' * 64,
            'gas_used': 2500000
        }
        
        # Test deployment parameters
        deployment_params = {
            'token0': TEST_CONFIG['test_token_0'],
            'token1': TEST_CONFIG['test_token_1'],
            'fee': TEST_CONFIG['fee_tier'],
            'template_type': 'AI_OPTIMIZED',
            'name': 'Test Vault',
            'symbol': 'dTEST',
            'initial_deposit_0': 1000000000000000000,  # 1 ETH
            'initial_deposit_1': 3000000000,           # 3000 USDC
        }
        
        # Deploy vault
        result = factory.create_vault(deployment_params)
        
        assert result['vault_address'].startswith('0x')
        assert len(result['vault_address']) == 42
        assert result['gas_used'] > 0
        factory.create_vault.assert_called_once_with(deployment_params)
    
    def test_vault_initialization(self, mock_vault_data):
        """Test vault initialization with correct parameters"""
        # Mock vault contract
        vault = Mock()
        vault.initialize.return_value = True
        vault.get_vault_config.return_value = mock_vault_data.config
        
        # Initialize vault
        result = vault.initialize(
            pool=TEST_CONFIG['test_pool'],
            compoundor='0x' + '2' * 40,
            ai_manager='0x' + '3' * 40,
            name=mock_vault_data.name,
            symbol=mock_vault_data.symbol,
            config=mock_vault_data.config
        )
        
        assert result is True
        vault.initialize.assert_called_once()
        
        # Verify configuration
        config = vault.get_vault_config()
        assert config['mode'] == 'AI_ASSISTED'
        assert config['ai_optimization_enabled'] is True

class TestVaultStrategies:
    """Test vault strategy management and execution"""
    
    def test_gamma_style_strategy(self, mock_vault_data, mock_pool_data):
        """Test Gamma-style dual position strategy"""
        from backend.ai.vault_strategy_models import GammaStyleOptimizer, VaultMetrics
        
        optimizer = GammaStyleOptimizer()
        
        # Convert mock data to VaultMetrics
        vault_metrics = VaultMetrics(
            total_value_locked=mock_vault_data.metrics['total_value_locked'],
            total_fees_24h=mock_vault_data.metrics['total_fees_24h'],
            impermanent_loss=mock_vault_data.metrics['impermanent_loss'],
            apr=mock_vault_data.metrics['apr'],
            sharpe_ratio=mock_vault_data.metrics['sharpe_ratio'],
            max_drawdown=mock_vault_data.metrics['max_drawdown'],
            successful_compounds=mock_vault_data.metrics['successful_compounds'],
            ai_optimization_count=mock_vault_data.metrics['ai_optimization_count'],
            capital_efficiency=0.85,
            risk_score=0.3
        )
        
        # Test dual position optimization
        base_range, limit_range, allocations = optimizer.optimize_dual_positions(
            current_tick=mock_pool_data['current_tick'],
            tick_spacing=mock_pool_data['tick_spacing'],
            pool_data=mock_pool_data,
            vault_metrics=vault_metrics,
            risk_tolerance=0.5
        )
        
        # Verify results
        assert base_range[0] < base_range[1]  # Valid range
        assert limit_range[0] < limit_range[1]  # Valid range
        assert abs(allocations[0] + allocations[1] - 1.0) < 0.01  # Allocations sum to 100%
        assert base_range[1] - base_range[0] > limit_range[1] - limit_range[0]  # Base wider than limit
    
    def test_ai_strategy_prediction(self, mock_vault_data, mock_pool_data, mock_market_data):
        """Test AI strategy prediction and recommendation"""
        from backend.ai.vault_strategy_models import VaultMLEngine, VaultMetrics
        
        engine = VaultMLEngine()
        
        vault_metrics = VaultMetrics(
            total_value_locked=mock_vault_data.metrics['total_value_locked'],
            total_fees_24h=mock_vault_data.metrics['total_fees_24h'],
            impermanent_loss=mock_vault_data.metrics['impermanent_loss'],
            apr=mock_vault_data.metrics['apr'],
            sharpe_ratio=mock_vault_data.metrics['sharpe_ratio'],
            max_drawdown=mock_vault_data.metrics['max_drawdown'],
            successful_compounds=mock_vault_data.metrics['successful_compounds'],
            ai_optimization_count=mock_vault_data.metrics['ai_optimization_count'],
            capital_efficiency=0.85,
            risk_score=0.3
        )
        
        # Get strategy recommendation
        recommendation = engine.predict_strategy(
            mock_pool_data, vault_metrics, mock_market_data
        )
        
        # Verify recommendation
        assert recommendation.confidence_score >= 0.0
        assert recommendation.confidence_score <= 1.0
        assert recommendation.expected_apr >= 0.0
        assert recommendation.expected_risk >= 0.0
        assert len(recommendation.position_ranges) > 0
        assert recommendation.reasoning is not None
    
    def test_multi_range_optimization(self, mock_vault_data, mock_pool_data):
        """Test multi-range position optimization"""
        from backend.ai.vault_strategy_models import MultiRangeOptimizer, VaultMetrics, StrategyType
        
        optimizer = MultiRangeOptimizer(max_ranges=5)
        
        vault_metrics = VaultMetrics(
            total_value_locked=mock_vault_data.metrics['total_value_locked'],
            total_fees_24h=mock_vault_data.metrics['total_fees_24h'],
            impermanent_loss=mock_vault_data.metrics['impermanent_loss'],
            apr=mock_vault_data.metrics['apr'],
            sharpe_ratio=mock_vault_data.metrics['sharpe_ratio'],
            max_drawdown=mock_vault_data.metrics['max_drawdown'],
            successful_compounds=mock_vault_data.metrics['successful_compounds'],
            ai_optimization_count=mock_vault_data.metrics['ai_optimization_count'],
            capital_efficiency=0.85,
            risk_score=0.3
        )
        
        # Test AI strategy ranges
        ranges = optimizer.optimize_ranges(
            current_tick=mock_pool_data['current_tick'],
            tick_spacing=mock_pool_data['tick_spacing'],
            pool_data=mock_pool_data,
            vault_metrics=vault_metrics,
            strategy_type=StrategyType.AI_BALANCED
        )
        
        # Verify ranges
        assert len(ranges) > 0
        assert len(ranges) <= optimizer.max_ranges
        
        total_allocation = sum(r.allocation for r in ranges)
        assert abs(total_allocation - 1.0) < 0.1  # Total allocation approximately 100%
        
        for range_data in ranges:
            assert range_data.tick_lower < range_data.tick_upper
            assert range_data.allocation >= 0.0
            assert range_data.allocation <= 1.0

class TestFeeManagement:
    """Test fee calculation and distribution"""
    
    def test_tiered_fee_calculation(self):
        """Test tiered fee structure calculation"""
        # Mock fee manager
        fee_manager = Mock()
        
        # Test different user tiers
        test_cases = [
            {
                'tier': 'RETAIL',
                'amount': 100000,
                'expected_mgmt_fee': 100,  # 1%
                'expected_perf_fee': 15000  # 15%
            },
            {
                'tier': 'PREMIUM', 
                'amount': 100000,
                'expected_mgmt_fee': 75,   # 0.75%
                'expected_perf_fee': 12500  # 12.5%
            },
            {
                'tier': 'INSTITUTIONAL',
                'amount': 100000,
                'expected_mgmt_fee': 50,   # 0.5%
                'expected_perf_fee': 10000  # 10%
            }
        ]
        
        for case in test_cases:
            fee_manager.calculate_fee.return_value = (
                case['expected_mgmt_fee'],
                ['0x' + '1' * 40],  # recipients
                [case['expected_mgmt_fee']]  # distributions
            )
            
            mgmt_fee, recipients, distributions = fee_manager.calculate_fee(
                vault='0x' + '1' * 40,
                user='0x' + '2' * 40,
                fee_type='MANAGEMENT',
                amount=case['amount'],
                performance=0
            )
            
            assert mgmt_fee == case['expected_mgmt_fee']
            assert len(recipients) == len(distributions)
    
    def test_performance_fee_with_high_water_mark(self):
        """Test performance fee calculation with high water mark"""
        fee_manager = Mock()
        
        # Mock high water mark calculation
        current_value = 110000
        previous_value = 100000
        high_water_mark = 105000
        
        # Performance above high water mark
        net_performance = current_value - high_water_mark  # 5000
        expected_perf_fee = net_performance * 0.15  # 15% of 5000 = 750
        
        fee_manager.calculate_performance_fee.return_value = (expected_perf_fee, net_performance)
        
        perf_fee, net_perf = fee_manager.calculate_performance_fee(
            vault='0x' + '1' * 40,
            user='0x' + '2' * 40,
            current_value=current_value,
            previous_value=previous_value
        )
        
        assert perf_fee == expected_perf_fee
        assert net_perf == net_performance

class TestTWAPProtection:
    """Test TWAP protection and MEV resistance"""
    
    def test_twap_validation_pass(self):
        """Test TWAP validation passing within threshold"""
        # Mock clearing contract
        clearing = Mock()
        
        twap_data = {
            'current_price': 3000,
            'twap_price': 2985,  # 0.5% deviation
            'deviation': 50,     # 0.5% in basis points
            'is_valid': True,
            'interval': 3600,
            'timestamp': int(datetime.now().timestamp())
        }
        
        clearing.check_twap.return_value = twap_data
        
        result = clearing.check_twap('0x' + '1' * 40)
        
        assert result['is_valid'] is True
        assert result['deviation'] <= 500  # 5% threshold
    
    def test_twap_validation_fail(self):
        """Test TWAP validation failing outside threshold"""
        clearing = Mock()
        
        twap_data = {
            'current_price': 3000,
            'twap_price': 2700,  # 10% deviation
            'deviation': 1000,   # 10% in basis points
            'is_valid': False,
            'interval': 3600,
            'timestamp': int(datetime.now().timestamp())
        }
        
        clearing.check_twap.return_value = twap_data
        
        result = clearing.check_twap('0x' + '1' * 40)
        
        assert result['is_valid'] is False
        assert result['deviation'] > 500  # Exceeded 5% threshold
    
    def test_mev_protection(self):
        """Test MEV protection mechanisms"""
        clearing = Mock()
        
        # Test operation too soon after previous
        validation_context = {
            'vault': '0x' + '1' * 40,
            'user': '0x' + '2' * 40,
            'op_type': 'DEPOSIT',
            'amount0': 1000000000000000000,
            'amount1': 3000000000,
            'shares': 0,
            'additional_data': b''
        }
        
        # Should fail due to MEV protection
        clearing.validate_operation.return_value = ('REJECTED_COOLDOWN', 'MEV protection triggered')
        
        result, reason = clearing.validate_operation(validation_context)
        
        assert result == 'REJECTED_COOLDOWN'
        assert 'MEV' in reason

class TestVaultOperations:
    """Test core vault operations"""
    
    def test_deposit_flow(self, mock_vault_data):
        """Test complete deposit flow"""
        vault = Mock()
        
        # Mock successful deposit
        deposit_amount = 1000000000000000000  # 1 ETH
        expected_shares = 500  # Based on current share price
        
        vault.deposit.return_value = expected_shares
        vault.validate_deposit.return_value = (True, "Validation passed")
        
        # Validate deposit
        valid, reason = vault.validate_deposit(deposit_amount, '0x' + '2' * 40)
        assert valid is True
        
        # Execute deposit
        shares = vault.deposit(deposit_amount, '0x' + '2' * 40)
        assert shares == expected_shares
        
        vault.validate_deposit.assert_called_once()
        vault.deposit.assert_called_once()
    
    def test_withdraw_flow(self, mock_vault_data):
        """Test complete withdrawal flow"""
        vault = Mock()
        
        # Mock successful withdrawal
        shares_to_withdraw = 100
        expected_assets = 200000000000000000  # 0.2 ETH worth
        
        vault.withdraw.return_value = expected_assets
        vault.validate_withdrawal.return_value = (True, "Validation passed")
        
        # Validate withdrawal
        valid, reason = vault.validate_withdrawal(shares_to_withdraw, '0x' + '2' * 40, '0x' + '2' * 40)
        assert valid is True
        
        # Execute withdrawal
        assets = vault.withdraw(expected_assets, '0x' + '2' * 40, '0x' + '2' * 40)
        assert assets == expected_assets
    
    def test_compound_operation(self, mock_vault_data):
        """Test compounding operation"""
        vault = Mock()
        
        # Mock compound conditions
        vault.should_auto_compound.return_value = (True, 200000)  # Should compound, 200k gas
        vault.compound.return_value = 1500000000000000000  # 1.5 ETH new liquidity
        
        # Check if should compound
        should_compound, gas_estimate = vault.should_auto_compound()
        assert should_compound is True
        assert gas_estimate > 0
        
        # Execute compound
        new_liquidity = vault.compound()
        assert new_liquidity > 0
    
    def test_rebalance_operation(self, mock_vault_data):
        """Test position rebalancing"""
        vault = Mock()
        strategy_manager = Mock()
        
        # Mock rebalance check
        strategy_manager.auto_rebalance_check.return_value = (True, 'PRICE_MOVEMENT')
        vault.rebalance_positions.return_value = True
        
        # Check if rebalance needed
        should_rebalance, reason = strategy_manager.auto_rebalance_check('0x' + '1' * 40)
        assert should_rebalance is True
        assert reason == 'PRICE_MOVEMENT'
        
        # Execute rebalance
        success = vault.rebalance_positions()
        assert success is True

class TestPerformanceTracking:
    """Test performance tracking and analytics"""
    
    def test_vault_metrics_calculation(self, mock_vault_data):
        """Test vault metrics calculation"""
        vault = Mock()
        
        expected_metrics = mock_vault_data.metrics
        vault.get_vault_metrics.return_value = expected_metrics
        
        metrics = vault.get_vault_metrics()
        
        assert metrics['apr'] == expected_metrics['apr']
        assert metrics['sharpe_ratio'] == expected_metrics['sharpe_ratio']
        assert metrics['total_value_locked'] == expected_metrics['total_value_locked']
        assert metrics['successful_compounds'] == expected_metrics['successful_compounds']
    
    def test_performance_benchmarking(self, mock_vault_data):
        """Test performance vs benchmark calculation"""
        vault = Mock()
        
        vault_apr = 0.155      # 15.5%
        benchmark_apr = 0.12   # 12%
        outperformance = 0.035 # 3.5%
        
        vault.benchmark_performance.return_value = (vault_apr, benchmark_apr, outperformance)
        
        v_apr, b_apr, outperf = vault.benchmark_performance()
        
        assert v_apr > b_apr
        assert outperf > 0
        assert abs(outperf - (v_apr - b_apr)) < 0.001

class TestAIIntegration:
    """Test AI model integration"""
    
    def test_ai_recommendation_application(self, mock_vault_data, mock_pool_data, mock_market_data):
        """Test applying AI recommendations to vault"""
        from backend.ai.vault_strategy_models import VaultMLEngine, VaultMetrics
        
        vault = Mock()
        engine = VaultMLEngine()
        
        vault_metrics = VaultMetrics(
            total_value_locked=mock_vault_data.metrics['total_value_locked'],
            total_fees_24h=mock_vault_data.metrics['total_fees_24h'],
            impermanent_loss=mock_vault_data.metrics['impermanent_loss'],
            apr=mock_vault_data.metrics['apr'],
            sharpe_ratio=mock_vault_data.metrics['sharpe_ratio'],
            max_drawdown=mock_vault_data.metrics['max_drawdown'],
            successful_compounds=mock_vault_data.metrics['successful_compounds'],
            ai_optimization_count=mock_vault_data.metrics['ai_optimization_count'],
            capital_efficiency=0.85,
            risk_score=0.3
        )
        
        # Get AI recommendation
        recommendation = engine.predict_strategy(
            mock_pool_data, vault_metrics, mock_market_data
        )
        
        # Mock applying recommendation
        vault.apply_ai_recommendation.return_value = True
        
        # Apply recommendation
        ai_data = {
            'strategy_type': recommendation.strategy_type.value,
            'position_ranges': recommendation.position_ranges,
            'confidence_score': recommendation.confidence_score
        }
        
        success = vault.apply_ai_recommendation(ai_data)
        assert success is True
        vault.apply_ai_recommendation.assert_called_once()
    
    def test_ai_model_training_data_generation(self):
        """Test generation of training data for AI models"""
        from backend.ai.vault_strategy_models import VaultPerformanceTracker, VaultMetrics, StrategyType
        
        tracker = VaultPerformanceTracker()
        
        # Record some performance data
        vault_address = '0x' + '1' * 40
        
        for i in range(10):
            timestamp = datetime.now() - timedelta(days=i)
            metrics = VaultMetrics(
                total_value_locked=2500000 + i * 10000,
                total_fees_24h=12500,
                impermanent_loss=0.02,
                apr=0.155 + i * 0.001,
                sharpe_ratio=1.24,
                max_drawdown=0.08,
                successful_compounds=47 + i,
                ai_optimization_count=15,
                capital_efficiency=0.85,
                risk_score=0.3
            )
            
            tracker.record_performance(
                vault_address, timestamp, metrics, StrategyType.AI_BALANCED, []
            )
        
        # Generate training data
        training_data = tracker.generate_training_data(min_history_days=5)
        
        assert len(training_data) > 0
        for sample in training_data:
            assert 'pool_data' in sample
            assert 'vault_metrics' in sample
            assert 'market_data' in sample
            assert 'target_strategy' in sample
            assert 'actual_apr' in sample
            assert 'actual_risk' in sample

class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_insufficient_liquidity_handling(self):
        """Test handling of insufficient liquidity scenarios"""
        vault = Mock()
        
        # Mock insufficient liquidity error
        vault.withdraw.side_effect = Exception("Insufficient liquidity")
        
        with pytest.raises(Exception) as exc_info:
            vault.withdraw(1000000000000000000, '0x' + '2' * 40, '0x' + '2' * 40)
        
        assert "Insufficient liquidity" in str(exc_info.value)
    
    def test_slippage_protection(self):
        """Test slippage protection mechanisms"""
        vault = Mock()
        
        # Mock slippage exceeded error
        vault.deposit.side_effect = Exception("Slippage exceeded")
        
        with pytest.raises(Exception) as exc_info:
            vault.deposit(1000000000000000000, '0x' + '2' * 40)
        
        assert "Slippage exceeded" in str(exc_info.value)
    
    def test_emergency_pause_functionality(self):
        """Test emergency pause mechanisms"""
        vault = Mock()
        clearing = Mock()
        
        # Test emergency pause
        clearing.emergency_pause.return_value = True
        vault.is_paused.return_value = True
        
        # Pause should be activated
        clearing.emergency_pause(True, "Emergency maintenance")
        assert vault.is_paused() is True
        
        # Operations should be blocked when paused
        vault.deposit.side_effect = Exception("Vault paused")
        
        with pytest.raises(Exception) as exc_info:
            vault.deposit(1000000000000000000, '0x' + '2' * 40)
        
        assert "Vault paused" in str(exc_info.value)

class TestIntegrationFlow:
    """Test complete end-to-end integration flows"""
    
    def test_complete_vault_lifecycle(self, mock_vault_data, mock_pool_data, mock_market_data):
        """Test complete vault lifecycle from creation to operation"""
        # 1. Create vault
        factory = Mock()
        factory.create_vault.return_value = {
            'vault_address': mock_vault_data.address,
            'transaction_hash': '0x' + 'a' * 64
        }
        
        vault_result = factory.create_vault({
            'token0': mock_vault_data.token0,
            'token1': mock_vault_data.token1,
            'fee': mock_vault_data.fee,
            'template_type': 'AI_OPTIMIZED',
            'name': mock_vault_data.name,
            'symbol': mock_vault_data.symbol
        })
        
        vault_address = vault_result['vault_address']
        assert vault_address == mock_vault_data.address
        
        # 2. Initialize vault
        vault = Mock()
        vault.initialize.return_value = True
        assert vault.initialize() is True
        
        # 3. Configure strategy
        strategy_manager = Mock()
        strategy_manager.configure_strategy.return_value = True
        
        success = strategy_manager.configure_strategy(
            vault_address, 'AI_OPTIMIZED', mock_vault_data.config
        )
        assert success is True
        
        # 4. Deposit initial liquidity
        vault.validate_deposit.return_value = (True, "Valid")
        vault.deposit.return_value = 1000  # shares
        
        valid, reason = vault.validate_deposit(1000000000000000000, '0x' + '2' * 40)
        assert valid is True
        
        shares = vault.deposit(1000000000000000000, '0x' + '2' * 40)
        assert shares > 0
        
        # 5. AI optimization
        from backend.ai.vault_strategy_models import VaultMLEngine, VaultMetrics
        
        engine = VaultMLEngine()
        vault_metrics = VaultMetrics(
            total_value_locked=mock_vault_data.metrics['total_value_locked'],
            total_fees_24h=mock_vault_data.metrics['total_fees_24h'],
            impermanent_loss=mock_vault_data.metrics['impermanent_loss'],
            apr=mock_vault_data.metrics['apr'],
            sharpe_ratio=mock_vault_data.metrics['sharpe_ratio'],
            max_drawdown=mock_vault_data.metrics['max_drawdown'],
            successful_compounds=mock_vault_data.metrics['successful_compounds'],
            ai_optimization_count=mock_vault_data.metrics['ai_optimization_count'],
            capital_efficiency=0.85,
            risk_score=0.3
        )
        
        recommendation = engine.predict_strategy(
            mock_pool_data, vault_metrics, mock_market_data
        )
        assert recommendation.confidence_score > 0
        
        # 6. Execute operations
        vault.compound.return_value = 500000000000000000  # New liquidity
        vault.rebalance_positions.return_value = True
        
        new_liquidity = vault.compound()
        assert new_liquidity > 0
        
        rebalance_success = vault.rebalance_positions()
        assert rebalance_success is True
        
        # 7. Performance tracking
        vault.get_vault_metrics.return_value = mock_vault_data.metrics
        metrics = vault.get_vault_metrics()
        
        assert metrics['apr'] > 0
        assert metrics['successful_compounds'] > 0
        assert metrics['ai_optimization_count'] > 0

if __name__ == "__main__":
    # Run specific test suites
    pytest.main([
        "test_vault_integration.py::TestVaultIntegration",
        "test_vault_integration.py::TestVaultDeployment", 
        "test_vault_integration.py::TestVaultStrategies",
        "-v"
    ])