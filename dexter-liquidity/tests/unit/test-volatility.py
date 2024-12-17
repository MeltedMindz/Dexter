import pytest
import numpy as np
from data.volatility import (
    EnhancedVolatilityCalculator,
    MarketRegime,
    VolatilityError,
    InsufficientDataError
)

@pytest.fixture
def calculator():
    return EnhancedVolatilityCalculator()

@pytest.fixture
def sample_data():
    # Generate realistic price data
    np.random.seed(42)
    n_points = 200
    base_price = 100
    returns = np.random.normal(0, 0.02, n_points)
    prices = base_price * np.exp(np.cumsum(returns))
    highs = prices * (1 + np.abs(np.random.normal(0, 0.01, n_points)))
    lows = prices * (1 - np.abs(np.random.normal(0, 0.01, n_points)))
    return prices.tolist(), highs.tolist(), lows.tolist()

class TestVolatilityCalculator:
    def test_data_validation(self, calculator):
        """Test input data validation"""
        with pytest.raises(InsufficientDataError):
            calculator._validate_price_data([], 10, "test")
            
        with pytest.raises(InsufficientDataError):
            calculator._validate_price_data([1, 2], 10, "test")
            
    async def test_comprehensive_volatility(self, calculator, sample_data):
        """Test comprehensive volatility calculation"""
        prices, highs, lows = sample_data
        metrics = await calculator.get_comprehensive_volatility(prices, highs, lows)
        
        assert metrics.historical_vol > 0
        assert metrics.forecast_vol > 0
        assert isinstance(metrics.regime, MarketRegime)
        assert 0 <= metrics.confidence <= 1
        
    def test_parkinson_volatility(self, calculator, sample_data):
        """Test Parkinson volatility calculation"""
        _, highs, lows = sample_data
        vol = calculator.calculate_parkinson_volatility(highs, lows)
        
        assert vol > 0
        assert isinstance(vol, float)
        
    def test_garch_forecast(self, calculator, sample_data):
        """Test GARCH forecasting"""
        prices, _, _ = sample_data
        returns = np.diff(np.log(prices)).tolist()
        
        forecast, confidence, params = calculator.calculate_garch_forecast(returns)
        
        assert forecast > 0
        assert 0 <= confidence <= 1
        assert 0 <= params['alpha'] + params['beta'] < 1
        
    @pytest.mark.parametrize("regime_data", [
        # Ranging market
        (np.linspace(100, 101, 50), MarketRegime.RANGING),
        # Trending up
        (np.exp(np.linspace(0, 0.5, 50)), MarketRegime.TRENDING_UP),
        # Trending down
        (np.exp(np.linspace(0.5, 0, 50)), MarketRegime.TRENDING_DOWN),
        # Volatile
        (100 + 10 * np.random.randn(50), MarketRegime.VOLATILE)
    ])
    def test_market_regimes(self, calculator, regime_data):
        """Test market regime detection"""
        prices, expected_regime = regime_data
        returns = np.diff(np.log(prices))
        
        regime = calculator.detect_market_regime(prices.tolist(), returns.tolist())
        assert regime == expected_regime
