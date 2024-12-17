import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import pandas as pd
from scipy import stats
import logging
from numpy.typing import NDArray
import warnings
from scipy.optimize import minimize
from memory_profiler import profile

# Configure logging
logger = logging.getLogger(__name__)

class VolatilityError(Exception):
    """Custom exception for volatility calculation errors"""
    pass

class InsufficientDataError(VolatilityError):
    """Exception for insufficient price history"""
    pass

class MarketRegime(Enum):
    RANGING = "ranging"
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    VOLATILE = "volatile"

@dataclass
class DataRequirements:
    MIN_PRICES: int = 30
    MIN_RETURNS: int = 20
    MIN_GARCH: int = 50
    OPTIMAL_HISTORY: int = 100

@dataclass
class VolatilityMetrics:
    historical_vol: float
    forecast_vol: float
    regime: MarketRegime
    confidence: float
    garch_parameters: Dict[str, float]

class EnhancedVolatilityCalculator:
    def __init__(
        self,
        ema_window: int = 24,
        garch_p: int = 1,
        garch_q: int = 1,
        regime_threshold: float = 0.15,
        use_memory_optimization: bool = True
    ):
        self.ema_window = ema_window
        self.garch_p = garch_p
        self.garch_q = garch_q
        self.regime_threshold = regime_threshold
        self.use_memory_optimization = use_memory_optimization
        self.requirements = DataRequirements()
        
        logger.info(
            f"volatility.py: Initialized calculator with window={ema_window}, "
            f"garch_p={garch_p}, garch_q={garch_q}, threshold={regime_threshold}, "
            f"memory_optimization={use_memory_optimization}"
        )

    def _validate_price_data(
        self,
        prices: List[float],
        min_length: int,
        data_type: str
    ) -> NDArray:
        """Validate price data and convert to numpy array with memory optimization"""
        try:
            if not prices:
                raise InsufficientDataError(f"Empty {data_type} price list provided")
                
            if len(prices) < min_length:
                raise InsufficientDataError(
                    f"Insufficient {data_type} data. Required: {min_length}, "
                    f"Provided: {len(prices)}"
                )
            
            # Convert to numpy array with memory optimization
            if self.use_memory_optimization:
                # Use float32 instead of float64 to save memory
                return np.array(prices, dtype=np.float32)
            else:
                return np.array(prices)
                
        except Exception as e:
            logger.error(f"volatility.py: Error validating {data_type} data: {str(e)}")
            raise

    @profile
    def calculate_parkinson_volatility(
        self,
        high_prices: List[float],
        low_prices: List[float]
    ) -> float:
        """Memory-optimized Parkinson volatility calculation"""
        logger.info(f"volatility.py: Calculating Parkinson volatility")
        
        try:
            # Validate and convert data
            highs = self._validate_price_data(high_prices, self.requirements.MIN_PRICES, "high")
            lows = self._validate_price_data(low_prices, self.requirements.MIN_PRICES, "low")
            
            if len(highs) != len(lows):
                raise VolatilityError(
                    f"Mismatched price lists lengths: high={len(highs)}, low={len(lows)}"
                )
            
            # Calculate in chunks to optimize memory usage
            chunk_size = 1000
            n_chunks = len(highs) // chunk_size + (1 if len(highs) % chunk_size else 0)
            
            sum_squared = 0
            for i in range(n_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, len(highs))
                
                chunk_high = highs[start_idx:end_idx]
                chunk_low = lows[start_idx:end_idx]
                
                ln_hl_ratio = np.log(chunk_high / chunk_low)
                sum_squared += np.sum(ln_hl_ratio ** 2)
                
                # Free memory
                del ln_hl_ratio
            
            vol = np.sqrt(sum_squared / (4 * len(highs) * np.log(2)))
            logger.debug(f"volatility.py: Parkinson volatility = {vol:.6f}")
            return float(vol)
            
        except Exception as e:
            logger.error(f"volatility.py: Parkinson volatility calculation failed: {str(e)}")
            raise

    def _garch_likelihood(
        self,
        params: NDArray,
        returns: NDArray
    ) -> float:
        """Calculate GARCH likelihood for parameter optimization"""
        try:
            omega, alpha, beta = params
            
            # Parameter constraints
            if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
                return np.inf
                
            n = len(returns)
            h = np.zeros(n)
            h[0] = np.var(returns)
            
            for t in range(1, n):
                h[t] = omega + alpha * returns[t-1]**2 + beta * h[t-1]
                
            likelihood = np.sum(np.log(h) + returns**2 / h)
            return likelihood
            
        except Exception as e:
            logger.error(f"volatility.py: GARCH likelihood calculation failed: {str(e)}")
            raise

    @profile
    def calculate_garch_forecast(
        self,
        returns: List[float]
    ) -> Tuple[float, float, Dict[str, float]]:
        """Enhanced GARCH calculation with parameter optimization"""
        logger.info("volatility.py: Calculating GARCH forecast")
        
        try:
            returns_array = self._validate_price_data(
                returns, 
                self.requirements.MIN_GARCH,
                "returns"
            )
            
            # Optimize GARCH parameters
            initial_guess = np.array([0.01, 0.1, 0.8])
            bounds = ((1e-6, 1), (0, 1), (0, 1))
            
            result = minimize(
                self._garch_likelihood,
                initial_guess,
                args=(returns_array,),
                bounds=bounds,
                method='L-BFGS-B'
            )
            
            if not result.success:
                logger.warning(f"volatility.py: GARCH optimization did not converge: {result.message}")
            
            omega, alpha, beta = result.x
            
            # Calculate forecast
            h = np.zeros(len(returns_array))
            h[0] = np.var(returns_array)
            
            for t in range(1, len(returns_array)):
                h[t] = omega + alpha * returns_array[t-1]**2 + beta * h[t-1]
            
            forecast = np.sqrt(h[-1])
            
            # Calculate forecast confidence using multiple metrics
            confidence_jb = 1 - stats.jarque_bera(returns_array)[1]
            confidence_norm = 1 - stats.normaltest(returns_array)[1]
            confidence = (confidence_jb + confidence_norm) / 2
            
            params = {
                'omega': float(omega),
                'alpha': float(alpha),
                'beta': float(beta),
                'persistence': float(alpha + beta)
            }
            
            logger.debug(
                f"volatility.py: GARCH forecast={forecast:.6f}, confidence={confidence:.4f}, "
                f"parameters={params}"
            )
            
            return float(forecast), float(confidence), params
            
        except Exception as e:
            logger.error(f"volatility.py: GARCH calculation failed: {str(e)}")
            raise

    # ... [Previous detect_market_regime method remains the same]

    async def get_comprehensive_volatility(
        self,
        prices: List[float],
        highs: List[float],
        lows: List[float]
    ) -> VolatilityMetrics:
        """Calculate comprehensive volatility metrics with memory optimization"""
        logger.info("volatility.py: Starting comprehensive volatility calculation")
        
        try:
            # Validate all input data
            prices_array = self._validate_price_data(prices, self.requirements.MIN_PRICES, "prices")
            
            # Calculate returns using numpy operations for memory efficiency
            returns = np.diff(np.log(prices_array))
            
            # Calculate metrics
            hist_vol = float(np.std(returns, ddof=1) * np.sqrt(252))
            park_vol = self.calculate_parkinson_volatility(highs, lows)
            garch_forecast, confidence, garch_params = self.calculate_garch_forecast(returns.tolist())
            
            regime = self.detect_market_regime(prices, returns.tolist())
            
            # Adjust volatility based on regime
            if regime == MarketRegime.VOLATILE:
                adjusted_vol = max(hist_vol, park_vol) * 1.2
            elif regime == MarketRegime.RANGING:
                adjusted_vol = min(hist_vol, park_vol) * 0.8
            else:
                adjusted_vol = (hist_vol + park_vol) / 2
            
            metrics = VolatilityMetrics(
                historical_vol=adjusted_vol,
                forecast_vol=garch_forecast,
                regime=regime,
                confidence=confidence,
                garch_parameters=garch_params
            )
            
            logger.info(
                f"volatility.py: Completed volatility calculation - "
                f"regime={regime.value}, confidence={confidence:.4f}"
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"volatility.py: Comprehensive volatility calculation failed: {str(e)}")
            raise