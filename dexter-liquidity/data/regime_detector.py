import numpy as np
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from scipy import stats
import pandas as pd
from sklearn.cluster import KMeans
import logging
from utils.error_handler import ErrorHandler, DataError
from utils.memory_monitor import MemoryMonitor
from utils.parallel_processor import ParallelDataProcessor
from .volatility import MarketRegime

error_handler = ErrorHandler()
memory_monitor = MemoryMonitor()

class RegimeDetector:
    def __init__(self):
        self.parallel_detector = ParallelRegimeDetector()
        
    @error_handler.with_retries(retries=2)
    @memory_monitor.monitor
    async def detect_regime(self, market_data: Dict[str, Any]) -> MarketRegime:
        """Detect market regime with error handling and memory monitoring"""

logger = logging.getLogger(__name__)

@dataclass
class MarketConditions:
    volatility: float
    trend_strength: float
    momentum: float
    volume_profile: float
    regime: MarketRegime
    confidence: float

class EnhancedRegimeDetector:
    def __init__(
        self,
        lookback_window: int = 50,
        n_clusters: int = 4,
        use_ml: bool = True
    ):
        self.lookback_window = lookback_window
        self.n_clusters = n_clusters
        self.use_ml = use_ml
        self.kmeans = KMeans(n_clusters=n_clusters) if use_ml else None
        
    def detect_regime(
        self,
        prices: List[float],
        volumes: List[float],
        returns: List[float]
    ) -> MarketConditions:
        """Detect market regime using multiple indicators and ML"""
        try:
            # Calculate technical indicators
            volatility = self._calculate_volatility(returns)
            trend = self._calculate_trend_strength(prices)
            momentum = self._calculate_momentum(returns)
            volume_profile = self._analyze_volume_profile(volumes, returns)
            
            if self.use_ml:
                regime, confidence = self._ml_regime_detection(
                    volatility, trend, momentum, volume_profile
                )
            else:
                regime, confidence = self._rule_based_detection(
                    volatility, trend, momentum, volume_profile
                )
                
            return MarketConditions(
                volatility=volatility,
                trend_strength=trend,
                momentum=momentum,
                volume_profile=volume_profile,
                regime=regime,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Regime detection failed: {str(e)}")
            raise
            
    def _calculate_trend_strength(self, prices: List[float]) -> float:
        """Calculate trend strength using multiple methods"""
        df = pd.DataFrame({'price': prices})
        
        # Calculate EMAs
        df['ema20'] = df['price'].ewm(span=20).mean()
        df['ema50'] = df['price'].ewm(span=50).mean()
        
        # ADX calculation
        high_low = df['price'].rolling(window=14).max() - df['price'].rolling(window=14).min()
        dm_plus = df['price'].diff().clip(lower=0)
        dm_minus = (-df['price'].diff()).clip(lower=0)
        
        tr = high_low.rolling(window=14).sum()
        di_plus = 100 * (dm_plus.rolling(window=14).sum() / tr)
        di_minus = 100 * (dm_minus.rolling(window=14).sum() / tr)
        
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = dx.rolling(window=14).mean()
        
        return adx.iloc[-1] / 100  # Normalize to 0-1
        
    def _ml_regime_detection(
        self,
        volatility: float,
        trend: float,
        momentum: float,
        volume: float
    ) -> Tuple[MarketRegime, float]:
        """Use ML to detect market regime"""
        features = np.array([[volatility, trend, momentum, volume]])
        
        # Predict regime
        regime_idx = self.kmeans.predict(features)[0]
        
        # Calculate confidence using distance to cluster center
        distances = self.kmeans.transform(features)
        confidence = 1 - (distances[0][regime_idx] / np.max(distances))
        
        regime_map = {
            0: MarketRegime.RANGING,
            1: MarketRegime.TRENDING_UP,
            2: MarketRegime.TRENDING_DOWN,
            3: MarketRegime.VOLATILE
        }
        
        return regime_map[regime_idx], float(confidence)
