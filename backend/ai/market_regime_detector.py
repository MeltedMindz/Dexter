"""
Advanced Market Regime Detection with ML Classification
Implements sophisticated market regime classification using multiple ML models
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta
import asyncio
import aiohttp
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import talib
from scipy import stats
from hmmlearn import hmm
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    """Market regime classifications"""
    TRENDING_UP = 0
    TRENDING_DOWN = 1
    RANGING = 2
    HIGH_VOLATILITY = 3
    LOW_VOLATILITY = 4
    BREAKOUT = 5
    REVERSAL = 6
    UNKNOWN = 8

@dataclass
class RegimeDetection:
    """Market regime detection result"""
    regime: MarketRegime
    confidence: float
    probability_distribution: Dict[MarketRegime, float]
    features: Dict[str, float]
    timestamp: datetime
    reasoning: str

@dataclass
class MarketFeatures:
    """Comprehensive market feature set"""
    # Price action features
    returns_1h: float
    returns_4h: float
    returns_24h: float
    returns_7d: float
    volatility_1h: float
    volatility_24h: float
    volatility_7d: float
    
    # Technical indicators
    rsi_14: float
    macd_signal: float
    bb_position: float  # Bollinger Band position
    atr_ratio: float
    volume_ratio: float
    
    # Market structure
    trend_strength: float
    support_resistance_ratio: float
    price_momentum: float
    volume_momentum: float
    
    # Regime-specific features
    regime_persistence: float
    regime_transition_prob: float
    market_stress_index: float
    liquidity_depth_ratio: float

class TechnicalIndicators:
    """Technical analysis indicators for regime detection"""
    
    @staticmethod
    def calculate_rsi(prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI with boundary handling"""
        if len(prices) < period + 1:
            return 50.0  # Neutral RSI
        
        try:
            rsi = talib.RSI(prices, timeperiod=period)
            return float(rsi[-1]) if not np.isnan(rsi[-1]) else 50.0
        except:
            return 50.0
    
    @staticmethod
    def calculate_macd(prices: np.ndarray) -> Tuple[float, float, float]:
        """Calculate MACD indicators"""
        if len(prices) < 26:
            return 0.0, 0.0, 0.0
        
        try:
            macd, macd_signal, macd_hist = talib.MACD(prices)
            return (
                float(macd[-1]) if not np.isnan(macd[-1]) else 0.0,
                float(macd_signal[-1]) if not np.isnan(macd_signal[-1]) else 0.0,
                float(macd_hist[-1]) if not np.isnan(macd_hist[-1]) else 0.0
            )
        except:
            return 0.0, 0.0, 0.0
    
    @staticmethod
    def calculate_bollinger_bands(prices: np.ndarray, period: int = 20) -> Tuple[float, float, float]:
        """Calculate Bollinger Bands"""
        if len(prices) < period:
            return float(prices[-1]), float(prices[-1]), float(prices[-1])
        
        try:
            upper, middle, lower = talib.BBANDS(prices, timeperiod=period)
            return (
                float(upper[-1]) if not np.isnan(upper[-1]) else float(prices[-1]),
                float(middle[-1]) if not np.isnan(middle[-1]) else float(prices[-1]),
                float(lower[-1]) if not np.isnan(lower[-1]) else float(prices[-1])
            )
        except:
            return float(prices[-1]), float(prices[-1]), float(prices[-1])
    
    @staticmethod
    def calculate_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
        """Calculate Average True Range"""
        if len(closes) < period:
            return 0.0
        
        try:
            atr = talib.ATR(highs, lows, closes, timeperiod=period)
            return float(atr[-1]) if not np.isnan(atr[-1]) else 0.0
        except:
            return 0.0

class HiddenMarkovRegimeDetector:
    """Hidden Markov Model for regime detection"""
    
    def __init__(self, n_regimes: int = 4):
        self.n_regimes = n_regimes
        self.model = hmm.GaussianHMM(
            n_components=n_regimes,
            covariance_type="full",
            n_iter=1000,
            random_state=42
        )
        self.is_fitted = False
    
    def fit(self, features: np.ndarray) -> None:
        """Fit HMM to market features"""
        try:
            # Ensure features are properly scaled
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)
            
            # Fit the model
            self.model.fit(scaled_features)
            self.scaler = scaler
            self.is_fitted = True
            
            logger.info(f"HMM fitted with {self.n_regimes} regimes")
            
        except Exception as e:
            logger.error(f"Error fitting HMM: {e}")
            self.is_fitted = False
    
    def predict_regime(self, features: np.ndarray) -> Tuple[int, np.ndarray]:
        """Predict current regime and probabilities"""
        if not self.is_fitted:
            return 0, np.ones(self.n_regimes) / self.n_regimes
        
        try:
            scaled_features = self.scaler.transform(features.reshape(1, -1))
            regime = self.model.predict(scaled_features)[0]
            probabilities = self.model.predict_proba(scaled_features)[0]
            
            return int(regime), probabilities
            
        except Exception as e:
            logger.error(f"Error predicting regime: {e}")
            return 0, np.ones(self.n_regimes) / self.n_regimes

class MarketRegimeDetector:
    """Advanced market regime detection system"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.hmm_detector = HiddenMarkovRegimeDetector()
        self.feature_importance = {}
        self.regime_history = []
        self.is_trained = False
        
        # Model configurations
        self.model_configs = {
            'random_forest': {
                'model': RandomForestClassifier(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42
                ),
                'weight': 0.4
            },
            'gradient_boost': {
                'model': GradientBoostingClassifier(
                    n_estimators=150,
                    max_depth=8,
                    learning_rate=0.1,
                    random_state=42
                ),
                'weight': 0.4
            },
            'hmm': {
                'weight': 0.2
            }
        }
    
    def extract_features(self, market_data: Dict[str, Any]) -> MarketFeatures:
        """Extract comprehensive market features"""
        try:
            prices = np.array(market_data.get('prices', []))
            volumes = np.array(market_data.get('volumes', []))
            timestamps = market_data.get('timestamps', [])
            
            if len(prices) < 50:  # Minimum data requirement
                return self._default_features()
            
            # Calculate returns for different timeframes
            returns = np.diff(np.log(prices))
            returns_1h = float(returns[-1]) if len(returns) > 0 else 0.0
            returns_4h = float(np.mean(returns[-4:])) if len(returns) >= 4 else 0.0
            returns_24h = float(np.mean(returns[-24:])) if len(returns) >= 24 else 0.0
            returns_7d = float(np.mean(returns[-168:])) if len(returns) >= 168 else 0.0
            
            # Calculate volatilities
            volatility_1h = float(np.std(returns[-1:]) * np.sqrt(24)) if len(returns) > 0 else 0.0
            volatility_24h = float(np.std(returns[-24:]) * np.sqrt(24)) if len(returns) >= 24 else 0.0
            volatility_7d = float(np.std(returns[-168:]) * np.sqrt(24)) if len(returns) >= 168 else 0.0
            
            # Technical indicators
            rsi_14 = TechnicalIndicators.calculate_rsi(prices)
            macd, macd_signal, _ = TechnicalIndicators.calculate_macd(prices)
            
            # Bollinger Bands position
            bb_upper, bb_middle, bb_lower = TechnicalIndicators.calculate_bollinger_bands(prices)
            bb_position = (prices[-1] - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
            
            # ATR ratio
            highs = prices  # Simplified - in production would use actual OHLC
            lows = prices
            atr = TechnicalIndicators.calculate_atr(highs, lows, prices)
            atr_ratio = atr / prices[-1] if prices[-1] > 0 else 0.0
            
            # Volume analysis
            volume_ratio = float(volumes[-1] / np.mean(volumes[-24:])) if len(volumes) >= 24 and np.mean(volumes[-24:]) > 0 else 1.0
            
            # Advanced features
            trend_strength = self._calculate_trend_strength(prices)
            support_resistance_ratio = self._calculate_support_resistance_ratio(prices)
            price_momentum = self._calculate_momentum(prices)
            volume_momentum = self._calculate_momentum(volumes) if len(volumes) > 10 else 0.0
            
            # Regime-specific features
            regime_persistence = self._calculate_regime_persistence()
            regime_transition_prob = self._calculate_transition_probability()
            market_stress_index = self._calculate_market_stress(returns, volumes)
            liquidity_depth_ratio = self._calculate_liquidity_depth(market_data)
            
            return MarketFeatures(
                returns_1h=returns_1h,
                returns_4h=returns_4h,
                returns_24h=returns_24h,
                returns_7d=returns_7d,
                volatility_1h=volatility_1h,
                volatility_24h=volatility_24h,
                volatility_7d=volatility_7d,
                rsi_14=rsi_14,
                macd_signal=macd_signal,
                bb_position=bb_position,
                atr_ratio=atr_ratio,
                volume_ratio=volume_ratio,
                trend_strength=trend_strength,
                support_resistance_ratio=support_resistance_ratio,
                price_momentum=price_momentum,
                volume_momentum=volume_momentum,
                regime_persistence=regime_persistence,
                regime_transition_prob=regime_transition_prob,
                market_stress_index=market_stress_index,
                liquidity_depth_ratio=liquidity_depth_ratio
            )
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return self._default_features()
    
    def train(self, historical_data: List[Dict[str, Any]], labels: List[MarketRegime]) -> Dict[str, float]:
        """Train all regime detection models"""
        try:
            # Extract features for all samples
            features_list = []
            for data in historical_data:
                features = self.extract_features(data)
                features_array = self._features_to_array(features)
                features_list.append(features_array)
            
            X = np.array(features_list)
            y = np.array([regime.value for regime in labels])
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            self.scalers['main'] = scaler
            
            # Train ensemble models
            model_scores = {}
            
            for name, config in self.model_configs.items():
                if name == 'hmm':
                    # Train HMM separately
                    self.hmm_detector.fit(X_train_scaled)
                    continue
                
                model = config['model']
                model.fit(X_train_scaled, y_train)
                
                # Evaluate
                train_score = model.score(X_train_scaled, y_train)
                test_score = model.score(X_test_scaled, y_test)
                
                self.models[name] = model
                model_scores[name] = {
                    'train_score': train_score,
                    'test_score': test_score
                }
                
                # Feature importance
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[name] = model.feature_importances_
            
            self.is_trained = True
            logger.info("Market regime detection models trained successfully")
            
            return model_scores
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
            return {}
    
    def detect_regime(self, market_data: Dict[str, Any]) -> RegimeDetection:
        """Detect current market regime using ensemble approach"""
        if not self.is_trained:
            return RegimeDetection(
                regime=MarketRegime.UNKNOWN,
                confidence=0.0,
                probability_distribution={regime: 1.0/len(MarketRegime) for regime in MarketRegime},
                features={},
                timestamp=datetime.now(),
                reasoning="Models not trained"
            )
        
        try:
            # Extract features
            features = self.extract_features(market_data)
            features_array = self._features_to_array(features)
            features_scaled = self.scalers['main'].transform(features_array.reshape(1, -1))
            
            # Get predictions from all models
            predictions = {}
            probabilities = {}
            
            for name, model in self.models.items():
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(features_scaled)[0]
                    probabilities[name] = proba
                    predictions[name] = np.argmax(proba)
                else:
                    pred = model.predict(features_scaled)[0]
                    predictions[name] = pred
                    # Convert to probability distribution
                    proba = np.zeros(len(MarketRegime))
                    proba[pred] = 1.0
                    probabilities[name] = proba
            
            # HMM prediction
            hmm_regime, hmm_proba = self.hmm_detector.predict_regime(features_array)
            predictions['hmm'] = hmm_regime
            probabilities['hmm'] = hmm_proba
            
            # Ensemble prediction using weighted voting
            ensemble_proba = np.zeros(len(MarketRegime))
            total_weight = 0
            
            for name, proba in probabilities.items():
                weight = self.model_configs.get(name, {}).get('weight', 1.0)
                if len(proba) >= len(MarketRegime):
                    ensemble_proba += weight * proba[:len(MarketRegime)]
                total_weight += weight
            
            ensemble_proba /= total_weight
            predicted_regime = MarketRegime(np.argmax(ensemble_proba))
            confidence = float(np.max(ensemble_proba))
            
            # Create probability distribution
            prob_dist = {
                regime: float(ensemble_proba[regime.value]) 
                for regime in MarketRegime
            }
            
            # Generate reasoning
            reasoning = self._generate_reasoning(features, predictions, confidence)
            
            # Update regime history
            detection = RegimeDetection(
                regime=predicted_regime,
                confidence=confidence,
                probability_distribution=prob_dist,
                features=self._features_to_dict(features),
                timestamp=datetime.now(),
                reasoning=reasoning
            )
            
            self.regime_history.append(detection)
            if len(self.regime_history) > 100:  # Keep last 100 detections
                self.regime_history.pop(0)
            
            return detection
            
        except Exception as e:
            logger.error(f"Error detecting regime: {e}")
            return RegimeDetection(
                regime=MarketRegime.UNKNOWN,
                confidence=0.0,
                probability_distribution={regime: 1.0/len(MarketRegime) for regime in MarketRegime},
                features={},
                timestamp=datetime.now(),
                reasoning=f"Error: {str(e)}"
            )
    
    def get_regime_transitions(self) -> Dict[str, Any]:
        """Analyze regime transition patterns"""
        if len(self.regime_history) < 2:
            return {}
        
        transitions = {}
        for i in range(1, len(self.regime_history)):
            prev_regime = self.regime_history[i-1].regime
            curr_regime = self.regime_history[i].regime
            
            if prev_regime != curr_regime:
                transition = f"{prev_regime.name}->{curr_regime.name}"
                transitions[transition] = transitions.get(transition, 0) + 1
        
        return transitions
    
    # Helper methods
    def _default_features(self) -> MarketFeatures:
        """Return default features when data is insufficient"""
        return MarketFeatures(
            returns_1h=0.0, returns_4h=0.0, returns_24h=0.0, returns_7d=0.0,
            volatility_1h=0.0, volatility_24h=0.0, volatility_7d=0.0,
            rsi_14=50.0, macd_signal=0.0, bb_position=0.5, atr_ratio=0.0,
            volume_ratio=1.0, trend_strength=0.0, support_resistance_ratio=0.5,
            price_momentum=0.0, volume_momentum=0.0, regime_persistence=0.5,
            regime_transition_prob=0.5, market_stress_index=0.0, liquidity_depth_ratio=1.0
        )
    
    def _features_to_array(self, features: MarketFeatures) -> np.ndarray:
        """Convert features to numpy array"""
        return np.array([
            features.returns_1h, features.returns_4h, features.returns_24h, features.returns_7d,
            features.volatility_1h, features.volatility_24h, features.volatility_7d,
            features.rsi_14, features.macd_signal, features.bb_position, features.atr_ratio,
            features.volume_ratio, features.trend_strength, features.support_resistance_ratio,
            features.price_momentum, features.volume_momentum, features.regime_persistence,
            features.regime_transition_prob, features.market_stress_index, features.liquidity_depth_ratio
        ])
    
    def _features_to_dict(self, features: MarketFeatures) -> Dict[str, float]:
        """Convert features to dictionary"""
        return {
            'returns_1h': features.returns_1h,
            'returns_4h': features.returns_4h,
            'returns_24h': features.returns_24h,
            'returns_7d': features.returns_7d,
            'volatility_1h': features.volatility_1h,
            'volatility_24h': features.volatility_24h,
            'volatility_7d': features.volatility_7d,
            'rsi_14': features.rsi_14,
            'macd_signal': features.macd_signal,
            'bb_position': features.bb_position,
            'atr_ratio': features.atr_ratio,
            'volume_ratio': features.volume_ratio,
            'trend_strength': features.trend_strength,
            'support_resistance_ratio': features.support_resistance_ratio,
            'price_momentum': features.price_momentum,
            'volume_momentum': features.volume_momentum,
            'regime_persistence': features.regime_persistence,
            'regime_transition_prob': features.regime_transition_prob,
            'market_stress_index': features.market_stress_index,
            'liquidity_depth_ratio': features.liquidity_depth_ratio
        }
    
    def _calculate_trend_strength(self, prices: np.ndarray) -> float:
        """Calculate trend strength using linear regression"""
        if len(prices) < 10:
            return 0.0
        
        x = np.arange(len(prices))
        slope, intercept, r_value, _, _ = stats.linregress(x, prices)
        return float(r_value ** 2)  # R-squared as trend strength
    
    def _calculate_support_resistance_ratio(self, prices: np.ndarray) -> float:
        """Calculate support/resistance strength ratio"""
        if len(prices) < 20:
            return 0.5
        
        # Simplified support/resistance calculation
        recent_high = np.max(prices[-20:])
        recent_low = np.min(prices[-20:])
        current_price = prices[-1]
        
        if recent_high == recent_low:
            return 0.5
        
        return (current_price - recent_low) / (recent_high - recent_low)
    
    def _calculate_momentum(self, data: np.ndarray) -> float:
        """Calculate momentum using rate of change"""
        if len(data) < 10:
            return 0.0
        
        return float((data[-1] - data[-10]) / data[-10] if data[-10] != 0 else 0.0)
    
    def _calculate_regime_persistence(self) -> float:
        """Calculate how persistent the current regime has been"""
        if len(self.regime_history) < 5:
            return 0.5
        
        current_regime = self.regime_history[-1].regime
        same_regime_count = 0
        
        for detection in reversed(self.regime_history[-5:]):
            if detection.regime == current_regime:
                same_regime_count += 1
            else:
                break
        
        return same_regime_count / 5.0
    
    def _calculate_transition_probability(self) -> float:
        """Calculate probability of regime transition"""
        if len(self.regime_history) < 10:
            return 0.5
        
        transitions = 0
        for i in range(1, min(10, len(self.regime_history))):
            if self.regime_history[-i].regime != self.regime_history[-i-1].regime:
                transitions += 1
        
        return transitions / 9.0  # 9 possible transitions in last 10 observations
    
    def _calculate_market_stress(self, returns: np.ndarray, volumes: np.ndarray) -> float:
        """Calculate market stress index"""
        if len(returns) < 24 or len(volumes) < 24:
            return 0.0
        
        # Combine volatility and volume spikes
        vol_stress = np.std(returns[-24:]) / np.std(returns) if len(returns) > 24 else 1.0
        volume_stress = np.std(volumes[-24:]) / np.std(volumes) if len(volumes) > 24 else 1.0
        
        return min(1.0, (vol_stress + volume_stress) / 2.0)
    
    def _calculate_liquidity_depth(self, market_data: Dict[str, Any]) -> float:
        """Calculate liquidity depth ratio"""
        # Simplified - would use order book data in production
        volumes = market_data.get('volumes', [])
        if len(volumes) < 24:
            return 1.0
        
        recent_volume = np.mean(volumes[-24:])
        historical_volume = np.mean(volumes[:-24]) if len(volumes) > 24 else recent_volume
        
        return recent_volume / historical_volume if historical_volume > 0 else 1.0
    
    def _generate_reasoning(self, features: MarketFeatures, predictions: Dict[str, int], confidence: float) -> str:
        """Generate human-readable reasoning for the regime detection"""
        reasoning_parts = []
        
        # Volatility analysis
        if features.volatility_24h > 0.05:  # 5% daily volatility
            reasoning_parts.append("High volatility detected")
        elif features.volatility_24h < 0.01:
            reasoning_parts.append("Low volatility environment")
        
        # Trend analysis
        if features.trend_strength > 0.8:
            if features.price_momentum > 0:
                reasoning_parts.append("Strong upward trend")
            else:
                reasoning_parts.append("Strong downward trend")
        elif features.trend_strength < 0.3:
            reasoning_parts.append("Sideways market movement")
        
        # Technical indicators
        if features.rsi_14 > 70:
            reasoning_parts.append("Overbought conditions (RSI > 70)")
        elif features.rsi_14 < 30:
            reasoning_parts.append("Oversold conditions (RSI < 30)")
        
        # Volume analysis
        if features.volume_ratio > 2.0:
            reasoning_parts.append("Abnormally high volume")
        elif features.volume_ratio < 0.5:
            reasoning_parts.append("Low volume activity")
        
        # Confidence assessment
        if confidence > 0.8:
            reasoning_parts.append("High confidence prediction")
        elif confidence < 0.6:
            reasoning_parts.append("Moderate confidence - conflicting signals")
        
        return "; ".join(reasoning_parts) if reasoning_parts else "Normal market conditions"

# Singleton instance
_detector = None

def get_market_regime_detector() -> MarketRegimeDetector:
    """Get singleton market regime detector instance"""
    global _detector
    if _detector is None:
        _detector = MarketRegimeDetector()
    return _detector