"""
Online Learning Engine with River ML
Continuous learning system for real-time DeFi strategy optimization
"""

import asyncio
import logging
import json
import time
import pickle
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
import os

# River ML imports for online learning
from river import linear_model, ensemble, tree, naive_bayes, preprocessing, feature_extraction
from river import metrics, evaluate, compose, stats, anomaly
from river.drift import ADWIN
import river

from ..dexbrain.config import Config

logger = logging.getLogger(__name__)

@dataclass
class OnlineLearningMetrics:
    """Metrics for online learning performance"""
    samples_processed: int = 0
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    concept_drifts_detected: int = 0
    model_updates: int = 0
    prediction_confidence: float = 0.0
    learning_rate: float = 0.01

@dataclass
class StrategyPrediction:
    """Online ML strategy prediction"""
    pool_address: str
    timestamp: int
    strategy_action: str  # 'hold', 'compound', 'rebalance', 'widen_range', 'narrow_range'
    confidence: float
    expected_return: float
    risk_score: float
    volatility_regime: str  # 'low', 'medium', 'high'
    market_trend: str  # 'bullish', 'bearish', 'sideways'
    optimal_fee_tier: Optional[int] = None
    recommended_range: Optional[Tuple[int, int]] = None
    model_version: str = "online_v1.0"

class OnlineDeFiOptimizer:
    """
    Online learning optimizer for DeFi strategies using River ML
    Continuously adapts to market conditions with real-time learning
    """
    
    def __init__(self):
        # Core online learning models
        self.strategy_classifier = self._create_strategy_classifier()
        self.return_regressor = self._create_return_regressor()
        self.volatility_detector = self._create_volatility_detector()
        self.anomaly_detector = self._create_anomaly_detector()
        
        # Concept drift detection
        self.drift_detector = ADWIN()
        
        # Feature preprocessing pipeline
        self.feature_pipeline = self._create_feature_pipeline()
        
        # Performance metrics
        self.metrics = {
            'strategy_accuracy': metrics.Accuracy(),
            'strategy_precision': metrics.Precision(),
            'strategy_recall': metrics.Recall(),
            'strategy_f1': metrics.F1(),
            'return_mae': metrics.MAE(),
            'return_rmse': metrics.RMSE(),
            'volatility_accuracy': metrics.Accuracy()
        }
        
        # Learning state
        self.samples_processed = 0
        self.concept_drifts = 0
        self.model_updates = 0
        
        # Feature importance tracking
        self.feature_importance = {}
        self.feature_statistics = {}
        
        # Model persistence
        self.model_save_interval = 1000  # Save every 1000 samples
        self.model_save_path = "/opt/dexter-ai/online_models/"
        
        logger.info("Online DeFi optimizer initialized with River ML")
    
    def _create_strategy_classifier(self):
        """Create online strategy classification model"""
        # Ensemble of online classifiers
        model = ensemble.VotingClassifier([
            ('adaptive_tree', tree.HoeffdingAdaptiveTreeClassifier(
                grace_period=200,
                split_confidence=0.0001,
                leaf_prediction='mc',
                bootstrap_sampling=True
            )),
            ('naive_bayes', naive_bayes.GaussianNB()),
            ('logistic', linear_model.LogisticRegression(
                optimizer=linear_model.optimizers.SGD(lr=0.01),
                l2=0.001
            )),
            ('perceptron', linear_model.Perceptron(
                l2=0.001,
                clip_gradient=1000
            ))
        ])
        
        return model
    
    def _create_return_regressor(self):
        """Create online return prediction model"""
        # Adaptive regression ensemble
        model = ensemble.AdaptiveRandomForestRegressor(
            n_models=10,
            max_depth=10,
            lambda_value=6,
            seed=42
        )
        
        return model
    
    def _create_volatility_detector(self):
        """Create volatility regime detection model"""
        model = tree.HoeffdingAdaptiveTreeClassifier(
            grace_period=100,
            split_confidence=0.001,
            leaf_prediction='mc'
        )
        
        return model
    
    def _create_anomaly_detector(self):
        """Create online anomaly detection model"""
        model = anomaly.HalfSpaceTrees(
            n_trees=25,
            height=10,
            window_size=250
        )
        
        return model
    
    def _create_feature_pipeline(self):
        """Create feature preprocessing pipeline"""
        pipeline = compose.Pipeline(
            ('scale', preprocessing.StandardScaler()),
            ('poly', preprocessing.PolynomialFeatures(degree=2, include_bias=False)),
            ('select', feature_extraction.RBFSampler(n_components=100, gamma=0.1))
        )
        
        return pipeline
    
    async def learn_from_stream(self, features: Dict[str, float], 
                               true_strategy: str, true_return: float,
                               true_volatility: str) -> Dict[str, Any]:
        """Learn from new streaming data sample"""
        try:
            start_time = time.time()
            
            # Preprocess features
            processed_features = self.feature_pipeline.transform_one(features)
            
            # Update strategy classifier
            strategy_pred = self.strategy_classifier.predict_one(processed_features)
            self.strategy_classifier.learn_one(processed_features, true_strategy)
            
            # Update return regressor
            return_pred = self.return_regressor.predict_one(processed_features)
            self.return_regressor.learn_one(processed_features, true_return)
            
            # Update volatility detector
            volatility_pred = self.volatility_detector.predict_one(processed_features)
            self.volatility_detector.learn_one(processed_features, true_volatility)
            
            # Update anomaly detector
            anomaly_score = self.anomaly_detector.score_one(processed_features)
            self.anomaly_detector.learn_one(processed_features)
            
            # Check for concept drift
            drift_detected = False
            strategy_correct = (strategy_pred == true_strategy) if strategy_pred else False
            self.drift_detector.update(1 if strategy_correct else 0)
            
            if self.drift_detector.drift_detected:
                drift_detected = True
                self.concept_drifts += 1
                await self._handle_concept_drift()
            
            # Update metrics
            if strategy_pred:
                self.metrics['strategy_accuracy'].update(true_strategy, strategy_pred)
                self.metrics['strategy_precision'].update(true_strategy, strategy_pred)
                self.metrics['strategy_recall'].update(true_strategy, strategy_pred)
                self.metrics['strategy_f1'].update(true_strategy, strategy_pred)
            
            if return_pred is not None:
                self.metrics['return_mae'].update(true_return, return_pred)
                self.metrics['return_rmse'].update(true_return, return_pred)
            
            if volatility_pred:
                self.metrics['volatility_accuracy'].update(true_volatility, volatility_pred)
            
            # Update counters
            self.samples_processed += 1
            self.model_updates += 1
            
            # Update feature importance (simplified)
            await self._update_feature_importance(features, true_strategy)
            
            # Periodic model saving
            if self.samples_processed % self.model_save_interval == 0:
                await self._save_models()
            
            processing_time = time.time() - start_time
            
            learning_result = {
                'samples_processed': self.samples_processed,
                'strategy_prediction': strategy_pred,
                'return_prediction': return_pred,
                'volatility_prediction': volatility_pred,
                'anomaly_score': anomaly_score,
                'drift_detected': drift_detected,
                'processing_time': processing_time,
                'accuracy': self.metrics['strategy_accuracy'].get(),
                'return_mae': self.metrics['return_mae'].get(),
                'concept_drifts': self.concept_drifts
            }
            
            logger.debug(f"Online learning update: accuracy={learning_result['accuracy']:.3f}, "
                        f"return_mae={learning_result['return_mae']:.3f}, "
                        f"samples={self.samples_processed}")
            
            return learning_result
            
        except Exception as e:
            logger.error(f"Online learning error: {e}")
            return {'error': str(e)}
    
    async def predict_strategy(self, features: Dict[str, float]) -> StrategyPrediction:
        """Generate strategy prediction for given features"""
        try:
            # Preprocess features
            processed_features = self.feature_pipeline.transform_one(features)
            
            # Get predictions from all models
            strategy_pred = self.strategy_classifier.predict_one(processed_features)
            return_pred = self.return_regressor.predict_one(processed_features)
            volatility_pred = self.volatility_detector.predict_one(processed_features)
            anomaly_score = self.anomaly_detector.score_one(processed_features)
            
            # Calculate prediction confidence
            confidence = await self._calculate_prediction_confidence(
                processed_features, strategy_pred, return_pred, volatility_pred
            )
            
            # Determine market trend
            market_trend = self._analyze_market_trend(features)
            
            # Generate optimal parameters
            optimal_fee, recommended_range = self._calculate_optimal_parameters(
                features, strategy_pred, volatility_pred
            )
            
            # Calculate risk score
            risk_score = self._calculate_risk_score(
                anomaly_score, volatility_pred, return_pred
            )
            
            prediction = StrategyPrediction(
                pool_address=features.get('pool_address', ''),
                timestamp=int(time.time()),
                strategy_action=strategy_pred or 'hold',
                confidence=confidence,
                expected_return=return_pred or 0.0,
                risk_score=risk_score,
                volatility_regime=volatility_pred or 'medium',
                market_trend=market_trend,
                optimal_fee_tier=optimal_fee,
                recommended_range=recommended_range
            )
            
            return prediction
            
        except Exception as e:
            logger.error(f"Strategy prediction error: {e}")
            # Return safe default prediction
            return StrategyPrediction(
                pool_address=features.get('pool_address', ''),
                timestamp=int(time.time()),
                strategy_action='hold',
                confidence=0.5,
                expected_return=0.0,
                risk_score=0.5,
                volatility_regime='medium',
                market_trend='sideways'
            )
    
    async def _calculate_prediction_confidence(self, features: Dict, 
                                             strategy_pred: str, 
                                             return_pred: float,
                                             volatility_pred: str) -> float:
        """Calculate prediction confidence based on model agreement"""
        try:
            # Base confidence from model accuracy
            base_confidence = self.metrics['strategy_accuracy'].get()
            
            # Adjust based on return prediction confidence
            return_error = self.metrics['return_mae'].get()
            return_confidence = max(0, 1 - return_error)
            
            # Adjust based on volatility prediction confidence
            volatility_confidence = self.metrics['volatility_accuracy'].get()
            
            # Weighted average confidence
            confidence = (
                base_confidence * 0.5 +
                return_confidence * 0.3 +
                volatility_confidence * 0.2
            )
            
            # Apply uncertainty penalty for drift
            if self.concept_drifts > 0:
                drift_penalty = min(0.1, self.concept_drifts / 100)
                confidence -= drift_penalty
            
            return max(0.1, min(1.0, confidence))
            
        except Exception:
            return 0.5  # Default moderate confidence
    
    def _analyze_market_trend(self, features: Dict[str, float]) -> str:
        """Analyze market trend from features"""
        try:
            price_change_1h = features.get('price_change_1h', 0)
            price_change_24h = features.get('price_change_24h', 0)
            volume_ratio = features.get('volume_1h', 0) / max(features.get('volume_24h', 1), 1)
            
            # Simple trend analysis
            if price_change_1h > 0.02 and price_change_24h > 0.05 and volume_ratio > 1.2:
                return 'bullish'
            elif price_change_1h < -0.02 and price_change_24h < -0.05 and volume_ratio > 1.2:
                return 'bearish'
            else:
                return 'sideways'
                
        except Exception:
            return 'sideways'
    
    def _calculate_optimal_parameters(self, features: Dict[str, float], 
                                    strategy: str, volatility: str) -> Tuple[Optional[int], Optional[Tuple[int, int]]]:
        """Calculate optimal fee tier and range based on predictions"""
        try:
            optimal_fee = None
            recommended_range = None
            
            current_volatility = features.get('price_volatility_24h', 0.1)
            
            # Fee tier optimization
            if current_volatility < 0.05:  # Low volatility
                optimal_fee = 500  # 0.05%
            elif current_volatility < 0.15:  # Medium volatility
                optimal_fee = 3000  # 0.3%
            else:  # High volatility
                optimal_fee = 10000  # 1%
            
            # Range optimization
            if strategy in ['widen_range', 'narrow_range', 'rebalance']:
                current_tick = int(features.get('current_tick', 0))
                tick_spacing = int(features.get('tick_spacing', 60))
                
                if volatility == 'low':
                    range_width = tick_spacing * 20  # Narrow range
                elif volatility == 'medium':
                    range_width = tick_spacing * 50  # Medium range
                else:
                    range_width = tick_spacing * 100  # Wide range
                
                recommended_range = (
                    current_tick - range_width // 2,
                    current_tick + range_width // 2
                )
            
            return optimal_fee, recommended_range
            
        except Exception:
            return None, None
    
    def _calculate_risk_score(self, anomaly_score: float, 
                            volatility: str, return_pred: float) -> float:
        """Calculate risk score from various factors"""
        try:
            # Base risk from anomaly score
            risk = anomaly_score
            
            # Adjust for volatility
            volatility_risk = {
                'low': 0.1,
                'medium': 0.3,
                'high': 0.7
            }.get(volatility, 0.3)
            
            # Adjust for expected return (higher returns often mean higher risk)
            return_risk = min(abs(return_pred) / 0.5, 0.5)  # Cap at 50% return
            
            # Weighted combination
            total_risk = (
                risk * 0.4 +
                volatility_risk * 0.4 +
                return_risk * 0.2
            )
            
            return max(0.0, min(1.0, total_risk))
            
        except Exception:
            return 0.5  # Default moderate risk
    
    async def _handle_concept_drift(self):
        """Handle detected concept drift"""
        logger.warning(f"Concept drift detected! Total drifts: {self.concept_drifts}")
        
        # Reset drift detector
        self.drift_detector = ADWIN()
        
        # Optionally reinitialize models with lower learning rates
        # or ensemble with new models
        
        # Log drift event for analysis
        drift_event = {
            'timestamp': datetime.now().isoformat(),
            'drift_count': self.concept_drifts,
            'samples_processed': self.samples_processed,
            'current_accuracy': self.metrics['strategy_accuracy'].get()
        }
        
        # Could trigger model retraining or ensemble updates here
    
    async def _update_feature_importance(self, features: Dict[str, float], target: str):
        """Update feature importance tracking"""
        try:
            # Simple frequency-based importance
            for feature_name, value in features.items():
                if feature_name not in self.feature_importance:
                    self.feature_importance[feature_name] = 0
                    self.feature_statistics[feature_name] = stats.Mean()
                
                # Update statistics
                self.feature_statistics[feature_name].update(value)
                
                # Simple importance based on usage frequency
                self.feature_importance[feature_name] += 1
                
        except Exception as e:
            logger.error(f"Feature importance update error: {e}")
    
    async def _save_models(self):
        """Save models to disk"""
        try:
            os.makedirs(self.model_save_path, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save models using pickle (River models are pickleable)
            models_to_save = {
                'strategy_classifier': self.strategy_classifier,
                'return_regressor': self.return_regressor,
                'volatility_detector': self.volatility_detector,
                'anomaly_detector': self.anomaly_detector,
                'feature_pipeline': self.feature_pipeline
            }
            
            for model_name, model in models_to_save.items():
                file_path = f"{self.model_save_path}/{model_name}_{timestamp}.pkl"
                with open(file_path, 'wb') as f:
                    pickle.dump(model, f)
            
            # Save metadata
            metadata = {
                'timestamp': timestamp,
                'samples_processed': self.samples_processed,
                'concept_drifts': self.concept_drifts,
                'metrics': {name: metric.get() for name, metric in self.metrics.items()},
                'feature_importance': self.feature_importance
            }
            
            metadata_path = f"{self.model_save_path}/metadata_{timestamp}.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Models saved to {self.model_save_path} at {timestamp}")
            
        except Exception as e:
            logger.error(f"Model saving error: {e}")
    
    async def load_models(self, timestamp: str = None):
        """Load models from disk"""
        try:
            if not timestamp:
                # Find latest timestamp
                files = os.listdir(self.model_save_path)
                timestamps = [f.split('_')[-1].split('.')[0] for f in files if 'metadata_' in f]
                timestamp = max(timestamps) if timestamps else None
            
            if not timestamp:
                logger.warning("No saved models found")
                return
            
            # Load models
            model_files = {
                'strategy_classifier': f"{self.model_save_path}/strategy_classifier_{timestamp}.pkl",
                'return_regressor': f"{self.model_save_path}/return_regressor_{timestamp}.pkl",
                'volatility_detector': f"{self.model_save_path}/volatility_detector_{timestamp}.pkl",
                'anomaly_detector': f"{self.model_save_path}/anomaly_detector_{timestamp}.pkl",
                'feature_pipeline': f"{self.model_save_path}/feature_pipeline_{timestamp}.pkl"
            }
            
            for model_name, file_path in model_files.items():
                if os.path.exists(file_path):
                    with open(file_path, 'rb') as f:
                        setattr(self, model_name, pickle.load(f))
            
            # Load metadata
            metadata_path = f"{self.model_save_path}/metadata_{timestamp}.json"
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                self.samples_processed = metadata.get('samples_processed', 0)
                self.concept_drifts = metadata.get('concept_drifts', 0)
                self.feature_importance = metadata.get('feature_importance', {})
            
            logger.info(f"Models loaded from timestamp {timestamp}")
            
        except Exception as e:
            logger.error(f"Model loading error: {e}")
    
    def get_learning_metrics(self) -> OnlineLearningMetrics:
        """Get current learning performance metrics"""
        return OnlineLearningMetrics(
            samples_processed=self.samples_processed,
            accuracy=self.metrics['strategy_accuracy'].get(),
            precision=self.metrics['strategy_precision'].get(),
            recall=self.metrics['strategy_recall'].get(),
            f1_score=self.metrics['strategy_f1'].get(),
            concept_drifts_detected=self.concept_drifts,
            model_updates=self.model_updates,
            prediction_confidence=0.8  # Would calculate dynamically
        )
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance rankings"""
        if not self.feature_importance:
            return {}
        
        # Normalize importance scores
        total_importance = sum(self.feature_importance.values())
        normalized_importance = {
            feature: importance / total_importance
            for feature, importance in self.feature_importance.items()
        }
        
        # Sort by importance
        return dict(sorted(normalized_importance.items(), 
                          key=lambda x: x[1], reverse=True))

# Factory function
async def create_online_optimizer() -> OnlineDeFiOptimizer:
    """Create online learning optimizer"""
    optimizer = OnlineDeFiOptimizer()
    
    # Try to load existing models
    await optimizer.load_models()
    
    return optimizer

# Example usage and testing
async def test_online_learning():
    """Test the online learning engine"""
    optimizer = await create_online_optimizer()
    
    # Simulate streaming data
    for i in range(100):
        # Generate synthetic features
        features = {
            'pool_address': '0x1234567890123456789012345678901234567890',
            'price': 3000 + np.random.normal(0, 100),
            'price_change_1h': np.random.normal(0, 0.05),
            'price_change_24h': np.random.normal(0, 0.15),
            'volume_1h': np.random.exponential(100000),
            'volume_24h': np.random.exponential(1000000),
            'price_volatility_24h': np.random.exponential(0.1),
            'liquidity': np.random.exponential(5000000),
            'current_tick': 200000 + np.random.randint(-1000, 1000),
            'tick_spacing': 60
        }
        
        # Generate synthetic ground truth
        true_strategy = np.random.choice(['hold', 'compound', 'rebalance', 'widen_range', 'narrow_range'])
        true_return = np.random.normal(0.05, 0.1)
        true_volatility = np.random.choice(['low', 'medium', 'high'])
        
        # Learn from sample
        result = await optimizer.learn_from_stream(features, true_strategy, true_return, true_volatility)
        
        if i % 20 == 0:
            print(f"Sample {i}: Accuracy = {result.get('accuracy', 0):.3f}, "
                  f"Drifts = {result.get('concept_drifts', 0)}")
    
    # Test prediction
    test_features = {
        'pool_address': '0x1234567890123456789012345678901234567890',
        'price': 3000,
        'price_change_1h': 0.02,
        'price_change_24h': 0.05,
        'volume_1h': 50000,
        'volume_24h': 500000,
        'price_volatility_24h': 0.1,
        'liquidity': 2000000,
        'current_tick': 200000,
        'tick_spacing': 60
    }
    
    prediction = await optimizer.predict_strategy(test_features)
    print(f"Strategy Prediction: {prediction.strategy_action} "
          f"(confidence: {prediction.confidence:.3f})")
    
    # Print metrics
    metrics = optimizer.get_learning_metrics()
    print(f"Learning Metrics: {asdict(metrics)}")
    
    # Print feature importance
    importance = optimizer.get_feature_importance()
    print(f"Top features: {list(importance.keys())[:5]}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_online_learning())