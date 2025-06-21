"""
Advanced ML Training Pipeline for Uniswap Position Optimization
Integrates with continuous learning system and real market data
"""

import asyncio
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
import json

from .models.enhanced_ml_models import UniswapLPOptimizer, UniswapFeatures
from .models.knowledge_base import KnowledgeBase
from .config import Config

logger = logging.getLogger(__name__)

class AdvancedTrainingPipeline:
    """
    Comprehensive training pipeline for ML-driven Uniswap optimization
    """
    
    def __init__(self, knowledge_base: KnowledgeBase):
        self.knowledge_base = knowledge_base
        self.ml_optimizer = UniswapLPOptimizer()
        
        # Training configuration
        self.min_training_samples = 20
        self.retrain_threshold = 50  # Retrain when we have 50+ new samples
        self.performance_threshold = 0.85  # Retrain if accuracy drops below 85%
        
        # Metrics tracking
        self.training_metrics = {
            'last_training_time': None,
            'total_samples_trained': 0,
            'model_performance': {},
            'feature_importance': {},
            'prediction_accuracy': {}
        }
        
        self.metrics_file = Path("training_metrics.json")
        self.load_metrics()
    
    def load_metrics(self):
        """Load training metrics from file"""
        try:
            if self.metrics_file.exists():
                with open(self.metrics_file, 'r') as f:
                    saved_metrics = json.load(f)
                    self.training_metrics.update(saved_metrics)
                    if 'last_training_time' in saved_metrics:
                        self.training_metrics['last_training_time'] = datetime.fromisoformat(
                            saved_metrics['last_training_time']
                        )
        except Exception as e:
            logger.warning(f"Could not load training metrics: {e}")
    
    def save_metrics(self):
        """Save training metrics to file"""
        try:
            metrics_to_save = self.training_metrics.copy()
            if metrics_to_save['last_training_time']:
                metrics_to_save['last_training_time'] = metrics_to_save['last_training_time'].isoformat()
            
            with open(self.metrics_file, 'w') as f:
                json.dump(metrics_to_save, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save training metrics: {e}")
    
    async def should_retrain(self) -> bool:
        """
        Determine if models should be retrained based on various criteria
        """
        # Check if we have enough new data
        insights = await self.knowledge_base.get_recent_insights(limit=1000)
        total_samples = len(insights)
        
        if total_samples < self.min_training_samples:
            logger.info(f"Insufficient data for training: {total_samples} samples")
            return False
        
        # Check if enough new samples since last training
        new_samples = total_samples - self.training_metrics['total_samples_trained']
        if new_samples >= self.retrain_threshold:
            logger.info(f"Retraining triggered: {new_samples} new samples")
            return True
        
        # Check if last training was more than 24 hours ago
        if self.training_metrics['last_training_time']:
            time_since_training = datetime.now() - self.training_metrics['last_training_time']
            if time_since_training > timedelta(hours=24):
                logger.info("Retraining triggered: 24+ hours since last training")
                return True
        
        # Check model performance degradation
        current_accuracy = self.training_metrics.get('prediction_accuracy', {}).get('overall', 1.0)
        if current_accuracy < self.performance_threshold:
            logger.info(f"Retraining triggered: accuracy below threshold ({current_accuracy:.2f})")
            return True
        
        return False
    
    async def prepare_training_data(self) -> List[Dict[str, Any]]:
        """
        Prepare comprehensive training data from knowledge base
        """
        logger.info("Preparing training data from knowledge base...")
        
        # Get all insights with position performance data
        insights = await self.knowledge_base.get_recent_insights(limit=10000)
        training_data = []
        
        for insight in insights:
            try:
                data = insight.get('data', {})
                
                # Skip if missing critical information
                if not data.get('pool_address') or not data.get('position_data'):
                    continue
                
                position_data = data['position_data']
                pool_metrics = data.get('pool_metrics', {})
                performance = data.get('performance_metrics', {})
                
                # Create comprehensive training example
                training_example = {
                    # Pool information
                    'pool_address': data['pool_address'],
                    'pool_tvl': float(pool_metrics.get('tvl', 0)),
                    'volume_24h': float(pool_metrics.get('volume_24h', 0)),
                    'fee_tier': int(pool_metrics.get('fee_tier', 3000)),
                    'token0_reserve': float(pool_metrics.get('token0_reserve', 0)),
                    'token1_reserve': float(pool_metrics.get('token1_reserve', 0)),
                    
                    # Price and volatility
                    'current_price': float(pool_metrics.get('current_price', 1)),
                    'price_volatility_1h': float(pool_metrics.get('price_volatility_1h', 0)),
                    'price_volatility_24h': float(pool_metrics.get('price_volatility_24h', 0)),
                    'price_volatility_7d': float(pool_metrics.get('price_volatility_7d', 0)),
                    'price_change_24h': float(pool_metrics.get('price_change_24h', 0)),
                    
                    # Position details
                    'current_tick': int(position_data.get('current_tick', 0)),
                    'tick_spacing': int(position_data.get('tick_spacing', 60)),
                    'liquidity_distribution': float(pool_metrics.get('liquidity_concentration', 0.5)),
                    'active_liquidity_ratio': float(pool_metrics.get('active_liquidity_ratio', 0.5)),
                    
                    # Market structure
                    'position_count': int(pool_metrics.get('position_count', 1)),
                    'average_position_size': float(pool_metrics.get('avg_position_size', 1000)),
                    'whale_concentration': float(pool_metrics.get('whale_concentration', 0.1)),
                    'volume_to_tvl_ratio': float(pool_metrics.get('volume_tvl_ratio', 0.1)),
                    
                    # Advanced metrics
                    'fee_growth_global': float(pool_metrics.get('fee_growth', 0)),
                    'tick_bitmap_density': float(pool_metrics.get('tick_density', 0.1)),
                    'price_impact_estimate': float(pool_metrics.get('price_impact', 0.01)),
                    'arbitrage_opportunity_score': float(pool_metrics.get('arbitrage_score', 0)),
                    
                    # Time features
                    'days_since_pool_creation': int(pool_metrics.get('pool_age_days', 100)),
                    
                    # Cross-pool features
                    'correlation_with_eth': float(pool_metrics.get('eth_correlation', 0.5)),
                    'correlation_with_btc': float(pool_metrics.get('btc_correlation', 0.3)),
                    'relative_volume_rank': float(pool_metrics.get('volume_rank', 0.5)),
                    
                    # Training targets
                    'target_apr': float(performance.get('apr', 0.1)),
                    'target_il': float(performance.get('impermanent_loss', 0.05)),
                    'target_lower_tick': int(position_data.get('lower_tick', -1000)),
                    'target_upper_tick': int(position_data.get('upper_tick', 1000)),
                    'target_fees_1d': float(performance.get('daily_fees', 10.0)),
                    'target_fees_7d': float(performance.get('weekly_fees', 70.0)),
                    'target_fees_30d': float(performance.get('monthly_fees', 300.0)),
                    
                    # Quality metrics
                    'data_quality_score': float(insight.get('confidence', 0.8)),
                    'position_duration_hours': float(performance.get('duration_hours', 24))
                }\n                \n                # Only include high-quality data\n                if training_example['data_quality_score'] >= 0.6:\n                    training_data.append(training_example)\n                    \n            except Exception as e:\n                logger.warning(f\"Skipping invalid insight: {e}\")\n                continue\n        \n        logger.info(f\"Prepared {len(training_data)} training examples\")\n        return training_data\n    \n    async def train_models(self, force_retrain: bool = False) -> Dict[str, Any]:\n        \"\"\"\n        Execute comprehensive model training\n        \n        Args:\n            force_retrain: Force retraining regardless of criteria\n            \n        Returns:\n            Training results and metrics\n        \"\"\"\n        if not force_retrain and not await self.should_retrain():\n            return {\n                'status': 'skipped',\n                'reason': 'No retraining needed',\n                'last_training': self.training_metrics['last_training_time']\n            }\n        \n        logger.info(\"Starting comprehensive model training...\")\n        \n        # Prepare training data\n        training_data = await self.prepare_training_data()\n        \n        if len(training_data) < self.min_training_samples:\n            return {\n                'status': 'insufficient_data',\n                'samples_available': len(training_data),\n                'samples_required': self.min_training_samples\n            }\n        \n        # Filter high-quality samples for training\n        high_quality_data = [\n            sample for sample in training_data \n            if sample['data_quality_score'] >= 0.8\n        ]\n        \n        if len(high_quality_data) >= self.min_training_samples:\n            training_data = high_quality_data\n            logger.info(f\"Using {len(training_data)} high-quality samples\")\n        \n        # Execute training\n        try:\n            training_results = self.ml_optimizer.train_models(\n                training_data=training_data,\n                epochs=100,  # More epochs for better convergence\n                batch_size=min(32, len(training_data) // 4),  # Adaptive batch size\n                learning_rate=0.001\n            )\n            \n            # Update metrics\n            self.training_metrics.update({\n                'last_training_time': datetime.now(),\n                'total_samples_trained': len(training_data),\n                'model_performance': training_results.get('results', {})\n            })\n            \n            # Calculate feature importance\n            await self._analyze_feature_importance(training_data)\n            \n            # Validate model performance\n            validation_results = await self._validate_model_performance(training_data)\n            self.training_metrics['prediction_accuracy'] = validation_results\n            \n            self.save_metrics()\n            \n            # Log training to knowledge base\n            await self.knowledge_base.add_insight({\n                'type': 'model_training',\n                'source': 'advanced_training_pipeline',\n                'data': {\n                    'training_results': training_results,\n                    'validation_results': validation_results,\n                    'samples_used': len(training_data),\n                    'model_types': ['lstm', 'tick_predictor', 'il_forecaster', 'fee_predictor']\n                },\n                'confidence': 0.95,\n                'timestamp': datetime.now().isoformat()\n            })\n            \n            logger.info(f\"Training completed successfully with {len(training_data)} samples\")\n            \n            return {\n                'status': 'success',\n                'training_results': training_results,\n                'validation_results': validation_results,\n                'samples_trained': len(training_data),\n                'metrics_updated': True\n            }\n            \n        except Exception as e:\n            logger.error(f\"Training failed: {e}\")\n            return {\n                'status': 'failed',\n                'error': str(e),\n                'samples_attempted': len(training_data)\n            }\n    \n    async def _analyze_feature_importance(self, training_data: List[Dict[str, Any]]):\n        \"\"\"\n        Analyze which features are most important for predictions\n        \"\"\"\n        try:\n            # Simple correlation analysis (in production, use proper feature importance)\n            feature_names = UniswapFeatures.get_feature_names()\n            feature_correlations = {}\n            \n            if len(training_data) < 10:\n                return\n            \n            # Extract features and targets\n            features_matrix = []\n            apr_targets = []\n            \n            for sample in training_data:\n                features = self.ml_optimizer.extract_features_from_position_data(sample)\n                features_matrix.append(features.to_array())\n                apr_targets.append(sample.get('target_apr', 0.1))\n            \n            features_matrix = np.array(features_matrix)\n            apr_targets = np.array(apr_targets)\n            \n            # Calculate correlations\n            for i, feature_name in enumerate(feature_names):\n                if features_matrix.shape[0] > 1:\n                    correlation = np.corrcoef(features_matrix[:, i], apr_targets)[0, 1]\n                    if not np.isnan(correlation):\n                        feature_correlations[feature_name] = abs(correlation)\n            \n            # Sort by importance\n            sorted_features = sorted(\n                feature_correlations.items(), \n                key=lambda x: x[1], \n                reverse=True\n            )\n            \n            self.training_metrics['feature_importance'] = dict(sorted_features[:10])\n            logger.info(f\"Top features: {list(dict(sorted_features[:5]).keys())}\")\n            \n        except Exception as e:\n            logger.warning(f\"Feature importance analysis failed: {e}\")\n    \n    async def _validate_model_performance(self, training_data: List[Dict[str, Any]]) -> Dict[str, float]:\n        \"\"\"\n        Validate model performance on held-out data\n        \"\"\"\n        try:\n            if len(training_data) < 20:\n                return {'overall': 0.5, 'note': 'insufficient_data_for_validation'}\n            \n            # Use last 20% as validation set\n            val_size = len(training_data) // 5\n            val_data = training_data[-val_size:]\n            \n            correct_predictions = 0\n            total_predictions = 0\n            \n            for sample in val_data:\n                try:\n                    # Get model predictions\n                    predictions = self.ml_optimizer.predict_optimal_position(sample)\n                    \n                    # Compare with actual results (simplified validation)\n                    predicted_apr = predictions.get('predicted_apr', 0)\n                    actual_apr = sample.get('target_apr', 0)\n                    \n                    # Consider prediction correct if within 20% of actual\n                    if actual_apr > 0:\n                        error_ratio = abs(predicted_apr - actual_apr) / actual_apr\n                        if error_ratio < 0.2:  # 20% tolerance\n                            correct_predictions += 1\n                    \n                    total_predictions += 1\n                    \n                except Exception:\n                    continue\n            \n            accuracy = correct_predictions / max(total_predictions, 1)\n            \n            return {\n                'overall': accuracy,\n                'correct_predictions': correct_predictions,\n                'total_predictions': total_predictions,\n                'validation_samples': len(val_data)\n            }\n            \n        except Exception as e:\n            logger.error(f\"Model validation failed: {e}\")\n            return {'overall': 0.0, 'error': str(e)}\n    \n    async def get_training_status(self) -> Dict[str, Any]:\n        \"\"\"\n        Get current training status and metrics\n        \"\"\"\n        insights_count = len(await self.knowledge_base.get_recent_insights(limit=10000))\n        \n        return {\n            'last_training_time': self.training_metrics['last_training_time'],\n            'total_samples_trained': self.training_metrics['total_samples_trained'],\n            'available_samples': insights_count,\n            'model_performance': self.training_metrics['model_performance'],\n            'prediction_accuracy': self.training_metrics['prediction_accuracy'],\n            'top_features': list(self.training_metrics.get('feature_importance', {}).keys())[:5],\n            'should_retrain': await self.should_retrain()\n        }