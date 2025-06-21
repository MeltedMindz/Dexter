"""
Learning Verification System for Dexter AI
Ensures ML models are actually learning from collected data with measurable improvements
"""

import asyncio
import logging
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import pickle
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import pandas as pd

from .advanced_data_collection import AdvancedUniswapDataCollector, PositionData, DataCollectionConfig
from .models.enhanced_ml_models import UniswapLPOptimizer, UniswapFeatures
from .models.knowledge_base import KnowledgeBase
from .training_pipeline import AdvancedTrainingPipeline

logger = logging.getLogger(__name__)

# Website logging
website_logger = logging.getLogger('website')
website_handler = logging.FileHandler('/var/log/dexter/liquidity.log')
website_handler.setFormatter(logging.Formatter('%(message)s'))
website_logger.addHandler(website_handler)
website_logger.setLevel(logging.INFO)

@dataclass
class LearningMetrics:
    """Comprehensive learning performance metrics"""
    timestamp: datetime
    model_version: str
    
    # Prediction accuracy metrics
    apr_prediction_mse: float = 0.0
    apr_prediction_mae: float = 0.0
    apr_prediction_r2: float = 0.0
    
    tick_range_accuracy: float = 0.0
    il_prediction_error: float = 0.0
    fee_yield_accuracy: float = 0.0
    
    # Learning progress metrics
    training_samples: int = 0
    validation_samples: int = 0
    convergence_epochs: int = 0
    final_loss: float = 0.0
    
    # Feature importance
    top_features: List[str] = field(default_factory=list)
    feature_stability: float = 0.0
    
    # Model behavior
    prediction_confidence: float = 0.0
    out_of_sample_performance: float = 0.0
    
    # Learning indicators
    is_learning: bool = False
    learning_rate: float = 0.0
    improvement_over_baseline: float = 0.0
    
    # Data quality impact
    data_quality_correlation: float = 0.0
    high_quality_data_percentage: float = 0.0

@dataclass
class BaselineModel:
    """Simple baseline model for comparison"""
    name: str = "naive_baseline"
    apr_prediction: float = 0.15  # 15% APR baseline
    tick_range_width: int = 2000  # Default tick range
    il_prediction: float = 0.05   # 5% IL baseline
    fee_yield: float = 0.02       # 2% fee yield baseline

class LearningVerificationSystem:
    """
    Comprehensive system to verify and monitor ML learning progress
    """
    
    def __init__(self, data_collector: AdvancedUniswapDataCollector, 
                 knowledge_base: KnowledgeBase, training_pipeline: AdvancedTrainingPipeline):
        self.data_collector = data_collector
        self.knowledge_base = knowledge_base
        self.training_pipeline = training_pipeline
        self.ml_optimizer = UniswapLPOptimizer()
        
        # Learning history
        self.learning_history: List[LearningMetrics] = []
        self.baseline_model = BaselineModel()
        
        # Verification configuration
        self.min_samples_for_verification = 50
        self.learning_threshold = 0.1  # 10% improvement over baseline
        self.data_quality_threshold = 0.8
        
        # Storage paths
        self.metrics_file = Path("/opt/dexter-ai/learning_metrics.json")
        self.models_dir = Path("/opt/dexter-ai/model_storage")
        self.plots_dir = Path("/opt/dexter-ai/learning_plots")
        self.plots_dir.mkdir(exist_ok=True)
        
        # Current learning state
        self.current_metrics: Optional[LearningMetrics] = None
        self.learning_verified = False
        
        logger.info("Learning verification system initialized")
    
    async def run_comprehensive_learning_verification(self) -> Dict[str, Any]:
        """
        Run complete learning verification pipeline
        """
        logger.info("🧪 Starting comprehensive learning verification...")
        
        verification_start = datetime.now()
        results = {
            'verification_time': verification_start.isoformat(),
            'verification_passed': False,
            'learning_detected': False,
            'data_quality_sufficient': False,
            'baseline_outperformed': False,
            'metrics': {},
            'recommendations': []
        }
        
        try:
            # Step 1: Collect fresh training data
            positions = await self.data_collector.collect_comprehensive_position_data()
            logger.info(f"📊 Collected {len(positions)} positions for verification")
            
            if len(positions) < self.min_samples_for_verification:
                results['recommendations'].append(
                    f"Insufficient data: {len(positions)} < {self.min_samples_for_verification} required"
                )
                await self._log_verification_results(results)
                return results
            
            # Step 2: Assess data quality
            data_quality = self._assess_data_quality(positions)
            results['data_quality_sufficient'] = data_quality['overall_score'] >= self.data_quality_threshold
            results['data_quality'] = data_quality
            
            # Step 3: Prepare training and validation sets
            train_data, validation_data = self._prepare_learning_datasets(positions)
            logger.info(f"📚 Prepared {len(train_data)} training, {len(validation_data)} validation samples")
            
            # Step 4: Train model and capture learning metrics
            training_results = await self._train_and_measure_learning(train_data)
            
            # Step 5: Validate model performance
            validation_results = await self._validate_model_performance(validation_data)
            
            # Step 6: Compare against baseline
            baseline_comparison = self._compare_against_baseline(validation_data, validation_results)
            results['baseline_outperformed'] = baseline_comparison['outperformed']
            
            # Step 7: Analyze learning progression
            learning_analysis = self._analyze_learning_progression()
            results['learning_detected'] = learning_analysis['learning_detected']
            
            # Step 8: Create comprehensive metrics
            self.current_metrics = LearningMetrics(
                timestamp=verification_start,
                model_version=f"v{len(self.learning_history) + 1}",
                training_samples=len(train_data),
                validation_samples=len(validation_data),
                **training_results,
                **validation_results,
                **baseline_comparison,
                **learning_analysis
            )
            
            # Step 9: Generate learning visualizations
            await self._generate_learning_visualizations()
            
            # Step 10: Overall verification assessment
            results['verification_passed'] = self._assess_overall_learning(results)
            results['metrics'] = self._format_metrics_for_output()
            
            # Step 11: Generate recommendations
            results['recommendations'] = self._generate_learning_recommendations(results)
            
            # Store results
            self.learning_history.append(self.current_metrics)
            await self._save_learning_metrics()
            
            await self._log_verification_results(results)
            
        except Exception as e:
            logger.error(f"Learning verification failed: {e}")
            results['error'] = str(e)
            results['recommendations'].append(f"Verification failed: {e}")
        
        verification_duration = (datetime.now() - verification_start).total_seconds()
        results['verification_duration'] = verification_duration
        
        return results
    
    def _assess_data_quality(self, positions: List[PositionData]) -> Dict[str, Any]:
        """Assess quality of collected position data"""
        if not positions:
            return {'overall_score': 0.0, 'issues': ['No positions collected']}
        
        quality_metrics = {
            'total_positions': len(positions),
            'complete_data_percentage': 0.0,
            'high_confidence_percentage': 0.0,
            'recent_data_percentage': 0.0,
            'value_diversity_score': 0.0,
            'issues': []
        }
        
        # Check data completeness
        complete_positions = [
            p for p in positions 
            if all([p.position_value_usd > 0, p.tick_lower != p.tick_upper, 
                   p.token0_symbol and p.token1_symbol])
        ]
        quality_metrics['complete_data_percentage'] = len(complete_positions) / len(positions)
        
        # Check confidence scores
        high_confidence = [p for p in positions if p.confidence_score >= 0.8]
        quality_metrics['high_confidence_percentage'] = len(high_confidence) / len(positions)
        
        # Check data recency
        recent_cutoff = datetime.now() - timedelta(days=7)
        recent_positions = [p for p in positions if p.timestamp and p.timestamp > recent_cutoff]
        quality_metrics['recent_data_percentage'] = len(recent_positions) / len(positions)
        
        # Check value diversity (avoid all similar positions)
        values = [p.position_value_usd for p in positions if p.position_value_usd > 0]
        if values:
            value_std = np.std(values)
            value_mean = np.mean(values)
            quality_metrics['value_diversity_score'] = min(1.0, value_std / max(value_mean, 1.0))
        
        # Identify issues
        if quality_metrics['complete_data_percentage'] < 0.7:
            quality_metrics['issues'].append("Low data completeness")
        if quality_metrics['high_confidence_percentage'] < 0.5:
            quality_metrics['issues'].append("Low confidence data")
        if quality_metrics['recent_data_percentage'] < 0.3:
            quality_metrics['issues'].append("Outdated data")
        if quality_metrics['value_diversity_score'] < 0.2:
            quality_metrics['issues'].append("Low data diversity")
        
        # Calculate overall score
        score_components = [
            quality_metrics['complete_data_percentage'],
            quality_metrics['high_confidence_percentage'],
            quality_metrics['recent_data_percentage'],
            quality_metrics['value_diversity_score']
        ]
        quality_metrics['overall_score'] = np.mean(score_components)
        
        return quality_metrics
    
    def _prepare_learning_datasets(self, positions: List[PositionData]) -> Tuple[List[Dict], List[Dict]]:
        """Prepare training and validation datasets"""
        # Convert positions to training format
        training_examples = []
        for position in positions:
            try:
                example = {
                    # Input features
                    'pool_tvl': position.position_value_usd,
                    'fee_tier': position.fee_tier,
                    'tick_lower': position.tick_lower,
                    'tick_upper': position.tick_upper,
                    'current_tick': position.current_tick,
                    'range_width': position.range_width,
                    'liquidity': position.liquidity,
                    'sqrt_price': position.sqrt_price,
                    'in_range': 1.0 if position.in_range else 0.0,
                    'capital_efficiency': position.capital_efficiency,
                    'deposited_token0': position.deposited_token0,
                    'deposited_token1': position.deposited_token1,
                    
                    # Target variables
                    'target_apr': position.apr_estimate,
                    'target_fee_yield': position.fee_yield,
                    'target_il': position.impermanent_loss,
                    'target_fees_collected': position.collected_fees_token0 + position.collected_fees_token1,
                    
                    # Metadata
                    'data_quality_score': position.confidence_score,
                    'data_source': position.data_source,
                    'timestamp': position.timestamp.isoformat() if position.timestamp else None
                }
                training_examples.append(example)
            except Exception as e:
                logger.warning(f"Failed to convert position to training example: {e}")
                continue
        
        # Split into train/validation (80/20)
        train_data, validation_data = train_test_split(
            training_examples, 
            test_size=0.2, 
            random_state=42,
            stratify=None  # Could stratify by fee_tier or other categorical features
        )
        
        return train_data, validation_data
    
    async def _train_and_measure_learning(self, train_data: List[Dict]) -> Dict[str, Any]:
        """Train model and measure learning metrics"""
        logger.info("🏋️ Training model and measuring learning...")
        
        training_metrics = {
            'convergence_epochs': 0,
            'final_loss': 0.0,
            'top_features': [],
            'feature_stability': 0.0
        }
        
        try:
            # Train the model
            training_results = self.ml_optimizer.train_models(
                training_data=train_data,
                epochs=100,
                batch_size=32,
                learning_rate=0.001
            )
            
            if training_results.get('status') == 'success':
                # Extract learning metrics
                results = training_results.get('results', {})
                
                training_metrics['convergence_epochs'] = results.get('epochs_to_convergence', 100)
                training_metrics['final_loss'] = results.get('final_loss', 1.0)
                
                # Analyze feature importance (if available)
                feature_importance = results.get('feature_importance', {})
                if feature_importance:
                    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                    training_metrics['top_features'] = [f[0] for f in sorted_features[:10]]
                    
                    # Feature stability (how consistent are top features across training runs)
                    if len(self.learning_history) > 0:
                        previous_features = self.learning_history[-1].top_features
                        overlap = len(set(training_metrics['top_features'][:5]) & set(previous_features[:5]))
                        training_metrics['feature_stability'] = overlap / 5.0
                
        except Exception as e:
            logger.error(f"Training and measurement failed: {e}")
            training_metrics['final_loss'] = 1.0  # High loss indicates poor training
        
        return training_metrics
    
    async def _validate_model_performance(self, validation_data: List[Dict]) -> Dict[str, Any]:
        """Validate model performance on held-out data"""
        logger.info("🔍 Validating model performance...")
        
        validation_metrics = {
            'apr_prediction_mse': 1.0,
            'apr_prediction_mae': 1.0,
            'apr_prediction_r2': 0.0,
            'tick_range_accuracy': 0.0,
            'il_prediction_error': 1.0,
            'fee_yield_accuracy': 0.0,
            'prediction_confidence': 0.0,
            'out_of_sample_performance': 0.0
        }
        
        if not validation_data:
            return validation_metrics
        
        try:
            # Collect predictions and actual values
            apr_predictions = []
            apr_actuals = []
            prediction_confidences = []
            
            for sample in validation_data:
                try:
                    # Get model prediction
                    prediction = self.ml_optimizer.predict_optimal_position(sample)
                    
                    if prediction and 'predicted_apr' in prediction:
                        apr_predictions.append(prediction['predicted_apr'])
                        apr_actuals.append(sample.get('target_apr', 0.0))
                        prediction_confidences.append(prediction.get('confidence', 0.5))
                
                except Exception as e:
                    continue
            
            # Calculate metrics if we have predictions
            if len(apr_predictions) >= 10:  # Need minimum samples for meaningful metrics
                apr_predictions = np.array(apr_predictions)
                apr_actuals = np.array(apr_actuals)
                
                # APR prediction metrics
                validation_metrics['apr_prediction_mse'] = mean_squared_error(apr_actuals, apr_predictions)
                validation_metrics['apr_prediction_mae'] = mean_absolute_error(apr_actuals, apr_predictions)
                
                # R-squared (coefficient of determination)
                try:
                    validation_metrics['apr_prediction_r2'] = r2_score(apr_actuals, apr_predictions)
                except:
                    validation_metrics['apr_prediction_r2'] = 0.0
                
                # Prediction confidence
                validation_metrics['prediction_confidence'] = np.mean(prediction_confidences)
                
                # Out-of-sample performance (inverse of normalized MSE)
                normalized_mse = validation_metrics['apr_prediction_mse'] / (np.var(apr_actuals) + 1e-8)
                validation_metrics['out_of_sample_performance'] = max(0.0, 1.0 - normalized_mse)
                
                logger.info(f"Validation metrics: MSE={validation_metrics['apr_prediction_mse']:.4f}, "
                           f"MAE={validation_metrics['apr_prediction_mae']:.4f}, "
                           f"R²={validation_metrics['apr_prediction_r2']:.4f}")
        
        except Exception as e:
            logger.error(f"Validation failed: {e}")
        
        return validation_metrics
    
    def _compare_against_baseline(self, validation_data: List[Dict], validation_results: Dict) -> Dict[str, Any]:
        """Compare model performance against simple baseline"""
        baseline_comparison = {
            'outperformed': False,
            'improvement_over_baseline': 0.0,
            'baseline_mse': 1.0,
            'model_mse': validation_results.get('apr_prediction_mse', 1.0)
        }
        
        try:
            # Calculate baseline performance
            actual_aprs = [sample.get('target_apr', 0.0) for sample in validation_data]
            baseline_predictions = [self.baseline_model.apr_prediction] * len(actual_aprs)
            
            if len(actual_aprs) >= 10:
                baseline_mse = mean_squared_error(actual_aprs, baseline_predictions)
                model_mse = validation_results.get('apr_prediction_mse', 1.0)
                
                baseline_comparison['baseline_mse'] = baseline_mse
                baseline_comparison['model_mse'] = model_mse
                
                # Calculate improvement
                if baseline_mse > 0:
                    improvement = (baseline_mse - model_mse) / baseline_mse
                    baseline_comparison['improvement_over_baseline'] = improvement
                    baseline_comparison['outperformed'] = improvement > self.learning_threshold
                
                logger.info(f"Baseline comparison: Baseline MSE={baseline_mse:.4f}, "
                           f"Model MSE={model_mse:.4f}, "
                           f"Improvement={baseline_comparison['improvement_over_baseline']:.1%}")
        
        except Exception as e:
            logger.warning(f"Baseline comparison failed: {e}")
        
        return baseline_comparison
    
    def _analyze_learning_progression(self) -> Dict[str, Any]:
        """Analyze learning progression over time"""
        learning_analysis = {
            'learning_detected': False,
            'learning_rate': 0.0,
            'is_learning': False
        }
        
        try:
            if len(self.learning_history) < 2:
                return learning_analysis
            
            # Get recent performance metrics
            recent_metrics = self.learning_history[-3:]  # Last 3 training sessions
            
            if len(recent_metrics) >= 2:
                # Calculate learning rate (improvement in R² over time)
                r2_scores = [m.apr_prediction_r2 for m in recent_metrics]
                
                if len(r2_scores) >= 2:
                    # Simple linear trend
                    x = np.arange(len(r2_scores))
                    learning_rate = np.polyfit(x, r2_scores, 1)[0]  # Slope of trend line
                    
                    learning_analysis['learning_rate'] = learning_rate
                    learning_analysis['learning_detected'] = learning_rate > 0.01  # 1% improvement per session
                    learning_analysis['is_learning'] = learning_rate > 0.005  # 0.5% improvement threshold
                
                logger.info(f"Learning analysis: Rate={learning_analysis['learning_rate']:.4f}, "
                           f"Learning detected={learning_analysis['learning_detected']}")
        
        except Exception as e:
            logger.warning(f"Learning progression analysis failed: {e}")
        
        return learning_analysis
    
    def _assess_overall_learning(self, results: Dict[str, Any]) -> bool:
        """Assess overall learning based on all verification criteria"""
        criteria = [
            results.get('data_quality_sufficient', False),
            results.get('baseline_outperformed', False),
            results.get('learning_detected', False),
            # Add model performance threshold
            self.current_metrics.apr_prediction_r2 > 0.1 if self.current_metrics else False,
            # Add confidence threshold
            self.current_metrics.prediction_confidence > 0.6 if self.current_metrics else False
        ]
        
        passed_criteria = sum(criteria)
        total_criteria = len(criteria)
        
        # Require at least 3 out of 5 criteria to pass
        learning_verified = passed_criteria >= 3
        
        logger.info(f"Learning verification: {passed_criteria}/{total_criteria} criteria passed")
        return learning_verified
    
    def _format_metrics_for_output(self) -> Dict[str, Any]:
        """Format current metrics for output"""
        if not self.current_metrics:
            return {}
        
        return {
            'model_version': self.current_metrics.model_version,
            'training_samples': self.current_metrics.training_samples,
            'validation_samples': self.current_metrics.validation_samples,
            'apr_prediction_accuracy': {
                'mse': self.current_metrics.apr_prediction_mse,
                'mae': self.current_metrics.apr_prediction_mae,
                'r2_score': self.current_metrics.apr_prediction_r2
            },
            'learning_indicators': {
                'is_learning': self.current_metrics.is_learning,
                'learning_rate': self.current_metrics.learning_rate,
                'improvement_over_baseline': self.current_metrics.improvement_over_baseline
            },
            'model_confidence': self.current_metrics.prediction_confidence,
            'feature_analysis': {
                'top_features': self.current_metrics.top_features[:5],
                'feature_stability': self.current_metrics.feature_stability
            }
        }
    
    def _generate_learning_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on verification results"""
        recommendations = []
        
        if not results.get('data_quality_sufficient', False):
            recommendations.append("Improve data quality: collect more complete and recent position data")
        
        if not results.get('baseline_outperformed', False):
            recommendations.append("Model underperforming: increase model complexity or feature engineering")
        
        if not results.get('learning_detected', False):
            recommendations.append("No learning detected: increase training data or adjust learning parameters")
        
        if self.current_metrics:
            if self.current_metrics.prediction_confidence < 0.6:
                recommendations.append("Low prediction confidence: improve model calibration")
            
            if self.current_metrics.apr_prediction_r2 < 0.1:
                recommendations.append("Poor predictive power: review feature selection and model architecture")
            
            if self.current_metrics.feature_stability < 0.4:
                recommendations.append("Unstable features: investigate feature engineering and data consistency")
        
        if not recommendations:
            recommendations.append("Learning verification successful: continue current training approach")
        
        return recommendations
    
    async def _generate_learning_visualizations(self):
        """Generate learning progress visualizations"""
        try:
            if len(self.learning_history) < 2:
                return
            
            # Learning progress over time
            timestamps = [m.timestamp for m in self.learning_history]
            r2_scores = [m.apr_prediction_r2 for m in self.learning_history]
            learning_rates = [m.learning_rate for m in self.learning_history]
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # R² score progression
            ax1.plot(timestamps, r2_scores, 'o-', label='R² Score')
            ax1.set_title('Model Performance Over Time')
            ax1.set_ylabel('R² Score')
            ax1.legend()
            ax1.grid(True)
            
            # Learning rate progression
            ax2.plot(timestamps, learning_rates, 's-', color='orange', label='Learning Rate')
            ax2.set_title('Learning Rate Over Time')
            ax2.set_ylabel('Learning Rate')
            ax2.set_xlabel('Time')
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            plt.savefig(self.plots_dir / 'learning_progression.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info("Learning visualizations generated")
            
        except Exception as e:
            logger.warning(f"Failed to generate visualizations: {e}")
    
    async def _save_learning_metrics(self):
        """Save learning metrics to file"""
        try:
            metrics_data = {
                'learning_history': [
                    {
                        'timestamp': m.timestamp.isoformat(),
                        'model_version': m.model_version,
                        'metrics': {
                            'apr_prediction_mse': m.apr_prediction_mse,
                            'apr_prediction_r2': m.apr_prediction_r2,
                            'learning_rate': m.learning_rate,
                            'improvement_over_baseline': m.improvement_over_baseline,
                            'is_learning': m.is_learning,
                            'training_samples': m.training_samples,
                            'prediction_confidence': m.prediction_confidence
                        }
                    }
                    for m in self.learning_history
                ],
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.metrics_file, 'w') as f:
                json.dump(metrics_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save learning metrics: {e}")
    
    async def _log_verification_results(self, results: Dict[str, Any]):
        """Log verification results to website"""
        try:
            log_data = {
                "timestamp": datetime.now().isoformat(),
                "level": "SUCCESS" if results['verification_passed'] else "WARNING",
                "source": "LearningVerification",
                "message": f"Learning verification {'PASSED' if results['verification_passed'] else 'FAILED'}",
                "details": {
                    "verification_passed": results['verification_passed'],
                    "learning_detected": results['learning_detected'],
                    "baseline_outperformed": results['baseline_outperformed'],
                    "data_quality_sufficient": results['data_quality_sufficient'],
                    "metrics_summary": results.get('metrics', {}),
                    "recommendations": results['recommendations'][:3],  # Top 3 recommendations
                    "verification_duration": results.get('verification_duration', 0)
                }
            }
            
            website_logger.info(json.dumps(log_data))
            
        except Exception as e:
            logger.warning(f"Failed to log verification results: {e}")
    
    async def continuous_learning_monitoring(self):
        """Continuously monitor learning progress"""
        logger.info("🔄 Starting continuous learning monitoring...")
        
        while True:
            try:
                # Run verification every 6 hours
                results = await self.run_comprehensive_learning_verification()
                
                if results['verification_passed']:
                    logger.info("✅ Learning verification passed")
                    self.learning_verified = True
                else:
                    logger.warning("❌ Learning verification failed")
                    self.learning_verified = False
                
                # Wait 6 hours before next verification
                await asyncio.sleep(6 * 3600)
                
            except Exception as e:
                logger.error(f"Continuous learning monitoring error: {e}")
                await asyncio.sleep(3600)  # Wait 1 hour on error


async def main():
    """Test the learning verification system"""
    import os
    
    # Initialize components
    config = DataCollectionConfig(
        graph_api_key="c6f241c1dd5aea81977a63b2614af70d",
        alchemy_api_key=os.getenv('ALCHEMY_API_KEY', ''),
        infura_api_key=os.getenv('INFURA_API_KEY', ''),
        base_rpc_url="https://base-mainnet.g.alchemy.com/v2/" + os.getenv('ALCHEMY_API_KEY', '')
    )
    
    data_collector = AdvancedUniswapDataCollector(config)
    knowledge_base = KnowledgeBase()
    training_pipeline = AdvancedTrainingPipeline(knowledge_base)
    
    # Run verification
    verification_system = LearningVerificationSystem(data_collector, knowledge_base, training_pipeline)
    results = await verification_system.run_comprehensive_learning_verification()
    
    print(f"Verification Results: {json.dumps(results, indent=2, default=str)}")

if __name__ == "__main__":
    asyncio.run(main())