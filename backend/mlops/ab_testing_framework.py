"""
A/B Testing Framework for DeFi Strategy Performance
Statistical testing and performance comparison for ML model strategies
"""

import asyncio
import logging
import json
import time
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
import pandas as pd

# Statistical libraries
from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu, chi2_contingency, norm
import statsmodels.stats.power as smp
from statsmodels.stats.proportion import proportions_ztest

from ..dexbrain.config import Config

logger = logging.getLogger(__name__)

class ExperimentStatus(Enum):
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    TERMINATED = "terminated"

class VariantType(Enum):
    CONTROL = "control"
    TREATMENT = "treatment"

@dataclass
class ExperimentVariant:
    """A/B test variant configuration"""
    variant_id: str
    variant_type: VariantType
    name: str
    description: str
    traffic_allocation: float  # 0.0 to 1.0
    model_config: Dict[str, Any]
    strategy_parameters: Dict[str, Any]
    active: bool = True

@dataclass
class ExperimentMetric:
    """Metric definition for A/B testing"""
    metric_name: str
    metric_type: str  # 'conversion', 'continuous', 'count'
    aggregation: str  # 'mean', 'sum', 'rate', 'percentage'
    higher_is_better: bool
    minimum_detectable_effect: float  # MDE
    statistical_power: float = 0.8
    significance_level: float = 0.05

@dataclass
class ExperimentConfig:
    """A/B test experiment configuration"""
    experiment_id: str
    name: str
    description: str
    hypothesis: str
    variants: List[ExperimentVariant]
    primary_metric: ExperimentMetric
    secondary_metrics: List[ExperimentMetric]
    target_audience: Dict[str, Any]  # Filtering criteria
    sample_size_per_variant: int
    duration_days: int
    created_by: str
    created_at: datetime
    status: ExperimentStatus = ExperimentStatus.DRAFT

@dataclass
class ExperimentObservation:
    """Single observation/measurement in experiment"""
    observation_id: str
    experiment_id: str
    variant_id: str
    user_id: str  # Pool address or user identifier
    timestamp: datetime
    metrics: Dict[str, float]
    metadata: Dict[str, Any]

@dataclass
class StatisticalResult:
    """Statistical test result"""
    metric_name: str
    test_type: str
    p_value: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    statistical_significance: bool
    practical_significance: bool
    sample_size_control: int
    sample_size_treatment: int

class ABTestingFramework:
    """
    A/B Testing Framework for DeFi Strategy Performance
    Provides statistical testing and performance comparison capabilities
    """
    
    def __init__(self):
        # Storage for experiments and observations
        self.experiments = {}
        self.observations = {}
        self.experiment_assignments = {}  # user_id -> variant_id mapping
        
        # Statistical configuration
        self.statistical_config = {
            'default_power': 0.8,
            'default_alpha': 0.05,
            'min_sample_size': 100,
            'max_duration_days': 90,
            'early_stopping_checks': True,
            'multiple_testing_correction': 'bonferroni'
        }
        
        # Hash salt for consistent user assignment
        self.assignment_salt = "dexter_ab_testing_v1"
        
        logger.info("A/B testing framework initialized")
    
    async def create_experiment(self, config: ExperimentConfig) -> str:
        """Create new A/B test experiment"""
        try:
            # Validate experiment configuration
            validation_result = self._validate_experiment_config(config)
            if not validation_result['is_valid']:
                raise ValueError(f"Invalid experiment config: {validation_result['errors']}")
            
            # Calculate required sample sizes
            sample_sizes = self._calculate_sample_sizes(config)
            config.sample_size_per_variant = max(sample_sizes.values())
            
            # Store experiment
            self.experiments[config.experiment_id] = config
            self.observations[config.experiment_id] = []
            
            logger.info(f"Experiment created: {config.experiment_id}")
            
            return config.experiment_id
            
        except Exception as e:
            logger.error(f"Error creating experiment: {e}")
            raise
    
    def _validate_experiment_config(self, config: ExperimentConfig) -> Dict[str, Any]:
        """Validate experiment configuration"""
        errors = []
        
        # Check variants
        if len(config.variants) < 2:
            errors.append("At least 2 variants required")
        
        control_variants = [v for v in config.variants if v.variant_type == VariantType.CONTROL]
        if len(control_variants) != 1:
            errors.append("Exactly 1 control variant required")
        
        # Check traffic allocation
        total_allocation = sum(v.traffic_allocation for v in config.variants)
        if abs(total_allocation - 1.0) > 0.01:
            errors.append(f"Traffic allocation must sum to 1.0, got {total_allocation}")
        
        # Check metrics
        if not config.primary_metric:
            errors.append("Primary metric is required")
        
        # Check duration
        if config.duration_days <= 0 or config.duration_days > self.statistical_config['max_duration_days']:
            errors.append(f"Duration must be 1-{self.statistical_config['max_duration_days']} days")
        
        return {
            'is_valid': len(errors) == 0,
            'errors': errors
        }
    
    def _calculate_sample_sizes(self, config: ExperimentConfig) -> Dict[str, int]:
        """Calculate required sample sizes for statistical power"""
        sample_sizes = {}
        
        # Calculate for primary metric
        primary_metric = config.primary_metric
        
        if primary_metric.metric_type == 'conversion':
            # Binary outcome (conversion rate)
            baseline_rate = 0.1  # Assume 10% baseline conversion
            effect_size = primary_metric.minimum_detectable_effect
            
            sample_size = smp.ttest_power(
                effect_size=effect_size,
                alpha=primary_metric.significance_level,
                power=primary_metric.statistical_power,
                alternative='two-sided'
            )
            
        else:
            # Continuous outcome
            effect_size = primary_metric.minimum_detectable_effect
            
            sample_size = smp.ttest_power(
                effect_size=effect_size,
                alpha=primary_metric.significance_level,
                power=primary_metric.statistical_power,
                alternative='two-sided'
            )
        
        # Apply minimum sample size
        sample_size = max(int(sample_size), self.statistical_config['min_sample_size'])
        
        # Same sample size for all variants for simplicity
        for variant in config.variants:
            sample_sizes[variant.variant_id] = sample_size
        
        logger.info(f"Calculated sample sizes: {sample_sizes}")
        
        return sample_sizes
    
    async def start_experiment(self, experiment_id: str) -> bool:
        """Start running an experiment"""
        try:
            if experiment_id not in self.experiments:
                raise ValueError(f"Experiment not found: {experiment_id}")
            
            experiment = self.experiments[experiment_id]
            
            if experiment.status != ExperimentStatus.DRAFT:
                raise ValueError(f"Experiment not in draft status: {experiment.status}")
            
            # Update status
            experiment.status = ExperimentStatus.RUNNING
            
            logger.info(f"Experiment started: {experiment_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error starting experiment: {e}")
            return False
    
    async def assign_variant(self, experiment_id: str, user_id: str) -> Optional[str]:
        """Assign user to experiment variant"""
        try:
            if experiment_id not in self.experiments:
                return None
            
            experiment = self.experiments[experiment_id]
            
            if experiment.status != ExperimentStatus.RUNNING:
                return None
            
            # Check if user already assigned
            assignment_key = f"{experiment_id}:{user_id}"
            if assignment_key in self.experiment_assignments:
                return self.experiment_assignments[assignment_key]
            
            # Check if user meets target audience criteria
            if not self._user_meets_criteria(user_id, experiment.target_audience):
                return None
            
            # Assign variant using consistent hashing
            variant_id = self._hash_assignment(experiment_id, user_id, experiment.variants)
            
            # Store assignment
            self.experiment_assignments[assignment_key] = variant_id
            
            logger.debug(f"User assigned: {user_id} -> {variant_id}")
            
            return variant_id
            
        except Exception as e:
            logger.error(f"Error assigning variant: {e}")
            return None
    
    def _user_meets_criteria(self, user_id: str, criteria: Dict[str, Any]) -> bool:
        """Check if user meets experiment target audience criteria"""
        # This would integrate with your user/pool data
        # For now, accept all users
        return True
    
    def _hash_assignment(self, experiment_id: str, user_id: str, variants: List[ExperimentVariant]) -> str:
        """Assign variant using consistent hashing"""
        # Create hash of experiment_id + user_id + salt
        hash_input = f"{experiment_id}:{user_id}:{self.assignment_salt}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        
        # Convert to 0-1 range
        hash_ratio = (hash_value % 1000000) / 1000000.0
        
        # Find variant based on traffic allocation
        cumulative_allocation = 0.0
        for variant in variants:
            cumulative_allocation += variant.traffic_allocation
            if hash_ratio <= cumulative_allocation:
                return variant.variant_id
        
        # Fallback to last variant
        return variants[-1].variant_id
    
    async def record_observation(self, observation: ExperimentObservation):
        """Record an observation/measurement"""
        try:
            if observation.experiment_id not in self.experiments:
                raise ValueError(f"Experiment not found: {observation.experiment_id}")
            
            # Store observation
            self.observations[observation.experiment_id].append(observation)
            
            logger.debug(f"Observation recorded: {observation.observation_id}")
            
        except Exception as e:
            logger.error(f"Error recording observation: {e}")
    
    async def analyze_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """Analyze experiment results with statistical testing"""
        try:
            if experiment_id not in self.experiments:
                raise ValueError(f"Experiment not found: {experiment_id}")
            
            experiment = self.experiments[experiment_id]
            observations = self.observations[experiment_id]
            
            if len(observations) < 2:
                return {'status': 'insufficient_data', 'observations_count': len(observations)}
            
            # Convert observations to DataFrame for analysis
            df = self._observations_to_dataframe(observations)
            
            # Analyze primary metric
            primary_result = await self._analyze_metric(
                df, experiment.primary_metric, experiment.variants
            )
            
            # Analyze secondary metrics
            secondary_results = []
            for metric in experiment.secondary_metrics:
                result = await self._analyze_metric(df, metric, experiment.variants)
                secondary_results.append(result)
            
            # Apply multiple testing correction
            if secondary_results:
                corrected_results = self._apply_multiple_testing_correction(
                    [primary_result] + secondary_results
                )
                primary_result = corrected_results[0]
                secondary_results = corrected_results[1:]
            
            # Calculate experiment summary
            summary = self._calculate_experiment_summary(df, experiment.variants)
            
            # Check for early stopping criteria
            early_stopping = self._check_early_stopping(primary_result, df, experiment)
            
            analysis_result = {
                'experiment_id': experiment_id,
                'analysis_timestamp': datetime.now().isoformat(),
                'observations_count': len(observations),
                'primary_metric_result': asdict(primary_result),
                'secondary_metric_results': [asdict(r) for r in secondary_results],
                'experiment_summary': summary,
                'early_stopping_recommendation': early_stopping,
                'statistical_confidence': primary_result.statistical_significance,
                'practical_significance': primary_result.practical_significance
            }
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error analyzing experiment: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _observations_to_dataframe(self, observations: List[ExperimentObservation]) -> pd.DataFrame:
        """Convert observations to pandas DataFrame"""
        data = []
        
        for obs in observations:
            row = {
                'observation_id': obs.observation_id,
                'experiment_id': obs.experiment_id,
                'variant_id': obs.variant_id,
                'user_id': obs.user_id,
                'timestamp': obs.timestamp
            }
            
            # Add metric values
            for metric_name, value in obs.metrics.items():
                row[metric_name] = value
            
            # Add metadata
            for key, value in obs.metadata.items():
                row[f"meta_{key}"] = value
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    async def _analyze_metric(self, df: pd.DataFrame, metric: ExperimentMetric, 
                            variants: List[ExperimentVariant]) -> StatisticalResult:
        """Analyze a single metric with statistical testing"""
        try:
            # Get control and treatment data
            control_variant = next(v for v in variants if v.variant_type == VariantType.CONTROL)
            treatment_variants = [v for v in variants if v.variant_type == VariantType.TREATMENT]
            
            control_data = df[df['variant_id'] == control_variant.variant_id][metric.metric_name].dropna()
            
            # For simplicity, combine all treatment variants
            treatment_data = df[df['variant_id'].isin([v.variant_id for v in treatment_variants])][metric.metric_name].dropna()
            
            if len(control_data) == 0 or len(treatment_data) == 0:
                return StatisticalResult(
                    metric_name=metric.metric_name,
                    test_type='insufficient_data',
                    p_value=1.0,
                    effect_size=0.0,
                    confidence_interval=(0.0, 0.0),
                    statistical_significance=False,
                    practical_significance=False,
                    sample_size_control=len(control_data),
                    sample_size_treatment=len(treatment_data)
                )
            
            # Choose appropriate statistical test
            if metric.metric_type == 'conversion':
                # Binary outcome - use proportion test
                control_successes = int(control_data.sum())
                treatment_successes = int(treatment_data.sum())
                
                stat, p_value = proportions_ztest(
                    [treatment_successes, control_successes],
                    [len(treatment_data), len(control_data)]
                )
                
                control_rate = control_successes / len(control_data)
                treatment_rate = treatment_successes / len(treatment_data)
                effect_size = treatment_rate - control_rate
                
                # Confidence interval for difference in proportions
                se = np.sqrt(
                    (control_rate * (1 - control_rate) / len(control_data)) +
                    (treatment_rate * (1 - treatment_rate) / len(treatment_data))
                )
                margin_of_error = norm.ppf(1 - metric.significance_level / 2) * se
                confidence_interval = (
                    effect_size - margin_of_error,
                    effect_size + margin_of_error
                )
                
                test_type = 'proportions_ztest'
                
            else:
                # Continuous outcome - use t-test or Mann-Whitney U
                # Check for normality (simplified)
                if len(control_data) > 20 and len(treatment_data) > 20:
                    # Use t-test for larger samples
                    stat, p_value = ttest_ind(treatment_data, control_data, equal_var=False)
                    test_type = 'welch_ttest'
                else:
                    # Use non-parametric test for smaller samples
                    stat, p_value = mannwhitneyu(treatment_data, control_data, alternative='two-sided')
                    test_type = 'mann_whitney_u'
                
                # Calculate effect size (Cohen's d)
                control_mean = control_data.mean()
                treatment_mean = treatment_data.mean()
                pooled_std = np.sqrt(
                    ((len(control_data) - 1) * control_data.var() + 
                     (len(treatment_data) - 1) * treatment_data.var()) /
                    (len(control_data) + len(treatment_data) - 2)
                )
                
                effect_size = (treatment_mean - control_mean) / pooled_std
                
                # Confidence interval for difference in means
                se = pooled_std * np.sqrt(1/len(control_data) + 1/len(treatment_data))
                df = len(control_data) + len(treatment_data) - 2
                t_critical = stats.t.ppf(1 - metric.significance_level / 2, df)
                margin_of_error = t_critical * se
                
                mean_diff = treatment_mean - control_mean
                confidence_interval = (
                    mean_diff - margin_of_error,
                    mean_diff + margin_of_error
                )
            
            # Determine significance
            statistical_significance = p_value < metric.significance_level
            practical_significance = abs(effect_size) >= metric.minimum_detectable_effect
            
            return StatisticalResult(
                metric_name=metric.metric_name,
                test_type=test_type,
                p_value=p_value,
                effect_size=effect_size,
                confidence_interval=confidence_interval,
                statistical_significance=statistical_significance,
                practical_significance=practical_significance,
                sample_size_control=len(control_data),
                sample_size_treatment=len(treatment_data)
            )
            
        except Exception as e:
            logger.error(f"Error analyzing metric {metric.metric_name}: {e}")
            return StatisticalResult(
                metric_name=metric.metric_name,
                test_type='error',
                p_value=1.0,
                effect_size=0.0,
                confidence_interval=(0.0, 0.0),
                statistical_significance=False,
                practical_significance=False,
                sample_size_control=0,
                sample_size_treatment=0
            )
    
    def _apply_multiple_testing_correction(self, results: List[StatisticalResult]) -> List[StatisticalResult]:
        """Apply multiple testing correction (Bonferroni)"""
        if self.statistical_config['multiple_testing_correction'] == 'bonferroni':
            correction_factor = len(results)
            
            for result in results:
                corrected_p_value = min(result.p_value * correction_factor, 1.0)
                result.p_value = corrected_p_value
                result.statistical_significance = corrected_p_value < 0.05
        
        return results
    
    def _calculate_experiment_summary(self, df: pd.DataFrame, variants: List[ExperimentVariant]) -> Dict[str, Any]:
        """Calculate experiment summary statistics"""
        summary = {}
        
        for variant in variants:
            variant_data = df[df['variant_id'] == variant.variant_id]
            
            summary[variant.variant_id] = {
                'variant_name': variant.name,
                'variant_type': variant.variant_type.value,
                'sample_size': len(variant_data),
                'traffic_allocation': variant.traffic_allocation,
                'conversion_metrics': {},
                'continuous_metrics': {}
            }
            
            # Calculate summary stats for each numeric column
            for col in df.select_dtypes(include=[np.number]).columns:
                if col in variant_data.columns:
                    values = variant_data[col].dropna()
                    if len(values) > 0:
                        summary[variant.variant_id]['continuous_metrics'][col] = {
                            'mean': float(values.mean()),
                            'std': float(values.std()),
                            'median': float(values.median()),
                            'count': int(len(values))
                        }
        
        return summary
    
    def _check_early_stopping(self, primary_result: StatisticalResult, 
                            df: pd.DataFrame, experiment: ExperimentConfig) -> Dict[str, Any]:
        """Check if experiment should be stopped early"""
        recommendation = {
            'should_stop': False,
            'reason': None,
            'confidence': 0.0
        }
        
        if not self.statistical_config['early_stopping_checks']:
            return recommendation
        
        # Check for strong statistical significance
        if primary_result.statistical_significance and primary_result.p_value < 0.01:
            recommendation['should_stop'] = True
            recommendation['reason'] = 'strong_statistical_significance'
            recommendation['confidence'] = 1 - primary_result.p_value
        
        # Check for minimum sample size
        min_sample_size = experiment.sample_size_per_variant
        if (primary_result.sample_size_control < min_sample_size or 
            primary_result.sample_size_treatment < min_sample_size):
            recommendation['should_stop'] = False
            recommendation['reason'] = 'insufficient_sample_size'
        
        # Check for futility (very unlikely to reach significance)
        if (primary_result.p_value > 0.8 and 
            primary_result.sample_size_control > min_sample_size * 0.8):
            recommendation['should_stop'] = True
            recommendation['reason'] = 'futility'
            recommendation['confidence'] = 0.8
        
        return recommendation
    
    async def stop_experiment(self, experiment_id: str, reason: str = "manual") -> bool:
        """Stop a running experiment"""
        try:
            if experiment_id not in self.experiments:
                raise ValueError(f"Experiment not found: {experiment_id}")
            
            experiment = self.experiments[experiment_id]
            
            if experiment.status != ExperimentStatus.RUNNING:
                raise ValueError(f"Experiment not running: {experiment.status}")
            
            experiment.status = ExperimentStatus.COMPLETED
            
            logger.info(f"Experiment stopped: {experiment_id} (reason: {reason})")
            
            return True
            
        except Exception as e:
            logger.error(f"Error stopping experiment: {e}")
            return False
    
    def get_experiment_status(self, experiment_id: str) -> Dict[str, Any]:
        """Get experiment status and basic metrics"""
        if experiment_id not in self.experiments:
            return {'error': 'Experiment not found'}
        
        experiment = self.experiments[experiment_id]
        observations = self.observations.get(experiment_id, [])
        
        # Calculate basic stats
        variant_stats = {}
        for variant in experiment.variants:
            variant_obs = [obs for obs in observations if obs.variant_id == variant.variant_id]
            variant_stats[variant.variant_id] = {
                'name': variant.name,
                'observations': len(variant_obs),
                'traffic_allocation': variant.traffic_allocation
            }
        
        return {
            'experiment_id': experiment_id,
            'name': experiment.name,
            'status': experiment.status.value,
            'created_at': experiment.created_at.isoformat(),
            'duration_days': experiment.duration_days,
            'total_observations': len(observations),
            'variants': variant_stats,
            'primary_metric': experiment.primary_metric.metric_name
        }
    
    def list_experiments(self, status_filter: Optional[ExperimentStatus] = None) -> List[Dict[str, Any]]:
        """List all experiments with optional status filter"""
        experiments = []
        
        for exp_id, experiment in self.experiments.items():
            if status_filter is None or experiment.status == status_filter:
                experiments.append({
                    'experiment_id': exp_id,
                    'name': experiment.name,
                    'status': experiment.status.value,
                    'created_at': experiment.created_at.isoformat(),
                    'variants_count': len(experiment.variants),
                    'observations_count': len(self.observations.get(exp_id, []))
                })
        
        return sorted(experiments, key=lambda x: x['created_at'], reverse=True)

# Factory function
def create_ab_testing_framework() -> ABTestingFramework:
    """Create A/B testing framework"""
    return ABTestingFramework()

# Example usage
async def example_usage():
    """Example usage of A/B testing framework"""
    framework = create_ab_testing_framework()
    
    # Create experiment
    variants = [
        ExperimentVariant(
            variant_id="control_v1",
            variant_type=VariantType.CONTROL,
            name="Current Strategy",
            description="Existing DeFi strategy",
            traffic_allocation=0.5,
            model_config={"model": "baseline"},
            strategy_parameters={"compound_frequency": "daily"}
        ),
        ExperimentVariant(
            variant_id="treatment_v1",
            variant_type=VariantType.TREATMENT,
            name="AI-Optimized Strategy",
            description="ML-enhanced strategy",
            traffic_allocation=0.5,
            model_config={"model": "enhanced_ml"},
            strategy_parameters={"compound_frequency": "dynamic"}
        )
    ]
    
    primary_metric = ExperimentMetric(
        metric_name="daily_return",
        metric_type="continuous",
        aggregation="mean",
        higher_is_better=True,
        minimum_detectable_effect=0.02  # 2% improvement
    )
    
    config = ExperimentConfig(
        experiment_id="strategy_optimization_001",
        name="AI Strategy vs Baseline",
        description="Compare AI-optimized strategy against baseline",
        hypothesis="AI-optimized strategy will improve daily returns by 2%",
        variants=variants,
        primary_metric=primary_metric,
        secondary_metrics=[],
        target_audience={},
        sample_size_per_variant=1000,
        duration_days=30,
        created_by="system",
        created_at=datetime.now()
    )
    
    # Create and start experiment
    exp_id = await framework.create_experiment(config)
    await framework.start_experiment(exp_id)
    
    # Simulate observations
    for i in range(200):
        user_id = f"pool_{i}"
        variant_id = await framework.assign_variant(exp_id, user_id)
        
        if variant_id:
            # Simulate different performance for variants
            if variant_id == "control_v1":
                daily_return = np.random.normal(0.05, 0.02)  # 5% ± 2%
            else:
                daily_return = np.random.normal(0.07, 0.02)  # 7% ± 2% (better)
            
            observation = ExperimentObservation(
                observation_id=f"obs_{i}",
                experiment_id=exp_id,
                variant_id=variant_id,
                user_id=user_id,
                timestamp=datetime.now(),
                metrics={"daily_return": daily_return},
                metadata={"pool_type": "USDC/ETH"}
            )
            
            await framework.record_observation(observation)
    
    # Analyze results
    results = await framework.analyze_experiment(exp_id)
    print("Experiment Results:")
    print(json.dumps(results, indent=2, default=str))

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(example_usage())