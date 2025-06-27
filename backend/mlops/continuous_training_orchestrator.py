"""
MLOps Level 2 Continuous Training Orchestrator
Automated training pipeline with performance monitoring and model deployment
"""

import asyncio
import logging
import json
import time
import os
import shutil
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import pandas as pd

# ML and MLOps imports
import mlflow
import mlflow.sklearn
import mlflow.pytorch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import joblib
import torch

# Custom imports
from ..dexbrain.config import Config
from ..streaming.online_learning_engine import OnlineDeFiOptimizer
from .model_registry import ModelRegistry
from .performance_monitor import PerformanceMonitor

logger = logging.getLogger(__name__)

@dataclass
class TrainingTrigger:
    """Training trigger configuration"""
    trigger_type: str  # 'schedule', 'performance', 'drift', 'data_volume'
    threshold: float
    enabled: bool = True
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0

@dataclass
class TrainingJob:
    """Training job configuration"""
    job_id: str
    model_name: str
    dataset_version: str
    hyperparameters: Dict[str, Any]
    training_config: Dict[str, Any]
    status: str = 'pending'  # 'pending', 'running', 'completed', 'failed'
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    metrics: Dict[str, float] = None
    model_path: Optional[str] = None

@dataclass
class ModelDeployment:
    """Model deployment configuration"""
    model_id: str
    model_version: str
    environment: str  # 'staging', 'production', 'canary'
    deployment_strategy: str  # 'blue_green', 'rolling', 'canary'
    health_check_url: str
    rollback_version: Optional[str] = None

class ContinuousTrainingOrchestrator:
    """
    MLOps Level 2 Continuous Training Orchestrator
    Manages automated training, validation, and deployment pipeline
    """
    
    def __init__(self):
        # MLflow configuration
        self.mlflow_tracking_uri = os.getenv('MLFLOW_TRACKING_URI', 'file:./mlruns')
        self.mlflow_experiment_name = "dexter_continuous_training"
        
        # Initialize MLflow
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        
        # Create or get experiment
        try:
            experiment = mlflow.get_experiment_by_name(self.mlflow_experiment_name)
            if experiment is None:
                self.experiment_id = mlflow.create_experiment(self.mlflow_experiment_name)
            else:
                self.experiment_id = experiment.experiment_id
        except Exception as e:
            logger.warning(f"MLflow experiment setup warning: {e}")
            self.experiment_id = None
        
        # Training triggers
        self.triggers = {
            'scheduled_daily': TrainingTrigger('schedule', 24.0),  # 24 hours
            'performance_degradation': TrainingTrigger('performance', 0.05),  # 5% drop
            'concept_drift': TrainingTrigger('drift', 0.1),  # 10% drift threshold
            'data_volume': TrainingTrigger('data_volume', 10000)  # 10k new samples
        }
        
        # Components
        self.model_registry = ModelRegistry()
        self.performance_monitor = PerformanceMonitor()
        self.online_optimizer = None
        
        # Training configuration
        self.training_config = {
            'batch_size': 1000,
            'validation_split': 0.2,
            'max_training_time': 3600,  # 1 hour max
            'early_stopping_patience': 10,
            'model_types': ['strategy_classifier', 'return_regressor', 'volatility_detector'],
            'hyperparameter_search': True,
            'cross_validation_folds': 5
        }
        
        # Deployment configuration
        self.deployment_config = {
            'staging_approval_required': True,
            'production_approval_required': True,
            'canary_traffic_percentage': 10,
            'health_check_timeout': 300,
            'rollback_on_error_rate': 0.1
        }
        
        # State tracking
        self.active_jobs = {}
        self.training_history = []
        self.last_training_check = time.time()
        
        # Paths
        self.data_path = Path("/opt/dexter-ai/training_data")
        self.model_artifacts_path = Path("/opt/dexter-ai/model_artifacts")
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.model_artifacts_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("Continuous training orchestrator initialized")
    
    async def start(self):
        """Start the continuous training orchestrator"""
        # Initialize components
        await self.model_registry.initialize()
        await self.performance_monitor.initialize()
        
        # Load existing online optimizer
        self.online_optimizer = OnlineDeFiOptimizer()
        await self.online_optimizer.load_models()
        
        logger.info("Continuous training orchestrator started")
    
    async def run_orchestration_loop(self):
        """Main orchestration loop"""
        while True:
            try:
                # Check training triggers
                triggered = await self._check_training_triggers()
                
                if triggered:
                    await self._execute_training_pipeline()
                
                # Monitor active jobs
                await self._monitor_training_jobs()
                
                # Check for deployment opportunities
                await self._check_deployment_candidates()
                
                # Cleanup old artifacts
                await self._cleanup_old_artifacts()
                
                # Sleep for check interval
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Orchestration loop error: {e}")
                await asyncio.sleep(60)  # Shorter sleep on error
    
    async def _check_training_triggers(self) -> List[str]:
        """Check if any training triggers are activated"""
        triggered_triggers = []
        current_time = datetime.now()
        
        for trigger_name, trigger in self.triggers.items():
            if not trigger.enabled:
                continue
            
            should_trigger = False
            
            if trigger.trigger_type == 'schedule':
                # Check if enough time has passed since last training
                if (trigger.last_triggered is None or 
                    (current_time - trigger.last_triggered).total_seconds() / 3600 >= trigger.threshold):
                    should_trigger = True
            
            elif trigger.trigger_type == 'performance':
                # Check for performance degradation
                current_performance = await self._get_current_performance()
                baseline_performance = await self._get_baseline_performance()
                
                if (baseline_performance and current_performance and
                    baseline_performance - current_performance >= trigger.threshold):
                    should_trigger = True
            
            elif trigger.trigger_type == 'drift':
                # Check for concept drift
                drift_detected = await self._check_concept_drift()
                if drift_detected and drift_detected >= trigger.threshold:
                    should_trigger = True
            
            elif trigger.trigger_type == 'data_volume':
                # Check for sufficient new data
                new_samples = await self._count_new_training_samples()
                if new_samples >= trigger.threshold:
                    should_trigger = True
            
            if should_trigger:
                trigger.last_triggered = current_time
                trigger.trigger_count += 1
                triggered_triggers.append(trigger_name)
                
                logger.info(f"Training trigger activated: {trigger_name}")
        
        return triggered_triggers
    
    async def _execute_training_pipeline(self):
        """Execute the complete training pipeline"""
        try:
            job_id = f"training_{int(time.time())}"
            
            logger.info(f"Starting training pipeline: {job_id}")
            
            # Step 1: Data preparation
            dataset_version = await self._prepare_training_data()
            
            # Step 2: Create training jobs for different models
            training_jobs = []
            for model_type in self.training_config['model_types']:
                job = TrainingJob(
                    job_id=f"{job_id}_{model_type}",
                    model_name=model_type,
                    dataset_version=dataset_version,
                    hyperparameters=await self._get_hyperparameters(model_type),
                    training_config=self.training_config
                )
                training_jobs.append(job)
                self.active_jobs[job.job_id] = job
            
            # Step 3: Execute training jobs in parallel
            training_tasks = []
            for job in training_jobs:
                task = asyncio.create_task(self._execute_training_job(job))
                training_tasks.append(task)
            
            # Wait for all training jobs to complete
            results = await asyncio.gather(*training_tasks, return_exceptions=True)
            
            # Step 4: Evaluate and select best models
            successful_jobs = [job for job, result in zip(training_jobs, results) 
                             if not isinstance(result, Exception) and job.status == 'completed']
            
            if successful_jobs:
                # Step 5: Model validation and staging
                for job in successful_jobs:
                    await self._validate_and_stage_model(job)
                
                logger.info(f"Training pipeline completed: {len(successful_jobs)}/{len(training_jobs)} successful")
            else:
                logger.error("Training pipeline failed: no successful jobs")
            
        except Exception as e:
            logger.error(f"Training pipeline execution error: {e}")
    
    async def _prepare_training_data(self) -> str:
        """Prepare training dataset"""
        try:
            # Collect recent data from various sources
            dataset_version = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Load streaming data from Kafka/database
            training_data = await self._load_training_data()
            
            # Feature engineering
            features, targets = await self._engineer_features(training_data)
            
            # Data validation and quality checks
            data_quality = await self._validate_data_quality(features, targets)
            
            if not data_quality['is_valid']:
                raise ValueError(f"Data quality validation failed: {data_quality['issues']}")
            
            # Save dataset
            dataset_path = self.data_path / f"dataset_{dataset_version}"
            dataset_path.mkdir(exist_ok=True)
            
            # Save features and targets
            features.to_parquet(dataset_path / "features.parquet")
            targets.to_parquet(dataset_path / "targets.parquet")
            
            # Save metadata
            metadata = {
                'version': dataset_version,
                'created_at': datetime.now().isoformat(),
                'sample_count': len(features),
                'feature_count': len(features.columns),
                'data_quality': data_quality,
                'source': 'continuous_training'
            }
            
            with open(dataset_path / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Training dataset prepared: {dataset_version} ({len(features)} samples)")
            
            return dataset_version
            
        except Exception as e:
            logger.error(f"Data preparation error: {e}")
            raise
    
    async def _load_training_data(self) -> pd.DataFrame:
        """Load training data from various sources"""
        # This would integrate with your data pipeline
        # For now, simulate with sample data
        
        n_samples = 10000
        data = {
            'pool_address': ['0x1234567890123456789012345678901234567890'] * n_samples,
            'timestamp': [time.time() - i * 60 for i in range(n_samples)],
            'price': np.random.normal(3000, 300, n_samples),
            'volume_1h': np.random.exponential(100000, n_samples),
            'liquidity': np.random.exponential(5000000, n_samples),
            'volatility_24h': np.random.exponential(0.1, n_samples),
            'strategy_action': np.random.choice(['hold', 'compound', 'rebalance'], n_samples),
            'actual_return': np.random.normal(0.05, 0.1, n_samples),
            'volatility_regime': np.random.choice(['low', 'medium', 'high'], n_samples)
        }
        
        return pd.DataFrame(data)
    
    async def _engineer_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Engineer features for training"""
        # Basic feature engineering
        features = data[['price', 'volume_1h', 'liquidity', 'volatility_24h']].copy()
        
        # Add derived features
        features['price_change'] = features['price'].pct_change().fillna(0)
        features['volume_ma'] = features['volume_1h'].rolling(window=10).mean().fillna(features['volume_1h'])
        features['liquidity_ratio'] = features['liquidity'] / features['liquidity'].mean()
        features['volatility_zscore'] = (features['volatility_24h'] - features['volatility_24h'].mean()) / features['volatility_24h'].std()
        
        # Target variables
        targets = data[['strategy_action', 'actual_return', 'volatility_regime']].copy()
        
        return features, targets
    
    async def _validate_data_quality(self, features: pd.DataFrame, targets: pd.DataFrame) -> Dict[str, Any]:
        """Validate data quality"""
        issues = []
        
        # Check for missing values
        if features.isnull().sum().sum() > 0:
            issues.append("Missing values in features")
        
        if targets.isnull().sum().sum() > 0:
            issues.append("Missing values in targets")
        
        # Check for data drift
        # (Would compare with reference dataset in production)
        
        # Check sample size
        if len(features) < 1000:
            issues.append("Insufficient training samples")
        
        # Check feature variance
        low_variance_features = features.columns[features.var() < 1e-6]
        if len(low_variance_features) > 0:
            issues.append(f"Low variance features: {list(low_variance_features)}")
        
        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'sample_count': len(features),
            'feature_count': len(features.columns)
        }
    
    async def _get_hyperparameters(self, model_type: str) -> Dict[str, Any]:
        """Get hyperparameters for model type"""
        hyperparameters = {
            'strategy_classifier': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'learning_rate': [0.01, 0.1, 0.2],
                'random_state': 42
            },
            'return_regressor': {
                'n_estimators': [100, 200],
                'max_depth': [10, 15, 20],
                'learning_rate': [0.01, 0.05, 0.1],
                'random_state': 42
            },
            'volatility_detector': {
                'C': [0.1, 1.0, 10.0],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto'],
                'random_state': 42
            }
        }
        
        return hyperparameters.get(model_type, {})
    
    async def _execute_training_job(self, job: TrainingJob):
        """Execute a single training job"""
        try:
            job.status = 'running'
            job.start_time = datetime.now()
            
            logger.info(f"Starting training job: {job.job_id}")
            
            # Load dataset
            dataset_path = self.data_path / f"dataset_{job.dataset_version}"
            features = pd.read_parquet(dataset_path / "features.parquet")
            targets = pd.read_parquet(dataset_path / "targets.parquet")
            
            # Start MLflow run
            with mlflow.start_run(experiment_id=self.experiment_id, run_name=job.job_id):
                # Log parameters
                mlflow.log_params(job.hyperparameters)
                mlflow.log_params(job.training_config)
                
                # Train model
                model, metrics = await self._train_model(job, features, targets)
                
                # Log metrics
                mlflow.log_metrics(metrics)
                
                # Save model
                model_path = self.model_artifacts_path / f"{job.job_id}.joblib"
                joblib.dump(model, model_path)
                job.model_path = str(model_path)
                
                # Log model to MLflow
                mlflow.sklearn.log_model(model, "model")
                
                job.metrics = metrics
                job.status = 'completed'
                job.end_time = datetime.now()
                
                logger.info(f"Training job completed: {job.job_id}")
                
        except Exception as e:
            job.status = 'failed'
            job.end_time = datetime.now()
            logger.error(f"Training job failed: {job.job_id} - {e}")
            raise
    
    async def _train_model(self, job: TrainingJob, features: pd.DataFrame, targets: pd.DataFrame) -> Tuple[Any, Dict[str, float]]:
        """Train model for specific job"""
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.svm import SVC
        from sklearn.model_selection import GridSearchCV
        
        # Select target variable based on model type
        if job.model_name == 'strategy_classifier':
            y = targets['strategy_action']
            model_class = RandomForestClassifier
            scoring = 'accuracy'
        elif job.model_name == 'return_regressor':
            y = targets['actual_return']
            model_class = RandomForestRegressor
            scoring = 'neg_mean_squared_error'
        elif job.model_name == 'volatility_detector':
            y = targets['volatility_regime']
            model_class = SVC
            scoring = 'accuracy'
        else:
            raise ValueError(f"Unknown model type: {job.model_name}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, y, test_size=job.training_config['validation_split'], random_state=42
        )
        
        # Hyperparameter search if enabled
        if job.training_config.get('hyperparameter_search', False):
            model = GridSearchCV(
                model_class(),
                job.hyperparameters,
                cv=job.training_config['cross_validation_folds'],
                scoring=scoring,
                n_jobs=-1
            )
        else:
            # Use default hyperparameters
            default_params = {k: v[0] if isinstance(v, list) else v 
                            for k, v in job.hyperparameters.items()}
            model = model_class(**default_params)
        
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        if job.model_name in ['strategy_classifier', 'volatility_detector']:
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1_score': f1_score(y_test, y_pred, average='weighted')
            }
        else:  # regression
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            metrics = {
                'mse': mean_squared_error(y_test, y_pred),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2_score': r2_score(y_test, y_pred)
            }
        
        return model, metrics
    
    async def _validate_and_stage_model(self, job: TrainingJob):
        """Validate model and move to staging"""
        try:
            # Load model
            model = joblib.load(job.model_path)
            
            # Performance validation
            performance_valid = await self._validate_model_performance(job, model)
            
            if performance_valid:
                # Register model in model registry
                model_version = await self.model_registry.register_model(
                    name=job.model_name,
                    model=model,
                    metrics=job.metrics,
                    metadata={
                        'training_job_id': job.job_id,
                        'dataset_version': job.dataset_version,
                        'training_time': (job.end_time - job.start_time).total_seconds(),
                        'hyperparameters': job.hyperparameters
                    }
                )
                
                # Move to staging
                await self.model_registry.transition_model_stage(
                    name=job.model_name,
                    version=model_version,
                    stage='staging'
                )
                
                logger.info(f"Model staged: {job.model_name} v{model_version}")
                
                # Schedule staging evaluation
                await self._schedule_staging_evaluation(job.model_name, model_version)
                
            else:
                logger.warning(f"Model validation failed: {job.model_name}")
                
        except Exception as e:
            logger.error(f"Model validation error: {e}")
    
    async def _validate_model_performance(self, job: TrainingJob, model: Any) -> bool:
        """Validate model performance against benchmarks"""
        try:
            # Get baseline performance
            baseline_metrics = await self._get_baseline_metrics(job.model_name)
            
            if not baseline_metrics:
                # No baseline, accept if metrics are reasonable
                return True
            
            # Compare key metrics
            current_metrics = job.metrics
            
            if job.model_name in ['strategy_classifier', 'volatility_detector']:
                key_metric = 'accuracy'
                improvement_threshold = 0.02  # 2% improvement required
            else:
                key_metric = 'r2_score'
                improvement_threshold = 0.05  # 5% improvement required
            
            current_score = current_metrics.get(key_metric, 0)
            baseline_score = baseline_metrics.get(key_metric, 0)
            
            # Check if improvement meets threshold
            improvement = current_score - baseline_score
            
            return improvement >= improvement_threshold
            
        except Exception as e:
            logger.error(f"Performance validation error: {e}")
            return False
    
    async def _monitor_training_jobs(self):
        """Monitor active training jobs"""
        for job_id, job in list(self.active_jobs.items()):
            if job.status in ['completed', 'failed']:
                # Move to history
                self.training_history.append(job)
                del self.active_jobs[job_id]
            elif job.status == 'running':
                # Check for timeout
                if (datetime.now() - job.start_time).total_seconds() > self.training_config['max_training_time']:
                    job.status = 'failed'
                    logger.warning(f"Training job timed out: {job_id}")
    
    async def _check_deployment_candidates(self):
        """Check for models ready for production deployment"""
        try:
            staging_models = await self.model_registry.get_models_by_stage('staging')
            
            for model_info in staging_models:
                # Check if staging evaluation is complete
                evaluation_complete = await self._check_staging_evaluation(model_info)
                
                if evaluation_complete:
                    # Deploy to production with canary strategy
                    await self._deploy_model_to_production(model_info)
                    
        except Exception as e:
            logger.error(f"Deployment check error: {e}")
    
    async def _deploy_model_to_production(self, model_info: Dict[str, Any]):
        """Deploy model to production with canary strategy"""
        try:
            model_name = model_info['name']
            model_version = model_info['version']
            
            logger.info(f"Deploying to production: {model_name} v{model_version}")
            
            # Create deployment plan
            deployment = ModelDeployment(
                model_id=f"{model_name}_{model_version}",
                model_version=model_version,
                environment='production',
                deployment_strategy='canary',
                health_check_url=f"/health/{model_name}"
            )
            
            # Execute canary deployment
            success = await self._execute_canary_deployment(deployment)
            
            if success:
                # Transition model stage
                await self.model_registry.transition_model_stage(
                    name=model_name,
                    version=model_version,
                    stage='production'
                )
                
                logger.info(f"Production deployment successful: {model_name} v{model_version}")
            else:
                logger.error(f"Production deployment failed: {model_name} v{model_version}")
                
        except Exception as e:
            logger.error(f"Production deployment error: {e}")
    
    async def _execute_canary_deployment(self, deployment: ModelDeployment) -> bool:
        """Execute canary deployment strategy"""
        try:
            # This would integrate with your deployment infrastructure
            # For now, simulate the process
            
            # 1. Deploy to canary environment
            logger.info(f"Deploying canary: {deployment.model_id}")
            
            # 2. Route small percentage of traffic
            canary_traffic = self.deployment_config['canary_traffic_percentage']
            logger.info(f"Routing {canary_traffic}% traffic to canary")
            
            # 3. Monitor for errors
            await asyncio.sleep(30)  # Simulate monitoring period
            
            # 4. Check health metrics
            health_check_passed = await self._check_deployment_health(deployment)
            
            if health_check_passed:
                # 5. Gradually increase traffic
                logger.info("Canary health check passed, increasing traffic")
                return True
            else:
                # 6. Rollback on failure
                logger.warning("Canary health check failed, rolling back")
                await self._rollback_deployment(deployment)
                return False
                
        except Exception as e:
            logger.error(f"Canary deployment error: {e}")
            return False
    
    async def _check_deployment_health(self, deployment: ModelDeployment) -> bool:
        """Check deployment health metrics"""
        # This would check actual health metrics
        # For simulation, return True
        return True
    
    async def _rollback_deployment(self, deployment: ModelDeployment):
        """Rollback deployment to previous version"""
        logger.info(f"Rolling back deployment: {deployment.model_id}")
        # Implementation would restore previous model version
    
    # Helper methods for monitoring and metrics
    async def _get_current_performance(self) -> Optional[float]:
        """Get current model performance"""
        if self.performance_monitor:
            return await self.performance_monitor.get_current_accuracy()
        return None
    
    async def _get_baseline_performance(self) -> Optional[float]:
        """Get baseline performance for comparison"""
        if self.performance_monitor:
            return await self.performance_monitor.get_baseline_accuracy()
        return None
    
    async def _check_concept_drift(self) -> Optional[float]:
        """Check for concept drift"""
        if self.online_optimizer:
            metrics = self.online_optimizer.get_learning_metrics()
            return metrics.concept_drifts_detected / max(metrics.samples_processed, 1)
        return None
    
    async def _count_new_training_samples(self) -> int:
        """Count new training samples since last training"""
        # This would query your data pipeline
        return 5000  # Simulated value
    
    async def _get_baseline_metrics(self, model_name: str) -> Optional[Dict[str, float]]:
        """Get baseline metrics for model"""
        if self.model_registry:
            return await self.model_registry.get_latest_metrics(model_name)
        return None
    
    async def _schedule_staging_evaluation(self, model_name: str, model_version: str):
        """Schedule staging environment evaluation"""
        logger.info(f"Scheduling staging evaluation: {model_name} v{model_version}")
        # Implementation would schedule evaluation tasks
    
    async def _check_staging_evaluation(self, model_info: Dict[str, Any]) -> bool:
        """Check if staging evaluation is complete"""
        # This would check actual staging metrics
        return True  # Simulated
    
    async def _cleanup_old_artifacts(self):
        """Cleanup old training artifacts"""
        try:
            cutoff_date = datetime.now() - timedelta(days=30)
            
            # Cleanup old datasets
            for dataset_dir in self.data_path.glob("dataset_*"):
                if dataset_dir.is_dir():
                    dir_time = datetime.fromtimestamp(dataset_dir.stat().st_mtime)
                    if dir_time < cutoff_date:
                        shutil.rmtree(dataset_dir)
            
            # Cleanup old model artifacts
            for model_file in self.model_artifacts_path.glob("*.joblib"):
                file_time = datetime.fromtimestamp(model_file.stat().st_mtime)
                if file_time < cutoff_date:
                    model_file.unlink()
                    
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
    
    def get_orchestrator_status(self) -> Dict[str, Any]:
        """Get orchestrator status"""
        return {
            'active_jobs': len(self.active_jobs),
            'completed_jobs': len([j for j in self.training_history if j.status == 'completed']),
            'failed_jobs': len([j for j in self.training_history if j.status == 'failed']),
            'triggers': {name: asdict(trigger) for name, trigger in self.triggers.items()},
            'last_training_check': self.last_training_check,
            'mlflow_experiment_id': self.experiment_id
        }

# Factory function
async def create_training_orchestrator() -> ContinuousTrainingOrchestrator:
    """Create and initialize training orchestrator"""
    orchestrator = ContinuousTrainingOrchestrator()
    await orchestrator.start()
    return orchestrator

# Example usage
async def main():
    """Main function for running orchestrator"""
    orchestrator = await create_training_orchestrator()
    await orchestrator.run_orchestration_loop()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())