"""
AI Models for Vault Strategy Optimization
Extends existing ML models with vault-specific features and Gamma-inspired strategies
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import json
import logging
from datetime import datetime, timedelta
import time

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Import existing models
from dexbrain.enhanced_ml_models import (
    LSTMModel, TickRangePredictor, EnhancedMLModels, DeFiMLEngine
)

logger = logging.getLogger(__name__)

# Configure structured logging for vault operations
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/opt/dexter-ai/vault-strategy.log'),
        logging.StreamHandler()
    ]
)

class StrategyType(Enum):
    GAMMA_CONSERVATIVE = "gamma_conservative"
    GAMMA_BALANCED = "gamma_balanced"
    GAMMA_AGGRESSIVE = "gamma_aggressive"
    AI_CONSERVATIVE = "ai_conservative"
    AI_BALANCED = "ai_balanced"
    AI_AGGRESSIVE = "ai_aggressive"
    HYBRID_MANUAL_AI = "hybrid_manual_ai"
    HYBRID_AI_MANUAL = "hybrid_ai_manual"
    CUSTOM = "custom"

class AllocationMode(Enum):
    FIXED = "fixed"
    DYNAMIC = "dynamic"
    AI_MANAGED = "ai_managed"
    VOLATILITY_BASED = "volatility_based"
    VOLUME_WEIGHTED = "volume_weighted"

@dataclass
class VaultMetrics:
    """Comprehensive vault performance metrics"""
    total_value_locked: float
    total_fees_24h: float
    impermanent_loss: float
    apr: float
    sharpe_ratio: float
    max_drawdown: float
    successful_compounds: int
    ai_optimization_count: int
    capital_efficiency: float
    risk_score: float

@dataclass
class RangeMetrics:
    """Multi-range position metrics"""
    range_id: int
    tick_lower: int
    tick_upper: int
    allocation: float
    liquidity: float
    fees_collected: float
    utilization_rate: float
    capital_efficiency: float
    roi: float

@dataclass
class StrategyRecommendation:
    """AI strategy recommendation"""
    strategy_type: StrategyType
    allocation_mode: AllocationMode
    confidence_score: float
    expected_apr: float
    expected_risk: float
    position_ranges: List[Tuple[int, int, float]]  # (tick_lower, tick_upper, allocation)
    reasoning: str
    timestamp: datetime

class VaultStrategyPredictor(nn.Module):
    """
    Neural network for predicting optimal vault strategies
    Combines market data, vault metrics, and performance history
    """
    
    def __init__(self, input_dim: int = 50, hidden_dims: List[int] = [128, 64, 32]):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        
        # Feature extraction layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Strategy classification head
        self.strategy_classifier = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.ReLU(),
            nn.Linear(32, len(StrategyType))
        )
        
        # APR prediction head
        self.apr_predictor = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Risk prediction head
        self.risk_predictor = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Risk score 0-1
        )
        
        # Allocation predictor (for multi-range)
        self.allocation_predictor = nn.Sequential(
            nn.Linear(prev_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 10),  # Max 10 ranges
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        features = self.feature_extractor(x)
        
        strategy_logits = self.strategy_classifier(features)
        apr_pred = self.apr_predictor(features)
        risk_pred = self.risk_predictor(features)
        allocation_pred = self.allocation_predictor(features)
        
        return {
            'strategy_logits': strategy_logits,
            'apr_prediction': apr_pred,
            'risk_prediction': risk_pred,
            'allocation_prediction': allocation_pred
        }

class GammaStyleOptimizer:
    """
    Optimizer implementing Gamma Strategies' dual-position approach
    with AI enhancements
    """
    
    def __init__(self):
        self.volatility_window = 24  # hours
        self.rebalance_threshold = 0.1  # 10%
        self.fee_growth_model = None
        
    def optimize_dual_positions(
        self,
        current_tick: int,
        tick_spacing: int,
        pool_data: Dict,
        vault_metrics: VaultMetrics,
        risk_tolerance: float = 0.5
    ) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[float, float]]:
        """
        Optimize dual positions (base + limit) similar to Gamma's approach
        
        Returns:
            base_range: (tick_lower, tick_upper)
            limit_range: (tick_lower, tick_upper)
            allocations: (base_allocation, limit_allocation)
        """
        start_time = time.time()
        
        logger.info(f"[GammaStyleOptimizer] Starting dual position optimization | Current Tick: {current_tick} | Risk Tolerance: {risk_tolerance}")
        
        # Calculate volatility-based range widths
        volatility = self._calculate_volatility(pool_data)
        logger.info(f"[GammaStyleOptimizer] Calculated volatility: {volatility:.4f}")
        
        # Base position: wider, stable range
        base_width = self._calculate_base_width(volatility, risk_tolerance)
        base_lower = self._align_tick(current_tick - base_width // 2, tick_spacing)
        base_upper = self._align_tick(current_tick + base_width // 2, tick_spacing)
        
        # Limit position: narrower, active range
        limit_width = self._calculate_limit_width(volatility, risk_tolerance)
        limit_lower = self._align_tick(current_tick - limit_width // 2, tick_spacing)
        limit_upper = self._align_tick(current_tick + limit_width // 2, tick_spacing)
        
        # Calculate optimal allocations
        base_allocation, limit_allocation = self._optimize_allocations(
            volatility, vault_metrics, risk_tolerance
        )
        
        execution_time = time.time() - start_time
        
        logger.info(f"[GammaStyleOptimizer] Dual position optimization complete | "
                   f"Base Range: [{base_lower}, {base_upper}] ({base_allocation:.2%}) | "
                   f"Limit Range: [{limit_lower}, {limit_upper}] ({limit_allocation:.2%}) | "
                   f"Execution Time: {execution_time:.3f}s")
        
        return (
            (base_lower, base_upper),
            (limit_lower, limit_upper),
            (base_allocation, limit_allocation)
        )
    
    def _calculate_volatility(self, pool_data: Dict) -> float:
        """Calculate recent price volatility"""
        prices = pool_data.get('prices', [])
        if len(prices) < 2:
            return 0.2  # Default volatility
        
        returns = np.diff(np.log(prices))
        return np.std(returns) * np.sqrt(24)  # Annualized volatility
    
    def _calculate_base_width(self, volatility: float, risk_tolerance: float) -> int:
        """Calculate base position width based on volatility"""
        # Wider range for higher volatility
        base_width = int(4000 * (1 + volatility) * (1 + risk_tolerance))
        return max(1000, min(base_width, 10000))  # Clamp to reasonable range
    
    def _calculate_limit_width(self, volatility: float, risk_tolerance: float) -> int:
        """Calculate limit position width"""
        # Narrower range, inversely related to risk tolerance
        limit_width = int(600 * (1 + volatility) * (2 - risk_tolerance))
        return max(200, min(limit_width, 2000))
    
    def _optimize_allocations(
        self,
        volatility: float,
        vault_metrics: VaultMetrics,
        risk_tolerance: float
    ) -> Tuple[float, float]:
        """Optimize allocation between base and limit positions"""
        
        # Base allocation: stable, higher in volatile markets
        base_allocation = 0.8 - (volatility * 0.2) + (risk_tolerance * 0.1)
        base_allocation = max(0.6, min(base_allocation, 0.9))
        
        # Limit allocation: remainder
        limit_allocation = 1.0 - base_allocation
        
        return base_allocation, limit_allocation
    
    def _align_tick(self, tick: int, tick_spacing: int) -> int:
        """Align tick to valid tick spacing"""
        return (tick // tick_spacing) * tick_spacing

class MultiRangeOptimizer:
    """
    Advanced optimizer for multi-range strategies
    Integrates with MultiRangeManager for complex position management
    """
    
    def __init__(self, max_ranges: int = 5):
        self.max_ranges = max_ranges
        self.range_predictor = None
        self.allocation_optimizer = None
        
    def optimize_ranges(
        self,
        current_tick: int,
        tick_spacing: int,
        pool_data: Dict,
        vault_metrics: VaultMetrics,
        strategy_type: StrategyType
    ) -> List[RangeMetrics]:
        """
        Optimize multiple position ranges based on strategy type
        """
        
        if strategy_type in [StrategyType.GAMMA_CONSERVATIVE, 
                           StrategyType.GAMMA_BALANCED, 
                           StrategyType.GAMMA_AGGRESSIVE]:
            return self._optimize_gamma_style_ranges(
                current_tick, tick_spacing, pool_data, vault_metrics, strategy_type
            )
        else:
            return self._optimize_ai_ranges(
                current_tick, tick_spacing, pool_data, vault_metrics, strategy_type
            )
    
    def _optimize_gamma_style_ranges(
        self,
        current_tick: int,
        tick_spacing: int,
        pool_data: Dict,
        vault_metrics: VaultMetrics,
        strategy_type: StrategyType
    ) -> List[RangeMetrics]:
        """Optimize ranges using Gamma-style approach"""
        
        risk_map = {
            StrategyType.GAMMA_CONSERVATIVE: 0.2,
            StrategyType.GAMMA_BALANCED: 0.5,
            StrategyType.GAMMA_AGGRESSIVE: 0.8
        }
        
        risk_tolerance = risk_map.get(strategy_type, 0.5)
        
        # Use GammaStyleOptimizer for dual positions
        gamma_optimizer = GammaStyleOptimizer()
        base_range, limit_range, allocations = gamma_optimizer.optimize_dual_positions(
            current_tick, tick_spacing, pool_data, vault_metrics, risk_tolerance
        )
        
        ranges = [
            RangeMetrics(
                range_id=0,
                tick_lower=base_range[0],
                tick_upper=base_range[1],
                allocation=allocations[0],
                liquidity=0,
                fees_collected=0,
                utilization_rate=0,
                capital_efficiency=0,
                roi=0
            ),
            RangeMetrics(
                range_id=1,
                tick_lower=limit_range[0],
                tick_upper=limit_range[1],
                allocation=allocations[1],
                liquidity=0,
                fees_collected=0,
                utilization_rate=0,
                capital_efficiency=0,
                roi=0
            )
        ]
        
        return ranges
    
    def _optimize_ai_ranges(
        self,
        current_tick: int,
        tick_spacing: int,
        pool_data: Dict,
        vault_metrics: VaultMetrics,
        strategy_type: StrategyType
    ) -> List[RangeMetrics]:
        """Optimize ranges using AI models"""
        
        # This would use trained ML models to determine optimal ranges
        # For now, create adaptive ranges based on volatility and volume
        
        volatility = self._calculate_volatility(pool_data)
        volume = pool_data.get('volume_24h', 0)
        
        ranges = []
        num_ranges = min(3, self.max_ranges)  # Start with 3 ranges for AI strategies
        
        for i in range(num_ranges):
            # Create ranges at different distances from current price
            distance_multiplier = (i + 1) * 0.5
            width = int(1000 * distance_multiplier * (1 + volatility))
            
            tick_lower = self._align_tick(
                current_tick - width // 2, tick_spacing
            )
            tick_upper = self._align_tick(
                current_tick + width // 2, tick_spacing
            )
            
            # Allocate more to closer ranges
            allocation = 0.5 / (i + 1) if i < num_ranges - 1 else 0.2
            
            ranges.append(RangeMetrics(
                range_id=i,
                tick_lower=tick_lower,
                tick_upper=tick_upper,
                allocation=allocation,
                liquidity=0,
                fees_collected=0,
                utilization_rate=0,
                capital_efficiency=0,
                roi=0
            ))
        
        return ranges
    
    def _calculate_volatility(self, pool_data: Dict) -> float:
        """Calculate recent price volatility"""
        prices = pool_data.get('prices', [])
        if len(prices) < 2:
            return 0.2
        
        returns = np.diff(np.log(prices))
        return np.std(returns) * np.sqrt(24)
    
    def _align_tick(self, tick: int, tick_spacing: int) -> int:
        """Align tick to valid tick spacing"""
        return (tick // tick_spacing) * tick_spacing

class VaultMLEngine:
    """
    Main ML engine for vault strategy optimization
    Integrates multiple models and optimizers
    """
    
    def __init__(self):
        self.strategy_predictor = VaultStrategyPredictor()
        self.gamma_optimizer = GammaStyleOptimizer()
        self.range_optimizer = MultiRangeOptimizer()
        
        # Existing models from enhanced_ml_models
        self.lstm_model = LSTMModel()
        self.tick_predictor = TickRangePredictor()
        self.defi_engine = DeFiMLEngine()
        
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def prepare_features(
        self,
        pool_data: Dict,
        vault_metrics: VaultMetrics,
        market_data: Dict
    ) -> np.ndarray:
        """
        Prepare feature vector for ML models
        Combines pool, vault, and market data
        """
        
        features = []
        
        # Pool features
        features.extend([
            pool_data.get('current_price', 0),
            pool_data.get('liquidity', 0),
            pool_data.get('volume_24h', 0),
            pool_data.get('fee_tier', 0),
            pool_data.get('tick_spacing', 0)
        ])
        
        # Vault features
        features.extend([
            vault_metrics.total_value_locked,
            vault_metrics.apr,
            vault_metrics.sharpe_ratio,
            vault_metrics.max_drawdown,
            vault_metrics.capital_efficiency,
            vault_metrics.risk_score
        ])
        
        # Market features
        features.extend([
            market_data.get('btc_price', 0),
            market_data.get('eth_price', 0),
            market_data.get('total_market_cap', 0),
            market_data.get('defi_tvl', 0),
            market_data.get('volatility_index', 0)
        ])
        
        # Technical indicators from existing models
        try:
            price_history = pool_data.get('price_history', [])
            if len(price_history) >= 20:
                # Use existing tick predictor features
                tick_features = self.tick_predictor.engineer_features(
                    pd.DataFrame({'price': price_history})
                )
                features.extend(tick_features[-10:])  # Last 10 features
            else:
                features.extend([0] * 10)  # Padding
        except:
            features.extend([0] * 10)
        
        # Ensure fixed feature size
        target_size = 50
        if len(features) > target_size:
            features = features[:target_size]
        else:
            features.extend([0] * (target_size - len(features)))
        
        return np.array(features, dtype=np.float32)
    
    def predict_strategy(
        self,
        pool_data: Dict,
        vault_metrics: VaultMetrics,
        market_data: Dict
    ) -> StrategyRecommendation:
        """
        Predict optimal strategy for a vault
        """
        start_time = time.time()
        
        logger.info(f"[VaultMLEngine] Starting strategy prediction | "
                   f"Current Price: ${pool_data.get('current_price', 0)} | "
                   f"Vault TVL: ${vault_metrics.total_value_locked:,.2f} | "
                   f"Current APR: {vault_metrics.apr:.2%}")
        
        if not self.is_trained:
            logger.warning("[VaultMLEngine] Model not trained, using heuristic approach")
            # Use heuristic approach if not trained
            return self._heuristic_strategy_selection(
                pool_data, vault_metrics, market_data
            )
        
        # Prepare features
        features = self.prepare_features(pool_data, vault_metrics, market_data)
        features_tensor = torch.FloatTensor(features).unsqueeze(0)
        
        # Get model predictions
        with torch.no_grad():
            predictions = self.strategy_predictor(features_tensor)
        
        # Extract predictions
        strategy_probs = torch.softmax(predictions['strategy_logits'], dim=1)
        best_strategy_idx = torch.argmax(strategy_probs).item()
        confidence = strategy_probs[0, best_strategy_idx].item()
        
        expected_apr = predictions['apr_prediction'].item()
        expected_risk = predictions['risk_prediction'].item()
        allocations = predictions['allocation_prediction'][0].numpy()
        
        # Map to strategy type
        strategy_types = list(StrategyType)
        recommended_strategy = strategy_types[best_strategy_idx]
        
        # Generate position ranges
        current_tick = pool_data.get('current_tick', 0)
        tick_spacing = pool_data.get('tick_spacing', 60)
        
        if recommended_strategy in [StrategyType.GAMMA_CONSERVATIVE,
                                   StrategyType.GAMMA_BALANCED,
                                   StrategyType.GAMMA_AGGRESSIVE]:
            # Use Gamma-style optimization
            ranges = self.range_optimizer._optimize_gamma_style_ranges(
                current_tick, tick_spacing, pool_data, vault_metrics, recommended_strategy
            )
        else:
            # Use AI optimization
            ranges = self.range_optimizer._optimize_ai_ranges(
                current_tick, tick_spacing, pool_data, vault_metrics, recommended_strategy
            )
        
        position_ranges = [
            (r.tick_lower, r.tick_upper, r.allocation) for r in ranges
        ]
        
        execution_time = time.time() - start_time
        
        logger.info(f"[VaultMLEngine] Strategy prediction complete | "
                   f"Recommended: {recommended_strategy.value} | "
                   f"Confidence: {confidence:.2%} | "
                   f"Expected APR: {expected_apr:.2%} | "
                   f"Expected Risk: {expected_risk:.2%} | "
                   f"Ranges: {len(position_ranges)} | "
                   f"Execution Time: {execution_time:.3f}s")
        
        return StrategyRecommendation(
            strategy_type=recommended_strategy,
            allocation_mode=AllocationMode.AI_MANAGED,
            confidence_score=confidence,
            expected_apr=expected_apr,
            expected_risk=expected_risk,
            position_ranges=position_ranges,
            reasoning=self._generate_reasoning(recommended_strategy, confidence),
            timestamp=datetime.now()
        )
    
    def _heuristic_strategy_selection(
        self,
        pool_data: Dict,
        vault_metrics: VaultMetrics,
        market_data: Dict
    ) -> StrategyRecommendation:
        """
        Fallback heuristic strategy selection when ML models aren't trained
        """
        
        volatility = self._calculate_volatility(pool_data)
        tvl = vault_metrics.total_value_locked
        
        # Heuristic rules
        if tvl > 1000000:  # $1M+
            if volatility < 0.2:
                strategy = StrategyType.GAMMA_CONSERVATIVE
            elif volatility < 0.5:
                strategy = StrategyType.GAMMA_BALANCED
            else:
                strategy = StrategyType.AI_AGGRESSIVE
        else:
            if volatility < 0.3:
                strategy = StrategyType.GAMMA_BALANCED
            else:
                strategy = StrategyType.AI_BALANCED
        
        # Generate ranges
        current_tick = pool_data.get('current_tick', 0)
        tick_spacing = pool_data.get('tick_spacing', 60)
        
        ranges = self.range_optimizer.optimize_ranges(
            current_tick, tick_spacing, pool_data, vault_metrics, strategy
        )
        
        position_ranges = [
            (r.tick_lower, r.tick_upper, r.allocation) for r in ranges
        ]
        
        return StrategyRecommendation(
            strategy_type=strategy,
            allocation_mode=AllocationMode.DYNAMIC,
            confidence_score=0.7,  # Moderate confidence for heuristics
            expected_apr=0.15,     # Conservative estimate
            expected_risk=volatility,
            position_ranges=position_ranges,
            reasoning=f"Heuristic selection based on volatility ({volatility:.2f}) and TVL (${tvl:,.0f})",
            timestamp=datetime.now()
        )
    
    def train_models(
        self,
        training_data: List[Dict],
        epochs: int = 100,
        batch_size: int = 32
    ):
        """
        Train the ML models on historical data
        """
        
        logger.info(f"Training vault strategy models on {len(training_data)} samples")
        
        # Prepare training data
        X = []
        y_strategy = []
        y_apr = []
        y_risk = []
        
        for sample in training_data:
            features = self.prepare_features(
                sample['pool_data'],
                VaultMetrics(**sample['vault_metrics']),
                sample['market_data']
            )
            X.append(features)
            
            y_strategy.append(sample['target_strategy'])
            y_apr.append(sample['actual_apr'])
            y_risk.append(sample['actual_risk'])
        
        X = np.array(X)
        y_strategy = np.array(y_strategy)
        y_apr = np.array(y_apr)
        y_risk = np.array(y_risk)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_scaled)
        y_strategy_tensor = torch.LongTensor(y_strategy)
        y_apr_tensor = torch.FloatTensor(y_apr).unsqueeze(1)
        y_risk_tensor = torch.FloatTensor(y_risk).unsqueeze(1)
        
        # Create data loader
        dataset = torch.utils.data.TensorDataset(
            X_tensor, y_strategy_tensor, y_apr_tensor, y_risk_tensor
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        
        # Setup optimizer and loss functions
        optimizer = optim.Adam(self.strategy_predictor.parameters(), lr=0.001)
        strategy_loss_fn = nn.CrossEntropyLoss()
        regression_loss_fn = nn.MSELoss()
        
        # Training loop
        self.strategy_predictor.train()
        
        for epoch in range(epochs):
            total_loss = 0
            for batch_x, batch_y_strategy, batch_y_apr, batch_y_risk in dataloader:
                optimizer.zero_grad()
                
                predictions = self.strategy_predictor(batch_x)
                
                # Calculate losses
                strategy_loss = strategy_loss_fn(
                    predictions['strategy_logits'], batch_y_strategy
                )
                apr_loss = regression_loss_fn(
                    predictions['apr_prediction'], batch_y_apr
                )
                risk_loss = regression_loss_fn(
                    predictions['risk_prediction'], batch_y_risk
                )
                
                # Combined loss
                loss = strategy_loss + apr_loss + risk_loss
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}, Loss: {total_loss:.4f}")
        
        self.is_trained = True
        logger.info("Model training completed")
    
    def _calculate_volatility(self, pool_data: Dict) -> float:
        """Calculate price volatility"""
        prices = pool_data.get('prices', [])
        if len(prices) < 2:
            return 0.2
        
        returns = np.diff(np.log(prices))
        return np.std(returns) * np.sqrt(24)
    
    def _generate_reasoning(self, strategy: StrategyType, confidence: float) -> str:
        """Generate human-readable reasoning for strategy selection"""
        
        base_reasoning = {
            StrategyType.GAMMA_CONSERVATIVE: "Low volatility market conditions favor stable, wide-range positions",
            StrategyType.GAMMA_BALANCED: "Moderate market conditions suit balanced dual-position approach",
            StrategyType.GAMMA_AGGRESSIVE: "High volatility enables profitable narrow-range strategies",
            StrategyType.AI_CONSERVATIVE: "AI models recommend conservative approach based on risk factors",
            StrategyType.AI_BALANCED: "AI optimization suggests balanced multi-range strategy",
            StrategyType.AI_AGGRESSIVE: "ML models predict high-reward opportunity with managed risk"
        }
        
        reasoning = base_reasoning.get(
            strategy, 
            "Custom strategy selected based on unique market conditions"
        )
        
        confidence_desc = "high" if confidence > 0.8 else "moderate" if confidence > 0.6 else "low"
        
        return f"{reasoning}. Model confidence: {confidence_desc} ({confidence:.1%})"

class VaultPerformanceTracker:
    """
    Tracks and analyzes vault performance for model training and optimization
    """
    
    def __init__(self):
        self.performance_history = {}
        self.benchmark_data = {}
        
    def record_performance(
        self,
        vault_address: str,
        timestamp: datetime,
        metrics: VaultMetrics,
        strategy_type: StrategyType,
        ranges: List[RangeMetrics]
    ):
        """Record vault performance data"""
        
        if vault_address not in self.performance_history:
            self.performance_history[vault_address] = []
        
        record = {
            'timestamp': timestamp,
            'metrics': metrics,
            'strategy_type': strategy_type,
            'ranges': ranges,
            'total_ranges': len(ranges),
            'active_ranges': len([r for r in ranges if r.liquidity > 0])
        }
        
        self.performance_history[vault_address].append(record)
    
    def calculate_strategy_performance(
        self,
        vault_address: str,
        period_days: int = 30
    ) -> Dict:
        """Calculate performance metrics for a strategy over time"""
        
        if vault_address not in self.performance_history:
            return {}
        
        history = self.performance_history[vault_address]
        cutoff_date = datetime.now() - timedelta(days=period_days)
        
        recent_records = [
            r for r in history 
            if r['timestamp'] >= cutoff_date
        ]
        
        if len(recent_records) < 2:
            return {}
        
        # Calculate metrics
        aprs = [r['metrics'].apr for r in recent_records]
        sharpe_ratios = [r['metrics'].sharpe_ratio for r in recent_records]
        drawdowns = [r['metrics'].max_drawdown for r in recent_records]
        
        return {
            'avg_apr': np.mean(aprs),
            'apr_volatility': np.std(aprs),
            'avg_sharpe': np.mean(sharpe_ratios),
            'max_drawdown': max(drawdowns),
            'total_compounds': recent_records[-1]['metrics'].successful_compounds,
            'ai_optimizations': recent_records[-1]['metrics'].ai_optimization_count,
            'consistency_score': 1 / (1 + np.std(aprs))  # Higher for consistent performance
        }
    
    def generate_training_data(
        self,
        min_history_days: int = 7
    ) -> List[Dict]:
        """Generate training data from performance history"""
        
        training_samples = []
        
        for vault_address, history in self.performance_history.items():
            if len(history) < min_history_days:
                continue
            
            for i in range(len(history) - 1):
                current = history[i]
                next_record = history[i + 1]
                
                # Calculate actual performance
                time_diff = (next_record['timestamp'] - current['timestamp']).total_seconds() / 3600
                if time_diff == 0:
                    continue
                
                actual_apr = (
                    next_record['metrics'].total_value_locked / 
                    current['metrics'].total_value_locked - 1
                ) * (365 * 24 / time_diff)
                
                actual_risk = current['metrics'].max_drawdown
                
                sample = {
                    'pool_data': {
                        'current_tick': 0,  # Would be populated from actual data
                        'liquidity': current['metrics'].total_value_locked,
                        'volume_24h': current['metrics'].total_fees_24h * 100,
                        'tick_spacing': 60,
                        'prices': []  # Would be populated
                    },
                    'vault_metrics': {
                        'total_value_locked': current['metrics'].total_value_locked,
                        'total_fees_24h': current['metrics'].total_fees_24h,
                        'impermanent_loss': current['metrics'].impermanent_loss,
                        'apr': current['metrics'].apr,
                        'sharpe_ratio': current['metrics'].sharpe_ratio,
                        'max_drawdown': current['metrics'].max_drawdown,
                        'successful_compounds': current['metrics'].successful_compounds,
                        'ai_optimization_count': current['metrics'].ai_optimization_count,
                        'capital_efficiency': current['metrics'].capital_efficiency,
                        'risk_score': current['metrics'].risk_score
                    },
                    'market_data': {
                        'btc_price': 50000,  # Would be populated from actual data
                        'eth_price': 3000,
                        'total_market_cap': 2000000000000,
                        'defi_tvl': 100000000000,
                        'volatility_index': 0.3
                    },
                    'target_strategy': list(StrategyType).index(current['strategy_type']),
                    'actual_apr': actual_apr,
                    'actual_risk': actual_risk
                }
                
                training_samples.append(sample)
        
        return training_samples

# Integration functions for existing Dexter systems

def integrate_with_compound_service():
    """
    Integration point with existing compound service
    """
    
    vault_engine = VaultMLEngine()
    
    def get_vault_strategy_recommendation(
        vault_address: str,
        pool_data: Dict,
        vault_metrics: Dict,
        market_data: Dict
    ) -> Dict:
        """
        Function to be called by compound service for strategy recommendations
        """
        
        # Convert dict to VaultMetrics
        metrics = VaultMetrics(**vault_metrics)
        
        # Get recommendation
        recommendation = vault_engine.predict_strategy(
            pool_data, metrics, market_data
        )
        
        return {
            'strategy_type': recommendation.strategy_type.value,
            'allocation_mode': recommendation.allocation_mode.value,
            'confidence_score': recommendation.confidence_score,
            'expected_apr': recommendation.expected_apr,
            'expected_risk': recommendation.expected_risk,
            'position_ranges': recommendation.position_ranges,
            'reasoning': recommendation.reasoning,
            'timestamp': recommendation.timestamp.isoformat()
        }
    
    return get_vault_strategy_recommendation

def update_claude_md_integration():
    """
    Update CLAUDE.md with vault strategy integration information
    """
    
    integration_info = """
    
## AI Vault Strategy Integration

### Enhanced ML Models for Vault Management
- **VaultStrategyPredictor**: Neural network for optimal strategy selection
- **GammaStyleOptimizer**: Dual-position optimization inspired by Gamma Strategies  
- **MultiRangeOptimizer**: Advanced multi-range position management
- **VaultMLEngine**: Main engine integrating all models and optimizers

### Strategy Types Supported
- Gamma-style strategies (Conservative, Balanced, Aggressive)
- AI-optimized strategies with dynamic rebalancing
- Hybrid manual/AI approaches for maximum flexibility
- Custom strategies for specialized use cases

### Key Features
- Real-time strategy recommendations with confidence scores
- Integration with existing LSTM and TickRangePredictor models
- Performance tracking and continuous learning
- Gamma-inspired dual position management
- Multi-range allocation optimization

### Integration Points
- Compound Service: `get_vault_strategy_recommendation()`
- Strategy Manager: AI recommendation application
- Performance Tracker: Historical data for model training
- Range Manager: Multi-range position optimization
"""
    
    return integration_info

if __name__ == "__main__":
    # Example usage
    engine = VaultMLEngine()
    
    # Sample data
    pool_data = {
        'current_tick': 100000,
        'current_price': 3000,
        'liquidity': 1000000,
        'volume_24h': 5000000,
        'fee_tier': 3000,
        'tick_spacing': 60,
        'prices': list(range(2900, 3100, 10))
    }
    
    vault_metrics = VaultMetrics(
        total_value_locked=2000000,
        total_fees_24h=5000,
        impermanent_loss=0.02,
        apr=0.15,
        sharpe_ratio=1.2,
        max_drawdown=0.05,
        successful_compounds=45,
        ai_optimization_count=12,
        capital_efficiency=0.85,
        risk_score=0.3
    )
    
    market_data = {
        'btc_price': 50000,
        'eth_price': 3000,
        'total_market_cap': 2000000000000,
        'defi_tvl': 100000000000,
        'volatility_index': 0.3
    }
    
    # Get strategy recommendation
    recommendation = engine.predict_strategy(pool_data, vault_metrics, market_data)
    
    print(f"Recommended Strategy: {recommendation.strategy_type}")
    print(f"Confidence: {recommendation.confidence_score:.2%}")
    print(f"Expected APR: {recommendation.expected_apr:.2%}")
    print(f"Position Ranges: {recommendation.position_ranges}")
    print(f"Reasoning: {recommendation.reasoning}")