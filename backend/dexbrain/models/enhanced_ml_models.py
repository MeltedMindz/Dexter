"""
Enhanced ML Models for Uniswap Concentrated Liquidity Optimization
Implements advanced feature extraction and multi-output prediction models
"""

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, RobustScaler
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
from pathlib import Path
import joblib
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta
from ..config import Config

logger = logging.getLogger(__name__)

@dataclass
class UniswapFeatures:
    """Comprehensive Uniswap feature set for ML models"""
    
    # Basic Pool Metrics
    pool_tvl: float
    volume_24h: float
    fee_tier: int
    token0_reserve: float
    token1_reserve: float
    
    # Price and Volatility Metrics
    current_price: float
    price_volatility_1h: float
    price_volatility_24h: float
    price_volatility_7d: float
    price_change_24h: float
    
    # Liquidity Distribution Metrics
    current_tick: int
    tick_spacing: int
    sqrt_price: float
    liquidity_distribution: float  # Concentration metric
    active_liquidity_ratio: float
    
    # Position and Trading Metrics
    position_count: int
    average_position_size: float
    whale_concentration: float  # Top 10 positions % of total
    volume_to_tvl_ratio: float
    
    # Advanced Metrics
    fee_growth_global: float
    tick_bitmap_density: float
    price_impact_estimate: float
    arbitrage_opportunity_score: float
    
    # Time-based Features
    hour_of_day: int
    day_of_week: int
    days_since_pool_creation: int
    
    # Cross-pool Features
    correlation_with_eth: float
    correlation_with_btc: float
    relative_volume_rank: float
    
    def to_array(self) -> np.ndarray:
        """Convert features to numpy array for ML training"""
        return np.array([
            self.pool_tvl,
            self.volume_24h,
            self.fee_tier,
            self.token0_reserve,
            self.token1_reserve,
            self.current_price,
            self.price_volatility_1h,
            self.price_volatility_24h,
            self.price_volatility_7d,
            self.price_change_24h,
            self.current_tick,
            self.tick_spacing,
            self.sqrt_price,
            self.liquidity_distribution,
            self.active_liquidity_ratio,
            self.position_count,
            self.average_position_size,
            self.whale_concentration,
            self.volume_to_tvl_ratio,
            self.fee_growth_global,
            self.tick_bitmap_density,
            self.price_impact_estimate,
            self.arbitrage_opportunity_score,
            self.hour_of_day,
            self.day_of_week,
            self.days_since_pool_creation,
            self.correlation_with_eth,
            self.correlation_with_btc,
            self.relative_volume_rank
        ])
    
    @classmethod
    def get_feature_names(cls) -> List[str]:
        """Get ordered list of feature names"""
        return [
            'pool_tvl', 'volume_24h', 'fee_tier', 'token0_reserve', 'token1_reserve',
            'current_price', 'price_volatility_1h', 'price_volatility_24h', 'price_volatility_7d', 'price_change_24h',
            'current_tick', 'tick_spacing', 'sqrt_price', 'liquidity_distribution', 'active_liquidity_ratio',
            'position_count', 'average_position_size', 'whale_concentration', 'volume_to_tvl_ratio',
            'fee_growth_global', 'tick_bitmap_density', 'price_impact_estimate', 'arbitrage_opportunity_score',
            'hour_of_day', 'day_of_week', 'days_since_pool_creation',
            'correlation_with_eth', 'correlation_with_btc', 'relative_volume_rank'
        ]


class LSTMConcentratedLiquidityModel(nn.Module):
    """LSTM model for time-series prediction of liquidity position performance"""
    
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, 
                 output_size: int = 3, dropout: float = 0.2):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)
        
        # Output layers for multi-output prediction
        self.fc_layers = nn.ModuleDict({
            'apr_prediction': nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, 1)
            ),
            'impermanent_loss': nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, 1)
            ),
            'optimal_range': nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, 2)  # Lower and upper tick
            )
        })
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size = x.size(0)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Apply attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Use last time step
        last_output = attn_out[:, -1, :]
        
        # Multi-output prediction
        outputs = {}
        for name, layer in self.fc_layers.items():
            outputs[name] = layer(last_output)
            
        return outputs


class TickRangePredictor(nn.Module):
    """Specialized model for predicting optimal tick ranges"""
    
    def __init__(self, input_size: int, hidden_sizes: Tuple[int, ...] = (256, 128, 64)):
        super().__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(0.3)
            ])
            prev_size = hidden_size
        
        # Separate heads for lower and upper ticks
        self.feature_extractor = nn.Sequential(*layers)
        self.lower_tick_head = nn.Linear(prev_size, 1)
        self.upper_tick_head = nn.Linear(prev_size, 1)
        self.confidence_head = nn.Linear(prev_size, 1)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.feature_extractor(x)
        
        return {
            'lower_tick': self.lower_tick_head(features),
            'upper_tick': self.upper_tick_head(features),
            'confidence': torch.sigmoid(self.confidence_head(features))
        }


class ImpermanentLossForecaster(nn.Module):
    """Model for predicting impermanent loss under different scenarios"""
    
    def __init__(self, input_size: int, scenario_count: int = 5):
        super().__init__()
        
        self.scenario_count = scenario_count
        
        # Shared feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2)
        )
        
        # Scenario-specific heads
        self.scenario_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            ) for _ in range(scenario_count)
        ])
        
        # Probability weights for scenarios
        self.scenario_weights = nn.Sequential(
            nn.Linear(128, scenario_count),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.feature_extractor(x)
        
        # Get predictions for each scenario
        scenario_predictions = torch.stack([
            head(features) for head in self.scenario_heads
        ], dim=2)
        
        # Get scenario probabilities
        scenario_probs = self.scenario_weights(features)
        
        # Weighted average prediction
        expected_il = torch.sum(scenario_predictions.squeeze(1) * scenario_probs, dim=1, keepdim=True)
        
        return {
            'expected_impermanent_loss': expected_il,
            'scenario_predictions': scenario_predictions.squeeze(1),
            'scenario_probabilities': scenario_probs
        }


class FeeEarningsPredictor(nn.Module):
    """Model for predicting fee earnings over different time horizons"""
    
    def __init__(self, input_size: int, time_horizons: List[int] = [1, 7, 30]):
        super().__init__()
        
        self.time_horizons = time_horizons
        
        # Shared layers
        self.shared_layers = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2)
        )
        
        # Time-horizon specific heads
        self.horizon_heads = nn.ModuleDict({
            f'fees_{horizon}d': nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            ) for horizon in time_horizons
        })
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.shared_layers(x)
        
        predictions = {}
        for horizon_key, head in self.horizon_heads.items():
            predictions[horizon_key] = head(features)
            
        return predictions


class UniswapLPOptimizer:
    """
    Comprehensive ML-driven optimizer for Uniswap concentrated liquidity positions
    """
    
    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path or Config.MODEL_STORAGE_PATH / 'uniswap_optimizer'
        self.model_path.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Feature scaling
        self.feature_scaler = RobustScaler()  # More robust to outliers than StandardScaler
        self.target_scalers = {
            'apr': StandardScaler(),
            'il': StandardScaler(),
            'fees': StandardScaler()
        }
        
        # Models
        self.lstm_model: Optional[LSTMConcentratedLiquidityModel] = None
        self.tick_predictor: Optional[TickRangePredictor] = None
        self.il_forecaster: Optional[ImpermanentLossForecaster] = None
        self.fee_predictor: Optional[FeeEarningsPredictor] = None
        
        # Training history
        self.training_history: Dict[str, Dict[str, List[float]]] = {
            'lstm': {'loss': [], 'val_loss': []},
            'tick_predictor': {'loss': [], 'val_loss': []},
            'il_forecaster': {'loss': [], 'val_loss': []},
            'fee_predictor': {'loss': [], 'val_loss': []}
        }
        
        self.load_or_initialize_models()
    
    def load_or_initialize_models(self, input_size: int = 29) -> None:
        """Load existing models or create new ones"""
        try:
            # Try to load existing models
            checkpoint_path = self.model_path / 'checkpoint.pth'
            if checkpoint_path.exists():
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                
                self.lstm_model = checkpoint['lstm_model']
                self.tick_predictor = checkpoint['tick_predictor']
                self.il_forecaster = checkpoint['il_forecaster']
                self.fee_predictor = checkpoint['fee_predictor']
                self.training_history = checkpoint.get('history', self.training_history)
                
                # Load scalers
                scaler_path = self.model_path / 'scalers.pkl'
                if scaler_path.exists():
                    scalers = joblib.load(scaler_path)
                    self.feature_scaler = scalers['feature_scaler']
                    self.target_scalers = scalers['target_scalers']
                
                logger.info("Loaded existing models from checkpoint")
            else:
                raise FileNotFoundError("No checkpoint found")
                
        except (FileNotFoundError, KeyError) as e:
            logger.info(f"Initializing new models: {e}")
            
            # Initialize new models
            self.lstm_model = LSTMConcentratedLiquidityModel(input_size=input_size)
            self.tick_predictor = TickRangePredictor(input_size=input_size)
            self.il_forecaster = ImpermanentLossForecaster(input_size=input_size)
            self.fee_predictor = FeeEarningsPredictor(input_size=input_size)
            
        # Move models to device
        self.lstm_model.to(self.device)
        self.tick_predictor.to(self.device)
        self.il_forecaster.to(self.device)
        self.fee_predictor.to(self.device)
        
        self.save_models()
    
    def save_models(self) -> None:
        """Save all models and scalers"""
        checkpoint = {
            'lstm_model': self.lstm_model,
            'tick_predictor': self.tick_predictor,
            'il_forecaster': self.il_forecaster,
            'fee_predictor': self.fee_predictor,
            'history': self.training_history
        }
        
        checkpoint_path = self.model_path / 'checkpoint.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save scalers
        scalers = {
            'feature_scaler': self.feature_scaler,
            'target_scalers': self.target_scalers
        }
        scaler_path = self.model_path / 'scalers.pkl'
        joblib.dump(scalers, scaler_path)
        
        logger.info("Models and scalers saved successfully")
    
    def extract_features_from_position_data(self, position_data: Dict[str, Any]) -> UniswapFeatures:
        """
        Extract comprehensive features from position/pool data
        
        Args:
            position_data: Dictionary containing pool and position information
            
        Returns:
            UniswapFeatures object with all extracted features
        """
        now = datetime.now()
        
        # Extract basic features with safe defaults
        features = UniswapFeatures(
            pool_tvl=float(position_data.get('pool_tvl', 0)),
            volume_24h=float(position_data.get('volume_24h', 0)),
            fee_tier=int(position_data.get('fee_tier', 3000)),
            token0_reserve=float(position_data.get('token0_reserve', 0)),
            token1_reserve=float(position_data.get('token1_reserve', 0)),
            
            current_price=float(position_data.get('current_price', 1)),
            price_volatility_1h=float(position_data.get('price_volatility_1h', 0)),
            price_volatility_24h=float(position_data.get('price_volatility_24h', 0)),
            price_volatility_7d=float(position_data.get('price_volatility_7d', 0)),
            price_change_24h=float(position_data.get('price_change_24h', 0)),
            
            current_tick=int(position_data.get('current_tick', 0)),
            tick_spacing=int(position_data.get('tick_spacing', 60)),
            sqrt_price=float(position_data.get('sqrt_price', 1)),
            liquidity_distribution=float(position_data.get('liquidity_distribution', 0.5)),
            active_liquidity_ratio=float(position_data.get('active_liquidity_ratio', 0.5)),
            
            position_count=int(position_data.get('position_count', 1)),
            average_position_size=float(position_data.get('average_position_size', 1000)),
            whale_concentration=float(position_data.get('whale_concentration', 0.1)),
            volume_to_tvl_ratio=float(position_data.get('volume_to_tvl_ratio', 0.1)),
            
            fee_growth_global=float(position_data.get('fee_growth_global', 0)),
            tick_bitmap_density=float(position_data.get('tick_bitmap_density', 0.1)),
            price_impact_estimate=float(position_data.get('price_impact_estimate', 0.01)),
            arbitrage_opportunity_score=float(position_data.get('arbitrage_opportunity_score', 0)),
            
            hour_of_day=now.hour,
            day_of_week=now.weekday(),
            days_since_pool_creation=int(position_data.get('days_since_pool_creation', 100)),
            
            correlation_with_eth=float(position_data.get('correlation_with_eth', 0.5)),
            correlation_with_btc=float(position_data.get('correlation_with_btc', 0.3)),
            relative_volume_rank=float(position_data.get('relative_volume_rank', 0.5))
        )
        
        return features
    
    def predict_optimal_position(self, pool_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict optimal liquidity position parameters
        
        Args:
            pool_data: Pool information dictionary
            
        Returns:
            Dictionary with optimization recommendations
        """
        # Extract features
        features = self.extract_features_from_position_data(pool_data)
        feature_array = features.to_array().reshape(1, -1)
        
        # Scale features
        feature_array_scaled = self.feature_scaler.transform(feature_array)
        feature_tensor = torch.FloatTensor(feature_array_scaled).to(self.device)
        
        # Get predictions from all models
        predictions = {}
        
        # LSTM predictions (if we have time-series data)
        # For now, we'll use current features repeated to simulate time series
        lstm_input = feature_tensor.unsqueeze(0).repeat(1, 10, 1)  # 10 time steps
        with torch.no_grad():
            lstm_outputs = self.lstm_model(lstm_input)
            predictions['lstm'] = {k: v.cpu().numpy()[0] for k, v in lstm_outputs.items()}
        
        # Tick range prediction
        with torch.no_grad():
            tick_outputs = self.tick_predictor(feature_tensor)
            predictions['optimal_ticks'] = {k: v.cpu().numpy()[0] for k, v in tick_outputs.items()}
        
        # Impermanent loss forecast
        with torch.no_grad():
            il_outputs = self.il_forecaster(feature_tensor)
            predictions['impermanent_loss'] = {k: v.cpu().numpy()[0] for k, v in il_outputs.items()}
        
        # Fee earnings prediction
        with torch.no_grad():
            fee_outputs = self.fee_predictor(feature_tensor)
            predictions['fee_earnings'] = {k: v.cpu().numpy()[0] for k, v in fee_outputs.items()}
        
        # Compile recommendations
        recommendations = {
            'predicted_apr': float(predictions['lstm']['apr_prediction'][0]),
            'predicted_il': float(predictions['lstm']['impermanent_loss'][0]),
            'optimal_lower_tick': int(predictions['optimal_ticks']['lower_tick'][0]),
            'optimal_upper_tick': int(predictions['optimal_ticks']['upper_tick'][0]),
            'tick_confidence': float(predictions['optimal_ticks']['confidence'][0]),
            'expected_impermanent_loss': float(predictions['impermanent_loss']['expected_impermanent_loss'][0]),
            'fee_earnings_1d': float(predictions['fee_earnings']['fees_1d'][0]),
            'fee_earnings_7d': float(predictions['fee_earnings']['fees_7d'][0]),
            'fee_earnings_30d': float(predictions['fee_earnings']['fees_30d'][0]),
            'risk_score': float(np.mean([
                abs(predictions['impermanent_loss']['expected_impermanent_loss'][0]),
                predictions['lstm']['impermanent_loss'][0]
            ])),
            'confidence_score': float(predictions['optimal_ticks']['confidence'][0])
        }
        
        return recommendations
    
    def _train_tick_predictor(self, X_train, X_val, targets, train_indices, val_indices, epochs, batch_size, learning_rate):
        """Train the tick range predictor"""
        # Prepare targets
        lower_ticks = torch.FloatTensor([targets['lower_tick'][i] for i in train_indices]).unsqueeze(1).to(self.device)
        upper_ticks = torch.FloatTensor([targets['upper_tick'][i] for i in train_indices]).unsqueeze(1).to(self.device)
        
        val_lower_ticks = torch.FloatTensor([targets['lower_tick'][i] for i in val_indices]).unsqueeze(1).to(self.device)
        val_upper_ticks = torch.FloatTensor([targets['upper_tick'][i] for i in val_indices]).unsqueeze(1).to(self.device)
        
        # Optimizer and loss
        optimizer = optim.Adam(self.tick_predictor.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            self.tick_predictor.train()
            epoch_loss = 0
            
            # Training loop
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_lower = lower_ticks[i:i+batch_size]
                batch_upper = upper_ticks[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = self.tick_predictor(batch_X)
                
                loss_lower = criterion(outputs['lower_tick'], batch_lower)
                loss_upper = criterion(outputs['upper_tick'], batch_upper)
                total_loss = loss_lower + loss_upper
                
                total_loss.backward()
                optimizer.step()
                epoch_loss += total_loss.item()
            
            # Validation
            self.tick_predictor.eval()
            with torch.no_grad():
                val_outputs = self.tick_predictor(X_val)
                val_loss_lower = criterion(val_outputs['lower_tick'], val_lower_ticks)
                val_loss_upper = criterion(val_outputs['upper_tick'], val_upper_ticks)
                val_loss = val_loss_lower + val_loss_upper
            
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss.item()
            
            self.training_history['tick_predictor']['loss'].append(epoch_loss / len(X_train))
            self.training_history['tick_predictor']['val_loss'].append(val_loss.item())
        
        return {'final_val_loss': best_val_loss, 'epochs_trained': epochs}
    
    def _train_il_forecaster(self, X_train, X_val, targets, train_indices, val_indices, epochs, batch_size, learning_rate):
        """Train the impermanent loss forecaster"""
        il_targets = torch.FloatTensor([targets['il'][i] for i in train_indices]).unsqueeze(1).to(self.device)
        val_il_targets = torch.FloatTensor([targets['il'][i] for i in val_indices]).unsqueeze(1).to(self.device)
        
        optimizer = optim.Adam(self.il_forecaster.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            self.il_forecaster.train()
            epoch_loss = 0
            
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_targets = il_targets[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = self.il_forecaster(batch_X)
                loss = criterion(outputs['expected_impermanent_loss'], batch_targets)
                
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            # Validation
            self.il_forecaster.eval()
            with torch.no_grad():
                val_outputs = self.il_forecaster(X_val)
                val_loss = criterion(val_outputs['expected_impermanent_loss'], val_il_targets)
            
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss.item()
            
            self.training_history['il_forecaster']['loss'].append(epoch_loss / len(X_train))
            self.training_history['il_forecaster']['val_loss'].append(val_loss.item())
        
        return {'final_val_loss': best_val_loss, 'epochs_trained': epochs}
    
    def _train_fee_predictor(self, X_train, X_val, targets, train_indices, val_indices, epochs, batch_size, learning_rate):
        """Train the fee earnings predictor"""
        fees_1d = torch.FloatTensor([targets['fees_1d'][i] for i in train_indices]).unsqueeze(1).to(self.device)
        fees_7d = torch.FloatTensor([targets['fees_7d'][i] for i in train_indices]).unsqueeze(1).to(self.device)
        fees_30d = torch.FloatTensor([targets['fees_30d'][i] for i in train_indices]).unsqueeze(1).to(self.device)
        
        val_fees_1d = torch.FloatTensor([targets['fees_1d'][i] for i in val_indices]).unsqueeze(1).to(self.device)
        val_fees_7d = torch.FloatTensor([targets['fees_7d'][i] for i in val_indices]).unsqueeze(1).to(self.device)
        val_fees_30d = torch.FloatTensor([targets['fees_30d'][i] for i in val_indices]).unsqueeze(1).to(self.device)
        
        optimizer = optim.Adam(self.fee_predictor.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            self.fee_predictor.train()
            epoch_loss = 0
            
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_1d = fees_1d[i:i+batch_size]
                batch_7d = fees_7d[i:i+batch_size]
                batch_30d = fees_30d[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = self.fee_predictor(batch_X)
                
                loss_1d = criterion(outputs['fees_1d'], batch_1d)
                loss_7d = criterion(outputs['fees_7d'], batch_7d)
                loss_30d = criterion(outputs['fees_30d'], batch_30d)
                total_loss = loss_1d + loss_7d + loss_30d
                
                total_loss.backward()
                optimizer.step()
                epoch_loss += total_loss.item()
            
            # Validation
            self.fee_predictor.eval()
            with torch.no_grad():
                val_outputs = self.fee_predictor(X_val)
                val_loss = (
                    criterion(val_outputs['fees_1d'], val_fees_1d) +
                    criterion(val_outputs['fees_7d'], val_fees_7d) +
                    criterion(val_outputs['fees_30d'], val_fees_30d)
                )
            
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss.item()
            
            self.training_history['fee_predictor']['loss'].append(epoch_loss / len(X_train))
            self.training_history['fee_predictor']['val_loss'].append(val_loss.item())
        
        return {'final_val_loss': best_val_loss, 'epochs_trained': epochs}
    
    def _train_lstm_model(self, X_train, X_val, targets, train_indices, val_indices, epochs, batch_size, learning_rate):
        """Train the LSTM model with sequence data"""
        # Create sequence data by repeating features (in production, use real time series)
        sequence_length = 10
        X_train_seq = X_train.unsqueeze(1).repeat(1, sequence_length, 1)
        X_val_seq = X_val.unsqueeze(1).repeat(1, sequence_length, 1)
        
        # Prepare targets
        apr_targets = torch.FloatTensor([targets['apr'][i] for i in train_indices]).unsqueeze(1).to(self.device)
        il_targets = torch.FloatTensor([targets['il'][i] for i in train_indices]).unsqueeze(1).to(self.device)
        
        val_apr_targets = torch.FloatTensor([targets['apr'][i] for i in val_indices]).unsqueeze(1).to(self.device)
        val_il_targets = torch.FloatTensor([targets['il'][i] for i in val_indices]).unsqueeze(1).to(self.device)
        
        optimizer = optim.Adam(self.lstm_model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            self.lstm_model.train()
            epoch_loss = 0
            
            for i in range(0, len(X_train_seq), batch_size):
                batch_X = X_train_seq[i:i+batch_size]
                batch_apr = apr_targets[i:i+batch_size]
                batch_il = il_targets[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = self.lstm_model(batch_X)
                
                loss_apr = criterion(outputs['apr_prediction'], batch_apr)
                loss_il = criterion(outputs['impermanent_loss'], batch_il)
                total_loss = loss_apr + loss_il
                
                total_loss.backward()
                optimizer.step()
                epoch_loss += total_loss.item()
            
            # Validation
            self.lstm_model.eval()
            with torch.no_grad():
                val_outputs = self.lstm_model(X_val_seq)
                val_loss = (
                    criterion(val_outputs['apr_prediction'], val_apr_targets) +
                    criterion(val_outputs['impermanent_loss'], val_il_targets)
                )
            
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss.item()
            
            self.training_history['lstm']['loss'].append(epoch_loss / len(X_train_seq))
            self.training_history['lstm']['val_loss'].append(val_loss.item())
        
        return {'final_val_loss': best_val_loss, 'epochs_trained': epochs}
    
    def train_models(self, training_data: List[Dict[str, Any]], epochs: int = 50, batch_size: int = 32, learning_rate: float = 0.001) -> Dict[str, Any]:
        """
        Train all models on provided data
        
        Args:
            training_data: List of training examples with features and targets
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimization
            
        Returns:
            Training results
        """
        logger.info(f"Training models on {len(training_data)} examples")
        
        if len(training_data) < 10:
            logger.warning(f"Insufficient training data: {len(training_data)} samples")
            return {'status': 'insufficient_data', 'training_samples': len(training_data)}
        
        # Prepare training data
        features_list = []
        targets = {
            'apr': [], 'il': [], 'lower_tick': [], 'upper_tick': [], 
            'fees_1d': [], 'fees_7d': [], 'fees_30d': []
        }
        
        for example in training_data:
            try:
                features = self.extract_features_from_position_data(example)
                features_list.append(features.to_array())
                
                # Extract targets with better defaults
                targets['apr'].append(float(example.get('target_apr', example.get('apr', 0.1))))
                targets['il'].append(float(example.get('target_il', example.get('impermanent_loss', 0.05))))
                targets['lower_tick'].append(int(example.get('target_lower_tick', example.get('lower_tick', -1000))))
                targets['upper_tick'].append(int(example.get('target_upper_tick', example.get('upper_tick', 1000))))
                targets['fees_1d'].append(float(example.get('target_fees_1d', example.get('daily_fees', 10.0))))
                targets['fees_7d'].append(float(example.get('target_fees_7d', example.get('weekly_fees', 70.0))))
                targets['fees_30d'].append(float(example.get('target_fees_30d', example.get('monthly_fees', 300.0))))
            except Exception as e:
                logger.warning(f"Skipping invalid training example: {e}")
                continue
        
        if len(features_list) == 0:
            logger.error("No valid training examples processed")
            return {'status': 'no_valid_data'}
        
        X = np.array(features_list)
        
        # Fit scalers
        X_scaled = self.feature_scaler.fit_transform(X)
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        # Training split
        val_size = max(1, len(X) // 5)  # 20% validation
        indices = np.random.permutation(len(X))
        train_indices = indices[val_size:]
        val_indices = indices[:val_size]
        
        X_train = X_tensor[train_indices]
        X_val = X_tensor[val_indices]
        
        training_results = {}
        
        # Train Tick Range Predictor
        logger.info("Training tick range predictor...")
        tick_results = self._train_tick_predictor(
            X_train, X_val, targets, train_indices, val_indices, 
            epochs, batch_size, learning_rate
        )
        training_results['tick_predictor'] = tick_results
        
        # Train IL Forecaster
        logger.info("Training impermanent loss forecaster...")
        il_results = self._train_il_forecaster(
            X_train, X_val, targets, train_indices, val_indices,
            epochs, batch_size, learning_rate
        )
        training_results['il_forecaster'] = il_results
        
        # Train Fee Predictor
        logger.info("Training fee earnings predictor...")
        fee_results = self._train_fee_predictor(
            X_train, X_val, targets, train_indices, val_indices,
            epochs, batch_size, learning_rate
        )
        training_results['fee_predictor'] = fee_results
        
        # Train LSTM Model (with sequence data)
        logger.info("Training LSTM model...")
        lstm_results = self._train_lstm_model(
            X_train, X_val, targets, train_indices, val_indices,
            epochs, batch_size, learning_rate
        )
        training_results['lstm_model'] = lstm_results
        
        logger.info("Model training completed successfully")
        self.save_models()
        
        return {
            'status': 'success',
            'training_samples': len(training_data),
            'valid_samples': len(features_list),
            'epochs': epochs,
            'results': training_results
        }