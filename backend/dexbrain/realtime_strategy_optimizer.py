"""
Real-time Strategy Optimizer with Reinforcement Learning for DeFi Position Management
Implements Deep Q-Network (DQN) for optimal liquidity position decisions
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque
import random
import pickle
from pathlib import Path

from .models.enhanced_ml_models import UniswapFeatures
from .config import Config

logger = logging.getLogger(__name__)

@dataclass
class PositionState:
    """Current state of a liquidity position for RL agent"""
    # Pool metrics
    pool_address: str
    tvl_usd: float
    volume_24h: float
    price_volatility: float
    current_tick: int
    
    # Position metrics
    position_liquidity: float
    tick_lower: int
    tick_upper: int
    fees_uncollected: float
    impermanent_loss: float
    capital_efficiency: float
    
    # Market context
    market_regime: str  # 'trending_up', 'trending_down', 'ranging', 'volatile'
    price_momentum: float
    volume_trend: float
    liquidity_concentration: float
    
    # Time features
    hours_since_last_compound: int
    days_since_position_open: int
    
    def to_array(self) -> np.ndarray:
        """Convert state to numpy array for RL model"""
        return np.array([
            np.log1p(self.tvl_usd),  # Log transform for stability
            np.log1p(self.volume_24h),
            self.price_volatility,
            self.current_tick / 1000,  # Normalize
            np.log1p(self.position_liquidity),
            (self.tick_upper - self.tick_lower) / 1000,  # Range width
            self.fees_uncollected / max(self.position_liquidity, 1),  # Fee yield
            self.impermanent_loss,
            self.capital_efficiency,
            self._encode_market_regime(),
            self.price_momentum,
            self.volume_trend,
            self.liquidity_concentration,
            self.hours_since_last_compound / 24,  # Normalize to days
            np.log1p(self.days_since_position_open)
        ])
    
    def _encode_market_regime(self) -> float:
        """Encode market regime as numerical value"""
        regime_map = {
            'trending_up': 1.0,
            'trending_down': -1.0,
            'ranging': 0.0,
            'volatile': 0.5
        }
        return regime_map.get(self.market_regime, 0.0)

@dataclass
class PositionAction:
    """Actions the RL agent can take"""
    action_type: str  # 'hold', 'compound', 'widen_range', 'narrow_range', 'rebalance'
    urgency: float    # 0.0 to 1.0
    confidence: float # 0.0 to 1.0
    reasoning: str    # Human-readable explanation

class DQNNetwork(nn.Module):
    """Deep Q-Network for position management decisions"""
    
    def __init__(self, state_size: int, action_size: int, hidden_sizes: List[int] = [512, 256, 128]):
        super().__init__()
        
        layers = []
        prev_size = state_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(0.2)
            ])
            prev_size = hidden_size
        
        # Output layer for Q-values
        layers.append(nn.Linear(prev_size, action_size))
        
        self.network = nn.Sequential(*layers)
        
        # Additional heads for confidence and urgency
        self.confidence_head = nn.Sequential(
            nn.Linear(prev_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.network[:-1](x)  # Get features before final layer
        q_values = self.network[-1](features)
        confidence = self.confidence_head(features)
        
        return q_values, confidence

class RealtimeStrategyOptimizer:
    """
    Production-ready reinforcement learning system for DeFi position management
    """
    
    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path or Config.MODEL_STORAGE_PATH / 'rl_strategy'
        self.model_path.mkdir(parents=True, exist_ok=True)
        
        # RL parameters
        self.state_size = 15  # Size of PositionState array
        self.action_size = 5  # Number of possible actions
        self.learning_rate = 0.001
        self.epsilon = 0.1  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.gamma = 0.95  # Discount factor
        self.batch_size = 64
        self.memory_size = 10000
        
        # Action mapping
        self.actions = [
            'hold',          # 0: Do nothing
            'compound',      # 1: Compound fees
            'widen_range',   # 2: Widen position range
            'narrow_range',  # 3: Narrow position range  
            'rebalance'      # 4: Rebalance position
        ]
        
        # Initialize networks
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.q_network = DQNNetwork(self.state_size, self.action_size).to(self.device)
        self.target_network = DQNNetwork(self.state_size, self.action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        
        # Experience replay
        self.memory = deque(maxlen=self.memory_size)
        
        # Performance tracking
        self.performance_history = {
            'rewards': [],
            'actions_taken': {},
            'confidence_scores': [],
            'portfolio_value': [],
            'sharpe_ratio': 0.0
        }
        
        # Load existing model if available
        self.load_model()
        
        logger.info("Real-time strategy optimizer initialized")
    
    def get_optimal_action(self, position_state: PositionState) -> PositionAction:
        """
        Get optimal action for current position state using trained DQN
        """
        state_array = position_state.to_array()
        state_tensor = torch.FloatTensor(state_array).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values, confidence = self.q_network(state_tensor)
            
            # Epsilon-greedy action selection
            if random.random() < self.epsilon:
                action_idx = random.randint(0, self.action_size - 1)
                exploration = True
            else:
                action_idx = q_values.argmax().item()
                exploration = False
            
            action_type = self.actions[action_idx]
            confidence_score = confidence.item()
            
            # Calculate urgency based on position state
            urgency = self._calculate_urgency(position_state)
            
            # Generate reasoning
            reasoning = self._generate_reasoning(position_state, action_type, confidence_score, exploration)
        
        return PositionAction(
            action_type=action_type,
            urgency=urgency,
            confidence=confidence_score,
            reasoning=reasoning
        )
    
    def _calculate_urgency(self, state: PositionState) -> float:
        """Calculate action urgency based on position state"""
        urgency_factors = []
        
        # High volatility increases urgency
        if state.price_volatility > 0.3:  # >30% volatility
            urgency_factors.append(0.8)
        
        # Large uncollected fees increase urgency
        fee_yield = state.fees_uncollected / max(state.position_liquidity, 1)
        if fee_yield > 0.05:  # >5% uncollected fees
            urgency_factors.append(0.7)
        
        # High impermanent loss increases urgency
        if state.impermanent_loss > 0.1:  # >10% IL
            urgency_factors.append(0.6)
        
        # Low capital efficiency increases urgency
        if state.capital_efficiency < 0.3:  # <30% efficiency
            urgency_factors.append(0.5)
        
        # Time since last action
        if state.hours_since_last_compound > 48:  # >48 hours
            urgency_factors.append(0.4)
        
        return max(urgency_factors) if urgency_factors else 0.1
    
    def _generate_reasoning(self, state: PositionState, action: str, confidence: float, exploration: bool) -> str:
        """Generate human-readable reasoning for action"""
        
        if exploration:
            return f"Exploration action: {action} (ε-greedy)"
        
        reasons = []
        
        if action == 'compound':
            fee_yield = state.fees_uncollected / max(state.position_liquidity, 1)
            reasons.append(f"High uncollected fees ({fee_yield:.1%})")
            
        elif action == 'widen_range':
            reasons.append(f"High volatility ({state.price_volatility:.1%})")
            reasons.append(f"IL risk mitigation (current IL: {state.impermanent_loss:.1%})")
            
        elif action == 'narrow_range':
            reasons.append(f"Low volatility market ({state.price_volatility:.1%})")
            reasons.append(f"Improve capital efficiency (current: {state.capital_efficiency:.1%})")
            
        elif action == 'rebalance':
            reasons.append(f"Position out of range (current tick: {state.current_tick})")
            reasons.append(f"Market regime: {state.market_regime}")
            
        elif action == 'hold':
            reasons.append("Optimal position, no action needed")
            reasons.append(f"Good capital efficiency ({state.capital_efficiency:.1%})")
        
        reasoning = f"{action.title()}: {', '.join(reasons)} (confidence: {confidence:.1%})"
        return reasoning
    
    def train_on_experience(self, state: PositionState, action_idx: int, reward: float, 
                          next_state: PositionState, done: bool):
        """Train the DQN on experience tuple"""
        
        # Store experience in replay buffer
        self.memory.append((
            state.to_array(),
            action_idx,
            reward,
            next_state.to_array(),
            done
        ))
        
        # Train if we have enough experiences
        if len(self.memory) >= self.batch_size:
            self._replay_experience()
    
    def _replay_experience(self):
        """Sample and learn from experience replay buffer"""
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)
        
        # Current Q values
        current_q_values, _ = self.q_network(states)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values, _ = self.target_network(next_states)
            max_next_q_values = next_q_values.max(1)[0]
            target_q_values = rewards + (self.gamma * max_next_q_values * ~dones)
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_network(self):
        """Update target network with current network weights"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def calculate_reward(self, prev_state: PositionState, action: str, new_state: PositionState) -> float:
        """Calculate reward for state transition"""
        reward = 0.0
        
        # Reward for fee collection
        fee_increase = new_state.fees_uncollected - prev_state.fees_uncollected
        reward += fee_increase * 10  # Scale fee rewards
        
        # Penalty for impermanent loss increase
        il_change = new_state.impermanent_loss - prev_state.impermanent_loss
        reward -= il_change * 20  # Higher penalty for IL
        
        # Reward for capital efficiency improvement
        efficiency_change = new_state.capital_efficiency - prev_state.capital_efficiency
        reward += efficiency_change * 5
        
        # Action-specific rewards
        if action == 'compound' and fee_increase > 0:
            reward += 1.0  # Bonus for successful compounding
        
        if action == 'widen_range' and new_state.impermanent_loss < prev_state.impermanent_loss:
            reward += 2.0  # Bonus for reducing IL
        
        if action == 'narrow_range' and new_state.capital_efficiency > prev_state.capital_efficiency:
            reward += 1.5  # Bonus for improving efficiency
        
        # Time penalty to encourage action
        time_penalty = -0.1 if action == 'hold' else 0.0
        reward += time_penalty
        
        return reward
    
    def analyze_market_regime(self, pool_data: Dict[str, Any]) -> str:
        """Analyze current market regime using ML"""
        
        # Simple regime detection (can be enhanced with ML models)
        volatility = pool_data.get('price_volatility_24h', 0)
        price_change = pool_data.get('price_change_24h', 0)
        volume_trend = pool_data.get('volume_trend', 0)
        
        if volatility > 0.3:  # High volatility
            return 'volatile'
        elif abs(price_change) > 0.1:  # Strong trend
            return 'trending_up' if price_change > 0 else 'trending_down'
        else:  # Low volatility, small price change
            return 'ranging'
    
    def generate_strategy_recommendations(self, positions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate strategy recommendations for multiple positions"""
        recommendations = []
        
        for position_data in positions:
            try:
                # Create position state
                position_state = self._create_position_state(position_data)
                
                # Get optimal action
                action = self.get_optimal_action(position_state)
                
                # Calculate expected outcomes
                expected_outcomes = self._predict_outcomes(position_state, action)
                
                recommendation = {
                    'position_id': position_data.get('id'),
                    'pool_address': position_data.get('pool_address'),
                    'action': action.action_type,
                    'urgency': action.urgency,
                    'confidence': action.confidence,
                    'reasoning': action.reasoning,
                    'expected_outcomes': expected_outcomes,
                    'timestamp': datetime.now().isoformat()
                }
                
                recommendations.append(recommendation)
                
                # Update performance tracking
                self.performance_history['actions_taken'][action.action_type] = \
                    self.performance_history['actions_taken'].get(action.action_type, 0) + 1
                self.performance_history['confidence_scores'].append(action.confidence)
                
            except Exception as e:
                logger.warning(f"Failed to generate recommendation for position {position_data.get('id')}: {e}")
        
        return recommendations
    
    def _create_position_state(self, position_data: Dict[str, Any]) -> PositionState:
        """Create PositionState from position data"""
        
        # Analyze market regime
        market_regime = self.analyze_market_regime(position_data)
        
        return PositionState(
            pool_address=position_data.get('pool_address', ''),
            tvl_usd=float(position_data.get('tvl_usd', 0)),
            volume_24h=float(position_data.get('volume_24h', 0)),
            price_volatility=float(position_data.get('price_volatility_24h', 0.1)),
            current_tick=int(position_data.get('current_tick', 0)),
            position_liquidity=float(position_data.get('liquidity', 0)),
            tick_lower=int(position_data.get('tick_lower', 0)),
            tick_upper=int(position_data.get('tick_upper', 0)),
            fees_uncollected=float(position_data.get('fees_uncollected', 0)),
            impermanent_loss=float(position_data.get('impermanent_loss', 0)),
            capital_efficiency=float(position_data.get('capital_efficiency', 0.5)),
            market_regime=market_regime,
            price_momentum=float(position_data.get('price_momentum', 0)),
            volume_trend=float(position_data.get('volume_trend', 0)),
            liquidity_concentration=float(position_data.get('liquidity_concentration', 0.5)),
            hours_since_last_compound=int(position_data.get('hours_since_last_compound', 0)),
            days_since_position_open=int(position_data.get('days_since_position_open', 1))
        )
    
    def _predict_outcomes(self, state: PositionState, action: PositionAction) -> Dict[str, float]:
        """Predict expected outcomes of taking action"""
        
        outcomes = {
            'expected_fee_increase': 0.0,
            'expected_il_change': 0.0,
            'expected_efficiency_change': 0.0,
            'risk_score': 0.5
        }
        
        if action.action_type == 'compound':
            outcomes['expected_fee_increase'] = state.fees_uncollected * 0.95  # Assume 95% collection
            outcomes['risk_score'] = 0.1  # Low risk action
            
        elif action.action_type == 'widen_range':
            outcomes['expected_il_change'] = -state.impermanent_loss * 0.3  # Reduce IL by 30%
            outcomes['expected_efficiency_change'] = -0.1  # Reduce efficiency
            outcomes['risk_score'] = 0.3
            
        elif action.action_type == 'narrow_range':
            outcomes['expected_efficiency_change'] = 0.2  # Improve efficiency
            outcomes['expected_il_change'] = state.price_volatility * 0.1  # Increase IL risk
            outcomes['risk_score'] = 0.7
            
        elif action.action_type == 'rebalance':
            outcomes['expected_efficiency_change'] = 0.15
            outcomes['risk_score'] = 0.4
            
        return outcomes
    
    def save_model(self):
        """Save trained model and performance history"""
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'performance_history': self.performance_history
        }
        
        model_file = self.model_path / 'dqn_strategy_model.pth'
        torch.save(checkpoint, model_file)
        
        logger.info(f"Strategy optimizer model saved to {model_file}")
    
    def load_model(self):
        """Load existing model if available"""
        model_file = self.model_path / 'dqn_strategy_model.pth'
        
        if model_file.exists():
            try:
                checkpoint = torch.load(model_file, map_location=self.device)
                
                self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
                self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.epsilon = checkpoint.get('epsilon', self.epsilon)
                self.performance_history = checkpoint.get('performance_history', self.performance_history)
                
                logger.info(f"Strategy optimizer model loaded from {model_file}")
            except Exception as e:
                logger.warning(f"Failed to load model: {e}, starting with fresh model")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for monitoring"""
        
        if not self.performance_history['rewards']:
            return {'status': 'insufficient_data'}
        
        rewards = self.performance_history['rewards']
        confidence_scores = self.performance_history['confidence_scores']
        
        metrics = {
            'avg_reward': np.mean(rewards),
            'total_reward': np.sum(rewards),
            'reward_volatility': np.std(rewards),
            'avg_confidence': np.mean(confidence_scores) if confidence_scores else 0,
            'actions_distribution': self.performance_history['actions_taken'],
            'exploration_rate': self.epsilon,
            'sharpe_ratio': self.performance_history['sharpe_ratio'],
            'total_decisions': len(rewards)
        }
        
        return metrics

# Factory function for easy instantiation
def create_strategy_optimizer(model_path: Optional[Path] = None) -> RealtimeStrategyOptimizer:
    """Create and return a strategy optimizer instance"""
    return RealtimeStrategyOptimizer(model_path)