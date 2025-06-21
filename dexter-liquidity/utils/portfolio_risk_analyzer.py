"""
Portfolio Risk Analyzer
Advanced risk management and portfolio analysis for DeFi positions
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats, optimize
import logging
import asyncio
from enum import Enum

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    """Risk level classifications"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"

@dataclass
class PositionRisk:
    """Risk metrics for individual positions"""
    position_id: str
    token_pair: str
    risk_level: RiskLevel
    var_1d: float
    var_7d: float
    expected_shortfall: float
    concentration_risk: float
    liquidity_risk: float
    correlation_risk: float
    impermanent_loss_risk: float
    
@dataclass
class PortfolioRisk:
    """Portfolio-level risk metrics"""
    total_var: float
    component_var: Dict[str, float]
    marginal_var: Dict[str, float]
    diversification_ratio: float
    concentration_index: float
    risk_budget_utilization: float
    stress_test_results: Dict[str, float]
    
@dataclass
class RiskAlert:
    """Risk alert data structure"""
    alert_id: str
    timestamp: datetime
    risk_type: str
    severity: str
    position_id: Optional[str]
    message: str
    recommended_action: str
    threshold_breached: float
    current_value: float

class PortfolioRiskAnalyzer:
    """
    Advanced portfolio risk analysis and monitoring
    """
    
    def __init__(self, 
                 risk_free_rate: float = 0.04,
                 confidence_levels: List[float] = None):
        """
        Initialize portfolio risk analyzer
        
        Args:
            risk_free_rate: Annual risk-free rate
            confidence_levels: VaR confidence levels to calculate
        """
        self.risk_free_rate = risk_free_rate
        self.confidence_levels = confidence_levels or [0.95, 0.99]
        
        # Position tracking
        self.positions: Dict[str, Dict] = {}
        self.returns_history: Dict[str, List[float]] = {}
        self.correlation_matrix: Optional[np.ndarray] = None
        
        # Risk thresholds
        self.risk_thresholds = {
            'var_1d': 0.05,  # 5% daily VaR threshold
            'var_7d': 0.15,  # 15% weekly VaR threshold
            'concentration': 0.25,  # 25% max concentration per position
            'correlation': 0.8,  # 80% correlation threshold
            'drawdown': 0.20,  # 20% max drawdown threshold
        }
        
        # Alert history
        self.alerts: List[RiskAlert] = []
        
        logger.info("Portfolio risk analyzer initialized")
    
    async def analyze_portfolio_risk(self, 
                                   positions: List[Dict],
                                   returns_data: Dict[str, List[float]]) -> PortfolioRisk:
        """
        Perform comprehensive portfolio risk analysis
        
        Args:
            positions: List of position dictionaries
            returns_data: Historical returns for each position
            
        Returns:
            PortfolioRisk object with analysis results
        """
        try:
            # Update internal data
            self.positions = {pos['id']: pos for pos in positions}
            self.returns_history = returns_data
            
            # Calculate correlation matrix
            self.correlation_matrix = await self._calculate_correlation_matrix()
            
            # Portfolio weights
            weights = await self._calculate_position_weights()
            
            # Portfolio VaR calculation
            portfolio_var = await self._calculate_portfolio_var(weights)
            
            # Component VaR
            component_var = await self._calculate_component_var(weights)
            
            # Marginal VaR
            marginal_var = await self._calculate_marginal_var(weights)
            
            # Diversification metrics
            diversification_ratio = await self._calculate_diversification_ratio(weights)
            
            # Concentration analysis
            concentration_index = await self._calculate_concentration_index(weights)
            
            # Risk budget utilization
            risk_budget = await self._calculate_risk_budget_utilization(component_var)
            
            # Stress testing
            stress_results = await self._run_stress_tests(weights)
            
            return PortfolioRisk(
                total_var=portfolio_var,
                component_var=component_var,
                marginal_var=marginal_var,
                diversification_ratio=diversification_ratio,
                concentration_index=concentration_index,
                risk_budget_utilization=risk_budget,
                stress_test_results=stress_results
            )
            
        except Exception as e:
            logger.error(f"Error analyzing portfolio risk: {e}")
            return self._empty_portfolio_risk()
    
    async def analyze_position_risk(self, position_id: str) -> Optional[PositionRisk]:
        """
        Analyze risk for individual position
        
        Args:
            position_id: ID of position to analyze
            
        Returns:
            PositionRisk object or None if analysis fails
        """
        try:
            if position_id not in self.positions:
                return None
            
            position = self.positions[position_id]
            returns = self.returns_history.get(position_id, [])
            
            if not returns:
                return None
            
            returns_array = np.array(returns)
            
            # Calculate VaR
            var_1d = self._calculate_var(returns_array, 0.95)
            var_7d = var_1d * np.sqrt(7)  # Scale to weekly
            
            # Expected shortfall
            expected_shortfall = self._calculate_expected_shortfall(returns_array, 0.95)
            
            # Risk classifications
            concentration_risk = await self._calculate_concentration_risk(position_id)
            liquidity_risk = await self._calculate_liquidity_risk(position)
            correlation_risk = await self._calculate_correlation_risk(position_id)
            il_risk = await self._calculate_impermanent_loss_risk(position)
            
            # Overall risk level
            risk_level = await self._classify_risk_level(
                var_1d, concentration_risk, liquidity_risk, correlation_risk
            )
            
            return PositionRisk(
                position_id=position_id,
                token_pair=f"{position.get('token0', 'UNK')}/{position.get('token1', 'UNK')}",
                risk_level=risk_level,
                var_1d=abs(var_1d),
                var_7d=abs(var_7d),
                expected_shortfall=abs(expected_shortfall),
                concentration_risk=concentration_risk,
                liquidity_risk=liquidity_risk,
                correlation_risk=correlation_risk,
                impermanent_loss_risk=il_risk
            )
            
        except Exception as e:
            logger.error(f"Error analyzing position risk for {position_id}: {e}")
            return None
    
    async def check_risk_alerts(self) -> List[RiskAlert]:
        """
        Check for risk threshold breaches and generate alerts
        
        Returns:
            List of new risk alerts
        """
        new_alerts = []
        
        try:
            # Check portfolio-level risks
            portfolio_risk = await self.analyze_portfolio_risk(
                list(self.positions.values()),
                self.returns_history
            )
            
            # VaR threshold check
            if portfolio_risk.total_var > self.risk_thresholds['var_1d']:
                alert = RiskAlert(
                    alert_id=f"portfolio_var_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    timestamp=datetime.now(),
                    risk_type="portfolio_var",
                    severity="HIGH",
                    position_id=None,
                    message=f"Portfolio VaR exceeds threshold: {portfolio_risk.total_var:.2%}",
                    recommended_action="Consider reducing position sizes or hedging",
                    threshold_breached=self.risk_thresholds['var_1d'],
                    current_value=portfolio_risk.total_var
                )
                new_alerts.append(alert)
            
            # Concentration risk check
            if portfolio_risk.concentration_index > self.risk_thresholds['concentration']:
                alert = RiskAlert(
                    alert_id=f"concentration_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    timestamp=datetime.now(),
                    risk_type="concentration",
                    severity="MEDIUM",
                    position_id=None,
                    message=f"Portfolio concentration risk: {portfolio_risk.concentration_index:.2%}",
                    recommended_action="Diversify positions across different token pairs",
                    threshold_breached=self.risk_thresholds['concentration'],
                    current_value=portfolio_risk.concentration_index
                )
                new_alerts.append(alert)
            
            # Check individual position risks
            for position_id in self.positions:
                position_risk = await self.analyze_position_risk(position_id)
                if position_risk:
                    # High VaR alert
                    if position_risk.var_1d > self.risk_thresholds['var_1d']:
                        alert = RiskAlert(
                            alert_id=f"position_var_{position_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                            timestamp=datetime.now(),
                            risk_type="position_var",
                            severity="HIGH" if position_risk.var_1d > 0.1 else "MEDIUM",
                            position_id=position_id,
                            message=f"Position {position_risk.token_pair} VaR: {position_risk.var_1d:.2%}",
                            recommended_action="Consider reducing position size or adding hedges",
                            threshold_breached=self.risk_thresholds['var_1d'],
                            current_value=position_risk.var_1d
                        )
                        new_alerts.append(alert)
                    
                    # High correlation alert
                    if position_risk.correlation_risk > self.risk_thresholds['correlation']:
                        alert = RiskAlert(
                            alert_id=f"correlation_{position_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                            timestamp=datetime.now(),
                            risk_type="correlation",
                            severity="MEDIUM",
                            position_id=position_id,
                            message=f"High correlation risk for {position_risk.token_pair}: {position_risk.correlation_risk:.2%}",
                            recommended_action="Consider diversifying into uncorrelated assets",
                            threshold_breached=self.risk_thresholds['correlation'],
                            current_value=position_risk.correlation_risk
                        )
                        new_alerts.append(alert)
            
            # Store alerts
            self.alerts.extend(new_alerts)
            
            # Keep only last 1000 alerts
            if len(self.alerts) > 1000:
                self.alerts = self.alerts[-1000:]
            
            return new_alerts
            
        except Exception as e:
            logger.error(f"Error checking risk alerts: {e}")
            return []
    
    async def calculate_optimal_position_sizes(self, 
                                             target_risk: float = 0.02,
                                             max_position_size: float = 0.25) -> Dict[str, float]:
        """
        Calculate optimal position sizes using risk parity approach
        
        Args:
            target_risk: Target portfolio volatility
            max_position_size: Maximum weight per position
            
        Returns:
            Dictionary of position IDs to optimal weights
        """
        try:
            if not self.positions or not self.returns_history:
                return {}
            
            # Get returns matrix
            returns_matrix = []
            position_ids = []
            
            for pos_id, returns in self.returns_history.items():
                if len(returns) >= 30:  # Minimum data points
                    returns_matrix.append(returns[-252:])  # Last year of data
                    position_ids.append(pos_id)
            
            if len(returns_matrix) < 2:
                return {pos_id: 1.0/len(self.positions) for pos_id in self.positions}
            
            # Convert to numpy array
            returns_matrix = np.array(returns_matrix).T
            
            # Calculate covariance matrix
            cov_matrix = np.cov(returns_matrix.T) * 252  # Annualized
            
            # Risk parity optimization
            n_assets = len(position_ids)
            
            def risk_parity_objective(weights):
                portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
                marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
                contrib = weights * marginal_contrib
                target_contrib = np.ones(n_assets) / n_assets
                return np.sum((contrib / np.sum(contrib) - target_contrib) ** 2)
            
            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Weights sum to 1
            ]
            
            bounds = [(0.01, max_position_size) for _ in range(n_assets)]
            
            # Initial guess (equal weights)
            x0 = np.ones(n_assets) / n_assets
            
            # Optimize
            result = optimize.minimize(
                risk_parity_objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            if result.success:
                optimal_weights = dict(zip(position_ids, result.x))
                
                # Add zero weights for positions not included
                for pos_id in self.positions:
                    if pos_id not in optimal_weights:
                        optimal_weights[pos_id] = 0.0
                
                return optimal_weights
            else:
                # Fallback to equal weights
                return {pos_id: 1.0/len(self.positions) for pos_id in self.positions}
            
        except Exception as e:
            logger.error(f"Error calculating optimal position sizes: {e}")
            return {pos_id: 1.0/len(self.positions) for pos_id in self.positions}
    
    async def run_monte_carlo_simulation(self, 
                                       num_simulations: int = 10000,
                                       time_horizon: int = 30) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation for portfolio risk assessment
        
        Args:
            num_simulations: Number of simulation paths
            time_horizon: Time horizon in days
            
        Returns:
            Dictionary with simulation results
        """
        try:
            if not self.returns_history:
                return {}
            
            # Prepare returns data
            returns_matrix = []
            position_ids = []
            
            for pos_id, returns in self.returns_history.items():
                if len(returns) >= 30:
                    returns_matrix.append(returns[-252:])  # Last year
                    position_ids.append(pos_id)
            
            if not returns_matrix:
                return {}
            
            returns_array = np.array(returns_matrix)
            
            # Calculate parameters
            mean_returns = np.mean(returns_array, axis=1)
            cov_matrix = np.cov(returns_array)
            
            # Portfolio weights
            weights = await self._calculate_position_weights()
            weight_vector = np.array([weights.get(pos_id, 0) for pos_id in position_ids])
            
            # Simulate portfolio returns
            simulated_returns = []
            
            for _ in range(num_simulations):
                # Generate random returns for time horizon
                random_returns = np.random.multivariate_normal(
                    mean_returns, cov_matrix, time_horizon
                )
                
                # Calculate portfolio returns
                portfolio_returns = np.dot(random_returns, weight_vector)
                
                # Calculate cumulative return
                cumulative_return = np.prod(1 + portfolio_returns) - 1
                simulated_returns.append(cumulative_return)
            
            simulated_returns = np.array(simulated_returns)
            
            # Calculate statistics
            results = {
                'expected_return': float(np.mean(simulated_returns)),
                'volatility': float(np.std(simulated_returns)),
                'var_95': float(np.percentile(simulated_returns, 5)),
                'var_99': float(np.percentile(simulated_returns, 1)),
                'expected_shortfall_95': float(np.mean(simulated_returns[simulated_returns <= np.percentile(simulated_returns, 5)])),
                'expected_shortfall_99': float(np.mean(simulated_returns[simulated_returns <= np.percentile(simulated_returns, 1)])),
                'probability_of_loss': float(np.mean(simulated_returns < 0)),
                'probability_of_large_loss': float(np.mean(simulated_returns < -0.1)),
                'max_loss': float(np.min(simulated_returns)),
                'max_gain': float(np.max(simulated_returns)),
                'simulation_params': {
                    'num_simulations': num_simulations,
                    'time_horizon_days': time_horizon,
                    'positions_included': len(position_ids)
                }
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error running Monte Carlo simulation: {e}")
            return {}
    
    # Helper methods
    
    async def _calculate_correlation_matrix(self) -> Optional[np.ndarray]:
        """Calculate correlation matrix for all positions"""
        try:
            if len(self.returns_history) < 2:
                return None
            
            returns_data = []
            for returns in self.returns_history.values():
                if len(returns) >= 30:  # Minimum data points
                    returns_data.append(returns[-252:])  # Last year of data
            
            if len(returns_data) < 2:
                return None
            
            # Ensure all series have same length
            min_length = min(len(r) for r in returns_data)
            returns_matrix = np.array([r[-min_length:] for r in returns_data])
            
            return np.corrcoef(returns_matrix)
            
        except Exception as e:
            logger.error(f"Error calculating correlation matrix: {e}")
            return None
    
    async def _calculate_position_weights(self) -> Dict[str, float]:
        """Calculate current position weights by value"""
        total_value = sum(pos.get('value', 0) for pos in self.positions.values())
        
        if total_value == 0:
            return {pos_id: 0 for pos_id in self.positions}
        
        return {
            pos_id: pos.get('value', 0) / total_value 
            for pos_id, pos in self.positions.items()
        }
    
    async def _calculate_portfolio_var(self, weights: Dict[str, float]) -> float:
        """Calculate portfolio Value at Risk"""
        try:
            if not self.returns_history or not weights:
                return 0.0
            
            # Create portfolio returns time series
            portfolio_returns = []
            
            # Get aligned returns
            min_length = min(len(returns) for returns in self.returns_history.values() if returns)
            if min_length < 2:
                return 0.0
            
            for i in range(min_length):
                period_return = sum(
                    weights.get(pos_id, 0) * returns[-min_length + i]
                    for pos_id, returns in self.returns_history.items()
                    if returns
                )
                portfolio_returns.append(period_return)
            
            if not portfolio_returns:
                return 0.0
            
            return float(abs(np.percentile(portfolio_returns, 5)))  # 95% VaR
            
        except Exception as e:
            logger.error(f"Error calculating portfolio VaR: {e}")
            return 0.0
    
    async def _calculate_component_var(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Calculate component VaR for each position"""
        try:
            component_var = {}
            
            for pos_id in self.positions:
                if pos_id in self.returns_history and self.returns_history[pos_id]:
                    # Simplified component VaR calculation
                    position_returns = np.array(self.returns_history[pos_id])
                    position_var = abs(np.percentile(position_returns, 5))
                    weight = weights.get(pos_id, 0)
                    
                    component_var[pos_id] = weight * position_var
                else:
                    component_var[pos_id] = 0.0
            
            return component_var
            
        except Exception as e:
            logger.error(f"Error calculating component VaR: {e}")
            return {}
    
    async def _calculate_marginal_var(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Calculate marginal VaR for each position"""
        try:
            marginal_var = {}
            base_var = await self._calculate_portfolio_var(weights)
            
            for pos_id in self.positions:
                # Calculate VaR with small increase in position weight
                modified_weights = weights.copy()
                delta = 0.01  # 1% increase
                
                if pos_id in modified_weights:
                    modified_weights[pos_id] += delta
                    # Renormalize
                    total_weight = sum(modified_weights.values())
                    if total_weight > 0:
                        modified_weights = {k: v/total_weight for k, v in modified_weights.items()}
                    
                    new_var = await self._calculate_portfolio_var(modified_weights)
                    marginal_var[pos_id] = (new_var - base_var) / delta
                else:
                    marginal_var[pos_id] = 0.0
            
            return marginal_var
            
        except Exception as e:
            logger.error(f"Error calculating marginal VaR: {e}")
            return {}
    
    def _calculate_var(self, returns: np.ndarray, confidence_level: float) -> float:
        """Calculate Value at Risk for given confidence level"""
        if len(returns) == 0:
            return 0.0
        
        percentile = (1 - confidence_level) * 100
        return float(np.percentile(returns, percentile))
    
    def _calculate_expected_shortfall(self, returns: np.ndarray, confidence_level: float) -> float:
        """Calculate Expected Shortfall (Conditional VaR)"""
        if len(returns) == 0:
            return 0.0
        
        var = self._calculate_var(returns, confidence_level)
        tail_returns = returns[returns <= var]
        
        return float(np.mean(tail_returns)) if len(tail_returns) > 0 else 0.0
    
    async def _calculate_concentration_risk(self, position_id: str) -> float:
        """Calculate concentration risk for a position"""
        weights = await self._calculate_position_weights()
        return weights.get(position_id, 0.0)
    
    async def _calculate_liquidity_risk(self, position: Dict) -> float:
        """Calculate liquidity risk based on position characteristics"""
        # Simplified liquidity risk based on TVL and volume
        tvl = position.get('tvl', 0)
        volume = position.get('volume_24h', 0)
        
        if tvl == 0:
            return 1.0  # Maximum risk
        
        # Liquidity ratio (higher is better)
        liquidity_ratio = volume / tvl if tvl > 0 else 0
        
        # Convert to risk score (0-1, where 1 is highest risk)
        if liquidity_ratio > 0.1:  # 10% daily turnover is good
            return 0.1
        elif liquidity_ratio > 0.05:  # 5% is moderate
            return 0.3
        elif liquidity_ratio > 0.01:  # 1% is concerning
            return 0.6
        else:
            return 1.0  # Very low liquidity
    
    async def _calculate_correlation_risk(self, position_id: str) -> float:
        """Calculate correlation risk for a position"""
        if not self.correlation_matrix or position_id not in self.returns_history:
            return 0.0
        
        position_ids = list(self.returns_history.keys())
        if position_id not in position_ids:
            return 0.0
        
        pos_index = position_ids.index(position_id)
        
        # Average absolute correlation with other positions
        correlations = self.correlation_matrix[pos_index]
        other_correlations = np.abs(correlations[correlations != 1.0])  # Exclude self-correlation
        
        return float(np.mean(other_correlations)) if len(other_correlations) > 0 else 0.0
    
    async def _calculate_impermanent_loss_risk(self, position: Dict) -> float:
        """Calculate impermanent loss risk"""
        # Simplified IL risk based on price volatility and correlation
        token0_vol = position.get('token0_volatility', 0.2)  # Default 20% volatility
        token1_vol = position.get('token1_volatility', 0.2)
        correlation = position.get('token_correlation', 0.3)  # Default moderate correlation
        
        # IL risk increases with volatility difference and low correlation
        vol_diff = abs(token0_vol - token1_vol)
        il_risk = vol_diff * (1 - correlation) * 0.5  # Scale factor
        
        return min(il_risk, 1.0)  # Cap at 100%
    
    async def _classify_risk_level(self, var: float, concentration: float, 
                                 liquidity: float, correlation: float) -> RiskLevel:
        """Classify overall risk level for a position"""
        risk_score = (
            abs(var) * 0.4 +
            concentration * 0.3 +
            liquidity * 0.2 +
            correlation * 0.1
        )
        
        if risk_score > 0.2:
            return RiskLevel.EXTREME
        elif risk_score > 0.1:
            return RiskLevel.HIGH
        elif risk_score > 0.05:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    async def _calculate_diversification_ratio(self, weights: Dict[str, float]) -> float:
        """Calculate diversification ratio"""
        try:
            if not self.correlation_matrix or len(weights) < 2:
                return 1.0
            
            weight_values = list(weights.values())
            weight_array = np.array(weight_values)
            
            # Weighted average volatility
            position_vols = []
            for pos_id, returns in self.returns_history.items():
                if returns:
                    vol = np.std(returns) * np.sqrt(252)
                    position_vols.append(vol)
                else:
                    position_vols.append(0.2)  # Default volatility
            
            vol_array = np.array(position_vols)
            weighted_avg_vol = np.dot(weight_array, vol_array)
            
            # Portfolio volatility
            cov_matrix = self.correlation_matrix * np.outer(vol_array, vol_array)
            portfolio_vol = np.sqrt(np.dot(weight_array, np.dot(cov_matrix, weight_array)))
            
            return float(weighted_avg_vol / portfolio_vol) if portfolio_vol > 0 else 1.0
            
        except Exception as e:
            logger.error(f"Error calculating diversification ratio: {e}")
            return 1.0
    
    async def _calculate_concentration_index(self, weights: Dict[str, float]) -> float:
        """Calculate Herfindahl concentration index"""
        weight_values = list(weights.values())
        return float(sum(w**2 for w in weight_values))
    
    async def _calculate_risk_budget_utilization(self, component_var: Dict[str, float]) -> float:
        """Calculate risk budget utilization"""
        total_component_var = sum(component_var.values())
        return min(total_component_var / 0.02, 1.0)  # 2% daily risk budget
    
    async def _run_stress_tests(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Run stress tests on the portfolio"""
        try:
            stress_results = {}
            
            # Market crash scenario (-20% across all positions)
            market_crash_return = -0.20
            portfolio_loss = sum(weights.get(pos_id, 0) * market_crash_return for pos_id in self.positions)
            stress_results['market_crash_20pct'] = portfolio_loss
            
            # Volatility spike (double volatility)
            if self.returns_history:
                avg_vol = np.mean([np.std(returns) for returns in self.returns_history.values() if returns])
                stress_results['volatility_spike'] = avg_vol * 2
            else:
                stress_results['volatility_spike'] = 0.4
            
            # Liquidity crisis (spread widening)
            liquidity_cost = sum(weights.get(pos_id, 0) * 0.05 for pos_id in self.positions)  # 5% cost
            stress_results['liquidity_crisis'] = -liquidity_cost
            
            return stress_results
            
        except Exception as e:
            logger.error(f"Error running stress tests: {e}")
            return {}
    
    def _empty_portfolio_risk(self) -> PortfolioRisk:
        """Return empty portfolio risk for error cases"""
        return PortfolioRisk(
            total_var=0.0,
            component_var={},
            marginal_var={},
            diversification_ratio=1.0,
            concentration_index=0.0,
            risk_budget_utilization=0.0,
            stress_test_results={}
        )