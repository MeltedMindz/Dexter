"""
Enhanced Performance Tracker with Advanced Financial Metrics
Implements comprehensive risk metrics, portfolio analysis, and benchmarking
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from scipy import stats
from prometheus_client import Gauge, Counter, Histogram
import asyncio
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Enhanced Prometheus metrics
ENHANCED_TVL = Gauge('dexter_enhanced_tvl', 'Enhanced Total Value Locked', ['agent', 'pool'])
ENHANCED_SHARPE = Gauge('dexter_enhanced_sharpe_ratio', 'Enhanced Sharpe Ratio', ['agent', 'timeframe'])
VAR_GAUGE = Gauge('dexter_value_at_risk', 'Value at Risk', ['agent', 'confidence_level'])
BETA_GAUGE = Gauge('dexter_beta', 'Beta vs Market', ['agent', 'benchmark'])
ALPHA_GAUGE = Gauge('dexter_alpha', 'Alpha vs Benchmark', ['agent', 'benchmark'])
SORTINO_GAUGE = Gauge('dexter_sortino_ratio', 'Sortino Ratio', ['agent'])
CALMAR_GAUGE = Gauge('dexter_calmar_ratio', 'Calmar Ratio', ['agent'])
INFORMATION_RATIO = Gauge('dexter_information_ratio', 'Information Ratio', ['agent', 'benchmark'])
TREYNOR_RATIO = Gauge('dexter_treynor_ratio', 'Treynor Ratio', ['agent'])
MAX_DRAWDOWN_DURATION = Gauge('dexter_max_drawdown_duration_days', 'Max Drawdown Duration', ['agent'])
PROFIT_FACTOR = Gauge('dexter_profit_factor', 'Profit Factor', ['agent'])
RECOVERY_FACTOR = Gauge('dexter_recovery_factor', 'Recovery Factor', ['agent'])

@dataclass
class EnhancedPerformanceMetrics:
    """Comprehensive performance metrics data structure"""
    # Basic metrics
    apy: float
    dpy: float
    total_value_locked: float
    total_fees_earned: float
    impermanent_loss: float
    
    # Risk metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    treynor_ratio: float
    information_ratio: float
    volatility: float
    downside_volatility: float
    
    # Drawdown metrics
    max_drawdown: float
    max_drawdown_duration: int
    current_drawdown: float
    recovery_factor: float
    
    # Return metrics
    time_weighted_return: float
    money_weighted_return: float
    geometric_mean_return: float
    arithmetic_mean_return: float
    
    # Risk measures
    value_at_risk_95: float
    value_at_risk_99: float
    conditional_var_95: float
    conditional_var_99: float
    skewness: float
    kurtosis: float
    
    # Performance ratios
    win_rate: float
    profit_factor: float
    payoff_ratio: float
    kelly_criterion: float
    
    # Market comparison
    alpha: float
    beta: float
    correlation: float
    tracking_error: float
    
    # Additional metrics
    ulcer_index: float
    sterling_ratio: float
    burke_ratio: float
    tail_ratio: float
    capture_ratio: float

@dataclass
class RiskMetrics:
    """Risk-specific metrics container"""
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    max_drawdown: float
    volatility: float
    downside_volatility: float
    skewness: float
    kurtosis: float
    ulcer_index: float

@dataclass
class BenchmarkComparison:
    """Benchmark comparison metrics"""
    alpha: float
    beta: float
    correlation: float
    tracking_error: float
    information_ratio: float
    capture_ratio: float

class EnhancedPerformanceTracker:
    """
    Advanced performance tracking with institutional-grade metrics
    """
    
    def __init__(self, 
                 agent_name: str = "default",
                 benchmark_data: Optional[List[float]] = None,
                 risk_free_rate: float = 0.04):
        """
        Initialize enhanced performance tracker
        
        Args:
            agent_name: Name of the trading agent
            benchmark_data: Benchmark returns for comparison (e.g., ETH/BTC returns)
            risk_free_rate: Annual risk-free rate (default 4%)
        """
        self.agent_name = agent_name
        self.risk_free_rate = risk_free_rate
        self.benchmark_data = benchmark_data or []
        
        # Performance data storage
        self.returns: List[float] = []
        self.timestamps: List[datetime] = []
        self.values: List[float] = []
        self.fees_history: List[float] = []
        self.il_history: List[float] = []
        self.volumes: List[float] = []
        
        # Position tracking
        self.positions: Dict[str, Dict] = {}
        self.trades: List[Dict] = []
        
        # Benchmark tracking
        self.benchmark_returns: List[float] = []
        
        # Cache for expensive calculations
        self._metrics_cache: Optional[EnhancedPerformanceMetrics] = None
        self._cache_timestamp: Optional[datetime] = None
        self._cache_ttl = timedelta(minutes=5)
        
        logger.info(f"Enhanced performance tracker initialized for agent: {agent_name}")
    
    async def update_performance(self,
                                value: float,
                                fees: float,
                                il: float,
                                volume: float,
                                timestamp: datetime,
                                benchmark_return: Optional[float] = None) -> None:
        """
        Update performance data with new values
        
        Args:
            value: Current portfolio value
            fees: Fees earned in this period
            il: Impermanent loss
            volume: Trading volume
            timestamp: Current timestamp
            benchmark_return: Benchmark return for this period
        """
        try:
            # Calculate return if we have previous data
            if self.values:
                previous_value = self.values[-1]
                if previous_value > 0:
                    period_return = (value + fees - il - previous_value) / previous_value
                else:
                    period_return = 0.0
            else:
                period_return = 0.0
            
            # Store data
            self.returns.append(period_return)
            self.values.append(value)
            self.fees_history.append(fees)
            self.il_history.append(il)
            self.volumes.append(volume)
            self.timestamps.append(timestamp)
            
            # Store benchmark return if provided
            if benchmark_return is not None:
                self.benchmark_returns.append(benchmark_return)
            
            # Invalidate cache
            self._metrics_cache = None
            
            # Update Prometheus metrics
            await self._update_prometheus_metrics()
            
            # Limit history to last 10,000 points for memory efficiency
            if len(self.returns) > 10000:
                self._trim_history()
            
        except Exception as e:
            logger.error(f"Error updating performance: {e}")
    
    async def calculate_enhanced_metrics(self) -> EnhancedPerformanceMetrics:
        """
        Calculate comprehensive performance metrics
        
        Returns:
            EnhancedPerformanceMetrics with all calculated metrics
        """
        # Check cache
        if (self._metrics_cache and self._cache_timestamp and 
            datetime.now() - self._cache_timestamp < self._cache_ttl):
            return self._metrics_cache
        
        try:
            if not self.returns or not self.values:
                return self._empty_enhanced_metrics()
            
            returns_array = np.array(self.returns)
            values_array = np.array(self.values)
            
            # Basic calculations
            total_return = (values_array[-1] - values_array[0]) / values_array[0] if values_array[0] > 0 else 0
            
            # Annualized metrics
            days = len(self.returns)
            periods_per_year = 365
            
            # Return metrics
            geometric_mean = self._calculate_geometric_mean(returns_array)
            arithmetic_mean = np.mean(returns_array)
            apy = (1 + geometric_mean) ** periods_per_year - 1 if geometric_mean > -1 else -1
            
            # Risk metrics
            volatility = np.std(returns_array) * np.sqrt(periods_per_year)
            downside_returns = returns_array[returns_array < 0]
            downside_volatility = np.std(downside_returns) * np.sqrt(periods_per_year) if len(downside_returns) > 0 else 0
            
            # Advanced risk measures
            var_95 = self._calculate_var(returns_array, 0.95)
            var_99 = self._calculate_var(returns_array, 0.99)
            cvar_95 = self._calculate_cvar(returns_array, 0.95)
            cvar_99 = self._calculate_cvar(returns_array, 0.99)
            
            # Distribution metrics
            skewness = stats.skew(returns_array) if len(returns_array) > 2 else 0
            kurtosis = stats.kurtosis(returns_array) if len(returns_array) > 2 else 0
            
            # Drawdown metrics
            drawdown_info = self._calculate_drawdown_metrics(values_array)
            
            # Performance ratios
            sharpe_ratio = self._calculate_sharpe_ratio(returns_array, volatility)
            sortino_ratio = self._calculate_sortino_ratio(returns_array, downside_volatility)
            calmar_ratio = apy / abs(drawdown_info['max_drawdown']) if drawdown_info['max_drawdown'] != 0 else 0
            
            # Win rate and profit metrics
            win_rate = len(returns_array[returns_array > 0]) / len(returns_array)
            profit_factor = self._calculate_profit_factor(returns_array)
            
            # Benchmark comparison
            benchmark_metrics = self._calculate_benchmark_metrics(returns_array)
            
            # Advanced metrics
            ulcer_index = self._calculate_ulcer_index(values_array)
            sterling_ratio = self._calculate_sterling_ratio(apy, drawdown_info['max_drawdown'])
            burke_ratio = self._calculate_burke_ratio(apy, values_array)
            tail_ratio = self._calculate_tail_ratio(returns_array)
            kelly_criterion = self._calculate_kelly_criterion(returns_array)
            recovery_factor = abs(total_return / drawdown_info['max_drawdown']) if drawdown_info['max_drawdown'] != 0 else 0
            
            # Create comprehensive metrics object
            metrics = EnhancedPerformanceMetrics(
                # Basic metrics
                apy=apy,
                dpy=arithmetic_mean,
                total_value_locked=values_array[-1],
                total_fees_earned=sum(self.fees_history),
                impermanent_loss=sum(self.il_history),
                
                # Risk metrics
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                treynor_ratio=benchmark_metrics.get('treynor_ratio', 0),
                information_ratio=benchmark_metrics.get('information_ratio', 0),
                volatility=volatility,
                downside_volatility=downside_volatility,
                
                # Drawdown metrics
                max_drawdown=drawdown_info['max_drawdown'],
                max_drawdown_duration=drawdown_info['max_duration'],
                current_drawdown=drawdown_info['current_drawdown'],
                recovery_factor=recovery_factor,
                
                # Return metrics
                time_weighted_return=total_return,
                money_weighted_return=self._calculate_money_weighted_return(),
                geometric_mean_return=geometric_mean,
                arithmetic_mean_return=arithmetic_mean,
                
                # Risk measures
                value_at_risk_95=var_95,
                value_at_risk_99=var_99,
                conditional_var_95=cvar_95,
                conditional_var_99=cvar_99,
                skewness=skewness,
                kurtosis=kurtosis,
                
                # Performance ratios
                win_rate=win_rate,
                profit_factor=profit_factor,
                payoff_ratio=self._calculate_payoff_ratio(returns_array),
                kelly_criterion=kelly_criterion,
                
                # Market comparison
                alpha=benchmark_metrics.get('alpha', 0),
                beta=benchmark_metrics.get('beta', 0),
                correlation=benchmark_metrics.get('correlation', 0),
                tracking_error=benchmark_metrics.get('tracking_error', 0),
                
                # Additional metrics
                ulcer_index=ulcer_index,
                sterling_ratio=sterling_ratio,
                burke_ratio=burke_ratio,
                tail_ratio=tail_ratio,
                capture_ratio=benchmark_metrics.get('capture_ratio', 0)
            )
            
            # Cache results
            self._metrics_cache = metrics
            self._cache_timestamp = datetime.now()
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating enhanced metrics: {e}")
            return self._empty_enhanced_metrics()
    
    def _calculate_geometric_mean(self, returns: np.ndarray) -> float:
        """Calculate geometric mean return"""
        if len(returns) == 0:
            return 0.0
        
        # Add 1 to convert returns to growth factors
        growth_factors = returns + 1
        
        # Handle negative values (losses greater than 100%)
        growth_factors = np.maximum(growth_factors, 0.001)  # Avoid zero or negative values
        
        # Calculate geometric mean
        geometric_mean = np.power(np.prod(growth_factors), 1.0 / len(growth_factors)) - 1
        return float(geometric_mean)
    
    def _calculate_var(self, returns: np.ndarray, confidence_level: float) -> float:
        """Calculate Value at Risk"""
        if len(returns) == 0:
            return 0.0
        
        percentile = (1 - confidence_level) * 100
        return float(np.percentile(returns, percentile))
    
    def _calculate_cvar(self, returns: np.ndarray, confidence_level: float) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        if len(returns) == 0:
            return 0.0
        
        var = self._calculate_var(returns, confidence_level)
        tail_returns = returns[returns <= var]
        
        return float(np.mean(tail_returns)) if len(tail_returns) > 0 else 0.0
    
    def _calculate_drawdown_metrics(self, values: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive drawdown metrics"""
        if len(values) == 0:
            return {'max_drawdown': 0, 'max_duration': 0, 'current_drawdown': 0}
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(values)
        
        # Calculate drawdowns
        drawdowns = (values - running_max) / running_max
        
        # Find maximum drawdown
        max_drawdown = float(np.min(drawdowns))
        
        # Calculate drawdown duration
        in_drawdown = drawdowns < 0
        durations = []
        current_duration = 0
        
        for is_dd in in_drawdown:
            if is_dd:
                current_duration += 1
            else:
                if current_duration > 0:
                    durations.append(current_duration)
                current_duration = 0
        
        if current_duration > 0:
            durations.append(current_duration)
        
        max_duration = max(durations) if durations else 0
        current_drawdown = float(drawdowns[-1])
        
        return {
            'max_drawdown': max_drawdown,
            'max_duration': max_duration,
            'current_drawdown': current_drawdown
        }
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray, volatility: float) -> float:
        """Calculate Sharpe ratio"""
        if volatility == 0 or len(returns) == 0:
            return 0.0
        
        excess_return = np.mean(returns) * 365 - self.risk_free_rate
        return float(excess_return / volatility)
    
    def _calculate_sortino_ratio(self, returns: np.ndarray, downside_volatility: float) -> float:
        """Calculate Sortino ratio"""
        if downside_volatility == 0 or len(returns) == 0:
            return 0.0
        
        excess_return = np.mean(returns) * 365 - self.risk_free_rate
        return float(excess_return / downside_volatility)
    
    def _calculate_profit_factor(self, returns: np.ndarray) -> float:
        """Calculate profit factor"""
        if len(returns) == 0:
            return 0.0
        
        gains = returns[returns > 0]
        losses = returns[returns < 0]
        
        total_gains = np.sum(gains) if len(gains) > 0 else 0
        total_losses = abs(np.sum(losses)) if len(losses) > 0 else 0
        
        return float(total_gains / total_losses) if total_losses > 0 else float('inf')
    
    def _calculate_payoff_ratio(self, returns: np.ndarray) -> float:
        """Calculate average win / average loss ratio"""
        if len(returns) == 0:
            return 0.0
        
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        
        avg_win = np.mean(wins) if len(wins) > 0 else 0
        avg_loss = abs(np.mean(losses)) if len(losses) > 0 else 0
        
        return float(avg_win / avg_loss) if avg_loss > 0 else float('inf')
    
    def _calculate_kelly_criterion(self, returns: np.ndarray) -> float:
        """Calculate Kelly Criterion for optimal position sizing"""
        if len(returns) == 0:
            return 0.0
        
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        
        if len(wins) == 0 or len(losses) == 0:
            return 0.0
        
        win_rate = len(wins) / len(returns)
        avg_win = np.mean(wins)
        avg_loss = abs(np.mean(losses))
        
        if avg_loss == 0:
            return 0.0
        
        kelly = win_rate - ((1 - win_rate) * avg_loss) / avg_win
        return float(np.clip(kelly, 0, 1))  # Cap at 100%
    
    def _calculate_benchmark_metrics(self, returns: np.ndarray) -> Dict[str, float]:
        """Calculate metrics relative to benchmark"""
        if len(self.benchmark_returns) == 0 or len(returns) != len(self.benchmark_returns):
            return {
                'alpha': 0, 'beta': 0, 'correlation': 0, 
                'tracking_error': 0, 'information_ratio': 0,
                'treynor_ratio': 0, 'capture_ratio': 0
            }
        
        benchmark_array = np.array(self.benchmark_returns[-len(returns):])
        
        # Correlation and beta
        correlation = float(np.corrcoef(returns, benchmark_array)[0, 1]) if len(returns) > 1 else 0
        
        if np.var(benchmark_array) > 0:
            beta = float(np.cov(returns, benchmark_array)[0, 1] / np.var(benchmark_array))
        else:
            beta = 0
        
        # Alpha (Jensen's Alpha)
        portfolio_return = np.mean(returns) * 365
        benchmark_return = np.mean(benchmark_array) * 365
        alpha = portfolio_return - (self.risk_free_rate + beta * (benchmark_return - self.risk_free_rate))
        
        # Tracking error
        excess_returns = returns - benchmark_array
        tracking_error = float(np.std(excess_returns) * np.sqrt(365))
        
        # Information ratio
        information_ratio = float(np.mean(excess_returns) * 365 / tracking_error) if tracking_error > 0 else 0
        
        # Treynor ratio
        treynor_ratio = float((portfolio_return - self.risk_free_rate) / beta) if beta != 0 else 0
        
        # Capture ratio (upside/downside capture)
        up_market = benchmark_array > 0
        down_market = benchmark_array < 0
        
        up_capture = 0
        down_capture = 0
        
        if np.any(up_market) and np.sum(benchmark_array[up_market]) > 0:
            up_capture = np.sum(returns[up_market]) / np.sum(benchmark_array[up_market])
        
        if np.any(down_market) and np.sum(benchmark_array[down_market]) < 0:
            down_capture = np.sum(returns[down_market]) / np.sum(benchmark_array[down_market])
        
        capture_ratio = up_capture / abs(down_capture) if down_capture != 0 else 0
        
        return {
            'alpha': float(alpha),
            'beta': float(beta),
            'correlation': correlation,
            'tracking_error': tracking_error,
            'information_ratio': information_ratio,
            'treynor_ratio': treynor_ratio,
            'capture_ratio': float(capture_ratio)
        }
    
    def _calculate_ulcer_index(self, values: np.ndarray) -> float:
        """Calculate Ulcer Index (alternative to standard deviation for downside risk)"""
        if len(values) == 0:
            return 0.0
        
        running_max = np.maximum.accumulate(values)
        drawdowns = (values - running_max) / running_max
        squared_drawdowns = drawdowns ** 2
        
        return float(np.sqrt(np.mean(squared_drawdowns)))
    
    def _calculate_sterling_ratio(self, annual_return: float, max_drawdown: float) -> float:
        """Calculate Sterling Ratio"""
        if max_drawdown == 0:
            return 0.0
        
        return float(annual_return / abs(max_drawdown))
    
    def _calculate_burke_ratio(self, annual_return: float, values: np.ndarray) -> float:
        """Calculate Burke Ratio"""
        if len(values) == 0:
            return 0.0
        
        running_max = np.maximum.accumulate(values)
        drawdowns = (values - running_max) / running_max
        squared_drawdowns = drawdowns ** 2
        
        sum_squared_drawdowns = np.sum(squared_drawdowns)
        
        if sum_squared_drawdowns == 0:
            return 0.0
        
        return float(annual_return / np.sqrt(sum_squared_drawdowns))
    
    def _calculate_tail_ratio(self, returns: np.ndarray) -> float:
        """Calculate Tail Ratio (95th percentile / 5th percentile)"""
        if len(returns) == 0:
            return 0.0
        
        p95 = np.percentile(returns, 95)
        p5 = np.percentile(returns, 5)
        
        if p5 == 0:
            return 0.0
        
        return float(p95 / abs(p5))
    
    def _calculate_money_weighted_return(self) -> float:
        """Calculate money-weighted return (IRR approximation)"""
        if len(self.values) < 2:
            return 0.0
        
        # Simple approximation using initial and final values
        initial_value = self.values[0]
        final_value = self.values[-1]
        periods = len(self.values)
        
        if initial_value <= 0 or periods <= 1:
            return 0.0
        
        return float((final_value / initial_value) ** (1/periods) - 1)
    
    async def _update_prometheus_metrics(self) -> None:
        """Update Prometheus metrics with latest values"""
        try:
            if not self.values:
                return
            
            # Update basic metrics
            ENHANCED_TVL.labels(agent=self.agent_name, pool="all").set(self.values[-1])
            
            # Calculate and update advanced metrics (cached)
            metrics = await self.calculate_enhanced_metrics()
            
            ENHANCED_SHARPE.labels(agent=self.agent_name, timeframe="daily").set(metrics.sharpe_ratio)
            VAR_GAUGE.labels(agent=self.agent_name, confidence_level="95").set(abs(metrics.value_at_risk_95))
            VAR_GAUGE.labels(agent=self.agent_name, confidence_level="99").set(abs(metrics.value_at_risk_99))
            BETA_GAUGE.labels(agent=self.agent_name, benchmark="market").set(metrics.beta)
            ALPHA_GAUGE.labels(agent=self.agent_name, benchmark="market").set(metrics.alpha)
            SORTINO_GAUGE.labels(agent=self.agent_name).set(metrics.sortino_ratio)
            CALMAR_GAUGE.labels(agent=self.agent_name).set(metrics.calmar_ratio)
            INFORMATION_RATIO.labels(agent=self.agent_name, benchmark="market").set(metrics.information_ratio)
            TREYNOR_RATIO.labels(agent=self.agent_name).set(metrics.treynor_ratio)
            MAX_DRAWDOWN_DURATION.labels(agent=self.agent_name).set(metrics.max_drawdown_duration)
            PROFIT_FACTOR.labels(agent=self.agent_name).set(metrics.profit_factor)
            RECOVERY_FACTOR.labels(agent=self.agent_name).set(metrics.recovery_factor)
            
        except Exception as e:
            logger.error(f"Error updating Prometheus metrics: {e}")
    
    def _trim_history(self) -> None:
        """Trim history to maintain memory efficiency"""
        max_points = 5000
        if len(self.returns) > max_points:
            # Keep last 5000 points
            self.returns = self.returns[-max_points:]
            self.values = self.values[-max_points:]
            self.fees_history = self.fees_history[-max_points:]
            self.il_history = self.il_history[-max_points:]
            self.volumes = self.volumes[-max_points:]
            self.timestamps = self.timestamps[-max_points:]
            
            if self.benchmark_returns:
                self.benchmark_returns = self.benchmark_returns[-max_points:]
    
    def _empty_enhanced_metrics(self) -> EnhancedPerformanceMetrics:
        """Return empty enhanced metrics for error cases"""
        return EnhancedPerformanceMetrics(
            apy=0.0, dpy=0.0, total_value_locked=0.0, total_fees_earned=0.0,
            impermanent_loss=0.0, sharpe_ratio=0.0, sortino_ratio=0.0,
            calmar_ratio=0.0, treynor_ratio=0.0, information_ratio=0.0,
            volatility=0.0, downside_volatility=0.0, max_drawdown=0.0,
            max_drawdown_duration=0, current_drawdown=0.0, recovery_factor=0.0,
            time_weighted_return=0.0, money_weighted_return=0.0,
            geometric_mean_return=0.0, arithmetic_mean_return=0.0,
            value_at_risk_95=0.0, value_at_risk_99=0.0, conditional_var_95=0.0,
            conditional_var_99=0.0, skewness=0.0, kurtosis=0.0, win_rate=0.0,
            profit_factor=0.0, payoff_ratio=0.0, kelly_criterion=0.0,
            alpha=0.0, beta=0.0, correlation=0.0, tracking_error=0.0,
            ulcer_index=0.0, sterling_ratio=0.0, burke_ratio=0.0,
            tail_ratio=0.0, capture_ratio=0.0
        )
    
    async def export_metrics_report(self, output_path: str) -> Dict[str, Any]:
        """
        Export comprehensive metrics report
        
        Args:
            output_path: Path to save the report
            
        Returns:
            Dictionary containing the full metrics report
        """
        try:
            metrics = await self.calculate_enhanced_metrics()
            
            report = {
                'agent_name': self.agent_name,
                'report_timestamp': datetime.now().isoformat(),
                'data_period': {
                    'start': self.timestamps[0].isoformat() if self.timestamps else None,
                    'end': self.timestamps[-1].isoformat() if self.timestamps else None,
                    'total_periods': len(self.returns)
                },
                'metrics': asdict(metrics),
                'raw_data_summary': {
                    'total_return_periods': len(self.returns),
                    'total_fees': sum(self.fees_history),
                    'total_volume': sum(self.volumes),
                    'final_value': self.values[-1] if self.values else 0
                }
            }
            
            # Save to file
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Metrics report exported to {output_path}")
            return report
            
        except Exception as e:
            logger.error(f"Error exporting metrics report: {e}")
            return {}
    
    def reset(self) -> None:
        """Reset all tracking data"""
        self.returns.clear()
        self.timestamps.clear()
        self.values.clear()
        self.fees_history.clear()
        self.il_history.clear()
        self.volumes.clear()
        self.positions.clear()
        self.trades.clear()
        self.benchmark_returns.clear()
        self._metrics_cache = None
        self._cache_timestamp = None
        
        logger.info(f"Performance tracker reset for agent: {self.agent_name}")