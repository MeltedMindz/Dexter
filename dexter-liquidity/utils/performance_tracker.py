from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
import numpy as np
from prometheus_client import Gauge, Counter
from utils.error_handler import ErrorHandler
from utils.memory_monitor import MemoryMonitor

# Add Prometheus metrics
TOTAL_TVL = Gauge('dexter_total_tvl', 'Total Value Locked across all pools')
APY_GAUGE = Gauge('dexter_apy', 'Current APY', ['risk_profile'])
DPY_GAUGE = Gauge('dexter_dpy', 'Current DPY', ['risk_profile'])
IL_GAUGE = Gauge('dexter_impermanent_loss', 'Impermanent Loss', ['risk_profile'])
SHARPE_GAUGE = Gauge('dexter_sharpe_ratio', 'Sharpe Ratio', ['risk_profile'])

@dataclass
class PerformanceMetrics:
    """Performance metrics data structure"""
    apy: float  # Annual Percentage Yield
    dpy: float  # Daily Percentage Yield
    total_value_locked: float
    total_fees_earned: float
    impermanent_loss: float
    sharpe_ratio: float
    volatility: float
    time_weighted_return: float
    max_drawdown: float
    win_rate: float


class PerformanceTracker:
    """Unified performance tracking with monitoring and error handling"""
    
    def __init__(self, error_handler: Optional[ErrorHandler] = None):
        self.error_handler = error_handler or ErrorHandler()
        self.memory_monitor = MemoryMonitor()
        
        # Performance data
        self.daily_returns: List[float] = []
        self.total_fees = 0.0
        self.initial_tvl = 0.0
        self.current_tvl = 0.0
        self.il_values: List[float] = []
        self.timestamps: List[datetime] = []
        
    async def update_metrics(
        self,
        current_tvl: float,
        daily_fees: float,
        il: float,
        timestamp: datetime,
        risk_profile: Optional[str] = None
    ) -> None:
        """Update performance metrics with new data
        
        Args:
            current_tvl: Current total value locked
            daily_fees: Fees earned today
            il: Current impermanent loss
            timestamp: Current timestamp
            risk_profile: Risk profile for Prometheus labels
        """
        try:
            # Initialize if first data point
            if self.initial_tvl == 0:
                self.initial_tvl = current_tvl
                
            self.current_tvl = current_tvl
            self.total_fees += daily_fees
            self.il_values.append(il)
            self.timestamps.append(timestamp)
            
            # Calculate daily return
            if len(self.daily_returns) > 0:
                previous_value = self.initial_tvl + sum(self.daily_returns) * self.initial_tvl
                daily_return = (current_tvl + daily_fees - il - previous_value) / previous_value
            else:
                daily_return = (current_tvl + daily_fees - il - self.initial_tvl) / self.initial_tvl
                
            self.daily_returns.append(daily_return)
            
            # Update Prometheus metrics
            TOTAL_TVL.set(current_tvl)
            
            if risk_profile:
                metrics = await self.calculate_metrics()
                APY_GAUGE.labels(risk_profile=risk_profile).set(metrics.apy)
                DPY_GAUGE.labels(risk_profile=risk_profile).set(metrics.dpy)
                IL_GAUGE.labels(risk_profile=risk_profile).set(metrics.impermanent_loss)
                SHARPE_GAUGE.labels(risk_profile=risk_profile).set(metrics.sharpe_ratio)
                
        except Exception as e:
            await self.error_handler.handle_error(e, "performance_tracker.update_metrics")
    
    async def calculate_metrics(self) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics
        
        Returns:
            PerformanceMetrics object with all calculated metrics
        """
        try:
            if not self.daily_returns:
                return self._empty_metrics()
            
            # Calculate returns
            total_return = sum(self.daily_returns)
            
            # Calculate APY using compound interest formula
            days_elapsed = len(self.daily_returns)
            if days_elapsed > 0:
                apy = ((1 + total_return) ** (365 / days_elapsed)) - 1
            else:
                apy = 0.0
            
            # Calculate DPY (average daily return)
            dpy = np.mean(self.daily_returns) if self.daily_returns else 0.0
            
            # Calculate volatility and Sharpe ratio
            if len(self.daily_returns) > 1:
                volatility = np.std(self.daily_returns) * np.sqrt(365)
                risk_free_rate = 0.04  # 4% annual risk-free rate
                sharpe_ratio = (apy - risk_free_rate) / volatility if volatility > 0 else 0.0
            else:
                volatility = 0.0
                sharpe_ratio = 0.0
            
            # Calculate time-weighted return
            twr = self._calculate_twr()
            
            # Calculate max drawdown
            max_drawdown = self._calculate_max_drawdown()
            
            # Calculate win rate
            win_rate = self._calculate_win_rate()
            
            # Calculate total impermanent loss
            total_il = sum(self.il_values) if self.il_values else 0.0
            
            return PerformanceMetrics(
                apy=apy,
                dpy=dpy,
                total_value_locked=self.current_tvl,
                total_fees_earned=self.total_fees,
                impermanent_loss=total_il,
                sharpe_ratio=sharpe_ratio,
                volatility=volatility,
                time_weighted_return=twr,
                max_drawdown=max_drawdown,
                win_rate=win_rate
            )
            
        except Exception as e:
            await self.error_handler.handle_error(e, "performance_tracker.calculate_metrics")
            return self._empty_metrics()
    
    def _calculate_twr(self) -> float:
        """Calculate time-weighted return"""
        if not self.daily_returns:
            return 0.0
        
        # Geometric mean of (1 + daily_return)
        product = 1.0
        for daily_return in self.daily_returns:
            product *= (1 + daily_return)
        
        return product - 1.0
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        if not self.daily_returns:
            return 0.0
        
        cumulative_returns = np.cumprod([1 + r for r in self.daily_returns])
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        
        return float(np.min(drawdowns))
    
    def _calculate_win_rate(self) -> float:
        """Calculate win rate (percentage of positive returns)"""
        if not self.daily_returns:
            return 0.0
        
        positive_returns = sum(1 for r in self.daily_returns if r > 0)
        return positive_returns / len(self.daily_returns)
    
    def _empty_metrics(self) -> PerformanceMetrics:
        """Return empty metrics for error cases"""
        return PerformanceMetrics(
            apy=0.0,
            dpy=0.0,
            total_value_locked=self.current_tvl,
            total_fees_earned=self.total_fees,
            impermanent_loss=0.0,
            sharpe_ratio=0.0,
            volatility=0.0,
            time_weighted_return=0.0,
            max_drawdown=0.0,
            win_rate=0.0
        )
    
    def reset(self) -> None:
        """Reset all tracking data"""
        self.daily_returns.clear()
        self.il_values.clear()
        self.timestamps.clear()
        self.total_fees = 0.0
        self.initial_tvl = 0.0
        self.current_tvl = 0.0
