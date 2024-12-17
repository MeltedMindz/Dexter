from datetime import datetime, timedelta
from typing import Dict, List
from dataclasses import dataclass
import numpy as np
from prometheus_client import Gauge, Counter
from utils.error_handler import ErrorHandler
from utils.memory_monitor import MemoryMonitor

# Add Prometheus metrics
TOTAL_TVL = Gauge('dexter_total_tvl', 'Total Value Locked across all pools')
APY_GAUGE = Gauge('dexter_apy', 'Current APY', ['risk_profile'])
DPY_GAUGE = Gauge('dexter_dpy', 'Current DPY', ['risk_profile'])

@dataclass
class PerformanceMetrics:
    apy: float  # Annual Percentage Yield
    dpy: float  # Daily Percentage Yield
    total_value_locked: float
    total_fees_earned: float
    impermanent_loss: float
    sharpe_ratio: float
    volatility: float
    time_weighted_return: float

class PerformanceTracker:
    def __init__(self):
        self.daily_returns = []
        self.total_fees = 0
        self.initial_tvl = 0
        self.current_tvl = 0
        
    def update_metrics(
        self,
        current_tvl: float,
        daily_fees: float,
        il: float,
        timestamp: datetime
    ):
        """Update performance metrics with new data"""
        
        if self.initial_tvl == 0:
            self.initial_tvl = current_tvl
            
        self.current_tvl = current_tvl
        self.total_fees += daily_fees
        
        # Calculate daily return
        daily_return = (
            (current_tvl - self.initial_tvl + daily_fees - il) /
            self.initial_tvl
        )
        self.daily_returns.append(daily_return)
        
    def calculate_metrics(self) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        
        # Calculate returns
        total_return = (
            (self.current_tvl - self.initial_tvl + self.total_fees) /
            self.initial_tvl
        )
        
        # Calculate APY using compound interest formula
        days_elapsed = len(self.daily_returns)
        apy = ((1 + total_return) ** (365 / days_elapsed)) - 1
        
        # Calculate DPY (average daily return)
        dpy = np.mean(self.daily_returns)
        
        # Calculate volatility and Sharpe ratio
        volatility = np.std(self.daily_returns) * np.sqrt(365)
        risk_free_rate = 0.04  # 4% annual risk-free rate
        sharpe_ratio = (apy - risk_free_rate) / volatility
        
        return PerformanceMetrics(
            apy=apy,
            dpy=dpy,
            total_value_locked=self.current_tvl,
            total_fees_earned=self.total_fees,
            impermanent_loss=self._calculate_total_il(),
            sharpe_ratio=sharpe_ratio,
            volatility=volatility,
            time_weighted_return=self._calculate_twr()
        )
    
class PerformanceTracker:
    def __init__(self, error_handler: ErrorHandler):
        self.error_handler = error_handler
        self.memory_monitor = MemoryMonitor()
        
    @memory_monitor.monitor
    async def update_metrics(
        self,
        risk_profile: RiskProfile,
        current_tvl: float,
        daily_fees: float,
        il: float,
        timestamp: datetime
    ):
        try:
            # Update Prometheus metrics
            TOTAL_TVL.set(current_tvl)
            
            metrics = self.calculate_metrics()
            APY_GAUGE.labels(risk_profile=risk_profile.value).set(metrics.apy)
            DPY_GAUGE.labels(risk_profile=risk_profile.value).set(metrics.dpy)
            
        except Exception as e:
            await self.error_handler._handle_error(e, "performance_tracker")
