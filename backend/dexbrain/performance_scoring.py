"""Advanced Performance Scoring System for DexBrain Network"""

import math
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from .schemas import PerformanceMetrics, LiquidityPosition


class ScoringMethod(Enum):
    """Scoring methodology options"""
    WEIGHTED_AVERAGE = "weighted_average"
    RISK_ADJUSTED = "risk_adjusted"
    CONSISTENCY_FOCUSED = "consistency_focused"
    VOLUME_WEIGHTED = "volume_weighted"


@dataclass
class ScoringWeights:
    """Configurable weights for scoring components"""
    profitability: float = 0.25  # 25%
    consistency: float = 0.20    # 20%
    risk_management: float = 0.20 # 20%
    volume: float = 0.15         # 15%
    duration: float = 0.10       # 10%
    data_quality: float = 0.10   # 10%
    
    def __post_init__(self):
        """Validate weights sum to 1.0"""
        total = (self.profitability + self.consistency + self.risk_management + 
                self.volume + self.duration + self.data_quality)
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total}")


@dataclass
class AgentPerformanceProfile:
    """Comprehensive agent performance profile"""
    agent_id: str
    total_score: float
    profitability_score: float
    consistency_score: float
    risk_score: float
    volume_score: float
    duration_score: float
    data_quality_score: float
    percentile_rank: float
    performance_tier: str
    last_updated: str
    
    # Detailed metrics
    total_positions: int = 0
    win_rate: float = 0.0
    average_apr: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    total_volume: float = 0.0
    average_position_duration: float = 0.0
    data_submissions: int = 0
    
    # Risk metrics
    volatility: float = 0.0
    var_95: float = 0.0  # Value at Risk (95%)
    calmar_ratio: float = 0.0
    
    # Consistency metrics
    return_consistency: float = 0.0
    profit_factor: float = 0.0
    
    def get_tier(self) -> str:
        """Determine performance tier based on score"""
        if self.total_score >= 85:
            return "Elite"
        elif self.total_score >= 70:
            return "Advanced"
        elif self.total_score >= 55:
            return "Intermediate"
        elif self.total_score >= 40:
            return "Novice"
        else:
            return "Developing"


class PerformanceScorer:
    """Advanced performance scoring engine"""
    
    def __init__(self, scoring_method: ScoringMethod = ScoringMethod.WEIGHTED_AVERAGE):
        self.scoring_method = scoring_method
        self.weights = ScoringWeights()
        self.benchmark_data = {}  # Store benchmark metrics
    
    def calculate_agent_score(
        self,
        agent_id: str,
        performance_metrics: List[PerformanceMetrics],
        positions: List[LiquidityPosition],
        time_window_days: int = 30
    ) -> AgentPerformanceProfile:
        """Calculate comprehensive performance score for an agent
        
        Args:
            agent_id: Agent identifier
            performance_metrics: List of performance metrics
            positions: List of liquidity positions
            time_window_days: Time window for calculations
            
        Returns:
            Complete performance profile
        """
        if not performance_metrics:
            return self._create_empty_profile(agent_id)
        
        # Filter recent data
        cutoff_date = datetime.now() - timedelta(days=time_window_days)
        recent_metrics = [
            m for m in performance_metrics 
            if datetime.fromisoformat(m.timestamp) >= cutoff_date
        ]
        
        if not recent_metrics:
            return self._create_empty_profile(agent_id)
        
        # Calculate component scores
        profitability_score = self._calculate_profitability_score(recent_metrics)
        consistency_score = self._calculate_consistency_score(recent_metrics)
        risk_score = self._calculate_risk_score(recent_metrics)
        volume_score = self._calculate_volume_score(recent_metrics)
        duration_score = self._calculate_duration_score(recent_metrics)
        data_quality_score = self._calculate_data_quality_score(recent_metrics, positions)
        
        # Calculate total score
        total_score = (
            profitability_score * self.weights.profitability +
            consistency_score * self.weights.consistency +
            risk_score * self.weights.risk_management +
            volume_score * self.weights.volume +
            duration_score * self.weights.duration +
            data_quality_score * self.weights.data_quality
        )
        
        # Calculate detailed metrics
        win_rate = sum(1 for m in recent_metrics if m.win) / len(recent_metrics)
        avg_apr = np.mean([m.apr for m in recent_metrics])
        total_volume = sum(m.total_return_usd for m in recent_metrics if m.total_return_usd > 0)
        
        returns = [m.total_return_percent for m in recent_metrics]
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        max_drawdown = self._calculate_max_drawdown(returns)
        volatility = np.std(returns) if returns else 0
        
        # Create performance profile
        profile = AgentPerformanceProfile(
            agent_id=agent_id,
            total_score=min(total_score, 100.0),
            profitability_score=profitability_score,
            consistency_score=consistency_score,
            risk_score=risk_score,
            volume_score=volume_score,
            duration_score=duration_score,
            data_quality_score=data_quality_score,
            percentile_rank=0.0,  # Will be calculated when comparing agents
            performance_tier="",
            last_updated=datetime.now().isoformat(),
            total_positions=len(recent_metrics),
            win_rate=win_rate,
            average_apr=avg_apr,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            total_volume=total_volume,
            average_position_duration=np.mean([m.duration_hours for m in recent_metrics]),
            volatility=volatility,
            var_95=np.percentile(returns, 5) if returns else 0,
            calmar_ratio=avg_apr / abs(max_drawdown) if max_drawdown != 0 else 0,
            return_consistency=1 - (np.std(returns) / np.mean(returns)) if returns and np.mean(returns) != 0 else 0,
            profit_factor=self._calculate_profit_factor(recent_metrics)
        )
        
        profile.performance_tier = profile.get_tier()
        return profile
    
    def _calculate_profitability_score(self, metrics: List[PerformanceMetrics]) -> float:
        """Calculate profitability component score"""
        if not metrics:
            return 0.0
        
        # Base score on average APR
        avg_apr = np.mean([m.apr for m in metrics])
        
        # Normalize APR to 0-100 scale (assuming 50% APR = 100 score)
        base_score = min(avg_apr * 2, 100)
        
        # Bonus for consistency of positive returns
        positive_returns = sum(1 for m in metrics if m.net_profit_usd > 0)
        consistency_bonus = (positive_returns / len(metrics)) * 20
        
        return min(base_score + consistency_bonus, 100)
    
    def _calculate_consistency_score(self, metrics: List[PerformanceMetrics]) -> float:
        """Calculate consistency component score"""
        if not metrics:
            return 0.0
        
        returns = [m.total_return_percent for m in metrics]
        
        # Coefficient of variation (lower is better)
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if mean_return == 0:
            return 0.0
        
        cv = std_return / abs(mean_return)
        
        # Convert to score (lower CV = higher score)
        consistency_score = max(0, 100 - (cv * 50))
        
        # Bonus for low drawdown periods
        drawdown_score = 100 - (self._calculate_max_drawdown(returns) * 100)
        
        return min((consistency_score + drawdown_score) / 2, 100)
    
    def _calculate_risk_score(self, metrics: List[PerformanceMetrics]) -> float:
        """Calculate risk management component score"""
        if not metrics:
            return 0.0
        
        returns = [m.total_return_percent for m in metrics]
        
        # Sharpe ratio component
        sharpe = self._calculate_sharpe_ratio(returns)
        sharpe_score = min(sharpe * 20, 50)  # Cap at 50 points
        
        # Maximum drawdown component (lower is better)
        max_dd = abs(self._calculate_max_drawdown(returns))
        dd_score = max(0, 50 - (max_dd * 100))
        
        return min(sharpe_score + dd_score, 100)
    
    def _calculate_volume_score(self, metrics: List[PerformanceMetrics]) -> float:
        """Calculate volume component score"""
        if not metrics:
            return 0.0
        
        total_volume = sum(abs(m.total_return_usd) for m in metrics)
        
        # Logarithmic scaling for volume (rewards high volume, diminishing returns)
        if total_volume <= 0:
            return 0.0
        
        # Base score increases with volume
        volume_score = min(math.log10(total_volume) * 10, 100)
        
        return max(0, volume_score)
    
    def _calculate_duration_score(self, metrics: List[PerformanceMetrics]) -> float:
        """Calculate position duration component score"""
        if not metrics:
            return 0.0
        
        durations = [m.duration_hours for m in metrics]
        avg_duration = np.mean(durations)
        
        # Optimal duration range: 24-168 hours (1-7 days)
        if 24 <= avg_duration <= 168:
            duration_score = 100
        elif avg_duration < 24:
            # Penalty for very short positions
            duration_score = (avg_duration / 24) * 100
        else:
            # Penalty for very long positions
            duration_score = max(0, 100 - ((avg_duration - 168) / 168) * 50)
        
        return min(duration_score, 100)
    
    def _calculate_data_quality_score(
        self,
        metrics: List[PerformanceMetrics],
        positions: List[LiquidityPosition]
    ) -> float:
        """Calculate data quality component score"""
        if not metrics:
            return 0.0
        
        # Completeness score
        complete_metrics = 0
        for m in metrics:
            if all([
                m.total_return_usd is not None,
                m.fees_earned_usd is not None,
                m.impermanent_loss_usd is not None,
                m.gas_costs_usd is not None,
                m.duration_hours > 0
            ]):
                complete_metrics += 1
        
        completeness_score = (complete_metrics / len(metrics)) * 100
        
        # Frequency score (regular submissions)
        if len(metrics) >= 10:
            frequency_score = 100
        else:
            frequency_score = (len(metrics) / 10) * 100
        
        return min((completeness_score + frequency_score) / 2, 100)
    
    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio"""
        if not returns or len(returns) < 2:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        # Assuming risk-free rate of 0 for simplicity
        return mean_return / std_return
    
    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """Calculate maximum drawdown"""
        if not returns:
            return 0.0
        
        cumulative = np.cumprod([1 + r/100 for r in returns])
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        
        return np.min(drawdown)
    
    def _calculate_profit_factor(self, metrics: List[PerformanceMetrics]) -> float:
        """Calculate profit factor (gross profit / gross loss)"""
        if not metrics:
            return 0.0
        
        gross_profit = sum(m.net_profit_usd for m in metrics if m.net_profit_usd > 0)
        gross_loss = abs(sum(m.net_profit_usd for m in metrics if m.net_profit_usd < 0))
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        
        return gross_profit / gross_loss
    
    def _create_empty_profile(self, agent_id: str) -> AgentPerformanceProfile:
        """Create empty performance profile for new agents"""
        return AgentPerformanceProfile(
            agent_id=agent_id,
            total_score=0.0,
            profitability_score=0.0,
            consistency_score=0.0,
            risk_score=0.0,
            volume_score=0.0,
            duration_score=0.0,
            data_quality_score=0.0,
            percentile_rank=0.0,
            performance_tier="Developing",
            last_updated=datetime.now().isoformat()
        )
    
    def rank_agents(self, profiles: List[AgentPerformanceProfile]) -> List[AgentPerformanceProfile]:
        """Rank agents and calculate percentile ranks"""
        if not profiles:
            return []
        
        # Sort by total score (descending)
        sorted_profiles = sorted(profiles, key=lambda p: p.total_score, reverse=True)
        
        # Calculate percentile ranks
        total_agents = len(sorted_profiles)
        for i, profile in enumerate(sorted_profiles):
            profile.percentile_rank = ((total_agents - i) / total_agents) * 100
        
        return sorted_profiles
    
    def get_network_benchmarks(self, profiles: List[AgentPerformanceProfile]) -> Dict[str, float]:
        """Calculate network-wide benchmark metrics"""
        if not profiles:
            return {}
        
        scores = [p.total_score for p in profiles]
        aprs = [p.average_apr for p in profiles]
        win_rates = [p.win_rate for p in profiles]
        
        return {
            'median_score': np.median(scores),
            'top_10_percent_score': np.percentile(scores, 90),
            'median_apr': np.median(aprs),
            'median_win_rate': np.median(win_rates),
            'total_agents': len(profiles),
            'elite_agents': len([p for p in profiles if p.performance_tier == "Elite"]),
            'advanced_agents': len([p for p in profiles if p.performance_tier == "Advanced"])
        }