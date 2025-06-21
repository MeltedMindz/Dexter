"""
Real-time Performance Monitoring Dashboard
Advanced visualization and monitoring of trading performance
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import numpy as np
import json
import asyncio
import logging
from pathlib import Path
from .enhanced_performance_tracker import EnhancedPerformanceTracker, EnhancedPerformanceMetrics
from .portfolio_risk_analyzer import PortfolioRiskAnalyzer, RiskAlert, PositionRisk

logger = logging.getLogger(__name__)

@dataclass
class DashboardData:
    """Dashboard data structure"""
    timestamp: datetime
    performance_metrics: EnhancedPerformanceMetrics
    risk_alerts: List[RiskAlert]
    position_risks: List[PositionRisk]
    portfolio_summary: Dict[str, Any]
    real_time_stats: Dict[str, Any]

@dataclass
class PerformanceAlert:
    """Performance-based alert"""
    alert_id: str
    timestamp: datetime
    metric_name: str
    current_value: float
    threshold: float
    severity: str
    message: str
    trend: str  # "improving", "deteriorating", "stable"

class PerformanceDashboard:
    """
    Real-time performance monitoring and alerting dashboard
    """
    
    def __init__(self, 
                 performance_tracker: EnhancedPerformanceTracker,
                 risk_analyzer: PortfolioRiskAnalyzer,
                 output_dir: str = "dashboard_data"):
        """
        Initialize performance dashboard
        
        Args:
            performance_tracker: Enhanced performance tracker instance
            risk_analyzer: Portfolio risk analyzer instance
            output_dir: Directory to save dashboard data
        """
        self.performance_tracker = performance_tracker
        self.risk_analyzer = risk_analyzer
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Dashboard state
        self.dashboard_data: List[DashboardData] = []
        self.performance_alerts: List[PerformanceAlert] = []
        
        # Alert thresholds
        self.alert_thresholds = {
            'sharpe_ratio': {'min': 1.0, 'target': 2.0},
            'max_drawdown': {'max': -0.15, 'warning': -0.10},
            'win_rate': {'min': 0.55, 'target': 0.65},
            'volatility': {'max': 0.30, 'warning': 0.25},
            'var_95': {'max': -0.05, 'warning': -0.03},
            'profit_factor': {'min': 1.2, 'target': 1.5}
        }
        
        # Trend tracking
        self.metric_history: Dict[str, List[float]] = {}
        
        logger.info("Performance dashboard initialized")
    
    async def update_dashboard(self) -> DashboardData:
        """
        Update dashboard with latest performance data
        
        Returns:
            Latest dashboard data
        """
        try:
            timestamp = datetime.now()
            
            # Get latest performance metrics
            performance_metrics = await self.performance_tracker.calculate_enhanced_metrics()
            
            # Get risk analysis
            positions = list(self.risk_analyzer.positions.values())
            portfolio_risk = await self.risk_analyzer.analyze_portfolio_risk(
                positions, self.risk_analyzer.returns_history
            )
            
            # Check for risk alerts
            risk_alerts = await self.risk_analyzer.check_risk_alerts()
            
            # Analyze individual position risks
            position_risks = []
            for pos_id in self.risk_analyzer.positions:
                pos_risk = await self.risk_analyzer.analyze_position_risk(pos_id)
                if pos_risk:
                    position_risks.append(pos_risk)
            
            # Generate portfolio summary
            portfolio_summary = await self._generate_portfolio_summary(
                performance_metrics, portfolio_risk
            )
            
            # Generate real-time stats
            real_time_stats = await self._generate_real_time_stats(performance_metrics)
            
            # Check for performance alerts
            perf_alerts = await self._check_performance_alerts(performance_metrics)
            self.performance_alerts.extend(perf_alerts)
            
            # Create dashboard data
            dashboard_data = DashboardData(
                timestamp=timestamp,
                performance_metrics=performance_metrics,
                risk_alerts=risk_alerts,
                position_risks=position_risks,
                portfolio_summary=portfolio_summary,
                real_time_stats=real_time_stats
            )
            
            # Store dashboard data
            self.dashboard_data.append(dashboard_data)
            
            # Keep last 1000 data points
            if len(self.dashboard_data) > 1000:
                self.dashboard_data = self.dashboard_data[-1000:]
            
            # Update metric history
            await self._update_metric_history(performance_metrics)
            
            # Save to file
            await self._save_dashboard_data(dashboard_data)
            
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Error updating dashboard: {e}")
            return self._empty_dashboard_data()
    
    async def generate_performance_report(self, 
                                        period_days: int = 30) -> Dict[str, Any]:
        """
        Generate comprehensive performance report
        
        Args:
            period_days: Number of days to include in report
            
        Returns:
            Comprehensive performance report
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=period_days)
            recent_data = [
                data for data in self.dashboard_data 
                if data.timestamp >= cutoff_date
            ]
            
            if not recent_data:
                return {}
            
            # Latest metrics
            latest_metrics = recent_data[-1].performance_metrics
            
            # Calculate trends
            trends = await self._calculate_metric_trends(recent_data)
            
            # Risk summary
            risk_summary = await self._summarize_risks(recent_data)
            
            # Performance highlights
            highlights = await self._generate_highlights(recent_data)
            
            # Recommendations
            recommendations = await self._generate_recommendations(latest_metrics, trends)
            
            report = {
                'report_metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'period_days': period_days,
                    'data_points': len(recent_data),
                    'agent_name': self.performance_tracker.agent_name
                },
                'performance_summary': {
                    'current_metrics': asdict(latest_metrics),
                    'trends': trends,
                    'highlights': highlights
                },
                'risk_analysis': risk_summary,
                'alerts_summary': {
                    'total_alerts': len(self.performance_alerts),
                    'recent_alerts': len([
                        alert for alert in self.performance_alerts
                        if alert.timestamp >= cutoff_date
                    ]),
                    'critical_alerts': len([
                        alert for alert in self.performance_alerts
                        if alert.severity == "CRITICAL" and alert.timestamp >= cutoff_date
                    ])
                },
                'recommendations': recommendations,
                'detailed_metrics': self._format_detailed_metrics(latest_metrics)
            }
            
            # Save report
            report_path = self.output_dir / f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Performance report generated: {report_path}")
            return report
            
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return {}
    
    async def get_real_time_dashboard_data(self) -> Dict[str, Any]:
        """
        Get real-time dashboard data for web interface
        
        Returns:
            Dashboard data formatted for web display
        """
        try:
            if not self.dashboard_data:
                await self.update_dashboard()
            
            latest_data = self.dashboard_data[-1] if self.dashboard_data else None
            if not latest_data:
                return {}
            
            # Format for web display
            web_data = {
                'timestamp': latest_data.timestamp.isoformat(),
                'performance': {
                    'apy': f"{latest_data.performance_metrics.apy:.2%}",
                    'sharpe_ratio': f"{latest_data.performance_metrics.sharpe_ratio:.2f}",
                    'max_drawdown': f"{latest_data.performance_metrics.max_drawdown:.2%}",
                    'win_rate': f"{latest_data.performance_metrics.win_rate:.1%}",
                    'volatility': f"{latest_data.performance_metrics.volatility:.1%}",
                    'total_value': f"${latest_data.performance_metrics.total_value_locked:,.2f}",
                    'total_fees': f"${latest_data.performance_metrics.total_fees_earned:,.2f}"
                },
                'risk_metrics': {
                    'var_95': f"{abs(latest_data.performance_metrics.value_at_risk_95):.2%}",
                    'var_99': f"{abs(latest_data.performance_metrics.value_at_risk_99):.2%}",
                    'beta': f"{latest_data.performance_metrics.beta:.2f}",
                    'correlation': f"{latest_data.performance_metrics.correlation:.2f}"
                },
                'alerts': {
                    'total_count': len(latest_data.risk_alerts),
                    'high_severity': len([
                        alert for alert in latest_data.risk_alerts 
                        if alert.severity == "HIGH"
                    ]),
                    'recent_alerts': [
                        {
                            'message': alert.message,
                            'severity': alert.severity,
                            'timestamp': alert.timestamp.strftime('%H:%M:%S')
                        }
                        for alert in latest_data.risk_alerts[-5:]  # Last 5 alerts
                    ]
                },
                'position_risks': [
                    {
                        'token_pair': risk.token_pair,
                        'risk_level': risk.risk_level.value,
                        'var_1d': f"{risk.var_1d:.2%}",
                        'concentration': f"{risk.concentration_risk:.1%}"
                    }
                    for risk in latest_data.position_risks
                ],
                'portfolio_summary': latest_data.portfolio_summary,
                'real_time_stats': latest_data.real_time_stats
            }
            
            return web_data
            
        except Exception as e:
            logger.error(f"Error getting real-time dashboard data: {e}")
            return {}
    
    async def _generate_portfolio_summary(self, 
                                        metrics: EnhancedPerformanceMetrics,
                                        portfolio_risk: Any) -> Dict[str, Any]:
        """Generate portfolio summary statistics"""
        try:
            # Performance score (0-100)
            performance_score = self._calculate_performance_score(metrics)
            
            # Risk score (0-100, where 100 is lowest risk)
            risk_score = self._calculate_risk_score(metrics)
            
            # Overall health score
            health_score = (performance_score + risk_score) / 2
            
            return {
                'performance_score': performance_score,
                'risk_score': risk_score,
                'overall_health': health_score,
                'health_status': self._get_health_status(health_score),
                'key_metrics': {
                    'apy': metrics.apy,
                    'sharpe_ratio': metrics.sharpe_ratio,
                    'max_drawdown': metrics.max_drawdown,
                    'win_rate': metrics.win_rate,
                    'volatility': metrics.volatility
                },
                'risk_metrics': {
                    'var_95': metrics.value_at_risk_95,
                    'concentration': getattr(portfolio_risk, 'concentration_index', 0),
                    'diversification': getattr(portfolio_risk, 'diversification_ratio', 1)
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating portfolio summary: {e}")
            return {}
    
    async def _generate_real_time_stats(self, 
                                      metrics: EnhancedPerformanceMetrics) -> Dict[str, Any]:
        """Generate real-time statistics"""
        try:
            # Recent performance (last 24 hours worth of data)
            recent_returns = self.performance_tracker.returns[-24:] if len(self.performance_tracker.returns) >= 24 else self.performance_tracker.returns
            
            current_stats = {
                'current_return': recent_returns[-1] if recent_returns else 0.0,
                'daily_pnl': sum(recent_returns),
                'recent_volatility': np.std(recent_returns) * np.sqrt(365) if len(recent_returns) > 1 else 0.0,
                'recent_sharpe': self._calculate_recent_sharpe(recent_returns),
                'streak': self._calculate_win_streak(recent_returns),
                'max_recent_gain': max(recent_returns) if recent_returns else 0.0,
                'max_recent_loss': min(recent_returns) if recent_returns else 0.0,
                'active_positions': len(self.risk_analyzer.positions),
                'total_trades': len(getattr(self.performance_tracker, 'trades', [])),
                'last_update': datetime.now().isoformat()
            }
            
            return current_stats
            
        except Exception as e:
            logger.error(f"Error generating real-time stats: {e}")
            return {}
    
    async def _check_performance_alerts(self, 
                                      metrics: EnhancedPerformanceMetrics) -> List[PerformanceAlert]:
        """Check for performance-based alerts"""
        alerts = []
        timestamp = datetime.now()
        
        try:
            # Sharpe ratio check
            if metrics.sharpe_ratio < self.alert_thresholds['sharpe_ratio']['min']:
                trend = self._calculate_metric_trend('sharpe_ratio', metrics.sharpe_ratio)
                alert = PerformanceAlert(
                    alert_id=f"sharpe_{timestamp.strftime('%Y%m%d_%H%M%S')}",
                    timestamp=timestamp,
                    metric_name="Sharpe Ratio",
                    current_value=metrics.sharpe_ratio,
                    threshold=self.alert_thresholds['sharpe_ratio']['min'],
                    severity="WARNING",
                    message=f"Sharpe ratio below minimum threshold: {metrics.sharpe_ratio:.2f}",
                    trend=trend
                )
                alerts.append(alert)
            
            # Max drawdown check
            if metrics.max_drawdown < self.alert_thresholds['max_drawdown']['max']:
                trend = self._calculate_metric_trend('max_drawdown', metrics.max_drawdown)
                severity = "CRITICAL" if metrics.max_drawdown < -0.20 else "HIGH"
                alert = PerformanceAlert(
                    alert_id=f"drawdown_{timestamp.strftime('%Y%m%d_%H%M%S')}",
                    timestamp=timestamp,
                    metric_name="Max Drawdown",
                    current_value=metrics.max_drawdown,
                    threshold=self.alert_thresholds['max_drawdown']['max'],
                    severity=severity,
                    message=f"Excessive drawdown detected: {metrics.max_drawdown:.2%}",
                    trend=trend
                )
                alerts.append(alert)
            
            # Win rate check
            if metrics.win_rate < self.alert_thresholds['win_rate']['min']:
                trend = self._calculate_metric_trend('win_rate', metrics.win_rate)
                alert = PerformanceAlert(
                    alert_id=f"winrate_{timestamp.strftime('%Y%m%d_%H%M%S')}",
                    timestamp=timestamp,
                    metric_name="Win Rate",
                    current_value=metrics.win_rate,
                    threshold=self.alert_thresholds['win_rate']['min'],
                    severity="WARNING",
                    message=f"Win rate below target: {metrics.win_rate:.1%}",
                    trend=trend
                )
                alerts.append(alert)
            
            # Volatility check
            if metrics.volatility > self.alert_thresholds['volatility']['max']:
                trend = self._calculate_metric_trend('volatility', metrics.volatility)
                alert = PerformanceAlert(
                    alert_id=f"volatility_{timestamp.strftime('%Y%m%d_%H%M%S')}",
                    timestamp=timestamp,
                    metric_name="Volatility",
                    current_value=metrics.volatility,
                    threshold=self.alert_thresholds['volatility']['max'],
                    severity="HIGH",
                    message=f"High volatility detected: {metrics.volatility:.1%}",
                    trend=trend
                )
                alerts.append(alert)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error checking performance alerts: {e}")
            return []
    
    async def _update_metric_history(self, metrics: EnhancedPerformanceMetrics) -> None:
        """Update metric history for trend analysis"""
        try:
            metric_values = {
                'apy': metrics.apy,
                'sharpe_ratio': metrics.sharpe_ratio,
                'max_drawdown': metrics.max_drawdown,
                'win_rate': metrics.win_rate,
                'volatility': metrics.volatility,
                'var_95': metrics.value_at_risk_95,
                'profit_factor': metrics.profit_factor
            }
            
            for metric, value in metric_values.items():
                if metric not in self.metric_history:
                    self.metric_history[metric] = []
                
                self.metric_history[metric].append(value)
                
                # Keep last 100 values
                if len(self.metric_history[metric]) > 100:
                    self.metric_history[metric] = self.metric_history[metric][-100:]
            
        except Exception as e:
            logger.error(f"Error updating metric history: {e}")
    
    def _calculate_performance_score(self, metrics: EnhancedPerformanceMetrics) -> float:
        """Calculate overall performance score (0-100)"""
        try:
            score = 0.0
            
            # APY component (0-30 points)
            apy_score = min(metrics.apy * 100, 30) if metrics.apy > 0 else 0
            score += apy_score
            
            # Sharpe ratio component (0-25 points)
            sharpe_score = min(metrics.sharpe_ratio * 12.5, 25) if metrics.sharpe_ratio > 0 else 0
            score += sharpe_score
            
            # Win rate component (0-25 points)
            win_score = metrics.win_rate * 25
            score += win_score
            
            # Drawdown penalty (0-20 points)
            drawdown_score = max(20 + metrics.max_drawdown * 100, 0)
            score += drawdown_score
            
            return min(score, 100)
            
        except Exception as e:
            logger.error(f"Error calculating performance score: {e}")
            return 50.0
    
    def _calculate_risk_score(self, metrics: EnhancedPerformanceMetrics) -> float:
        """Calculate risk score (0-100, where 100 is lowest risk)"""
        try:
            score = 100.0
            
            # Volatility penalty
            vol_penalty = min(metrics.volatility * 200, 30)
            score -= vol_penalty
            
            # VaR penalty
            var_penalty = min(abs(metrics.value_at_risk_95) * 400, 25)
            score -= var_penalty
            
            # Drawdown penalty
            drawdown_penalty = min(abs(metrics.max_drawdown) * 100, 25)
            score -= drawdown_penalty
            
            # Skewness penalty (negative skew is bad)
            if metrics.skewness < 0:
                skew_penalty = min(abs(metrics.skewness) * 10, 20)
                score -= skew_penalty
            
            return max(score, 0)
            
        except Exception as e:
            logger.error(f"Error calculating risk score: {e}")
            return 50.0
    
    def _get_health_status(self, health_score: float) -> str:
        """Get health status based on score"""
        if health_score >= 85:
            return "Excellent"
        elif health_score >= 70:
            return "Good"
        elif health_score >= 55:
            return "Fair"
        elif health_score >= 40:
            return "Poor"
        else:
            return "Critical"
    
    def _calculate_metric_trend(self, metric_name: str, current_value: float) -> str:
        """Calculate trend for a specific metric"""
        if metric_name not in self.metric_history or len(self.metric_history[metric_name]) < 5:
            return "stable"
        
        recent_values = self.metric_history[metric_name][-5:]
        
        # Calculate trend using linear regression
        x = np.arange(len(recent_values))
        slope = np.polyfit(x, recent_values, 1)[0]
        
        # Determine trend direction
        if abs(slope) < 0.001:  # Very small change
            return "stable"
        elif slope > 0:
            return "improving" if metric_name not in ['volatility', 'max_drawdown', 'var_95'] else "deteriorating"
        else:
            return "deteriorating" if metric_name not in ['volatility', 'max_drawdown', 'var_95'] else "improving"
    
    def _calculate_recent_sharpe(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio for recent returns"""
        if len(returns) < 2:
            return 0.0
        
        mean_return = np.mean(returns)
        volatility = np.std(returns)
        
        if volatility == 0:
            return 0.0
        
        # Annualized
        return (mean_return * 365 - self.performance_tracker.risk_free_rate) / (volatility * np.sqrt(365))
    
    def _calculate_win_streak(self, returns: List[float]) -> int:
        """Calculate current win/loss streak"""
        if not returns:
            return 0
        
        streak = 0
        for ret in reversed(returns):
            if (ret > 0 and streak >= 0) or (ret <= 0 and streak <= 0):
                streak += 1 if ret > 0 else -1
            else:
                break
        
        return streak
    
    async def _calculate_metric_trends(self, recent_data: List[DashboardData]) -> Dict[str, str]:
        """Calculate trends for all key metrics"""
        if len(recent_data) < 5:
            return {}
        
        trends = {}
        
        # Extract metric time series
        metric_series = {
            'apy': [d.performance_metrics.apy for d in recent_data],
            'sharpe_ratio': [d.performance_metrics.sharpe_ratio for d in recent_data],
            'max_drawdown': [d.performance_metrics.max_drawdown for d in recent_data],
            'win_rate': [d.performance_metrics.win_rate for d in recent_data],
            'volatility': [d.performance_metrics.volatility for d in recent_data]
        }
        
        for metric, values in metric_series.items():
            if len(values) >= 5:
                x = np.arange(len(values))
                slope = np.polyfit(x, values, 1)[0]
                
                if abs(slope) < 0.001:
                    trends[metric] = "stable"
                elif slope > 0:
                    trends[metric] = "improving" if metric not in ['volatility', 'max_drawdown'] else "deteriorating"
                else:
                    trends[metric] = "deteriorating" if metric not in ['volatility', 'max_drawdown'] else "improving"
        
        return trends
    
    async def _summarize_risks(self, recent_data: List[DashboardData]) -> Dict[str, Any]:
        """Summarize risk analysis from recent data"""
        if not recent_data:
            return {}
        
        latest_data = recent_data[-1]
        
        return {
            'total_risk_alerts': len(latest_data.risk_alerts),
            'high_risk_positions': len([
                risk for risk in latest_data.position_risks 
                if risk.risk_level.value in ['high', 'extreme']
            ]),
            'avg_position_var': np.mean([
                risk.var_1d for risk in latest_data.position_risks
            ]) if latest_data.position_risks else 0.0,
            'risk_trend': 'stable'  # Could be enhanced with actual trend calculation
        }
    
    async def _generate_highlights(self, recent_data: List[DashboardData]) -> List[str]:
        """Generate performance highlights"""
        if not recent_data:
            return []
        
        latest = recent_data[-1].performance_metrics
        highlights = []
        
        # Positive highlights
        if latest.sharpe_ratio > 2.0:
            highlights.append(f"Excellent Sharpe ratio of {latest.sharpe_ratio:.2f}")
        
        if latest.win_rate > 0.7:
            highlights.append(f"Strong win rate of {latest.win_rate:.1%}")
        
        if latest.apy > 0.3:
            highlights.append(f"High annual return of {latest.apy:.1%}")
        
        # Areas for improvement
        if latest.max_drawdown < -0.15:
            highlights.append(f"High drawdown risk: {latest.max_drawdown:.1%}")
        
        if latest.volatility > 0.4:
            highlights.append(f"High volatility: {latest.volatility:.1%}")
        
        return highlights[:5]  # Top 5 highlights
    
    async def _generate_recommendations(self, 
                                      metrics: EnhancedPerformanceMetrics,
                                      trends: Dict[str, str]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Performance recommendations
        if metrics.sharpe_ratio < 1.0:
            recommendations.append("Consider optimizing position sizing to improve risk-adjusted returns")
        
        if metrics.max_drawdown < -0.15:
            recommendations.append("Implement stricter stop-loss rules to limit drawdowns")
        
        if metrics.win_rate < 0.6:
            recommendations.append("Review entry/exit criteria to improve trade success rate")
        
        # Risk recommendations
        if metrics.volatility > 0.3:
            recommendations.append("Consider diversifying positions to reduce portfolio volatility")
        
        if abs(metrics.value_at_risk_95) > 0.05:
            recommendations.append("Reduce position sizes to lower daily Value at Risk")
        
        # Trend-based recommendations
        if trends.get('sharpe_ratio') == 'deteriorating':
            recommendations.append("Monitor strategy performance - Sharpe ratio declining")
        
        return recommendations[:5]  # Top 5 recommendations
    
    def _format_detailed_metrics(self, metrics: EnhancedPerformanceMetrics) -> Dict[str, Any]:
        """Format detailed metrics for report"""
        return {
            'returns': {
                'apy': f"{metrics.apy:.2%}",
                'geometric_mean': f"{metrics.geometric_mean_return:.2%}",
                'arithmetic_mean': f"{metrics.arithmetic_mean_return:.2%}",
                'time_weighted_return': f"{metrics.time_weighted_return:.2%}"
            },
            'risk_metrics': {
                'volatility': f"{metrics.volatility:.2%}",
                'downside_volatility': f"{metrics.downside_volatility:.2%}",
                'var_95': f"{abs(metrics.value_at_risk_95):.2%}",
                'var_99': f"{abs(metrics.value_at_risk_99):.2%}",
                'max_drawdown': f"{metrics.max_drawdown:.2%}"
            },
            'ratios': {
                'sharpe_ratio': f"{metrics.sharpe_ratio:.2f}",
                'sortino_ratio': f"{metrics.sortino_ratio:.2f}",
                'calmar_ratio': f"{metrics.calmar_ratio:.2f}",
                'profit_factor': f"{metrics.profit_factor:.2f}"
            },
            'distribution': {
                'skewness': f"{metrics.skewness:.2f}",
                'kurtosis': f"{metrics.kurtosis:.2f}",
                'win_rate': f"{metrics.win_rate:.1%}"
            }
        }
    
    async def _save_dashboard_data(self, data: DashboardData) -> None:
        """Save dashboard data to file"""
        try:
            # Save latest data
            latest_file = self.output_dir / "latest_dashboard_data.json"
            with open(latest_file, 'w') as f:
                json.dump(asdict(data), f, default=str, indent=2)
            
            # Save historical data (daily)
            date_str = data.timestamp.strftime('%Y%m%d')
            daily_file = self.output_dir / f"dashboard_data_{date_str}.json"
            
            # Append to daily file
            daily_data = []
            if daily_file.exists():
                with open(daily_file, 'r') as f:
                    daily_data = json.load(f)
            
            daily_data.append(asdict(data))
            
            with open(daily_file, 'w') as f:
                json.dump(daily_data, f, default=str, indent=2)
            
        except Exception as e:
            logger.error(f"Error saving dashboard data: {e}")
    
    def _empty_dashboard_data(self) -> DashboardData:
        """Return empty dashboard data for error cases"""
        from .enhanced_performance_tracker import EnhancedPerformanceMetrics
        
        empty_metrics = EnhancedPerformanceMetrics(
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
        
        return DashboardData(
            timestamp=datetime.now(),
            performance_metrics=empty_metrics,
            risk_alerts=[],
            position_risks=[],
            portfolio_summary={},
            real_time_stats={}
        )