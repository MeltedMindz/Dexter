#!/usr/bin/env python3
"""
Enhanced Performance Tracking Integration Example
Demonstrates how to use all the enhanced performance tracking components together
"""

import asyncio
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
import random

# Import our enhanced components
from utils.enhanced_performance_tracker import EnhancedPerformanceTracker
from utils.portfolio_risk_analyzer import PortfolioRiskAnalyzer
from utils.performance_dashboard import PerformanceDashboard

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

class DexterEnhancedPerformanceSystem:
    """
    Complete enhanced performance tracking system integration
    """
    
    def __init__(self, agent_name: str = "aggressive"):
        """Initialize the enhanced performance system"""
        self.agent_name = agent_name
        
        # Initialize components
        self.performance_tracker = EnhancedPerformanceTracker(
            agent_name=agent_name,
            risk_free_rate=0.04  # 4% annual risk-free rate
        )
        
        self.risk_analyzer = PortfolioRiskAnalyzer(
            risk_free_rate=0.04
        )
        
        self.dashboard = PerformanceDashboard(
            performance_tracker=self.performance_tracker,
            risk_analyzer=self.risk_analyzer,
            output_dir=f"performance_data/{agent_name}"
        )
        
        # System state
        self.is_running = False
        self.update_interval = 300  # 5 minutes
        
        logger.info(f"Enhanced performance system initialized for agent: {agent_name}")
    
    async def start_monitoring(self):
        """Start the enhanced performance monitoring system"""
        logger.info("🚀 Starting enhanced performance monitoring...")
        
        self.is_running = True
        
        # Start monitoring tasks
        tasks = [
            asyncio.create_task(self._performance_update_loop()),
            asyncio.create_task(self._risk_monitoring_loop()),
            asyncio.create_task(self._dashboard_update_loop()),
            asyncio.create_task(self._alert_monitoring_loop())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Error in monitoring system: {e}")
        finally:
            self.is_running = False
    
    async def update_performance_data(self,
                                    portfolio_value: float,
                                    fees_earned: float,
                                    impermanent_loss: float,
                                    trading_volume: float,
                                    positions: List[Dict],
                                    benchmark_return: float = None):
        """
        Update performance data for all components
        
        Args:
            portfolio_value: Current total portfolio value
            fees_earned: Fees earned in this period
            impermanent_loss: Current impermanent loss
            trading_volume: Trading volume for this period
            positions: List of current positions
            benchmark_return: Benchmark return for comparison
        """
        try:
            timestamp = datetime.now()
            
            # Update performance tracker
            await self.performance_tracker.update_performance(
                value=portfolio_value,
                fees=fees_earned,
                il=impermanent_loss,
                volume=trading_volume,
                timestamp=timestamp,
                benchmark_return=benchmark_return
            )
            
            # Update risk analyzer with positions
            self.risk_analyzer.positions = {
                pos.get('id', f"pos_{i}"): pos 
                for i, pos in enumerate(positions)
            }
            
            # Update returns history for risk analysis
            if hasattr(self, '_previous_values'):
                for i, pos in enumerate(positions):
                    pos_id = pos.get('id', f"pos_{i}")
                    current_value = pos.get('value', 0)
                    
                    if pos_id in self._previous_values:
                        previous_value = self._previous_values[pos_id]
                        if previous_value > 0:
                            return_pct = (current_value - previous_value) / previous_value
                        else:
                            return_pct = 0.0
                        
                        if pos_id not in self.risk_analyzer.returns_history:
                            self.risk_analyzer.returns_history[pos_id] = []
                        
                        self.risk_analyzer.returns_history[pos_id].append(return_pct)
                        
                        # Keep last 252 returns (1 year of daily data)
                        if len(self.risk_analyzer.returns_history[pos_id]) > 252:
                            self.risk_analyzer.returns_history[pos_id] = \
                                self.risk_analyzer.returns_history[pos_id][-252:]
            
            # Store current values for next calculation
            self._previous_values = {
                pos.get('id', f"pos_{i}"): pos.get('value', 0)
                for i, pos in enumerate(positions)
            }
            
            logger.info(f"Performance data updated: ${portfolio_value:,.2f} portfolio value")
            
        except Exception as e:
            logger.error(f"Error updating performance data: {e}")
    
    async def get_comprehensive_metrics(self) -> Dict:
        """Get comprehensive performance and risk metrics"""
        try:
            # Get performance metrics
            performance_metrics = await self.performance_tracker.calculate_enhanced_metrics()
            
            # Get risk analysis
            positions = list(self.risk_analyzer.positions.values())
            portfolio_risk = await self.risk_analyzer.analyze_portfolio_risk(
                positions, self.risk_analyzer.returns_history
            )
            
            # Get dashboard data
            dashboard_data = await self.dashboard.get_real_time_dashboard_data()
            
            # Compile comprehensive report
            comprehensive_metrics = {
                'timestamp': datetime.now().isoformat(),
                'agent_name': self.agent_name,
                'performance_metrics': {
                    'apy': performance_metrics.apy,
                    'sharpe_ratio': performance_metrics.sharpe_ratio,
                    'max_drawdown': performance_metrics.max_drawdown,
                    'win_rate': performance_metrics.win_rate,
                    'volatility': performance_metrics.volatility,
                    'var_95': performance_metrics.value_at_risk_95,
                    'sortino_ratio': performance_metrics.sortino_ratio,
                    'calmar_ratio': performance_metrics.calmar_ratio,
                    'profit_factor': performance_metrics.profit_factor,
                    'kelly_criterion': performance_metrics.kelly_criterion
                },
                'risk_metrics': {
                    'portfolio_var': portfolio_risk.total_var,
                    'concentration_index': portfolio_risk.concentration_index,
                    'diversification_ratio': portfolio_risk.diversification_ratio,
                    'risk_budget_utilization': portfolio_risk.risk_budget_utilization
                },
                'portfolio_summary': dashboard_data.get('portfolio_summary', {}),
                'alerts': {
                    'risk_alerts_count': len(dashboard_data.get('alerts', {}).get('recent_alerts', [])),
                    'performance_alerts_count': len(self.dashboard.performance_alerts)
                }
            }
            
            return comprehensive_metrics
            
        except Exception as e:
            logger.error(f"Error getting comprehensive metrics: {e}")
            return {}
    
    async def run_performance_analysis(self) -> Dict:
        """Run comprehensive performance analysis"""
        try:
            logger.info("📊 Running comprehensive performance analysis...")
            
            # Generate performance report
            performance_report = await self.dashboard.generate_performance_report(period_days=30)
            
            # Run Monte Carlo simulation
            mc_results = await self.risk_analyzer.run_monte_carlo_simulation(
                num_simulations=5000,
                time_horizon=30
            )
            
            # Calculate optimal position sizes
            optimal_sizes = await self.risk_analyzer.calculate_optimal_position_sizes(
                target_risk=0.02,
                max_position_size=0.25
            )
            
            # Get position-level risk analysis
            position_risks = []
            for pos_id in self.risk_analyzer.positions:
                pos_risk = await self.risk_analyzer.analyze_position_risk(pos_id)
                if pos_risk:
                    position_risks.append({
                        'position_id': pos_risk.position_id,
                        'token_pair': pos_risk.token_pair,
                        'risk_level': pos_risk.risk_level.value,
                        'var_1d': pos_risk.var_1d,
                        'concentration_risk': pos_risk.concentration_risk,
                        'liquidity_risk': pos_risk.liquidity_risk
                    })
            
            analysis_results = {
                'analysis_timestamp': datetime.now().isoformat(),
                'performance_report': performance_report,
                'monte_carlo_simulation': mc_results,
                'optimal_position_sizes': optimal_sizes,
                'position_risk_analysis': position_risks,
                'summary': {
                    'total_positions_analyzed': len(position_risks),
                    'high_risk_positions': len([
                        pos for pos in position_risks 
                        if pos['risk_level'] in ['high', 'extreme']
                    ]),
                    'portfolio_health_score': performance_report.get('performance_summary', {}).get('highlights', [])
                }
            }
            
            logger.info("✅ Performance analysis completed")
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error running performance analysis: {e}")
            return {}
    
    async def _performance_update_loop(self):
        """Main performance update loop"""
        logger.info("🔄 Starting performance update loop...")
        
        while self.is_running:
            try:
                # Update dashboard with latest data
                await self.dashboard.update_dashboard()
                
                # Wait for next update
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in performance update loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error
    
    async def _risk_monitoring_loop(self):
        """Risk monitoring and alert loop"""
        logger.info("⚠️ Starting risk monitoring loop...")
        
        while self.is_running:
            try:
                # Check for risk alerts
                risk_alerts = await self.risk_analyzer.check_risk_alerts()
                
                if risk_alerts:
                    logger.warning(f"🚨 {len(risk_alerts)} new risk alerts generated")
                    for alert in risk_alerts:
                        logger.warning(f"Risk Alert: {alert.message}")
                
                # Wait 10 minutes between risk checks
                await asyncio.sleep(600)
                
            except Exception as e:
                logger.error(f"Error in risk monitoring loop: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    async def _dashboard_update_loop(self):
        """Dashboard update loop"""
        logger.info("📊 Starting dashboard update loop...")
        
        while self.is_running:
            try:
                # Update dashboard
                dashboard_data = await self.dashboard.update_dashboard()
                
                # Log key metrics
                metrics = dashboard_data.performance_metrics
                logger.info(
                    f"📈 Performance Update - "
                    f"APY: {metrics.apy:.1%}, "
                    f"Sharpe: {metrics.sharpe_ratio:.2f}, "
                    f"Drawdown: {metrics.max_drawdown:.1%}, "
                    f"Win Rate: {metrics.win_rate:.1%}"
                )
                
                # Wait 15 minutes between dashboard updates
                await asyncio.sleep(900)
                
            except Exception as e:
                logger.error(f"Error in dashboard update loop: {e}")
                await asyncio.sleep(300)
    
    async def _alert_monitoring_loop(self):
        """Performance alert monitoring loop"""
        logger.info("🔔 Starting alert monitoring loop...")
        
        while self.is_running:
            try:
                # Check for performance alerts
                if hasattr(self.dashboard, 'performance_alerts'):
                    recent_alerts = [
                        alert for alert in self.dashboard.performance_alerts
                        if (datetime.now() - alert.timestamp).total_seconds() < 3600  # Last hour
                    ]
                    
                    if recent_alerts:
                        logger.info(f"📢 {len(recent_alerts)} performance alerts in last hour")
                        for alert in recent_alerts[-3:]:  # Show last 3
                            logger.info(f"Alert: {alert.message} (Trend: {alert.trend})")
                
                # Wait 30 minutes between alert checks
                await asyncio.sleep(1800)
                
            except Exception as e:
                logger.error(f"Error in alert monitoring loop: {e}")
                await asyncio.sleep(600)
    
    async def stop_monitoring(self):
        """Stop the monitoring system"""
        logger.info("🛑 Stopping enhanced performance monitoring...")
        self.is_running = False
        
        # Export final report
        try:
            final_report = await self.performance_tracker.export_metrics_report(
                f"performance_data/{self.agent_name}/final_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            logger.info("📄 Final performance report exported")
        except Exception as e:
            logger.error(f"Error exporting final report: {e}")

# Demo function to show system in action
async def demo_enhanced_performance_system():
    """Demonstrate the enhanced performance system with simulated data"""
    logger.info("🎯 Starting Enhanced Performance System Demo")
    
    # Initialize system
    system = DexterEnhancedPerformanceSystem("demo_aggressive")
    
    # Simulate some performance data
    base_value = 100000  # $100k starting portfolio
    current_value = base_value
    
    logger.info("📊 Simulating 30 days of trading data...")
    
    for day in range(30):
        # Simulate daily performance
        daily_return = np.random.normal(0.001, 0.02)  # 0.1% daily return, 2% volatility
        fees = abs(np.random.normal(50, 20))  # $50 average daily fees
        il = abs(np.random.normal(0, 10))  # Small random IL
        volume = abs(np.random.normal(10000, 3000))  # $10k average volume
        
        current_value *= (1 + daily_return)
        
        # Simulate positions
        positions = [
            {
                'id': 'eth_usdc_500',
                'token0': 'ETH',
                'token1': 'USDC',
                'fee_tier': 500,
                'value': current_value * 0.4,
                'tvl': 50000000,
                'volume_24h': 5000000
            },
            {
                'id': 'btc_eth_3000',
                'token0': 'BTC',
                'token1': 'ETH',
                'fee_tier': 3000,
                'value': current_value * 0.35,
                'tvl': 30000000,
                'volume_24h': 2000000
            },
            {
                'id': 'usdc_dai_100',
                'token0': 'USDC',
                'token1': 'DAI',
                'fee_tier': 100,
                'value': current_value * 0.25,
                'tvl': 80000000,
                'volume_24h': 8000000
            }
        ]
        
        # Update system
        await system.update_performance_data(
            portfolio_value=current_value,
            fees_earned=fees,
            impermanent_loss=il,
            trading_volume=volume,
            positions=positions,
            benchmark_return=np.random.normal(0.0005, 0.015)  # Market benchmark
        )
        
        # Log progress every 10 days
        if (day + 1) % 10 == 0:
            logger.info(f"Day {day + 1}: Portfolio value ${current_value:,.2f}")
    
    # Get comprehensive metrics
    logger.info("📈 Calculating comprehensive metrics...")
    comprehensive_metrics = await system.get_comprehensive_metrics()
    
    logger.info("🔍 Performance Summary:")
    perf = comprehensive_metrics.get('performance_metrics', {})
    logger.info(f"  📊 APY: {perf.get('apy', 0):.1%}")
    logger.info(f"  📈 Sharpe Ratio: {perf.get('sharpe_ratio', 0):.2f}")
    logger.info(f"  📉 Max Drawdown: {perf.get('max_drawdown', 0):.1%}")
    logger.info(f"  🎯 Win Rate: {perf.get('win_rate', 0):.1%}")
    logger.info(f"  📊 Volatility: {perf.get('volatility', 0):.1%}")
    
    # Run full analysis
    logger.info("🔬 Running comprehensive analysis...")
    analysis = await system.run_performance_analysis()
    
    if analysis:
        logger.info("✅ Analysis completed successfully")
        mc_results = analysis.get('monte_carlo_simulation', {})
        if mc_results:
            logger.info(f"🎲 Monte Carlo VaR (95%): {abs(mc_results.get('var_95', 0)):.1%}")
            logger.info(f"🎲 Expected Return: {mc_results.get('expected_return', 0):.1%}")
    
    logger.info("🎉 Enhanced Performance System Demo completed!")
    
    return comprehensive_metrics

if __name__ == "__main__":
    # Run the demo
    asyncio.run(demo_enhanced_performance_system())