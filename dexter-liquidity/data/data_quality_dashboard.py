"""
Data Quality Dashboard
Integrated dashboard for monitoring data quality, completeness, and triggering backfills
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import asdict
import json
from pathlib import Path
import aiohttp
from aiohttp import web

from .data_quality_monitor import DataQualityMonitor
from .historical_backfill_service import HistoricalBackfillService
from .completeness_checker import DataCompletenessChecker

logger = logging.getLogger(__name__)

class DataQualityDashboard:
    """
    Integrated data quality dashboard and management system
    """
    
    def __init__(self, 
                 database_url: str,
                 alchemy_api_key: str,
                 graph_api_url: str = "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3",
                 dashboard_port: int = 8090):
        """
        Initialize data quality dashboard
        
        Args:
            database_url: Database connection string
            alchemy_api_key: Alchemy API key
            graph_api_url: Graph API endpoint
            dashboard_port: Port for web dashboard
        """
        self.database_url = database_url
        self.alchemy_api_key = alchemy_api_key
        self.graph_api_url = graph_api_url
        self.dashboard_port = dashboard_port
        
        # Initialize components
        self.quality_monitor = DataQualityMonitor(database_url, alchemy_api_key, graph_api_url)
        self.backfill_service = HistoricalBackfillService(database_url, alchemy_api_key, graph_api_url)
        self.completeness_checker = DataCompletenessChecker(database_url)
        
        # Dashboard state
        self.is_running = False
        self.last_quality_check = None
        self.last_completeness_check = None
        self.system_alerts = []
        
        # Web app setup
        self.app = web.Application()
        self._setup_routes()
        
        logger.info("Data quality dashboard initialized")
    
    async def start_dashboard(self):
        """Start the integrated data quality dashboard"""
        try:
            logger.info("🚀 Starting Data Quality Dashboard...")
            
            self.is_running = True
            
            # Start background monitoring
            monitoring_task = asyncio.create_task(self._monitoring_loop())
            
            # Start web server
            runner = web.AppRunner(self.app)
            await runner.setup()
            
            site = web.TCPSite(runner, '0.0.0.0', self.dashboard_port)
            await site.start()
            
            logger.info(f"📊 Dashboard available at http://localhost:{self.dashboard_port}")
            logger.info("🔍 Background monitoring started")
            
            # Keep running
            try:
                await monitoring_task
            except KeyboardInterrupt:
                logger.info("Dashboard shutdown requested")
            finally:
                await runner.cleanup()
                self.is_running = False
            
        except Exception as e:
            logger.error(f"Error starting dashboard: {e}")
            self.is_running = False
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            # Get latest quality metrics
            quality_data = await self.quality_monitor.get_quality_dashboard_data()
            
            # Get completeness status
            completeness_results = await self.completeness_checker.check_all_sources_completeness(6)
            
            # Get backfill status
            backfill_status = await self.backfill_service.get_backfill_status()
            
            # Calculate overall system health
            if quality_data and 'system_overview' in quality_data:
                overall_quality = quality_data['system_overview']['overall_quality']
            else:
                overall_quality = 0.0
            
            if completeness_results:
                overall_completeness = sum(r.completeness_percentage for r in completeness_results.values()) / len(completeness_results)
            else:
                overall_completeness = 0.0
            
            system_health_score = (overall_quality + overall_completeness) / 2
            
            # Get critical issues
            critical_issues = await self.completeness_checker.get_critical_issues()
            
            status = {
                'timestamp': datetime.now().isoformat(),
                'system_health': {
                    'overall_score': system_health_score,
                    'quality_score': overall_quality,
                    'completeness_score': overall_completeness,
                    'health_status': self._get_health_status(system_health_score),
                    'critical_issues_count': len([i for i in critical_issues if i['severity'] == 'critical']),
                    'high_issues_count': len([i for i in critical_issues if i['severity'] == 'high'])
                },
                'data_quality': quality_data,
                'completeness': {
                    'sources_checked': len(completeness_results),
                    'sources_healthy': len([r for r in completeness_results.values() if r.completeness_percentage >= 0.95]),
                    'total_missing_records': sum(r.missing_records for r in completeness_results.values()),
                    'sources_with_gaps': len([r for r in completeness_results.values() if r.missing_time_periods]),
                    'results': {
                        source: {
                            'completeness_percentage': result.completeness_percentage,
                            'missing_records': result.missing_records,
                            'priority_level': result.priority_level,
                            'largest_gap_hours': result.largest_gap_hours
                        }
                        for source, result in completeness_results.items()
                    }
                },
                'backfill_operations': backfill_status,
                'critical_issues': critical_issues[:10],  # Top 10 critical issues
                'system_alerts': self.system_alerts[-20:],  # Last 20 alerts
                'last_checks': {
                    'quality_check': self.last_quality_check.isoformat() if self.last_quality_check else None,
                    'completeness_check': self.last_completeness_check.isoformat() if self.last_completeness_check else None
                }
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    async def trigger_emergency_backfill(self, 
                                       source: str,
                                       hours_back: int = 24) -> Dict[str, Any]:
        """
        Trigger emergency backfill for data gaps
        
        Args:
            source: Data source to backfill
            hours_back: How many hours back to backfill
            
        Returns:
            Backfill operation details
        """
        try:
            logger.info(f"🚨 Triggering emergency backfill for {source} ({hours_back} hours)")
            
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours_back)
            
            # Add system alert
            self.system_alerts.append({
                'timestamp': datetime.now().isoformat(),
                'type': 'emergency_backfill',
                'severity': 'high',
                'message': f"Emergency backfill triggered for {source}",
                'source': source,
                'details': f"Backfilling {hours_back} hours of data"
            })
            
            if source == 'graph_api' or source == 'positions':
                progress = await self.backfill_service.backfill_uniswap_positions(
                    start_time, end_time, chunk_hours=4
                )
            elif source == 'alchemy_rpc' or source == 'alchemy_positions':
                # Estimate block range (approximately 12 seconds per block)
                blocks_per_hour = 300
                total_blocks = hours_back * blocks_per_hour
                current_block = 18000000  # Approximate current block
                start_block = current_block - total_blocks
                
                progress = await self.backfill_service.backfill_alchemy_position_data(
                    start_block, current_block, block_chunk_size=500
                )
            elif source == 'price_data' or source == 'token_prices':
                # Get monitored tokens
                token_addresses = [
                    "0xA0b86a33E6c4ffbb6d07C8c72E7c90Fc7C92A7F6",  # ETH
                    "0xA0b86a33E6c4ffbb6d07C8c72E7c90Fc7C92A7F7",  # USDC
                    "0xA0b86a33E6c4ffbb6d07C8c72E7c90Fc7C92A7F8",  # WBTC
                ]
                
                progress = await self.backfill_service.backfill_token_prices(
                    token_addresses, start_time, end_time
                )
            else:
                raise ValueError(f"Unknown source for emergency backfill: {source}")
            
            result = {
                'success': True,
                'source': source,
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'records_processed': progress.records_processed,
                'errors_encountered': progress.errors_encountered,
                'estimated_completion': progress.estimated_completion.isoformat(),
                'rate_per_minute': progress.rate_per_minute
            }
            
            logger.info(f"✅ Emergency backfill initiated: {progress.records_processed} records processed")
            return result
            
        except Exception as e:
            logger.error(f"Error in emergency backfill: {e}")
            return {'success': False, 'error': str(e)}
    
    async def auto_heal_data_issues(self) -> Dict[str, Any]:
        """
        Automatically detect and heal data quality issues
        
        Returns:
            Summary of healing actions taken
        """
        try:
            logger.info("🔧 Starting auto-heal process...")
            
            healing_actions = []
            
            # Check for critical completeness issues
            critical_issues = await self.completeness_checker.get_critical_issues()
            
            for issue in critical_issues:
                if issue['severity'] == 'critical' and issue['type'] == 'low_completeness':
                    source = issue['source']
                    
                    # Trigger automatic backfill
                    logger.info(f"Auto-healing: Backfilling {source}")
                    backfill_result = await self.trigger_emergency_backfill(source, hours_back=12)
                    
                    healing_actions.append({
                        'action': 'auto_backfill',
                        'source': source,
                        'result': backfill_result,
                        'timestamp': datetime.now().isoformat()
                    })
                
                elif issue['type'] == 'large_data_gap':
                    source = issue['source']
                    gap_hours = issue.get('gap_hours', 24)
                    
                    # Trigger targeted backfill for the gap
                    logger.info(f"Auto-healing: Filling {gap_hours:.1f} hour gap in {source}")
                    backfill_result = await self.trigger_emergency_backfill(source, hours_back=int(gap_hours) + 6)
                    
                    healing_actions.append({
                        'action': 'gap_fill',
                        'source': source,
                        'gap_hours': gap_hours,
                        'result': backfill_result,
                        'timestamp': datetime.now().isoformat()
                    })
            
            # Smart gap detection and filling
            for source in ['graph_api', 'alchemy_rpc', 'price_data']:
                gap_result = await self.backfill_service.smart_gap_detection_and_fill(source, max_gap_hours=4)
                
                if gap_result['gaps_filled'] > 0:
                    healing_actions.append({
                        'action': 'smart_gap_fill',
                        'source': source,
                        'gaps_filled': gap_result['gaps_filled'],
                        'records_added': gap_result['records_added'],
                        'timestamp': datetime.now().isoformat()
                    })
            
            # Add system alert
            if healing_actions:
                self.system_alerts.append({
                    'timestamp': datetime.now().isoformat(),
                    'type': 'auto_heal',
                    'severity': 'info',
                    'message': f"Auto-heal completed: {len(healing_actions)} actions taken",
                    'details': f"Healing actions: {[a['action'] for a in healing_actions]}"
                })
            
            result = {
                'success': True,
                'actions_taken': len(healing_actions),
                'healing_actions': healing_actions,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"🔧 Auto-heal completed: {len(healing_actions)} actions taken")
            return result
            
        except Exception as e:
            logger.error(f"Error in auto-heal process: {e}")
            return {'success': False, 'error': str(e)}
    
    # Web routes
    
    def _setup_routes(self):
        """Setup web dashboard routes"""
        self.app.router.add_get('/', self._dashboard_home)
        self.app.router.add_get('/api/status', self._api_status)
        self.app.router.add_get('/api/quality', self._api_quality)
        self.app.router.add_get('/api/completeness', self._api_completeness)
        self.app.router.add_get('/api/backfill', self._api_backfill_status)
        self.app.router.add_post('/api/backfill/trigger', self._api_trigger_backfill)
        self.app.router.add_post('/api/auto-heal', self._api_auto_heal)
        self.app.router.add_get('/api/critical-issues', self._api_critical_issues)
    
    async def _dashboard_home(self, request):
        """Main dashboard page"""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Dexter Data Quality Dashboard</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
                .container { max-width: 1200px; margin: 0 auto; }
                .header { background: #2c3e50; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
                .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 20px; }
                .card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .metric { font-size: 2em; font-weight: bold; color: #27ae60; }
                .status-good { color: #27ae60; }
                .status-warning { color: #f39c12; }
                .status-critical { color: #e74c3c; }
                .btn { background: #3498db; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; margin: 5px; }
                .btn:hover { background: #2980b9; }
                .btn-danger { background: #e74c3c; }
                .btn-danger:hover { background: #c0392b; }
                .alerts { max-height: 300px; overflow-y: auto; }
                .alert { padding: 10px; margin: 5px 0; border-radius: 4px; }
                .alert-critical { background: #ffebee; border-left: 4px solid #e74c3c; }
                .alert-high { background: #fff3e0; border-left: 4px solid #f39c12; }
                .alert-info { background: #e3f2fd; border-left: 4px solid #2196f3; }
            </style>
            <script>
                async function refreshData() {
                    try {
                        const response = await fetch('/api/status');
                        const data = await response.json();
                        updateDashboard(data);
                    } catch (error) {
                        console.error('Error fetching data:', error);
                    }
                }
                
                function updateDashboard(data) {
                    // Update system health
                    const healthScore = (data.system_health?.overall_score * 100 || 0).toFixed(1);
                    document.getElementById('health-score').textContent = healthScore + '%';
                    document.getElementById('health-score').className = 'metric ' + getStatusClass(data.system_health?.overall_score || 0);
                    
                    // Update completeness
                    const completeness = (data.completeness?.results ? 
                        Object.values(data.completeness.results).reduce((sum, r) => sum + r.completeness_percentage, 0) / 
                        Object.keys(data.completeness.results).length * 100 : 0).toFixed(1);
                    document.getElementById('completeness-score').textContent = completeness + '%';
                    document.getElementById('completeness-score').className = 'metric ' + getStatusClass(completeness / 100);
                    
                    // Update critical issues
                    document.getElementById('critical-issues').textContent = data.system_health?.critical_issues_count || 0;
                    document.getElementById('high-issues').textContent = data.system_health?.high_issues_count || 0;
                    
                    // Update backfill status
                    document.getElementById('active-backfills').textContent = data.backfill_operations?.active_backfills || 0;
                    
                    // Update alerts
                    const alertsContainer = document.getElementById('alerts-container');
                    alertsContainer.innerHTML = '';
                    (data.critical_issues || []).slice(0, 5).forEach(issue => {
                        const alertDiv = document.createElement('div');
                        alertDiv.className = 'alert alert-' + issue.severity;
                        alertDiv.innerHTML = '<strong>' + issue.type + ':</strong> ' + issue.description;
                        alertsContainer.appendChild(alertDiv);
                    });
                }
                
                function getStatusClass(score) {
                    if (score >= 0.9) return 'status-good';
                    if (score >= 0.7) return 'status-warning';
                    return 'status-critical';
                }
                
                async function triggerAutoHeal() {
                    try {
                        const response = await fetch('/api/auto-heal', { method: 'POST' });
                        const result = await response.json();
                        alert('Auto-heal completed: ' + result.actions_taken + ' actions taken');
                        refreshData();
                    } catch (error) {
                        alert('Error triggering auto-heal: ' + error.message);
                    }
                }
                
                // Auto-refresh every 30 seconds
                setInterval(refreshData, 30000);
                refreshData();
            </script>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🔍 Dexter Data Quality Dashboard</h1>
                    <p>Real-time monitoring of data quality, completeness, and system health</p>
                </div>
                
                <div class="cards">
                    <div class="card">
                        <h3>System Health</h3>
                        <div id="health-score" class="metric">Loading...</div>
                        <p>Overall system health score</p>
                    </div>
                    
                    <div class="card">
                        <h3>Data Completeness</h3>
                        <div id="completeness-score" class="metric">Loading...</div>
                        <p>Average data completeness across sources</p>
                    </div>
                    
                    <div class="card">
                        <h3>Critical Issues</h3>
                        <div id="critical-issues" class="metric status-critical">Loading...</div>
                        <p>Issues requiring immediate attention</p>
                    </div>
                    
                    <div class="card">
                        <h3>High Priority Issues</h3>
                        <div id="high-issues" class="metric status-warning">Loading...</div>
                        <p>Issues requiring attention soon</p>
                    </div>
                    
                    <div class="card">
                        <h3>Active Backfills</h3>
                        <div id="active-backfills" class="metric">Loading...</div>
                        <p>Currently running backfill operations</p>
                    </div>
                    
                    <div class="card">
                        <h3>Actions</h3>
                        <button class="btn" onclick="refreshData()">Refresh Data</button><br>
                        <button class="btn btn-danger" onclick="triggerAutoHeal()">Trigger Auto-Heal</button>
                        <p>Manual dashboard controls</p>
                    </div>
                </div>
                
                <div class="card">
                    <h3>🚨 Current Issues</h3>
                    <div id="alerts-container" class="alerts">
                        Loading alerts...
                    </div>
                </div>
                
                <div class="card">
                    <h3>📊 API Endpoints</h3>
                    <ul>
                        <li><a href="/api/status">/api/status</a> - Complete system status</li>
                        <li><a href="/api/quality">/api/quality</a> - Data quality metrics</li>
                        <li><a href="/api/completeness">/api/completeness</a> - Completeness analysis</li>
                        <li><a href="/api/critical-issues">/api/critical-issues</a> - Critical issues list</li>
                        <li><a href="/api/backfill">/api/backfill</a> - Backfill operations status</li>
                    </ul>
                </div>
            </div>
        </body>
        </html>
        """
        return web.Response(text=html, content_type='text/html')
    
    async def _api_status(self, request):
        """API endpoint for system status"""
        status = await self.get_system_status()
        return web.json_response(status)
    
    async def _api_quality(self, request):
        """API endpoint for quality data"""
        quality_data = await self.quality_monitor.get_quality_dashboard_data()
        return web.json_response(quality_data)
    
    async def _api_completeness(self, request):
        """API endpoint for completeness data"""
        hours = int(request.query.get('hours', 24))
        results = await self.completeness_checker.check_all_sources_completeness(hours)
        
        # Convert results to JSON-serializable format
        serializable_results = {
            source: asdict(result) for source, result in results.items()
        }
        
        return web.json_response(serializable_results)
    
    async def _api_backfill_status(self, request):
        """API endpoint for backfill status"""
        status = await self.backfill_service.get_backfill_status()
        return web.json_response(status)
    
    async def _api_trigger_backfill(self, request):
        """API endpoint to trigger backfill"""
        try:
            data = await request.json()
            source = data.get('source')
            hours_back = data.get('hours_back', 24)
            
            if not source:
                return web.json_response({'error': 'Source parameter required'}, status=400)
            
            result = await self.trigger_emergency_backfill(source, hours_back)
            return web.json_response(result)
            
        except Exception as e:
            return web.json_response({'error': str(e)}, status=500)
    
    async def _api_auto_heal(self, request):
        """API endpoint to trigger auto-heal"""
        result = await self.auto_heal_data_issues()
        return web.json_response(result)
    
    async def _api_critical_issues(self, request):
        """API endpoint for critical issues"""
        issues = await self.completeness_checker.get_critical_issues()
        return web.json_response(issues)
    
    # Background monitoring
    
    async def _monitoring_loop(self):
        """Background monitoring loop"""
        logger.info("🔄 Starting background monitoring loop...")
        
        while self.is_running:
            try:
                # Run quality check every 30 minutes
                if (not self.last_quality_check or 
                    datetime.now() - self.last_quality_check > timedelta(minutes=30)):
                    
                    logger.info("Running scheduled quality check...")
                    await self._run_quality_check()
                    self.last_quality_check = datetime.now()
                
                # Run completeness check every hour
                if (not self.last_completeness_check or 
                    datetime.now() - self.last_completeness_check > timedelta(hours=1)):
                    
                    logger.info("Running scheduled completeness check...")
                    await self._run_completeness_check()
                    self.last_completeness_check = datetime.now()
                
                # Check for auto-heal triggers
                await self._check_auto_heal_triggers()
                
                # Sleep for 5 minutes between checks
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error
    
    async def _run_quality_check(self):
        """Run scheduled quality check"""
        try:
            for source_id in self.quality_monitor.data_sources.keys():
                metrics = await self.quality_monitor.check_data_quality(source_id)
                
                # Check for quality issues
                if metrics.overall_score < 0.85:
                    self.system_alerts.append({
                        'timestamp': datetime.now().isoformat(),
                        'type': 'quality_degradation',
                        'severity': 'warning',
                        'message': f"Quality degradation detected in {source_id}",
                        'source': source_id,
                        'score': metrics.overall_score
                    })
            
        except Exception as e:
            logger.error(f"Error in quality check: {e}")
    
    async def _run_completeness_check(self):
        """Run scheduled completeness check"""
        try:
            results = await self.completeness_checker.check_all_sources_completeness(6)
            
            for source, result in results.items():
                if result.priority_level in ['critical', 'high']:
                    self.system_alerts.append({
                        'timestamp': datetime.now().isoformat(),
                        'type': 'completeness_issue',
                        'severity': result.priority_level,
                        'message': f"Completeness issue in {source}: {result.completeness_percentage:.1%}",
                        'source': source,
                        'missing_records': result.missing_records
                    })
            
        except Exception as e:
            logger.error(f"Error in completeness check: {e}")
    
    async def _check_auto_heal_triggers(self):
        """Check if auto-heal should be triggered"""
        try:
            # Get critical issues
            critical_issues = await self.completeness_checker.get_critical_issues()
            
            # Auto-heal if we have critical completeness issues
            critical_completeness_issues = [
                issue for issue in critical_issues 
                if issue['severity'] == 'critical' and issue['type'] == 'low_completeness'
            ]
            
            if len(critical_completeness_issues) >= 2:  # Multiple critical issues
                logger.info("🔧 Auto-triggering heal process due to multiple critical issues")
                await self.auto_heal_data_issues()
            
        except Exception as e:
            logger.error(f"Error checking auto-heal triggers: {e}")
    
    def _get_health_status(self, score: float) -> str:
        """Get health status string"""
        if score >= 0.9:
            return "healthy"
        elif score >= 0.7:
            return "warning"
        elif score >= 0.5:
            return "degraded"
        else:
            return "critical"
    
    async def stop_dashboard(self):
        """Stop the dashboard"""
        logger.info("🛑 Stopping data quality dashboard...")
        self.is_running = False

# Demo function
async def demo_data_quality_dashboard():
    """Demonstrate the data quality dashboard"""
    logger.info("🎯 Starting Data Quality Dashboard Demo")
    
    # Initialize dashboard (with mock credentials)
    dashboard = DataQualityDashboard(
        database_url="postgresql://mock:mock@localhost/mock",
        alchemy_api_key="mock_key",
        dashboard_port=8091
    )
    
    # Demo: Get system status
    logger.info("📊 Demo: Getting system status...")
    status = await dashboard.get_system_status()
    
    if status and 'system_health' in status:
        health_score = status['system_health']['overall_score']
        health_status = status['system_health']['health_status']
        critical_issues = status['system_health']['critical_issues_count']
        
        logger.info(f"  📈 System health: {health_score:.1%} ({health_status})")
        logger.info(f"  🚨 Critical issues: {critical_issues}")
    
    # Demo: Auto-heal process
    logger.info("🔧 Demo: Running auto-heal process...")
    heal_result = await dashboard.auto_heal_data_issues()
    
    if heal_result['success']:
        actions_taken = heal_result['actions_taken']
        logger.info(f"  ✅ Auto-heal completed: {actions_taken} actions taken")
    
    # Demo: Emergency backfill
    logger.info("🚨 Demo: Triggering emergency backfill...")
    backfill_result = await dashboard.trigger_emergency_backfill('graph_api', hours_back=6)
    
    if backfill_result['success']:
        records_processed = backfill_result['records_processed']
        logger.info(f"  ✅ Emergency backfill: {records_processed} records processed")
    
    logger.info("🎉 Data Quality Dashboard Demo completed!")
    logger.info(f"💡 In production, run: python -m data.data_quality_dashboard")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run demo
    asyncio.run(demo_data_quality_dashboard())