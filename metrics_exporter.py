#!/usr/bin/env python3
"""
Dexter AI Custom Metrics Exporter
Exposes Prometheus metrics for all Dexter services
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime, timedelta
from aiohttp import web
import logging
import subprocess
import psutil
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DexterMetricsExporter:
    """Export custom Dexter AI metrics for Prometheus"""
    
    def __init__(self):
        self.app = web.Application()
        self.app.router.add_get('/metrics', self.metrics_handler)
        self.app.router.add_get('/health', self.health_handler)
        
        # Metrics cache
        self.metrics_cache = {}
        self.cache_ttl = 30  # 30 seconds
        self.last_update = 0
        
        logger.info("Dexter metrics exporter initialized")
    
    async def metrics_handler(self, request):
        """Handle /metrics endpoint for Prometheus"""
        try:
            metrics = await self.collect_metrics()
            return web.Response(text=metrics, content_type='text/plain')
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            return web.Response(text="# Error collecting metrics\n", status=500)
    
    async def health_handler(self, request):
        """Health check endpoint"""
        return web.json_response({"status": "healthy", "timestamp": datetime.now().isoformat()})
    
    async def collect_metrics(self):
        """Collect all Dexter metrics"""
        current_time = time.time()
        
        # Use cache if recent
        if current_time - self.last_update < self.cache_ttl and self.metrics_cache:
            return self.metrics_cache.get('metrics', '')
        
        metrics_lines = []
        
        # Add timestamp
        metrics_lines.append(f"# HELP dexter_metrics_collection_timestamp_seconds Last metrics collection time")
        metrics_lines.append(f"# TYPE dexter_metrics_collection_timestamp_seconds gauge")
        metrics_lines.append(f"dexter_metrics_collection_timestamp_seconds {current_time}")
        
        # System metrics
        metrics_lines.extend(await self._collect_system_metrics())
        
        # Service status metrics
        metrics_lines.extend(await self._collect_service_metrics())
        
        # Position harvester metrics
        metrics_lines.extend(await self._collect_harvester_metrics())
        
        # Trading performance metrics
        metrics_lines.extend(await self._collect_trading_metrics())
        
        # Data quality metrics
        metrics_lines.extend(await self._collect_data_quality_metrics())
        
        metrics_text = "\n".join(metrics_lines) + "\n"
        
        # Update cache
        self.metrics_cache['metrics'] = metrics_text
        self.last_update = current_time
        
        return metrics_text
    
    async def _collect_system_metrics(self):
        """Collect system-level metrics"""
        metrics = []
        
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            metrics.append("# HELP dexter_system_cpu_percent System CPU usage percentage")
            metrics.append("# TYPE dexter_system_cpu_percent gauge")
            metrics.append(f"dexter_system_cpu_percent {cpu_percent}")
            
            # Memory usage
            memory = psutil.virtual_memory()
            metrics.append("# HELP dexter_system_memory_usage_bytes System memory usage in bytes")
            metrics.append("# TYPE dexter_system_memory_usage_bytes gauge")
            metrics.append(f"dexter_system_memory_usage_bytes {memory.used}")
            
            metrics.append("# HELP dexter_system_memory_total_bytes System total memory in bytes")
            metrics.append("# TYPE dexter_system_memory_total_bytes gauge")
            metrics.append(f"dexter_system_memory_total_bytes {memory.total}")
            
            # Disk usage
            disk = psutil.disk_usage('/')
            metrics.append("# HELP dexter_system_disk_usage_bytes System disk usage in bytes")
            metrics.append("# TYPE dexter_system_disk_usage_bytes gauge")
            metrics.append(f"dexter_system_disk_usage_bytes {disk.used}")
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
        
        return metrics
    
    async def _collect_service_metrics(self):
        """Collect service status metrics"""
        metrics = []
        
        try:
            dexter_services = [
                'dexter-position-harvester',
                'dexter-enhanced-alchemy', 
                'dexter-analysis',
                'dexter-log-stream',
                'dexter-log-api',
                'dexter-trading-logs',
                'dexter-twitter-bot'
            ]
            
            metrics.append("# HELP dexter_service_status Service status (1=active, 0=inactive)")
            metrics.append("# TYPE dexter_service_status gauge")
            
            for service in dexter_services:
                try:
                    result = subprocess.run(
                        ['systemctl', 'is-active', f'{service}.service'],
                        capture_output=True, text=True, timeout=5
                    )
                    status = 1 if result.stdout.strip() == 'active' else 0
                    metrics.append(f'dexter_service_status{{service="{service}"}} {status}')
                except Exception as e:
                    logger.warning(f"Failed to check service {service}: {e}")
                    metrics.append(f'dexter_service_status{{service="{service}"}} 0')
            
        except Exception as e:
            logger.error(f"Error collecting service metrics: {e}")
        
        return metrics
    
    async def _collect_harvester_metrics(self):
        """Collect position harvester metrics from logs"""
        metrics = []
        
        try:
            # Parse recent harvester logs
            log_file = '/var/log/dexter/liquidity.log'
            if os.path.exists(log_file):
                positions_harvested = 0
                graph_collections = 0
                alchemy_collections = 0
                
                # Read last 50 lines
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    recent_lines = lines[-50:] if len(lines) > 50 else lines
                    
                    for line in recent_lines:
                        try:
                            if 'GraphHarvest' in line:
                                data = json.loads(line)
                                if 'details' in data and 'positions_processed' in data['details']:
                                    positions_harvested += data['details']['positions_processed']
                                    graph_collections += 1
                            elif 'AlchemyHarvest' in line:
                                data = json.loads(line)
                                if 'details' in data and 'positions_processed' in data['details']:
                                    positions_harvested += data['details']['positions_processed']
                                    alchemy_collections += 1
                        except (json.JSONDecodeError, KeyError):
                            continue
                
                metrics.append("# HELP dexter_positions_harvested_total Total positions harvested")
                metrics.append("# TYPE dexter_positions_harvested_total counter")
                metrics.append(f"dexter_positions_harvested_total {positions_harvested}")
                
                metrics.append("# HELP dexter_graph_collections_total Total Graph API collections")
                metrics.append("# TYPE dexter_graph_collections_total counter")
                metrics.append(f"dexter_graph_collections_total {graph_collections}")
                
                metrics.append("# HELP dexter_alchemy_collections_total Total Alchemy RPC collections")
                metrics.append("# TYPE dexter_alchemy_collections_total counter")
                metrics.append(f"dexter_alchemy_collections_total {alchemy_collections}")
            
        except Exception as e:
            logger.error(f"Error collecting harvester metrics: {e}")
        
        return metrics
    
    async def _collect_trading_metrics(self):
        """Collect enhanced trading performance metrics"""
        metrics = []
        
        try:
            log_file = '/var/log/dexter/liquidity.log'
            if os.path.exists(log_file):
                total_volume = 0
                arbitrage_count = 0
                avg_apr = 0
                apr_values = []
                sharpe_values = []
                drawdown_values = []
                volatility_values = []
                win_count = 0
                loss_count = 0
                
                # Read last 200 lines for enhanced trading data
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    recent_lines = lines[-200:] if len(lines) > 200 else lines
                    
                    for line in recent_lines:
                        try:
                            data = json.loads(line)
                            if 'action' in data and data['action'] == 'ARBITRAGE_EXECUTION':
                                arbitrage_count += 1
                                if 'amount_usd' in data:
                                    total_volume += data['amount_usd']
                                if 'apr_current' in data:
                                    apr_values.append(data['apr_current'])
                                
                                # Enhanced metrics from logs
                                if 'price_impact' in data:
                                    impact = data['price_impact']
                                    if impact > 0:
                                        win_count += 1
                                    else:
                                        loss_count += 1
                            
                            # Look for performance tracking logs
                            elif 'source' in data and data['source'] == 'PerformanceTracker':
                                details = data.get('details', {})
                                if 'sharpe_ratio' in details:
                                    sharpe_values.append(details['sharpe_ratio'])
                                if 'max_drawdown' in details:
                                    drawdown_values.append(abs(details['max_drawdown']))
                                if 'volatility' in details:
                                    volatility_values.append(details['volatility'])
                                    
                        except (json.JSONDecodeError, KeyError):
                            continue
                
                # Calculate averages
                if apr_values:
                    avg_apr = sum(apr_values) / len(apr_values)
                
                avg_sharpe = sum(sharpe_values) / len(sharpe_values) if sharpe_values else 0
                avg_drawdown = sum(drawdown_values) / len(drawdown_values) if drawdown_values else 0
                avg_volatility = sum(volatility_values) / len(volatility_values) if volatility_values else 0
                win_rate = win_count / (win_count + loss_count) if (win_count + loss_count) > 0 else 0
                
                # Basic trading metrics
                metrics.append("# HELP dexter_trading_volume_usd_total Total trading volume in USD")
                metrics.append("# TYPE dexter_trading_volume_usd_total counter")
                metrics.append(f"dexter_trading_volume_usd_total {total_volume}")
                
                metrics.append("# HELP dexter_arbitrage_executions_total Total arbitrage executions")
                metrics.append("# TYPE dexter_arbitrage_executions_total counter")
                metrics.append(f"dexter_arbitrage_executions_total {arbitrage_count}")
                
                metrics.append("# HELP dexter_average_apr_percent Average APR percentage")
                metrics.append("# TYPE dexter_average_apr_percent gauge")
                metrics.append(f"dexter_average_apr_percent {avg_apr}")
                
                # Enhanced performance metrics
                metrics.append("# HELP dexter_sharpe_ratio_avg Average Sharpe ratio")
                metrics.append("# TYPE dexter_sharpe_ratio_avg gauge")
                metrics.append(f"dexter_sharpe_ratio_avg {avg_sharpe}")
                
                metrics.append("# HELP dexter_max_drawdown_avg Average maximum drawdown")
                metrics.append("# TYPE dexter_max_drawdown_avg gauge")
                metrics.append(f"dexter_max_drawdown_avg {avg_drawdown}")
                
                metrics.append("# HELP dexter_volatility_avg Average volatility")
                metrics.append("# TYPE dexter_volatility_avg gauge")
                metrics.append(f"dexter_volatility_avg {avg_volatility}")
                
                metrics.append("# HELP dexter_win_rate Win rate percentage")
                metrics.append("# TYPE dexter_win_rate gauge")
                metrics.append(f"dexter_win_rate {win_rate}")
                
                # Risk metrics
                metrics.append("# HELP dexter_total_trades_count Total number of trades")
                metrics.append("# TYPE dexter_total_trades_count counter")
                metrics.append(f"dexter_total_trades_count {win_count + loss_count}")
                
                metrics.append("# HELP dexter_winning_trades_count Total winning trades")
                metrics.append("# TYPE dexter_winning_trades_count counter")
                metrics.append(f"dexter_winning_trades_count {win_count}")
                
                metrics.append("# HELP dexter_losing_trades_count Total losing trades")
                metrics.append("# TYPE dexter_losing_trades_count counter")
                metrics.append(f"dexter_losing_trades_count {loss_count}")
            
        except Exception as e:
            logger.error(f"Error collecting trading metrics: {e}")
        
        return metrics
    
    async def _collect_data_quality_metrics(self):
        """Collect data quality and ML metrics"""
        metrics = []
        
        try:
            log_file = '/var/log/dexter/liquidity.log'
            if os.path.exists(log_file):
                ml_triggers = 0
                data_quality_score = 0
                
                # Read last 50 lines for ML data
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    recent_lines = lines[-50:] if len(lines) > 50 else lines
                    
                    for line in recent_lines:
                        try:
                            data = json.loads(line)
                            if 'source' in data:
                                if data['source'] == 'MLTrigger':
                                    ml_triggers += 1
                                elif data['source'] == 'DataQuality' and 'details' in data:
                                    details = data['details']
                                    if 'overall_score' in details:
                                        data_quality_score = details['overall_score']
                        except (json.JSONDecodeError, KeyError):
                            continue
                
                metrics.append("# HELP dexter_ml_training_triggers_total ML training triggers")
                metrics.append("# TYPE dexter_ml_training_triggers_total counter")
                metrics.append(f"dexter_ml_training_triggers_total {ml_triggers}")
                
                metrics.append("# HELP dexter_data_quality_score Data quality score (0-1)")
                metrics.append("# TYPE dexter_data_quality_score gauge")
                metrics.append(f"dexter_data_quality_score {data_quality_score}")
            
        except Exception as e:
            logger.error(f"Error collecting data quality metrics: {e}")
        
        return metrics
    
    async def start_server(self, host='0.0.0.0', port=9091):
        """Start the metrics server"""
        logger.info(f"Starting Dexter metrics exporter on {host}:{port}")
        
        runner = web.AppRunner(self.app)
        await runner.setup()
        
        site = web.TCPSite(runner, host, port)
        await site.start()
        
        logger.info(f"Metrics server running at http://{host}:{port}/metrics")
        
        # Keep the server running
        try:
            while True:
                await asyncio.sleep(3600)  # Sleep for 1 hour
        except KeyboardInterrupt:
            logger.info("Shutting down metrics server")
        finally:
            await runner.cleanup()

async def main():
    """Run the metrics exporter"""
    exporter = DexterMetricsExporter()
    await exporter.start_server()

if __name__ == "__main__":
    asyncio.run(main())