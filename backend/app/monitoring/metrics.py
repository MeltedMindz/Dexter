"""
Monitoring and Alerting System for DexBrain
"""
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Callable
import asyncio
import logging
import aiohttp
from prometheus_client import Counter, Gauge, Histogram, start_http_server

class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

@dataclass
class Alert:
    severity: AlertSeverity
    message: str
    timestamp: datetime
    pool_id: Optional[str] = None
    metadata: Optional[Dict] = None

class MetricsCollector:
    """Collects and exports system metrics"""
    
    def __init__(self):
        # Pool metrics
        self.pool_tvl = Gauge('pool_tvl', 'Pool TVL in USD', ['pool_id'])
        self.pool_volume = Counter('pool_volume', 'Pool trading volume', ['pool_id'])
        self.pool_fees = Counter('pool_fees', 'Pool fees earned', ['pool_id'])
        
        # Strategy metrics
        self.strategy_confidence = Gauge(
            'strategy_confidence',
            'Strategy confidence score',
            ['pool_id']
        )
        self.strategy_performance = Gauge(
            'strategy_performance',
            'Strategy performance vs benchmark',
            ['pool_id']
        )
        
        # System metrics
        self.api_request_duration = Histogram(
            'api_request_duration_seconds',
            'API request duration in seconds',
            ['endpoint']
        )
        self.cache_hits = Counter('cache_hits', 'Cache hit count')
        self.cache_misses = Counter('cache_misses', 'Cache miss count')
        
        # RPC metrics
        self.rpc_errors = Counter('rpc_errors', 'RPC error count')
        self.rpc_latency = Histogram(
            'rpc_latency_seconds',
            'RPC request latency in seconds'
        )

    async def update_pool_metrics(self, pool_id: str, metrics: Dict) -> None:
        """Update Prometheus metrics for pool"""
        self.pool_tvl.labels(pool_id=pool_id).set(float(metrics['tvl_usd']))
        self.pool_volume.labels(pool_id=pool_id).inc(float(metrics['volume_24h']))
        self.pool_fees.labels(pool_id=pool_id).inc(float(metrics['fees_24h']))

    async def record_api_duration(self, endpoint: str, duration: float) -> None:
        """Record API request duration"""
        self.api_request_duration.labels(endpoint=endpoint).observe(duration)

class AlertManager:
    """Manages system alerts and notifications"""
    
    def __init__(
        self,
        slack_webhook: Optional[str] = None,
        pagerduty_key: Optional[str] = None
    ):
        self.slack_webhook = slack_webhook
        self.pagerduty_key = pagerduty_key
        self.alert_history: List[Alert] = []
        self.alert_handlers: Dict[AlertSeverity, List[Callable]] = {
            AlertSeverity.INFO: [],
            AlertSeverity.WARNING: [],
            AlertSeverity.CRITICAL: []
        }

    async def send_alert(self, alert: Alert) -> None:
        """Process and send alert"""
        self.alert_history.append(alert)
        
        # Execute handlers for alert severity
        for handler in self.alert_handlers[alert.severity]:
            try:
                await handler(alert)
            except Exception as e:
                logging.error(f"Alert handler failed: {e}")

        # Send to external services based on severity
        if alert.severity == AlertSeverity.CRITICAL:
            await self._send_pagerduty_alert(alert)
        await self._send_slack_alert(alert)

    async def _send_slack_alert(self, alert: Alert) -> None:
        """Send alert to Slack"""
        if not self.slack_webhook:
            return
            
        message = {
            "text": f"[{alert.severity.value.upper()}] {alert.message}",
            "attachments": [{
                "fields": [
                    {"title": "Pool ID", "value": alert.pool_id or "N/A"},
                    {"title": "Timestamp", "value": alert.timestamp.isoformat()}
                ]
            }]
        }
        
        if alert.metadata:
            message["attachments"][0]["fields"].extend([
                {"title": k, "value": str(v)}
                for k, v in alert.metadata.items()
            ])
            
        async with aiohttp.ClientSession() as session:
            try:
                await session.post(self.slack_webhook, json=message)
            except Exception as e:
                logging.error(f"Failed to send Slack alert: {e}")

    async def _send_pagerduty_alert(self, alert: Alert) -> None:
        """Send critical alert to PagerDuty"""
        if not self.pagerduty_key:
            return
            
        payload = {
            "routing_key": self.pagerduty_key,
            "event_action": "trigger",
            "payload": {
                "summary": alert.message,
                "severity": alert.severity.value,
                "source": "dexbrain",
                "custom_details": {
                    "pool_id": alert.pool_id,
                    **(alert.metadata or {})
                }
            }
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                await session.post(
                    "https://events.pagerduty.com/v2/enqueue",
                    json=payload
                )
            except Exception as e:
                logging.error(f"Failed to send PagerDuty alert: {e}")

    def add_alert_handler(
        self,
        severity: AlertSeverity,
        handler: Callable
    ) -> None:
        """Add custom alert handler"""
        self.alert_handlers[severity].append(handler)

class HealthMonitor:
    """Monitors system health and generates alerts"""
    
    def __init__(
        self,
        alert_manager: AlertManager,
        metrics_collector: MetricsCollector
    ):
        self.alert_manager = alert_manager
        self.metrics = metrics_collector
        self.health_checks = []

    async def add_health_check(
        self,
        check_func: Callable,
        interval: int,
        severity: AlertSeverity
    ) -> None:
        """Add new health check"""
        self.health_checks.append({
            "func": check_func,
            "interval": interval,
            "severity": severity,
            "last_run": None,
            "last_success": None
        })

    async def monitor_pool_health(self, pool_id: str, metrics: Dict) -> None:
        """Monitor pool health metrics"""
        # Check for zero TVL
        if Decimal(metrics['tvl_usd']) == 0:
            await self.alert_manager.send_alert(Alert(
                severity=AlertSeverity.WARNING,
                message=f"Pool {pool_id} has zero TVL",
                timestamp=datetime.utcnow(),
                pool_id=pool_id,
                metadata={"tvl": metrics['tvl_usd']}
            ))

        # Check for unusual volume changes
        if self._is_volume_anomaly(metrics['volume_24h']):
            await self.alert_manager.send_alert(Alert(
                severity=AlertSeverity.INFO,
                message=f"Unusual volume detected in pool {pool_id}",
                timestamp=datetime.utcnow(),
                pool_id=pool_id,
                metadata={"volume": metrics['volume_24h']}
            ))

    async def monitor_strategy_performance(
        self,
        pool_id: str,
        performance: Dict
    ) -> None:
        """Monitor strategy performance metrics"""
        if performance['returns'] < -0.10:  # 10% loss threshold
            await self.alert_manager.send_alert(Alert(
                severity=AlertSeverity.CRITICAL,
                message=f"Strategy significant loss in pool {pool_id}",
                timestamp=datetime.utcnow(),
                pool_id=pool_id,
                metadata={"returns": performance['returns']}
            ))

    async def start_monitoring(self) -> None:
        """Start health monitoring loop"""
        while True:
            for check in self.health_checks:
                now = datetime.utcnow()
                
                # Skip if not due to run
                if (check['last_run'] and 
                    (now - check['last_run']).seconds < check['interval']):
                    continue

                try:
                    check['last_run'] = now
                    result = await check['func']()
                    
                    if not result['healthy']:
                        await self.alert_manager.send_alert(Alert(
                            severity=check['severity'],
                            message=result['message'],
                            timestamp=now,
                            metadata=result.get('metadata')
                        ))
                    check['last_success'] = now
                    
                except Exception as e:
                    logging.error(f"Health check failed: {e}")
                    
            await asyncio.sleep(60)  # Check every minute

    def _is_volume_anomaly(self, current_volume: Decimal) -> bool:
        """Detect unusual volume changes"""
        # Implement anomaly detection logic
        pass

# Example usage:
async def setup_monitoring(config: Dict):
    # Initialize components
    metrics = MetricsCollector()
    alerts = AlertManager(
        slack_webhook=config['slack_webhook'],
        pagerduty_key=config['pagerduty_key']
    )
    monitor = HealthMonitor(alerts, metrics)
    
    # Start Prometheus metrics server
    start_http_server(8000)
    
    # Add health checks
    await monitor.add_health_check(
        check_func=check_rpc_connection,
        interval=300,  # 5 minutes
        severity=AlertSeverity.CRITICAL
    )
    
    await monitor.add_health_check(
        check_func=check_cache_health,
        interval=60,  # 1 minute
        severity=AlertSeverity.WARNING
    )
    
    # Start monitoring
    await monitor.start_monitoring()
