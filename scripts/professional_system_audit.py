#!/usr/bin/env python3
"""
Professional System Audit and Optimization for Dexter AI
Comprehensive monitoring, optimization, and real-time analytics dashboard
"""

import asyncio
import subprocess
import logging
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import psutil

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProfessionalSystemAudit:
    """
    Enterprise-grade system monitoring and optimization for Dexter AI
    """
    
    def __init__(self):
        self.vps_ip = "5.78.71.231"
        self.vps_user = "root"
        self.remote_path = "/opt/dexter-ai"
        
        # Services to monitor
        self.critical_services = [
            "dexter-enhanced-learning.service",
            "postgresql",
            "redis-server",
            "nginx"
        ]
        
        # Performance thresholds
        self.thresholds = {
            'cpu_usage': 80.0,
            'memory_usage': 85.0,
            'disk_usage': 90.0,
            'load_average': 4.0,
            'network_errors': 100,
            'log_error_rate': 10  # errors per minute
        }
        
        # Monitoring state
        self.monitoring_data = {
            'system_health': {},
            'service_status': {},
            'performance_metrics': {},
            'ml_pipeline_stats': {},
            'blockchain_data_quality': {},
            'alerts': []
        }
    
    async def run_comprehensive_audit(self):
        """
        Execute comprehensive system audit and optimization
        """
        logger.info("🔍 Starting comprehensive professional system audit...")
        
        try:
            # System health checks
            await self._audit_system_health()
            
            # Service status monitoring
            await self._audit_services()
            
            # Performance analysis
            await self._audit_performance()
            
            # ML pipeline health
            await self._audit_ml_pipeline()
            
            # Blockchain data quality
            await self._audit_blockchain_data()
            
            # Security audit
            await self._audit_security()
            
            # Optimization recommendations
            await self._generate_optimizations()
            
            # Real-time monitoring setup
            await self._setup_realtime_monitoring()
            
            # Generate comprehensive report
            await self._generate_audit_report()
            
        except Exception as e:
            logger.error(f"Audit failed: {e}")
            raise
    
    async def _audit_system_health(self):
        """Comprehensive system health audit"""
        logger.info("📊 Auditing system health...")
        
        commands = [
            # System resources
            ("cpu_info", "cat /proc/cpuinfo | grep 'model name' | head -1"),
            ("memory_info", "free -h"),
            ("disk_usage", "df -h"),
            ("load_average", "uptime"),
            ("network_stats", "ss -tuln | wc -l"),
            
            # Process monitoring
            ("top_processes", "ps aux --sort=-%mem | head -10"),
            ("python_processes", "ps aux | grep python"),
            
            # System logs
            ("kernel_errors", "dmesg | grep -i error | tail -5"),
            ("system_errors", "journalctl -p err --since '1 hour ago' --no-pager"),
            
            # Network connectivity
            ("external_connectivity", "curl -s -o /dev/null -w '%{http_code}' https://api.thegraph.com"),
            ("dns_resolution", "nslookup google.com")
        ]
        
        health_data = {}
        for name, command in commands:
            try:
                result = await self._run_ssh_command(command, capture_output=True)
                health_data[name] = result
            except Exception as e:
                health_data[name] = f"Error: {e}"
                logger.warning(f"Health check {name} failed: {e}")
        
        self.monitoring_data['system_health'] = health_data
        logger.info(f"✅ System health audit completed ({len(health_data)} metrics)")
    
    async def _audit_services(self):
        """Audit critical services status"""
        logger.info("🔧 Auditing service status...")
        
        service_data = {}
        for service in self.critical_services:
            try:
                # Service status
                status = await self._run_ssh_command(f"systemctl is-active {service}", capture_output=True)
                enabled = await self._run_ssh_command(f"systemctl is-enabled {service}", capture_output=True)
                
                # Service logs
                logs = await self._run_ssh_command(f"journalctl -u {service} --since '10 minutes ago' --no-pager -n 20", capture_output=True)
                
                # Process info if running
                if "active" in status:
                    process_info = await self._run_ssh_command(f"systemctl show {service} --property=MainPID,MemoryCurrent,CPUUsageNSec", capture_output=True)
                else:
                    process_info = "Service not running"
                
                service_data[service] = {
                    'status': status.strip(),
                    'enabled': enabled.strip(),
                    'recent_logs': logs.split('\n')[-5:],  # Last 5 log lines
                    'process_info': process_info
                }
                
                # Health check for enhanced learning service
                if service == "dexter-enhanced-learning.service" and "active" in status:
                    health = await self._check_ml_service_health()
                    service_data[service]['ml_health'] = health
                
            except Exception as e:
                service_data[service] = {'error': str(e)}
                logger.warning(f"Service audit for {service} failed: {e}")
        
        self.monitoring_data['service_status'] = service_data
        logger.info(f"✅ Service audit completed ({len(service_data)} services)")
    
    async def _audit_performance(self):
        """Comprehensive performance analysis"""
        logger.info("⚡ Auditing system performance...")
        
        performance_data = {}
        
        try:
            # CPU and Memory
            cpu_usage = await self._run_ssh_command("top -bn1 | grep 'Cpu(s)' | awk '{print $2}' | cut -d'%' -f1", capture_output=True)
            memory_usage = await self._run_ssh_command("free | grep Mem | awk '{printf \"%.1f\", ($3/$2) * 100.0}'", capture_output=True)
            
            # Disk I/O
            disk_io = await self._run_ssh_command("iostat -x 1 1 | tail -n +4", capture_output=True)
            
            # Network traffic
            network_stats = await self._run_ssh_command("cat /proc/net/dev | grep eth0", capture_output=True)
            
            # Database performance (if PostgreSQL is running)
            try:
                db_connections = await self._run_ssh_command("sudo -u postgres psql -c 'SELECT count(*) FROM pg_stat_activity;' -t", capture_output=True)
                db_size = await self._run_ssh_command("sudo -u postgres psql -c 'SELECT pg_size_pretty(pg_database_size(current_database()));' -t", capture_output=True)
            except:
                db_connections = "N/A"
                db_size = "N/A"
            
            # Redis performance (if Redis is running)
            try:
                redis_info = await self._run_ssh_command("redis-cli info memory | grep used_memory_human", capture_output=True)
            except:
                redis_info = "N/A"
            
            performance_data = {
                'cpu_usage_percent': float(cpu_usage.strip()) if cpu_usage.strip().replace('.', '').isdigit() else 0,
                'memory_usage_percent': float(memory_usage.strip()) if memory_usage.strip().replace('.', '').isdigit() else 0,
                'disk_io_stats': disk_io,
                'network_stats': network_stats,
                'database_connections': db_connections.strip(),
                'database_size': db_size.strip(),
                'redis_memory': redis_info.strip(),
                'timestamp': datetime.now().isoformat()
            }
            
            # Performance alerts
            alerts = []
            if performance_data['cpu_usage_percent'] > self.thresholds['cpu_usage']:
                alerts.append(f"HIGH CPU: {performance_data['cpu_usage_percent']:.1f}%")
            
            if performance_data['memory_usage_percent'] > self.thresholds['memory_usage']:
                alerts.append(f"HIGH MEMORY: {performance_data['memory_usage_percent']:.1f}%")
            
            performance_data['alerts'] = alerts
            
        except Exception as e:
            performance_data = {'error': str(e)}
            logger.error(f"Performance audit failed: {e}")
        
        self.monitoring_data['performance_metrics'] = performance_data
        logger.info(f"✅ Performance audit completed")
    
    async def _audit_ml_pipeline(self):
        """Audit ML pipeline health and performance"""
        logger.info("🤖 Auditing ML pipeline...")
        
        ml_data = {}
        
        try:
            # Check ML service logs
            ml_logs = await self._run_ssh_command("tail -50 /var/log/dexter/enhanced_learning.log", capture_output=True)
            
            # Check website logs for ML activity
            website_logs = await self._run_ssh_command("tail -20 /var/log/dexter/liquidity.log | grep DexBrain", capture_output=True)
            
            # Check model files
            model_files = await self._run_ssh_command("find /opt/dexter-ai/model_storage -name '*.pth' -o -name '*.pkl' | wc -l", capture_output=True)
            
            # Check knowledge base
            kb_size = await self._run_ssh_command("find /opt/dexter-ai/knowledge_base -type f | wc -l", capture_output=True)
            
            # GPU/PyTorch info (if available)
            try:
                torch_info = await self._run_ssh_command("cd /opt/dexter-ai && python3 -c 'import torch; print(f\"PyTorch: {torch.__version__}, Device: {torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")}'\"", capture_output=True)
            except:
                torch_info = "PyTorch CPU-only"
            
            # Memory usage of ML processes
            ml_memory = await self._run_ssh_command("ps aux | grep 'enhanced_continuous_learning' | grep -v grep | awk '{print $6}'", capture_output=True)
            
            ml_data = {
                'service_logs': ml_logs.split('\n')[-10:],  # Last 10 lines
                'website_activity': website_logs.split('\n')[-5:],  # Last 5 DexBrain logs
                'model_files_count': int(model_files.strip()) if model_files.strip().isdigit() else 0,
                'knowledge_base_files': int(kb_size.strip()) if kb_size.strip().isdigit() else 0,
                'pytorch_info': torch_info.strip(),
                'memory_usage_kb': ml_memory.strip(),
                'pipeline_health': 'healthy' if int(model_files.strip() or 0) > 0 else 'initializing'
            }
            
        except Exception as e:
            ml_data = {'error': str(e)}
            logger.error(f"ML pipeline audit failed: {e}")
        
        self.monitoring_data['ml_pipeline_stats'] = ml_data
        logger.info(f"✅ ML pipeline audit completed")
    
    async def _audit_blockchain_data(self):
        """Audit blockchain data quality and connectivity"""
        logger.info("⛓️ Auditing blockchain data quality...")
        
        blockchain_data = {}
        
        try:
            # Test The Graph API connectivity
            graph_test = await self._run_ssh_command(
                "curl -s -H 'Authorization: Bearer c6f241c1dd5aea81977a63b2614af70d' -H 'Content-Type: application/json' -d '{\"query\": \"{pools(first: 1) {id}}\"}' https://gateway-arbitrum.network.thegraph.com/api/c6f241c1dd5aea81977a63b2614af70d/subgraphs/id/5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV",
                capture_output=True
            )
            
            # Test external API endpoints
            endpoints_test = {}
            test_urls = [
                ("coingecko", "https://api.coingecko.com/api/v3/ping"),
                ("defillama", "https://api.llama.fi/protocols"),
                ("base_rpc", "https://base-mainnet.g.alchemy.com/v2/demo")
            ]
            
            for name, url in test_urls:
                try:
                    response = await self._run_ssh_command(f"curl -s -o /dev/null -w '%{{http_code}}' {url}", capture_output=True)
                    endpoints_test[name] = response.strip()
                except:
                    endpoints_test[name] = "error"
            
            # Check recent data ingestion
            try:
                recent_insights = await self._run_ssh_command("grep -c 'insights created' /var/log/dexter/enhanced_learning.log | tail -1", capture_output=True)
            except:
                recent_insights = "0"
            
            blockchain_data = {
                'graph_api_test': "success" if "pools" in graph_test else "failed",
                'graph_response_size': len(graph_test),
                'external_endpoints': endpoints_test,
                'recent_insights_count': recent_insights.strip(),
                'data_quality_score': self._calculate_data_quality_score(endpoints_test),
                'last_check': datetime.now().isoformat()
            }
            
        except Exception as e:
            blockchain_data = {'error': str(e)}
            logger.error(f"Blockchain data audit failed: {e}")
        
        self.monitoring_data['blockchain_data_quality'] = blockchain_data
        logger.info(f"✅ Blockchain data audit completed")
    
    async def _audit_security(self):
        """Security audit and hardening check"""
        logger.info("🔒 Auditing security...")
        
        security_checks = [
            # System security
            ("ssh_config", "cat /etc/ssh/sshd_config | grep -E '(PermitRootLogin|PasswordAuthentication|Port)'"),
            ("firewall_status", "ufw status"),
            ("failed_logins", "grep 'Failed password' /var/log/auth.log | tail -5"),
            
            # File permissions
            ("sensitive_files", "ls -la /opt/dexter-ai/backend/dexbrain/config.py"),
            ("log_permissions", "ls -la /var/log/dexter/"),
            
            # Process security
            ("running_as_root", "ps aux | grep -E '(python|node)' | grep root | wc -l"),
            
            # Network security
            ("open_ports", "netstat -tuln | grep LISTEN"),
            ("active_connections", "netstat -tuln | wc -l")
        ]
        
        security_data = {}
        for name, command in security_checks:
            try:
                result = await self._run_ssh_command(command, capture_output=True, ignore_errors=True)
                security_data[name] = result
            except Exception as e:
                security_data[name] = f"Error: {e}"
        
        # Security recommendations
        recommendations = []
        if "yes" in security_data.get("ssh_config", "").lower():
            recommendations.append("Consider disabling root SSH login")
        
        if int(security_data.get("running_as_root", "0").strip() or "0") > 3:
            recommendations.append("Multiple processes running as root - consider using dedicated users")
        
        security_data['recommendations'] = recommendations
        
        self.monitoring_data['security_audit'] = security_data
        logger.info(f"✅ Security audit completed")
    
    def _calculate_data_quality_score(self, endpoints_test: Dict[str, str]) -> float:
        """Calculate overall data quality score"""
        working_endpoints = sum(1 for status in endpoints_test.values() if status.startswith("20"))
        total_endpoints = len(endpoints_test)
        return working_endpoints / max(total_endpoints, 1)
    
    async def _generate_optimizations(self):
        """Generate system optimization recommendations"""
        logger.info("🎯 Generating optimization recommendations...")
        
        optimizations = []
        
        # Performance optimizations
        perf = self.monitoring_data.get('performance_metrics', {})
        if perf.get('cpu_usage_percent', 0) > 80:
            optimizations.append({
                'type': 'performance',
                'priority': 'high',
                'issue': 'High CPU usage',
                'recommendation': 'Consider reducing ML training frequency or optimizing algorithms',
                'command': 'systemctl status dexter-enhanced-learning.service'
            })
        
        if perf.get('memory_usage_percent', 0) > 85:
            optimizations.append({
                'type': 'performance',
                'priority': 'high',
                'issue': 'High memory usage',
                'recommendation': 'Implement memory optimization in ML pipeline',
                'command': 'free -h && ps aux --sort=-%mem | head -10'
            })
        
        # Service optimizations
        for service, data in self.monitoring_data.get('service_status', {}).items():
            if data.get('status') != 'active':
                optimizations.append({
                    'type': 'service',
                    'priority': 'critical',
                    'issue': f'{service} not running',
                    'recommendation': f'Restart {service} and check configuration',
                    'command': f'systemctl restart {service} && systemctl status {service}'
                })
        
        # ML pipeline optimizations
        ml_data = self.monitoring_data.get('ml_pipeline_stats', {})
        if ml_data.get('model_files_count', 0) == 0:
            optimizations.append({
                'type': 'ml',
                'priority': 'medium',
                'issue': 'No ML model files found',
                'recommendation': 'Initialize ML models with training data',
                'command': 'Check /opt/dexter-ai/model_storage/ and training logs'
            })
        
        # Data quality optimizations
        blockchain_data = self.monitoring_data.get('blockchain_data_quality', {})
        if blockchain_data.get('data_quality_score', 1.0) < 0.8:
            optimizations.append({
                'type': 'data',
                'priority': 'medium',
                'issue': 'Poor blockchain data quality',
                'recommendation': 'Check API endpoints and network connectivity',
                'command': 'Test external API connections and GraphQL endpoints'
            })
        
        self.monitoring_data['optimizations'] = optimizations
        logger.info(f"✅ Generated {len(optimizations)} optimization recommendations")
    
    async def _setup_realtime_monitoring(self):
        """Setup real-time monitoring and alerting"""
        logger.info("📡 Setting up real-time monitoring...")
        
        monitoring_script = f"""#!/bin/bash
# Real-time monitoring script for Dexter AI
while true; do
    timestamp=$(date --iso-8601=seconds)
    
    # System metrics
    cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{{print $2}}' | cut -d'%' -f1)
    memory_usage=$(free | grep Mem | awk '{{printf "%.1f", ($3/$2) * 100.0}}')
    
    # Service status
    ml_service_status=$(systemctl is-active dexter-enhanced-learning.service)
    
    # Log to monitoring file
    echo "$timestamp,CPU:$cpu_usage,MEM:$memory_usage,ML:$ml_service_status" >> /var/log/dexter/realtime_monitoring.log
    
    # Alert on critical issues
    if (( $(echo "$cpu_usage > 90" | bc -l) )); then
        echo "$timestamp CRITICAL: CPU usage $cpu_usage%" >> /var/log/dexter/alerts.log
    fi
    
    if (( $(echo "$memory_usage > 95" | bc -l) )); then
        echo "$timestamp CRITICAL: Memory usage $memory_usage%" >> /var/log/dexter/alerts.log
    fi
    
    if [ "$ml_service_status" != "active" ]; then
        echo "$timestamp CRITICAL: ML service down" >> /var/log/dexter/alerts.log
        systemctl restart dexter-enhanced-learning.service
    fi
    
    sleep 30
done
"""
        
        # Upload monitoring script
        await self._run_ssh_command(f"cat > /opt/dexter-ai/realtime_monitor.sh << 'EOF'\n{monitoring_script}EOF")
        await self._run_ssh_command("chmod +x /opt/dexter-ai/realtime_monitor.sh")
        
        # Create systemd service for monitoring
        monitoring_service = f"""[Unit]
Description=Dexter AI Real-time Monitoring
After=network.target

[Service]
Type=simple
User=root
ExecStart=/opt/dexter-ai/realtime_monitor.sh
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""
        
        await self._run_ssh_command(f"cat > /etc/systemd/system/dexter-monitoring.service << 'EOF'\n{monitoring_service}EOF")
        await self._run_ssh_command("systemctl daemon-reload")
        await self._run_ssh_command("systemctl enable dexter-monitoring.service")
        await self._run_ssh_command("systemctl start dexter-monitoring.service")
        
        logger.info("✅ Real-time monitoring setup completed")
    
    async def _check_ml_service_health(self) -> Dict[str, Any]:
        """Detailed health check for ML service"""
        try:
            # Check if service is processing data
            recent_logs = await self._run_ssh_command("tail -20 /var/log/dexter/enhanced_learning.log", capture_output=True)
            
            # Look for learning activity
            learning_activity = "learning cycle" in recent_logs.lower()
            ml_predictions = "ml predictions" in recent_logs.lower()
            training_activity = "training" in recent_logs.lower()
            
            # Check error rate
            errors = recent_logs.lower().count("error")
            warnings = recent_logs.lower().count("warning")
            
            # Calculate health score
            health_score = 1.0
            if errors > 5:
                health_score -= 0.3
            if warnings > 10:
                health_score -= 0.2
            if not learning_activity:
                health_score -= 0.3
            if not ml_predictions:
                health_score -= 0.2
            
            return {
                'health_score': max(0, health_score),
                'learning_activity': learning_activity,
                'ml_predictions': ml_predictions,
                'training_activity': training_activity,
                'error_count': errors,
                'warning_count': warnings,
                'status': 'healthy' if health_score > 0.7 else 'degraded' if health_score > 0.4 else 'critical'
            }
            
        except Exception as e:
            return {'error': str(e), 'status': 'unknown'}
    
    async def _generate_audit_report(self):
        """Generate comprehensive audit report"""
        logger.info("📋 Generating comprehensive audit report...")
        
        report = {
            'audit_timestamp': datetime.now().isoformat(),
            'system_summary': self._generate_system_summary(),
            'detailed_findings': self.monitoring_data,
            'optimization_priorities': self._prioritize_optimizations(),
            'next_steps': self._generate_next_steps()
        }
        
        # Save report locally
        report_path = Path("dexter_audit_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Display summary
        self._display_audit_summary()
        
        logger.info(f"✅ Audit report saved to {report_path}")
    
    def _generate_system_summary(self) -> Dict[str, Any]:
        """Generate high-level system summary"""
        perf = self.monitoring_data.get('performance_metrics', {})
        services = self.monitoring_data.get('service_status', {})
        ml_stats = self.monitoring_data.get('ml_pipeline_stats', {})
        blockchain = self.monitoring_data.get('blockchain_data_quality', {})
        
        # Count active services
        active_services = sum(1 for service_data in services.values() 
                            if isinstance(service_data, dict) and service_data.get('status') == 'active')
        
        return {
            'overall_health': self._calculate_overall_health(),
            'cpu_usage': perf.get('cpu_usage_percent', 0),
            'memory_usage': perf.get('memory_usage_percent', 0),
            'active_services': f"{active_services}/{len(self.critical_services)}",
            'ml_pipeline_status': ml_stats.get('pipeline_health', 'unknown'),
            'data_quality': blockchain.get('data_quality_score', 0),
            'critical_issues': len([opt for opt in self.monitoring_data.get('optimizations', []) 
                                  if opt.get('priority') == 'critical'])
        }
    
    def _calculate_overall_health(self) -> str:
        """Calculate overall system health score"""
        scores = []
        
        # Performance score
        perf = self.monitoring_data.get('performance_metrics', {})
        cpu_score = max(0, 1 - (perf.get('cpu_usage_percent', 0) / 100))
        mem_score = max(0, 1 - (perf.get('memory_usage_percent', 0) / 100))
        scores.extend([cpu_score, mem_score])
        
        # Service score
        services = self.monitoring_data.get('service_status', {})
        active_services = sum(1 for data in services.values() 
                            if isinstance(data, dict) and data.get('status') == 'active')
        service_score = active_services / len(self.critical_services)
        scores.append(service_score)
        
        # Data quality score
        data_score = self.monitoring_data.get('blockchain_data_quality', {}).get('data_quality_score', 0)
        scores.append(data_score)
        
        overall_score = sum(scores) / len(scores)
        
        if overall_score > 0.8:
            return "excellent"
        elif overall_score > 0.6:
            return "good"
        elif overall_score > 0.4:
            return "fair"
        else:
            return "poor"
    
    def _prioritize_optimizations(self) -> List[Dict[str, Any]]:
        """Prioritize optimizations by impact and urgency"""
        optimizations = self.monitoring_data.get('optimizations', [])
        
        priority_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        
        return sorted(optimizations, key=lambda x: priority_order.get(x.get('priority', 'low'), 3))
    
    def _generate_next_steps(self) -> List[str]:
        """Generate actionable next steps"""
        next_steps = [
            "Monitor real-time metrics via /var/log/dexter/realtime_monitoring.log",
            "Check ML service logs regularly: tail -f /var/log/dexter/enhanced_learning.log",
            "Verify blockchain data ingestion every hour",
            "Review optimization recommendations and implement high-priority items",
            "Set up automated alerting for critical thresholds"
        ]
        
        # Add specific steps based on findings
        optimizations = self.monitoring_data.get('optimizations', [])
        critical_issues = [opt for opt in optimizations if opt.get('priority') == 'critical']
        
        if critical_issues:
            next_steps.insert(0, f"URGENT: Address {len(critical_issues)} critical issues immediately")
        
        return next_steps
    
    def _display_audit_summary(self):
        """Display comprehensive audit summary"""
        summary = self._generate_system_summary()
        optimizations = self.monitoring_data.get('optimizations', [])
        
        print("\n" + "="*80)
        print("🏆 DEXTER AI PROFESSIONAL SYSTEM AUDIT SUMMARY")
        print("="*80)
        
        print(f"\n📊 OVERALL HEALTH: {summary['overall_health'].upper()}")
        print(f"🖥️  CPU Usage: {summary['cpu_usage']:.1f}%")
        print(f"💾 Memory Usage: {summary['memory_usage']:.1f}%")
        print(f"⚙️  Active Services: {summary['active_services']}")
        print(f"🤖 ML Pipeline: {summary['ml_pipeline_status']}")
        print(f"⛓️  Data Quality: {summary['data_quality']:.1%}")
        
        print(f"\n🚨 ISSUES FOUND:")
        if optimizations:
            for opt in optimizations[:5]:  # Show top 5
                priority_icon = "🔴" if opt['priority'] == 'critical' else "🟡" if opt['priority'] == 'high' else "🟢"
                print(f"   {priority_icon} {opt['issue']} ({opt['priority']})")
        else:
            print("   ✅ No issues found")
        
        print(f"\n🎯 TOP RECOMMENDATIONS:")
        for i, step in enumerate(self._generate_next_steps()[:3], 1):
            print(f"   {i}. {step}")
        
        print("\n" + "="*80)
    
    async def _run_ssh_command(self, command: str, capture_output: bool = False, ignore_errors: bool = False, timeout: int = 30) -> str:
        """Execute SSH command on VPS"""
        ssh_cmd = f"ssh {self.vps_user}@{self.vps_ip} '{command}'"
        
        try:
            if capture_output:
                result = subprocess.run(ssh_cmd, shell=True, capture_output=True, text=True, timeout=timeout)
                if result.returncode != 0 and not ignore_errors:
                    raise subprocess.CalledProcessError(result.returncode, ssh_cmd, result.stderr)
                return result.stdout.strip()
            else:
                result = subprocess.run(ssh_cmd, shell=True, timeout=timeout)
                if result.returncode != 0 and not ignore_errors:
                    raise subprocess.CalledProcessError(result.returncode, ssh_cmd)
                return ""
        except subprocess.TimeoutExpired:
            logger.error(f"SSH command timed out: {command}")
            raise
        except subprocess.CalledProcessError as e:
            if not ignore_errors:
                logger.error(f"SSH command failed: {command} - {e}")
                raise
            return ""


async def main():
    """Main function to run professional system audit"""
    audit = ProfessionalSystemAudit()
    await audit.run_comprehensive_audit()

if __name__ == "__main__":
    asyncio.run(main())