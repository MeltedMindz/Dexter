# DexBrain Production Monitoring Guide

## Overview

Comprehensive monitoring and alerting system for the DexBrain production environment on VPS `157.90.230.148`.

## Architecture

### Monitoring Stack

- **Prometheus** - Metrics collection and storage
- **Grafana** - Visualization and dashboards  
- **AlertManager** - Alert routing and notifications
- **Node Exporter** - System metrics (CPU, memory, disk)
- **cAdvisor** - Container metrics
- **Uptime Kuma** - Service uptime monitoring
- **Loki** - Log aggregation
- **Promtail** - Log shipping

### Custom Metrics

- **DexBrain Metrics Exporter** - Business-specific metrics
  - Active/total agents
  - API request rates and errors
  - Data quality scores
  - Network TVL
  - Intelligence query rates

## Quick Setup

### 1. VPS Setup

```bash
# SSH to VPS
ssh root@157.90.230.148

# Clone monitoring configuration
git clone https://github.com/MeltedMindz/Dexter.git
cd Dexter/backend/monitoring

# Make setup script executable
chmod +x setup-monitoring.sh

# Run setup
./setup-monitoring.sh
```

### 2. Configure Credentials

Edit `.env` file:
```env
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK
DATABASE_URL=postgresql://postgres:password@localhost:5432/dexter_db
```

### 3. Start Monitoring

```bash
# Start all monitoring services
docker-compose -f docker-compose.monitoring.yml up -d

# Check status
docker-compose -f docker-compose.monitoring.yml ps

# View logs
docker-compose -f docker-compose.monitoring.yml logs -f
```

## Access URLs

| Service | URL | Credentials |
|---------|-----|-------------|
| Grafana | http://157.90.230.148:3001 | admin/DexBrain2024! |
| Prometheus | http://157.90.230.148:9090 | None |
| AlertManager | http://157.90.230.148:9093 | None |
| Uptime Kuma | http://157.90.230.148:3002 | Setup required |
| Node Exporter | http://157.90.230.148:9100 | None |
| cAdvisor | http://157.90.230.148:8080 | None |

## Key Dashboards

### System Overview
- CPU, Memory, Disk usage
- Network I/O
- Load averages
- System uptime

### DexBrain Application
- API request rates
- Response times
- Error rates
- Active agents
- Data quality metrics

### Database Performance
- Connection counts
- Query performance
- Lock statistics
- Cache hit rates

### Container Monitoring
- Container resource usage
- Restart counts
- Health status
- Log errors

## Critical Alerts

### System Alerts
- **Instance Down** - Service unavailable
- **High CPU Usage** - Above 80% for 10 minutes
- **High Memory Usage** - Above 85% for 10 minutes  
- **Low Disk Space** - Below 10% available
- **Container Restarting** - Unexpected restarts

### Application Alerts
- **DexBrain API Down** - API server unavailable
- **High API Error Rate** - Above 10% error rate
- **Slow API Response** - 95th percentile above 2 seconds
- **Low Agent Activity** - Less than 5 active agents
- **Data Quality Issues** - Above 20% failure rate

### Database Alerts
- **PostgreSQL Down** - Database unavailable
- **High Connection Usage** - Above 80% of max connections
- **Slow Queries** - Performance degradation detected

## Alert Notifications

### Email Notifications
- **Critical alerts** → admin@dexteragent.com
- **Warning alerts** → alerts@dexteragent.com

### Slack Notifications
- **Critical alerts** → #dexbrain-alerts channel
- Configure webhook URL in AlertManager

### Alert Escalation
1. **Immediate** - Email/Slack notification
2. **5 minutes** - Repeat notification if unresolved
3. **30 minutes** - Escalate to on-call engineer
4. **2 hours** - Executive notification for critical issues

## Maintenance

### Daily Tasks
- Review Grafana dashboards for anomalies
- Check AlertManager for active alerts
- Verify backup completion
- Monitor disk space usage

### Weekly Tasks
- Review alert thresholds and tune as needed
- Update monitoring documentation
- Test alert notification channels
- Analyze performance trends

### Monthly Tasks
- Rotate log files and clean old data
- Update monitoring stack (security patches)
- Review and optimize alert rules
- Capacity planning based on growth trends

## Troubleshooting

### Common Issues

#### Prometheus Not Scraping Targets
```bash
# Check target status
curl http://157.90.230.148:9090/api/v1/targets

# Verify network connectivity
docker exec dexbrain-prometheus nc -zv target-host port

# Check configuration
docker exec dexbrain-prometheus cat /etc/prometheus/prometheus.yml
```

#### Grafana Dashboard Not Loading
```bash
# Check Grafana logs
docker logs dexbrain-grafana

# Verify data source connectivity
curl http://157.90.230.148:3001/api/datasources/proxy/1/api/v1/query?query=up

# Reset admin password
docker exec -it dexbrain-grafana grafana-cli admin reset-admin-password DexBrain2024!
```

#### Alerts Not Firing
```bash
# Check AlertManager status
curl http://157.90.230.148:9093/api/v1/status

# Verify rule syntax
docker exec dexbrain-prometheus promtool check rules /etc/prometheus/alert-rules.yml

# Test notification channels
docker exec -it dexbrain-alertmanager amtool alert add alertname=test severity=warning
```

### Log Locations

```bash
# Application logs
docker logs dexbrain-api
docker logs dexter-frontend

# Monitoring logs
docker logs dexbrain-prometheus
docker logs dexbrain-grafana
docker logs dexbrain-alertmanager

# System logs
tail -f /var/log/syslog
journalctl -u docker.service -f
```

## Security

### Network Security
- Configure firewall rules for monitoring ports
- Use VPN or SSH tunneling for external access
- Enable HTTPS with proper certificates

### Access Control
- Change default Grafana admin password
- Create read-only users for dashboard viewing
- Use API keys for programmatic access

### Data Protection
- Regular backups of monitoring data
- Encrypt sensitive alert configurations
- Monitor for unauthorized access attempts

## Backup Strategy

### Configuration Backup
```bash
# Backup monitoring configuration
tar -czf monitoring-config-$(date +%Y%m%d).tar.gz \
  /opt/dexbrain-monitoring/

# Store in secure location
scp monitoring-config-*.tar.gz backup-server:/backups/
```

### Data Backup
```bash
# Backup Prometheus data
docker run --rm -v prometheus_data:/source -v /backup:/backup \
  alpine tar -czf /backup/prometheus-$(date +%Y%m%d).tar.gz -C /source .

# Backup Grafana data
docker run --rm -v grafana_data:/source -v /backup:/backup \
  alpine tar -czf /backup/grafana-$(date +%Y%m%d).tar.gz -C /source .
```

## Performance Optimization

### Resource Allocation
- **Prometheus**: 2GB RAM, 50GB disk
- **Grafana**: 512MB RAM
- **AlertManager**: 256MB RAM
- **Exporters**: 128MB RAM each

### Data Retention
- **Prometheus**: 30 days (configurable)
- **Loki**: 7 days (log retention)
- **AlertManager**: 120 hours (alert history)

### Query Optimization
- Use recording rules for expensive queries
- Limit dashboard time ranges
- Optimize PromQL queries for performance

## Integration

### API Monitoring
Add metrics to your application:
```python
from prometheus_client import Counter, Histogram, generate_latest

REQUEST_COUNT = Counter('api_requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_LATENCY = Histogram('api_request_duration_seconds', 'Request latency')

@REQUEST_LATENCY.time()
def api_endpoint():
    REQUEST_COUNT.labels(method='GET', endpoint='/api/data').inc()
    # Your API logic here
```

### Custom Alerts
Add business-specific alerts:
```yaml
- alert: HighUserSignupRate
  expr: rate(user_registrations_total[5m]) > 10
  for: 5m
  labels:
    severity: info
  annotations:
    summary: "High user signup rate detected"
```

---

**🔧 Support**: For monitoring issues, check logs and contact the infrastructure team.