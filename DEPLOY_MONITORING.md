# Deploy Monitoring to VPS 157.90.230.148

## Quick Deployment

I've created a complete deployment script that will set up monitoring on your VPS with your email (meltedmindz1@gmail.com) configured for alerts.

### 1. Run Deployment Script

```bash
# From your local machine (in the Dexter repository)
./deploy-monitoring-vps.sh
```

This script will:
- ✅ Copy all monitoring configuration to your VPS
- ✅ Configure firewall rules for monitoring ports  
- ✅ Install Docker/Docker Compose if needed
- ✅ Set up your email for alerts
- ✅ Deploy the complete monitoring stack
- ✅ Start custom DexBrain metrics exporter
- ✅ Test all services for health

### 2. Complete Email Setup

To receive email alerts, you need a Gmail App Password:

1. **Go to Google Account Settings**
   - Visit: https://myaccount.google.com/security
   
2. **Enable 2-Factor Authentication** (if not already enabled)

3. **Generate App Password**
   - Go to "App passwords" 
   - Select "Mail" and "Other (custom name)"
   - Name it "DexBrain Monitoring"
   - Copy the 16-character password

4. **Update Configuration on VPS**
   ```bash
   # SSH to your VPS
   ssh root@157.90.230.148
   
   # Edit the environment file
   nano /opt/dexbrain-monitoring/.env
   
   # Replace "your-gmail-app-password-here" with your actual app password
   SMTP_PASSWORD=your-actual-16-char-app-password
   
   # Restart AlertManager to apply changes
   cd /opt/dexbrain-monitoring
   docker-compose -f docker-compose.monitoring.yml restart alertmanager
   ```

### 3. Access Monitoring Dashboards

#### Option A: SSH Tunnel (Recommended)
```bash
# Create SSH tunnel for secure access
ssh -L 3001:localhost:3001 -L 9090:localhost:9090 -L 9093:localhost:9093 root@157.90.230.148

# Then access locally:
# http://localhost:3001 (Grafana - admin/DexBrain2024!)
# http://localhost:9090 (Prometheus)
# http://localhost:9093 (AlertManager)
```

#### Option B: Direct Access (Less Secure)
If you want direct internet access, the firewall is already configured:
- Grafana: http://157.90.230.148:3001 (admin/DexBrain2024!)
- Prometheus: http://157.90.230.148:9090
- AlertManager: http://157.90.230.148:9093

## What's Monitoring

### System Health
- **CPU, Memory, Disk usage** across the VPS
- **Network I/O and connectivity**
- **Docker container performance**
- **Process monitoring**

### DexBrain Application
- **API response times and error rates**
- **Active agent counts**
- **Data quality scores**
- **Database performance**
- **Intelligence query rates**

### Automated Alerts
- **Critical**: System down, high resource usage, API failures
- **Warning**: Performance degradation, capacity approaching limits
- **Email notifications** to meltedmindz1@gmail.com
- **Escalation procedures** for unresolved issues

## Firewall Configuration Applied

The script automatically configures these firewall rules:
```bash
ufw allow 3001/tcp  # Grafana
ufw allow 9090/tcp  # Prometheus  
ufw allow 9093/tcp  # AlertManager
ufw allow 3002/tcp  # Uptime Kuma
ufw allow 9100/tcp  # Node Exporter
ufw allow 8080/tcp  # cAdvisor
ufw allow 8081/tcp  # DexBrain Metrics
```

## Testing Alerts

Once email is configured, test an alert:
```bash
# SSH to VPS
ssh root@157.90.230.148

# Send test alert
docker exec -it dexbrain-alertmanager amtool alert add \
  alertname=TestAlert \
  severity=warning \
  instance=test \
  summary="This is a test alert"

# Check AlertManager UI to see the alert
# You should receive an email within 1-2 minutes
```

## Troubleshooting

### Check Service Status
```bash
ssh root@157.90.230.148
cd /opt/dexbrain-monitoring

# Check all services
docker-compose -f docker-compose.monitoring.yml ps

# Check logs
docker-compose -f docker-compose.monitoring.yml logs grafana
docker-compose -f docker-compose.monitoring.yml logs prometheus
docker-compose -f docker-compose.monitoring.yml logs alertmanager
```

### Restart Services
```bash
# Restart all monitoring services
docker-compose -f docker-compose.monitoring.yml restart

# Or restart individual services
docker-compose -f docker-compose.monitoring.yml restart grafana
```

### Check Custom Metrics
```bash
# Check DexBrain metrics exporter
systemctl status dexbrain-metrics.service
curl http://localhost:8081/metrics
```

## Next Steps After Deployment

1. **Access Grafana** and explore the dashboards
2. **Set up custom dashboards** for your specific KPIs  
3. **Configure Slack notifications** (optional)
4. **Test alert notifications** 
5. **Review alert thresholds** and tune as needed

The monitoring system will now proactively watch your DexBrain infrastructure and alert you to any issues! 🚀