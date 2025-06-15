#!/bin/bash

# DexBrain VPS Monitoring Deployment Script
# Run this script on your local machine to deploy monitoring to VPS 157.90.230.148

set -e

VPS_IP="157.90.230.148"
VPS_USER="root"
EMAIL="meltedmindz1@gmail.com"

echo "🚀 Deploying DexBrain monitoring to VPS ${VPS_IP}..."

# Check if we can connect to VPS
if ! ssh -o ConnectTimeout=10 ${VPS_USER}@${VPS_IP} "echo 'Connection successful'"; then
    echo "❌ Cannot connect to VPS. Please check your SSH access."
    exit 1
fi

echo "✅ VPS connection verified"

# Create monitoring directory on VPS and copy files
echo "📁 Setting up monitoring directory on VPS..."
ssh ${VPS_USER}@${VPS_IP} "mkdir -p /opt/dexbrain-monitoring"

# Copy monitoring configuration files
echo "📋 Copying monitoring configuration to VPS..."
scp -r backend/monitoring/* ${VPS_USER}@${VPS_IP}:/opt/dexbrain-monitoring/

# Create environment file with your email
echo "⚙️ Creating environment configuration..."
ssh ${VPS_USER}@${VPS_IP} "cat > /opt/dexbrain-monitoring/.env << 'EOF'
SMTP_USER=${EMAIL}
SMTP_PASSWORD=your-gmail-app-password-here
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK
DATABASE_URL=postgresql://postgres:password@localhost:5432/dexter_db
DEXBRAIN_API_URL=http://localhost:8080
EOF"

# Update firewall rules for monitoring ports
echo "🔒 Configuring firewall for monitoring services..."
ssh ${VPS_USER}@${VPS_IP} "
# Allow monitoring ports
ufw allow 3001/tcp comment 'Grafana'
ufw allow 9090/tcp comment 'Prometheus' 
ufw allow 9093/tcp comment 'AlertManager'
ufw allow 3002/tcp comment 'Uptime Kuma'
ufw allow 9100/tcp comment 'Node Exporter'
ufw allow 8080/tcp comment 'cAdvisor'
ufw allow 8081/tcp comment 'DexBrain Metrics'

# Reload firewall
ufw reload
ufw status
"

# Install Docker and Docker Compose if not present
echo "🐳 Ensuring Docker is installed..."
ssh ${VPS_USER}@${VPS_IP} "
if ! command -v docker &> /dev/null; then
    echo 'Installing Docker...'
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    systemctl enable docker
    systemctl start docker
fi

if ! command -v docker-compose &> /dev/null; then
    echo 'Installing Docker Compose...'
    curl -L 'https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)' -o /usr/local/bin/docker-compose
    chmod +x /usr/local/bin/docker-compose
fi

docker --version
docker-compose --version
"

# Update AlertManager configuration with your email
echo "📧 Configuring AlertManager with your email..."
ssh ${VPS_USER}@${VPS_IP} "
cd /opt/dexbrain-monitoring
sed -i 's/admin@dexteragent.com/${EMAIL}/g' alertmanager.yml
sed -i 's/alerts@dexteragent.com/${EMAIL}/g' alertmanager.yml
"

# Deploy monitoring stack
echo "🚀 Deploying monitoring stack..."
ssh ${VPS_USER}@${VPS_IP} "
cd /opt/dexbrain-monitoring

# Pull all images first
docker-compose -f docker-compose.monitoring.yml pull

# Create required directories
mkdir -p grafana/provisioning/{dashboards,datasources}
mkdir -p grafana/dashboards
chown -R 472:472 grafana/

# Start monitoring services
docker-compose -f docker-compose.monitoring.yml up -d

# Wait for services to start
echo 'Waiting for services to initialize...'
sleep 30

# Check service status
docker-compose -f docker-compose.monitoring.yml ps
"

# Deploy custom metrics exporter
echo "📊 Setting up custom DexBrain metrics exporter..."
ssh ${VPS_USER}@${VPS_IP} "
cd /opt/dexbrain-monitoring

# Install Python dependencies for metrics exporter
pip3 install prometheus_client psycopg2-binary aiohttp

# Create systemd service for metrics exporter
cat > /etc/systemd/system/dexbrain-metrics.service << 'EOF'
[Unit]
Description=DexBrain Metrics Exporter
After=network.target postgresql.service

[Service]
Type=simple
User=root
WorkingDirectory=/opt/dexbrain-monitoring
Environment=DATABASE_URL=postgresql://postgres:password@localhost:5432/dexter_db
Environment=DEXBRAIN_API_URL=http://localhost:8080
ExecStart=/usr/bin/python3 dexbrain-metrics.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start metrics service
systemctl daemon-reload
systemctl enable dexbrain-metrics.service
systemctl start dexbrain-metrics.service
systemctl status dexbrain-metrics.service --no-pager
"

# Test connectivity and health
echo "🧪 Testing monitoring services..."
ssh ${VPS_USER}@${VPS_IP} "
echo 'Testing service endpoints...'

# Test Prometheus
if curl -s http://localhost:9090/-/healthy > /dev/null; then
    echo '✅ Prometheus is healthy'
else
    echo '❌ Prometheus is not responding'
fi

# Test Grafana
sleep 10  # Give Grafana more time to start
if curl -s http://localhost:3001/api/health > /dev/null; then
    echo '✅ Grafana is healthy'
else
    echo '❌ Grafana is not responding'
fi

# Test AlertManager
if curl -s http://localhost:9093/-/healthy > /dev/null; then
    echo '✅ AlertManager is healthy'
else
    echo '❌ AlertManager is not responding'
fi

# Test custom metrics
if curl -s http://localhost:8081/metrics > /dev/null; then
    echo '✅ DexBrain metrics exporter is healthy'
else
    echo '❌ DexBrain metrics exporter is not responding'
fi

echo ''
echo '🎉 Monitoring deployment complete!'
echo ''
echo '📊 Access URLs (use SSH tunnel or configure nginx for external access):'
echo '  • Grafana:      http://${VPS_IP}:3001 (admin/DexBrain2024!)'
echo '  • Prometheus:   http://${VPS_IP}:9090'
echo '  • AlertManager: http://${VPS_IP}:9093'
echo '  • Uptime Kuma:  http://${VPS_IP}:3002'
echo ''
echo '🔧 Next steps:'
echo '  1. Set up Gmail App Password and update .env file'
echo '  2. Configure Slack webhook if desired'
echo '  3. Access Grafana and import/create dashboards'
echo '  4. Test alert notifications'
echo ''
echo '📖 Documentation: See MONITORING.md for detailed configuration'
"

echo ""
echo "🎯 IMPORTANT: To complete email alerts setup:"
echo "1. Go to Google Account settings"
echo "2. Enable 2-Factor Authentication"
echo "3. Generate an App Password for Gmail"
echo "4. SSH to VPS and update /opt/dexbrain-monitoring/.env with the app password"
echo "5. Restart AlertManager: docker-compose -f docker-compose.monitoring.yml restart alertmanager"
echo ""
echo "💡 To access Grafana from outside the VPS, set up an SSH tunnel:"
echo "ssh -L 3001:localhost:3001 root@${VPS_IP}"
echo "Then access: http://localhost:3001"