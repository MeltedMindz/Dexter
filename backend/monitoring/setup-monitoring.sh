#!/bin/bash

# DexBrain Production Monitoring Setup Script
# Run this on the VPS (157.90.230.148) to set up comprehensive monitoring

set -e

echo "🔧 Setting up DexBrain Production Monitoring..."

# Create monitoring directory
mkdir -p /opt/dexbrain-monitoring
cd /opt/dexbrain-monitoring

# Copy configuration files
echo "📋 Copying monitoring configuration files..."

# Create environment file
cat > .env << EOF
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK
EOF

echo "⚠️  Please update .env file with your actual credentials"

# Create directories
mkdir -p grafana/provisioning/{dashboards,datasources}
mkdir -p grafana/dashboards

# Set permissions
chown -R 472:472 grafana/  # Grafana user
chmod 755 grafana/

# Pull Docker images
echo "🐳 Pulling Docker images..."
docker-compose -f docker-compose.monitoring.yml pull

# Start monitoring stack
echo "🚀 Starting monitoring stack..."
docker-compose -f docker-compose.monitoring.yml up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 30

# Check service status
echo "🔍 Checking service status..."
docker-compose -f docker-compose.monitoring.yml ps

# Test connectivity
echo "🧪 Testing connectivity..."

# Test Prometheus
if curl -s http://localhost:9090/-/healthy > /dev/null; then
    echo "✅ Prometheus is healthy"
else
    echo "❌ Prometheus is not responding"
fi

# Test Grafana
if curl -s http://localhost:3001/api/health > /dev/null; then
    echo "✅ Grafana is healthy"
else
    echo "❌ Grafana is not responding"
fi

# Test AlertManager
if curl -s http://localhost:9093/-/healthy > /dev/null; then
    echo "✅ AlertManager is healthy"
else
    echo "❌ AlertManager is not responding"
fi

# Create initial Grafana dashboard
echo "📊 Setting up initial Grafana dashboard..."

# Wait for Grafana to be fully ready
sleep 10

# Import DexBrain dashboard
curl -X POST \
  http://admin:DexBrain2024!@localhost:3001/api/dashboards/db \
  -H 'Content-Type: application/json' \
  -d @dexbrain-dashboard.json || echo "⚠️  Dashboard import failed - will need manual setup"

echo "🎉 Monitoring setup complete!"
echo ""
echo "📊 Access URLs:"
echo "  • Grafana:      http://157.90.230.148:3001 (admin/DexBrain2024!)"
echo "  • Prometheus:   http://157.90.230.148:9090"
echo "  • AlertManager: http://157.90.230.148:9093"
echo "  • Uptime Kuma:  http://157.90.230.148:3002"
echo ""
echo "🔧 Next steps:"
echo "  1. Update .env file with your email/Slack credentials"
echo "  2. Configure firewall rules for monitoring ports"
echo "  3. Set up SSL certificates for external access"
echo "  4. Configure backup for monitoring data"
echo ""
echo "📖 Documentation: See MONITORING.md for detailed configuration"