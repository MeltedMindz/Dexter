#!/bin/bash

# Deploy Dexter Log API Server to VPS
# Usage: ./deploy-log-api.sh [VPS_IP]

set -e

VPS_IP=${1:-"5.78.71.231"}
VPS_USER="root"

echo "🚀 Deploying Dexter Log API Server to $VPS_IP..."

# Create temporary deployment directory
TEMP_DIR=$(mktemp -d)
cp log-api-server.js "$TEMP_DIR/"
cp dexter-log-api.service "$TEMP_DIR/"

echo "📦 Files prepared for deployment"

# Deploy to VPS
echo "📡 Connecting to VPS and deploying..."

ssh -o StrictHostKeyChecking=no $VPS_USER@$VPS_IP << 'ENDSSH'
set -e

echo "🛠️  Setting up directories..."
sudo mkdir -p /opt/dexter
sudo mkdir -p /var/log/dexter
sudo chown -R root:root /opt/dexter
sudo chown -R root:root /var/log/dexter

echo "📝 Creating test log files if they don't exist..."
if [ ! -f /var/log/dexter/dexter.log ]; then
    echo "[$(date -Iseconds)] INFO [ConservativeAgent] System initialized | Monitoring ETH/USDC pool" > /var/log/dexter/dexter.log
fi

if [ ! -f /var/log/dexter/liquidity.log ]; then
    echo '{"timestamp":"'$(date -Iseconds)'","agent":"Conservative","pool_name":"ETH/USDC","action":"MONITOR","amount_usd":25000,"apr_current":15.2}' > /var/log/dexter/liquidity.log
fi

echo "📋 Log directory setup complete"
ENDSSH

# Copy files to VPS
echo "📋 Copying files to VPS..."
scp "$TEMP_DIR/log-api-server.js" $VPS_USER@$VPS_IP:/opt/dexter/
scp "$TEMP_DIR/dexter-log-api.service" $VPS_USER@$VPS_IP:/tmp/

# Install and start service
echo "🔧 Installing and starting service..."
ssh $VPS_USER@$VPS_IP << 'ENDSSH'
set -e

# Make script executable
chmod +x /opt/dexter/log-api-server.js

# Install systemd service
sudo cp /tmp/dexter-log-api.service /etc/systemd/system/
sudo systemctl daemon-reload

# Stop existing services if running
sudo systemctl stop dexter-log-api 2>/dev/null || true

# Enable and start new service
sudo systemctl enable dexter-log-api
sudo systemctl start dexter-log-api

echo "⏱️  Waiting for service to start..."
sleep 3

# Check service status
if sudo systemctl is-active --quiet dexter-log-api; then
    echo "✅ Dexter Log API Service is running!"
    sudo systemctl status dexter-log-api --no-pager -l
else
    echo "❌ Service failed to start"
    sudo journalctl -u dexter-log-api -n 20 --no-pager
    exit 1
fi

ENDSSH

# Test the API
echo "🧪 Testing API endpoints..."
sleep 2

echo "Testing health endpoint..."
if curl -s "http://$VPS_IP:3004/health" | grep -q "healthy"; then
    echo "✅ Health check passed"
else
    echo "❌ Health check failed"
fi

echo "Testing logs endpoint..."
if curl -s "http://$VPS_IP:3004/api/recent-logs" | grep -q "logs"; then
    echo "✅ Logs endpoint working"
else
    echo "❌ Logs endpoint failed"
fi

# Cleanup
rm -rf "$TEMP_DIR"

echo "🎉 Deployment complete!"
echo ""
echo "📊 Service URLs:"
echo "  • Health Check: http://$VPS_IP:3004/health"
echo "  • Recent Logs:  http://$VPS_IP:3004/api/recent-logs"
echo "  • Refresh Logs: http://$VPS_IP:3004/api/refresh-logs (POST)"
echo ""
echo "🔧 Service Management:"
echo "  • Status:  sudo systemctl status dexter-log-api"
echo "  • Logs:    sudo journalctl -u dexter-log-api -f"
echo "  • Restart: sudo systemctl restart dexter-log-api"
echo ""
echo "📁 Log Files:"
echo "  • Dexter:    /var/log/dexter/dexter.log"
echo "  • Liquidity: /var/log/dexter/liquidity.log"