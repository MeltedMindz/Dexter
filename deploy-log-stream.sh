#!/bin/bash

# Deploy Dexter AI Log Stream Server to VPS
# Usage: ./deploy-log-stream.sh [VPS_IP]

VPS_IP=${1:-"5.78.71.231"}
VPS_USER="root"

echo "Deploying Dexter AI Log Stream Server to $VPS_IP..."

# Check if Node.js server file exists
if [ ! -f "log-stream-server.js" ]; then
    echo "Error: log-stream-server.js not found in current directory"
    exit 1
fi

# Check if systemd service file exists
if [ ! -f "dexter-log-stream.service" ]; then
    echo "Error: dexter-log-stream.service not found in current directory"
    exit 1
fi

echo "1. Creating /opt/dexter-ai directory on VPS..."
ssh $VPS_USER@$VPS_IP "mkdir -p /opt/dexter-ai && chown root:root /opt/dexter-ai"

echo "2. Copying log stream server to VPS..."
scp log-stream-server.js $VPS_USER@$VPS_IP:/opt/dexter-ai/

echo "3. Making log stream server executable..."
ssh $VPS_USER@$VPS_IP "chmod +x /opt/dexter-ai/log-stream-server.js"

echo "4. Installing systemd service..."
scp dexter-log-stream.service $VPS_USER@$VPS_IP:/etc/systemd/system/

echo "5. Enabling and starting the service..."
ssh $VPS_USER@$VPS_IP "
    systemctl daemon-reload
    systemctl enable dexter-log-stream
    systemctl start dexter-log-stream
"

echo "6. Checking service status..."
ssh $VPS_USER@$VPS_IP "systemctl status dexter-log-stream --no-pager"

echo "7. Testing the log stream endpoint..."
echo "Testing connection to http://$VPS_IP:3003/health"
curl -s "http://$VPS_IP:3003/health" | jq . || echo "Health check failed or jq not available"

echo ""
echo "Deployment complete!"
echo ""
echo "Service commands:"
echo "  Start:   systemctl start dexter-log-stream"
echo "  Stop:    systemctl stop dexter-log-stream"
echo "  Restart: systemctl restart dexter-log-stream"
echo "  Status:  systemctl status dexter-log-stream"
echo "  Logs:    journalctl -u dexter-log-stream -f"
echo ""
echo "Endpoints:"
echo "  SSE Stream: http://$VPS_IP:3003/logs"
echo "  Health:     http://$VPS_IP:3003/health"
echo ""
echo "To create test log files for monitoring:"
echo "  ssh $VPS_USER@$VPS_IP"
echo "  echo 'INFO: Test log entry' >> /opt/dexter-ai/dexter.log"
echo "  echo 'ERROR: Test error entry' >> /opt/dexter-ai/liquidity.log"