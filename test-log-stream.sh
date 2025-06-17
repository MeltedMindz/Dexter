#!/bin/bash

# Test script for Dexter AI Log Stream Server
# Usage: ./test-log-stream.sh [VPS_IP]

VPS_IP=${1:-"5.78.71.231"}
VPS_USER="root"

echo "Testing Dexter AI Log Stream Server on $VPS_IP..."

echo "1. Checking service status..."
ssh $VPS_USER@$VPS_IP "systemctl status dexter-log-stream --no-pager -l"

echo ""
echo "2. Testing health endpoint..."
curl -s "http://$VPS_IP:3003/health" | jq . 2>/dev/null || curl -s "http://$VPS_IP:3003/health"

echo ""
echo "3. Creating test log files..."
ssh $VPS_USER@$VPS_IP "
    mkdir -p /opt/dexter-ai
    echo '$(date) INFO: Dexter AI system started' >> /opt/dexter-ai/dexter.log
    echo '$(date) INFO: Liquidity management initialized' >> /opt/dexter-ai/liquidity.log
    echo '$(date) DEBUG: Pool analysis started' >> /opt/dexter-ai/dexbrain.log
    echo '$(date) WARN: High volatility detected in USDC/ETH pool' >> /opt/dexter-ai/dexter.log
    echo '$(date) ERROR: Failed to connect to Base RPC endpoint' >> /opt/dexter-ai/liquidity.log
"

echo ""
echo "4. Checking log files exist..."
ssh $VPS_USER@$VPS_IP "ls -la /opt/dexter-ai/*.log"

echo ""
echo "5. Testing SSE stream (will run for 10 seconds)..."
echo "Connect to: http://$VPS_IP:3003/logs"
echo "Testing with curl (timeout after 10 seconds)..."

timeout 10 curl -N -H "Accept: text/event-stream" "http://$VPS_IP:3003/logs" 2>/dev/null || echo "Stream test completed"

echo ""
echo "6. Adding more test log entries..."
ssh $VPS_USER@$VPS_IP "
    echo '$(date) INFO: Test log entry from test script' >> /opt/dexter-ai/dexter.log
    echo '$(date) CRITICAL: Test critical error for monitoring' >> /opt/dexter-ai/liquidity.log
"

echo ""
echo "Test complete!"
echo ""
echo "To monitor logs in real-time:"
echo "  curl -N -H 'Accept: text/event-stream' 'http://$VPS_IP:3003/logs'"
echo ""
echo "Or open in browser:"
echo "  http://$VPS_IP:3003/logs"