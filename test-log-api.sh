#!/bin/bash

# Test Dexter Log API Server
# Usage: ./test-log-api.sh [VPS_IP]

VPS_IP=${1:-"5.78.71.231"}

echo "🧪 Testing Dexter Log API on $VPS_IP..."

echo ""
echo "1. Testing health endpoint..."
HEALTH_RESPONSE=$(curl -s -w "%{http_code}" "http://$VPS_IP:3004/health")
HTTP_CODE=${HEALTH_RESPONSE: -3}
HEALTH_BODY=${HEALTH_RESPONSE%???}

if [ "$HTTP_CODE" = "200" ]; then
    echo "✅ Health check passed"
    echo "   Response: $HEALTH_BODY"
else
    echo "❌ Health check failed (HTTP $HTTP_CODE)"
    echo "   Response: $HEALTH_BODY"
fi

echo ""
echo "2. Testing logs endpoint..."
LOGS_RESPONSE=$(curl -s -w "%{http_code}" "http://$VPS_IP:3004/api/recent-logs")
HTTP_CODE=${LOGS_RESPONSE: -3}
LOGS_BODY=${LOGS_RESPONSE%???}

if [ "$HTTP_CODE" = "200" ]; then
    echo "✅ Logs endpoint working"
    echo "   Log count: $(echo "$LOGS_BODY" | grep -o '"count":[0-9]*' | cut -d: -f2)"
else
    echo "❌ Logs endpoint failed (HTTP $HTTP_CODE)"
    echo "   Response: $LOGS_BODY"
fi

echo ""
echo "3. Testing service status..."
ssh ubuntu@$VPS_IP "sudo systemctl is-active dexter-log-api" > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✅ Service is running"
    ssh ubuntu@$VPS_IP "sudo systemctl status dexter-log-api --no-pager -l | head -10"
else
    echo "❌ Service is not running"
    echo "Service status:"
    ssh ubuntu@$VPS_IP "sudo systemctl status dexter-log-api --no-pager -l | head -10"
fi

echo ""
echo "4. Checking log files..."
LOG_COUNT=$(ssh ubuntu@$VPS_IP "find /var/log/dexter -name '*.log' -type f | wc -l" 2>/dev/null || echo "0")
if [ "$LOG_COUNT" -gt 0 ]; then
    echo "✅ Found $LOG_COUNT log file(s)"
    ssh ubuntu@$VPS_IP "ls -la /var/log/dexter/"
else
    echo "⚠️  No log files found in /var/log/dexter/"
fi

echo ""
echo "🎯 Quick test summary:"
echo "   Health API: $([ "$HTTP_CODE" = "200" ] && echo "✅ Working" || echo "❌ Failed")"
echo "   Service:    $(ssh ubuntu@$VPS_IP "sudo systemctl is-active dexter-log-api" 2>/dev/null || echo "inactive")"
echo "   Log files:  $LOG_COUNT found"