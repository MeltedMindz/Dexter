#!/bin/bash
echo "🔧 Grafana Dashboard Quick Fix"
echo "=============================="

# Test Prometheus connectivity
echo "1. Testing Prometheus..."
if curl -s http://localhost:9090/api/v1/query?query=up | jq -e '.status == "success"' > /dev/null; then
    echo "  ✅ Prometheus API working"
else
    echo "  ❌ Prometheus API failed"
    exit 1
fi

# Test metrics availability
echo "2. Testing Dexter metrics..."
METRIC_COUNT=$(curl -s 'http://localhost:9090/api/v1/label/__name__/values' | jq -r '.data[]' | grep dexter | wc -l)
echo "  📊 Found $METRIC_COUNT Dexter metrics"

# Test specific metrics that should show in dashboard
echo "3. Testing key dashboard metrics..."

# Service Status
SERVICE_COUNT=$(curl -s 'http://localhost:9090/api/v1/query?query=dexter_service_status' | jq '.data.result | length')
echo "  🔧 Service status metrics: $SERVICE_COUNT"

# CPU
CPU_VALUE=$(curl -s 'http://localhost:9090/api/v1/query?query=dexter_system_cpu_percent' | jq -r '.data.result[0].value[1]')
echo "  💻 CPU usage: $CPU_VALUE%"

# Memory
MEMORY_VALUE=$(curl -s 'http://localhost:9090/api/v1/query?query=dexter_system_memory_usage_bytes' | jq -r '.data.result[0].value[1]')
echo "  🧠 Memory usage: $(echo "scale=1; $MEMORY_VALUE / 1024 / 1024 / 1024" | bc)GB"

# Data Quality (might be 0 initially)
QUALITY_VALUE=$(curl -s 'http://localhost:9090/api/v1/query?query=dexter_data_quality_score' | jq -r '.data.result[0].value[1] // "0"')
echo "  📊 Data quality: $QUALITY_VALUE"

echo ""
echo "4. Dashboard Access Info:"
echo "  🌐 URL: http://$(curl -s ifconfig.me):3000/d/dexter-ai-main"
echo "  👤 Username: admin"
echo "  🔑 Password: dexteradmin123"

echo ""
echo "5. Troubleshooting:"
if [ "$SERVICE_COUNT" -gt 0 ] && [ "$CPU_VALUE" != "null" ] && [ "$MEMORY_VALUE" != "null" ]; then
    echo "  ✅ All core metrics available - Dashboard should work!"
    echo "  💡 If still showing 'No data', try refreshing the page"
    echo "  💡 Check time range (top right) - try 'Last 5 minutes'"
else
    echo "  ⚠️  Some metrics missing - Check service status:"
    echo "     systemctl status dexter-metrics-exporter"
fi