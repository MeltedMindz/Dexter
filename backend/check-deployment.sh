#!/bin/bash
# Check deployment status on VPS

VPS_IP="5.78.71.231"

echo "=== Dexter AI Deployment Status ==="
echo "VPS: $VPS_IP"
echo ""

echo "1. DexBrain API Status:"
ssh root@$VPS_IP "systemctl status dexbrain --no-pager | head -10"
echo ""

echo "2. DexBrain Health Check:"
ssh root@$VPS_IP "curl -s http://localhost:8080/health | python3 -m json.tool"
echo ""

echo "3. Running Services:"
ssh root@$VPS_IP "ps aux | grep -E 'dexter|dexbrain|node' | grep -v grep | wc -l"
echo ""

echo "4. Log Files:"
ssh root@$VPS_IP "ls -lh /opt/dexter-ai/*.log | tail -5"
echo ""

echo "5. Port Status:"
ssh root@$VPS_IP "netstat -tlnp | grep -E '3000|3003|8080' || ss -tlnp | grep -E '3000|3003|8080'"
echo ""

echo "=== Integration Points ==="
echo "- DexBrain API: http://$VPS_IP:8080"
echo "- Log Stream: http://$VPS_IP:3003"
echo "- Dexter Trading: Running at /opt/dexter-trading/"
echo ""

echo "To register an agent with DexBrain:"
echo "curl -X POST http://$VPS_IP:8080/api/register -H 'Content-Type: application/json' -d '{\"agent_id\": \"test-agent\"}'"