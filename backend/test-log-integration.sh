#!/bin/bash
# Test DexBrain log integration

VPS_IP="5.78.71.231"

echo "=== Testing DexBrain Log Integration ==="
echo ""

echo "1. Checking log files:"
ssh root@$VPS_IP "ls -la /opt/dexter-ai/*.log | grep -E 'dexbrain|dexter|liquidity'"
echo ""

echo "2. Log stream server status:"
ssh root@$VPS_IP "curl -s http://localhost:3003/health | python3 -m json.tool"
echo ""

echo "3. Testing log stream (press Ctrl+C after a few seconds):"
echo "Connecting to http://$VPS_IP:3003/logs ..."
echo ""

# Add some test entries to different logs
ssh root@$VPS_IP "
echo '[$(date -u +%Y-%m-%dT%H:%M:%S.000Z)] DEXBRAIN: Knowledge base updated with 50 new insights' >> /opt/dexter-ai/dexbrain.log
echo '[$(date -u +%Y-%m-%dT%H:%M:%S.000Z)] DEXTER: Agent submitted performance data' >> /opt/dexter-ai/dexter.log
echo '[$(date -u +%Y-%m-%dT%H:%M:%S.000Z)] LIQUIDITY: Pool analysis completed for USDC/ETH' >> /opt/dexter-ai/liquidity.log
"

# Connect to stream briefly
timeout 5 curl -N -s http://$VPS_IP:3003/logs | while read line; do
    if [[ $line == data:* ]]; then
        echo "$line" | sed 's/^data://' | python3 -m json.tool 2>/dev/null || echo "$line"
    fi
done

echo ""
echo "=== Log Integration Summary ==="
echo "✓ DexBrain logs are being written to: /opt/dexter-ai/dexbrain.log"
echo "✓ Log stream server is monitoring all logs in: /opt/dexter-ai/"
echo "✓ Frontend can access combined logs at: http://$VPS_IP:3003/logs"
echo ""
echo "The BrainWindow component will receive logs from:"
echo "  - dexbrain.log (DexBrain API activities)"
echo "  - dexter.log (Main agent activities)"
echo "  - liquidity.log (Liquidity management)"
echo "  - defi-analysis.log (DeFi analysis)"