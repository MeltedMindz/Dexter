#!/bin/bash
# DexBrain Monitoring SSH Tunnel Script
# This creates secure tunnels to access monitoring services

echo "🔒 Creating secure SSH tunnels to DexBrain monitoring..."
echo ""
echo "After running this script, access services at:"
echo "  📊 Grafana:      http://localhost:3001 (admin/DexBrain2024!)"
echo "  📈 Prometheus:   http://localhost:9090"
echo "  🚨 AlertManager: http://localhost:9093"
echo "  📍 Uptime Kuma:  http://localhost:3002"
echo ""
echo "Press Ctrl+C to close the tunnels when done."
echo ""

ssh -L 3001:localhost:3001 \
    -L 9090:localhost:9090 \
    -L 9093:localhost:9093 \
    -L 3002:localhost:3002 \
    -L 8080:localhost:8080 \
    -L 8081:localhost:8081 \
    root@157.90.230.148 \
    -o ServerAliveInterval=60 \
    -o ServerAliveCountMax=3 \
    -N