#!/bin/bash

# Hetzner Server Information Script
# Usage: ./hetzner-server-info.sh

HETZNER_API_TOKEN="WSR7DB4CThdJEWKRL8LIZpHPTWFaBDVFbka16XxmR28OO3WBW5TfPCBW1Egla80L"

echo "🔍 Fetching your Hetzner server information..."

# Check if jq is installed
if ! command -v jq &> /dev/null; then
    echo "Installing jq for JSON parsing..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        brew install jq
    else
        sudo apt-get update && sudo apt-get install -y jq
    fi
fi

# Get all servers
echo "📊 Your Hetzner Cloud Servers:"
echo "================================"

SERVERS=$(curl -s -H "Authorization: Bearer $HETZNER_API_TOKEN" \
    "https://api.hetzner.cloud/v1/servers")

# Check if API call was successful
if [[ $(echo "$SERVERS" | jq -r '.error // empty') ]]; then
    echo "❌ Error accessing Hetzner API:"
    echo "$SERVERS" | jq -r '.error.message'
    exit 1
fi

# Display server information
echo "$SERVERS" | jq -r '.servers[] | 
    "Name: \(.name)",
    "IP: \(.public_net.ipv4.ip)",
    "Status: \(.status)",
    "Type: \(.server_type.name)",
    "Location: \(.datacenter.location.name)",
    "RAM: \(.server_type.memory)GB",
    "Disk: \(.server_type.disk)GB", 
    "CPU: \(.server_type.cores) cores",
    "Created: \(.created)",
    "---"'

# Get specific server info for our target IP
TARGET_IP="157.90.230.148"
TARGET_SERVER=$(echo "$SERVERS" | jq -r --arg ip "$TARGET_IP" '.servers[] | select(.public_net.ipv4.ip == $ip)')

if [[ -n "$TARGET_SERVER" ]]; then
    echo ""
    echo "🎯 Target Server ($TARGET_IP) Details:"
    echo "======================================"
    echo "Name: $(echo "$TARGET_SERVER" | jq -r '.name')"
    echo "Status: $(echo "$TARGET_SERVER" | jq -r '.status')"
    echo "Type: $(echo "$TARGET_SERVER" | jq -r '.server_type.name')"
    echo "RAM: $(echo "$TARGET_SERVER" | jq -r '.server_type.memory')GB"
    echo "Disk: $(echo "$TARGET_SERVER" | jq -r '.server_type.disk')GB"
    echo "CPU: $(echo "$TARGET_SERVER" | jq -r '.server_type.cores') cores"
    echo "Location: $(echo "$TARGET_SERVER" | jq -r '.datacenter.location.name')"
    echo "Created: $(echo "$TARGET_SERVER" | jq -r '.created')"
    echo ""
    
    # Check if server is ready for deployment
    STATUS=$(echo "$TARGET_SERVER" | jq -r '.status')
    if [[ "$STATUS" == "running" ]]; then
        echo "✅ Server is running and ready for DexBrain deployment!"
        echo ""
        echo "🚀 To deploy DexBrain API, run:"
        echo "   chmod +x scripts/deploy-hetzner.sh"
        echo "   ./scripts/deploy-hetzner.sh"
    else
        echo "⚠️ Server status is '$STATUS' - it should be 'running' for deployment"
    fi
else
    echo ""
    echo "❌ Server with IP $TARGET_IP not found in your account"
    echo "Please check the IP address or create a new server"
fi

echo ""
echo "📚 Useful Hetzner API endpoints:"
echo "================================"
echo "List servers: curl -H 'Authorization: Bearer \$TOKEN' https://api.hetzner.cloud/v1/servers"
echo "Server details: curl -H 'Authorization: Bearer \$TOKEN' https://api.hetzner.cloud/v1/servers/\$SERVER_ID"
echo "API Documentation: https://docs.hetzner.com/cloud/api/"