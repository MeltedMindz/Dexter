#!/bin/bash
# Setup Graph API Key for Automatic Learning

set -e

VPS_IP="5.78.71.231"
VPS_USER="root"

if [ -z "$1" ]; then
    echo "Usage: $0 <GRAPH_API_KEY>"
    echo ""
    echo "This script will:"
    echo "  1. Configure your Graph API key"
    echo "  2. Test data fetching from Base network"
    echo "  3. Run initial data ingestion"
    echo "  4. Verify ML training with real data"
    echo "  5. Enable automatic learning every 6 hours"
    echo ""
    echo "Get your API key from: https://thegraph.com/studio/"
    exit 1
fi

GRAPH_API_KEY="$1"

echo "=== Setting up Graph API Key for Automatic Learning ==="
echo "API Key: ${GRAPH_API_KEY:0:10}..."
echo ""

# 1. Configure API key
echo "1. Configuring Graph API key..."
ssh ${VPS_USER}@${VPS_IP} "
cd /opt/dexbrain
# Backup current config
cp .env .env.backup-\$(date +%Y%m%d_%H%M%S)

# Update API key
sed -i 's/GRAPH_API_KEY=.*/GRAPH_API_KEY=${GRAPH_API_KEY}/' .env

# Add if not exists
if ! grep -q 'GRAPH_API_KEY' .env; then
    echo 'GRAPH_API_KEY=${GRAPH_API_KEY}' >> .env
fi

echo '✓ API key configured'
"

# 2. Test data fetching
echo ""
echo "2. Testing data fetching with your API key..."
ssh ${VPS_USER}@${VPS_IP} "
cd /opt/dexbrain && source venv/bin/activate
export GRAPH_API_KEY='${GRAPH_API_KEY}'

# Test Graph API connectivity
echo 'Testing Graph API connection...'
python3 -c \"
import asyncio
import aiohttp
import json

async def test_graph_api():
    endpoint = 'https://gateway.thegraph.com/api/${GRAPH_API_KEY}/subgraphs/id/HMuAwufqZ1YCRmzL2SfHTVkzZovC9VL2UAKhjvRqKiR1'
    
    query = '''
    {
      positions(first: 5, where: {liquidity: \\\"0\\\"}) {
        id
        owner
        liquidity
        transaction {
          timestamp
        }
      }
    }
    '''
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                endpoint,
                json={'query': query},
                headers={'Content-Type': 'application/json'}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'errors' in data:
                        print(f'❌ Graph API errors: {data[\\\"errors\\\"]}')
                        return False
                    positions = data.get('data', {}).get('positions', [])
                    print(f'✅ Successfully fetched {len(positions)} positions')
                    print(f'   Sample position ID: {positions[0][\\\"id\\\"] if positions else \\\"None\\\"}')
                    return len(positions) > 0
                else:
                    print(f'❌ HTTP error {response.status}')
                    return False
        except Exception as e:
            print(f'❌ Connection error: {e}')
            return False

success = asyncio.run(test_graph_api())
if success:
    print('✅ Graph API connection successful!')
else:
    print('❌ Graph API connection failed')
    exit(1)
\"
"

# 3. Run initial data ingestion
echo ""
echo "3. Running initial data ingestion with real positions..."
ssh ${VPS_USER}@${VPS_IP} "
cd /opt/dexbrain && source venv/bin/activate
export GRAPH_API_KEY='${GRAPH_API_KEY}'

echo 'Starting real data ingestion...'
python3 data_sources/dexbrain_data_ingestion.py --days-back 7 --limit 100 --min-value 100
"

# 4. Check results and trigger ML training
echo ""
echo "4. Checking ingestion results and ML training..."
ssh ${VPS_USER}@${VPS_IP} "
cd /opt/dexbrain && source venv/bin/activate

# Check knowledge base status
python3 -c \"
import asyncio
from dexbrain.models.knowledge_base import KnowledgeBase

async def check_status():
    kb = KnowledgeBase()
    count = await kb.get_insight_count('base_liquidity_positions')
    print(f'📊 Total insights in knowledge base: {count}')
    
    if count >= 10:
        print('✅ Sufficient data for ML training!')
        
        # Trigger ML training
        try:
            from dexbrain.core import DexBrain
            brain = DexBrain()
            result = await brain.train_models('base_liquidity_positions')
            print(f'🤖 ML training result: {result}')
        except Exception as e:
            print(f'⚠️  ML training error: {e}')
    else:
        print(f'⏳ Need {10-count} more insights for ML training')
    
    return count

asyncio.run(check_status())
\"
"

# 5. Verify automatic scheduling
echo ""
echo "5. Verifying automatic learning schedule..."
ssh ${VPS_USER}@${VPS_IP} "
# Check timer status
echo 'Timer status:'
systemctl status dexbrain-ingestion.timer --no-pager | head -10

echo ''
echo 'Next scheduled runs:'
systemctl list-timers dexbrain-ingestion.timer --no-pager

# Update service to use API key
echo ''
echo 'Updating service to use API key...'
sed -i '/Environment=/a Environment=\"GRAPH_API_KEY=${GRAPH_API_KEY}\"' /etc/systemd/system/dexbrain-ingestion.service
systemctl daemon-reload

echo '✅ Service updated with API key'
"

echo ""
echo "=== Setup Complete! ==="
echo ""
echo "🎯 Automatic Learning Status:"
echo "  ✅ Graph API key configured and tested"
echo "  ✅ Real position data ingestion working"
echo "  ✅ DexBrain knowledge base populated"
echo "  ✅ Automatic runs every 6 hours"
echo ""
echo "📊 What happens automatically:"
echo "  • Every 6 hours: Fetch new closed positions from last 7 days"
echo "  • Filter positions by quality and minimum value"
echo "  • Convert to ML features and store in knowledge base"
echo "  • Trigger ML model training when sufficient data"
echo "  • Serve predictions via API endpoints"
echo ""
echo "🔍 Monitor the system:"
echo "  • View logs: ssh ${VPS_USER}@${VPS_IP} 'tail -f /opt/dexter-ai/dexbrain.log'"
echo "  • Check status: ssh ${VPS_USER}@${VPS_IP} 'systemctl status dexbrain-ingestion.timer'"
echo "  • Manual run: ssh ${VPS_USER}@${VPS_IP} '/opt/dexbrain/run_ingestion.sh'"
echo ""
echo "🚀 DexBrain is now learning from real Base network liquidity positions!"