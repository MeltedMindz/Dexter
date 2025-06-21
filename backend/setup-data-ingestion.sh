#!/bin/bash
# Setup Historical Data Ingestion for DexBrain

set -e

VPS_IP="5.78.71.231"
VPS_USER="root"

echo "=== Setting up Historical Data Ingestion for DexBrain ==="

# Copy data ingestion files to VPS
echo "1. Copying data ingestion files to VPS..."
scp -r ./data_sources/ ${VPS_USER}@${VPS_IP}:/opt/dexbrain/
echo "✓ Files copied"

# Install additional dependencies
echo "2. Installing additional Python dependencies..."
ssh ${VPS_USER}@${VPS_IP} "
cd /opt/dexbrain && source venv/bin/activate
pip install aiohttp web3 requests python-dateutil
pip install --upgrade aiohttp asyncio
"
echo "✓ Dependencies installed"

# Setup data directories
echo "3. Setting up data directories..."
ssh ${VPS_USER}@${VPS_IP} "
mkdir -p /opt/dexbrain/data/historical
mkdir -p /opt/dexbrain/data/backups
chmod 755 /opt/dexbrain/data_sources/*.py
"
echo "✓ Data directories created"

# Create environment variables for API access
echo "4. Configuring API access..."
ssh ${VPS_USER}@${VPS_IP} "
# Add data ingestion environment variables
cat >> /opt/dexbrain/.env << 'EOF'

# Data Sources Configuration
GRAPH_API_KEY=
BASE_RPC_URL=https://base-mainnet.g.alchemy.com/v2/ory0F2cLFNIXsovAmrtJj
ALCHEMY_API_KEY=ory0F2cLFNIXsovAmrtJj

# Data Quality Filters
MIN_POSITION_VALUE_USD=100
MAX_DAYS_BACK=30
DEFAULT_BATCH_SIZE=50

# Ingestion Schedule
AUTO_INGESTION_ENABLED=true
INGESTION_INTERVAL_HOURS=6
EOF
"
echo "✓ Environment configured"

# Create ingestion service
echo "5. Creating data ingestion service..."
ssh ${VPS_USER}@${VPS_IP} "
cat > /etc/systemd/system/dexbrain-ingestion.service << 'EOF'
[Unit]
Description=DexBrain Historical Data Ingestion Service
After=network.target dexbrain.service
Requires=dexbrain.service

[Service]
Type=oneshot
User=root
WorkingDirectory=/opt/dexbrain
Environment=\"PYTHONPATH=/opt/dexbrain\"
ExecStart=/opt/dexbrain/venv/bin/python /opt/dexbrain/data_sources/dexbrain_data_ingestion.py --days-back 7 --limit 500 --min-value 100
StandardOutput=append:/opt/dexter-ai/dexbrain.log
StandardError=append:/opt/dexter-ai/dexbrain.log

[Install]
WantedBy=multi-user.target
EOF
"
echo "✓ Ingestion service created"

# Create timer for automatic ingestion
echo "6. Creating automatic ingestion timer..."
ssh ${VPS_USER}@${VPS_IP} "
cat > /etc/systemd/system/dexbrain-ingestion.timer << 'EOF'
[Unit]
Description=Run DexBrain data ingestion every 6 hours
Requires=dexbrain-ingestion.service

[Timer]
OnCalendar=*-*-* 00,06,12,18:00:00
Persistent=true
RandomizedDelaySec=300

[Install]
WantedBy=timers.target
EOF
"
echo "✓ Ingestion timer created"

# Enable services
echo "7. Enabling services..."
ssh ${VPS_USER}@${VPS_IP} "
systemctl daemon-reload
systemctl enable dexbrain-ingestion.service
systemctl enable dexbrain-ingestion.timer
systemctl start dexbrain-ingestion.timer
"
echo "✓ Services enabled"

# Create manual ingestion script
echo "8. Creating manual ingestion script..."
ssh ${VPS_USER}@${VPS_IP} "
cat > /opt/dexbrain/run_ingestion.sh << 'EOF'
#!/bin/bash
# Manual data ingestion script

cd /opt/dexbrain
source venv/bin/activate

echo \"=== DexBrain Historical Data Ingestion ===\" 
echo \"Started at: \$(date)\"
echo \"\"

# Run with default parameters
python data_sources/dexbrain_data_ingestion.py --days-back 7 --limit 500 --min-value 100

echo \"\"
echo \"Completed at: \$(date)\"
echo \"\"

# Show status
echo \"=== Current Status ===\" 
python data_sources/dexbrain_data_ingestion.py --status-only
EOF

chmod +x /opt/dexbrain/run_ingestion.sh
"
echo "✓ Manual ingestion script created"

# Test the setup
echo "9. Testing data ingestion setup..."
ssh ${VPS_USER}@${VPS_IP} "
cd /opt/dexbrain && source venv/bin/activate
echo 'Testing imports...'
python -c 'from data_sources.historical_position_fetcher import HistoricalPositionFetcher; print(\"✓ Historical fetcher import OK\")'
python -c 'from data_sources.dexbrain_data_ingestion import DexBrainDataIngestion; print(\"✓ Data ingestion import OK\")'
echo 'Testing DexBrain connectivity...'
curl -s http://localhost:8080/health | python3 -m json.tool
"
echo "✓ Setup testing complete"

# Run initial ingestion (small batch)
echo "10. Running initial data ingestion (test batch)..."
ssh ${VPS_USER}@${VPS_IP} "
cd /opt/dexbrain && source venv/bin/activate
timeout 60 python data_sources/dexbrain_data_ingestion.py --days-back 3 --limit 10 --min-value 50 || echo 'Initial ingestion timed out (normal for first run)'
"
echo "✓ Initial ingestion attempted"

echo ""
echo "=== Setup Complete! ==="
echo ""
echo "Data ingestion system is now configured:"
echo "  • Automatic ingestion every 6 hours"
echo "  • Manual ingestion: ssh ${VPS_USER}@${VPS_IP} '/opt/dexbrain/run_ingestion.sh'"
echo "  • Service status: ssh ${VPS_USER}@${VPS_IP} 'systemctl status dexbrain-ingestion.timer'"
echo "  • View logs: ssh ${VPS_USER}@${VPS_IP} 'tail -f /opt/dexter-ai/dexbrain.log | grep -E \"ingestion|position\"'"
echo ""
echo "Next steps:"
echo "  1. Set GRAPH_API_KEY in /opt/dexbrain/.env for better rate limits"
echo "  2. Monitor the first few automatic runs"
echo "  3. Adjust ingestion parameters if needed"
echo ""
echo "API endpoints:"
echo "  • DexBrain health: http://${VPS_IP}:8080/health"
echo "  • Register agent: http://${VPS_IP}:8080/api/register"
echo "  • Get intelligence: http://${VPS_IP}:8080/api/intelligence"