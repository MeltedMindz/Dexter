#!/bin/bash
# DexBrain Deployment Script for VPS

set -e

echo "=== DexBrain Deployment Script ==="

# Configuration
VPS_IP="5.78.71.231"
VPS_USER="root"
DEPLOY_DIR="/opt/dexbrain"
SERVICE_NAME="dexbrain"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}Starting DexBrain deployment to VPS...${NC}"

# Create deployment directory on VPS
echo "Creating deployment directory..."
ssh ${VPS_USER}@${VPS_IP} "mkdir -p ${DEPLOY_DIR}"

# Copy backend files to VPS
echo "Copying DexBrain files to VPS..."
rsync -avz --exclude='__pycache__' --exclude='*.pyc' --exclude='.env' \
    ./dexbrain/ ${VPS_USER}@${VPS_IP}:${DEPLOY_DIR}/dexbrain/
rsync -avz requirements.txt ${VPS_USER}@${VPS_IP}:${DEPLOY_DIR}/

# Create environment file for DexBrain
echo "Creating environment configuration..."
cat > .env.dexbrain << 'EOF'
# DexBrain Configuration
ALCHEMY_API_KEY=${ALCHEMY_API_KEY}
BASE_RPC_URL=https://base-mainnet.g.alchemy.com/v2/${ALCHEMY_API_KEY}
BASESCAN_API_KEY=${BASESCAN_API_KEY}

# Database Configuration (using SQLite for simplicity)
DATABASE_URL=sqlite:///opt/dexbrain/dexbrain.db

# Redis Configuration (optional, will use in-memory if not available)
REDIS_URL=redis://localhost:6379

# DexBrain Specific
DEXBRAIN_PORT=8000
KNOWLEDGE_BASE_PATH=/opt/dexbrain/knowledge_base
MODEL_PATH=/opt/dexbrain/models
LOG_LEVEL=INFO
ENVIRONMENT=production

# Integration with existing services
DEXTER_API_URL=http://localhost:3000
LOG_STREAM_URL=http://localhost:3003
EOF

# Copy environment file
scp .env.dexbrain ${VPS_USER}@${VPS_IP}:${DEPLOY_DIR}/.env
rm .env.dexbrain

# Create startup script
echo "Creating startup script..."
cat > dexbrain-start.py << 'EOF'
#!/usr/bin/env python3
import os
import sys
import asyncio
import logging
from pathlib import Path

# Add dexbrain to path
sys.path.insert(0, '/opt/dexbrain')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/opt/dexter-ai/dexbrain.log'),
        logging.StreamHandler()
    ]
)

async def main():
    """Start DexBrain service."""
    try:
        # Import after path setup
        from dexbrain.core import DexBrain
        from dexbrain.config import Config
        
        # Load configuration
        config = Config()
        
        # Initialize DexBrain
        brain = DexBrain(config)
        
        # Start the service
        logging.info("Starting DexBrain service...")
        await brain.start()
        
        # Keep running
        try:
            await asyncio.Event().wait()
        except KeyboardInterrupt:
            logging.info("Shutting down DexBrain...")
            await brain.stop()
            
    except Exception as e:
        logging.error(f"Failed to start DexBrain: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
EOF

# Copy startup script
scp dexbrain-start.py ${VPS_USER}@${VPS_IP}:${DEPLOY_DIR}/
rm dexbrain-start.py

# Create systemd service file
echo "Creating systemd service..."
cat > dexbrain.service << 'EOF'
[Unit]
Description=DexBrain AI Knowledge Hub
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/dexbrain
Environment="PYTHONPATH=/opt/dexbrain"
ExecStart=/usr/bin/python3 /opt/dexbrain/dexbrain-start.py
Restart=always
RestartSec=10
StandardOutput=append:/opt/dexter-ai/dexbrain.log
StandardError=append:/opt/dexter-ai/dexbrain.log

[Install]
WantedBy=multi-user.target
EOF

# Deploy service file
scp dexbrain.service ${VPS_USER}@${VPS_IP}:/tmp/
rm dexbrain.service

# Install dependencies and start service on VPS
echo "Installing dependencies and starting service on VPS..."
ssh ${VPS_USER}@${VPS_IP} << 'REMOTE_SCRIPT'
set -e

# Install Python dependencies
cd /opt/dexbrain
apt-get update
apt-get install -y python3-pip python3-venv

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install --upgrade pip
pip install aiohttp asyncio sqlalchemy aioredis numpy pandas scikit-learn

# Create necessary directories
mkdir -p /opt/dexbrain/knowledge_base
mkdir -p /opt/dexbrain/models
mkdir -p /opt/dexbrain/data

# Make startup script executable
chmod +x dexbrain-start.py

# Install systemd service
mv /tmp/dexbrain.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable dexbrain

# Stop any existing instance
systemctl stop dexbrain 2>/dev/null || true

# Start DexBrain service
systemctl start dexbrain

# Check status
sleep 5
systemctl status dexbrain --no-pager

echo "DexBrain deployment completed!"
REMOTE_SCRIPT

echo -e "${GREEN}DexBrain deployment complete!${NC}"
echo "Service status:"
ssh ${VPS_USER}@${VPS_IP} "systemctl status dexbrain --no-pager"
echo ""
echo "View logs with:"
echo "  ssh ${VPS_USER}@${VPS_IP} 'tail -f /opt/dexter-ai/dexbrain.log'"