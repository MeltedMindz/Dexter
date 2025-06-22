#!/bin/bash
# Vault Infrastructure Deployment Script for VPS
# Deploys complete DexBrain with vault integration, logging, and monitoring

set -e

echo "=== Vault Infrastructure Deployment Script ==="

# Configuration
VPS_IP="5.78.71.231"
VPS_USER="root"
DEPLOY_DIR="/opt/dexter-ai"
DOCKER_COMPOSE_FILE="docker-compose.vault.yml"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}Starting Vault Infrastructure deployment to VPS...${NC}"

# Create deployment directories on VPS
echo "Creating deployment directories..."
ssh ${VPS_USER}@${VPS_IP} << 'EOF'
mkdir -p /opt/dexter-ai/{backend,monitoring,logs}
mkdir -p /opt/dexter-ai/backend/{dexbrain,ai,services,logging,monitoring}
mkdir -p /opt/dexter-ai/monitoring/{prometheus,grafana/dashboards,grafana/datasources}
EOF

# Copy all backend files to VPS
echo "Copying vault infrastructure files to VPS..."
rsync -avz --exclude='__pycache__' --exclude='*.pyc' --exclude='.env' --exclude='venv' \
    ./dexbrain/ ${VPS_USER}@${VPS_IP}:${DEPLOY_DIR}/backend/dexbrain/
rsync -avz --exclude='__pycache__' --exclude='*.pyc' \
    ./ai/ ${VPS_USER}@${VPS_IP}:${DEPLOY_DIR}/backend/ai/
rsync -avz --exclude='__pycache__' --exclude='*.pyc' \
    ./services/ ${VPS_USER}@${VPS_IP}:${DEPLOY_DIR}/backend/services/
rsync -avz --exclude='__pycache__' --exclude='*.pyc' \
    ./logging/ ${VPS_USER}@${VPS_IP}:${DEPLOY_DIR}/backend/logging/
rsync -avz --exclude='__pycache__' --exclude='*.pyc' \
    ./monitoring/ ${VPS_USER}@${VPS_IP}:${DEPLOY_DIR}/backend/monitoring/

# Copy Docker files
echo "Copying Docker configuration..."
rsync -avz requirements.txt ${VPS_USER}@${VPS_IP}:${DEPLOY_DIR}/backend/
rsync -avz docker-compose.vault.yml ${VPS_USER}@${VPS_IP}:${DEPLOY_DIR}/
rsync -avz Dockerfile.* ${VPS_USER}@${VPS_IP}:${DEPLOY_DIR}/backend/

# Create environment file for vault infrastructure
echo "Creating environment configuration..."
cat > .env.vault << 'EOF'
# Database Configuration
DATABASE_URL=postgresql://postgres:dexter_secure_password_2024@db:5432/dexter_db
DB_PASSWORD=dexter_secure_password_2024

# Redis Configuration
REDIS_PASSWORD=dexter_redis_2024

# Blockchain Configuration
ALCHEMY_API_KEY=${ALCHEMY_API_KEY}
BASE_RPC_URL=https://base-mainnet.g.alchemy.com/v2/${ALCHEMY_API_KEY}
BASESCAN_API_KEY=${BASESCAN_API_KEY}

# DexBrain Configuration
DEXBRAIN_API_URL=http://dexbrain-api:8080
API_PORT=8080
LOG_LEVEL=INFO
LOG_FORMAT=%(asctime)s - %(name)s - %(levelname)s - %(message)s
ENVIRONMENT=production
DEBUG=false
VAULT_INTEGRATION=true

# Monitoring
GRAFANA_PASSWORD=dexter_admin_2024

# Service URLs (internal Docker network)
VAULT_STRATEGY_URL=http://vault-strategy:8081
COMPOUND_SERVICE_URL=http://compound-service:8082
LOG_AGGREGATOR_URL=http://log-aggregator:8084
EOF

# Copy environment file
scp .env.vault ${VPS_USER}@${VPS_IP}:${DEPLOY_DIR}/.env
rm .env.vault

# Create Prometheus configuration
echo "Creating Prometheus configuration..."
cat > prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'dexbrain-api'
    static_configs:
      - targets: ['dexbrain-api:8080']
  
  - job_name: 'vault-strategy'
    static_configs:
      - targets: ['vault-strategy:8081']
  
  - job_name: 'compound-service'
    static_configs:
      - targets: ['compound-service:8082']
  
  - job_name: 'dexbrain-metrics'
    static_configs:
      - targets: ['dexbrain-metrics:8081']
  
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']
EOF

scp prometheus.yml ${VPS_USER}@${VPS_IP}:${DEPLOY_DIR}/monitoring/
rm prometheus.yml

# Create Grafana datasource configuration
echo "Creating Grafana configuration..."
cat > datasources.yml << 'EOF'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
EOF

scp datasources.yml ${VPS_USER}@${VPS_IP}:${DEPLOY_DIR}/monitoring/grafana/datasources/
rm datasources.yml

# Create startup script for API server integration
echo "Creating API server startup script..."
cat > start-api-server.sh << 'EOF'
#!/bin/bash
# Start the DexBrain API server with vault integration

cd /app

# Ensure log directory exists
mkdir -p /opt/dexter-ai

# Start the API server
python -m dexbrain.api_server
EOF

scp start-api-server.sh ${VPS_USER}@${VPS_IP}:${DEPLOY_DIR}/backend/
rm start-api-server.sh

# Deploy and start services on VPS
echo "Deploying services on VPS..."
ssh ${VPS_USER}@${VPS_IP} << 'REMOTE_SCRIPT'
set -e

cd /opt/dexter-ai

# Make startup script executable
chmod +x backend/start-api-server.sh

# Stop existing services if running
echo "Stopping existing services..."
docker-compose -f docker-compose.vault.yml down 2>/dev/null || true

# Pull latest images
echo "Pulling Docker images..."
docker-compose -f docker-compose.vault.yml pull || true

# Build custom images
echo "Building custom Docker images..."
docker-compose -f docker-compose.vault.yml build

# Start all services
echo "Starting vault infrastructure services..."
docker-compose -f docker-compose.vault.yml up -d

# Wait for services to start
echo "Waiting for services to start..."
sleep 30

# Check service status
echo "Checking service status..."
docker-compose -f docker-compose.vault.yml ps

# Show logs from key services
echo ""
echo "=== DexBrain API Logs ==="
docker-compose -f docker-compose.vault.yml logs --tail=20 dexbrain-api

echo ""
echo "=== Vault Strategy Logs ==="
docker-compose -f docker-compose.vault.yml logs --tail=20 vault-strategy

echo ""
echo "=== Compound Service Logs ==="
docker-compose -f docker-compose.vault.yml logs --tail=20 compound-service

echo ""
echo "Vault infrastructure deployment completed!"
REMOTE_SCRIPT

# Create nginx configuration for external access
echo "Creating nginx configuration for external access..."
cat > vault-api.nginx << 'EOF'
# DexBrain API endpoints
location /api/vault/ {
    proxy_pass http://localhost:8080/api/vault/;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection 'upgrade';
    proxy_set_header Host $host;
    proxy_cache_bypass $http_upgrade;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
}

location /api/logs/recent {
    proxy_pass http://localhost:8084/api/logs/recent;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
}

location /api/logs/stream {
    proxy_pass http://localhost:8084/api/logs/stream;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection 'upgrade';
    proxy_set_header Host $host;
    proxy_cache_bypass $http_upgrade;
    
    # SSE specific headers
    proxy_set_header Cache-Control no-cache;
    proxy_set_header X-Accel-Buffering no;
    proxy_buffering off;
    chunked_transfer_encoding off;
}
EOF

# Deploy nginx configuration
scp vault-api.nginx ${VPS_USER}@${VPS_IP}:/tmp/
rm vault-api.nginx

ssh ${VPS_USER}@${VPS_IP} << 'NGINX_UPDATE'
# Add vault API configuration to existing nginx config
cat /tmp/vault-api.nginx >> /etc/nginx/sites-available/dexter-api

# Test nginx configuration
nginx -t

# Reload nginx
systemctl reload nginx

echo "Nginx configuration updated!"
NGINX_UPDATE

echo -e "${GREEN}Vault Infrastructure deployment complete!${NC}"
echo ""
echo "Service URLs:"
echo "  - DexBrain API: http://${VPS_IP}:8080"
echo "  - Prometheus: http://${VPS_IP}:9090"
echo "  - Grafana: http://${VPS_IP}:3002 (admin/dexter_admin_2024)"
echo "  - Log Stream: http://${VPS_IP}:8084/api/logs/stream"
echo ""
echo "External API endpoints:"
echo "  - https://api.dexteragent.com/api/vault/intelligence"
echo "  - https://api.dexteragent.com/api/vault/compound-opportunities"
echo "  - https://api.dexteragent.com/api/vault/analytics"
echo "  - https://api.dexteragent.com/api/logs/recent"
echo ""
echo "View logs with:"
echo "  ssh ${VPS_USER}@${VPS_IP} 'docker-compose -f /opt/dexter-ai/docker-compose.vault.yml logs -f'"
echo ""
echo "Monitor services with:"
echo "  ssh ${VPS_USER}@${VPS_IP} 'docker-compose -f /opt/dexter-ai/docker-compose.vault.yml ps'"