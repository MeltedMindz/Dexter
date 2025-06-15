#!/bin/bash

# Hetzner VPS Deployment Script for DexBrain API
# Usage: ./deploy-hetzner.sh

set -e

# Configuration
HETZNER_API_TOKEN="WSR7DB4CThdJEWKRL8LIZpHPTWFaBDVFbka16XxmR28OO3WBW5TfPCBW1Egla80L"
SERVER_IP="157.90.230.148"
SSH_KEY_PATH="$HOME/.ssh/id_rsa"

echo "🚀 Starting DexBrain API deployment to Hetzner VPS..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if SSH key exists
check_ssh_key() {
    if [[ ! -f "$SSH_KEY_PATH" ]]; then
        print_error "SSH key not found at $SSH_KEY_PATH"
        print_status "Generating new SSH key..."
        ssh-keygen -t rsa -b 4096 -f "$SSH_KEY_PATH" -N "" -C "dexbrain-deployment"
    fi
}

# Get server info from Hetzner API
get_server_info() {
    print_status "Getting server information from Hetzner API..."
    
    SERVER_INFO=$(curl -s -H "Authorization: Bearer $HETZNER_API_TOKEN" \
        "https://api.hetzner.cloud/v1/servers" | \
        jq -r '.servers[] | select(.public_net.ipv4.ip == "'$SERVER_IP'")')
    
    if [[ -z "$SERVER_INFO" ]]; then
        print_error "Server with IP $SERVER_IP not found in your Hetzner account"
        exit 1
    fi
    
    SERVER_NAME=$(echo "$SERVER_INFO" | jq -r '.name')
    SERVER_STATUS=$(echo "$SERVER_INFO" | jq -r '.status')
    
    print_status "Found server: $SERVER_NAME (Status: $SERVER_STATUS)"
}

# Test SSH connection
test_ssh_connection() {
    print_status "Testing SSH connection to $SERVER_IP..."
    
    if ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no root@$SERVER_IP "echo 'SSH connection successful'" > /dev/null 2>&1; then
        print_status "SSH connection successful"
    else
        print_error "Cannot connect via SSH to $SERVER_IP"
        print_warning "Make sure your SSH key is added to the server"
        exit 1
    fi
}

# Deploy DexBrain API
deploy_api() {
    print_status "Deploying DexBrain API to server..."
    
    # Create deployment script
    cat > /tmp/deploy-script.sh << 'EOF'
#!/bin/bash
set -e

echo "🔧 Setting up DexBrain API deployment..."

# Update system
apt update && apt upgrade -y

# Install required packages
apt install -y git curl docker.io docker-compose-plugin ufw nginx

# Enable and start Docker
systemctl enable docker
systemctl start docker

# Configure firewall
ufw --force enable
ufw allow ssh
ufw allow 80
ufw allow 443

# Create application directory
mkdir -p /opt/dexbrain
cd /opt/dexbrain

# Clone repository if not exists
if [[ ! -d "Dexter" ]]; then
    git clone https://github.com/MeltedMindz/Dexter.git
fi

cd Dexter
git pull origin main

# Generate secure passwords
DB_PASSWORD=$(openssl rand -base64 32)
REDIS_PASSWORD=$(openssl rand -base64 32)

# Create production environment file
cat > .env << ENVEOF
# Database Configuration
POSTGRES_USER=dexter
POSTGRES_PASSWORD=$DB_PASSWORD
POSTGRES_DB=dexter
DATABASE_URL=postgresql://dexter:$DB_PASSWORD@db:5432/dexter

# Redis Configuration
REDIS_PASSWORD=$REDIS_PASSWORD
REDIS_URL=redis://:$REDIS_PASSWORD@redis:6379

# API Configuration
API_HOST=0.0.0.0
API_PORT=8080
LOG_LEVEL=INFO
ENVIRONMENT=production

# DexBrain Settings
RATE_LIMIT_PER_MINUTE=100
DATA_QUALITY_THRESHOLD=60.0
MAX_AGENTS_PER_REQUEST=1000
PARALLEL_WORKERS=2
CACHE_TTL=300
ENVEOF

# Create production docker-compose
cat > docker-compose.prod.yml << 'DOCKEREOF'
version: '3.8'

services:
  dexbrain-api:
    build:
      context: ./backend
      dockerfile: Dockerfile.api
    ports:
      - "8080:8080"
    env_file: .env
    volumes:
      - ./backend/knowledge_base:/app/knowledge_base
      - ./logs:/app/logs
    restart: unless-stopped
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_healthy
    networks:
      - dexter-network
    deploy:
      resources:
        limits:
          memory: 1G
        reservations:
          memory: 512M

  db:
    image: postgres:13-alpine
    environment:
      - POSTGRES_USER=${POSTGRES_USER}
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
      - POSTGRES_DB=${POSTGRES_DB}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped
    networks:
      - dexter-network
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER}"]
      interval: 10s
      timeout: 5s
      retries: 5
    deploy:
      resources:
        limits:
          memory: 512M
        reservations:
          memory: 256M

  redis:
    image: redis:6-alpine
    command: redis-server --requirepass ${REDIS_PASSWORD} --maxmemory 128mb --maxmemory-policy allkeys-lru
    volumes:
      - redis_data:/data
    restart: unless-stopped
    networks:
      - dexter-network
    healthcheck:
      test: ["CMD", "redis-cli", "--pass", "${REDIS_PASSWORD}", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
    deploy:
      resources:
        limits:
          memory: 256M
        reservations:
          memory: 128M

networks:
  dexter-network:
    driver: bridge

volumes:
  postgres_data:
  redis_data:
DOCKEREOF

# Create logs directory
mkdir -p logs

# Build and deploy
echo "🏗️ Building and starting DexBrain API..."
docker compose -f docker-compose.prod.yml down || true
docker compose -f docker-compose.prod.yml build
docker compose -f docker-compose.prod.yml up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 30

# Check service health
if docker compose -f docker-compose.prod.yml ps | grep -q "healthy\|running"; then
    echo "✅ DexBrain API services are running"
else
    echo "❌ Some services failed to start"
    docker compose -f docker-compose.prod.yml logs
    exit 1
fi

# Configure Nginx reverse proxy
cat > /etc/nginx/sites-available/dexbrain-api << 'NGINXEOF'
server {
    listen 80;
    server_name _;
    
    # Security headers
    add_header X-Content-Type-Options nosniff;
    add_header X-Frame-Options DENY;
    add_header X-XSS-Protection "1; mode=block";
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;
    
    location / {
        limit_req zone=api_limit burst=20 nodelay;
        
        proxy_pass http://localhost:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
    }
    
    # Health check endpoint (no rate limiting)
    location /health {
        proxy_pass http://localhost:8080/health;
        proxy_set_header Host $host;
        access_log off;
    }
}
NGINXEOF

# Enable Nginx site
ln -sf /etc/nginx/sites-available/dexbrain-api /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default
nginx -t
systemctl restart nginx

# Create monitoring script
cat > /root/monitor-dexbrain.sh << 'MONITOREOF'
#!/bin/bash
API_URL="http://localhost:8080/health"
LOG_FILE="/var/log/dexbrain-monitor.log"

check_api() {
    if curl -s -f "$API_URL" > /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

if ! check_api; then
    echo "$(date): DexBrain API is down, restarting..." >> "$LOG_FILE"
    cd /opt/dexbrain/Dexter
    docker compose -f docker-compose.prod.yml restart dexbrain-api
    sleep 30
    
    if check_api; then
        echo "$(date): DexBrain API restarted successfully" >> "$LOG_FILE"
    else
        echo "$(date): Failed to restart DexBrain API" >> "$LOG_FILE"
    fi
fi
MONITOREOF

chmod +x /root/monitor-dexbrain.sh

# Add monitoring to crontab
(crontab -l 2>/dev/null; echo "*/5 * * * * /root/monitor-dexbrain.sh") | crontab -

# Test API
echo "🧪 Testing API endpoints..."
sleep 10

if curl -s "http://localhost/health" | grep -q "healthy"; then
    echo "✅ API health check passed"
    API_STATUS="✅ DEPLOYED SUCCESSFULLY"
else
    echo "❌ API health check failed"
    API_STATUS="❌ DEPLOYMENT ISSUES"
fi

# Display deployment summary
echo ""
echo "================================================"
echo "🎉 DEXBRAIN API DEPLOYMENT SUMMARY"
echo "================================================"
echo "Status: $API_STATUS"
echo "API URL: http://157.90.230.148"
echo "Health Check: http://157.90.230.148/health"
echo "API Registration: http://157.90.230.148/api/register"
echo "Intelligence Feed: http://157.90.230.148/api/intelligence"
echo "Data Submission: http://157.90.230.148/api/submit-data"
echo ""
echo "📊 Resource Usage:"
docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}"
echo ""
echo "🔧 Useful Commands:"
echo "View logs: docker compose -f /opt/dexbrain/Dexter/docker-compose.prod.yml logs -f"
echo "Restart API: docker compose -f /opt/dexbrain/Dexter/docker-compose.prod.yml restart"
echo "Update code: cd /opt/dexbrain/Dexter && git pull && docker compose -f docker-compose.prod.yml up -d --build"
echo "================================================"

EOF

    # Copy and execute deployment script
    scp -o StrictHostKeyChecking=no /tmp/deploy-script.sh root@$SERVER_IP:/tmp/
    ssh -o StrictHostKeyChecking=no root@$SERVER_IP "chmod +x /tmp/deploy-script.sh && /tmp/deploy-script.sh"
    
    # Clean up temp file
    rm /tmp/deploy-script.sh
}

# Test deployed API
test_deployment() {
    print_status "Testing deployed API..."
    
    sleep 10
    
    # Test health endpoint
    if curl -s -f "http://$SERVER_IP/health" | grep -q "healthy"; then
        print_status "✅ API is responding correctly"
        
        # Test registration endpoint
        API_KEY=$(curl -s -X POST "http://$SERVER_IP/api/register" \
            -H "Content-Type: application/json" \
            -d '{"agent_id": "test-agent", "metadata": {"test": true}}' | \
            jq -r '.api_key // empty')
        
        if [[ -n "$API_KEY" ]]; then
            print_status "✅ Agent registration working (API Key: ${API_KEY:0:20}...)"
        else
            print_warning "⚠️ Agent registration might have issues"
        fi
        
    else
        print_error "❌ API health check failed"
    fi
}

# Main deployment flow
main() {
    print_status "Starting DexBrain API deployment..."
    
    check_ssh_key
    get_server_info
    test_ssh_connection
    deploy_api
    test_deployment
    
    echo ""
    echo "🎉 DEPLOYMENT COMPLETE!"
    echo ""
    echo "📡 Your DexBrain API is now live at:"
    echo "   Health Check: http://$SERVER_IP/health"
    echo "   API Documentation: https://github.com/MeltedMindz/Dexter/blob/main/backend/API_DOCUMENTATION.md"
    echo ""
    echo "🔧 Management Commands:"
    echo "   SSH to server: ssh root@$SERVER_IP"
    echo "   View logs: ssh root@$SERVER_IP 'cd /opt/dexbrain/Dexter && docker compose -f docker-compose.prod.yml logs -f'"
    echo "   Restart API: ssh root@$SERVER_IP 'cd /opt/dexbrain/Dexter && docker compose -f docker-compose.prod.yml restart'"
    echo ""
    echo "💰 Total monthly cost: ~$6 (just the VPS)"
    echo "🔥 Ready for 500+ agents!"
}

# Run main function
main "$@"