#!/bin/bash

# Simple DexBrain Deployment with Full Monitoring
# Usage: ./deploy-simple.sh

set -e

SERVER_IP="157.90.230.148"

echo "🚀 Deploying DexBrain with Full Monitoring Stack..."

# Create comprehensive deployment script
cat > /tmp/simple-deploy.sh << 'EOF'
#!/bin/bash
set -e

echo "🔧 Installing Docker and dependencies..."

# Update system
apt update

# Install Docker the simple way
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
systemctl enable docker
systemctl start docker

# Install Docker Compose V1 (more compatible)
curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Install other tools
apt install -y git nginx ufw

# Setup firewall
ufw --force enable
ufw allow ssh
ufw allow 80
ufw allow 443
ufw allow 3000  # Grafana
ufw allow 9090  # Prometheus

echo "✅ Dependencies installed"

# Create application directory
mkdir -p /opt/dexbrain
cd /opt/dexbrain

# Clone repository
if [[ ! -d "Dexter" ]]; then
    git clone https://github.com/MeltedMindz/Dexter.git
fi

cd Dexter
git pull origin main

echo "🗃️ Setting up databases and monitoring..."

# Generate secure passwords
DB_PASS=$(openssl rand -base64 32)
REDIS_PASS=$(openssl rand -base64 32)

# Create production environment
cat > .env << ENVEOF
# Database Configuration
POSTGRES_USER=dexter
POSTGRES_PASSWORD=$DB_PASS
POSTGRES_DB=dexter
DATABASE_URL=postgresql://dexter:$DB_PASS@db:5432/dexter

# Redis Configuration
REDIS_PASSWORD=$REDIS_PASS
REDIS_URL=redis://:$REDIS_PASS@redis:6379

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

# Monitoring
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=dexbrain_$(openssl rand -base64 16)
ENVEOF

# Create enhanced docker-compose with monitoring
cat > docker-compose.monitoring.yml << 'DOCKEREOF'
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
      - db
      - redis
    networks:
      - dexter-network
    labels:
      - "prometheus.io/scrape=true"
      - "prometheus.io/port=8080"
      - "prometheus.io/path=/metrics"

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

  redis:
    image: redis:6-alpine
    command: redis-server --requirepass ${REDIS_PASSWORD} --maxmemory 256mb --maxmemory-policy allkeys-lru
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

  prometheus:
    image: prom/prometheus:v2.44.0
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--web.enable-lifecycle'
      - '--storage.tsdb.retention.time=30d'
    restart: unless-stopped
    networks:
      - dexter-network

  grafana:
    image: grafana/grafana:9.5.2
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/provisioning:/etc/grafana/provisioning
      - ./monitoring/grafana/dashboards:/var/lib/grafana/dashboards
    environment:
      - GF_SECURITY_ADMIN_USER=${GRAFANA_ADMIN_USER:-admin}
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_ADMIN_PASSWORD}
      - GF_USERS_ALLOW_SIGN_UP=false
      - GF_SECURITY_ALLOW_EMBEDDING=true
      - GF_AUTH_ANONYMOUS_ENABLED=true
      - GF_AUTH_ANONYMOUS_ORG_ROLE=Viewer
    restart: unless-stopped
    networks:
      - dexter-network

  node-exporter:
    image: prom/node-exporter:v1.5.0
    ports:
      - "9100:9100"
    volumes:
      - /proc:/host/proc:ro
      - /sys:/host/sys:ro
      - /:/rootfs:ro
    command:
      - '--path.procfs=/host/proc'
      - '--path.rootfs=/rootfs'
      - '--path.sysfs=/host/sys'
      - '--collector.filesystem.mount-points-exclude=^/(sys|proc|dev|host|etc)($$|/)'
    restart: unless-stopped
    networks:
      - dexter-network

networks:
  dexter-network:
    driver: bridge

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:
DOCKEREOF

# Create monitoring configuration
mkdir -p monitoring/grafana/{provisioning/{datasources,dashboards},dashboards}

# Prometheus config
cat > monitoring/prometheus.yml << 'PROMEOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  # - "first_rules.yml"

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'dexbrain-api'
    static_configs:
      - targets: ['dexbrain-api:8080']
    metrics_path: '/metrics'
    scrape_interval: 5s

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']

  - job_name: 'postgres'
    static_configs:
      - targets: ['db:5432']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']
PROMEOF

# Grafana datasource
cat > monitoring/grafana/provisioning/datasources/prometheus.yml << 'DSEOF'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
DSEOF

# Grafana dashboard provisioning
cat > monitoring/grafana/provisioning/dashboards/dashboards.yml << 'DASHEOF'
apiVersion: 1

providers:
  - name: 'DexBrain Dashboards'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /var/lib/grafana/dashboards
DASHEOF

echo "🏗️ Building and starting all services..."

# Create logs directory
mkdir -p logs

# Build and start everything
docker-compose -f docker-compose.monitoring.yml down || true
docker-compose -f docker-compose.monitoring.yml build
docker-compose -f docker-compose.monitoring.yml up -d

# Wait for services to start
echo "⏳ Waiting for services to initialize..."
sleep 45

# Check service status
echo "📊 Service Status:"
docker-compose -f docker-compose.monitoring.yml ps

# Test API health
echo "🧪 Testing API health..."
if curl -s http://localhost:8080/health | grep -q "healthy"; then
    echo "✅ DexBrain API is healthy"
else
    echo "❌ API health check failed"
    docker-compose -f docker-compose.monitoring.yml logs dexbrain-api
fi

# Configure Nginx reverse proxy with monitoring access
cat > /etc/nginx/sites-available/dexbrain << 'NGINXEOF'
server {
    listen 80;
    server_name _;
    
    # Security headers
    add_header X-Content-Type-Options nosniff;
    add_header X-Frame-Options SAMEORIGIN;
    add_header X-XSS-Protection "1; mode=block";
    
    # API endpoints
    location /api/ {
        proxy_pass http://localhost:8080/api/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    # Health endpoint
    location /health {
        proxy_pass http://localhost:8080/health;
        proxy_set_header Host $host;
        access_log off;
    }
    
    # Metrics endpoint (for monitoring)
    location /metrics {
        proxy_pass http://localhost:8080/metrics;
        proxy_set_header Host $host;
    }
    
    # Grafana dashboard (public read-only)
    location /monitoring/ {
        proxy_pass http://localhost:3000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Allow iframe embedding for homepage
        add_header X-Frame-Options SAMEORIGIN;
    }
    
    # Prometheus (admin only)
    location /prometheus/ {
        proxy_pass http://localhost:9090/;
        proxy_set_header Host $host;
        
        # Basic auth for admin access
        auth_basic "Prometheus Admin";
        auth_basic_user_file /etc/nginx/.htpasswd;
    }
}
NGINXEOF

# Create admin password for Prometheus access
echo "admin:$(openssl passwd -apr1 'dexbrain_admin')" > /etc/nginx/.htpasswd

# Enable the site
ln -sf /etc/nginx/sites-available/dexbrain /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default
nginx -t
systemctl restart nginx

# Create monitoring health check script
cat > /root/monitor-full-stack.sh << 'MONITOREOF'
#!/bin/bash
LOG_FILE="/var/log/dexbrain-monitor.log"
API_URL="http://localhost:8080/health"
GRAFANA_URL="http://localhost:3000/api/health"

check_service() {
    local service_name=$1
    local url=$2
    
    if curl -s -f "$url" > /dev/null 2>&1; then
        return 0
    else
        echo "$(date): $service_name is down, restarting..." >> "$LOG_FILE"
        cd /opt/dexbrain/Dexter
        docker-compose -f docker-compose.monitoring.yml restart $service_name
        return 1
    fi
}

# Check all critical services
check_service "dexbrain-api" "$API_URL"
check_service "grafana" "$GRAFANA_URL"

# Check disk space
DISK_USAGE=$(df / | awk 'NR==2 {print $5}' | sed 's/%//')
if [ "$DISK_USAGE" -gt 85 ]; then
    echo "$(date): High disk usage: ${DISK_USAGE}%" >> "$LOG_FILE"
fi

# Check memory usage
MEM_USAGE=$(free | awk '/^Mem:/ {printf "%.0f", $3/$2 * 100}')
if [ "$MEM_USAGE" -gt 90 ]; then
    echo "$(date): High memory usage: ${MEM_USAGE}%" >> "$LOG_FILE"
fi
MONITOREOF

chmod +x /root/monitor-full-stack.sh

# Add to crontab (check every 2 minutes)
(crontab -l 2>/dev/null; echo "*/2 * * * * /root/monitor-full-stack.sh") | crontab -

# Get Grafana admin password
GRAFANA_PASS=$(grep GRAFANA_ADMIN_PASSWORD .env | cut -d'=' -f2)

# Display final summary
echo ""
echo "🎉 DEXBRAIN FULL MONITORING STACK DEPLOYED!"
echo "============================================="
echo ""
echo "🔗 API Endpoints:"
echo "   Health: http://157.90.230.148/health"
echo "   API: http://157.90.230.148/api/"
echo "   Metrics: http://157.90.230.148/metrics"
echo ""
echo "📊 Monitoring Dashboards:"
echo "   Grafana (Public): http://157.90.230.148/monitoring/"
echo "   Grafana (Admin): http://157.90.230.148:3000 (admin / $GRAFANA_PASS)"
echo "   Prometheus: http://157.90.230.148/prometheus/ (admin / dexbrain_admin)"
echo ""
echo "🔧 Embed on Homepage:"
echo "   <iframe src='http://157.90.230.148/monitoring/d/dexbrain/dexbrain-intelligence?orgId=1&refresh=5s&kiosk' width='100%' height='600px'></iframe>"
echo ""
echo "📈 Monitoring Features:"
echo "   ✅ Real-time API metrics"
echo "   ✅ Agent activity tracking" 
echo "   ✅ Database performance"
echo "   ✅ System resources"
echo "   ✅ Network statistics"
echo "   ✅ Auto-restart on failures"
echo ""
echo "💾 Data Retention: 30 days"
echo "🔄 Auto-monitoring: Every 2 minutes"
echo "============================================="

EOF

# Copy and execute the deployment
scp -o StrictHostKeyChecking=no /tmp/simple-deploy.sh root@$SERVER_IP:/tmp/
ssh -o StrictHostKeyChecking=no root@$SERVER_IP "chmod +x /tmp/simple-deploy.sh && /tmp/simple-deploy.sh"

echo ""
echo "🎯 Next Step: Create Grafana Dashboard for Homepage"
echo "Visit http://157.90.230.148:3000 to configure the public monitoring view"