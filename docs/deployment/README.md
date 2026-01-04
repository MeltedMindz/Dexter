# Dexter Protocol Deployment Guide

> **Complete deployment instructions for the Dexter Protocol infrastructure**

[![Infrastructure](https://img.shields.io/badge/Infrastructure-Production-green)](https://status.dexteragent.com)
[![Monitoring](https://img.shields.io/badge/Monitoring-Grafana-orange)](https://monitoring.dexteragent.com)
[![Uptime](https://img.shields.io/badge/Uptime-99.9%25-brightgreen)](https://status.dexteragent.com)

## 🏗️ Deployment Overview

Dexter Protocol uses a multi-tier architecture with separate deployment strategies for each component:

1. **Smart Contracts**: On-chain deployment to Base Network and Ethereum
2. **AI Services**: VPS deployment with Docker orchestration
3. **Frontend Website**: Vercel deployment with automatic CI/CD
4. **Monitoring**: Prometheus + Grafana stack for observability

## 🔗 Smart Contract Deployment

### Prerequisites

```bash
# Required tools
npm install -g @foundry/cli
npm install -g hardhat
npm install -g @openzeppelin/contracts

# Environment setup
git clone https://github.com/MeltedMindz/Dexter.git
cd Dexter/contracts
npm install
```

### Environment Configuration

Create `.env` file in contracts directory:
```bash
# Network RPCs
BASE_RPC_URL=https://base-mainnet.g.alchemy.com/v2/YOUR_KEY
ETHEREUM_RPC_URL=https://eth-mainnet.g.alchemy.com/v2/YOUR_KEY
BASE_TESTNET_RPC_URL=https://base-sepolia.g.alchemy.com/v2/YOUR_KEY

# Deployment keys
DEPLOYER_PRIVATE_KEY=your_deployer_private_key
MULTISIG_ADDRESS=0x742d35Cc6637C0532c2c1dD7A2b43F8E7b6b8D9f

# Block explorers
BASESCAN_API_KEY=your_basescan_api_key
ETHERSCAN_API_KEY=your_etherscan_api_key

# Protocol configuration
MAX_POSITIONS_PER_ADDRESS=200
MAX_REWARD_X64=1844674407370955161  # 2% in Q64
MAX_GAS_PER_COMPOUND=300000
```

### Deployment Sequence

#### 1. Core Infrastructure Deployment

```bash
# Deploy core contracts first
npx hardhat run scripts/deploy-core.js --network base

# Contracts deployed:
# - ConfigurationManager.sol
# - AdvancedSecurityGuard.sol  
# - TWAPOracle.sol
# - PriceAggregator.sol
```

#### 2. Position Management Deployment

```bash
# Deploy position management system
npx hardhat run scripts/deploy-position-management.js --network base

# Contracts deployed:
# - DexterCompoundor.sol
# - ConfigurableDexterCompoundor.sol
# - GasOptimizedDexterCompoundor.sol
# - DexterMultiCompoundor.sol
# - DexterV3Utils.sol
```

#### 3. Vault System Deployment

```bash
# Deploy ERC4626 vault infrastructure
npx hardhat run scripts/deploy-vaults.js --network base

# Contracts deployed:
# - DexterVault.sol (implementation)
# - VaultFactory.sol
# - FeeManager.sol
# - StrategyManager.sol
# - MultiRangeManager.sol
# - VaultClearing.sol
```

#### 4. Oracle System Deployment

```bash
# Deploy ML and price oracles
npx hardhat run scripts/deploy-oracles.js --network base

# Contracts deployed:
# - MLValidationOracle.sol
# - MultiOracleValidator.sol
```

#### 5. Uniswap V4 Hook Deployment

```bash
# Deploy V4 hooks (when V4 is live)
cd dexter-liquidity/uniswap-v4
forge script script/DeployHook.s.sol --rpc-url $BASE_RPC_URL --broadcast

# Contracts deployed:
# - SimpleDexterHook.sol
# - DexterMath.sol
```

### Contract Verification

```bash
# Verify all contracts on block explorer
npx hardhat verify --network base CONTRACT_ADDRESS

# Bulk verification script
bash scripts/verify-all-contracts.sh
```

### Deployment Configuration

Update contract addresses in configuration files:
```typescript
// contracts/deployed-addresses.json
{
  "base": {
    "ConfigurationManager": "0x...",
    "DexterCompoundor": "0x...",
    "VaultFactory": "0x...",
    "MLValidationOracle": "0x..."
  }
}
```

## 🤖 AI Services Deployment

### VPS Infrastructure Setup

**Server Requirements:**
- **CPU**: 8 cores minimum (16 cores recommended)
- **RAM**: 32GB minimum (64GB recommended)
- **Storage**: 500GB SSD minimum (1TB recommended)
- **Network**: 1Gbps connection
- **OS**: Ubuntu 22.04 LTS

### Docker Environment Setup

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Install Python and dependencies
sudo apt install python3.11 python3.11-venv python3-pip git htop -y
```

### Service Deployment

#### 1. Clone Repository and Setup

```bash
# Clone main repository
git clone https://github.com/MeltedMindz/Dexter.git
cd Dexter

# Setup backend services
cd backend
python3.11 -m venv env
source env/bin/activate
pip install -r requirements.txt

# Setup AI services
cd ../dexter-liquidity
python3.11 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

#### 2. Environment Configuration

Create production environment files:

**Backend Environment** (`backend/.env`):
```bash
# Database
DATABASE_URL=postgresql://dexter:secure_password@localhost:5432/dexter_production
REDIS_PASSWORD=secure_redis_password

# Blockchain
ALCHEMY_API_KEY=your_production_alchemy_key
BASE_RPC_URL=https://base-mainnet.g.alchemy.com/v2/YOUR_KEY
BASESCAN_API_KEY=your_basescan_key

# AI Services
OPENAI_API_KEY=your_openai_key
ML_MODEL_PATH=/opt/dexter-ai/models
PREDICTION_CACHE_TTL=300

# Production settings
ENVIRONMENT=production
LOG_LEVEL=INFO
DEBUG=false

# Service ports
DEXBRAIN_PORT=8001
ENHANCED_ALCHEMY_PORT=8002
```

**AI Services Environment** (`dexter-liquidity/.env`):
```bash
# ML Configuration
ML_TRAINING_INTERVAL=1800  # 30 minutes
MODEL_PERSISTENCE=local
FEATURE_COUNT=20
CONFIDENCE_THRESHOLD=0.7

# Data pipeline
PARALLEL_WORKERS=8
MAX_SLIPPAGE=0.005
POSITION_BATCH_SIZE=50

# Performance
CACHE_TTL=300
MAX_MEMORY_USAGE=16GB
```

#### 3. Database Setup

```bash
# Install PostgreSQL
sudo apt install postgresql postgresql-contrib -y

# Setup database
sudo -u postgres createuser dexter
sudo -u postgres createdb dexter_production
sudo -u postgres psql -c "ALTER USER dexter PASSWORD 'secure_password';"

# Run migrations
cd backend
python -m alembic upgrade head
```

#### 4. Redis Setup

```bash
# Install Redis
sudo apt install redis-server -y

# Configure Redis
sudo nano /etc/redis/redis.conf
# Add: requirepass secure_redis_password

sudo systemctl restart redis-server
```

#### 5. Service Deployment

Create systemd service files:

**DexBrain Service** (`/etc/systemd/system/dexter-dexbrain.service`):
```ini
[Unit]
Description=Dexter DexBrain Intelligence Hub
After=network.target postgresql.service redis.service

[Service]
Type=simple
User=dexter
WorkingDirectory=/opt/dexter/backend
Environment=PATH=/opt/dexter/backend/env/bin
ExecStart=/opt/dexter/backend/env/bin/python -m dexbrain.core
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=dexter-dexbrain
MemoryLimit=4G
TimeoutStopSec=30

[Install]
WantedBy=multi-user.target
```

**ML Pipeline Service** (`/etc/systemd/system/dexter-ml-pipeline.service`):
```ini
[Unit]
Description=Dexter ML Pipeline Service
After=network.target

[Service]
Type=simple
User=dexter
WorkingDirectory=/opt/dexter/backend
Environment=PATH=/opt/dexter/backend/env/bin
ExecStart=/opt/dexter/backend/env/bin/python -m ai.simplified_ml_pipeline
Restart=always
RestartSec=30
StandardOutput=journal
StandardError=journal
SyslogIdentifier=dexter-ml-pipeline
MemoryLimit=8G
TimeoutStopSec=60

[Install]
WantedBy=multi-user.target
```

**Position Harvester Service** (`/etc/systemd/system/dexter-position-harvester.service`):
```ini
[Unit]
Description=Dexter Position Harvester
After=network.target

[Service]
Type=simple
User=dexter
WorkingDirectory=/opt/dexter/dexter-liquidity
Environment=PATH=/opt/dexter/dexter-liquidity/env/bin
ExecStart=/opt/dexter/dexter-liquidity/env/bin/python -m agents.position_harvester
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=dexter-position-harvester
MemoryLimit=2G

[Install]
WantedBy=multi-user.target
```

**Enhanced Alchemy Service** (`/etc/systemd/system/dexter-enhanced-alchemy.service`):
```ini
[Unit]
Description=Dexter Enhanced Alchemy Service
After=network.target

[Service]
Type=simple
User=dexter
WorkingDirectory=/opt/dexter/dexter-liquidity
Environment=PATH=/opt/dexter/dexter-liquidity/env/bin
ExecStart=/opt/dexter/dexter-liquidity/env/bin/python -m data.enhanced_alchemy
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=dexter-enhanced-alchemy
MemoryLimit=2G

[Install]
WantedBy=multi-user.target
```

#### 6. Start All Services

```bash
# Enable and start all services
sudo systemctl enable dexter-dexbrain
sudo systemctl enable dexter-ml-pipeline
sudo systemctl enable dexter-position-harvester
sudo systemctl enable dexter-enhanced-alchemy

sudo systemctl start dexter-dexbrain
sudo systemctl start dexter-ml-pipeline
sudo systemctl start dexter-position-harvester
sudo systemctl start dexter-enhanced-alchemy

# Check status
sudo systemctl status dexter-*
```

## 🌐 Frontend Website Deployment

### Vercel Deployment Setup

#### 1. Repository Configuration

The frontend is deployed from a separate repository:
```bash
# Website repository
git clone https://github.com/MeltedMindz/dexter-website.git
```

#### 2. Vercel Project Setup

**Import Project:**
1. Connect GitHub repository to Vercel
2. Configure build settings:
   - Framework: Next.js
   - Build Command: `npm run build`
   - Output Directory: `.next`
   - Install Command: `npm install`

#### 3. Environment Variables

Configure in Vercel dashboard:

**Required Variables:**
```bash
# Blockchain integration
NEXT_PUBLIC_ALCHEMY_API_KEY=your_alchemy_key
NEXT_PUBLIC_WC_PROJECT_ID=your_walletconnect_id
NEXT_PUBLIC_BASE_RPC_URL=https://base-mainnet.g.alchemy.com/v2/[key]

# Backend integration
NEXT_PUBLIC_API_URL=https://api.dexteragent.com
NEXT_PUBLIC_DEXBRAIN_URL=https://dexbrain.dexteragent.com

# AI services
OPENAI_API_KEY=your_openai_key
CHAT_ADMIN_PASSWORD=secure_admin_password

# Application
NEXT_PUBLIC_APP_URL=https://www.dexteragent.com
NEXT_PUBLIC_ENVIRONMENT=production
NODE_ENV=production
```

#### 4. Domain Configuration

**Custom Domain Setup:**
1. Add domain: `www.dexteragent.com`
2. Configure DNS records:
   ```
   CNAME www 76.76.19.61
   ```
3. Enable SSL certificate

#### 5. Deployment Pipeline

**Automatic Deployment:**
- Production: `main` branch → `www.dexteragent.com`
- Preview: Feature branches → Preview URLs

## 📊 Monitoring Setup

### Prometheus Configuration

**Prometheus Config** (`/etc/prometheus/prometheus.yml`):
```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'dexter-services'
    static_configs:
      - targets: ['localhost:8001', 'localhost:8002']
    metrics_path: /metrics
    scrape_interval: 10s

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['localhost:9100']

  - job_name: 'postgres-exporter'
    static_configs:
      - targets: ['localhost:9187']
```

### Grafana Dashboard

**Installation:**
```bash
# Install Grafana
sudo apt-get install -y software-properties-common
sudo add-apt-repository "deb https://packages.grafana.com/oss/deb stable main"
wget -q -O - https://packages.grafana.com/gpg.key | sudo apt-key add -
sudo apt-get update
sudo apt-get install grafana

# Enable and start
sudo systemctl enable grafana-server
sudo systemctl start grafana-server
```

**Dashboard Configuration:**
1. Import Dexter Protocol dashboard: `dashboards/dexter-overview.json`
2. Configure data source: Prometheus at `localhost:9090`
3. Set up alerting for critical metrics

### Log Aggregation

**Centralized Logging with rsyslog:**
```bash
# Configure log forwarding
echo "*.* @@localhost:514" >> /etc/rsyslog.conf
sudo systemctl restart rsyslog

# Log rotation
cat > /etc/logrotate.d/dexter-services << EOF
/var/log/dexter/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    copytruncate
}
EOF
```

## 🔧 Production Configuration

### SSL/TLS Setup

**Nginx Configuration** (`/etc/nginx/sites-available/dexter-api`):
```nginx
server {
    listen 443 ssl http2;
    server_name api.dexteragent.com;

    ssl_certificate /etc/letsencrypt/live/api.dexteragent.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.dexteragent.com/privkey.pem;

    location / {
        proxy_pass http://localhost:8001;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /data/ {
        proxy_pass http://localhost:8002/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Firewall Configuration

```bash
# Configure UFW firewall
sudo ufw enable
sudo ufw allow ssh
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw allow 8001/tcp  # DexBrain (internal)
sudo ufw allow 8002/tcp  # Enhanced Alchemy (internal)
sudo ufw allow 3000/tcp  # Grafana
```

### Backup Strategy

**Database Backup** (`/opt/dexter/scripts/backup-db.sh`):
```bash
#!/bin/bash
BACKUP_DIR="/opt/dexter/backups"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup
pg_dump dexter_production > "$BACKUP_DIR/dexter_$DATE.sql"

# Compress backup
gzip "$BACKUP_DIR/dexter_$DATE.sql"

# Keep only last 15 days
find "$BACKUP_DIR" -name "dexter_*.sql.gz" -mtime +15 -delete

# Upload to cloud storage (optional)
# aws s3 cp "$BACKUP_DIR/dexter_$DATE.sql.gz" s3://dexter-backups/
```

**Cron Setup:**
```bash
# Add to crontab
0 2 * * * /opt/dexter/scripts/backup-db.sh
0 */6 * * * /opt/dexter/scripts/backup-logs.sh
```

## ⚙️ Health Checks & Monitoring

### Service Health Endpoints

**Health Check Script** (`/opt/dexter/scripts/health-check.sh`):
```bash
#!/bin/bash

# Check DexBrain
curl -f http://localhost:8001/health || echo "DexBrain DOWN"

# Check Enhanced Alchemy
curl -f http://localhost:8002/health || echo "Enhanced Alchemy DOWN"

# Check database
pg_isready -h localhost -p 5432 -U dexter || echo "Database DOWN"

# Check Redis
redis-cli ping || echo "Redis DOWN"

# Check system resources
df -h | grep -E '9[0-9]%|100%' && echo "Disk space critical"
free | awk 'NR==2{printf "Memory: %.2f%%\n", $3*100/$2}' | grep -E '9[0-9]|100' && echo "Memory usage high"
```

### Alerting Rules

**Prometheus Alerts** (`/etc/prometheus/alert.rules.yml`):
```yaml
groups:
- name: dexter.rules
  rules:
  - alert: ServiceDown
    expr: up == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "Service {{ $labels.instance }} is down"

  - alert: HighMemoryUsage
    expr: (node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes * 100 > 90
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High memory usage on {{ $labels.instance }}"

  - alert: DatabaseConnectionFailure
    expr: pg_up == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "Database connection failed"
```

## 🚀 Deployment Checklist

### Pre-Deployment

- [ ] Environment variables configured
- [ ] Database migrations tested
- [ ] Smart contracts compiled and tested
- [ ] API keys and secrets secured
- [ ] SSL certificates obtained
- [ ] Monitoring configured
- [ ] Backup strategy implemented

### Smart Contract Deployment

- [ ] Testnet deployment successful
- [ ] Contract verification complete
- [ ] Access controls configured
- [ ] Emergency procedures tested
- [ ] Gas optimization verified
- [ ] Integration tests passed

### Backend Services

- [ ] All services start without errors
- [ ] Database connections established
- [ ] ML models loaded successfully
- [ ] API endpoints responding
- [ ] Rate limiting functional
- [ ] Log aggregation working

### Frontend Deployment

- [ ] Build process successful
- [ ] Environment variables set
- [ ] Domain configuration complete
- [ ] SSL certificate active
- [ ] Web3 integration functional
- [ ] Analytics tracking enabled

### Post-Deployment

- [ ] Health checks passing
- [ ] Monitoring alerts configured
- [ ] Performance metrics baseline established
- [ ] Documentation updated
- [ ] Team notified of deployment
- [ ] Rollback procedure verified

## 🆘 Troubleshooting Guide

### Common Issues

**Service Won't Start:**
```bash
# Check service logs
sudo journalctl -u dexter-dexbrain -f

# Check port conflicts
sudo netstat -tlnp | grep :8001

# Verify environment variables
systemctl show dexter-dexbrain --property=Environment
```

**Database Connection Issues:**
```bash
# Test connection
psql -h localhost -U dexter -d dexter_production

# Check PostgreSQL status
sudo systemctl status postgresql

# Verify firewall rules
sudo ufw status
```

**High Memory Usage:**
```bash
# Check process memory usage
ps aux --sort=-%mem | head

# Monitor ML model memory
sudo -u dexter ps -eo pid,user,cmd,%mem,%cpu --sort=-%mem
```

**ML Model Issues:**
```bash
# Check model files
ls -la /opt/dexter-ai/models/

# Verify training logs
tail -f /var/log/dexter/ml-pipeline.log

# Test model loading
python -c "import joblib; model = joblib.load('model.pkl')"
```

### Emergency Procedures

**Service Recovery:**
```bash
# Restart all services
sudo systemctl restart dexter-*

# Emergency stop
sudo systemctl stop dexter-*

# Check system resources
htop
df -h
free -h
```

**Database Recovery:**
```bash
# Restore from backup
gunzip dexter_YYYYMMDD_HHMMSS.sql.gz
psql dexter_production < dexter_YYYYMMDD_HHMMSS.sql
```

## 📞 Support & Maintenance

### Maintenance Schedule

**Daily:**
- Health check monitoring
- Log review and analysis
- Performance metrics review

**Weekly:**
- Security updates
- Database optimization
- Backup verification

**Monthly:**
- Dependency updates
- Performance optimization
- Security audit

### Support Contacts

- **Infrastructure**: devops@dexteragent.com
- **Smart Contracts**: blockchain@dexteragent.com
- **AI Services**: ai@dexteragent.com
- **Frontend**: frontend@dexteragent.com
- **Emergency**: emergency@dexteragent.com

---

*Last Updated: December 2024 | Version: 2.0.0 | Deployment Guide*