# Vault Infrastructure Deployment Guide

## Overview
This document describes the complete vault infrastructure deployment including DexBrain AI integration, compound services, and real-time monitoring.

## Components Deployed

### 1. DexBrain API Server (Port 8080)
- Main intelligence API with vault integration
- Endpoints:
  - `/api/vault/intelligence` - AI strategy recommendations
  - `/api/vault/compound-opportunities` - Real-time compound opportunities
  - `/api/vault/analytics` - Comprehensive vault analytics
  - `/api/logs/recent` - Structured log streaming

### 2. Vault Strategy Service (Port 8081)
- ML models for vault optimization
- Gamma-style dual position strategies
- Multi-range position management

### 3. Compound Service (Port 8082)
- Automated compound opportunity detection
- AI-optimized compound execution
- Batch compound operations

### 4. Log Aggregator (Port 8084)
- Real-time log streaming
- Structured log categorization
- SSE endpoint for live updates

### 5. Monitoring Stack
- Prometheus (Port 9090) - Metrics collection
- Grafana (Port 3002) - Visualization dashboards
- DexBrain Metrics Exporter - Custom vault metrics

## Deployment Instructions

1. **Deploy to VPS:**
   ```bash
   cd backend
   ./deploy-vault-infrastructure.sh
   ```

2. **Check Service Status:**
   ```bash
   ssh root@5.78.71.231 'docker-compose -f /opt/dexter-ai/docker-compose.vault.yml ps'
   ```

3. **View Logs:**
   ```bash
   ssh root@5.78.71.231 'docker-compose -f /opt/dexter-ai/docker-compose.vault.yml logs -f'
   ```

## API Endpoints

### External Access
All endpoints are available at `https://api.dexteragent.com`:

- **Vault Intelligence:** `GET /api/vault/intelligence?vault_address=0x...`
- **Compound Opportunities:** `GET /api/vault/compound-opportunities?limit=10`
- **Vault Analytics:** `GET /api/vault/analytics?days=30`
- **Recent Logs:** `GET /api/logs/recent?limit=50&type=all`
- **Log Stream (SSE):** `GET /api/logs/stream`

### Authentication
Use Bearer token authentication with API key:
```
Authorization: Bearer YOUR_API_KEY
```

## Monitoring

### Grafana Dashboard
Access at: http://5.78.71.231:3002
- Username: admin
- Password: dexter_admin_2024

### Prometheus Metrics
Access at: http://5.78.71.231:9090

## Log Categories

The system logs are categorized for easy filtering:
- `vault_strategy` - AI strategy predictions
- `vault_optimization` - Position optimization events
- `compound_success` - Successful compound operations
- `compound_opportunities` - Opportunity detection
- `vault_intelligence` - Intelligence generation
- `gamma_optimization` - Gamma-style optimizations

## Troubleshooting

### Check Container Logs
```bash
docker-compose -f /opt/dexter-ai/docker-compose.vault.yml logs [service-name]
```

### Restart Services
```bash
docker-compose -f /opt/dexter-ai/docker-compose.vault.yml restart [service-name]
```

### View Real-time Logs
```bash
curl https://api.dexteragent.com/api/logs/recent?limit=50&type=vault_strategy
```

## Frontend Integration

The BrainWindow component automatically connects to the vault infrastructure APIs:
- Polls `/api/logs/recent` every 3 seconds
- Displays vault operations with color-coded categories
- Shows real-time compound opportunities and AI decisions