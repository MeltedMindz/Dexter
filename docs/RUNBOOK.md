# RUNBOOK.md - Dexter Protocol Operations Guide

**Last Updated:** 2026-01-19
**Maintained By:** Ralph A (Systems & DevEx)

---

## Quick Start

### Prerequisites

| Requirement | Version | Check Command |
|-------------|---------|---------------|
| Node.js | 18+ | `node --version` |
| Python | 3.9+ | `python3 --version` |
| npm | 8+ | `npm --version` |
| Git | 2.0+ | `git --version` |
| Docker (optional) | 20+ | `docker --version` |
| PostgreSQL (optional) | 14+ | `psql --version` |

### Minimal Setup (Development)

```bash
# Clone repository
git clone https://github.com/MeltedMindz/Dexter.git
cd Dexter

# Copy environment file
cp .env.example .env

# For development without external services:
# Edit .env and set:
#   DEV_MODE=true
#   USE_SQLITE_FALLBACK=true
#   USE_MOCK_DATA=true

# Install dependencies
make install

# Build contracts
make build

# Run tests
make test
```

### Full Setup (All Services)

```bash
# 1. Clone and configure
git clone https://github.com/MeltedMindz/Dexter.git
cd Dexter
cp .env.example .env

# 2. Configure .env with real values:
#    - ALCHEMY_API_KEY (get from alchemy.com)
#    - DATABASE_URL (PostgreSQL connection string)
#    - REDIS_URL (if using Redis)

# 3. Install all dependencies
make install

# 4. Set up database (if using PostgreSQL)
createdb dexter
# Run migrations (when available):
# cd backend && alembic upgrade head

# 5. Build and test
make build
make test

# 6. Start services (if Docker available)
make docker-up
```

---

## Make Targets Reference

| Target | Description | Prerequisites |
|--------|-------------|---------------|
| `make install` | Install all npm and pip dependencies | Node.js, Python |
| `make build` | Compile Solidity contracts | `make install` |
| `make test` | Run all tests (contracts, backend, ML) | `make build` |
| `make test-contracts` | Run contract tests only | `make build` |
| `make test-backend` | Run backend tests only | `make install` |
| `make docker-up` | Start Docker services | Docker installed |
| `make docker-down` | Stop Docker services | Docker installed |
| `make clean` | Remove build artifacts | None |

---

## Component-Specific Commands

### Contracts (contracts/mvp/)

```bash
# Compile contracts
cd contracts/mvp
npm run compile

# Run tests
npm test

# Run tests with gas report
REPORT_GAS=true npm test

# Run Slither security scan (requires slither-analyzer)
pip install slither-analyzer
slither .

# Start local Hardhat node
npx hardhat node

# Deploy to local network
npx hardhat run scripts/deploy.js --network localhost

# Deploy to Base testnet
npx hardhat run scripts/deploy.js --network baseGoerli
```

### Backend (backend/)

```bash
cd backend

# Run tests
python3 -m pytest tests/ -v

# Run with coverage
python3 -m pytest tests/ --cov=dexbrain --cov-report=html

# Start API server (development)
python3 -m dexbrain.api_server

# Run database migrations (when available)
alembic upgrade head

# Create new migration
alembic revision --autogenerate -m "description"
```

### ML Pipeline (dexter-liquidity/)

```bash
cd dexter-liquidity

# Run tests
python3 -m pytest tests/ -v

# Start MLflow server (optional)
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns

# Run training (with real data)
python3 -m agents.train --data-source alchemy

# Run training (with mock data for development)
USE_MOCK_DATA=true python3 -m agents.train
```

---

## Environment Configuration

### Development Mode

For local development without external services:

```bash
# .env settings for development
DEV_MODE=true
USE_SQLITE_FALLBACK=true
USE_MOCK_DATA=true
```

### Production Mode

For production deployment:

```bash
# .env settings for production
DEV_MODE=false
USE_SQLITE_FALLBACK=false
USE_MOCK_DATA=false

# Required values (no defaults):
ALCHEMY_API_KEY=<your-key>
DATABASE_URL=postgresql://user:pass@host:5432/dexter
REDIS_URL=redis://:password@host:6379
```

---

## Database Setup

### PostgreSQL (Production)

```bash
# Create database
createdb dexter

# Create user (if needed)
createuser dexter_user
psql -c "ALTER USER dexter_user WITH PASSWORD 'your_password';"
psql -c "GRANT ALL PRIVILEGES ON DATABASE dexter TO dexter_user;"

# Run migrations
cd backend
alembic upgrade head
```

### SQLite (Development)

```bash
# Set in .env
USE_SQLITE_FALLBACK=true

# Database file will be created automatically at:
# backend/dexter_dev.db
```

---

## MLflow Setup

```bash
# Install MLflow
pip install mlflow

# Start tracking server
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 0.0.0.0 \
  --port 5000

# Access UI at http://localhost:5000
```

---

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| `ModuleNotFoundError: solana_connector` | Module removed from scope | Check test expects skip for this |
| `Connection refused (PostgreSQL)` | DB not running | Start PostgreSQL or set `USE_SQLITE_FALLBACK=true` |
| `hardhat: command not found` | Not in contracts/mvp | `cd contracts/mvp && npm install` |
| `python: command not found` | System uses python3 | Use `python3` or create alias |
| Docker socket error | Docker not running | Start Docker Desktop |

### Test Failures

```bash
# If backend tests fail with import errors:
# 1. Check Python path
export PYTHONPATH=$PWD/backend:$PWD/dexter-liquidity:$PYTHONPATH

# 2. Install missing dependencies
pip install -r backend/requirements.txt
pip install -r dexter-liquidity/requirements.txt

# 3. Run specific test for debugging
python3 -m pytest backend/tests/test_imports.py -v
```

---

## Monitoring (When Available)

### Prometheus

```bash
# Access at http://localhost:9090
# Metrics endpoint: http://localhost:8080/metrics
```

### Grafana

```bash
# Access at http://localhost:3002
# Default credentials: admin / (see GRAFANA_ADMIN_PASSWORD in .env)
```

---

## Deployment

### Local Hardhat Network

```bash
# Terminal 1: Start node
cd contracts/mvp
npx hardhat node

# Terminal 2: Deploy
npx hardhat run scripts/deploy.js --network localhost
```

### Base Testnet

```bash
# Ensure .env has:
# - ALCHEMY_API_KEY
# - PRIVATE_KEY (testnet wallet with Base Goerli ETH)

cd contracts/mvp
npx hardhat run scripts/deploy.js --network baseGoerli
```

---

## Verification Commands

Run these to verify setup is complete:

```bash
# Full verification
make install && make build && make test

# Check contract compilation
cd contracts/mvp && npm run compile

# Check backend imports
cd backend && python3 -c "import dexbrain.config; print('OK')"

# Check ML imports
cd dexter-liquidity && python3 -c "import agents; print('OK')"
```

---

## Contact & Support

- **Repository**: https://github.com/MeltedMindz/Dexter
- **Issues**: https://github.com/MeltedMindz/Dexter/issues
- **Documentation**: See `docs/` directory
