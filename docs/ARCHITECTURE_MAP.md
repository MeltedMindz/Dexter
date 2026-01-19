# ARCHITECTURE_MAP.md - The Ralph Collective Phase 1 Audit

**Date:** 2026-01-19
**Auditor:** The Ralph Collective
**Scope:** Complete architecture mapping of Dexter Protocol repository

---

## 1. Repository Structure

```
Dexter/
├── contracts/                    # Smart Contracts (Solidity)
│   ├── mvp/                     # MVP contracts - PRIMARY BUILD TARGET
│   │   ├── contracts/           # DexterMVP, BinRebalancer, UltraFrequentCompounder
│   │   │   └── vendor/          # Vendored Uniswap interfaces
│   │   ├── test/                # JavaScript tests (17 tests)
│   │   └── hardhat.config.js    # Hardhat build config
│   ├── core/                    # Production contracts (UNBUILT)
│   ├── vaults/                  # ERC4626 vault contracts (UNBUILT)
│   ├── oracles/                 # ML validation oracles (UNBUILT)
│   ├── security/                # Security guards (UNBUILT)
│   └── [other modules]/         # Various contract modules
│
├── backend/                      # Python AI/ML Backend
│   ├── dexbrain/                # Core AI hub (28 files)
│   │   ├── api_server.py        # Flask REST API
│   │   ├── core.py              # DexBrain intelligence class
│   │   ├── models/              # ML model definitions
│   │   └── [many modules]       # Data pipeline, learning, etc.
│   ├── mlops/                   # MLOps orchestration
│   │   ├── continuous_training_orchestrator.py
│   │   └── ab_testing_framework.py
│   ├── streaming/               # Kafka/Flink processors
│   │   ├── kafka_producer.py
│   │   ├── kafka_consumer.py
│   │   ├── flink_processor.py
│   │   └── online_learning_engine.py
│   ├── ai/                      # AI models
│   ├── data_sources/            # Data ingestion
│   ├── db/                      # Database schema
│   └── tests/                   # Test files (4 files)
│
├── dexter-liquidity/            # ML Liquidity Management System
│   ├── agents/                  # Trading agents (conservative, aggressive, hyper)
│   ├── data/                    # Data processing modules
│   ├── execution/               # Execution layer
│   ├── utils/                   # Utilities (14 files)
│   ├── tests/                   # Test suite (10 files)
│   ├── uniswap-v4/              # V4 submodule (2692 files)
│   └── requirements.txt
│
├── frontend/                    # React/Next.js Frontend (MINIMAL)
│   ├── components/              # 3 components + UI folder
│   ├── pages/                   # 1 page
│   ├── lib/                     # 1 utility
│   └── mvp/                     # MVP dashboard
│
├── monitoring/                  # Prometheus/Grafana configs
├── scripts/                     # Utility scripts
├── automation/                  # Node.js keeper automation
├── docs/                        # Documentation
├── tests/                       # Root-level tests (mvp subfolder)
└── [config files]               # Makefile, Docker, CI/CD
```

---

## 2. Component Breakdown

### 2.1 Smart Contracts Layer

**Location:** `contracts/`

**Build System:**
- `contracts/mvp/`: Hardhat + npm (WORKING)
- `contracts/` (other): Foundry expected (NOT CONFIGURED)

**Components:**

| Contract | Purpose | Build Status | Test Status |
|----------|---------|--------------|-------------|
| `mvp/DexterMVP.sol` | Position deposit, compound, rebalance | Compiles | 5 tests pass |
| `mvp/BinRebalancer.sol` | Bin-based rebalancing | Compiles | 6 tests pass |
| `mvp/UltraFrequentCompounder.sol` | High-frequency compounding | Compiles | 6 tests pass |
| `core/DexterCompoundor.sol` | Production compoundor | NOT BUILT | No tests |
| `vaults/DexterVault.sol` | ERC4626 vault | NOT BUILT | No tests |
| `oracles/MLValidationOracle.sol` | ML price validation | NOT BUILT | No tests |

**Key Interfaces:**
- INonfungiblePositionManager (Uniswap V3)
- ISwapRouter
- IUniswapV3Pool

### 2.2 Backend/API Layer

**Location:** `backend/`

**Build System:** Python 3.9-3.11 + pip

**Components:**

| Module | Purpose | Status |
|--------|---------|--------|
| `dexbrain/api_server.py` | REST API (Flask) | Code exists, mock data |
| `dexbrain/core.py` | DexBrain intelligence | Code exists |
| `dexbrain/models/` | ML model definitions | Code exists |
| `mlops/continuous_training_orchestrator.py` | Training pipeline | Simulated data |
| `mlops/ab_testing_framework.py` | A/B testing | Code exists |
| `streaming/kafka_producer.py` | Kafka producer | Code exists |
| `streaming/kafka_consumer.py` | Kafka consumer | Code exists |
| `streaming/flink_processor.py` | Stream processing | Code exists |
| `streaming/online_learning_engine.py` | Online learning | Code exists |

**Endpoints (api_server.py):**
- `GET /health` - Health check
- `POST /api/register` - Agent registration
- `GET /api/intelligence` - Market intelligence
- `POST /api/submit-data` - Data submission
- `GET /api/agents` - List agents
- `GET /api/stats` - Network stats
- `POST /api/models/retrain` - Trigger retraining
- `GET /api/vault/intelligence` - Vault recommendations
- `GET /api/vault/compound-opportunities` - Compound opportunities
- `GET /api/vault/analytics` - Vault analytics
- `GET /api/logs/recent` - Recent logs

### 2.3 ML/Data Layer

**Location:** `dexter-liquidity/`, `backend/mlops/`, `backend/ai/`

**Components:**

| Module | Purpose | Data Source |
|--------|---------|-------------|
| `agents/conservative.py` | Low-risk agent | Config-based |
| `agents/aggressive.py` | Medium-risk agent | Config-based |
| `agents/hyper_aggressive.py` | High-risk agent | Config-based |
| `mlops/continuous_training_orchestrator.py` | MLOps Level 2 | Simulated |
| `ai/market_regime_detector.py` | Market classification | Not connected |
| `ai/vault_strategy_models.py` | Strategy prediction | Not connected |

**Claimed Models:**
1. Fee Predictor (RandomForestRegressor)
2. Range Optimizer (GradientBoosting)
3. Volatility Predictor (RandomForest)
4. Yield Optimizer (GradientBoosting)

**Actual Training Data:** Generated random data in `_load_training_data()`

### 2.4 Infrastructure Layer

**Location:** Root level configs

| Config | Purpose | Status |
|--------|---------|--------|
| `docker-compose.streaming.yml` | Full ML infrastructure | Not tested |
| `docker-compose.vault.yml` | Vault infrastructure | Not tested |
| `Dockerfile.streaming` | Multi-stage build | Defined |
| `Makefile` | Build orchestration | Working |
| `.github/workflows/ci-cd.yml` | CI/CD pipeline | Exists |
| `.github/workflows/test.yml` | Test automation | Exists |

---

## 3. Data Flow

### 3.1 Intended Data Flow (from documentation)

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Blockchain    │────▶│  Alchemy API    │────▶│ Kafka Producer  │
│   (Base/ETH)    │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                                                         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   DexBrain      │◀────│  Kafka Consumer │◀────│     Kafka       │
│   API Server    │     │  + Processing   │     │     Topics      │
└────────┬────────┘     └─────────────────┘     └─────────────────┘
         │                      │
         │                      ▼
         │              ┌─────────────────┐
         │              │  Flink Processor│
         │              │  (Features)     │
         │              └────────┬────────┘
         │                       │
         │                       ▼
         │              ┌─────────────────┐     ┌─────────────────┐
         │              │  ML Training    │────▶│   MLflow        │
         │              │  Pipeline       │     │   Registry      │
         │              └────────┬────────┘     └─────────────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│   REST API      │◀────│  Model Serving  │
│   (Predictions) │     │                 │
└─────────────────┘     └─────────────────┘
```

### 3.2 Actual Data Flow (from code analysis)

```
┌─────────────────┐
│   User Request  │
│   to API        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Flask API     │
│   api_server.py │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│   DexBrain      │────▶│   Mock/         │
│   core.py       │     │   Placeholder   │
└─────────────────┘     │   Data          │
                        └─────────────────┘

Note: Kafka, Flink, MLflow defined but not integrated
```

---

## 4. Onchain vs Offchain Boundaries

### Onchain (Ethereum/Base)

| Function | Contract | Status |
|----------|----------|--------|
| Position custody | DexterMVP | EXISTS |
| Fee collection | DexterMVP | EXISTS |
| Liquidity increase | DexterMVP | EXISTS |
| Position rebalance | DexterMVP | PARTIAL |
| Vault shares | DexterVault | NOT BUILT |
| TWAP validation | MLValidationOracle | NOT BUILT |

### Offchain (Backend)

| Function | Module | Status |
|----------|--------|--------|
| API gateway | api_server.py | EXISTS |
| ML prediction | core.py + models | MOCK DATA |
| Training pipeline | continuous_training_orchestrator.py | SIMULATED |
| Data ingestion | kafka_producer.py | NOT TESTED |
| Stream processing | flink_processor.py | NOT TESTED |
| A/B testing | ab_testing_framework.py | EXISTS |

### Bridge Points

1. **Keeper → Contract**: Authorized keepers call compound/rebalance
2. **API → User**: Recommendations served via REST
3. **Oracle → Contract**: Price validation (NOT IMPLEMENTED)

---

## 5. ML vs Deterministic Logic

### Deterministic Logic (Always Same Output)

| Component | Logic |
|-----------|-------|
| `shouldCompound()` | Time interval OR fee threshold |
| `shouldRebalance()` | Bins from price >= max threshold |
| `_calculateConcentratedRange()` | Fixed formula based on tick spacing |
| API authentication | JWT/API key validation |
| Rate limiting | Fixed requests per minute |

### ML/Probabilistic Logic (Variable Output)

| Component | Intended Use | Actual Status |
|-----------|--------------|---------------|
| Fee Predictor | Predict position fees | Code exists, not trained on real data |
| Range Optimizer | Optimal tick range | Code exists, not trained on real data |
| Volatility Predictor | Market regime | Code exists, not trained on real data |
| Yield Optimizer | APR maximization | Code exists, not trained on real data |
| Online Learning | Adaptive models | Code exists, not connected |

### Hybrid Logic

| Component | Deterministic | ML |
|-----------|---------------|-----|
| Compound timing | Min interval | Optimal timing prediction |
| Rebalance trigger | Max bin drift | Predicted optimal range |
| Gas optimization | Batch size limits | Optimal batch prediction |

---

## 6. External Dependencies

### Blockchain
- Uniswap V3 Position Manager
- Uniswap V3 Pools
- Base Network RPC (Alchemy)

### Infrastructure (Defined, Not Verified)
- PostgreSQL 15
- Redis 7
- Kafka (Confluent)
- Prometheus
- Grafana
- MLflow

### Python Packages (Key)
- Flask 3.0.0
- scikit-learn 1.3.0
- web3 6.11.1
- mlflow 2.5.0
- kafka-python (implied)

### Node.js Packages (contracts/mvp)
- Hardhat
- ethers v6
- OpenZeppelin Contracts v4

---

## 7. Build Dependency Graph

```
┌─────────────────────────────────────────────────────────────────┐
│                        make install                              │
└─────────────────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ install-contracts│  │ install-backend │  │install-dexter-  │
│                 │  │                 │  │liquidity        │
└────────┬────────┘  └────────┬────────┘  └────────┬────────┘
         │                    │                    │
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ npm install     │  │ pip install     │  │ pip install     │
│ (contracts/mvp) │  │ (backend)       │  │ (dexter-liquidity)│
└─────────────────┘  └─────────────────┘  └─────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                        make build                                │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│ npm run compile │
│ (contracts/mvp) │
└─────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                        make test                                 │
└─────────────────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ npm run test    │  │ pytest backend/ │  │ pytest dexter-  │
│ (17 passing)    │  │ (4 passing)     │  │ liquidity/      │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

---

## 8. Service Port Allocation

| Service | Port | Status |
|---------|------|--------|
| DexBrain API | 8080 | Defined |
| Enhanced Alchemy | 8002 | Defined |
| Kafka Producer | 8003 | Defined |
| Flink Processor | 8004 | Defined |
| Online Learning | 8005 | Defined |
| MLOps Orchestrator | 8006 | Defined |
| A/B Testing | 8007 | Defined |
| Stream Monitor | 8008 | Defined |
| Kafka | 9092 | Docker |
| Kafka UI | 8080 | Docker |
| MLflow | 5000 | Docker |
| PostgreSQL | 5432 | Docker |
| Redis | 6379 | Docker |
| Prometheus | 9090 | Docker |
| Grafana | 3001 | Docker |

---

## 9. Architecture Summary

**What Exists:**
- Well-structured monorepo with clear separation of concerns
- Functional smart contract build system (Hardhat MVP)
- Comprehensive Docker infrastructure definitions
- Backend API structure with Flask
- ML pipeline design with MLOps patterns

**What's Missing:**
- End-to-end integration between components
- Real data connections
- Production deployment evidence
- Verified infrastructure operability

**Architecture Quality:** Good design, incomplete implementation.
