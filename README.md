# Dexter Protocol

**AI-Powered DeFi Liquidity Management Framework**

> *Dexter Protocol is an ambitious DeFi infrastructure project with comprehensive AI/ML pipeline design and smart contract architecture for automated liquidity management on Uniswap V3/V4.*

[![License: Source Available](https://img.shields.io/badge/License-Source%20Available-blue.svg)](LICENSE)
[![Build](https://img.shields.io/badge/Build-Passing-brightgreen)](contracts/mvp/)
[![Tests](https://img.shields.io/badge/Contract%20Tests-17%20Passing-brightgreen)](contracts/mvp/)
[![Status](https://img.shields.io/badge/Status-Development-yellow)](#current-status)
[![Website](https://img.shields.io/badge/Website-dexteragent.com-blue)](https://dexteragent.com)

## Project Status

**Development Phase** - Core infrastructure is designed and partially implemented. This repository contains:

| Component | Status | Details |
|-----------|--------|---------|
| MVP Smart Contracts | **Compiles + Tests Pass** | 17 tests passing, core logic implemented |
| Backend API Structure | **Code Exists** | Flask API defined, endpoints return placeholder data |
| ML Pipeline Design | **Code Exists** | Training infrastructure defined, uses simulated data |
| Docker Infrastructure | **Defined** | Compose files exist, not verified operational |
| Production Deployment | **Unverified** | External infrastructure claims not verifiable from code |

**What Works Today:**
- Smart contracts compile and pass all 17 tests
- Position limit enforcement implemented
- Basic API structure exists
- Development environment can be set up

**What Needs Work:**
- ML models need real blockchain data connection
- API endpoints need real data sources
- Infrastructure needs operational verification
- Documentation accuracy needs improvement

## What Dexter Aims to Be

Dexter Protocol is designed as an AI-powered liquidity management system that:

1. **Automates Position Management**: Deposit Uniswap V3 positions for automated compounding and rebalancing
2. **Uses ML for Optimization**: Machine learning models to predict optimal timing and parameters
3. **Provides Institutional Infrastructure**: ERC4626 vault standard for institutional DeFi adoption
4. **Optimizes Gas Costs**: Batch operations for multiple positions

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              Website (dexter-website repo)                 │
│        https://github.com/MeltedMindz/dexter-website        │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────┴───────────────────────────────┐
│                 Smart Contracts (Solidity)                 │
│  ┌─────────────────┐ ┌─────────────────┐ ┌──────────────┐   │
│  │  DexterMVP      │ │  BinRebalancer  │ │ Compounder   │   │
│  │  (Working)      │ │  (Working)      │ │ (Working)    │   │
│  └─────────────────┘ └─────────────────┘ └──────────────┘   │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────┴───────────────────────────────┐
│                   Backend (Python)                         │
│  ┌─────────────────┐ ┌─────────────────┐ ┌──────────────┐   │
│  │ DexBrain API    │ │ ML Pipeline     │ │  Data Layer  │   │
│  │ (Structure OK)  │ │ (Needs Data)    │ │ (Needs Data) │   │
│  └─────────────────┘ └─────────────────┘ └──────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Getting Started

### Prerequisites
- Node.js 18+ (for contracts)
- Python 3.11 (for backend)
- npm/yarn

### Quick Start

```bash
# Clone repository
git clone https://github.com/MeltedMindz/Dexter.git
cd Dexter

# Set up environment
cp .env.example .env

# Install and build
make install
make build
make test
```

### Component Setup

**Smart Contracts (Working):**
```bash
cd contracts/mvp
npm install
npm run compile  # Compiles all contracts
npm run test     # Runs 17 tests
```

**Backend (Structure Exists):**
```bash
cd backend
pip install -r requirements.txt
python -m pytest tests/  # Runs smoke tests
```

**Dexter-Liquidity (ML System):**
```bash
cd dexter-liquidity
pip install -r requirements.txt
pytest tests/
```

## Repository Structure

```
Dexter/
├── contracts/mvp/           # Working smart contracts (Hardhat)
│   ├── contracts/           # DexterMVP, BinRebalancer, etc.
│   └── test/                # 17 passing tests
├── backend/                 # Python backend (Flask API)
│   ├── dexbrain/            # API server and ML models
│   ├── mlops/               # Training orchestration
│   └── streaming/           # Kafka processors (defined)
├── dexter-liquidity/        # ML liquidity system
│   ├── agents/              # Trading agents
│   └── tests/               # Test suite
├── docs/                    # Documentation
│   ├── SYSTEM_INTENT.md     # What Dexter aims to be
│   ├── ARCHITECTURE_MAP.md  # Component breakdown
│   ├── GAP_ANALYSIS.md      # Claims vs reality
│   └── RISK_REGISTER.md     # Known issues
└── frontend/                # Minimal UI (main site separate)
```

## Smart Contracts

The MVP contracts are the most complete part of the system:

- **DexterMVP.sol**: Position deposit, compound, and rebalance logic
- **BinRebalancer.sol**: Bin-based rebalancing for concentrated liquidity
- **UltraFrequentCompounder.sol**: High-frequency fee compounding

**Capabilities:**
- Accept Uniswap V3 position NFT deposits
- Configure automation settings per position
- Execute compounds when conditions met
- Track position performance metrics
- Enforce position limits (200 per address)

**Current Limitations:**
- Fee calculation returns placeholder values
- Needs oracle integration for production
- Not deployed to any network yet

## ML Pipeline

The ML pipeline design exists but uses simulated data:

```python
# Training pipeline exists in backend/mlops/
# Currently generates random data for development
# Needs connection to real blockchain data sources
```

**Designed Models:**
- Fee Predictor
- Range Optimizer
- Volatility Predictor
- Yield Optimizer

**TODO:**
- Connect to real Alchemy/blockchain data
- Validate model accuracy on real data
- Deploy MLflow for experiment tracking

## Documentation

**Phase 1 Audit Documents:**
- [SYSTEM_INTENT.md](docs/SYSTEM_INTENT.md) - What Dexter is trying to be
- [ARCHITECTURE_MAP.md](docs/ARCHITECTURE_MAP.md) - Component breakdown
- [GAP_ANALYSIS.md](docs/GAP_ANALYSIS.md) - Claims vs reality analysis
- [RISK_REGISTER.md](docs/RISK_REGISTER.md) - Known risks and issues

**Technical Docs:**
- [Backend README](backend/README.md) - API documentation
- [Contract Docs](contracts/mvp/README.md) - Smart contract specs

## Known Issues

See [RISK_REGISTER.md](docs/RISK_REGISTER.md) for complete list. Key issues:

1. **ML uses simulated data** - Training pipeline needs real blockchain data
2. **API returns mock data** - Several endpoints return hardcoded values
3. **Contracts not deployed** - Code exists but not on any network
4. **Infrastructure unverified** - Docker services not tested end-to-end

## Technology Stack

- **Contracts**: Solidity ^0.8.19, Hardhat, OpenZeppelin
- **Backend**: Python 3.11, Flask, scikit-learn
- **Infrastructure**: Docker, Kafka, PostgreSQL, Redis (defined)
- **Testing**: Hardhat (contracts), pytest (Python)

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Key areas needing help:
1. Connecting ML pipeline to real data
2. Expanding test coverage
3. Implementing missing contract functions
4. Verifying Docker infrastructure

## License

Source Available License - see [LICENSE](LICENSE) for details.
- View and learn from the code
- Non-commercial use permitted
- Commercial use requires separate agreement

## Links

- **Website**: [dexteragent.com](https://dexteragent.com)
- **Frontend Repo**: [dexter-website](https://github.com/MeltedMindz/dexter-website)
- **Twitter**: [@Dexter_AI_](https://x.com/Dexter_AI_)

---

*This README reflects the actual current state of the codebase. For the vision of what Dexter aims to become, see [SYSTEM_INTENT.md](docs/SYSTEM_INTENT.md).*
