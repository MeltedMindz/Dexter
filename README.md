# Dexter Protocol

**AI-Powered DeFi Liquidity Management Framework**

> *Dexter Protocol is an ambitious DeFi infrastructure project with comprehensive AI/ML pipeline design and smart contract architecture for automated liquidity management on Uniswap V3/V4.*

[![License: Source Available](https://img.shields.io/badge/License-Source%20Available-blue.svg)](LICENSE)
[![Build](https://img.shields.io/badge/Build-Passing-brightgreen)](contracts/mvp/)
[![Tests](https://img.shields.io/badge/Contract%20Tests-62%20Passing-brightgreen)](contracts/mvp/)
[![Status](https://img.shields.io/badge/Status-Development-yellow)](#current-status)
[![Website](https://img.shields.io/badge/Website-dexteragent.com-blue)](https://dexteragent.com)

## Project Status

**Development Phase** - Core infrastructure is designed and partially implemented. This repository contains:

| Component | Status | Details |
|-----------|--------|---------|
| MVP Smart Contracts | **Production-Ready** | 62 tests passing (42 unit + 20 integration), security features implemented |
| Oracle Integration | **Implemented** | IPriceAggregator + TWAPOracle for MEV protection |
| Emergency Controls | **Implemented** | Pausable pattern on all contracts |
| Backend API Structure | **Code Exists** | Flask API defined, real data fetchers available |
| ML Pipeline Design | **Code Exists** | Training infrastructure with real data pipeline option |
| Docker Infrastructure | **Defined** | Compose files exist, needs deployment verification |

**What Works Today:**
- Smart contracts compile and pass all 62 tests
- Position limit enforcement (200 per address)
- Emergency pause capability on all contracts
- TWAP protection against MEV attacks
- Price oracle integration for real fee calculations
- Environment-based credential management
- Pre-commit hooks and CI/CD pipeline

**What Needs Work:**
- ML models need production data validation
- Testnet/mainnet deployment
- End-to-end infrastructure verification
- Performance benchmarking under load

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

**Smart Contracts (Production-Ready):**
```bash
cd contracts/mvp
npm install
npm run compile  # Compiles all contracts
npm run test     # Runs 62 tests (42 unit + 20 integration)
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
├── contracts/mvp/           # Production-ready smart contracts (Hardhat)
│   ├── contracts/           # DexterMVP, BinRebalancer, UltraFrequentCompounder
│   └── test/                # 62 passing tests (unit + integration)
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

The MVP contracts are production-ready with comprehensive security features:

- **DexterMVP.sol**: Position deposit, compound, and rebalance with TWAP protection
- **BinRebalancer.sol**: Bin-based rebalancing with emergency pause capability
- **UltraFrequentCompounder.sol**: High-frequency compounding with oracle integration

**Security Features:**
- Emergency pause (Pausable) on all contracts
- TWAP oracle protection against MEV/sandwich attacks
- ReentrancyGuard on all state-changing functions
- Position limits enforced (200 per address)
- Environment-based credential management

**Capabilities:**
- Accept Uniswap V3 position NFT deposits
- Configure automation settings per position
- Execute compounds with price oracle validation
- Track position performance metrics
- Batch operations for gas efficiency

**Next Steps:**
- Testnet deployment and verification
- Production oracle integration
- Mainnet deployment after audit

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

See [RISK_REGISTER.md](docs/RISK_REGISTER.md) for complete list.

**Resolved in Recent Audit:**
- RISK-001: Fee calculations now use oracle integration (not hardcoded)
- RISK-003: Position limits enforced in depositPosition()
- RISK-005: TWAP protection integrated against MEV attacks
- RISK-006: Emergency pause implemented on all contracts
- RISK-008: Lock files committed, dependencies pinned

**Remaining Work:**
1. **ML production validation** - Training pipeline has real data path, needs verification
2. **Network deployment** - Contracts ready but not deployed to testnet/mainnet
3. **Infrastructure testing** - Docker services need end-to-end verification

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
