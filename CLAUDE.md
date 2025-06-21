# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Dexter Protocol is an advanced AI-powered liquidity management platform for decentralized exchanges (DEXs). The project has evolved into a comprehensive DeFi infrastructure with multiple integrated components:

### Core Components:
1. **Position Management System**: Advanced Uniswap V3 position management with auto-compounding
2. **AI Optimization Engine**: ML-driven strategies for yield optimization and risk management  
3. **Backend/DexBrain**: Centralized intelligence system processing data across all agents
4. **Frontend Dashboard**: Professional web interface for position tracking and management
5. **Smart Contracts**: On-chain infrastructure for position management and fee distribution

### Current Development Status (December 2024):
- ✅ **Complete Position Management System**: Professional-grade auto-compounding with AI integration
- ✅ **Advanced Smart Contracts**: DexterCompoundor.sol with TWAP protection and AI optimization
- ✅ **Production-Ready Frontend**: React/Next.js interface with real-time analytics
- ✅ **Comprehensive Data Quality System**: 4-dimensional data monitoring with auto-healing
- ✅ **Enhanced ML Pipeline**: Advanced feature engineering and performance tracking
- ✅ **Production Infrastructure**: Docker, monitoring, CI/CD, and VPS deployment

## Major Development Phases Completed (2024)

### Phase 1: Infrastructure & Testing (Q3 2024)
- **Testing Infrastructure**: Comprehensive pytest setup with integration/unit tests
- **Solana Cleanup**: Removed Solana components, focused on Base Network
- **Production Infrastructure**: Database backups, blue-green deployment, monitoring

### Phase 2: ML & Data Pipeline (Q3 2024)  
- **Enhanced ML Models**: 20+ Uniswap features, LSTM, TickRangePredictor
- **Advanced Data Pipeline**: Multi-source collection (GraphQL, RPC, events)
- **Learning Verification**: ML training validation and performance monitoring

### Phase 3: Advanced Features (Q4 2024)
- **Enhanced Uniswap Integration**: Optimal tick ranges, fee prediction  
- **Performance Tracking**: Sharpe ratio, risk metrics, comprehensive analytics
- **VPS Integration**: CI/CD, monitoring, production deployment

### Phase 4: Data Quality & Position Management (Q4 2024)
- **Data Quality System**: 4-dimensional monitoring with auto-healing workflows
- **Position Management**: Complete auto-compounding system with AI integration
- **Smart Contracts**: DexterCompoundor.sol with advanced features
- **Professional Frontend**: Production-ready React interface

### Recent Technical Improvements:
- **DexBrain Refactoring**: Modular structure with `config.py`, `blockchain/`, `models/`, `core.py`
- **Performance Tracking**: Unified implementation with Prometheus integration
- **Test Standardization**: Consistent `test_*.py` naming convention
- **Configuration Management**: Enhanced Settings class with validation

## Key Commands

### Development Environment

```bash
# Setup Python environment (Backend)
python -m venv env
source env/bin/activate  # Linux/Mac
pip install -r dexter-liquidity/requirements.txt

# Setup Frontend
cd frontend
npm install
npm run dev  # Development server

# Setup Full Stack
npm run dev  # Frontend (port 3000)
python -m main  # Backend (from dexter-liquidity/)
```

### Running the Application

```bash
# Production with Docker
sudo docker compose build
sudo docker compose up -d

# Development - Backend
cd dexter-liquidity
python -m main

# Development - Frontend  
cd frontend
npm run dev

# Run DexBrain separately
cd backend
python -m dexbrain.core

# Run Position Management
# Smart contracts in contracts/core/
# Frontend component: frontend/components/PositionManager.tsx
```

### Testing

```bash
# Run all tests
pytest

# Run unit tests only
pytest tests/unit/

# Run integration tests
pytest tests/integration/

# Run specific test file
pytest tests/unit/test_volatility.py

# Run tests with coverage
pytest --cov
```

### Code Quality

```bash
# Format code with Black
black .

# Sort imports
isort .

# Type checking
mypy .
```

## Architecture Overview

### Core Components

1. **Position Management System** (`contracts/core/`, `frontend/components/`):
   - `DexterCompoundor.sol`: Smart contract for auto-compounding with AI integration
   - `PositionManager.tsx`: Frontend interface for position tracking and management
   - **Features**: NFT position deposits, AI-optimized compounding, reward distribution
   - **Capacity**: 200 positions per address, configurable reward structures

2. **AI Agents** (`dexter-liquidity/agents/`): Risk-based trading strategies
   - `base_agent.py`: Abstract base class defining agent interface
   - `conservative.py`: Low-risk strategy ($100k min liquidity, 15% max volatility)
   - `aggressive.py`: Medium-risk strategy ($50k min liquidity, 30% max volatility)
   - `hyper_aggressive.py`: High-risk strategy ($25k min liquidity, no volatility limit)

3. **Data Infrastructure** (`dexter-liquidity/data/`): Multi-layered data collection
   - `fetchers/`: DEX integration layer (Uniswap V3/V4, base interfaces)
   - `data_quality_monitor.py`: 4-dimensional data quality scoring
   - `historical_backfill_service.py`: Intelligent gap-filling with 45-180 records/minute
   - `completeness_checker.py`: Automated missing data detection

4. **ML Pipeline** (`backend/dexbrain/`): **ENHANCED** intelligence system
   - `enhanced_ml_models.py`: 20+ features, LSTM, TickRangePredictor, DeFiMLEngine
   - `learning_verification_system.py`: Training validation and performance monitoring
   - `alchemy_position_collector.py`: Direct RPC data collection
   - `models/knowledge_base.py`: Centralized ML model storage and retrieval

5. **Frontend Dashboard** (`frontend/`): **PRODUCTION-READY** web interface
   - `components/PositionManager.tsx`: Complete position management interface
   - `components/EnhancedPortfolioOverview.tsx`: Real-time analytics dashboard
   - `lib/wagmi.ts`: Web3 integration with SSR handling
   - **Features**: AI-managed position toggles, compound interface, performance tracking

### Key Design Patterns

- **AI-First Architecture**: Every component designed with ML optimization in mind
- **Position-Centric Design**: All systems built around Uniswap V3 NFT position management
- **Auto-Compounding**: Automated fee collection and reinvestment with configurable rewards
- **Data Quality Assurance**: 4-dimensional monitoring (completeness, accuracy, consistency, timeliness)
- **Parallel Processing**: Multi-threaded execution with `parallel_processor.py`
- **Error Handling**: Circuit breakers and comprehensive error handling
- **Performance Monitoring**: Real-time metrics with Prometheus integration
- **SSR-Compatible Web3**: Proper client-side initialization for Next.js
- **Modular Smart Contracts**: Upgradeable, secure, and gas-optimized

### Database Schema

The system uses PostgreSQL for persistent storage. Key tables are defined in `backend/db/schema.sql`.

### Environment Configuration

Critical environment variables (set in `.env`):

**Blockchain & API Access:**
- `NEXT_PUBLIC_ALCHEMY_API_KEY`: Required for Base Network RPC access (Frontend)
- `ALCHEMY_API_KEY`: Backend RPC access
- `BASESCAN_API_KEY`: Block explorer API access
- `BASE_RPC_URL`: Base Network RPC endpoint
- `NEXT_PUBLIC_BASE_RPC_FALLBACK`: Public RPC fallback (https://mainnet.base.org)

**Database & Caching:**
- `DATABASE_URL`: PostgreSQL connection string
- `REDIS_PASSWORD`: Redis cache authentication

**Application Configuration:**
- `ENVIRONMENT`: Set to 'production' or 'development'
- `NODE_ENV`: Node.js environment ('production', 'development')
- `NEXT_PUBLIC_ENVIRONMENT`: Frontend environment detection

**Performance & Security:**
- `PARALLEL_WORKERS`: Number of concurrent workers (default: 4)
- `MAX_SLIPPAGE`: Maximum allowed slippage (default: 0.005)
- `NEXT_PUBLIC_WC_PROJECT_ID`: WalletConnect project ID

**Frontend Deployment (Vercel):**
- All `NEXT_PUBLIC_*` variables must be set in Vercel dashboard
- Use public RPC fallbacks for reliable operation
- See `frontend/.env.example` for complete list

### Configuration Classes

Both components now use centralized configuration:
- **DexBrain**: `backend/dexbrain/config.py` - Config class with validation
- **Liquidity System**: `dexter-liquidity/config/settings.py` - Settings class with agent configs

## Docker Services

The `docker-compose.yml` defines:
- `dexter`: Main application service
- `db`: PostgreSQL database
- `redis`: Caching layer
- `prometheus`: Metrics collection (port 9093)
- `grafana`: Visualization dashboards (port 3002)
- `node-exporter`: System metrics

## Development Notes

### Import Patterns
Use the enhanced modular structure:

**Backend/ML Imports:**
```python
# DexBrain imports
from dexbrain.core import DexBrain
from dexbrain.config import Config
from dexbrain.models import KnowledgeBase, DeFiMLEngine, EnhancedMLModels
from dexbrain.enhanced_ml_models import LSTMModel, TickRangePredictor

# Data quality imports
from data.data_quality_monitor import DataQualityMonitor, DataQualityMetrics
from data.historical_backfill_service import HistoricalBackfillService

# Liquidity system imports  
from config.settings import Settings
from utils.performance_tracker import PerformanceTracker, PerformanceMetrics
```

**Frontend Imports:**
```typescript
// Position management
import { PositionManager } from '@/components/PositionManager'
import { EnhancedPortfolioOverview } from '@/components/EnhancedPortfolioOverview'

// Web3 integration
import { useAccount, useReadContract, useWriteContract } from 'wagmi'
import { alchemyBase, getTokenBalances } from '@/lib/alchemy'

// SEO and utilities
import { generateSEOMetadata, pageSEO } from '@/lib/seo'
```

### Error Handling
All async operations should use proper error handling:
```python
try:
    async with connector as conn:
        data = await conn.fetch_liquidity_data(pool_address)
except Exception as e:
    await error_handler.handle_error(e, context="operation_name")
```

### Configuration Usage
```python
# Get agent-specific configuration
agent_config = Settings.get_agent_config('conservative')
min_liquidity = agent_config['min_liquidity']

# Validate configuration
Settings.validate()  # Raises ValueError if invalid
```

## Current Implementation Status

### ✅ Completed Features

**Smart Contract Infrastructure:**
- `DexterCompoundor.sol`: Complete auto-compounding system
- AI agent integration with bypass capabilities
- TWAP protection against MEV attacks
- Configurable reward structures (compounder + AI optimizer)
- Support for 200 positions per address

**Frontend Dashboard:**
- `PositionManager.tsx`: Professional position management interface
- Real-time analytics with APR, fees, impermanent loss tracking
- AI-managed vs manual position filtering
- One-click compounding with advanced options
- Dark mode support with clean, responsive design

**Data Infrastructure:**
- 4-dimensional data quality monitoring system
- Historical backfill service (45-180 records/minute capacity)
- Automated gap detection and healing workflows
- Comprehensive completeness checking

**ML Pipeline:**
- 20+ engineered features for Uniswap position analysis
- LSTM models for price prediction
- TickRangePredictor for optimal range selection
- Performance tracking with Sharpe ratio, max drawdown
- Learning verification and model validation

**Production Infrastructure:**
- Docker-based deployment with monitoring
- Grafana dashboards for real-time metrics
- CI/CD pipeline with automated testing
- VPS deployment with log streaming
- SEO optimization with social media cards

### 🔄 Next Development Priorities

**Immediate (Next Sprint):**
1. **Smart Contract Deployment**: Deploy DexterCompoundor to Base testnet
2. **Frontend Integration**: Connect PositionManager to actual contracts
3. **AI Integration**: Connect ML models to position optimization
4. **Testing**: Comprehensive end-to-end testing

**Short Term (1-2 weeks):**
1. **Advanced Compounding**: Implement optimal swapping logic
2. **Batch Operations**: Multi-position compound operations
3. **Analytics Enhancement**: Advanced performance metrics
4. **Mobile Optimization**: Responsive design improvements

**Medium Term (1 month):**
1. **Mainnet Deployment**: Production smart contract deployment
2. **Advanced AI Features**: Dynamic range adjustment, risk assessment
3. **Cross-Chain Support**: Extend to other networks
4. **Premium Features**: Advanced analytics, professional tools

### 🎯 Project Goals

**Primary Objective**: Launch the most advanced AI-powered position management protocol in DeFi
**Target Users**: DeFi power users, institutional liquidity providers, yield farmers
**Competitive Advantage**: AI optimization, superior UX, institutional-grade features
**Market Position**: First AI-native auto-compounding protocol

### 📊 Key Metrics to Track

**Technical Metrics:**
- Position compound frequency and success rate
- AI optimization performance vs manual
- Gas efficiency and cost savings
- Data quality scores and uptime

**Business Metrics:**
- Total Value Locked (TVL)
- Number of active positions
- User retention and engagement
- Fee revenue generation

**Performance Metrics:**
- Average APR improvement
- Impermanent loss mitigation
- Risk-adjusted returns (Sharpe ratio)
- Compound timing optimization