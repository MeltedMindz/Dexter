# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Dexter is an AI-powered liquidity management system for decentralized exchanges (DEXs). The project consists of two main components:

1. **dexter-liquidity**: The core liquidity management system that deploys multiple risk-based agents (Conservative, Aggressive, Hyper-Aggressive) to optimize liquidity positions on Base Network DEXs
2. **backend/dexbrain**: A shared knowledge hub that aggregates and processes data across all Dexter AI agents

## Recent Architecture Improvements (2024)

The codebase has been significantly refactored to improve maintainability and organization:

### DexBrain Refactoring
- **hub.py** was a god file containing multiple unrelated classes
- **Refactored into modular structure**:
  - `config.py`: Centralized configuration with validation
  - `blockchain/`: Blockchain connector interfaces and implementations
  - `models/`: ML models and knowledge base management  
  - `core.py`: Main DexBrain orchestrator with proper async patterns

### Performance Tracker Fix
- **Fixed duplicate PerformanceTracker class** in `performance_tracker.py`
- **Unified implementation** with comprehensive metrics (APR, Sharpe ratio, max drawdown, win rate)
- **Added Prometheus integration** for real-time monitoring

### Test Standardization
- **Renamed test files** to follow consistent `test_*.py` convention:
  - `fetcher-tests.py` → `test_fetchers.py`
  - `agent-tests.py` → `test_agents.py` 
  - `base-fetcher-tests.py` → `test_base_fetcher.py`

### Configuration Management
- **Enhanced Settings class** in `dexter-liquidity/config/settings.py`
- **Environment-based configuration** with validation
- **Agent-specific configs** with risk parameters
- **Production/development mode detection**

## Key Commands

### Development Environment

```bash
# Setup Python environment
python -m venv env
source env/bin/activate  # Linux/Mac
.\env\Scripts\activate    # Windows

# Install dependencies
pip install -r dexter-liquidity/requirements.txt
```

### Running the Application

```bash
# Using Docker (Production)
sudo docker compose build
sudo docker compose up -d

# Manual run (Development) 
cd dexter-liquidity
python -m main

# Run DexBrain separately
cd backend
python -m dexbrain.core
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

1. **Agents** (`dexter-liquidity/agents/`): Risk-based trading strategies
   - `base_agent.py`: Abstract base class defining agent interface
   - `conservative.py`: Low-risk strategy ($100k min liquidity, 15% max volatility)
   - `aggressive.py`: Medium-risk strategy ($50k min liquidity, 30% max volatility)
   - `hyper_aggressive.py`: High-risk strategy ($25k min liquidity, no volatility limit)

2. **Data Fetchers** (`dexter-liquidity/data/fetchers/`): DEX integration layer
   - `base_interface.py`: Common interface for all DEX fetchers
   - `uniswap_v4_fetcher.py`: Uniswap V4 integration
   - `meteora_fetcher.py`: Meteora protocol integration

3. **Execution Engine** (`dexter-liquidity/execution/`): Strategy execution and management
   - `manager.py`: Orchestrates agent execution with parallel processing
   - `execution_config.py`: Configuration for execution parameters

4. **DexBrain** (`backend/dexbrain/`): **REFACTORED** shared intelligence system
   - `core.py`: Main DexBrain orchestrator with proper shutdown handling
   - `config.py`: Centralized configuration management  
   - `blockchain/`: Modular blockchain connectors with async context managers
   - `models/`: ML models and async knowledge base with file locking

### Key Design Patterns

- **Parallel Processing**: The system uses `parallel_processor.py` for concurrent execution of multiple agents
- **Error Handling**: Comprehensive error handling through `error_handler.py` with circuit breakers
- **Caching**: Memory-efficient caching via `cache.py` for API responses
- **Performance Monitoring**: **Enhanced** real-time metrics with unified PerformanceTracker
- **Async Context Managers**: Blockchain connectors use proper async patterns for connection management
- **Configuration Validation**: Settings classes validate configuration on startup

### Database Schema

The system uses PostgreSQL for persistent storage. Key tables are defined in `backend/db/schema.sql`.

### Environment Configuration

Critical environment variables (set in `.env`):
- `ALCHEMY_API_KEY`: Required for Base Network RPC access
- `BASESCAN_API_KEY`: Required for production deployment
- `BASE_RPC_URL`: Base Network RPC endpoint
- `DATABASE_URL`: PostgreSQL connection string
- `REDIS_PASSWORD`: Redis cache authentication
- `ENVIRONMENT`: Set to 'production' or 'development'
- `PARALLEL_WORKERS`: Number of concurrent workers (default: 4)
- `MAX_SLIPPAGE`: Maximum allowed slippage (default: 0.005)

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
Use the new modular structure:
```python
# DexBrain imports
from dexbrain.core import DexBrain
from dexbrain.config import Config
from dexbrain.blockchain import SolanaConnector
from dexbrain.models import KnowledgeBase, DeFiMLEngine

# Liquidity system imports  
from config.settings import Settings
from utils.performance_tracker import PerformanceTracker, PerformanceMetrics
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