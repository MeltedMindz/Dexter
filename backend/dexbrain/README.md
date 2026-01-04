# DexBrain

## Purpose
DexBrain is the core intelligence hub of the Dexter Protocol, serving as a centralized knowledge base and AI coordination system. It aggregates data from all deployed Dexter agents, processes it through machine learning models, and provides shared intelligence for superior decision-making in DeFi.

## What Lives Here
- **core.py** - Main DexBrain orchestrator and entry point
- **api_server.py** - RESTful API server for the global intelligence network
- **config.py** - Centralized configuration management
- **models/** - ML models and knowledge base implementations
- **blockchain/** - Blockchain connectors (Solana, EVM chains)
- **auth.py** - API key management and authentication
- **agent_registry.py** - Agent registration and tracking
- **performance_scoring.py** - Agent performance evaluation
- **data_quality.py** - Data quality validation engine

## How It Fits Into the System
- **Interacts with**: All backend services, smart contracts, and external data sources
- **Depends on**: Database (PostgreSQL), Redis cache, blockchain RPC endpoints
- **Provides**: Centralized intelligence, ML model inference, knowledge base access, agent coordination
- **Part of**: The core backend infrastructure, enabling collective learning across all Dexter agents

## Current Status
✅ **Active / In use** - Production system running with vault infrastructure integration

## What This Is NOT
- This is not the individual AI models (those are in `backend/ai/`)
- This is not the data ingestion scripts (those are in `backend/data_sources/`)
- This is not the ML training pipeline (that's in `backend/mlops/`)

## Relevant Docs / Entry Points
- **Main entry point**: `core.py` - `DexBrain` class
- **API server**: `api_server.py` - Flask REST API
- **Configuration**: `config.py` - Environment-based config
- **Backend documentation**: See `../README.md`
- **Root documentation**: See `../../README.md`

## Key Features
- **Shared Knowledge Database**: Collective intelligence from all agents
- **ML Model Integration**: LSTM, TickRangePredictor, and DeFi ML engine
- **Agent Registry**: Track and manage all registered Dexter agents
- **Performance Scoring**: Evaluate agent performance and data quality
- **Vault Integration**: Optional integration with vault strategy models

