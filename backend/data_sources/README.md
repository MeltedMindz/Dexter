# Data Sources

## Purpose
The data_sources directory contains scripts and utilities for ingesting historical and real-time data from various blockchain and DeFi data sources. This data feeds into DexBrain for ML model training and knowledge base updates.

## What Lives Here
- **dexbrain_data_ingestion.py** - Main script for ingesting historical position data into DexBrain
- **historical_position_fetcher.py** - Fetches closed liquidity positions from The Graph, blockchain, and APIs
- **ingestion_fix.py** - Utility scripts for data ingestion fixes

## How It Fits Into the System
- **Interacts with**: DexBrain (`backend/dexbrain/`), The Graph subgraphs, Alchemy API, blockchain RPC endpoints
- **Depends on**: Web3.py, aiohttp, database connections
- **Provides**: Historical position data, market data, pool metrics for ML training
- **Part of**: The data pipeline infrastructure, feeding the ML training system

## Current Status
✅ **Active / In use** - Data ingestion scripts operational for Base network

## What This Is NOT
- This is not the real-time data streaming (that's in `backend/streaming/`)
- This is not the ML training pipeline (that's in `backend/mlops/`)
- This is not the DexBrain core (that's in `backend/dexbrain/`)

## Relevant Docs / Entry Points
- **Main ingestion**: `dexbrain_data_ingestion.py` - `DexBrainDataIngestion` class
- **Position fetcher**: `historical_position_fetcher.py` - `HistoricalPositionFetcher` class
- **Backend documentation**: See `../README.md`
- **Root documentation**: See `../../README.md`

## Data Sources
- **The Graph**: Uniswap V3 subgraphs for Base network
- **Alchemy API**: Historical blockchain data
- **Direct RPC**: Real-time blockchain queries
- **Messari**: Additional DeFi metrics

## Usage
```bash
# Ingest historical data
python data_sources/dexbrain_data_ingestion.py --days-back 7 --limit 500

# Check ingestion status
python data_sources/dexbrain_data_ingestion.py --status-only
```

