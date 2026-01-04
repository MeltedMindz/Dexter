# AI

## Purpose
The ai directory contains specialized AI models and services for vault strategy optimization, market regime detection, and arbitrage detection. These models extend the core DexBrain ML capabilities with vault-specific features and advanced market analysis.

## What Lives Here
- **vault_strategy_models.py** - Vault strategy optimization models (Gamma-style, AI-optimized, hybrid)
- **market_regime_detector.py** - Advanced market regime classification using ML
- **arbitrage_detector.py** - Cross-chain and cross-DEX arbitrage opportunity detection
- **error_handler.py** - AI-powered error classification and recovery strategies

## How It Fits Into the System
- **Interacts with**: DexBrain core (`backend/dexbrain/`), vault services (`backend/services/`), compound service
- **Depends on**: PyTorch, scikit-learn, and other ML libraries
- **Provides**: Strategy recommendations, market analysis, arbitrage signals, error recovery
- **Part of**: The AI/ML infrastructure, extending core ML capabilities with specialized models

## Current Status
✅ **Active / In use** - Production models integrated with vault infrastructure

## What This Is NOT
- This is not the core DexBrain system (that's in `backend/dexbrain/`)
- This is not the ML training pipeline (that's in `backend/mlops/`)
- This is not the data sources (those are in `backend/data_sources/`)

## Relevant Docs / Entry Points
- **Vault strategies**: `vault_strategy_models.py` - `VaultMLEngine` class
- **Market analysis**: `market_regime_detector.py` - `MarketRegimeDetector` class
- **Arbitrage**: `arbitrage_detector.py` - `ArbitrageDetector` class
- **Backend documentation**: See `../README.md`
- **Root documentation**: See `../../README.md`

## Key Models
- **VaultStrategyPredictor**: Neural network for optimal strategy selection
- **GammaStyleOptimizer**: Dual-position optimization inspired by Gamma Strategies
- **MultiRangeOptimizer**: Advanced multi-range position management
- **MarketRegimeDetector**: Real-time market condition classification
- **ArbitrageDetector**: Cross-chain and cross-DEX opportunity detection

