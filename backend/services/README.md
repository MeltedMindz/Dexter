# Services

## Purpose
The services directory contains high-level service implementations that orchestrate multiple components of the Dexter Protocol. These services provide business logic and coordination between AI models, smart contracts, and data sources.

## What Lives Here
- **compound_service.py** - Advanced AI-powered compounding service that integrates with DexBrain ML models for optimal compound timing and strategy selection

## How It Fits Into the System
- **Interacts with**: DexBrain (`backend/dexbrain/`), AI models (`backend/ai/`), smart contracts, Uniswap data fetchers
- **Depends on**: Web3.py, ML models, blockchain RPC endpoints
- **Provides**: Compound opportunity detection, AI-optimized compound execution, performance tracking
- **Part of**: The service layer, providing high-level business logic for the Dexter Protocol

## Current Status
✅ **Active / In use** - Production service integrated with vault infrastructure and AI models

## What This Is NOT
- This is not the core DexBrain system (that's in `backend/dexbrain/`)
- This is not the AI models themselves (those are in `backend/ai/`)
- This is not the data fetchers (those are in `dexter-liquidity/data/`)

## Relevant Docs / Entry Points
- **Compound service**: `compound_service.py` - `CompoundService` class
- **Backend documentation**: See `../README.md`
- **Root documentation**: See `../../README.md`

## Key Features
- **AI-Optimized Compounding**: Uses ML models to determine optimal compound timing
- **Multiple Strategies**: Conservative, Balanced, Aggressive, AI-Optimized, Gas-Optimized, Fee-Maximized
- **Vault Integration**: Integrates with vault strategy models for advanced optimization
- **Performance Tracking**: Tracks compound results and APR improvements
- **Gas Optimization**: Considers gas costs in compound decisions

