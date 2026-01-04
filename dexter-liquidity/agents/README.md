# Agents

## Purpose
The agents directory contains trading strategy agents that implement different risk profiles for automated liquidity management. These agents evaluate liquidity pairs and make decisions about position management based on their risk tolerance.

## What Lives Here
- **base_agent.py** - Base agent interface and abstract class
- **conservative.py** - Low-risk strategy agent
- **aggressive.py** - Medium-risk strategy agent
- **hyper_aggressive.py** - High-risk strategy agent
- **types.py** - Type definitions for agents (LiquidityPair, StrategyMetrics, RiskProfile, HealthStatus)

## How It Fits Into the System
- **Interacts with**: Execution manager (`../execution/`), data fetchers (`../data/`), Web3 provider
- **Depends on**: Web3.py, data processing utilities
- **Provides**: Strategy evaluation, risk-based decision making, position recommendations
- **Part of**: The dexter-liquidity system, providing the core trading logic

## Current Status
✅ **Active / In use** - Production agents operational with multiple risk profiles

## What This Is NOT
- This is not the execution system (that's in `../execution/`)
- This is not the data fetchers (those are in `../data/`)
- This is not the utility functions (those are in `../utils/`)

## Relevant Docs / Entry Points
- **Base agent**: `base_agent.py` - `DexterAgent` abstract class
- **Strategy agents**: `conservative.py`, `aggressive.py`, `hyper_aggressive.py`
- **Dexter-liquidity documentation**: See `../README.md`
- **Root documentation**: See `../../README.md`

## Risk Profiles
- **Conservative**: Low-risk, stable pairs, wide ranges
- **Aggressive**: Medium-risk, higher volatility, tighter ranges
- **Hyper-Aggressive**: High-risk, maximum volatility, ultra-tight ranges

