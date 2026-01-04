# Data

## Purpose
The data directory contains data processing, analysis, and fetching utilities for the dexter-liquidity system. This includes DEX data fetchers, volatility calculations, and market regime detection.

## What Lives Here
- **fetchers/** - DEX data fetchers (Uniswap V4, Meteora, base interface)
- **volatility.py** - Volatility calculation utilities
- **regime_detector.py** - Market regime detection algorithms

## How It Fits Into the System
- **Interacts with**: Agents (`../agents/`), execution manager (`../execution/`)
- **Depends on**: Web3.py, data processing libraries
- **Provides**: Real-time and historical DEX data, volatility metrics, market regime classification
- **Part of**: The dexter-liquidity system, providing data for decision-making

## Current Status
✅ **Active / In use** - Data fetchers operational for Uniswap V4 and Meteora

## What This Is NOT
- This is not the backend data sources (those are in `../../backend/data_sources/`)
- This is not the streaming data (that's in `../../backend/streaming/`)

## Relevant Docs / Entry Points
- **Data fetchers**: `fetchers/` - Uniswap V4 and Meteora fetchers
- **Dexter-liquidity documentation**: See `../README.md`
- **Root documentation**: See `../../README.md`

## Supported DEXs
- **Uniswap V4**: Primary DEX integration
- **Meteora**: Additional DEX support

