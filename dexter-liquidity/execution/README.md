# Execution

## Purpose
The execution directory contains the execution manager and configuration for strategy execution in the dexter-liquidity system. This orchestrates the execution of agent strategies and manages position operations.

## What Lives Here
- **manager.py** - Execution manager that orchestrates strategy execution
- **execution_config.py** - Execution configuration and types

## How It Fits Into the System
- **Interacts with**: Agents (`../agents/`), Web3 provider, smart contracts
- **Depends on**: Web3.py, agents, utilities
- **Provides**: Strategy execution orchestration, position management, transaction handling
- **Part of**: The dexter-liquidity system, executing the strategies determined by agents

## Current Status
✅ **Active / In use** - Execution manager operational with parallel processing

## What This Is NOT
- This is not the agents themselves (those are in `../agents/`)
- This is not the data fetchers (those are in `../data/`)

## Relevant Docs / Entry Points
- **Execution manager**: `manager.py` - `ExecutionManager` class
- **Execution config**: `execution_config.py` - `ExecutionConfig` class
- **Dexter-liquidity documentation**: See `../README.md`
- **Root documentation**: See `../../README.md`

## Execution Types
- **Event-based**: Triggered by blockchain events
- **Periodic**: Scheduled execution at intervals
- **Contract automation**: Automated contract interactions

