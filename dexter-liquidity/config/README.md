# Configuration

## Purpose
The config directory contains configuration management for the dexter-liquidity system, including environment settings, execution parameters, and system configuration.

## What Lives Here
- **settings.py** - Configuration settings and environment management

## How It Fits Into the System
- **Interacts with**: All dexter-liquidity components (agents, execution, data)
- **Depends on**: Environment variables, configuration files
- **Provides**: Centralized configuration management for the system
- **Part of**: The dexter-liquidity infrastructure, enabling configurable system behavior

## Current Status
✅ **Active / In use** - Configuration system operational

## What This Is NOT
- This is not the execution configuration (that's in `../execution/execution_config.py`)
- This is not the environment files (those are `.env` files in the root)

## Relevant Docs / Entry Points
- **Settings**: `settings.py` - `Settings` class
- **Dexter-liquidity documentation**: See `../README.md`
- **Root documentation**: See `../../README.md`

