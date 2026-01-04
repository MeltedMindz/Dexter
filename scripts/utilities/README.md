# Utilities

## Purpose
The utilities directory contains operational utility scripts for the Dexter Protocol, including backup systems, data collection, learning pipeline runners, and integration utilities.

## What Lives Here
- **backup_system.py** - Database backup and recovery utilities
- **demo_alchemy_collection.py** - Alchemy API data collection demo
- **enhanced_alchemy_service.py** - Enhanced Alchemy SDK integration
- **run_integrated_learning.py** - ML training pipeline runner
- **simple_learning_demo.py** - Simple ML demonstration
- **uniswap_position_harvester.py** - Uniswap position data harvester
- **vercel-env-config.txt** - Vercel environment configuration template

## How It Fits Into the System
- **Interacts with**: Backend services, databases, external APIs (Alchemy, The Graph)
- **Depends on**: Python, Web3.py, database connections
- **Provides**: Operational utilities, data collection, backup, and integration scripts
- **Part of**: The scripts directory, providing operational support

## Current Status
✅ **Active / In use** - Utility scripts operational for various operational tasks

## What This Is NOT
- This is not the core backend services (those are in `../../backend/`)
- This is not the monitoring scripts (those are in `../monitoring/`)

## Relevant Docs / Entry Points
- **Backup system**: `backup_system.py`
- **Data collection**: `demo_alchemy_collection.py`, `uniswap_position_harvester.py`
- **ML pipeline**: `run_integrated_learning.py`
- **Scripts documentation**: See `../README.md`
- **Root documentation**: See `../../README.md`

