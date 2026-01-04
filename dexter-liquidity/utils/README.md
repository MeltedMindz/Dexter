# Utilities

## Purpose
The utils directory contains utility functions and helper modules for the dexter-liquidity system, including error handling, memory monitoring, performance tracking, and parallel processing.

## What Lives Here
- **error_handler.py** - Error handling and retry logic
- **memory_monitor.py** - Memory usage monitoring
- **performance.py** - Performance tracking utilities
- **cache.py** - Caching utilities
- **parallel_processor.py** - Parallel data processing
- **statistics.py** - Statistical analysis utilities
- **pool_share_calculator.py** - Pool share calculations
- **performance_tracker.py** - Performance tracking

## How It Fits Into the System
- **Interacts with**: All dexter-liquidity components (agents, execution, data)
- **Depends on**: Python standard library, system monitoring APIs
- **Provides**: Error handling, monitoring, performance tracking, caching, parallel processing
- **Part of**: The dexter-liquidity infrastructure, providing cross-cutting concerns

## Current Status
✅ **Active / In use** - Utility modules operational across the system

## What This Is NOT
- This is not the core business logic (that's in agents and execution)
- This is not the data processing (that's in `../data/`)

## Relevant Docs / Entry Points
- **Error handling**: `error_handler.py`
- **Memory monitoring**: `memory_monitor.py`
- **Performance tracking**: `performance_tracker.py`
- **Dexter-liquidity documentation**: See `../README.md`
- **Root documentation**: See `../../README.md`

