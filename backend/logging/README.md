# Logging

## Purpose
The logging directory contains structured logging configuration and log aggregation utilities for the Dexter Protocol. This provides comprehensive logging for vault operations, compound services, DexBrain intelligence, and system events.

## What Lives Here
- **log_config.py** - Structured logging configuration with categories and formatters
- **log_aggregator.py** - Real-time log aggregation service that streams logs to frontend components

## How It Fits Into the System
- **Interacts with**: All backend services (vault operations, compound service, DexBrain, AI models)
- **Depends on**: Python logging framework, Flask (for log aggregator API)
- **Provides**: Structured logging, log aggregation, real-time log streaming to frontend
- **Part of**: The observability infrastructure, enabling debugging and monitoring

## Current Status
✅ **Active / In use** - Structured logging operational for all backend services

## What This Is NOT
- This is not the monitoring dashboards (those are in `monitoring/`)
- This is not the metrics collection (that's Prometheus in `monitoring/`)
- This is not the log storage (logs are written to files, may be aggregated elsewhere)

## Relevant Docs / Entry Points
- **Log configuration**: `log_config.py` - `StructuredLogger` class and log categories
- **Log aggregator**: `log_aggregator.py` - `LogAggregator` class and Flask API
- **Backend documentation**: See `../README.md`
- **Root documentation**: See `../../README.md`

## Log Categories
- **Vault Operations**: Strategy selection, optimization, range management
- **Compound Operations**: Compound successes, failures, opportunities
- **AI Predictions**: ML model predictions and recommendations
- **Performance Tracking**: Performance metrics and analytics
- **System Events**: Network, system, and error events

## Key Features
- **Structured Logging**: JSON-formatted logs with metadata
- **Category-Based**: Organized by component (vault, compound, AI, etc.)
- **Real-Time Aggregation**: Stream logs to frontend BrainWindow component
- **File and Console**: Dual output for development and production

