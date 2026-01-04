# Streaming

## Purpose
The streaming directory contains the real-time data streaming infrastructure for the Dexter Protocol, implementing Kafka-based event streaming, Apache Flink-style stream processing, and online learning capabilities. This enables continuous learning and real-time ML predictions.

## What Lives Here
- **kafka_producer.py** - Kafka producer for publishing pool events, ML predictions, and system metrics
- **kafka_consumer.py** - Kafka consumer for processing streaming events
- **flink_processor.py** - Apache Flink-style stream processor for real-time feature engineering and ML predictions
- **online_learning_engine.py** - Online learning engine for concept drift detection and adaptive model updates

## How It Fits Into the System
- **Interacts with**: DexBrain (`backend/dexbrain/`), ML models (`backend/ai/`), MLOps pipeline (`backend/mlops/`)
- **Depends on**: Kafka cluster, ML models, feature engineering pipelines
- **Provides**: Real-time event processing, ML feature generation, online learning, streaming predictions
- **Part of**: The MLOps Level 2 infrastructure, enabling continuous learning from live blockchain data

## Current Status
✅ **Active / In use** - Streaming infrastructure operational for production MLOps pipeline

## What This Is NOT
- This is not the batch data ingestion (that's in `backend/data_sources/`)
- This is not the ML training pipeline (that's in `backend/mlops/`)
- This is not the Kafka cluster itself (external infrastructure)

## Relevant Docs / Entry Points
- **Kafka producer**: `kafka_producer.py` - `DexterKafkaProducer` class
- **Kafka consumer**: `kafka_consumer.py` - `DexterKafkaConsumer` class
- **Stream processor**: `flink_processor.py` - `FlinkStreamProcessor` class
- **Online learning**: `online_learning_engine.py` - `OnlineDeFiOptimizer` class
- **Backend documentation**: See `../README.md`
- **Root documentation**: See `../../README.md`

## Kafka Topics
- **dexter.pool.events** - Uniswap V3 pool events (swaps, mints, burns)
- **dexter.ml.predictions** - ML model predictions and recommendations
- **dexter.price.updates** - Real-time price feeds and TWAP data
- **dexter.liquidity.changes** - TVL and liquidity depth changes
- **dexter.system.metrics** - System performance metrics
- **dexter.alerts** - System alerts and notifications

## Key Features
- **Real-Time Processing**: Processes 10,000+ events/second
- **Feature Engineering**: Generates 20+ engineered features for ML models
- **Online Learning**: Adaptive model updates based on concept drift detection
- **Stream Monitoring**: Performance tracking and observability

