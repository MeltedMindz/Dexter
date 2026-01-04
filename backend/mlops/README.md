# MLOps

## Purpose
The mlops directory contains the MLOps Level 2 continuous learning infrastructure for the Dexter Protocol. This includes automated training pipelines, model registry, performance monitoring, A/B testing framework, and continuous deployment capabilities.

## What Lives Here
- **continuous_training_orchestrator.py** - Automated training pipeline with MLflow integration
- **ab_testing_framework.py** - Statistical A/B testing framework for strategy comparison
- **model_registry.py** - Model versioning and registry management
- **performance_monitor.py** - Model performance monitoring and drift detection

## How It Fits Into the System
- **Interacts with**: DexBrain (`backend/dexbrain/`), streaming infrastructure (`backend/streaming/`), AI models (`backend/ai/`)
- **Depends on**: MLflow, scikit-learn, PyTorch, Kafka streaming
- **Provides**: Automated model training, A/B testing, model versioning, performance monitoring
- **Part of**: The MLOps Level 2 infrastructure, enabling continuous learning and model improvement

## Current Status
✅ **Active / In use** - MLOps pipeline operational with automated training every 30 minutes

## What This Is NOT
- This is not the ML models themselves (those are in `backend/ai/` and `backend/dexbrain/models/`)
- This is not the data ingestion (that's in `backend/data_sources/`)
- This is not the streaming infrastructure (that's in `backend/streaming/`)

## Relevant Docs / Entry Points
- **Training orchestrator**: `continuous_training_orchestrator.py` - `ContinuousTrainingOrchestrator` class
- **A/B testing**: `ab_testing_framework.py` - Statistical testing framework
- **Model registry**: `model_registry.py` - Model versioning
- **Performance monitor**: `performance_monitor.py` - Model monitoring
- **Backend documentation**: See `../README.md`
- **Root documentation**: See `../../README.md`

## Key Features
- **Automated Training**: Triggers training based on schedule, performance degradation, concept drift, or data volume
- **MLflow Integration**: Experiment tracking, model registry, and deployment management
- **A/B Testing**: Statistical comparison of control vs treatment strategies
- **Performance Monitoring**: Tracks model accuracy, drift detection, and automatic rollback
- **Continuous Deployment**: Hot-swappable model updates with zero downtime

## Training Triggers
- **Scheduled**: Daily training runs
- **Performance Degradation**: Triggered when model performance drops >5%
- **Concept Drift**: Triggered when distribution shifts >10%
- **Data Volume**: Triggered when 10k+ new samples are available

