# Phase 2 Deployment Guide: MLOps Level 2 Continuous Learning Pipeline

## Overview

This guide covers the deployment and operation of the complete AI Learning Pipelines and VPS-to-Liquidity Pool Communication Architecture implemented in Phase 2.

## 🏗️ Architecture Components

### Core Infrastructure
- **Kafka Streaming**: Real-time data ingestion and processing
- **Apache Flink**: Stream processing and feature engineering
- **River ML**: Online learning with concept drift detection
- **MLflow**: Experiment tracking and model registry
- **A/B Testing**: Statistical testing framework for strategy performance

### Services Deployed
1. `kafka-producer` - Real-time data streaming
2. `kafka-consumer` - Stream processing (2 replicas)
3. `flink-processor` - Feature engineering and ML inference
4. `online-learning` - Continuous model adaptation
5. `continuous-training` - MLOps orchestration
6. `ab-testing` - Experiment management
7. `mlflow` - Model tracking and registry
8. `stream-monitor` - System monitoring

## 🚀 Quick Start

### Prerequisites
```bash
# Ensure Docker and Docker Compose are installed
docker --version
docker-compose --version

# Set environment variables
export DB_PASSWORD="your_secure_password"
export REDIS_PASSWORD="your_redis_password"
export GRAFANA_PASSWORD="your_grafana_password"
```

### 1. Deploy Infrastructure
```bash
# Start the complete streaming infrastructure
docker-compose -f docker-compose.streaming.yml up -d

# Verify all services are healthy
docker-compose -f docker-compose.streaming.yml ps

# Check logs for any issues
docker-compose -f docker-compose.streaming.yml logs -f
```

### 2. Initialize Kafka Topics
```bash
# Create Kafka topics for streaming
docker exec dexter-kafka kafka-topics --create \
  --bootstrap-server localhost:9092 \
  --topic dexter.pool.events \
  --partitions 6 \
  --replication-factor 1

docker exec dexter-kafka kafka-topics --create \
  --bootstrap-server localhost:9092 \
  --topic dexter.ml.predictions \
  --partitions 6 \
  --replication-factor 1

docker exec dexter-kafka kafka-topics --create \
  --bootstrap-server localhost:9092 \
  --topic dexter.price.updates \
  --partitions 3 \
  --replication-factor 1

docker exec dexter-kafka kafka-topics --create \
  --bootstrap-server localhost:9092 \
  --topic dexter.liquidity.changes \
  --partitions 3 \
  --replication-factor 1

# Verify topics created
docker exec dexter-kafka kafka-topics --list --bootstrap-server localhost:9092
```

### 3. Access Web Interfaces

| Service | URL | Purpose |
|---------|-----|---------|
| Kafka UI | http://localhost:8080 | Kafka cluster management |
| MLflow | http://localhost:5000 | Experiment tracking |
| Grafana | http://localhost:3001 | Monitoring dashboards |
| Prometheus | http://localhost:9090 | Metrics collection |
| A/B Testing API | http://localhost:8007 | Experiment management |
| Stream Monitor | http://localhost:8008/metrics | Service metrics |

## 📊 Monitoring & Observability

### Health Checks
```bash
# Check service health
curl http://localhost:8003/health  # Kafka Producer
curl http://localhost:8004/health  # Flink Processor
curl http://localhost:8005/health  # Online Learning
curl http://localhost:8006/health  # MLOps Orchestrator
curl http://localhost:8007/health  # A/B Testing
curl http://localhost:8008/health  # Stream Monitor
```

### Service Metrics
```bash
# Get Prometheus metrics
curl http://localhost:8008/metrics

# Check Kafka topics
docker exec dexter-kafka kafka-run-class kafka.tools.GetOffsetShell \
  --broker-list localhost:9092 \
  --topic dexter.pool.events

# Monitor consumer lag
docker exec dexter-kafka kafka-consumer-groups \
  --bootstrap-server localhost:9092 \
  --describe \
  --group dexter_ml_pipeline
```

### Log Aggregation
```bash
# Follow logs from all streaming services
docker-compose -f docker-compose.streaming.yml logs -f kafka-producer kafka-consumer flink-processor

# Check specific service logs
docker logs dexter-online-learning --tail 100 -f
docker logs dexter-continuous-training --tail 100 -f
```

## 🧪 Testing the Pipeline

### 1. Generate Test Data
```python
import asyncio
import time
from backend.streaming.kafka_producer import DexterKafkaProducer, PoolDataEvent

async def generate_test_data():
    producer = await DexterKafkaProducer().start()
    
    # Simulate pool events
    for i in range(100):
        event = PoolDataEvent(
            pool_address="0x1234567890123456789012345678901234567890",
            timestamp=int(time.time()),
            event_type="swap",
            token0_symbol="USDC",
            token1_symbol="ETH",
            fee_tier=3000,
            liquidity=1000000 + i * 1000,
            tick=200000 + i,
            price=3000 + i * 0.1,
            volume_usd=50000.0,
            fees_usd=150.0,
            tvl_usd=5000000.0,
            transaction_hash=f"0xhash{i}",
            block_number=12345678 + i
        )
        
        await producer.send_pool_event(event)
        await asyncio.sleep(1)  # 1 event per second
    
    await producer.stop()

# Run test
asyncio.run(generate_test_data())
```

### 2. Verify ML Pipeline
```bash
# Check online learning metrics
curl http://localhost:8005/metrics

# Check MLflow experiments
curl http://localhost:5000/api/2.0/mlflow/experiments/list

# Monitor A/B testing
curl http://localhost:8007/experiments
```

### 3. Create A/B Test
```python
import asyncio
from backend.mlops.ab_testing_framework import *
from datetime import datetime

async def create_test_experiment():
    framework = ABTestingFramework()
    
    # Define variants
    variants = [
        ExperimentVariant(
            variant_id="control_v1",
            variant_type=VariantType.CONTROL,
            name="Baseline Strategy",
            description="Current auto-compounding strategy",
            traffic_allocation=0.5,
            model_config={"model": "baseline"},
            strategy_parameters={"compound_frequency": "daily"}
        ),
        ExperimentVariant(
            variant_id="treatment_v1",
            variant_type=VariantType.TREATMENT,
            name="AI-Enhanced Strategy",
            description="ML-optimized strategy with dynamic parameters",
            traffic_allocation=0.5,
            model_config={"model": "enhanced_ml"},
            strategy_parameters={"compound_frequency": "dynamic"}
        )
    ]
    
    # Define primary metric
    primary_metric = ExperimentMetric(
        metric_name="daily_return",
        metric_type="continuous",
        aggregation="mean",
        higher_is_better=True,
        minimum_detectable_effect=0.02  # 2% improvement
    )
    
    # Create experiment
    config = ExperimentConfig(
        experiment_id="strategy_optimization_001",
        name="AI Strategy vs Baseline Performance",
        description="Compare AI-optimized strategy against current baseline",
        hypothesis="AI-optimized strategy will improve daily returns by 2%",
        variants=variants,
        primary_metric=primary_metric,
        secondary_metrics=[],
        target_audience={},
        sample_size_per_variant=1000,
        duration_days=30,
        created_by="system",
        created_at=datetime.now()
    )
    
    exp_id = await framework.create_experiment(config)
    await framework.start_experiment(exp_id)
    
    print(f"Experiment created and started: {exp_id}")

# Run test
asyncio.run(create_test_experiment())
```

## 🔧 Configuration

### Environment Variables
```bash
# Required environment variables
export KAFKA_BOOTSTRAP_SERVERS="localhost:9092"
export REDIS_URL="redis://localhost:6379"
export DATABASE_URL="postgresql://dexter:password@localhost:5432/dexter_production"
export MLFLOW_TRACKING_URI="http://localhost:5000"

# Optional performance tuning
export KAFKA_BATCH_SIZE="1000"
export KAFKA_LINGER_MS="100"
export ML_UPDATE_INTERVAL="60"
export TRAINING_CHECK_INTERVAL="300"
```

### Service Configuration
Each service can be configured via environment variables:

#### Kafka Producer
- `KAFKA_BATCH_SIZE`: Batch size for message sending (default: 1000)
- `KAFKA_LINGER_MS`: Time to wait for batching (default: 100ms)
- `KAFKA_MAX_REQUEST_SIZE`: Maximum message size (default: 1MB)

#### Online Learning
- `ML_UPDATE_INTERVAL`: Model update frequency in seconds (default: 60)
- `MODEL_STORAGE_PATH`: Path for model persistence
- `CONCEPT_DRIFT_THRESHOLD`: Drift detection threshold (default: 0.1)

#### Continuous Training
- `TRAINING_DATA_PATH`: Path for training datasets
- `MODEL_ARTIFACTS_PATH`: Path for model artifacts
- `MAX_TRAINING_TIME`: Maximum training time per job (default: 3600s)

## 🔄 Operational Procedures

### Model Deployment
```bash
# Check model registry
curl http://localhost:5000/api/2.0/mlflow/registered-models/list

# Promote model to production
curl -X POST http://localhost:5000/api/2.0/mlflow/model-versions/transition-stage \
  -H "Content-Type: application/json" \
  -d '{
    "name": "strategy_classifier",
    "version": "1",
    "stage": "Production"
  }'
```

### Backup and Recovery
```bash
# Backup ML models
docker exec dexter-continuous-training \
  tar -czf /app/artifacts/models_backup_$(date +%Y%m%d).tar.gz /app/models

# Backup training data
docker exec dexter-continuous-training \
  tar -czf /app/artifacts/data_backup_$(date +%Y%m%d).tar.gz /app/data

# Backup PostgreSQL database
docker exec dexter-postgres \
  pg_dump -U dexter dexter_production > backup_$(date +%Y%m%d).sql
```

### Scaling Operations
```bash
# Scale Kafka consumers
docker-compose -f docker-compose.streaming.yml up -d --scale kafka-consumer=4

# Monitor resource usage
docker stats dexter-online-learning dexter-continuous-training

# Adjust resource limits
docker update --memory=4g --cpus=2 dexter-continuous-training
```

## 🐛 Troubleshooting

### Common Issues

#### Kafka Connection Issues
```bash
# Check Kafka broker status
docker exec dexter-kafka kafka-broker-api-versions --bootstrap-server localhost:9092

# Reset consumer group offset
docker exec dexter-kafka kafka-consumer-groups \
  --bootstrap-server localhost:9092 \
  --group dexter_ml_pipeline \
  --reset-offsets --to-earliest \
  --topic dexter.pool.events --execute
```

#### MLflow Database Issues
```bash
# Check MLflow database connection
docker exec dexter-postgres psql -U dexter -d mlflow_db -c "SELECT COUNT(*) FROM experiments;"

# Reset MLflow database
docker exec dexter-postgres psql -U dexter -c "DROP DATABASE IF EXISTS mlflow_db; CREATE DATABASE mlflow_db;"
```

#### Memory Issues
```bash
# Check memory usage
docker stats --no-stream

# Clear model caches
docker exec dexter-online-learning rm -rf /app/models/cache/*

# Restart memory-intensive services
docker restart dexter-continuous-training dexter-online-learning
```

### Performance Optimization

#### Kafka Optimization
```bash
# Increase Kafka heap size
docker exec dexter-kafka bash -c 'export KAFKA_HEAP_OPTS="-Xmx2G -Xms2G"'

# Optimize producer settings
export KAFKA_BATCH_SIZE=16384
export KAFKA_LINGER_MS=5
export KAFKA_BUFFER_MEMORY=33554432
```

#### ML Pipeline Optimization
```bash
# Increase worker threads
export ML_WORKER_THREADS=4

# Optimize batch processing
export ML_BATCH_SIZE=256
export FEATURE_CACHE_SIZE=10000
```

## 📈 Performance Metrics

### Key Performance Indicators
- **Message Throughput**: Target >10,000 messages/second
- **ML Prediction Latency**: Target <100ms per prediction
- **Model Accuracy**: Target >85% for strategy classification
- **Concept Drift Detection**: Target <24 hours to detect significant drift
- **Training Pipeline**: Target <2 hours for complete model retraining

### Monitoring Queries
```promql
# Kafka message rate
rate(kafka_producer_record_send_total[5m])

# ML prediction accuracy
avg(ml_model_accuracy) by (model_name)

# System resource usage
avg(container_cpu_usage_seconds_total) by (container_name)
avg(container_memory_usage_bytes) by (container_name)

# Service availability
up{job=~"dexter-.*"}
```

## 🔐 Security Considerations

### Network Security
```bash
# Use internal Docker networks
docker network create --driver bridge dexter-internal

# Implement service mesh for encryption
# (Consider Istio or Linkerd for production)
```

### Data Security
```bash
# Encrypt sensitive data at rest
export ENCRYPT_ML_MODELS=true

# Use secrets management
docker secret create kafka_password /path/to/kafka_password.txt
```

### Access Control
```bash
# Implement RBAC for MLflow
export MLFLOW_AUTH_CONFIG_PATH=/app/auth/mlflow_auth.yaml

# Secure Kafka with SASL/SSL
export KAFKA_SECURITY_PROTOCOL=SASL_SSL
```

## 📚 Further Reading

- [Kafka Documentation](https://kafka.apache.org/documentation/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [River ML Documentation](https://riverml.xyz/)
- [Prometheus Monitoring](https://prometheus.io/docs/)
- [A/B Testing Best Practices](https://exp-platform.com/)

## 🆘 Support

For issues and support:
1. Check service logs: `docker-compose logs [service-name]`
2. Verify health endpoints: `curl http://localhost:[port]/health`
3. Monitor system resources: `docker stats`
4. Review Grafana dashboards: http://localhost:3001

---

**Next Steps**: Proceed to Phase 3 for oracle infrastructure deployment and smart contract integration testing.