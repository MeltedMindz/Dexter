"""
Kafka Producer for Real-time DeFi Data Streaming
High-performance producer for streaming pool data, trades, and ML predictions
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
import aiofiles
import aiokafka
from aiokafka import AIOKafkaProducer
from aiokafka.errors import KafkaError
import os

from ..dexbrain.config import Config

logger = logging.getLogger(__name__)

@dataclass
class PoolDataEvent:
    """Pool data event for streaming"""
    pool_address: str
    timestamp: int
    event_type: str  # 'swap', 'mint', 'burn', 'collect'
    token0_symbol: str
    token1_symbol: str
    fee_tier: int
    liquidity: int
    tick: int
    price: float
    volume_usd: float
    fees_usd: float
    tvl_usd: float
    transaction_hash: str
    block_number: int
    volatility_1h: Optional[float] = None
    volatility_24h: Optional[float] = None
    
    def to_kafka_message(self) -> Dict[str, Any]:
        """Convert to Kafka message format"""
        return {
            'key': self.pool_address,
            'value': asdict(self),
            'timestamp_ms': self.timestamp * 1000,
            'headers': {
                'event_type': self.event_type,
                'pool_address': self.pool_address,
                'source': 'dexter_pipeline'
            }
        }

@dataclass
class MLPredictionEvent:
    """ML prediction event for streaming"""
    pool_address: str
    timestamp: int
    model_name: str
    prediction_type: str
    action_recommendation: str
    confidence: float
    expected_return: float
    risk_score: float
    features_used: List[str]
    model_version: str
    
    def to_kafka_message(self) -> Dict[str, Any]:
        """Convert to Kafka message format"""
        return {
            'key': f"{self.pool_address}_{self.model_name}",
            'value': asdict(self),
            'timestamp_ms': self.timestamp * 1000,
            'headers': {
                'model': self.model_name,
                'pool_address': self.pool_address,
                'prediction_type': self.prediction_type,
                'source': 'dexter_ml'
            }
        }

class DexterKafkaProducer:
    """
    High-performance Kafka producer for Dexter Protocol streaming data
    Handles pool events, ML predictions, and system metrics
    """
    
    def __init__(self):
        # Kafka configuration
        self.bootstrap_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
        self.security_protocol = os.getenv('KAFKA_SECURITY_PROTOCOL', 'PLAINTEXT')
        
        # Topic configuration
        self.topics = {
            'pool_events': 'dexter.pool.events',
            'ml_predictions': 'dexter.ml.predictions', 
            'price_updates': 'dexter.price.updates',
            'liquidity_changes': 'dexter.liquidity.changes',
            'system_metrics': 'dexter.system.metrics',
            'alerts': 'dexter.alerts'
        }
        
        # Producer instance
        self.producer = None
        
        # Performance tracking
        self.message_count = 0
        self.error_count = 0
        self.last_send_time = 0
        
        # Batching configuration
        self.batch_size = int(os.getenv('KAFKA_BATCH_SIZE', '100'))
        self.linger_ms = int(os.getenv('KAFKA_LINGER_MS', '100'))
        self.max_request_size = int(os.getenv('KAFKA_MAX_REQUEST_SIZE', '1048576'))  # 1MB
        
        logger.info("Dexter Kafka producer initialized")
    
    async def start(self):
        """Start the Kafka producer"""
        try:
            self.producer = AIOKafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                security_protocol=self.security_protocol,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                compression_type='snappy',
                batch_size=self.batch_size,
                linger_ms=self.linger_ms,
                max_request_size=self.max_request_size,
                request_timeout_ms=30000,
                retry_backoff_ms=100,
                retries=5,
                enable_idempotence=True,
                acks='all'  # Wait for all replicas
            )
            
            await self.producer.start()
            
            logger.info(f"Kafka producer started successfully")
            logger.info(f"Connected to: {self.bootstrap_servers}")
            logger.info(f"Topics configured: {list(self.topics.values())}")
            
        except Exception as e:
            logger.error(f"Failed to start Kafka producer: {e}")
            raise
    
    async def stop(self):
        """Stop the Kafka producer"""
        if self.producer:
            await self.producer.stop()
            logger.info("Kafka producer stopped")
    
    async def send_pool_event(self, event: PoolDataEvent) -> bool:
        """Send pool data event to Kafka"""
        try:
            message = event.to_kafka_message()
            
            await self.producer.send_and_wait(
                topic=self.topics['pool_events'],
                key=message['key'],
                value=message['value'],
                timestamp_ms=message['timestamp_ms'],
                headers=[(k, v.encode()) for k, v in message['headers'].items()]
            )
            
            self.message_count += 1
            self.last_send_time = time.time()
            
            logger.debug(f"Pool event sent: {event.pool_address} - {event.event_type}")
            return True
            
        except KafkaError as e:
            logger.error(f"Kafka error sending pool event: {e}")
            self.error_count += 1
            return False
        except Exception as e:
            logger.error(f"Error sending pool event: {e}")
            self.error_count += 1
            return False
    
    async def send_ml_prediction(self, prediction: MLPredictionEvent) -> bool:
        """Send ML prediction to Kafka"""
        try:
            message = prediction.to_kafka_message()
            
            await self.producer.send_and_wait(
                topic=self.topics['ml_predictions'],
                key=message['key'],
                value=message['value'],
                timestamp_ms=message['timestamp_ms'],
                headers=[(k, v.encode()) for k, v in message['headers'].items()]
            )
            
            self.message_count += 1
            self.last_send_time = time.time()
            
            logger.debug(f"ML prediction sent: {prediction.pool_address} - {prediction.action_recommendation}")
            return True
            
        except KafkaError as e:
            logger.error(f"Kafka error sending ML prediction: {e}")
            self.error_count += 1
            return False
        except Exception as e:
            logger.error(f"Error sending ML prediction: {e}")
            self.error_count += 1
            return False
    
    async def send_price_update(self, pool_address: str, price_data: Dict[str, Any]) -> bool:
        """Send price update event"""
        try:
            message = {
                'pool_address': pool_address,
                'timestamp': int(time.time()),
                'price': price_data.get('price', 0),
                'price_change_1h': price_data.get('price_change_1h', 0),
                'price_change_24h': price_data.get('price_change_24h', 0),
                'volume_1h': price_data.get('volume_1h', 0),
                'volume_24h': price_data.get('volume_24h', 0)
            }
            
            await self.producer.send_and_wait(
                topic=self.topics['price_updates'],
                key=pool_address,
                value=message,
                headers=[
                    ('event_type', b'price_update'),
                    ('pool_address', pool_address.encode()),
                    ('source', b'dexter_price_feed')
                ]
            )
            
            self.message_count += 1
            return True
            
        except Exception as e:
            logger.error(f"Error sending price update: {e}")
            self.error_count += 1
            return False
    
    async def send_liquidity_change(self, pool_address: str, liquidity_data: Dict[str, Any]) -> bool:
        """Send liquidity change event"""
        try:
            message = {
                'pool_address': pool_address,
                'timestamp': int(time.time()),
                'liquidity_delta': liquidity_data.get('liquidity_delta', 0),
                'new_liquidity': liquidity_data.get('new_liquidity', 0),
                'tick_lower': liquidity_data.get('tick_lower'),
                'tick_upper': liquidity_data.get('tick_upper'),
                'amount0': liquidity_data.get('amount0', 0),
                'amount1': liquidity_data.get('amount1', 0),
                'transaction_hash': liquidity_data.get('transaction_hash', '')
            }
            
            await self.producer.send_and_wait(
                topic=self.topics['liquidity_changes'],
                key=pool_address,
                value=message,
                headers=[
                    ('event_type', b'liquidity_change'),
                    ('pool_address', pool_address.encode()),
                    ('source', b'dexter_liquidity_monitor')
                ]
            )
            
            self.message_count += 1
            return True
            
        except Exception as e:
            logger.error(f"Error sending liquidity change: {e}")
            self.error_count += 1
            return False
    
    async def send_system_metrics(self, metrics: Dict[str, Any]) -> bool:
        """Send system performance metrics"""
        try:
            message = {
                'timestamp': int(time.time()),
                'service_name': metrics.get('service_name', 'dexter_pipeline'),
                'cpu_usage': metrics.get('cpu_usage', 0),
                'memory_usage': metrics.get('memory_usage', 0),
                'active_pools': metrics.get('active_pools', 0),
                'predictions_per_minute': metrics.get('predictions_per_minute', 0),
                'error_rate': metrics.get('error_rate', 0),
                'avg_response_time': metrics.get('avg_response_time', 0)
            }
            
            await self.producer.send_and_wait(
                topic=self.topics['system_metrics'],
                key=metrics.get('service_name', 'default'),
                value=message,
                headers=[
                    ('metric_type', b'system_performance'),
                    ('source', b'dexter_monitoring')
                ]
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error sending system metrics: {e}")
            return False
    
    async def send_alert(self, alert_data: Dict[str, Any]) -> bool:
        """Send system alert"""
        try:
            message = {
                'timestamp': int(time.time()),
                'alert_type': alert_data.get('alert_type', 'info'),
                'severity': alert_data.get('severity', 'medium'),
                'title': alert_data.get('title', ''),
                'description': alert_data.get('description', ''),
                'pool_address': alert_data.get('pool_address'),
                'service': alert_data.get('service', 'dexter'),
                'metadata': alert_data.get('metadata', {})
            }
            
            await self.producer.send_and_wait(
                topic=self.topics['alerts'],
                key=f"{alert_data.get('alert_type', 'info')}_{int(time.time())}",
                value=message,
                headers=[
                    ('alert_type', alert_data.get('alert_type', 'info').encode()),
                    ('severity', alert_data.get('severity', 'medium').encode()),
                    ('source', b'dexter_alerts')
                ]
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error sending alert: {e}")
            return False
    
    async def batch_send_pool_events(self, events: List[PoolDataEvent]) -> int:
        """Send multiple pool events in batch"""
        successful_sends = 0
        
        tasks = []
        for event in events:
            task = asyncio.create_task(self.send_pool_event(event))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if result is True:
                successful_sends += 1
            elif isinstance(result, Exception):
                logger.error(f"Batch send error: {result}")
        
        logger.info(f"Batch sent {successful_sends}/{len(events)} pool events")
        return successful_sends
    
    async def batch_send_ml_predictions(self, predictions: List[MLPredictionEvent]) -> int:
        """Send multiple ML predictions in batch"""
        successful_sends = 0
        
        tasks = []
        for prediction in predictions:
            task = asyncio.create_task(self.send_ml_prediction(prediction))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if result is True:
                successful_sends += 1
            elif isinstance(result, Exception):
                logger.error(f"Batch send error: {result}")
        
        logger.info(f"Batch sent {successful_sends}/{len(predictions)} ML predictions")
        return successful_sends
    
    def get_producer_stats(self) -> Dict[str, Any]:
        """Get producer performance statistics"""
        uptime = time.time() - self.last_send_time if self.last_send_time > 0 else 0
        
        return {
            'total_messages_sent': self.message_count,
            'total_errors': self.error_count,
            'error_rate': self.error_count / max(self.message_count, 1),
            'last_send_time': self.last_send_time,
            'uptime_seconds': uptime,
            'topics_configured': len(self.topics),
            'is_connected': self.producer is not None
        }

# Factory function
async def create_kafka_producer() -> DexterKafkaProducer:
    """Create and start Kafka producer"""
    producer = DexterKafkaProducer()
    await producer.start()
    return producer

# Example usage for testing
async def test_kafka_producer():
    """Test function for Kafka producer"""
    producer = await create_kafka_producer()
    
    try:
        # Test pool event
        pool_event = PoolDataEvent(
            pool_address="0x1234567890123456789012345678901234567890",
            timestamp=int(time.time()),
            event_type="swap",
            token0_symbol="USDC",
            token1_symbol="ETH",
            fee_tier=3000,
            liquidity=1000000,
            tick=200000,
            price=3000.50,
            volume_usd=100000.0,
            fees_usd=300.0,
            tvl_usd=5000000.0,
            transaction_hash="0xabcdef1234567890",
            block_number=12345678,
            volatility_1h=0.05,
            volatility_24h=0.15
        )
        
        success = await producer.send_pool_event(pool_event)
        print(f"Pool event sent: {success}")
        
        # Test ML prediction
        ml_prediction = MLPredictionEvent(
            pool_address="0x1234567890123456789012345678901234567890",
            timestamp=int(time.time()),
            model_name="dexter_strategy_optimizer",
            prediction_type="rebalance_recommendation",
            action_recommendation="narrow_range",
            confidence=0.85,
            expected_return=0.12,
            risk_score=0.3,
            features_used=["price_volatility", "volume_24h", "liquidity_depth"],
            model_version="v2.1.0"
        )
        
        success = await producer.send_ml_prediction(ml_prediction)
        print(f"ML prediction sent: {success}")
        
        # Print stats
        stats = producer.get_producer_stats()
        print(f"Producer stats: {stats}")
        
    finally:
        await producer.stop()

if __name__ == "__main__":
    asyncio.run(test_kafka_producer())