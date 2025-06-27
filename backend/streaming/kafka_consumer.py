"""
Kafka Consumer for Real-time DeFi Data Processing
High-performance consumer for processing streaming pool data and ML predictions
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Any, Callable, Optional
from datetime import datetime
from dataclasses import dataclass
import aiokafka
from aiokafka import AIOKafkaConsumer
from aiokafka.errors import KafkaError, ConsumerStoppedError
import os
import signal
import sys

from ..dexbrain.config import Config

logger = logging.getLogger(__name__)

@dataclass
class ConsumerMetrics:
    """Consumer performance metrics"""
    messages_processed: int = 0
    processing_errors: int = 0
    average_processing_time: float = 0.0
    last_message_time: int = 0
    consumer_lag: int = 0
    
class DexterKafkaConsumer:
    """
    High-performance Kafka consumer for Dexter Protocol streaming data
    Processes pool events, ML predictions, and system metrics in real-time
    """
    
    def __init__(self, consumer_group: str = "dexter_ml_pipeline"):
        # Kafka configuration
        self.bootstrap_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
        self.consumer_group = consumer_group
        self.auto_offset_reset = os.getenv('KAFKA_AUTO_OFFSET_RESET', 'latest')
        
        # Topics to consume
        self.topics = [
            'dexter.pool.events',
            'dexter.ml.predictions',
            'dexter.price.updates',
            'dexter.liquidity.changes',
            'dexter.system.metrics',
            'dexter.alerts'
        ]
        
        # Consumer instance
        self.consumer = None
        
        # Message handlers
        self.handlers = {
            'dexter.pool.events': self._handle_pool_event,
            'dexter.ml.predictions': self._handle_ml_prediction,
            'dexter.price.updates': self._handle_price_update,
            'dexter.liquidity.changes': self._handle_liquidity_change,
            'dexter.system.metrics': self._handle_system_metrics,
            'dexter.alerts': self._handle_alert
        }
        
        # Performance tracking
        self.metrics = ConsumerMetrics()
        self.processing_times = []
        
        # State management
        self.running = False
        self.shutdown_event = asyncio.Event()
        
        # Batch processing
        self.batch_size = int(os.getenv('KAFKA_CONSUMER_BATCH_SIZE', '100'))
        self.batch_timeout = int(os.getenv('KAFKA_CONSUMER_BATCH_TIMEOUT', '5'))  # seconds
        
        logger.info(f"Dexter Kafka consumer initialized for group: {consumer_group}")
    
    async def start(self):
        """Start the Kafka consumer"""
        try:
            self.consumer = AIOKafkaConsumer(
                *self.topics,
                bootstrap_servers=self.bootstrap_servers,
                group_id=self.consumer_group,
                auto_offset_reset=self.auto_offset_reset,
                enable_auto_commit=True,
                auto_commit_interval_ms=1000,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')) if m else None,
                key_deserializer=lambda k: k.decode('utf-8') if k else None,
                session_timeout_ms=30000,
                heartbeat_interval_ms=3000,
                max_poll_records=self.batch_size,
                fetch_max_wait_ms=500,
                max_partition_fetch_bytes=1048576  # 1MB
            )
            
            await self.consumer.start()
            self.running = True
            
            logger.info(f"Kafka consumer started successfully")
            logger.info(f"Connected to: {self.bootstrap_servers}")
            logger.info(f"Consumer group: {self.consumer_group}")
            logger.info(f"Subscribed to topics: {self.topics}")
            
            # Set up signal handlers for graceful shutdown
            self._setup_signal_handlers()
            
        except Exception as e:
            logger.error(f"Failed to start Kafka consumer: {e}")
            raise
    
    async def stop(self):
        """Stop the Kafka consumer"""
        self.running = False
        self.shutdown_event.set()
        
        if self.consumer:
            await self.consumer.stop()
            logger.info("Kafka consumer stopped")
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            asyncio.create_task(self.stop())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def consume_messages(self):
        """Main consumption loop"""
        logger.info("Starting message consumption...")
        
        try:
            while self.running:
                # Get batch of messages
                message_batch = await self._get_message_batch()
                
                if message_batch:
                    # Process batch
                    await self._process_message_batch(message_batch)
                else:
                    # Short sleep if no messages
                    await asyncio.sleep(0.1)
                
                # Check for shutdown
                if self.shutdown_event.is_set():
                    break
                    
        except ConsumerStoppedError:
            logger.info("Consumer stopped")
        except Exception as e:
            logger.error(f"Consumption loop error: {e}")
            raise
        finally:
            logger.info("Message consumption ended")
    
    async def _get_message_batch(self) -> List[Any]:
        """Get a batch of messages with timeout"""
        batch = []
        start_time = time.time()
        
        try:
            # Poll for messages with timeout
            msg_pack = await asyncio.wait_for(
                self.consumer.getmany(timeout_ms=1000),
                timeout=self.batch_timeout
            )
            
            for tp, messages in msg_pack.items():
                for message in messages:
                    batch.append(message)
                    if len(batch) >= self.batch_size:
                        break
                if len(batch) >= self.batch_size:
                    break
                    
        except asyncio.TimeoutError:
            # Return whatever we have
            pass
        except Exception as e:
            logger.error(f"Error getting message batch: {e}")
        
        return batch
    
    async def _process_message_batch(self, messages: List[Any]):
        """Process a batch of messages"""
        start_time = time.time()
        
        # Group messages by topic for efficient processing
        topic_groups = {}
        for message in messages:
            topic = message.topic
            if topic not in topic_groups:
                topic_groups[topic] = []
            topic_groups[topic].append(message)
        
        # Process each topic group
        processing_tasks = []
        for topic, topic_messages in topic_groups.items():
            if topic in self.handlers:
                task = asyncio.create_task(
                    self._process_topic_messages(topic, topic_messages)
                )
                processing_tasks.append(task)
        
        # Wait for all processing tasks to complete
        if processing_tasks:
            results = await asyncio.gather(*processing_tasks, return_exceptions=True)
            
            # Count successes and errors
            successful = sum(1 for r in results if r is True)
            errors = sum(1 for r in results if isinstance(r, Exception))
            
            # Update metrics
            processing_time = time.time() - start_time
            self.metrics.messages_processed += len(messages)
            self.metrics.processing_errors += errors
            self.metrics.last_message_time = int(time.time())
            
            # Update average processing time
            self.processing_times.append(processing_time)
            if len(self.processing_times) > 100:  # Keep last 100 measurements
                self.processing_times.pop(0)
            self.metrics.average_processing_time = sum(self.processing_times) / len(self.processing_times)
            
            logger.debug(f"Processed batch: {len(messages)} messages, {successful} successful, {errors} errors, {processing_time:.3f}s")
    
    async def _process_topic_messages(self, topic: str, messages: List[Any]) -> bool:
        """Process messages for a specific topic"""
        try:
            handler = self.handlers[topic]
            
            # Process messages in parallel for better throughput
            tasks = []
            for message in messages:
                task = asyncio.create_task(handler(message))
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Check for errors
            errors = [r for r in results if isinstance(r, Exception)]
            if errors:
                logger.warning(f"Topic {topic} processing errors: {len(errors)}/{len(messages)}")
                for error in errors[:3]:  # Log first 3 errors
                    logger.error(f"Processing error: {error}")
            
            return len(errors) == 0
            
        except Exception as e:
            logger.error(f"Error processing topic {topic}: {e}")
            return False
    
    async def _handle_pool_event(self, message) -> bool:
        """Handle pool event message"""
        try:
            data = message.value
            pool_address = data.get('pool_address')
            event_type = data.get('event_type')
            
            logger.debug(f"Processing pool event: {pool_address} - {event_type}")
            
            # Add your pool event processing logic here
            # For example:
            # - Update real-time pool state
            # - Trigger ML model updates
            # - Update price/volume metrics
            # - Store in time-series database
            
            # Example processing
            if event_type == 'swap':
                await self._process_swap_event(data)
            elif event_type == 'mint':
                await self._process_mint_event(data)
            elif event_type == 'burn':
                await self._process_burn_event(data)
            elif event_type == 'collect':
                await self._process_collect_event(data)
            
            return True
            
        except Exception as e:
            logger.error(f"Error handling pool event: {e}")
            return False
    
    async def _handle_ml_prediction(self, message) -> bool:
        """Handle ML prediction message"""
        try:
            data = message.value
            pool_address = data.get('pool_address')
            model_name = data.get('model_name')
            action = data.get('action_recommendation')
            confidence = data.get('confidence')
            
            logger.debug(f"Processing ML prediction: {pool_address} - {action} ({confidence:.2f})")
            
            # Add your ML prediction processing logic here
            # For example:
            # - Validate prediction quality
            # - Update position recommendations
            # - Trigger automated actions
            # - Store prediction history
            # - Send to oracle bridge
            
            # Example: High-confidence predictions trigger immediate actions
            if confidence > 0.8:
                await self._process_high_confidence_prediction(data)
            
            return True
            
        except Exception as e:
            logger.error(f"Error handling ML prediction: {e}")
            return False
    
    async def _handle_price_update(self, message) -> bool:
        """Handle price update message"""
        try:
            data = message.value
            pool_address = data.get('pool_address')
            price = data.get('price')
            
            logger.debug(f"Processing price update: {pool_address} - ${price:.2f}")
            
            # Add your price update processing logic here
            # For example:
            # - Update price feeds
            # - Calculate price volatility
            # - Trigger volatility alerts
            # - Update ML model features
            
            return True
            
        except Exception as e:
            logger.error(f"Error handling price update: {e}")
            return False
    
    async def _handle_liquidity_change(self, message) -> bool:
        """Handle liquidity change message"""
        try:
            data = message.value
            pool_address = data.get('pool_address')
            liquidity_delta = data.get('liquidity_delta')
            
            logger.debug(f"Processing liquidity change: {pool_address} - Δ{liquidity_delta}")
            
            # Add your liquidity change processing logic here
            # For example:
            # - Update liquidity metrics
            # - Recalculate capital efficiency
            # - Trigger rebalancing recommendations
            # - Update position tracking
            
            return True
            
        except Exception as e:
            logger.error(f"Error handling liquidity change: {e}")
            return False
    
    async def _handle_system_metrics(self, message) -> bool:
        """Handle system metrics message"""
        try:
            data = message.value
            service_name = data.get('service_name')
            cpu_usage = data.get('cpu_usage')
            
            logger.debug(f"Processing system metrics: {service_name} - CPU {cpu_usage:.1f}%")
            
            # Add your system metrics processing logic here
            # For example:
            # - Store in monitoring system
            # - Check for resource alerts
            # - Update service health status
            # - Trigger auto-scaling
            
            return True
            
        except Exception as e:
            logger.error(f"Error handling system metrics: {e}")
            return False
    
    async def _handle_alert(self, message) -> bool:
        """Handle alert message"""
        try:
            data = message.value
            alert_type = data.get('alert_type')
            severity = data.get('severity')
            title = data.get('title')
            
            logger.info(f"Processing alert: [{severity}] {alert_type} - {title}")
            
            # Add your alert processing logic here
            # For example:
            # - Send notifications
            # - Update alert dashboard
            # - Trigger automated responses
            # - Log to alert history
            
            return True
            
        except Exception as e:
            logger.error(f"Error handling alert: {e}")
            return False
    
    # Event processing methods
    async def _process_swap_event(self, data: Dict[str, Any]):
        """Process swap event specific logic"""
        # Example: Update volume metrics, trigger volatility calculations
        pass
    
    async def _process_mint_event(self, data: Dict[str, Any]):
        """Process mint event specific logic"""
        # Example: Update liquidity metrics, recalculate position efficiency
        pass
    
    async def _process_burn_event(self, data: Dict[str, Any]):
        """Process burn event specific logic"""
        # Example: Update liquidity metrics, check for exit signals
        pass
    
    async def _process_collect_event(self, data: Dict[str, Any]):
        """Process collect event specific logic"""
        # Example: Update fee collection metrics, compound recommendations
        pass
    
    async def _process_high_confidence_prediction(self, data: Dict[str, Any]):
        """Process high-confidence ML predictions"""
        # Example: Trigger immediate action, notify oracle bridge
        logger.info(f"High-confidence prediction received: {data.get('action_recommendation')} ({data.get('confidence'):.2f})")
    
    def get_consumer_metrics(self) -> Dict[str, Any]:
        """Get consumer performance metrics"""
        return {
            'messages_processed': self.metrics.messages_processed,
            'processing_errors': self.metrics.processing_errors,
            'error_rate': self.metrics.processing_errors / max(self.metrics.messages_processed, 1),
            'average_processing_time': self.metrics.average_processing_time,
            'last_message_time': self.metrics.last_message_time,
            'consumer_lag': self.metrics.consumer_lag,
            'is_running': self.running,
            'topics_subscribed': len(self.topics)
        }

# Factory function
async def create_kafka_consumer(consumer_group: str = "dexter_ml_pipeline") -> DexterKafkaConsumer:
    """Create and start Kafka consumer"""
    consumer = DexterKafkaConsumer(consumer_group)
    await consumer.start()
    return consumer

# Main function for running consumer as standalone service
async def main():
    """Main function to run the Kafka consumer"""
    consumer = await create_kafka_consumer("dexter_stream_processor")
    
    try:
        await consumer.consume_messages()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        await consumer.stop()

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    asyncio.run(main())