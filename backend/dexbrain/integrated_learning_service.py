"""
Integrated Learning Service for Dexter AI
Combines advanced data collection, parsing, and learning verification
"""

import asyncio
import logging
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path

from .advanced_data_collection import AdvancedUniswapDataCollector, DataCollectionConfig
from .learning_verification_system import LearningVerificationSystem
from .models.knowledge_base import KnowledgeBase
from .training_pipeline import AdvancedTrainingPipeline

logger = logging.getLogger(__name__)

# Website logging
website_logger = logging.getLogger('website')
website_handler = logging.FileHandler('/var/log/dexter/liquidity.log')
website_handler.setFormatter(logging.Formatter('%(message)s'))
website_logger.addHandler(website_handler)
website_logger.setLevel(logging.INFO)

class IntegratedLearningService:
    """
    Comprehensive learning service that orchestrates data collection,
    parsing, training, and verification
    """
    
    def __init__(self):
        # Initialize configuration
        self.config = DataCollectionConfig(
            graph_api_key="c6f241c1dd5aea81977a63b2614af70d",
            alchemy_api_key=os.getenv('ALCHEMY_API_KEY', ''),
            infura_api_key=os.getenv('INFURA_API_KEY', ''),
            base_rpc_url=os.getenv('BASE_RPC_URL', 'https://base-mainnet.g.alchemy.com/v2/demo')
        )
        
        # Initialize core components
        self.data_collector = AdvancedUniswapDataCollector(self.config)
        self.knowledge_base = KnowledgeBase()
        self.training_pipeline = AdvancedTrainingPipeline(self.knowledge_base)
        self.verification_system = LearningVerificationSystem(
            self.data_collector, 
            self.knowledge_base, 
            self.training_pipeline
        )
        
        # Service state
        self.service_stats = {
            'service_start_time': datetime.now(),
            'total_learning_cycles': 0,
            'successful_learning_cycles': 0,
            'total_positions_processed': 0,
            'current_learning_verified': False,
            'last_successful_learning': None,
            'data_collection_errors': 0,
            'learning_verification_errors': 0
        }
        
        # Learning schedule
        self.data_collection_interval = 900  # 15 minutes
        self.learning_cycle_interval = 3600  # 1 hour
        self.verification_interval = 21600  # 6 hours
        
        logger.info("Integrated learning service initialized")
    
    async def start_integrated_learning_service(self):
        """
        Start the comprehensive learning service with all components
        """
        logger.info("🚀 Starting integrated learning service...")
        
        await self._log_service_start()
        
        # Start concurrent tasks
        tasks = [
            asyncio.create_task(self._continuous_data_collection_loop()),
            asyncio.create_task(self._continuous_learning_loop()),
            asyncio.create_task(self._continuous_verification_loop()),
            asyncio.create_task(self._service_monitoring_loop()),
            asyncio.create_task(self._data_quality_monitoring_loop())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Integrated learning service error: {e}")
            await self._handle_service_error(e)
    
    async def _continuous_data_collection_loop(self):
        """Continuously collect Uniswap position data"""
        logger.info("📊 Starting continuous data collection...")
        
        while True:
            try:
                collection_start = datetime.now()
                
                # Collect comprehensive position data
                positions = await self.data_collector.collect_comprehensive_position_data()
                
                # Process and store positions
                processed_count = await self._process_and_store_positions(positions)
                
                self.service_stats['total_positions_processed'] += processed_count
                
                collection_duration = (datetime.now() - collection_start).total_seconds()
                
                await self._log_data_collection_cycle(positions, processed_count, collection_duration)
                
                # Wait for next collection cycle
                await asyncio.sleep(self.data_collection_interval)
                
            except Exception as e:
                logger.error(f"Data collection loop error: {e}")
                self.service_stats['data_collection_errors'] += 1
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    async def _continuous_learning_loop(self):
        """Continuously train ML models on collected data"""
        logger.info("🧠 Starting continuous learning loop...")
        
        while True:
            try:
                learning_start = datetime.now()
                
                # Check if we should trigger learning
                should_train = await self._should_trigger_learning()
                
                if should_train:
                    self.service_stats['total_learning_cycles'] += 1
                    
                    # Execute learning cycle
                    learning_results = await self._execute_learning_cycle()
                    
                    if learning_results.get('status') == 'success':
                        self.service_stats['successful_learning_cycles'] += 1
                        self.service_stats['last_successful_learning'] = learning_start
                        
                        await self._log_successful_learning(learning_results)
                    else:
                        await self._log_failed_learning(learning_results)
                
                # Wait for next learning cycle
                await asyncio.sleep(self.learning_cycle_interval)
                
            except Exception as e:
                logger.error(f"Learning loop error: {e}")
                await asyncio.sleep(600)  # Wait 10 minutes on error
    
    async def _continuous_verification_loop(self):
        """Continuously verify learning progress"""
        logger.info("🔍 Starting continuous verification loop...")
        
        while True:
            try:
                verification_start = datetime.now()
                
                # Run comprehensive learning verification
                verification_results = await self.verification_system.run_comprehensive_learning_verification()
                
                # Update service state
                self.service_stats['current_learning_verified'] = verification_results['verification_passed']
                
                if verification_results['verification_passed']:
                    await self._log_successful_verification(verification_results)
                else:
                    await self._log_failed_verification(verification_results)
                    # Take corrective actions
                    await self._handle_verification_failure(verification_results)
                
                # Wait for next verification cycle
                await asyncio.sleep(self.verification_interval)
                
            except Exception as e:
                logger.error(f"Verification loop error: {e}")
                self.service_stats['learning_verification_errors'] += 1
                await asyncio.sleep(1800)  # Wait 30 minutes on error
    
    async def _service_monitoring_loop(self):
        """Monitor overall service health and performance"""
        logger.info("📈 Starting service monitoring...")
        
        while True:
            try:
                # Calculate service metrics
                uptime = datetime.now() - self.service_stats['service_start_time']
                uptime_hours = uptime.total_seconds() / 3600
                
                learning_success_rate = (
                    self.service_stats['successful_learning_cycles'] / 
                    max(self.service_stats['total_learning_cycles'], 1)
                )
                
                positions_per_hour = self.service_stats['total_positions_processed'] / max(uptime_hours, 1)
                
                service_health = {
                    'service_uptime_hours': uptime_hours,
                    'total_learning_cycles': self.service_stats['total_learning_cycles'],
                    'learning_success_rate': learning_success_rate,
                    'current_learning_verified': self.service_stats['current_learning_verified'],
                    'positions_per_hour': positions_per_hour,
                    'total_positions_processed': self.service_stats['total_positions_processed'],
                    'data_collection_errors': self.service_stats['data_collection_errors'],
                    'verification_errors': self.service_stats['learning_verification_errors']
                }
                
                await self._log_service_health(service_health)
                
                # Alert on issues
                if learning_success_rate < 0.5 and self.service_stats['total_learning_cycles'] > 5:
                    await self._alert_low_learning_success()
                
                if positions_per_hour < 1.0 and uptime_hours > 2:
                    await self._alert_low_data_collection()
                
                # Wait 30 minutes before next health check
                await asyncio.sleep(1800)
                
            except Exception as e:
                logger.error(f"Service monitoring error: {e}")
                await asyncio.sleep(1800)
    
    async def _data_quality_monitoring_loop(self):
        """Monitor data quality and take corrective actions"""
        logger.info("🔎 Starting data quality monitoring...")
        
        while True:
            try:
                # Get data collection summary
                collection_summary = self.data_collector.get_collection_summary()
                
                # Assess data quality
                data_quality_issues = []
                
                if collection_summary['success_rate'] < 0.7:
                    data_quality_issues.append("Low data collection success rate")
                
                if collection_summary['positions_collected'] == 0:
                    data_quality_issues.append("No positions collected in recent cycles")
                
                # Check data source health
                data_quality = collection_summary.get('data_quality_distribution', {})
                if 'graph_api' not in data_quality or data_quality['graph_api'] == 0:
                    data_quality_issues.append("Primary Graph API not providing data")
                
                if data_quality_issues:
                    await self._handle_data_quality_issues(data_quality_issues)
                else:
                    await self._log_good_data_quality(collection_summary)
                
                # Wait 1 hour before next quality check
                await asyncio.sleep(3600)
                
            except Exception as e:
                logger.error(f"Data quality monitoring error: {e}")
                await asyncio.sleep(3600)
    
    async def _process_and_store_positions(self, positions) -> int:
        """Process collected positions and store in knowledge base"""
        processed_count = 0
        
        for position in positions:
            try:
                # Convert position to insight format
                insight = {
                    'type': 'position_data',
                    'source': 'integrated_learning_service',
                    'data': {
                        'position_id': position.position_id,
                        'pool_address': position.pool_address,
                        'tokens': f"{position.token0_symbol}/{position.token1_symbol}",
                        'fee_tier': position.fee_tier,
                        'tick_range': [position.tick_lower, position.tick_upper],
                        'position_value_usd': position.position_value_usd,
                        'apr_estimate': position.apr_estimate,
                        'fee_yield': position.fee_yield,
                        'impermanent_loss': position.impermanent_loss,
                        'in_range': position.in_range,
                        'capital_efficiency': position.capital_efficiency,
                        'data_source': position.data_source
                    },
                    'confidence': position.confidence_score,
                    'timestamp': position.timestamp.isoformat() if position.timestamp else datetime.now().isoformat()
                }
                
                await self.knowledge_base.add_insight(insight)
                processed_count += 1
                
            except Exception as e:
                logger.warning(f"Failed to process position {position.position_id}: {e}")
                continue
        
        return processed_count
    
    async def _should_trigger_learning(self) -> bool:
        """Determine if we should trigger a learning cycle"""
        try:
            # Check if we have enough new data
            recent_insights = await self.knowledge_base.get_recent_insights(limit=1000)
            
            if len(recent_insights) < 50:
                return False
            
            # Check if enough time has passed since last training
            training_status = await self.training_pipeline.get_training_status()
            
            if training_status['should_retrain']:
                return True
            
            # Check if data quality has improved significantly
            last_training = training_status.get('last_training_time')
            if last_training:
                hours_since_training = (datetime.now() - last_training).total_seconds() / 3600
                if hours_since_training > 6:  # Force retraining every 6 hours
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking learning trigger: {e}")
            return False
    
    async def _execute_learning_cycle(self) -> Dict[str, Any]:
        """Execute a complete learning cycle"""
        logger.info("🎯 Executing learning cycle...")
        
        try:
            # Force retrain with latest data
            training_results = await self.training_pipeline.train_models(force_retrain=True)
            
            if training_results.get('status') == 'success':
                logger.info("✅ Learning cycle completed successfully")
                return {
                    'status': 'success',
                    'training_results': training_results,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                logger.warning(f"⚠️ Learning cycle failed: {training_results}")
                return {
                    'status': 'failed',
                    'error': training_results.get('error', 'Unknown error'),
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Learning cycle execution error: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    # Logging methods
    
    async def _log_service_start(self):
        """Log service startup"""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "level": "INFO",
            "source": "IntegratedLearningService",
            "message": "Integrated learning service started with comprehensive data pipeline",
            "details": {
                "components": ["AdvancedDataCollector", "TrainingPipeline", "LearningVerification"],
                "data_collection_interval": self.data_collection_interval,
                "learning_cycle_interval": self.learning_cycle_interval,
                "verification_interval": self.verification_interval,
                "data_sources": ["GraphQL", "RPC Events", "Position NFTs", "Backup APIs"]
            }
        }
        website_logger.info(json.dumps(log_data))
    
    async def _log_data_collection_cycle(self, positions, processed_count, duration):
        """Log data collection cycle results"""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "level": "INFO",
            "source": "DataCollection",
            "message": f"Collected and processed {processed_count} positions in {duration:.1f}s",
            "details": {
                "positions_collected": len(positions),
                "positions_processed": processed_count,
                "collection_duration": duration,
                "data_sources_used": list(self.data_collector.collection_stats['data_sources_used']),
                "success_rate": processed_count / max(len(positions), 1)
            }
        }
        website_logger.info(json.dumps(log_data))
    
    async def _log_successful_learning(self, learning_results):
        """Log successful learning cycle"""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "level": "SUCCESS",
            "source": "LearningCycle",
            "message": f"ML training completed successfully - Cycle {self.service_stats['total_learning_cycles']}",
            "details": {
                "training_results": learning_results['training_results'],
                "cycle_number": self.service_stats['total_learning_cycles'],
                "success_rate": self.service_stats['successful_learning_cycles'] / self.service_stats['total_learning_cycles']
            }
        }
        website_logger.info(json.dumps(log_data))
    
    async def _log_service_health(self, health_metrics):
        """Log service health metrics"""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "level": "INFO",
            "source": "ServiceHealth",
            "message": f"Service health: {health_metrics['learning_success_rate']:.1%} learning success, {health_metrics['positions_per_hour']:.1f} positions/hour",
            "details": health_metrics
        }
        website_logger.info(json.dumps(log_data))
    
    # Error handling and alerts
    
    async def _handle_service_error(self, error):
        """Handle critical service errors"""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "level": "CRITICAL",
            "source": "ServiceError",
            "message": f"Critical service error: {error}",
            "details": {
                "error_type": type(error).__name__,
                "error_message": str(error),
                "service_stats": self.service_stats
            }
        }
        website_logger.info(json.dumps(log_data))
    
    async def _handle_verification_failure(self, verification_results):
        """Handle learning verification failures"""
        recommendations = verification_results.get('recommendations', [])
        
        # Take automated corrective actions
        if "Insufficient data" in str(recommendations):
            # Trigger more aggressive data collection
            self.data_collection_interval = min(300, self.data_collection_interval)  # Increase frequency
        
        if "Model underperforming" in str(recommendations):
            # Trigger immediate retraining
            asyncio.create_task(self._execute_learning_cycle())
        
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "level": "WARNING",
            "source": "VerificationFailure",
            "message": "Learning verification failed - taking corrective actions",
            "details": {
                "verification_results": verification_results,
                "corrective_actions": "Adjusted data collection frequency and triggered retraining"
            }
        }
        website_logger.info(json.dumps(log_data))
    
    async def get_service_status(self) -> Dict[str, Any]:
        """Get comprehensive service status"""
        uptime = datetime.now() - self.service_stats['service_start_time']
        
        return {
            'service_uptime': str(uptime),
            'service_stats': self.service_stats,
            'data_collection_summary': self.data_collector.get_collection_summary(),
            'training_status': await self.training_pipeline.get_training_status(),
            'learning_verified': self.service_stats['current_learning_verified'],
            'last_successful_learning': self.service_stats['last_successful_learning']
        }


async def main():
    """Main function to run the integrated learning service"""
    service = IntegratedLearningService()
    await service.start_integrated_learning_service()

if __name__ == "__main__":
    asyncio.run(main())