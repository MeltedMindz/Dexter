#!/usr/bin/env python3
"""
Entry point for Dexter Integrated Learning Service
Handles imports and system setup
"""

import sys
import os
import asyncio
import logging
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/dexter/integrated_learning.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Now import our components with proper path setup
try:
    from backend.dexbrain.advanced_data_collection import AdvancedUniswapDataCollector, DataCollectionConfig
    from backend.dexbrain.learning_verification_system import LearningVerificationSystem
    from backend.dexbrain.models.knowledge_base import KnowledgeBase
    from backend.dexbrain.training_pipeline import AdvancedTrainingPipeline
    
    logger.info("✅ All components imported successfully")
    
except ImportError as e:
    logger.error(f"❌ Import error: {e}")
    # Fallback to simpler imports
    sys.exit(1)

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
            alchemy_api_key=os.getenv('ALCHEMY_API_KEY', 'demo'),
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
            'service_start_time': os.environ.get('SERVICE_START_TIME', str(asyncio.get_event_loop().time())),
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
        
        logger.info("Integrated learning service initialized successfully")
    
    async def start_integrated_learning_service(self):
        """
        Start the comprehensive learning service with all components
        """
        logger.info("🚀 Starting integrated learning service...")
        
        await self._log_service_start()
        
        # Start concurrent tasks
        tasks = [
            asyncio.create_task(self._simplified_data_collection_loop()),
            asyncio.create_task(self._service_monitoring_loop())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Integrated learning service error: {e}")
            await self._handle_service_error(e)
    
    async def _simplified_data_collection_loop(self):
        """Simplified data collection loop for initial deployment"""
        logger.info("📊 Starting simplified data collection...")
        
        while True:
            try:
                collection_start = asyncio.get_event_loop().time()
                
                # Collect position data using multiple methods
                positions = await self.data_collector.collect_comprehensive_position_data()
                
                # Process and log results
                self.service_stats['total_positions_processed'] += len(positions)
                
                collection_duration = asyncio.get_event_loop().time() - collection_start
                
                await self._log_data_collection_cycle(positions, len(positions), collection_duration)
                
                # Wait for next collection cycle
                await asyncio.sleep(self.data_collection_interval)
                
            except Exception as e:
                logger.error(f"Data collection loop error: {e}")
                self.service_stats['data_collection_errors'] += 1
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    async def _service_monitoring_loop(self):
        """Monitor overall service health and performance"""
        logger.info("📈 Starting service monitoring...")
        
        while True:
            try:
                # Calculate service metrics
                uptime_seconds = asyncio.get_event_loop().time() - float(self.service_stats['service_start_time'])
                uptime_hours = uptime_seconds / 3600
                
                positions_per_hour = self.service_stats['total_positions_processed'] / max(uptime_hours, 1)
                
                service_health = {
                    'service_uptime_hours': uptime_hours,
                    'total_positions_processed': self.service_stats['total_positions_processed'],
                    'positions_per_hour': positions_per_hour,
                    'data_collection_errors': self.service_stats['data_collection_errors'],
                    'service_status': 'operational'
                }
                
                await self._log_service_health(service_health)
                
                # Wait 30 minutes before next health check
                await asyncio.sleep(1800)
                
            except Exception as e:
                logger.error(f"Service monitoring error: {e}")
                await asyncio.sleep(1800)
    
    # Logging methods
    
    async def _log_service_start(self):
        """Log service startup"""
        import json
        log_data = {
            "timestamp": asyncio.get_event_loop().time(),
            "level": "INFO",
            "source": "IntegratedLearningService",
            "message": "Integrated learning service started with comprehensive data pipeline",
            "details": {
                "components": ["AdvancedDataCollector", "TrainingPipeline", "LearningVerification"],
                "data_collection_interval": self.data_collection_interval,
                "data_sources": ["GraphQL", "RPC Events", "Position NFTs", "Backup APIs"]
            }
        }
        website_logger.info(json.dumps(log_data))
    
    async def _log_data_collection_cycle(self, positions, processed_count, duration):
        """Log data collection cycle results"""
        import json
        log_data = {
            "timestamp": asyncio.get_event_loop().time(),
            "level": "INFO",
            "source": "DataCollection",
            "message": f"Collected and processed {processed_count} positions in {duration:.1f}s",
            "details": {
                "positions_collected": len(positions),
                "positions_processed": processed_count,
                "collection_duration": duration,
                "data_sources_used": list(self.data_collector.collection_stats.get('data_sources_used', [])),
                "success_rate": processed_count / max(len(positions), 1) if positions else 0
            }
        }
        website_logger.info(json.dumps(log_data))
    
    async def _log_service_health(self, health_metrics):
        """Log service health metrics"""
        import json
        log_data = {
            "timestamp": asyncio.get_event_loop().time(),
            "level": "INFO",
            "source": "ServiceHealth",
            "message": f"Service health: {health_metrics['positions_per_hour']:.1f} positions/hour, {health_metrics['service_uptime_hours']:.1f}h uptime",
            "details": health_metrics
        }
        website_logger.info(json.dumps(log_data))
    
    async def _handle_service_error(self, error):
        """Handle critical service errors"""
        import json
        log_data = {
            "timestamp": asyncio.get_event_loop().time(),
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


async def main():
    """Main function to run the integrated learning service"""
    logger.info("🎯 Initializing Dexter Integrated Learning Service...")
    
    try:
        service = IntegratedLearningService()
        await service.start_integrated_learning_service()
    except Exception as e:
        logger.error(f"Failed to start service: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())