import os
import asyncio
import logging
from web3 import Web3
from execution.manager import ExecutionManager
from execution.execution_config import EXECUTION_CONFIG
from agents import ConservativeAgent, AggressiveAgent, HyperAggressiveAgent
from utils.error_handler import ErrorHandler
from utils.memory_monitor import MemoryMonitor

logger = logging.getLogger(__name__)

async def initialize_system():
    """Initialize all system components"""
    try:
        # Initialize Web3
        web3 = Web3(Web3.HTTPProvider(os.getenv('WEB3_PROVIDER')))
        
        # Initialize agents
        agents = [
            ConservativeAgent(web3),
            AggressiveAgent(web3),
            HyperAggressiveAgent(web3)
        ]
        
        # Initialize execution manager
        manager = ExecutionManager(
            web3=web3,
            agents=agents,
            config=EXECUTION_CONFIG
        )
        
        return manager
        
    except Exception as e:
        logger.error(f"System initialization failed: {str(e)}")
        raise

async def main():
    """Main entry point"""
    try:
        manager = await initialize_system()
        await manager.start()
        
        # Keep the system running
        while True:
            await asyncio.sleep(60)
            
            # Log performance metrics
            memory_stats = MemoryMonitor().get_stats()
            error_stats = ErrorHandler().get_error_stats()
            
            logger.info("System performance metrics:")
            logger.info(f"Memory usage: {memory_stats}")
            logger.info(f"Error statistics: {error_stats}")
            
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        await manager.stop()
    except Exception as e:
        logger.error(f"System error: {str(e)}")
        raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
