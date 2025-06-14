import asyncio
import logging
from typing import List, Dict, Any, Optional
import numpy as np
from .blockchain.solana_connector import SolanaConnector
from .blockchain.base_connector import BlockchainConnector
from .models.knowledge_base import KnowledgeBase
from .models.ml_models import DeFiMLEngine
from .config import Config


class DexBrain:
    """Core orchestrator for DexBrain system"""
    
    def __init__(self):
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, Config.LOG_LEVEL),
            format=Config.LOG_FORMAT
        )
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Validate configuration
        Config.validate()
        
        # Initialize components
        self.blockchain_connectors: Dict[str, BlockchainConnector] = {
            'solana': SolanaConnector()
        }
        
        self.knowledge_base = KnowledgeBase()
        self.ml_engine = DeFiMLEngine()
        self._shutdown_event = asyncio.Event()
    
    async def aggregate_data(
        self, 
        blockchain: str, 
        pool_addresses: List[str]
    ) -> List[Dict[str, Any]]:
        """Aggregate liquidity data from specified blockchain and pool addresses
        
        Args:
            blockchain: Target blockchain (e.g., 'solana')
            pool_addresses: List of pool addresses to analyze
            
        Returns:
            List of liquidity data for each pool
        """
        try:
            connector = self.blockchain_connectors.get(blockchain.lower())
            if not connector:
                raise ValueError(f"Unsupported blockchain: {blockchain}")
            
            # Use context manager for connection
            async with connector as conn:
                results = []
                
                # Fetch data for each pool concurrently
                tasks = []
                for pool_address in pool_addresses:
                    tasks.append(conn.fetch_liquidity_data(pool_address))
                
                liquidity_data_list = await asyncio.gather(*tasks, return_exceptions=True)
                
                for pool_address, liquidity_data in zip(pool_addresses, liquidity_data_list):
                    if isinstance(liquidity_data, Exception):
                        self.logger.error(f"Error fetching data for {pool_address}: {liquidity_data}")
                        continue
                    
                    if liquidity_data:
                        # Store insight in knowledge base
                        await self.knowledge_base.store_insight(
                            category=f"{blockchain}_liquidity",
                            insight={
                                'pool_address': pool_address,
                                'blockchain': blockchain,
                                **liquidity_data
                            }
                        )
                        results.append(liquidity_data)
                
                return results
                
        except Exception as e:
            self.logger.error(f"Data aggregation failed: {e}")
            raise
    
    async def train_models(self, category: str) -> Dict[str, Any]:
        """Train ML models based on stored insights
        
        Args:
            category: Insights category to use for training
            
        Returns:
            Training metrics
        """
        try:
            # Retrieve insights
            insights = await self.knowledge_base.retrieve_insights(category, limit=1000)
            
            if len(insights) < 10:
                self.logger.warning(f"Insufficient insights for training: {len(insights)}")
                return {'status': 'skipped', 'reason': 'insufficient_data'}
            
            # Prepare training data
            features = []
            targets = []
            
            for insight in insights:
                # Extract features
                feature_vector = [
                    insight.get('total_liquidity', 0),
                    insight.get('volume_24h', 0),
                    insight.get('fee_tier', 0),
                    insight.get('token0_reserves', 0),
                    insight.get('token1_reserves', 0),
                ]
                
                # Target is APR for now
                target = insight.get('apr', 0)
                
                features.append(feature_vector)
                targets.append(target)
            
            X = np.array(features)
            y = np.array(targets)
            
            # Train model
            metrics = self.ml_engine.train(X, y)
            
            self.logger.info(f"Model training completed: {metrics}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Model training failed: {e}")
            raise
    
    async def predict_liquidity_metrics(
        self, 
        pool_data: Dict[str, Any]
    ) -> float:
        """Predict liquidity metrics for a given pool
        
        Args:
            pool_data: Pool information
            
        Returns:
            Predicted APR
        """
        feature_vector = np.array([[
            pool_data.get('total_liquidity', 0),
            pool_data.get('volume_24h', 0),
            pool_data.get('fee_tier', 0),
            pool_data.get('token0_reserves', 0),
            pool_data.get('token1_reserves', 0),
        ]])
        
        prediction = self.ml_engine.predict(feature_vector)
        return float(prediction[0])
    
    async def run(
        self, 
        blockchain: str, 
        pool_addresses: List[str],
        interval: int = 3600
    ) -> None:
        """Main execution loop for DexBrain
        
        Args:
            blockchain: Blockchain to analyze
            pool_addresses: Pools to monitor
            interval: Update interval in seconds
        """
        self.logger.info(f"Starting DexBrain for {blockchain} with {len(pool_addresses)} pools")
        
        try:
            while not self._shutdown_event.is_set():
                # Aggregate data
                data = await self.aggregate_data(blockchain, pool_addresses)
                self.logger.info(f"Aggregated data for {len(data)} pools")
                
                # Train models if enough data
                count = await self.knowledge_base.get_insight_count(f"{blockchain}_liquidity")
                if count >= 10:
                    await self.train_models(f"{blockchain}_liquidity")
                
                # Wait for next iteration or shutdown
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(), 
                        timeout=interval
                    )
                except asyncio.TimeoutError:
                    continue
                    
        except Exception as e:
            self.logger.error(f"DexBrain execution failed: {e}")
            raise
        finally:
            self.logger.info("DexBrain shutting down")
    
    async def shutdown(self) -> None:
        """Gracefully shutdown DexBrain"""
        self.logger.info("Shutdown requested")
        self._shutdown_event.set()


async def main():
    """Main entry point"""
    dexbrain = DexBrain()
    
    # Setup shutdown handlers
    loop = asyncio.get_event_loop()
    for sig in (asyncio.SIGTERM, asyncio.SIGINT):
        loop.add_signal_handler(
            sig, lambda: asyncio.create_task(dexbrain.shutdown())
        )
    
    try:
        await dexbrain.run(
            blockchain='solana', 
            pool_addresses=['pool1', 'pool2']  # Replace with actual addresses
        )
    except KeyboardInterrupt:
        await dexbrain.shutdown()


if __name__ == "__main__":
    asyncio.run(main())