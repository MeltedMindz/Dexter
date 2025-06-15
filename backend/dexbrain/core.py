import asyncio
import logging
from typing import List, Dict, Any, Optional
import numpy as np
from .blockchain.solana_connector import SolanaConnector
from .blockchain.base_connector import BlockchainConnector
from .models.knowledge_base import KnowledgeBase
from .models.ml_models import DeFiMLEngine
from .config import Config
from .auth import APIKeyManager
from .agent_registry import AgentRegistry
from .performance_scoring import PerformanceScorer
from .data_quality import DataQualityEngine
from .schemas import AgentSubmission


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
        
        # Global intelligence network components
        self.api_key_manager = APIKeyManager()
        self.agent_registry = AgentRegistry()
        self.performance_scorer = PerformanceScorer()
        self.data_quality_engine = DataQualityEngine()
    
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
    
    # Global Intelligence Network Methods
    
    async def process_agent_submission(self, submission: AgentSubmission) -> Dict[str, Any]:
        """Process data submission from external agent
        
        Args:
            submission: Agent data submission
            
        Returns:
            Processing results with quality report
        """
        try:
            # Validate data quality
            quality_report = self.data_quality_engine.validate_submission(submission)
            
            if not quality_report.is_accepted:
                self.logger.warning(f"Submission rejected for agent {submission.agent_id}: Quality score {quality_report.overall_score}")
                return {
                    'status': 'rejected',
                    'quality_report': quality_report,
                    'message': 'Submission failed quality checks'
                }
            
            # Store validated data in knowledge base
            category = f"{submission.blockchain}_liquidity"
            for position in submission.positions:
                insight_data = {
                    'agent_id': submission.agent_id,
                    'pool_address': position.pool.address,
                    'blockchain': submission.blockchain,
                    'dex_protocol': submission.dex_protocol,
                    'total_liquidity': position.position_value_usd,
                    'volume_24h': position.pool.volume_24h_usd,
                    'fee_tier': position.pool.fee_tier,
                    'token0_reserves': position.token0_amount,
                    'token1_reserves': position.token1_amount,
                    'submission_id': submission.submission_id,
                    'quality_score': quality_report.overall_score
                }
                
                await self.knowledge_base.store_insight(category, insight_data)
            
            # Update agent metrics
            self.agent_registry.update_agent_activity(submission.agent_id)
            
            # Calculate performance score if we have enough data
            agent_profile = self.performance_scorer.calculate_agent_score(
                submission.agent_id,
                submission.performance_metrics,
                submission.positions
            )
            
            # Update agent registry with new metrics
            metrics_update = {
                'total_positions': agent_profile.total_positions,
                'total_volume': agent_profile.total_volume,
                'average_apr': agent_profile.average_apr,
                'win_rate': agent_profile.win_rate,
                'sharpe_ratio': agent_profile.sharpe_ratio,
                'max_drawdown': agent_profile.max_drawdown
            }
            self.agent_registry.update_agent_metrics(submission.agent_id, metrics_update)
            
            self.logger.info(f"Successfully processed submission from agent {submission.agent_id}")
            
            return {
                'status': 'accepted',
                'quality_report': quality_report,
                'performance_profile': agent_profile,
                'message': 'Submission processed successfully'
            }
            
        except Exception as e:
            self.logger.error(f"Failed to process submission from {submission.agent_id}: {e}")
            return {
                'status': 'error',
                'message': f'Processing failed: {str(e)}'
            }
    
    async def get_intelligence_feed(
        self,
        agent_id: str,
        blockchain: str = 'base',
        category: Optional[str] = None,
        limit: int = 100
    ) -> Dict[str, Any]:
        """Provide intelligence feed to external agent
        
        Args:
            agent_id: Requesting agent ID
            blockchain: Target blockchain
            category: Specific category filter
            limit: Maximum number of insights
            
        Returns:
            Intelligence data and predictions
        """
        try:
            category = category or f"{blockchain}_liquidity"
            
            # Get recent insights
            insights = await self.knowledge_base.retrieve_insights(category, limit=limit)
            
            # Filter high-quality insights
            quality_threshold = 70.0
            high_quality_insights = [
                insight for insight in insights 
                if insight.get('quality_score', 0) >= quality_threshold
            ]
            
            # Get network statistics
            network_stats = self.agent_registry.get_network_statistics()
            
            # Get top performers for benchmarking
            all_agents = self.agent_registry.list_agents()
            if all_agents:
                agent_profiles = []
                for agent in all_agents[:50]:  # Process top 50 agents
                    try:
                        # This would be more efficient with cached profiles
                        profile = self.performance_scorer.calculate_agent_score(
                            agent.agent_id, [], []  # Would need actual metrics
                        )
                        agent_profiles.append(profile)
                    except:
                        continue
                
                benchmarks = self.performance_scorer.get_network_benchmarks(agent_profiles)
            else:
                benchmarks = {}
            
            # Generate predictions for popular pools
            predictions = []
            pool_addresses = list(set([
                insight.get('pool_address') for insight in high_quality_insights[-20:]
                if insight.get('pool_address')
            ]))
            
            for pool_address in pool_addresses[:5]:  # Limit to 5 predictions
                try:
                    # Find recent data for this pool
                    pool_insights = [
                        i for i in high_quality_insights 
                        if i.get('pool_address') == pool_address
                    ]
                    
                    if pool_insights:
                        latest = pool_insights[-1]
                        pool_data = {
                            'pool_address': pool_address,
                            'total_liquidity': latest.get('total_liquidity', 0),
                            'volume_24h': latest.get('volume_24h', 0),
                            'fee_tier': latest.get('fee_tier', 0),
                            'token0_reserves': latest.get('token0_reserves', 0),
                            'token1_reserves': latest.get('token1_reserves', 0)
                        }
                        
                        predicted_apr = await self.predict_liquidity_metrics(pool_data)
                        
                        predictions.append({
                            'pool_address': pool_address,
                            'predicted_apr': predicted_apr,
                            'confidence': min(len(pool_insights) / 10, 1.0),
                            'data_points': len(pool_insights)
                        })
                except Exception as e:
                    self.logger.debug(f"Prediction failed for pool {pool_address}: {e}")
                    continue
            
            return {
                'insights': high_quality_insights,
                'predictions': predictions,
                'network_stats': network_stats,
                'benchmarks': benchmarks,
                'metadata': {
                    'blockchain': blockchain,
                    'category': category,
                    'total_insights': len(insights),
                    'high_quality_insights': len(high_quality_insights),
                    'quality_threshold': quality_threshold,
                    'timestamp': asyncio.get_event_loop().time()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate intelligence feed for {agent_id}: {e}")
            return {
                'insights': [],
                'predictions': [],
                'network_stats': {},
                'benchmarks': {},
                'error': str(e)
            }


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