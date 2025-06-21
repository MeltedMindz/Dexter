#!/usr/bin/env python3
"""
Enhanced Data Pipeline for Dexter AI
Combines The Graph, Alchemy RPC, and multiple data sources for comprehensive position collection
"""

import asyncio
import logging
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path

# Import our data collection components
from .alchemy_position_collector import AlchemyPositionCollector, AlchemyPositionData

logger = logging.getLogger(__name__)

# Website logging
website_logger = logging.getLogger('website')
website_handler = logging.FileHandler('/var/log/dexter/liquidity.log')
website_handler.setFormatter(logging.Formatter('%(message)s'))
website_logger.addHandler(website_handler)
website_logger.setLevel(logging.INFO)

class EnhancedDataPipeline:
    """
    Enhanced data pipeline that combines multiple sources:
    1. The Graph API for indexed historical data
    2. Alchemy RPC for direct on-chain data
    3. Price feeds from CoinGecko/DeFiLlama
    4. Real-time event monitoring
    """
    
    def __init__(self):
        # API keys
        self.graph_api_key = "c6f241c1dd5aea81977a63b2614af70d"
        self.alchemy_api_key = os.getenv('ALCHEMY_API_KEY', 'demo')
        
        # Initialize collectors
        self.alchemy_collector = AlchemyPositionCollector(self.alchemy_api_key)
        
        # Graph API endpoint
        self.graph_url = f"https://gateway-arbitrum.network.thegraph.com/api/{self.graph_api_key}/subgraphs/id/5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV"
        
        # Pipeline statistics
        self.pipeline_stats = {
            'start_time': datetime.now(),
            'total_positions_collected': 0,
            'alchemy_positions': 0,
            'graph_positions': 0,
            'data_quality_score': 0.0,
            'learning_cycles_triggered': 0,
            'last_successful_collection': None
        }
        
        # Storage for ML training
        self.training_data_buffer = []
        self.max_buffer_size = 1000
        
        logger.info("Enhanced data pipeline initialized with Alchemy + Graph API")
    
    async def run_enhanced_pipeline(self):
        """
        Run the enhanced data collection pipeline
        """
        logger.info("🚀 Starting enhanced data pipeline with multiple sources...")
        
        await self._log_pipeline_start()
        
        # Start concurrent data collection tasks
        tasks = [
            asyncio.create_task(self._alchemy_collection_loop()),
            asyncio.create_task(self._graph_collection_loop()),
            asyncio.create_task(self._data_quality_monitoring()),
            asyncio.create_task(self._ml_training_trigger()),
            asyncio.create_task(self._pipeline_health_monitoring())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Enhanced pipeline error: {e}")
            await self._handle_pipeline_error(e)
    
    async def _alchemy_collection_loop(self):
        """
        Collect position data using Alchemy RPC
        """
        logger.info("⚡ Starting Alchemy RPC collection loop...")
        
        while True:
            try:
                collection_start = datetime.now()
                
                # Collect closed positions from last 10,000 blocks (~2-3 hours on Base)
                closed_positions = await self.alchemy_collector.collect_closed_positions(
                    block_range=10000,
                    limit=100
                )
                
                # Process and store positions
                processed_count = await self._process_alchemy_positions(closed_positions)
                
                self.pipeline_stats['alchemy_positions'] += processed_count
                self.pipeline_stats['total_positions_collected'] += processed_count
                
                collection_duration = (datetime.now() - collection_start).total_seconds()
                
                await self._log_alchemy_collection(closed_positions, processed_count, collection_duration)
                
                # Wait 30 minutes between Alchemy collections (to manage RPC rate limits)
                await asyncio.sleep(1800)
                
            except Exception as e:
                logger.error(f"Alchemy collection error: {e}")
                await asyncio.sleep(900)  # Wait 15 minutes on error
    
    async def _graph_collection_loop(self):
        """
        Collect position data using The Graph API
        """
        logger.info("📊 Starting Graph API collection loop...")
        
        while True:
            try:
                collection_start = datetime.now()
                
                # Collect from multiple Graph queries
                graph_positions = await self._collect_from_graph_api()
                
                # Process and store positions
                processed_count = await self._process_graph_positions(graph_positions)
                
                self.pipeline_stats['graph_positions'] += processed_count
                self.pipeline_stats['total_positions_collected'] += processed_count
                
                collection_duration = (datetime.now() - collection_start).total_seconds()
                
                await self._log_graph_collection(graph_positions, processed_count, collection_duration)
                
                # Wait 15 minutes between Graph collections
                await asyncio.sleep(900)
                
            except Exception as e:
                logger.error(f"Graph collection error: {e}")
                await asyncio.sleep(600)  # Wait 10 minutes on error
    
    async def _collect_from_graph_api(self) -> List[Dict]:
        """
        Collect positions from The Graph API
        """
        import aiohttp
        
        positions = []
        
        # Query for high-value closed positions
        query = """
        {
          positions(
            first: 200,
            where: {
              liquidity: "0",
              depositedToken0_gt: "1000"
            },
            orderBy: collectedFeesToken0,
            orderDirection: desc
          ) {
            id
            owner
            pool {
              id
              token0 { symbol, decimals }
              token1 { symbol, decimals }
              feeTier
              sqrtPrice
              tick
              totalValueLockedUSD
              volumeUSD
            }
            tickLower { tickIdx }
            tickUpper { tickIdx }
            liquidity
            depositedToken0
            depositedToken1
            withdrawnToken0
            withdrawnToken1
            collectedFeesToken0
            collectedFeesToken1
            transaction { timestamp, blockNumber }
          }
        }
        """
        
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {self.graph_api_key}'
                }
                
                async with session.post(
                    self.graph_url,
                    json={'query': query},
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    
                    if response.status == 200:
                        data = await response.json()
                        if 'data' in data and 'positions' in data['data']:
                            positions = data['data']['positions']
                            logger.info(f"Graph API: Retrieved {len(positions)} positions")
                    else:
                        logger.error(f"Graph API error: HTTP {response.status}")
                        
        except Exception as e:
            logger.error(f"Graph API request failed: {e}")
        
        return positions
    
    async def _process_alchemy_positions(self, positions: List[AlchemyPositionData]) -> int:
        """
        Process Alchemy positions for ML training
        """
        processed_count = 0
        
        for position in positions:
            try:
                # Convert to training format
                training_sample = {
                    'source': 'alchemy_rpc',
                    'position_id': f"alchemy_{position.token_id}",
                    'token0': position.token0,
                    'token1': position.token1,
                    'fee_tier': position.fee_tier,
                    'tick_lower': position.tick_lower,
                    'tick_upper': position.tick_upper,
                    'range_width': position.range_width,
                    'liquidity': float(position.liquidity),
                    'tokens_owed0': float(position.tokens_owed0),
                    'tokens_owed1': float(position.tokens_owed1),
                    'closed_at': position.closed_at_timestamp.isoformat(),
                    'is_closed': position.is_closed,
                    'data_quality': 0.95,  # High quality - direct from chain
                    'timestamp': datetime.now().isoformat()
                }
                
                # Add to training buffer
                self.training_data_buffer.append(training_sample)
                processed_count += 1
                
                # Maintain buffer size
                if len(self.training_data_buffer) > self.max_buffer_size:
                    self.training_data_buffer = self.training_data_buffer[-self.max_buffer_size:]
                
            except Exception as e:
                logger.warning(f"Failed to process Alchemy position: {e}")
                continue
        
        return processed_count
    
    async def _process_graph_positions(self, positions: List[Dict]) -> int:
        """
        Process Graph positions for ML training
        """
        processed_count = 0
        
        for position in positions:
            try:
                # Convert to training format
                training_sample = {
                    'source': 'graph_api',
                    'position_id': position['id'],
                    'token0': position['pool']['token0']['symbol'],
                    'token1': position['pool']['token1']['symbol'],
                    'fee_tier': int(position['pool']['feeTier']),
                    'tick_lower': int(position['tickLower']['tickIdx']),
                    'tick_upper': int(position['tickUpper']['tickIdx']),
                    'range_width': int(position['tickUpper']['tickIdx']) - int(position['tickLower']['tickIdx']),
                    'liquidity': float(position['liquidity']),
                    'deposited_token0': float(position['depositedToken0']),
                    'deposited_token1': float(position['depositedToken1']),
                    'withdrawn_token0': float(position['withdrawnToken0']),
                    'withdrawn_token1': float(position['withdrawnToken1']),
                    'collected_fees_token0': float(position['collectedFeesToken0']),
                    'collected_fees_token1': float(position['collectedFeesToken1']),
                    'pool_tvl': float(position['pool'].get('totalValueLockedUSD', 0)),
                    'is_closed': float(position['liquidity']) == 0,
                    'data_quality': 0.85,  # Good quality - indexed data
                    'timestamp': datetime.now().isoformat()
                }
                
                # Calculate derived metrics
                total_deposited = training_sample['deposited_token0'] + training_sample['deposited_token1']
                total_fees = training_sample['collected_fees_token0'] + training_sample['collected_fees_token1']
                
                if total_deposited > 0:
                    training_sample['fee_yield'] = total_fees / total_deposited
                    training_sample['apr_estimate'] = (total_fees / total_deposited) * 365  # Simplified
                
                # Add to training buffer
                self.training_data_buffer.append(training_sample)
                processed_count += 1
                
            except Exception as e:
                logger.warning(f"Failed to process Graph position: {e}")
                continue
        
        return processed_count
    
    async def _data_quality_monitoring(self):
        """
        Monitor data quality across sources
        """
        logger.info("🔍 Starting data quality monitoring...")
        
        while True:
            try:
                if self.training_data_buffer:
                    # Analyze data quality
                    alchemy_data = [d for d in self.training_data_buffer if d['source'] == 'alchemy_rpc']
                    graph_data = [d for d in self.training_data_buffer if d['source'] == 'graph_api']
                    
                    quality_metrics = {
                        'total_samples': len(self.training_data_buffer),
                        'alchemy_samples': len(alchemy_data),
                        'graph_samples': len(graph_data),
                        'alchemy_percentage': len(alchemy_data) / len(self.training_data_buffer),
                        'graph_percentage': len(graph_data) / len(self.training_data_buffer),
                        'data_freshness': self._calculate_data_freshness(),
                        'data_diversity': self._calculate_data_diversity()
                    }
                    
                    # Calculate overall quality score
                    quality_score = (
                        0.4 * min(quality_metrics['alchemy_percentage'], 0.5) * 2 +  # Balance of sources
                        0.3 * quality_metrics['data_freshness'] +
                        0.3 * quality_metrics['data_diversity']
                    )
                    
                    self.pipeline_stats['data_quality_score'] = quality_score
                    
                    await self._log_data_quality(quality_metrics, quality_score)
                
                # Monitor every 10 minutes
                await asyncio.sleep(600)
                
            except Exception as e:
                logger.error(f"Data quality monitoring error: {e}")
                await asyncio.sleep(600)
    
    async def _ml_training_trigger(self):
        """
        Trigger ML training when sufficient quality data is available
        """
        logger.info("🧠 Starting ML training trigger monitor...")
        
        while True:
            try:
                # Check if we have enough quality data
                if len(self.training_data_buffer) >= 100:
                    # Check data quality
                    if self.pipeline_stats['data_quality_score'] >= 0.7:
                        # Trigger learning
                        await self._trigger_ml_learning()
                        self.pipeline_stats['learning_cycles_triggered'] += 1
                        
                        # Clear old data after training
                        self.training_data_buffer = self.training_data_buffer[-500:]
                    else:
                        logger.info("Data quality insufficient for training")
                
                # Check every 30 minutes
                await asyncio.sleep(1800)
                
            except Exception as e:
                logger.error(f"ML training trigger error: {e}")
                await asyncio.sleep(1800)
    
    async def _pipeline_health_monitoring(self):
        """
        Monitor overall pipeline health
        """
        logger.info("💓 Starting pipeline health monitoring...")
        
        while True:
            try:
                uptime = datetime.now() - self.pipeline_stats['start_time']
                uptime_hours = uptime.total_seconds() / 3600
                
                positions_per_hour = self.pipeline_stats['total_positions_collected'] / max(uptime_hours, 1)
                
                health_metrics = {
                    'uptime_hours': uptime_hours,
                    'total_positions': self.pipeline_stats['total_positions_collected'],
                    'alchemy_positions': self.pipeline_stats['alchemy_positions'],
                    'graph_positions': self.pipeline_stats['graph_positions'],
                    'positions_per_hour': positions_per_hour,
                    'data_quality_score': self.pipeline_stats['data_quality_score'],
                    'learning_cycles': self.pipeline_stats['learning_cycles_triggered'],
                    'buffer_size': len(self.training_data_buffer)
                }
                
                await self._log_pipeline_health(health_metrics)
                
                # Check every 15 minutes
                await asyncio.sleep(900)
                
            except Exception as e:
                logger.error(f"Pipeline health monitoring error: {e}")
                await asyncio.sleep(900)
    
    def _calculate_data_freshness(self) -> float:
        """Calculate how fresh the data is"""
        if not self.training_data_buffer:
            return 0.0
        
        try:
            # Get timestamps of recent data
            recent_data = self.training_data_buffer[-50:]
            current_time = datetime.now()
            
            fresh_count = 0
            for data in recent_data:
                timestamp = datetime.fromisoformat(data['timestamp'])
                age = (current_time - timestamp).total_seconds() / 3600  # Hours
                if age < 1:  # Less than 1 hour old
                    fresh_count += 1
            
            return fresh_count / len(recent_data)
            
        except:
            return 0.5
    
    def _calculate_data_diversity(self) -> float:
        """Calculate diversity of data (fee tiers, ranges, etc)"""
        if not self.training_data_buffer:
            return 0.0
        
        try:
            # Count unique characteristics
            fee_tiers = set(d['fee_tier'] for d in self.training_data_buffer)
            unique_pools = set((d.get('token0', ''), d.get('token1', '')) for d in self.training_data_buffer)
            range_widths = set(d.get('range_width', 0) // 1000 for d in self.training_data_buffer)  # Bucket by 1000s
            
            diversity_score = (
                min(len(fee_tiers) / 4, 1.0) * 0.3 +  # Up to 4 fee tiers
                min(len(unique_pools) / 20, 1.0) * 0.4 +  # Up to 20 unique pools
                min(len(range_widths) / 10, 1.0) * 0.3  # Up to 10 range width buckets
            )
            
            return diversity_score
            
        except:
            return 0.5
    
    async def _trigger_ml_learning(self):
        """Trigger ML learning with collected data"""
        logger.info(f"🎯 Triggering ML learning with {len(self.training_data_buffer)} samples")
        
        # Here you would integrate with your ML training pipeline
        # For now, we'll just log the trigger
        
        await self._log_ml_trigger(len(self.training_data_buffer))
    
    # Logging methods
    
    async def _log_pipeline_start(self):
        """Log pipeline startup"""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "level": "INFO",
            "source": "EnhancedDataPipeline",
            "message": "Enhanced data pipeline started with Alchemy RPC + Graph API",
            "details": {
                "data_sources": ["Alchemy RPC (Direct)", "The Graph API (Indexed)"],
                "capabilities": ["Closed positions", "Active positions", "Lifecycle events", "ML training"],
                "alchemy_features": ["On-chain data", "Block-level precision", "Event logs"],
                "graph_features": ["Historical data", "Complex queries", "Aggregations"]
            }
        }
        website_logger.info(json.dumps(log_data))
    
    async def _log_alchemy_collection(self, positions, processed, duration):
        """Log Alchemy collection results"""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "level": "SUCCESS",
            "source": "AlchemyCollection",
            "message": f"Collected {len(positions)} positions from Alchemy RPC in {duration:.1f}s",
            "details": {
                "positions_collected": len(positions),
                "positions_processed": processed,
                "collection_duration": duration,
                "data_source": "alchemy_direct_rpc",
                "blocks_scanned": self.alchemy_collector.collection_stats['blocks_scanned'],
                "rpc_calls": self.alchemy_collector.collection_stats['rpc_calls']
            }
        }
        website_logger.info(json.dumps(log_data))
    
    async def _log_graph_collection(self, positions, processed, duration):
        """Log Graph collection results"""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "level": "SUCCESS",
            "source": "GraphCollection",
            "message": f"Collected {len(positions)} positions from Graph API in {duration:.1f}s",
            "details": {
                "positions_collected": len(positions),
                "positions_processed": processed,
                "collection_duration": duration,
                "data_source": "graph_api_indexed"
            }
        }
        website_logger.info(json.dumps(log_data))
    
    async def _log_data_quality(self, metrics, score):
        """Log data quality metrics"""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "level": "INFO" if score >= 0.7 else "WARNING",
            "source": "DataQuality",
            "message": f"Data quality score: {score:.1%}",
            "details": {
                "quality_metrics": metrics,
                "overall_score": score,
                "ready_for_training": score >= 0.7
            }
        }
        website_logger.info(json.dumps(log_data))
    
    async def _log_pipeline_health(self, metrics):
        """Log pipeline health metrics"""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "level": "INFO",
            "source": "PipelineHealth",
            "message": f"Pipeline health: {metrics['total_positions']} positions, {metrics['positions_per_hour']:.1f}/hour",
            "details": metrics
        }
        website_logger.info(json.dumps(log_data))
    
    async def _log_ml_trigger(self, sample_count):
        """Log ML training trigger"""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "level": "SUCCESS",
            "source": "MLTrigger",
            "message": f"ML training triggered with {sample_count} samples",
            "details": {
                "sample_count": sample_count,
                "data_quality_score": self.pipeline_stats['data_quality_score'],
                "alchemy_samples": len([d for d in self.training_data_buffer if d['source'] == 'alchemy_rpc']),
                "graph_samples": len([d for d in self.training_data_buffer if d['source'] == 'graph_api'])
            }
        }
        website_logger.info(json.dumps(log_data))
    
    async def _handle_pipeline_error(self, error):
        """Handle pipeline errors"""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "level": "ERROR",
            "source": "PipelineError",
            "message": f"Pipeline error: {error}",
            "details": {
                "error_type": type(error).__name__,
                "error_message": str(error),
                "pipeline_stats": self.pipeline_stats
            }
        }
        website_logger.info(json.dumps(log_data))


async def main():
    """Run the enhanced data pipeline"""
    pipeline = EnhancedDataPipeline()
    await pipeline.run_enhanced_pipeline()

if __name__ == "__main__":
    asyncio.run(main())