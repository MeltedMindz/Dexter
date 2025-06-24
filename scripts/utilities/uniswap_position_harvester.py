#!/usr/bin/env python3
"""
Uniswap Position Harvester
Specifically designed to collect and parse closed Uniswap V3 positions for ML training
"""

import asyncio
import logging
import json
import os
from datetime import datetime, timedelta
from web3 import Web3
import aiohttp
from typing import Dict, List, Any, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Website logging
website_logger = logging.getLogger('website')
website_handler = logging.FileHandler('/var/log/dexter/liquidity.log')
website_handler.setFormatter(logging.Formatter('%(message)s'))
website_logger.addHandler(website_handler)
website_logger.setLevel(logging.INFO)

class UniswapPositionHarvester:
    """
    Harvests closed Uniswap V3 positions from Base network for ML training
    """
    
    def __init__(self):
        # API configurations - use environment variables
        self.alchemy_api_key = os.getenv('ALCHEMY_API_KEY')
        self.graph_api_key = os.getenv('GRAPH_API_KEY')
        
        if not self.alchemy_api_key or not self.graph_api_key:
            raise ValueError("Missing required environment variables: ALCHEMY_API_KEY, GRAPH_API_KEY")
        
        # Web3 setup
        self.w3 = Web3(Web3.HTTPProvider(f"https://base-mainnet.g.alchemy.com/v2/{self.alchemy_api_key}"))
        
        # Contract addresses
        self.npm_address = "0xC36442b4a4522E871399CD717aBDD847Ab11FE88"  # Position Manager
        self.graph_url = f"https://gateway-arbitrum.network.thegraph.com/api/{self.graph_api_key}/subgraphs/id/5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV"
        
        # Harvesting statistics
        self.harvest_stats = {
            'start_time': datetime.now(),
            'positions_harvested': 0,
            'high_quality_positions': 0,
            'graph_api_positions': 0,
            'alchemy_rpc_positions': 0,
            'last_harvest_time': None,
            'learning_samples_generated': 0
        }
        
        # ML training buffer
        self.training_samples = []
        self.max_samples = 500
        
        logger.info("Uniswap Position Harvester initialized")
    
    async def start_position_harvesting(self):
        """
        Start comprehensive position harvesting
        """
        logger.info("🌾 Starting Uniswap position harvesting for ML training...")
        
        await self._log_harvest_start()
        
        # Start harvesting tasks
        tasks = [
            asyncio.create_task(self._graph_position_harvesting()),
            asyncio.create_task(self._alchemy_event_harvesting()),
            asyncio.create_task(self._ml_sample_processing()),
            asyncio.create_task(self._harvest_monitoring())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Harvesting error: {e}")
            await self._handle_harvest_error(e)
    
    async def _graph_position_harvesting(self):
        """
        Harvest positions from The Graph API
        """
        logger.info("📊 Starting Graph API position harvesting...")
        
        while True:
            try:
                harvest_start = datetime.now()
                
                # Multiple queries for comprehensive data collection
                queries = [
                    # High-value closed positions with fees
                    {
                        "name": "high_value_closed",
                        "query": """
                        {
                          positions(
                            first: 100,
                            where: {
                              liquidity: "0",
                              collectedFeesToken0_gt: "100",
                              depositedToken0_gt: "1000"
                            },
                            orderBy: collectedFeesToken0,
                            orderDirection: desc
                          ) {
                            id
                            owner
                            pool {
                              id
                              token0 { symbol, decimals, id }
                              token1 { symbol, decimals, id }
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
                            transaction { timestamp, blockNumber, id }
                          }
                        }
                        """
                    },
                    # Recently closed positions
                    {
                        "name": "recently_closed",
                        "query": """
                        {
                          positions(
                            first: 150,
                            where: {
                              liquidity: "0",
                              depositedToken0_gt: "500"
                            },
                            orderBy: transaction__timestamp,
                            orderDirection: desc
                          ) {
                            id
                            owner
                            pool {
                              id
                              token0 { symbol }
                              token1 { symbol }
                              feeTier
                              sqrtPrice
                              tick
                              totalValueLockedUSD
                            }
                            tickLower { tickIdx }
                            tickUpper { tickIdx }
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
                    }
                ]
                
                total_positions = []
                
                for query_info in queries:
                    try:
                        positions = await self._execute_graph_query(query_info["query"])
                        if positions:
                            total_positions.extend(positions)
                            logger.info(f"✅ {query_info['name']}: {len(positions)} positions")
                    except Exception as e:
                        logger.warning(f"Query {query_info['name']} failed: {e}")
                        continue
                
                # Process harvested positions
                processed_count = await self._process_graph_positions(total_positions)
                
                self.harvest_stats['graph_api_positions'] += processed_count
                self.harvest_stats['positions_harvested'] += processed_count
                self.harvest_stats['last_harvest_time'] = harvest_start
                
                harvest_duration = (datetime.now() - harvest_start).total_seconds()
                
                await self._log_graph_harvest(len(total_positions), processed_count, harvest_duration)
                
                # Wait 20 minutes between graph harvests
                await asyncio.sleep(1200)
                
            except Exception as e:
                logger.error(f"Graph harvesting error: {e}")
                await asyncio.sleep(600)
    
    async def _alchemy_event_harvesting(self):
        """
        Harvest position events using Alchemy RPC
        """
        logger.info("⚡ Starting Alchemy event harvesting...")
        
        while True:
            try:
                harvest_start = datetime.now()
                
                # Get current block and scan range
                current_block = self.w3.eth.block_number
                scan_range = 5000  # Last 5000 blocks
                start_block = max(0, current_block - scan_range)
                
                # Look for Transfer events to zero address (burns)
                transfer_topic = self.w3.keccak(text="Transfer(address,address,uint256)").hex()
                zero_address = "0x0000000000000000000000000000000000000000000000000000000000000000"
                
                # Get burn events (closed positions)
                try:
                    burn_logs = self.w3.eth.get_logs({
                        "fromBlock": start_block,
                        "toBlock": current_block,
                        "address": self.npm_address,
                        "topics": [transfer_topic, None, zero_address]
                    })
                    
                    logger.info(f"Found {len(burn_logs)} position burn events")
                    
                    # Process events for position data
                    alchemy_positions = []
                    for log in burn_logs[:20]:  # Process first 20 to avoid rate limits
                        try:
                            token_id = int(log['topics'][3].hex(), 16)
                            
                            # Get block timestamp
                            block_info = self.w3.eth.get_block(log['blockNumber'])
                            
                            position_info = {
                                'token_id': token_id,
                                'block_number': log['blockNumber'],
                                'transaction_hash': log['transactionHash'].hex(),
                                'timestamp': datetime.fromtimestamp(block_info['timestamp']),
                                'source': 'alchemy_rpc'
                            }
                            
                            alchemy_positions.append(position_info)
                            
                        except Exception as e:
                            logger.warning(f"Failed to process burn event: {e}")
                            continue
                    
                    # Process Alchemy positions
                    processed_count = await self._process_alchemy_positions(alchemy_positions)
                    
                    self.harvest_stats['alchemy_rpc_positions'] += processed_count
                    self.harvest_stats['positions_harvested'] += processed_count
                    
                    harvest_duration = (datetime.now() - harvest_start).total_seconds()
                    
                    await self._log_alchemy_harvest(len(alchemy_positions), processed_count, harvest_duration)
                    
                except Exception as e:
                    logger.error(f"Failed to get burn logs: {e}")
                
                # Wait 25 minutes between Alchemy harvests
                await asyncio.sleep(1500)
                
            except Exception as e:
                logger.error(f"Alchemy harvesting error: {e}")
                await asyncio.sleep(900)
    
    async def _ml_sample_processing(self):
        """
        Process collected data into ML training samples
        """
        logger.info("🧠 Starting ML sample processing...")
        
        while True:
            try:
                if len(self.training_samples) >= 50:
                    # Analyze collected samples
                    high_quality_samples = [
                        sample for sample in self.training_samples
                        if sample.get('data_quality_score', 0) >= 0.8
                    ]
                    
                    # Calculate metrics
                    avg_fee_yield = 0
                    if high_quality_samples:
                        fee_yields = [s.get('fee_yield', 0) for s in high_quality_samples if s.get('fee_yield', 0) > 0]
                        if fee_yields:
                            avg_fee_yield = sum(fee_yields) / len(fee_yields)
                    
                    # Generate learning insight
                    learning_insight = {
                        'total_samples': len(self.training_samples),
                        'high_quality_samples': len(high_quality_samples),
                        'average_fee_yield': avg_fee_yield,
                        'data_sources': list(set(s.get('source', 'unknown') for s in self.training_samples)),
                        'sample_diversity': self._calculate_sample_diversity(),
                        'ready_for_training': len(high_quality_samples) >= 30
                    }
                    
                    self.harvest_stats['learning_samples_generated'] += 1
                    
                    await self._log_ml_processing(learning_insight)
                    
                    # Clear old samples to maintain memory
                    if len(self.training_samples) > self.max_samples:
                        self.training_samples = self.training_samples[-self.max_samples:]
                
                # Process every 15 minutes
                await asyncio.sleep(900)
                
            except Exception as e:
                logger.error(f"ML processing error: {e}")
                await asyncio.sleep(900)
    
    async def _harvest_monitoring(self):
        """
        Monitor harvesting progress and performance
        """
        logger.info("📈 Starting harvest monitoring...")
        
        while True:
            try:
                uptime = datetime.now() - self.harvest_stats['start_time']
                uptime_hours = uptime.total_seconds() / 3600
                
                harvest_rate = self.harvest_stats['positions_harvested'] / max(uptime_hours, 1)
                
                monitor_data = {
                    'uptime_hours': uptime_hours,
                    'total_positions_harvested': self.harvest_stats['positions_harvested'],
                    'high_quality_positions': self.harvest_stats['high_quality_positions'],
                    'graph_api_positions': self.harvest_stats['graph_api_positions'],
                    'alchemy_rpc_positions': self.harvest_stats['alchemy_rpc_positions'],
                    'harvest_rate_per_hour': harvest_rate,
                    'training_samples_buffer': len(self.training_samples),
                    'learning_samples_generated': self.harvest_stats['learning_samples_generated'],
                    'last_harvest': self.harvest_stats['last_harvest_time'].isoformat() if self.harvest_stats['last_harvest_time'] else None
                }
                
                await self._log_harvest_monitoring(monitor_data)
                
                # Monitor every 10 minutes
                await asyncio.sleep(600)
                
            except Exception as e:
                logger.error(f"Harvest monitoring error: {e}")
                await asyncio.sleep(600)
    
    async def _execute_graph_query(self, query: str) -> List[Dict]:
        """Execute GraphQL query"""
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
                            return data['data']['positions']
                    else:
                        logger.error(f"GraphQL query failed: {response.status}")
                        
        except Exception as e:
            logger.error(f"GraphQL query error: {e}")
        
        return []
    
    async def _process_graph_positions(self, positions: List[Dict]) -> int:
        """Process Graph API positions into training samples"""
        processed_count = 0
        
        for position in positions:
            try:
                # Calculate key metrics
                deposited_total = float(position.get('depositedToken0', 0)) + float(position.get('depositedToken1', 0))
                withdrawn_total = float(position.get('withdrawnToken0', 0)) + float(position.get('withdrawnToken1', 0))
                fees_total = float(position.get('collectedFeesToken0', 0)) + float(position.get('collectedFeesToken1', 0))
                
                if deposited_total > 0:
                    fee_yield = fees_total / deposited_total
                    pnl = (withdrawn_total + fees_total) - deposited_total
                    roi = pnl / deposited_total
                else:
                    fee_yield = 0
                    roi = 0
                
                # Create training sample
                training_sample = {
                    'source': 'graph_api',
                    'position_id': position['id'],
                    'pool_tokens': f"{position['pool']['token0']['symbol']}/{position['pool']['token1']['symbol']}",
                    'fee_tier': int(position['pool']['feeTier']),
                    'tick_lower': int(position['tickLower']['tickIdx']),
                    'tick_upper': int(position['tickUpper']['tickIdx']),
                    'range_width': int(position['tickUpper']['tickIdx']) - int(position['tickLower']['tickIdx']),
                    'deposited_total': deposited_total,
                    'fees_collected': fees_total,
                    'fee_yield': fee_yield,
                    'roi': roi,
                    'pool_tvl': float(position['pool'].get('totalValueLockedUSD', 0)),
                    'data_quality_score': 0.9,  # High quality from Graph
                    'timestamp': datetime.now().isoformat()
                }
                
                # Quality check
                if deposited_total >= 100 and fee_yield >= 0:  # Minimum thresholds
                    self.training_samples.append(training_sample)
                    processed_count += 1
                    
                    if fee_yield > 0.01:  # 1%+ yield considered high quality
                        self.harvest_stats['high_quality_positions'] += 1
                
            except Exception as e:
                logger.warning(f"Failed to process Graph position: {e}")
                continue
        
        return processed_count
    
    async def _process_alchemy_positions(self, positions: List[Dict]) -> int:
        """Process Alchemy positions into training samples"""
        processed_count = 0
        
        for position in positions:
            try:
                training_sample = {
                    'source': 'alchemy_rpc',
                    'token_id': position['token_id'],
                    'block_number': position['block_number'],
                    'transaction_hash': position['transaction_hash'],
                    'closed_timestamp': position['timestamp'].isoformat(),
                    'data_quality_score': 0.95,  # Very high quality - direct from chain
                    'timestamp': datetime.now().isoformat()
                }
                
                self.training_samples.append(training_sample)
                processed_count += 1
                
            except Exception as e:
                logger.warning(f"Failed to process Alchemy position: {e}")
                continue
        
        return processed_count
    
    def _calculate_sample_diversity(self) -> float:
        """Calculate diversity of training samples"""
        if not self.training_samples:
            return 0.0
        
        try:
            # Count unique characteristics
            sources = set(s.get('source', '') for s in self.training_samples)
            fee_tiers = set(s.get('fee_tier', 0) for s in self.training_samples if 'fee_tier' in s)
            pool_tokens = set(s.get('pool_tokens', '') for s in self.training_samples if 'pool_tokens' in s)
            
            diversity = (
                min(len(sources) / 2, 1.0) * 0.3 +  # Data source diversity
                min(len(fee_tiers) / 4, 1.0) * 0.4 +  # Fee tier diversity
                min(len(pool_tokens) / 10, 1.0) * 0.3  # Pool diversity
            )
            
            return diversity
            
        except:
            return 0.5
    
    # Logging methods
    
    async def _log_harvest_start(self):
        """Log harvest startup"""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "level": "INFO",
            "source": "PositionHarvester",
            "message": "Uniswap position harvesting started for ML training",
            "details": {
                "target": "Closed Uniswap V3 positions",
                "data_sources": ["The Graph API", "Alchemy RPC"],
                "ml_focus": "Fee yield prediction and position optimization",
                "quality_threshold": "80%+ data quality for training"
            }
        }
        website_logger.info(json.dumps(log_data))
    
    async def _log_graph_harvest(self, collected, processed, duration):
        """Log Graph API harvest results"""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "level": "SUCCESS",
            "source": "GraphHarvest",
            "message": f"Harvested {processed} high-quality positions from Graph API",
            "details": {
                "positions_collected": collected,
                "positions_processed": processed,
                "harvest_duration": duration,
                "data_source": "graph_api",
                "processing_rate": processed / max(duration, 1)
            }
        }
        website_logger.info(json.dumps(log_data))
    
    async def _log_alchemy_harvest(self, collected, processed, duration):
        """Log Alchemy harvest results"""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "level": "SUCCESS",
            "source": "AlchemyHarvest",
            "message": f"Harvested {processed} positions from Alchemy RPC events",
            "details": {
                "positions_collected": collected,
                "positions_processed": processed,
                "harvest_duration": duration,
                "data_source": "alchemy_rpc",
                "event_type": "position_burns"
            }
        }
        website_logger.info(json.dumps(log_data))
    
    async def _log_ml_processing(self, insight):
        """Log ML processing results"""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "level": "SUCCESS",
            "source": "MLProcessing",
            "message": f"Processed {insight['total_samples']} samples for ML training",
            "details": insight
        }
        website_logger.info(json.dumps(log_data))
    
    async def _log_harvest_monitoring(self, monitor_data):
        """Log harvest monitoring data"""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "level": "INFO",
            "source": "HarvestMonitoring",
            "message": f"Harvest progress: {monitor_data['total_positions_harvested']} positions, {monitor_data['harvest_rate_per_hour']:.1f}/hour",
            "details": monitor_data
        }
        website_logger.info(json.dumps(log_data))
    
    async def _handle_harvest_error(self, error):
        """Handle harvest errors"""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "level": "ERROR",
            "source": "HarvestError",
            "message": f"Harvest error: {error}",
            "details": {
                "error_type": type(error).__name__,
                "error_message": str(error),
                "harvest_stats": self.harvest_stats
            }
        }
        website_logger.info(json.dumps(log_data))


async def main():
    """Run the position harvester"""
    harvester = UniswapPositionHarvester()
    await harvester.start_position_harvesting()

if __name__ == "__main__":
    asyncio.run(main())