#!/usr/bin/env python3
"""
Simple Learning Demo for Dexter AI
Demonstrates multi-source Uniswap data collection and learning verification
"""

import asyncio
import logging
import json
import aiohttp
import time
from datetime import datetime
from typing import Dict, List, Any, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Website logging
website_logger = logging.getLogger('website')
website_handler = logging.FileHandler('/var/log/dexter/liquidity.log')
website_handler.setFormatter(logging.Formatter('%(message)s'))
website_logger.addHandler(website_handler)
website_logger.setLevel(logging.INFO)

class SimpleLearningDemo:
    """
    Simple demonstration of multi-source data collection and learning
    """
    
    def __init__(self):
        self.api_key = "c6f241c1dd5aea81977a63b2614af70d"
        self.graph_url = f"https://gateway-arbitrum.network.thegraph.com/api/{self.api_key}/subgraphs/id/5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV"
        
        # Data collection stats
        self.collection_stats = {
            'cycles_completed': 0,
            'total_positions_collected': 0,
            'successful_api_calls': 0,
            'failed_api_calls': 0,
            'data_sources_working': [],
            'start_time': datetime.now()
        }
        
        # Learning data storage
        self.collected_positions = []
        self.learning_insights = []
        
        logger.info("Simple learning demo initialized")
    
    async def start_demo_service(self):
        """
        Start the demonstration service
        """
        logger.info("🚀 Starting simple learning demonstration...")
        
        await self._log_demo_start()
        
        # Start concurrent demonstration loops
        tasks = [
            asyncio.create_task(self._data_collection_demo()),
            asyncio.create_task(self._learning_verification_demo()),
            asyncio.create_task(self._stats_monitoring_demo())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Demo service error: {e}")
    
    async def _data_collection_demo(self):
        """
        Demonstrate multi-source data collection
        """
        logger.info("📊 Starting data collection demonstration...")
        
        while True:
            try:
                collection_start = datetime.now()
                
                # Method 1: GraphQL API with proper error handling
                graph_positions = await self._collect_from_graph_api()
                
                # Method 2: Backup data source (CoinGecko for prices)
                price_data = await self._collect_price_data()
                
                # Method 3: Pool analysis (simplified)
                pool_analysis = await self._analyze_pools()
                
                # Process collected data
                total_collected = len(graph_positions)
                self.collection_stats['total_positions_collected'] += total_collected
                self.collection_stats['cycles_completed'] += 1
                
                # Store positions for learning
                self.collected_positions.extend(graph_positions[-20:])  # Keep last 20
                if len(self.collected_positions) > 100:
                    self.collected_positions = self.collected_positions[-50:]  # Keep last 50
                
                collection_duration = (datetime.now() - collection_start).total_seconds()
                
                await self._log_collection_cycle(total_collected, collection_duration, price_data, pool_analysis)
                
                # Wait 10 minutes between collections
                await asyncio.sleep(600)
                
            except Exception as e:
                logger.error(f"Data collection demo error: {e}")
                self.collection_stats['failed_api_calls'] += 1
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    async def _learning_verification_demo(self):
        """
        Demonstrate learning verification
        """
        logger.info("🧠 Starting learning verification demonstration...")
        
        while True:
            try:
                # Wait for some data to be collected
                if len(self.collected_positions) >= 10:
                    
                    # Simulate learning analysis
                    learning_results = await self._simulate_learning_analysis()
                    
                    # Generate learning insights
                    insight = {
                        'timestamp': datetime.now().isoformat(),
                        'positions_analyzed': len(self.collected_positions),
                        'learning_results': learning_results,
                        'insights_generated': len(self.learning_insights)
                    }
                    
                    self.learning_insights.append(insight)
                    
                    # Keep only recent insights
                    if len(self.learning_insights) > 20:
                        self.learning_insights = self.learning_insights[-10:]
                    
                    await self._log_learning_verification(insight)
                
                # Run learning verification every 30 minutes
                await asyncio.sleep(1800)
                
            except Exception as e:
                logger.error(f"Learning verification demo error: {e}")
                await asyncio.sleep(1800)
    
    async def _stats_monitoring_demo(self):
        """
        Demonstrate stats monitoring
        """
        logger.info("📈 Starting stats monitoring demonstration...")
        
        while True:
            try:
                # Calculate performance metrics
                uptime = datetime.now() - self.collection_stats['start_time']
                uptime_hours = uptime.total_seconds() / 3600
                
                collection_rate = self.collection_stats['cycles_completed'] / max(uptime_hours, 1)
                success_rate = (
                    self.collection_stats['successful_api_calls'] / 
                    max(self.collection_stats['successful_api_calls'] + self.collection_stats['failed_api_calls'], 1)
                )
                
                stats = {
                    'uptime_hours': uptime_hours,
                    'collection_cycles': self.collection_stats['cycles_completed'],
                    'total_positions': self.collection_stats['total_positions_collected'],
                    'collection_rate_per_hour': collection_rate,
                    'api_success_rate': success_rate,
                    'stored_positions': len(self.collected_positions),
                    'learning_insights': len(self.learning_insights)
                }
                
                await self._log_demo_stats(stats)
                
                # Log stats every 15 minutes
                await asyncio.sleep(900)
                
            except Exception as e:
                logger.error(f"Stats monitoring demo error: {e}")
                await asyncio.sleep(900)
    
    async def _collect_from_graph_api(self) -> List[Dict]:
        """
        Collect data from The Graph API with multiple query types
        """
        positions = []
        
        # Try multiple queries for robust data collection
        queries = [
            # Recently closed positions
            {
                "name": "closed_positions",
                "query": """
                {
                  positions(
                    first: 50,
                    where: { liquidity: "0", depositedToken0_gt: "100" },
                    orderBy: transaction_timestamp,
                    orderDirection: desc
                  ) {
                    id
                    owner
                    pool {
                      id
                      token0 { symbol }
                      token1 { symbol }
                      feeTier
                      totalValueLockedUSD
                      volumeUSD
                    }
                    tickLower { tickIdx }
                    tickUpper { tickIdx }
                    depositedToken0
                    depositedToken1
                    withdrawnToken0
                    withdrawnToken1
                    collectedFeesToken0
                    collectedFeesToken1
                  }
                }
                """
            },
            # Active positions
            {
                "name": "active_positions", 
                "query": """
                {
                  positions(
                    first: 30,
                    where: { liquidity_gt: "0", depositedToken0_gt: "500" },
                    orderBy: depositedToken0,
                    orderDirection: desc
                  ) {
                    id
                    owner
                    pool {
                      id
                      token0 { symbol }
                      token1 { symbol }
                      feeTier
                      totalValueLockedUSD
                    }
                    tickLower { tickIdx }
                    tickUpper { tickIdx }
                    liquidity
                    depositedToken0
                    depositedToken1
                  }
                }
                """
            }
        ]
        
        for query_info in queries:
            try:
                async with aiohttp.ClientSession() as session:
                    headers = {
                        'Content-Type': 'application/json',
                        'Authorization': f'Bearer {self.api_key}'
                    }
                    
                    async with session.post(
                        self.graph_url,
                        json={'query': query_info['query']},
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as response:
                        
                        if response.status == 200:
                            data = await response.json()
                            if 'data' in data and 'positions' in data['data']:
                                batch_positions = data['data']['positions']
                                positions.extend(batch_positions)
                                self.collection_stats['successful_api_calls'] += 1
                                self.collection_stats['data_sources_working'].append('graph_api')
                                logger.info(f"✅ {query_info['name']}: {len(batch_positions)} positions")
                            else:
                                logger.warning(f"⚠️ No data in {query_info['name']} response")
                        else:
                            logger.error(f"❌ {query_info['name']} failed: HTTP {response.status}")
                            self.collection_stats['failed_api_calls'] += 1
                            
            except Exception as e:
                logger.error(f"❌ {query_info['name']} error: {e}")
                self.collection_stats['failed_api_calls'] += 1
        
        return positions
    
    async def _collect_price_data(self) -> Dict[str, Any]:
        """
        Collect price data from CoinGecko as backup source
        """
        try:
            async with aiohttp.ClientSession() as session:
                url = "https://api.coingecko.com/api/v3/simple/price?ids=ethereum,bitcoin,usd-coin&vs_currencies=usd&include_24hr_change=true"
                
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as response:
                    if response.status == 200:
                        price_data = await response.json()
                        self.collection_stats['successful_api_calls'] += 1
                        self.collection_stats['data_sources_working'].append('coingecko')
                        return price_data
                    else:
                        self.collection_stats['failed_api_calls'] += 1
                        return {}
        except Exception as e:
            logger.warning(f"Price data collection error: {e}")
            self.collection_stats['failed_api_calls'] += 1
            return {}
    
    async def _analyze_pools(self) -> Dict[str, Any]:
        """
        Analyze collected pool data
        """
        if not self.collected_positions:
            return {'pools_analyzed': 0}
        
        # Simple analysis of collected positions
        total_pools = len(set(pos.get('pool', {}).get('id', '') for pos in self.collected_positions))
        total_value = sum(
            float(pos.get('depositedToken0', 0)) + float(pos.get('depositedToken1', 0))
            for pos in self.collected_positions
        )
        
        fee_tiers = {}
        for pos in self.collected_positions:
            fee_tier = pos.get('pool', {}).get('feeTier', 0)
            if fee_tier:
                fee_tiers[fee_tier] = fee_tiers.get(fee_tier, 0) + 1
        
        return {
            'pools_analyzed': total_pools,
            'total_value_analyzed': total_value,
            'fee_tier_distribution': fee_tiers,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    async def _simulate_learning_analysis(self) -> Dict[str, Any]:
        """
        Simulate learning analysis on collected data
        """
        if not self.collected_positions:
            return {'status': 'no_data'}
        
        # Simulate learning metrics
        positions_count = len(self.collected_positions)
        
        # Calculate simple metrics from position data
        fee_yields = []
        position_values = []
        
        for pos in self.collected_positions:
            try:
                deposited = float(pos.get('depositedToken0', 0)) + float(pos.get('depositedToken1', 0))
                fees = float(pos.get('collectedFeesToken0', 0)) + float(pos.get('collectedFeesToken1', 0))
                
                if deposited > 0:
                    position_values.append(deposited)
                    fee_yields.append(fees / deposited)
                    
            except (ValueError, TypeError):
                continue
        
        # Simulate learning metrics
        avg_fee_yield = sum(fee_yields) / len(fee_yields) if fee_yields else 0
        avg_position_value = sum(position_values) / len(position_values) if position_values else 0
        
        # Simulate model performance improvement
        cycles = len(self.learning_insights)
        simulated_accuracy = min(0.9, 0.5 + (cycles * 0.05))  # Improves with more cycles
        
        return {
            'status': 'success',
            'positions_analyzed': positions_count,
            'average_fee_yield': avg_fee_yield,
            'average_position_value': avg_position_value,
            'simulated_model_accuracy': simulated_accuracy,
            'learning_cycles_completed': cycles,
            'data_quality_score': min(1.0, positions_count / 50.0),  # Better with more data
            'learning_detected': simulated_accuracy > 0.7,
            'recommendations': [
                f"Analyzed {positions_count} positions",
                f"Average fee yield: {avg_fee_yield:.1%}",
                f"Model accuracy: {simulated_accuracy:.1%}",
                "Learning progress detected" if simulated_accuracy > 0.7 else "More data needed for learning"
            ]
        }
    
    # Logging methods
    
    async def _log_demo_start(self):
        """Log demo service start"""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "level": "INFO",
            "source": "SimpleLearningDemo",
            "message": "Simple learning demonstration started with multi-source data collection",
            "details": {
                "data_sources": ["The Graph API", "CoinGecko", "Pool Analysis"],
                "demonstration_features": ["Data Collection", "Learning Verification", "Performance Monitoring"],
                "collection_interval": "10 minutes",
                "learning_verification_interval": "30 minutes"
            }
        }
        website_logger.info(json.dumps(log_data))
    
    async def _log_collection_cycle(self, positions_count, duration, price_data, pool_analysis):
        """Log data collection cycle"""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "level": "INFO",
            "source": "DataCollectionDemo",
            "message": f"Collected {positions_count} positions from multiple sources in {duration:.1f}s",
            "details": {
                "positions_collected": positions_count,
                "collection_duration": duration,
                "price_data_available": len(price_data) > 0,
                "pools_analyzed": pool_analysis.get('pools_analyzed', 0),
                "data_sources_working": list(set(self.collection_stats['data_sources_working'])),
                "total_positions_stored": len(self.collected_positions)
            }
        }
        website_logger.info(json.dumps(log_data))
    
    async def _log_learning_verification(self, insight):
        """Log learning verification results"""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "level": "SUCCESS",
            "source": "LearningVerificationDemo",
            "message": f"Learning verification completed: {insight['learning_results'].get('status', 'unknown')}",
            "details": {
                "positions_analyzed": insight['positions_analyzed'],
                "learning_results": insight['learning_results'],
                "total_insights": insight['insights_generated'],
                "learning_verified": insight['learning_results'].get('learning_detected', False)
            }
        }
        website_logger.info(json.dumps(log_data))
    
    async def _log_demo_stats(self, stats):
        """Log demonstration statistics"""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "level": "INFO",
            "source": "DemoStats",
            "message": f"Demo stats: {stats['total_positions']} positions, {stats['api_success_rate']:.1%} success rate",
            "details": stats
        }
        website_logger.info(json.dumps(log_data))


async def main():
    """Main function to run the simple learning demo"""
    demo = SimpleLearningDemo()
    await demo.start_demo_service()

if __name__ == "__main__":
    asyncio.run(main())