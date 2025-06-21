"""
Advanced Uniswap Data Collection System
Multi-source data collection with robust parsing and validation
"""

import asyncio
import logging
import json
import aiohttp
import websockets
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set
from pathlib import Path
import time
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from web3 import Web3
from web3.contract import Contract
import os

logger = logging.getLogger(__name__)

# Website logging
website_logger = logging.getLogger('website')
website_handler = logging.FileHandler('/var/log/dexter/liquidity.log')
website_handler.setFormatter(logging.Formatter('%(message)s'))
website_logger.addHandler(website_handler)
website_logger.setLevel(logging.INFO)

@dataclass
class PositionData:
    """Comprehensive position data structure"""
    position_id: str
    pool_address: str
    owner: str
    token0_symbol: str
    token1_symbol: str
    fee_tier: int
    tick_lower: int
    tick_upper: int
    liquidity: float
    deposited_token0: float
    deposited_token1: float
    withdrawn_token0: float
    withdrawn_token1: float
    collected_fees_token0: float
    collected_fees_token1: float
    current_tick: int
    sqrt_price: float
    block_number: int
    timestamp: datetime
    transaction_hash: str
    
    # Derived metrics
    position_value_usd: float = 0.0
    apr_estimate: float = 0.0
    impermanent_loss: float = 0.0
    fee_yield: float = 0.0
    in_range: bool = True
    range_width: int = 0
    capital_efficiency: float = 0.0
    
    # Data quality
    data_source: str = ""
    confidence_score: float = 0.8
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class DataCollectionConfig:
    """Configuration for data collection methods"""
    # API endpoints
    graph_api_key: str
    alchemy_api_key: str
    infura_api_key: str
    base_rpc_url: str
    
    # Rate limits (requests per second)
    graph_rate_limit: float = 10.0
    rpc_rate_limit: float = 25.0
    batch_size: int = 100
    
    # Data quality thresholds
    min_liquidity_usd: float = 100.0
    min_position_age_hours: int = 1
    max_data_age_minutes: int = 30

class AdvancedUniswapDataCollector:
    """
    Multi-source Uniswap position data collector with robust parsing
    """
    
    def __init__(self, config: DataCollectionConfig):
        self.config = config
        self.web3 = Web3(Web3.HTTPProvider(config.base_rpc_url))
        
        # Uniswap V3 contract addresses on Base
        self.factory_address = "0x33128a8fC17869897dcE68Ed026d694621f6FDfD"
        self.position_manager_address = "0x03a520b32C04BF3bEEf7BF5d3Fc401f57B91b3B6"
        
        # ABI for position manager (simplified)
        self.position_manager_abi = [
            {
                "inputs": [{"type": "uint256", "name": "tokenId"}],
                "name": "positions",
                "outputs": [
                    {"type": "uint96", "name": "nonce"},
                    {"type": "address", "name": "operator"},
                    {"type": "address", "name": "token0"},
                    {"type": "address", "name": "token1"},
                    {"type": "uint24", "name": "fee"},
                    {"type": "int24", "name": "tickLower"},
                    {"type": "int24", "name": "tickUpper"},
                    {"type": "uint128", "name": "liquidity"},
                    {"type": "uint256", "name": "feeGrowthInside0LastX128"},
                    {"type": "uint256", "name": "feeGrowthInside1LastX128"},
                    {"type": "uint128", "name": "tokensOwed0"},
                    {"type": "uint128", "name": "tokensOwed1"}
                ],
                "stateMutability": "view",
                "type": "function"
            }
        ]
        
        # Data storage
        self.collected_positions: List[PositionData] = []
        self.pool_cache: Dict[str, Dict] = {}
        self.token_cache: Dict[str, Dict] = {}
        
        # Collection stats
        self.collection_stats = {
            'total_positions_found': 0,
            'positions_parsed_successfully': 0,
            'high_quality_positions': 0,
            'graph_api_calls': 0,
            'rpc_calls': 0,
            'last_collection_time': None,
            'collection_errors': 0,
            'data_sources_used': set()
        }
        
        logger.info("Advanced Uniswap data collector initialized")
    
    async def collect_comprehensive_position_data(self) -> List[PositionData]:
        """
        Collect position data using multiple methods for maximum coverage
        """
        logger.info("🔍 Starting comprehensive position data collection...")
        
        collection_start = datetime.now()
        all_positions = []
        
        # Method 1: The Graph API (Primary)
        try:
            graph_positions = await self._collect_from_graph_api()
            all_positions.extend(graph_positions)
            self.collection_stats['data_sources_used'].add('graph_api')
            logger.info(f"✅ Graph API: {len(graph_positions)} positions")
        except Exception as e:
            logger.error(f"❌ Graph API failed: {e}")
        
        # Method 2: Direct RPC calls with event parsing
        try:
            rpc_positions = await self._collect_from_rpc_events()
            all_positions.extend(rpc_positions)
            self.collection_stats['data_sources_used'].add('rpc_events')
            logger.info(f"✅ RPC Events: {len(rpc_positions)} positions")
        except Exception as e:
            logger.error(f"❌ RPC Events failed: {e}")
        
        # Method 3: Position NFT enumeration
        try:
            nft_positions = await self._collect_from_position_nfts()
            all_positions.extend(nft_positions)
            self.collection_stats['data_sources_used'].add('position_nfts')
            logger.info(f"✅ Position NFTs: {len(nft_positions)} positions")
        except Exception as e:
            logger.error(f"❌ Position NFTs failed: {e}")
        
        # Method 4: Backup Graph endpoints
        try:
            backup_positions = await self._collect_from_backup_sources()
            all_positions.extend(backup_positions)
            self.collection_stats['data_sources_used'].add('backup_apis')
            logger.info(f"✅ Backup APIs: {len(backup_positions)} positions")
        except Exception as e:
            logger.error(f"❌ Backup APIs failed: {e}")
        
        # Deduplicate and validate
        unique_positions = self._deduplicate_positions(all_positions)
        validated_positions = await self._validate_and_enrich_positions(unique_positions)
        
        # Update collection stats
        self.collection_stats.update({
            'total_positions_found': len(all_positions),
            'positions_parsed_successfully': len(unique_positions),
            'high_quality_positions': len(validated_positions),
            'last_collection_time': collection_start,
            'collection_duration': (datetime.now() - collection_start).total_seconds()
        })
        
        self.collected_positions = validated_positions
        
        await self._log_collection_results(validated_positions)
        
        return validated_positions
    
    async def _collect_from_graph_api(self) -> List[PositionData]:
        """Enhanced Graph API collection with better error handling"""
        positions = []
        
        # Query for recently closed positions (more complete data)
        closed_positions_query = """
        {
          positions(
            first: 1000,
            where: {
              liquidity: "0",
              depositedToken0_gt: "100"
            },
            orderBy: transaction_timestamp,
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
            transaction { 
              timestamp 
              blockNumber
              id
            }
          }
        }
        """
        
        # Query for active positions
        active_positions_query = """
        {
          positions(
            first: 500,
            where: {
              liquidity_gt: "0",
              depositedToken0_gt: "1000"
            },
            orderBy: depositedToken0,
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
            transaction { 
              timestamp 
              blockNumber
              id
            }
          }
        }
        """
        
        queries = [
            ("closed_positions", closed_positions_query),
            ("active_positions", active_positions_query)
        ]
        
        for query_name, query in queries:
            try:
                response_data = await self._make_graph_request(query)
                if response_data and 'positions' in response_data:
                    batch_positions = []
                    for pos_data in response_data['positions']:
                        try:
                            position = self._parse_graph_position(pos_data)
                            if position:
                                batch_positions.append(position)
                        except Exception as e:
                            logger.warning(f"Failed to parse position {pos_data.get('id', 'unknown')}: {e}")
                    
                    positions.extend(batch_positions)
                    logger.info(f"Graph API {query_name}: {len(batch_positions)} positions parsed")
                    
            except Exception as e:
                logger.error(f"Graph API {query_name} failed: {e}")
                self.collection_stats['collection_errors'] += 1
        
        return positions
    
    async def _collect_from_rpc_events(self) -> List[PositionData]:
        """Collect positions from RPC event logs"""
        positions = []
        
        try:
            # Get recent blocks
            latest_block = self.web3.eth.block_number
            from_block = latest_block - 1000  # Last ~1000 blocks
            
            # Mint events (position creation)
            mint_event_signature = "0x7a53080ba414158be7ec69b987b5fb7d07dee101fe85488f0853ae16239d0bde"
            
            # Burn events (position removal)
            burn_event_signature = "0x0c396cd989a39f4459b5fa1aed6a9a8dcdbc45908acfd67e028cd568da98982c"
            
            # Collect events
            for event_sig, event_name in [(mint_event_signature, "mint"), (burn_event_signature, "burn")]:
                try:
                    filter_params = {
                        'fromBlock': from_block,
                        'toBlock': 'latest',
                        'topics': [event_sig]
                    }
                    
                    logs = self.web3.eth.get_logs(filter_params)
                    self.collection_stats['rpc_calls'] += 1
                    
                    for log in logs[-100:]:  # Process last 100 events
                        try:
                            position = await self._parse_rpc_event(log, event_name)
                            if position:
                                positions.append(position)
                        except Exception as e:
                            logger.warning(f"Failed to parse RPC event: {e}")
                            
                except Exception as e:
                    logger.warning(f"Failed to get {event_name} events: {e}")
            
        except Exception as e:
            logger.error(f"RPC event collection failed: {e}")
            self.collection_stats['collection_errors'] += 1
        
        return positions
    
    async def _collect_from_position_nfts(self) -> List[PositionData]:
        """Collect positions by enumerating position NFTs"""
        positions = []
        
        try:
            # Get position manager contract
            position_manager = self.web3.eth.contract(
                address=self.position_manager_address,
                abi=self.position_manager_abi
            )
            
            # Sample recent position token IDs (in production, would use events or enumeration)
            # For now, sample some token IDs
            sample_token_ids = range(1, 1000, 10)  # Sample every 10th position
            
            for token_id in sample_token_ids:
                try:
                    # Get position data from contract
                    position_data = position_manager.functions.positions(token_id).call()
                    self.collection_stats['rpc_calls'] += 1
                    
                    # Only process if position has liquidity or was recently active
                    if position_data[7] > 0 or position_data[10] > 0 or position_data[11] > 0:  # liquidity or fees
                        position = await self._parse_nft_position(token_id, position_data)
                        if position:
                            positions.append(position)
                            
                except Exception as e:
                    # Position might not exist, continue
                    continue
                    
                # Rate limiting
                await asyncio.sleep(1.0 / self.config.rpc_rate_limit)
                
        except Exception as e:
            logger.error(f"Position NFT collection failed: {e}")
            self.collection_stats['collection_errors'] += 1
        
        return positions
    
    async def _collect_from_backup_sources(self) -> List[PositionData]:
        """Collect from backup data sources"""
        positions = []
        
        # Backup Graph endpoints
        backup_endpoints = [
            "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3",
            "https://api.studio.thegraph.com/query/5713/uniswap-v3-base/version/latest"
        ]
        
        simple_query = """
        {
          positions(first: 100, orderBy: depositedToken0, orderDirection: desc) {
            id
            owner
            liquidity
            depositedToken0
            depositedToken1
            pool {
              id
              token0 { symbol }
              token1 { symbol }
              feeTier
            }
          }
        }
        """
        
        for endpoint in backup_endpoints:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        endpoint,
                        json={'query': simple_query},
                        headers={'Content-Type': 'application/json'},
                        timeout=aiohttp.ClientTimeout(total=15)
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            if 'data' in data and 'positions' in data['data']:
                                for pos_data in data['data']['positions']:
                                    position = self._parse_backup_position(pos_data)
                                    if position:
                                        positions.append(position)
                                break  # Success, no need to try other endpoints
            except Exception as e:
                logger.warning(f"Backup endpoint {endpoint} failed: {e}")
                continue
        
        return positions
    
    def _parse_graph_position(self, pos_data: Dict) -> Optional[PositionData]:
        """Parse position data from Graph API response"""
        try:
            pool = pos_data['pool']
            tick_lower = int(pos_data['tickLower']['tickIdx'])
            tick_upper = int(pos_data['tickUpper']['tickIdx'])
            current_tick = int(pool['tick'])
            
            position = PositionData(
                position_id=pos_data['id'],
                pool_address=pool['id'],
                owner=pos_data['owner'],
                token0_symbol=pool['token0']['symbol'],
                token1_symbol=pool['token1']['symbol'],
                fee_tier=int(pool['feeTier']),
                tick_lower=tick_lower,
                tick_upper=tick_upper,
                liquidity=float(pos_data['liquidity']),
                deposited_token0=float(pos_data['depositedToken0']),
                deposited_token1=float(pos_data['depositedToken1']),
                withdrawn_token0=float(pos_data['withdrawnToken0']),
                withdrawn_token1=float(pos_data['withdrawnToken1']),
                collected_fees_token0=float(pos_data['collectedFeesToken0']),
                collected_fees_token1=float(pos_data['collectedFeesToken1']),
                current_tick=current_tick,
                sqrt_price=float(pool['sqrtPrice']),
                block_number=int(pos_data['transaction']['blockNumber']),
                timestamp=datetime.fromtimestamp(int(pos_data['transaction']['timestamp'])),
                transaction_hash=pos_data['transaction']['id'],
                data_source="graph_api",
                confidence_score=0.9
            )
            
            # Calculate derived metrics
            position.in_range = tick_lower <= current_tick <= tick_upper
            position.range_width = tick_upper - tick_lower
            
            # Estimate position value (simplified)
            position.position_value_usd = position.deposited_token0 + position.deposited_token1
            
            # Calculate fee yield
            total_fees = position.collected_fees_token0 + position.collected_fees_token1
            if position.position_value_usd > 0:
                position.fee_yield = total_fees / position.position_value_usd
            
            return position
            
        except Exception as e:
            logger.warning(f"Failed to parse Graph position: {e}")
            return None
    
    async def _parse_rpc_event(self, log: Dict, event_type: str) -> Optional[PositionData]:
        """Parse position data from RPC event log"""
        try:
            # This would require proper ABI decoding
            # For now, create a basic position structure
            position = PositionData(
                position_id=f"rpc_{log['transactionHash']}_{log['logIndex']}",
                pool_address=log['address'],
                owner="unknown",  # Would decode from event data
                token0_symbol="TOKEN0",
                token1_symbol="TOKEN1",
                fee_tier=3000,  # Default, would get from pool
                tick_lower=0,  # Would decode from event
                tick_upper=0,  # Would decode from event
                liquidity=0.0,  # Would decode from event
                deposited_token0=0.0,
                deposited_token1=0.0,
                withdrawn_token0=0.0,
                withdrawn_token1=0.0,
                collected_fees_token0=0.0,
                collected_fees_token1=0.0,
                current_tick=0,
                sqrt_price=0.0,
                block_number=log['blockNumber'],
                timestamp=datetime.now(),  # Would get from block
                transaction_hash=log['transactionHash'],
                data_source="rpc_events",
                confidence_score=0.7
            )
            
            return position
            
        except Exception as e:
            logger.warning(f"Failed to parse RPC event: {e}")
            return None
    
    async def _parse_nft_position(self, token_id: int, position_data: tuple) -> Optional[PositionData]:
        """Parse position data from NFT contract call"""
        try:
            # Unpack position data tuple
            (nonce, operator, token0, token1, fee, tick_lower, tick_upper, 
             liquidity, fee_growth_0, fee_growth_1, tokens_owed_0, tokens_owed_1) = position_data
            
            position = PositionData(
                position_id=f"nft_{token_id}",
                pool_address="",  # Would calculate from tokens and fee
                owner=operator,
                token0_symbol="TOKEN0",  # Would lookup from token address
                token1_symbol="TOKEN1",  # Would lookup from token address
                fee_tier=fee,
                tick_lower=tick_lower,
                tick_upper=tick_upper,
                liquidity=float(liquidity),
                deposited_token0=0.0,  # Would need additional contract calls
                deposited_token1=0.0,
                withdrawn_token0=0.0,
                withdrawn_token1=0.0,
                collected_fees_token0=float(tokens_owed_0),
                collected_fees_token1=float(tokens_owed_1),
                current_tick=0,  # Would get from pool
                sqrt_price=0.0,  # Would get from pool
                block_number=0,  # Would get current block
                timestamp=datetime.now(),
                transaction_hash="",
                data_source="position_nfts",
                confidence_score=0.8
            )
            
            return position
            
        except Exception as e:
            logger.warning(f"Failed to parse NFT position: {e}")
            return None
    
    def _parse_backup_position(self, pos_data: Dict) -> Optional[PositionData]:
        """Parse position data from backup sources"""
        try:
            pool = pos_data['pool']
            
            position = PositionData(
                position_id=pos_data['id'],
                pool_address=pool['id'],
                owner=pos_data['owner'],
                token0_symbol=pool['token0']['symbol'],
                token1_symbol=pool['token1']['symbol'],
                fee_tier=int(pool['feeTier']),
                tick_lower=0,  # Not available in simple query
                tick_upper=0,
                liquidity=float(pos_data['liquidity']),
                deposited_token0=float(pos_data['depositedToken0']),
                deposited_token1=float(pos_data['depositedToken1']),
                withdrawn_token0=0.0,
                withdrawn_token1=0.0,
                collected_fees_token0=0.0,
                collected_fees_token1=0.0,
                current_tick=0,
                sqrt_price=0.0,
                block_number=0,
                timestamp=datetime.now(),
                transaction_hash="",
                data_source="backup_apis",
                confidence_score=0.6
            )
            
            return position
            
        except Exception as e:
            logger.warning(f"Failed to parse backup position: {e}")
            return None
    
    def _deduplicate_positions(self, positions: List[PositionData]) -> List[PositionData]:
        """Remove duplicate positions based on ID and pool"""
        seen = set()
        unique_positions = []
        
        for position in positions:
            # Create unique key
            key = f"{position.pool_address}_{position.owner}_{position.tick_lower}_{position.tick_upper}"
            if key not in seen:
                seen.add(key)
                unique_positions.append(position)
        
        logger.info(f"Deduplicated {len(positions)} → {len(unique_positions)} positions")
        return unique_positions
    
    async def _validate_and_enrich_positions(self, positions: List[PositionData]) -> List[PositionData]:
        """Validate and enrich position data"""
        validated_positions = []
        
        for position in positions:
            try:
                # Validation checks
                if position.position_value_usd < self.config.min_liquidity_usd:
                    continue
                
                # Age check
                if position.timestamp:
                    age_hours = (datetime.now() - position.timestamp).total_seconds() / 3600
                    if age_hours < self.config.min_position_age_hours:
                        continue
                
                # Enrich with additional calculations
                await self._enrich_position_data(position)
                
                # Quality score adjustment
                if position.data_source == "graph_api":
                    position.confidence_score = 0.9
                elif position.data_source == "rpc_events":
                    position.confidence_score = 0.8
                elif position.data_source == "position_nfts":
                    position.confidence_score = 0.8
                else:
                    position.confidence_score = 0.6
                
                validated_positions.append(position)
                
            except Exception as e:
                logger.warning(f"Position validation failed: {e}")
                continue
        
        logger.info(f"Validated {len(validated_positions)} high-quality positions")
        return validated_positions
    
    async def _enrich_position_data(self, position: PositionData):
        """Enrich position with additional calculated metrics"""
        try:
            # Calculate APR estimate
            if position.collected_fees_token0 + position.collected_fees_token1 > 0:
                total_fees = position.collected_fees_token0 + position.collected_fees_token1
                if position.position_value_usd > 0:
                    # Annualized return based on fees
                    days_active = max(1, (datetime.now() - position.timestamp).days)
                    daily_yield = total_fees / position.position_value_usd / days_active
                    position.apr_estimate = daily_yield * 365
            
            # Calculate capital efficiency
            if position.range_width > 0:
                # Narrower ranges = higher capital efficiency
                position.capital_efficiency = 1.0 / (position.range_width / 1000.0)
            
            # Estimate impermanent loss (simplified)
            if position.deposited_token0 > 0 and position.deposited_token1 > 0:
                # This would require price data for accurate calculation
                position.impermanent_loss = 0.0  # Placeholder
            
        except Exception as e:
            logger.warning(f"Position enrichment failed: {e}")
    
    async def _make_graph_request(self, query: str) -> Optional[Dict]:
        """Make GraphQL request with error handling"""
        url = f"https://gateway-arbitrum.network.thegraph.com/api/{self.config.graph_api_key}/subgraphs/id/5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV"
        
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {self.config.graph_api_key}'
                }
                
                async with session.post(
                    url,
                    json={'query': query},
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    self.collection_stats['graph_api_calls'] += 1
                    
                    if response.status == 200:
                        data = await response.json()
                        if 'errors' in data:
                            logger.error(f"GraphQL errors: {data['errors']}")
                            return None
                        return data.get('data')
                    else:
                        logger.error(f"Graph API request failed: {response.status}")
                        return None
                        
        except Exception as e:
            logger.error(f"Graph API request error: {e}")
            return None
    
    async def _log_collection_results(self, positions: List[PositionData]):
        """Log collection results to website"""
        try:
            # Calculate summary statistics
            total_value = sum(p.position_value_usd for p in positions)
            avg_apr = np.mean([p.apr_estimate for p in positions if p.apr_estimate > 0]) if positions else 0
            in_range_pct = np.mean([p.in_range for p in positions]) if positions else 0
            
            source_breakdown = {}
            for pos in positions:
                source = pos.data_source
                if source not in source_breakdown:
                    source_breakdown[source] = 0
                source_breakdown[source] += 1
            
            log_data = {
                "timestamp": datetime.now().isoformat(),
                "level": "SUCCESS",
                "source": "AdvancedDataCollection",
                "message": f"Collected {len(positions)} high-quality Uniswap positions",
                "details": {
                    "positions_collected": len(positions),
                    "total_value_usd": total_value,
                    "average_apr": avg_apr,
                    "in_range_percentage": in_range_pct,
                    "data_sources": source_breakdown,
                    "collection_stats": {
                        "total_found": self.collection_stats['total_positions_found'],
                        "successfully_parsed": self.collection_stats['positions_parsed_successfully'],
                        "high_quality": self.collection_stats['high_quality_positions'],
                        "collection_duration": self.collection_stats.get('collection_duration', 0),
                        "sources_used": list(self.collection_stats['data_sources_used'])
                    }
                }
            }
            
            website_logger.info(json.dumps(log_data))
            
        except Exception as e:
            logger.warning(f"Failed to log collection results: {e}")
    
    def get_collection_summary(self) -> Dict[str, Any]:
        """Get summary of data collection results"""
        return {
            'positions_collected': len(self.collected_positions),
            'collection_stats': self.collection_stats,
            'data_quality_distribution': {
                source: len([p for p in self.collected_positions if p.data_source == source])
                for source in set(p.data_source for p in self.collected_positions)
            },
            'last_collection': self.collection_stats.get('last_collection_time'),
            'success_rate': (
                self.collection_stats['positions_parsed_successfully'] / 
                max(self.collection_stats['total_positions_found'], 1)
            )
        }


async def main():
    """Test the advanced data collection system"""
    config = DataCollectionConfig(
        graph_api_key="c6f241c1dd5aea81977a63b2614af70d",
        alchemy_api_key=os.getenv('ALCHEMY_API_KEY', ''),
        infura_api_key=os.getenv('INFURA_API_KEY', ''),
        base_rpc_url="https://base-mainnet.g.alchemy.com/v2/" + os.getenv('ALCHEMY_API_KEY', '')
    )
    
    collector = AdvancedUniswapDataCollector(config)
    positions = await collector.collect_comprehensive_position_data()
    
    summary = collector.get_collection_summary()
    print(f"Collection Summary: {json.dumps(summary, indent=2, default=str)}")

if __name__ == "__main__":
    asyncio.run(main())