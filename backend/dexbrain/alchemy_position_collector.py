"""
Alchemy-based Uniswap V3 Position Data Collector
Direct on-chain data collection from Base network using Alchemy RPC
"""

from web3 import Web3
import json
import time
import csv
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import aiohttp
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Website logging
website_logger = logging.getLogger('website')
website_handler = logging.FileHandler('/var/log/dexter/liquidity.log')
website_handler.setFormatter(logging.Formatter('%(message)s'))
website_logger.addHandler(website_handler)
website_logger.setLevel(logging.INFO)

@dataclass
class AlchemyPositionData:
    """Data structure for Uniswap V3 position from Alchemy"""
    token_id: int
    owner: str
    token0: str
    token1: str
    fee_tier: int
    tick_lower: int
    tick_upper: int
    liquidity: int
    fee_growth_inside0_last_x128: int
    fee_growth_inside1_last_x128: int
    tokens_owed0: int
    tokens_owed1: int
    closed_at_block: int
    closed_at_timestamp: datetime
    transaction_hash: str
    
    # Derived metrics
    is_closed: bool = True
    range_width: int = 0
    data_source: str = "alchemy_rpc"
    
    def __post_init__(self):
        self.range_width = self.tick_upper - self.tick_lower

class AlchemyPositionCollector:
    """
    Collects Uniswap V3 position data directly from Base network using Alchemy RPC
    """
    
    def __init__(self, alchemy_api_key: str):
        self.alchemy_api_key = alchemy_api_key
        self.alchemy_base_rpc = f"https://base-mainnet.g.alchemy.com/v2/{alchemy_api_key}"
        
        # Initialize Web3
        self.w3 = Web3(Web3.HTTPProvider(self.alchemy_base_rpc))
        if not self.w3.is_connected():
            raise ConnectionError("Failed to connect to Alchemy RPC")
        
        # Uniswap V3 Position Manager on Base
        self.npm_address = Web3.to_checksum_address("0x03a520b32C04BF3bEEf7BF5d3Fc401f57B91b3B6")
        
        # Position Manager ABI (simplified)
        self.npm_abi = [
            {
                "inputs": [{"internalType": "uint256", "name": "tokenId", "type": "uint256"}],
                "name": "positions",
                "outputs": [
                    {"internalType": "uint96", "name": "nonce", "type": "uint96"},
                    {"internalType": "address", "name": "operator", "type": "address"},
                    {"internalType": "address", "name": "token0", "type": "address"},
                    {"internalType": "address", "name": "token1", "type": "address"},
                    {"internalType": "uint24", "name": "fee", "type": "uint24"},
                    {"internalType": "int24", "name": "tickLower", "type": "int24"},
                    {"internalType": "int24", "name": "tickUpper", "type": "int24"},
                    {"internalType": "uint128", "name": "liquidity", "type": "uint128"},
                    {"internalType": "uint256", "name": "feeGrowthInside0LastX128", "type": "uint256"},
                    {"internalType": "uint256", "name": "feeGrowthInside1LastX128", "type": "uint256"},
                    {"internalType": "uint128", "name": "tokensOwed0", "type": "uint128"},
                    {"internalType": "uint128", "name": "tokensOwed1", "type": "uint128"}
                ],
                "stateMutability": "view",
                "type": "function",
            }
        ]
        
        # Initialize contract
        self.npm_contract = self.w3.eth.contract(address=self.npm_address, abi=self.npm_abi)
        
        # Event signatures
        self.transfer_event_sig = self.w3.keccak(text="Transfer(address,address,uint256)").hex()
        self.increase_liquidity_sig = self.w3.keccak(text="IncreaseLiquidity(uint256,uint128,uint256,uint256)").hex()
        self.decrease_liquidity_sig = self.w3.keccak(text="DecreaseLiquidity(uint256,uint128,uint256,uint256)").hex()
        self.collect_sig = self.w3.keccak(text="Collect(uint256,address,uint256,uint256)").hex()
        
        # Zero address for burn detection
        self.zero_address_topic = "0x0000000000000000000000000000000000000000000000000000000000000000"
        
        # Collection statistics
        self.collection_stats = {
            'total_positions_found': 0,
            'closed_positions': 0,
            'active_positions': 0,
            'blocks_scanned': 0,
            'rpc_calls': 0,
            'errors': 0,
            'last_collection_time': None
        }
        
        logger.info(f"Alchemy position collector initialized. Connected to Base network: {self.w3.is_connected()}")
    
    async def collect_closed_positions(self, block_range: int = 5000, limit: int = 100) -> List[AlchemyPositionData]:
        """
        Collect recently closed Uniswap V3 positions from Base network
        
        Args:
            block_range: Number of blocks to scan backwards from latest
            limit: Maximum number of positions to collect
            
        Returns:
            List of closed position data
        """
        logger.info(f"🔍 Collecting closed positions from last {block_range} blocks...")
        
        collection_start = datetime.now()
        closed_positions = []
        
        try:
            # Get current block
            current_block = self.w3.eth.block_number
            start_block = max(0, current_block - block_range)
            
            logger.info(f"Scanning blocks {start_block} to {current_block}")
            
            # Get Transfer events to zero address (burns = closed positions)
            burn_logs = self.w3.eth.get_logs({
                "fromBlock": start_block,
                "toBlock": current_block,
                "address": self.npm_address,
                "topics": [
                    self.transfer_event_sig,  # Transfer event
                    None,  # from (any address)
                    self.zero_address_topic  # to (zero address = burn)
                ]
            })
            
            self.collection_stats['blocks_scanned'] = block_range
            self.collection_stats['rpc_calls'] += 1
            
            logger.info(f"Found {len(burn_logs)} burn events")
            
            # Process each burn event
            for i, log in enumerate(burn_logs[:limit]):
                try:
                    # Extract token ID from event
                    token_id = int(log['topics'][3].hex(), 16)
                    
                    # Get position data from contract
                    position_data = await self._fetch_position_data(token_id)
                    
                    if position_data:
                        # Get block timestamp
                        block = self.w3.eth.get_block(log['blockNumber'])
                        close_timestamp = datetime.fromtimestamp(block['timestamp'])
                        
                        # Create position object
                        position = AlchemyPositionData(
                            token_id=token_id,
                            owner=position_data[1],  # operator
                            token0=position_data[2],
                            token1=position_data[3],
                            fee_tier=position_data[4],
                            tick_lower=position_data[5],
                            tick_upper=position_data[6],
                            liquidity=position_data[7],
                            fee_growth_inside0_last_x128=position_data[8],
                            fee_growth_inside1_last_x128=position_data[9],
                            tokens_owed0=position_data[10],
                            tokens_owed1=position_data[11],
                            closed_at_block=log['blockNumber'],
                            closed_at_timestamp=close_timestamp,
                            transaction_hash=log['transactionHash'].hex()
                        )
                        
                        closed_positions.append(position)
                        self.collection_stats['closed_positions'] += 1
                        
                        if (i + 1) % 10 == 0:
                            logger.info(f"Processed {i + 1}/{len(burn_logs)} positions")
                        
                except Exception as e:
                    logger.warning(f"Error processing token {token_id}: {e}")
                    self.collection_stats['errors'] += 1
                    continue
            
            self.collection_stats['total_positions_found'] = len(closed_positions)
            self.collection_stats['last_collection_time'] = collection_start
            
            collection_duration = (datetime.now() - collection_start).total_seconds()
            
            await self._log_collection_results(closed_positions, collection_duration)
            
            logger.info(f"✅ Collected {len(closed_positions)} closed positions in {collection_duration:.1f}s")
            
        except Exception as e:
            logger.error(f"Failed to collect closed positions: {e}")
            self.collection_stats['errors'] += 1
            
        return closed_positions
    
    async def collect_active_positions(self, token_ids: List[int]) -> List[AlchemyPositionData]:
        """
        Collect data for specific active positions by token ID
        
        Args:
            token_ids: List of position token IDs to fetch
            
        Returns:
            List of active position data
        """
        logger.info(f"📊 Collecting data for {len(token_ids)} active positions...")
        
        active_positions = []
        
        for token_id in token_ids:
            try:
                position_data = await self._fetch_position_data(token_id)
                
                if position_data and position_data[7] > 0:  # Has liquidity
                    position = AlchemyPositionData(
                        token_id=token_id,
                        owner=position_data[1],
                        token0=position_data[2],
                        token1=position_data[3],
                        fee_tier=position_data[4],
                        tick_lower=position_data[5],
                        tick_upper=position_data[6],
                        liquidity=position_data[7],
                        fee_growth_inside0_last_x128=position_data[8],
                        fee_growth_inside1_last_x128=position_data[9],
                        tokens_owed0=position_data[10],
                        tokens_owed1=position_data[11],
                        closed_at_block=0,  # Not closed
                        closed_at_timestamp=datetime.now(),  # Current time
                        transaction_hash="",
                        is_closed=False
                    )
                    
                    active_positions.append(position)
                    self.collection_stats['active_positions'] += 1
                    
            except Exception as e:
                logger.warning(f"Error fetching position {token_id}: {e}")
                self.collection_stats['errors'] += 1
                continue
        
        logger.info(f"✅ Collected {len(active_positions)} active positions")
        return active_positions
    
    async def collect_position_lifecycle_events(self, token_id: int) -> Dict[str, List[Dict]]:
        """
        Collect complete lifecycle events for a specific position
        
        Args:
            token_id: Position token ID
            
        Returns:
            Dictionary of event types and their data
        """
        logger.info(f"🔄 Collecting lifecycle events for position {token_id}...")
        
        events = {
            'mint': [],
            'increase_liquidity': [],
            'decrease_liquidity': [],
            'collect': [],
            'burn': []
        }
        
        try:
            # Get all logs for this position
            # Note: This is simplified - in production, you'd need to handle the token ID encoding
            
            # Get Transfer events for mint/burn
            transfer_logs = self.w3.eth.get_logs({
                "fromBlock": 0,  # Would use a reasonable start block
                "toBlock": "latest",
                "address": self.npm_address,
                "topics": [self.transfer_event_sig]
            })
            
            # Filter for our token ID
            for log in transfer_logs:
                if len(log['topics']) > 3:
                    log_token_id = int(log['topics'][3].hex(), 16)
                    if log_token_id == token_id:
                        from_address = log['topics'][1]
                        to_address = log['topics'][2]
                        
                        if from_address == self.zero_address_topic:
                            events['mint'].append(self._parse_event_log(log))
                        elif to_address == self.zero_address_topic:
                            events['burn'].append(self._parse_event_log(log))
            
            # Similar collection for other events would go here
            # (IncreaseLiquidity, DecreaseLiquidity, Collect)
            
            self.collection_stats['rpc_calls'] += 1
            
        except Exception as e:
            logger.error(f"Error collecting lifecycle events: {e}")
            self.collection_stats['errors'] += 1
        
        return events
    
    async def _fetch_position_data(self, token_id: int) -> Optional[tuple]:
        """
        Fetch position data from contract
        
        Args:
            token_id: Position NFT token ID
            
        Returns:
            Position data tuple or None
        """
        try:
            position_data = self.npm_contract.functions.positions(token_id).call()
            self.collection_stats['rpc_calls'] += 1
            return position_data
        except Exception as e:
            logger.warning(f"Failed to fetch position {token_id}: {e}")
            return None
    
    def _parse_event_log(self, log: Dict) -> Dict[str, Any]:
        """Parse event log into readable format"""
        return {
            'block_number': log['blockNumber'],
            'transaction_hash': log['transactionHash'].hex(),
            'log_index': log['logIndex'],
            'topics': [topic.hex() for topic in log['topics']],
            'data': log['data'].hex() if log['data'] else ''
        }
    
    async def export_to_csv(self, positions: List[AlchemyPositionData], filename: str = "alchemy_positions.csv"):
        """
        Export collected positions to CSV file
        
        Args:
            positions: List of position data
            filename: Output CSV filename
        """
        if not positions:
            logger.warning("No positions to export")
            return
        
        filepath = Path(filename)
        
        with open(filepath, 'w', newline='') as f:
            fieldnames = [
                'token_id', 'owner', 'token0', 'token1', 'fee_tier',
                'tick_lower', 'tick_upper', 'liquidity', 'range_width',
                'fee_growth_inside0_last_x128', 'fee_growth_inside1_last_x128',
                'tokens_owed0', 'tokens_owed1', 'closed_at_block',
                'closed_at_timestamp', 'transaction_hash', 'is_closed'
            ]
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for position in positions:
                writer.writerow({
                    'token_id': position.token_id,
                    'owner': position.owner,
                    'token0': position.token0,
                    'token1': position.token1,
                    'fee_tier': position.fee_tier,
                    'tick_lower': position.tick_lower,
                    'tick_upper': position.tick_upper,
                    'liquidity': position.liquidity,
                    'range_width': position.range_width,
                    'fee_growth_inside0_last_x128': position.fee_growth_inside0_last_x128,
                    'fee_growth_inside1_last_x128': position.fee_growth_inside1_last_x128,
                    'tokens_owed0': position.tokens_owed0,
                    'tokens_owed1': position.tokens_owed1,
                    'closed_at_block': position.closed_at_block,
                    'closed_at_timestamp': position.closed_at_timestamp.isoformat(),
                    'transaction_hash': position.transaction_hash,
                    'is_closed': position.is_closed
                })
        
        logger.info(f"✅ Exported {len(positions)} positions to {filepath}")
    
    async def _log_collection_results(self, positions: List[AlchemyPositionData], duration: float):
        """Log collection results to website"""
        try:
            log_data = {
                "timestamp": datetime.now().isoformat(),
                "level": "SUCCESS",
                "source": "AlchemyPositionCollector",
                "message": f"Collected {len(positions)} closed positions from Alchemy RPC",
                "details": {
                    "positions_collected": len(positions),
                    "blocks_scanned": self.collection_stats['blocks_scanned'],
                    "rpc_calls": self.collection_stats['rpc_calls'],
                    "collection_duration": duration,
                    "data_source": "alchemy_direct_rpc",
                    "network": "base",
                    "average_range_width": sum(p.range_width for p in positions) / len(positions) if positions else 0,
                    "unique_pools": len(set((p.token0, p.token1, p.fee_tier) for p in positions))
                }
            }
            
            website_logger.info(json.dumps(log_data))
            
        except Exception as e:
            logger.warning(f"Failed to log collection results: {e}")
    
    def get_collection_summary(self) -> Dict[str, Any]:
        """Get summary of collection statistics"""
        return {
            'collection_stats': self.collection_stats,
            'last_collection': self.collection_stats.get('last_collection_time'),
            'total_positions': self.collection_stats['closed_positions'] + self.collection_stats['active_positions'],
            'error_rate': self.collection_stats['errors'] / max(self.collection_stats['rpc_calls'], 1)
        }
    
    async def continuous_collection_loop(self, interval_minutes: int = 30):
        """
        Continuously collect position data at specified intervals
        
        Args:
            interval_minutes: Minutes between collection cycles
        """
        logger.info(f"🔄 Starting continuous collection loop (every {interval_minutes} minutes)")
        
        while True:
            try:
                # Collect closed positions
                closed_positions = await self.collect_closed_positions(block_range=10000, limit=200)
                
                # Export to CSV
                if closed_positions:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    await self.export_to_csv(closed_positions, f"closed_positions_base_{timestamp}.csv")
                
                # Log summary
                summary = self.get_collection_summary()
                logger.info(f"Collection summary: {json.dumps(summary, indent=2)}")
                
                # Wait for next cycle
                await asyncio.sleep(interval_minutes * 60)
                
            except Exception as e:
                logger.error(f"Error in continuous collection: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error


async def main():
    """Test the Alchemy position collector"""
    import os
    
    # Get API key from environment or use demo key
    alchemy_key = os.getenv('ALCHEMY_API_KEY', 'demo')
    
    collector = AlchemyPositionCollector(alchemy_key)
    
    # Collect closed positions from last 5000 blocks
    closed_positions = await collector.collect_closed_positions(block_range=5000, limit=50)
    
    # Export to CSV
    if closed_positions:
        await collector.export_to_csv(closed_positions, "test_closed_positions.csv")
    
    # Print summary
    summary = collector.get_collection_summary()
    print(f"Collection Summary: {json.dumps(summary, indent=2)}")

if __name__ == "__main__":
    asyncio.run(main())