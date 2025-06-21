#!/usr/bin/env python3
"""
Historical Liquidity Position Fetcher for DexBrain

This module fetches closed liquidity positions from various sources:
1. The Graph subgraphs for Base network
2. Direct blockchain queries for position data
3. Third-party APIs like Alchemy for historical data

The fetched data is parsed and formatted for DexBrain ML training.
"""

import asyncio
import aiohttp
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging
from pathlib import Path
import os
from web3 import Web3
from web3.exceptions import Web3Exception

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ClosedPosition:
    """Data structure for a closed liquidity position"""
    position_id: str
    owner: str
    pool_address: str
    token0: str
    token1: str
    fee_tier: int
    tick_lower: int
    tick_upper: int
    liquidity: int
    
    # Lifecycle timestamps
    created_at: datetime
    closed_at: datetime
    duration_hours: float
    
    # Financial metrics
    initial_token0_amount: float
    initial_token1_amount: float
    final_token0_amount: float
    final_token1_amount: float
    fees_earned_token0: float
    fees_earned_token1: float
    
    # Performance metrics
    initial_usd_value: float
    final_usd_value: float
    total_fees_usd: float
    impermanent_loss_usd: float
    net_pnl_usd: float
    apy: float
    roi_percentage: float
    
    # Market context
    entry_price: float
    exit_price: float
    price_change_percentage: float
    volume_during_position: float
    avg_volatility: float
    
    # Position management
    close_reason: str
    gas_costs_usd: float
    rebalance_count: int
    time_in_range_percentage: float


class BaseNetworkConfig:
    """Configuration for Base network data sources"""
    
    # Base RPC endpoint
    BASE_RPC = os.getenv('BASE_RPC_URL', 'https://base-mainnet.g.alchemy.com/v2/ory0F2cLFNIXsovAmrtJj')
    
    # The Graph endpoints for Base network
    GRAPH_ENDPOINTS = {
        'uniswap_v3_base': 'https://gateway.thegraph.com/api/[API_KEY]/subgraphs/id/HMuAwufqZ1YCRmzL2SfHTVkzZovC9VL2UAKhjvRqKiR1',
        'messari_base': 'https://gateway.thegraph.com/api/[API_KEY]/subgraphs/id/FUbEPQw1oMghy39fwWBFY5fE6MXPXZQtjncQy2cXdrNS'
    }
    
    # Contract addresses on Base
    UNISWAP_V3_FACTORY = '0x33128a8fC17869897dcE68Ed026d694621f6FDfD'
    NONFUNGIBLE_POSITION_MANAGER = '0x03a520b32C04BF3bEEf7BF8082933982d0f07ea0'
    
    # API keys (should be set as environment variables)
    GRAPH_API_KEY = os.getenv('GRAPH_API_KEY', '')
    ALCHEMY_API_KEY = os.getenv('ALCHEMY_API_KEY', 'ory0F2cLFNIXsovAmrtJj')


class HistoricalPositionFetcher:
    """Fetches historical closed liquidity positions from multiple sources"""
    
    def __init__(self):
        self.config = BaseNetworkConfig()
        self.w3 = Web3(Web3.HTTPProvider(self.config.BASE_RPC))
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def fetch_positions_from_graph(
        self, 
        endpoint_name: str = 'uniswap_v3_base',
        limit: int = 1000,
        days_back: int = 30
    ) -> List[Dict[str, Any]]:
        """Fetch closed positions from The Graph subgraph"""
        
        if not self.config.GRAPH_API_KEY:
            logger.warning("GRAPH_API_KEY not set, using public endpoint (rate limited)")
            endpoint = self.config.GRAPH_ENDPOINTS[endpoint_name].replace('[API_KEY]/', '')
        else:
            endpoint = self.config.GRAPH_ENDPOINTS[endpoint_name].replace('[API_KEY]', self.config.GRAPH_API_KEY)
        
        # Calculate timestamp for X days back
        days_ago = datetime.now() - timedelta(days=days_back)
        timestamp_filter = int(days_ago.timestamp())
        
        # GraphQL query to fetch closed positions
        query = f"""
        {{
          positions(
            first: {limit}
            where: {{
              liquidity: "0"
              depositedToken0: "0"
              depositedToken1: "0"
              transaction_: {{
                timestamp_gte: "{timestamp_filter}"
              }}
            }}
            orderBy: transaction__timestamp
            orderDirection: desc
          ) {{
            id
            owner
            pool {{
              id
              token0 {{
                id
                symbol
                decimals
              }}
              token1 {{
                id
                symbol
                decimals
              }}
              feeTier
            }}
            tickLower {{
              tickIdx
            }}
            tickUpper {{
              tickIdx
            }}
            liquidity
            depositedToken0
            depositedToken1
            withdrawnToken0
            withdrawnToken1
            collectedFeesToken0
            collectedFeesToken1
            transaction {{
              id
              timestamp
              blockNumber
              gasUsed
              gasPrice
            }}
            positionSnapshot(first: 1, orderBy: timestamp, orderDirection: asc) {{
              timestamp
              liquidity
              depositedToken0
              depositedToken1
            }}
          }}
        }}
        """
        
        try:
            async with self.session.post(
                endpoint,
                json={'query': query},
                headers={'Content-Type': 'application/json'}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'errors' in data:
                        logger.error(f"GraphQL errors: {data['errors']}")
                        return []
                    return data.get('data', {}).get('positions', [])
                else:
                    logger.error(f"Failed to fetch from Graph: {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Error fetching from Graph: {e}")
            return []
    
    async def get_position_creation_data(self, position_id: str) -> Optional[Dict[str, Any]]:
        """Get position creation data from blockchain"""
        try:
            # Query the NonfungiblePositionManager for position creation events
            position_manager_abi = [
                {
                    "anonymous": False,
                    "inputs": [
                        {"indexed": True, "name": "tokenId", "type": "uint256"},
                        {"indexed": True, "name": "liquidity", "type": "uint128"},
                        {"indexed": False, "name": "amount0", "type": "uint256"},
                        {"indexed": False, "name": "amount1", "type": "uint256"}
                    ],
                    "name": "IncreaseLiquidity",
                    "type": "event"
                }
            ]
            
            contract = self.w3.eth.contract(
                address=self.config.NONFUNGIBLE_POSITION_MANAGER,
                abi=position_manager_abi
            )
            
            # Get events for this position
            events = contract.events.IncreaseLiquidity.create_filter(
                argument_filters={'tokenId': int(position_id)},
                fromBlock='earliest'
            ).get_all_entries()
            
            if events:
                creation_event = events[0]  # First event is creation
                return {
                    'block_number': creation_event.blockNumber,
                    'transaction_hash': creation_event.transactionHash.hex(),
                    'initial_liquidity': creation_event.args.liquidity,
                    'initial_amount0': creation_event.args.amount0,
                    'initial_amount1': creation_event.args.amount1
                }
            
        except Exception as e:
            logger.error(f"Error getting position creation data: {e}")
        
        return None
    
    async def calculate_position_metrics(
        self, 
        position_data: Dict[str, Any],
        token0_price_history: List[float],
        token1_price_history: List[float]
    ) -> Dict[str, float]:
        """Calculate performance metrics for a position"""
        
        # Extract basic data
        deposited_token0 = float(position_data.get('depositedToken0', 0))
        deposited_token1 = float(position_data.get('depositedToken1', 0))
        withdrawn_token0 = float(position_data.get('withdrawnToken0', 0))
        withdrawn_token1 = float(position_data.get('withdrawnToken1', 0))
        fees_token0 = float(position_data.get('collectedFeesToken0', 0))
        fees_token1 = float(position_data.get('collectedFeesToken1', 0))
        
        # Use average prices for USD calculations (simplified)
        avg_token0_price = sum(token0_price_history) / len(token0_price_history) if token0_price_history else 1
        avg_token1_price = sum(token1_price_history) / len(token1_price_history) if token1_price_history else 1
        
        # Calculate USD values
        initial_usd = (deposited_token0 * avg_token0_price) + (deposited_token1 * avg_token1_price)
        final_usd = (withdrawn_token0 * avg_token0_price) + (withdrawn_token1 * avg_token1_price)
        fees_usd = (fees_token0 * avg_token0_price) + (fees_token1 * avg_token1_price)
        
        # Calculate metrics
        total_return_usd = final_usd + fees_usd - initial_usd
        roi_percentage = (total_return_usd / initial_usd * 100) if initial_usd > 0 else 0
        
        # Estimate impermanent loss (simplified calculation)
        entry_ratio = deposited_token0 / deposited_token1 if deposited_token1 > 0 else 0
        exit_ratio = withdrawn_token0 / withdrawn_token1 if withdrawn_token1 > 0 else 0
        price_change = abs(exit_ratio - entry_ratio) / entry_ratio if entry_ratio > 0 else 0
        estimated_il = price_change * 0.25 * initial_usd  # Simplified IL estimation
        
        return {
            'initial_usd_value': initial_usd,
            'final_usd_value': final_usd,
            'total_fees_usd': fees_usd,
            'impermanent_loss_usd': estimated_il,
            'net_pnl_usd': total_return_usd - estimated_il,
            'roi_percentage': roi_percentage,
            'price_change_percentage': price_change * 100
        }
    
    async def parse_position_data(self, raw_position: Dict[str, Any]) -> Optional[ClosedPosition]:
        """Parse raw position data into structured ClosedPosition object"""
        
        try:
            # Extract timestamps
            creation_snapshot = raw_position.get('positionSnapshot', [{}])[0]
            creation_timestamp = int(creation_snapshot.get('timestamp', 0))
            close_timestamp = int(raw_position['transaction']['timestamp'])
            
            created_at = datetime.fromtimestamp(creation_timestamp)
            closed_at = datetime.fromtimestamp(close_timestamp)
            duration_hours = (closed_at - created_at).total_seconds() / 3600
            
            # Get pool info
            pool = raw_position['pool']
            token0 = pool['token0']
            token1 = pool['token1']
            
            # Calculate position metrics (mock price data for now)
            mock_token0_prices = [1.0] * 10  # Replace with real price history
            mock_token1_prices = [1.0] * 10  # Replace with real price history
            
            metrics = await self.calculate_position_metrics(
                raw_position, 
                mock_token0_prices, 
                mock_token1_prices
            )
            
            # Calculate APY
            days_active = duration_hours / 24
            annual_multiplier = 365 / days_active if days_active > 0 else 0
            apy = metrics['roi_percentage'] * annual_multiplier
            
            return ClosedPosition(
                position_id=raw_position['id'],
                owner=raw_position['owner'],
                pool_address=pool['id'],
                token0=token0['id'],
                token1=token1['id'],
                fee_tier=int(pool['feeTier']),
                tick_lower=int(raw_position['tickLower']['tickIdx']),
                tick_upper=int(raw_position['tickUpper']['tickIdx']),
                liquidity=int(raw_position['liquidity']),
                
                created_at=created_at,
                closed_at=closed_at,
                duration_hours=duration_hours,
                
                initial_token0_amount=float(raw_position.get('depositedToken0', 0)),
                initial_token1_amount=float(raw_position.get('depositedToken1', 0)),
                final_token0_amount=float(raw_position.get('withdrawnToken0', 0)),
                final_token1_amount=float(raw_position.get('withdrawnToken1', 0)),
                fees_earned_token0=float(raw_position.get('collectedFeesToken0', 0)),
                fees_earned_token1=float(raw_position.get('collectedFeesToken1', 0)),
                
                initial_usd_value=metrics['initial_usd_value'],
                final_usd_value=metrics['final_usd_value'],
                total_fees_usd=metrics['total_fees_usd'],
                impermanent_loss_usd=metrics['impermanent_loss_usd'],
                net_pnl_usd=metrics['net_pnl_usd'],
                apy=apy,
                roi_percentage=metrics['roi_percentage'],
                
                entry_price=1.0,  # Replace with real price data
                exit_price=1.0,   # Replace with real price data
                price_change_percentage=metrics['price_change_percentage'],
                volume_during_position=0.0,  # TODO: Calculate from historical data
                avg_volatility=0.0,          # TODO: Calculate from historical data
                
                close_reason='manual',  # TODO: Determine from transaction data
                gas_costs_usd=0.0,      # TODO: Calculate from gas usage
                rebalance_count=0,      # TODO: Count rebalance transactions
                time_in_range_percentage=0.0  # TODO: Calculate from price history
            )
        
        except Exception as e:
            logger.error(f"Error parsing position data: {e}")
            return None
    
    async def fetch_closed_positions(
        self, 
        limit: int = 1000, 
        days_back: int = 30
    ) -> List[ClosedPosition]:
        """Main method to fetch and parse closed positions"""
        
        logger.info(f"Fetching closed positions from last {days_back} days...")
        
        # Fetch raw data from The Graph
        raw_positions = await self.fetch_positions_from_graph(
            limit=limit, 
            days_back=days_back
        )
        
        logger.info(f"Found {len(raw_positions)} raw positions")
        
        # Parse positions
        parsed_positions = []
        for raw_position in raw_positions:
            parsed = await self.parse_position_data(raw_position)
            if parsed:
                parsed_positions.append(parsed)
        
        logger.info(f"Successfully parsed {len(parsed_positions)} positions")
        return parsed_positions
    
    def save_positions_to_json(self, positions: List[ClosedPosition], filename: str):
        """Save positions to JSON file for inspection/backup"""
        
        data = []
        for pos in positions:
            pos_dict = {
                'position_id': pos.position_id,
                'owner': pos.owner,
                'pool_address': pos.pool_address,
                'token0': pos.token0,
                'token1': pos.token1,
                'fee_tier': pos.fee_tier,
                'created_at': pos.created_at.isoformat(),
                'closed_at': pos.closed_at.isoformat(),
                'duration_hours': pos.duration_hours,
                'initial_usd_value': pos.initial_usd_value,
                'final_usd_value': pos.final_usd_value,
                'net_pnl_usd': pos.net_pnl_usd,
                'apy': pos.apy,
                'roi_percentage': pos.roi_percentage
            }
            data.append(pos_dict)
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved {len(positions)} positions to {filename}")


async def main():
    """Example usage of the HistoricalPositionFetcher"""
    
    async with HistoricalPositionFetcher() as fetcher:
        # Fetch positions from the last 7 days
        positions = await fetcher.fetch_closed_positions(limit=100, days_back=7)
        
        if positions:
            # Save to JSON for inspection
            fetcher.save_positions_to_json(positions, 'closed_positions_sample.json')
            
            # Print sample position
            pos = positions[0]
            print(f"\nSample Position:")
            print(f"ID: {pos.position_id}")
            print(f"Pool: {pos.token0} / {pos.token1}")
            print(f"Duration: {pos.duration_hours:.1f} hours")
            print(f"ROI: {pos.roi_percentage:.2f}%")
            print(f"APY: {pos.apy:.2f}%")
            print(f"Net P&L: ${pos.net_pnl_usd:.2f}")
        else:
            print("No positions found")


if __name__ == "__main__":
    asyncio.run(main())