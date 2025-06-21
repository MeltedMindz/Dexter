"""
Historical Data Backfill Service
Efficiently backfills missing historical data from multiple sources
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import aiohttp
import json
import time
from dataclasses import dataclass
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class BackfillProgress:
    """Track backfill progress"""
    source: str
    start_date: datetime
    end_date: datetime
    total_periods: int
    completed_periods: int
    records_processed: int
    errors_encountered: int
    current_period: datetime
    estimated_completion: datetime
    rate_per_minute: float

class HistoricalBackfillService:
    """
    Service for efficiently backfilling historical data
    """
    
    def __init__(self, 
                 database_url: str,
                 alchemy_api_key: str,
                 graph_api_url: str = "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3"):
        """
        Initialize backfill service
        
        Args:
            database_url: Database connection string
            alchemy_api_key: Alchemy API key
            graph_api_url: Graph API endpoint
        """
        self.database_url = database_url
        self.alchemy_api_key = alchemy_api_key
        self.graph_api_url = graph_api_url
        
        # Database connection
        self.engine = create_engine(database_url)
        self.SessionLocal = sessionmaker(bind=self.engine)
        
        # Rate limiting
        self.rate_limits = {
            'graph_api': {'calls_per_minute': 60, 'last_call': 0},
            'alchemy_rpc': {'calls_per_minute': 300, 'last_call': 0},
            'coingecko': {'calls_per_minute': 50, 'last_call': 0}
        }
        
        # Progress tracking
        self.active_backfills: Dict[str, BackfillProgress] = {}
        
        logger.info("Historical backfill service initialized")
    
    async def backfill_uniswap_positions(self, 
                                       start_date: datetime, 
                                       end_date: datetime,
                                       chunk_hours: int = 6) -> BackfillProgress:
        """
        Backfill Uniswap position data from Graph API
        
        Args:
            start_date: Start date for backfill
            end_date: End date for backfill
            chunk_hours: Hours per chunk for processing
            
        Returns:
            BackfillProgress tracking the operation
        """
        try:
            backfill_id = f"uniswap_positions_{int(time.time())}"
            
            # Calculate time chunks
            current_time = start_date
            chunks = []
            while current_time < end_date:
                chunk_end = min(current_time + timedelta(hours=chunk_hours), end_date)
                chunks.append((current_time, chunk_end))
                current_time = chunk_end
            
            # Initialize progress tracking
            progress = BackfillProgress(
                source="graph_api",
                start_date=start_date,
                end_date=end_date,
                total_periods=len(chunks),
                completed_periods=0,
                records_processed=0,
                errors_encountered=0,
                current_period=start_date,
                estimated_completion=datetime.now() + timedelta(hours=len(chunks) * 0.1),
                rate_per_minute=0.0
            )
            
            self.active_backfills[backfill_id] = progress
            
            logger.info(f"Starting Uniswap position backfill: {len(chunks)} chunks from {start_date} to {end_date}")
            
            start_time = time.time()
            
            for i, (chunk_start, chunk_end) in enumerate(chunks):
                try:
                    # Rate limiting
                    await self._respect_rate_limit('graph_api')
                    
                    # Fetch positions for this time chunk
                    positions = await self._fetch_positions_chunk(chunk_start, chunk_end)
                    
                    if positions:
                        # Store in database
                        stored_count = await self._store_positions_batch(positions)
                        progress.records_processed += stored_count
                        
                        logger.info(f"Processed chunk {i+1}/{len(chunks)}: {stored_count} positions")
                    
                    # Update progress
                    progress.completed_periods = i + 1
                    progress.current_period = chunk_end
                    
                    # Calculate rate
                    elapsed_time = time.time() - start_time
                    if elapsed_time > 0:
                        progress.rate_per_minute = (progress.records_processed / elapsed_time) * 60
                    
                    # Update estimated completion
                    remaining_chunks = len(chunks) - (i + 1)
                    if progress.rate_per_minute > 0:
                        remaining_time_minutes = (remaining_chunks * chunk_hours * 60) / progress.rate_per_minute
                        progress.estimated_completion = datetime.now() + timedelta(minutes=remaining_time_minutes)
                    
                    # Small delay to be respectful
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Error processing chunk {i+1}: {e}")
                    progress.errors_encountered += 1
                    
                    # Continue with next chunk
                    continue
            
            logger.info(f"Backfill completed: {progress.records_processed} positions processed")
            return progress
            
        except Exception as e:
            logger.error(f"Error in Uniswap position backfill: {e}")
            if backfill_id in self.active_backfills:
                self.active_backfills[backfill_id].errors_encountered += 1
            raise
    
    async def backfill_token_prices(self, 
                                  token_addresses: List[str],
                                  start_date: datetime,
                                  end_date: datetime) -> BackfillProgress:
        """
        Backfill historical token price data
        
        Args:
            token_addresses: List of token addresses to backfill
            start_date: Start date for backfill
            end_date: End date for backfill
            
        Returns:
            BackfillProgress tracking the operation
        """
        try:
            backfill_id = f"token_prices_{int(time.time())}"
            
            # Calculate daily chunks
            current_date = start_date.date()
            end_date_only = end_date.date()
            dates = []
            
            while current_date <= end_date_only:
                dates.append(current_date)
                current_date += timedelta(days=1)
            
            total_operations = len(token_addresses) * len(dates)
            
            # Initialize progress
            progress = BackfillProgress(
                source="coingecko",
                start_date=start_date,
                end_date=end_date,
                total_periods=total_operations,
                completed_periods=0,
                records_processed=0,
                errors_encountered=0,
                current_period=start_date,
                estimated_completion=datetime.now() + timedelta(minutes=total_operations * 0.2),
                rate_per_minute=0.0
            )
            
            self.active_backfills[backfill_id] = progress
            
            logger.info(f"Starting token price backfill: {len(token_addresses)} tokens × {len(dates)} days")
            
            start_time = time.time()
            
            for token_address in token_addresses:
                for date in dates:
                    try:
                        # Rate limiting
                        await self._respect_rate_limit('coingecko')
                        
                        # Fetch price data for this token and date
                        price_data = await self._fetch_token_price_history(token_address, date)
                        
                        if price_data:
                            # Store in database
                            await self._store_price_data(token_address, date, price_data)
                            progress.records_processed += 1
                        
                        # Update progress
                        progress.completed_periods += 1
                        progress.current_period = datetime.combine(date, datetime.min.time())
                        
                        # Calculate rate
                        elapsed_time = time.time() - start_time
                        if elapsed_time > 0:
                            progress.rate_per_minute = (progress.completed_periods / elapsed_time) * 60
                        
                        # Small delay
                        await asyncio.sleep(0.5)
                        
                    except Exception as e:
                        logger.error(f"Error fetching price for {token_address} on {date}: {e}")
                        progress.errors_encountered += 1
                        continue
            
            logger.info(f"Price backfill completed: {progress.records_processed} price records processed")
            return progress
            
        except Exception as e:
            logger.error(f"Error in token price backfill: {e}")
            if backfill_id in self.active_backfills:
                self.active_backfills[backfill_id].errors_encountered += 1
            raise
    
    async def backfill_alchemy_position_data(self,
                                           start_block: int,
                                           end_block: int,
                                           block_chunk_size: int = 1000) -> BackfillProgress:
        """
        Backfill position data using Alchemy RPC
        
        Args:
            start_block: Starting block number
            end_block: Ending block number
            block_chunk_size: Number of blocks per chunk
            
        Returns:
            BackfillProgress tracking the operation
        """
        try:
            backfill_id = f"alchemy_positions_{int(time.time())}"
            
            # Calculate block chunks
            current_block = start_block
            chunks = []
            while current_block <= end_block:
                chunk_end = min(current_block + block_chunk_size, end_block)
                chunks.append((current_block, chunk_end))
                current_block = chunk_end + 1
            
            # Initialize progress
            progress = BackfillProgress(
                source="alchemy_rpc",
                start_date=datetime.now() - timedelta(days=7),  # Approximate
                end_date=datetime.now(),
                total_periods=len(chunks),
                completed_periods=0,
                records_processed=0,
                errors_encountered=0,
                current_period=datetime.now() - timedelta(days=7),
                estimated_completion=datetime.now() + timedelta(minutes=len(chunks) * 2),
                rate_per_minute=0.0
            )
            
            self.active_backfills[backfill_id] = progress
            
            logger.info(f"Starting Alchemy position backfill: {len(chunks)} block chunks")
            
            start_time = time.time()
            
            for i, (chunk_start, chunk_end) in enumerate(chunks):
                try:
                    # Rate limiting
                    await self._respect_rate_limit('alchemy_rpc')
                    
                    # Fetch position events for this block range
                    position_events = await self._fetch_position_events_alchemy(chunk_start, chunk_end)
                    
                    if position_events:
                        # Store in database
                        stored_count = await self._store_alchemy_positions(position_events)
                        progress.records_processed += stored_count
                        
                        logger.info(f"Processed blocks {chunk_start}-{chunk_end}: {stored_count} events")
                    
                    # Update progress
                    progress.completed_periods = i + 1
                    
                    # Calculate rate
                    elapsed_time = time.time() - start_time
                    if elapsed_time > 0:
                        progress.rate_per_minute = (progress.records_processed / elapsed_time) * 60
                    
                    await asyncio.sleep(0.2)  # Small delay
                    
                except Exception as e:
                    logger.error(f"Error processing blocks {chunk_start}-{chunk_end}: {e}")
                    progress.errors_encountered += 1
                    continue
            
            logger.info(f"Alchemy backfill completed: {progress.records_processed} events processed")
            return progress
            
        except Exception as e:
            logger.error(f"Error in Alchemy position backfill: {e}")
            if backfill_id in self.active_backfills:
                self.active_backfills[backfill_id].errors_encountered += 1
            raise
    
    async def smart_gap_detection_and_fill(self, 
                                         source: str,
                                         max_gap_hours: int = 6) -> Dict[str, Any]:
        """
        Intelligently detect data gaps and fill them
        
        Args:
            source: Data source to analyze
            max_gap_hours: Maximum acceptable gap in hours
            
        Returns:
            Report of gaps found and filled
        """
        try:
            logger.info(f"🔍 Detecting data gaps for {source}")
            
            # Analyze existing data to find gaps
            gaps = await self._detect_data_gaps(source, max_gap_hours)
            
            if not gaps:
                logger.info(f"No significant gaps found in {source}")
                return {'gaps_found': 0, 'gaps_filled': 0, 'records_added': 0}
            
            logger.info(f"Found {len(gaps)} data gaps in {source}")
            
            gaps_filled = 0
            total_records_added = 0
            
            for gap in gaps:
                try:
                    start_time = gap['start']
                    end_time = gap['end']
                    gap_hours = (end_time - start_time).total_seconds() / 3600
                    
                    logger.info(f"Filling gap: {start_time} to {end_time} ({gap_hours:.1f} hours)")
                    
                    if source == 'graph_api':
                        progress = await self.backfill_uniswap_positions(start_time, end_time, chunk_hours=2)
                        total_records_added += progress.records_processed
                    elif source == 'price_data':
                        # Get all unique token addresses
                        token_addresses = await self._get_monitored_token_addresses()
                        progress = await self.backfill_token_prices(token_addresses, start_time, end_time)
                        total_records_added += progress.records_processed
                    
                    gaps_filled += 1
                    
                    # Small delay between gaps
                    await asyncio.sleep(2)
                    
                except Exception as e:
                    logger.error(f"Error filling gap {start_time} to {end_time}: {e}")
                    continue
            
            result = {
                'gaps_found': len(gaps),
                'gaps_filled': gaps_filled,
                'records_added': total_records_added,
                'source': source,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Gap filling completed: {gaps_filled}/{len(gaps)} gaps filled, {total_records_added} records added")
            return result
            
        except Exception as e:
            logger.error(f"Error in smart gap detection and fill: {e}")
            return {'gaps_found': 0, 'gaps_filled': 0, 'records_added': 0, 'error': str(e)}
    
    async def get_backfill_status(self) -> Dict[str, Any]:
        """Get status of all active backfill operations"""
        try:
            status = {
                'active_backfills': len(self.active_backfills),
                'backfill_details': {},
                'system_status': 'operational'
            }
            
            for backfill_id, progress in self.active_backfills.items():
                completion_percentage = (progress.completed_periods / progress.total_periods) * 100 if progress.total_periods > 0 else 0
                
                status['backfill_details'][backfill_id] = {
                    'source': progress.source,
                    'start_date': progress.start_date.isoformat(),
                    'end_date': progress.end_date.isoformat(),
                    'completion_percentage': completion_percentage,
                    'records_processed': progress.records_processed,
                    'errors_encountered': progress.errors_encountered,
                    'current_period': progress.current_period.isoformat(),
                    'estimated_completion': progress.estimated_completion.isoformat(),
                    'rate_per_minute': progress.rate_per_minute
                }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting backfill status: {e}")
            return {'active_backfills': 0, 'system_status': 'error', 'error': str(e)}
    
    # Helper methods
    
    async def _respect_rate_limit(self, service: str):
        """Respect rate limits for external APIs"""
        if service not in self.rate_limits:
            return
        
        rate_info = self.rate_limits[service]
        calls_per_minute = rate_info['calls_per_minute']
        last_call = rate_info['last_call']
        
        # Calculate minimum time between calls
        min_interval = 60.0 / calls_per_minute
        
        # Check if we need to wait
        time_since_last_call = time.time() - last_call
        if time_since_last_call < min_interval:
            wait_time = min_interval - time_since_last_call
            await asyncio.sleep(wait_time)
        
        # Update last call time
        self.rate_limits[service]['last_call'] = time.time()
    
    async def _fetch_positions_chunk(self, start_time: datetime, end_time: datetime) -> List[Dict]:
        """Fetch positions from Graph API for a time chunk"""
        try:
            # Convert timestamps to Unix format
            start_timestamp = int(start_time.timestamp())
            end_timestamp = int(end_time.timestamp())
            
            query = """
            query GetPositions($startTime: Int!, $endTime: Int!) {
                positions(
                    first: 1000
                    where: {
                        transaction_: {timestamp_gte: $startTime, timestamp_lte: $endTime}
                    }
                    orderBy: transaction__timestamp
                    orderDirection: asc
                ) {
                    id
                    owner
                    pool {
                        id
                        token0 {
                            id
                            symbol
                            decimals
                        }
                        token1 {
                            id
                            symbol
                            decimals
                        }
                        feeTier
                    }
                    tickLower {
                        tickIdx
                    }
                    tickUpper {
                        tickIdx
                    }
                    liquidity
                    depositedToken0
                    depositedToken1
                    withdrawnToken0
                    withdrawnToken1
                    collectedFeesToken0
                    collectedFeesToken1
                    transaction {
                        id
                        timestamp
                        blockNumber
                    }
                }
            }
            """
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.graph_api_url,
                    json={
                        'query': query,
                        'variables': {
                            'startTime': start_timestamp,
                            'endTime': end_timestamp
                        }
                    },
                    headers={'Content-Type': 'application/json'}
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        if 'data' in data and 'positions' in data['data']:
                            return data['data']['positions']
                    else:
                        logger.error(f"Graph API error: {response.status}")
                        return []
            
        except Exception as e:
            logger.error(f"Error fetching positions chunk: {e}")
            return []
    
    async def _fetch_token_price_history(self, token_address: str, date) -> Optional[Dict]:
        """Fetch historical price data for a token"""
        try:
            # This is a simplified implementation
            # In reality, you'd use CoinGecko API or similar
            
            date_str = date.strftime("%d-%m-%Y")
            
            # Simulate price data (replace with actual API call)
            await asyncio.sleep(0.1)  # Simulate API delay
            
            # Return mock data for now
            return {
                'price_usd': np.random.uniform(0.1, 100.0),
                'market_cap': np.random.uniform(1000000, 1000000000),
                'volume_24h': np.random.uniform(10000, 10000000),
                'price_change_24h': np.random.uniform(-10, 10)
            }
            
        except Exception as e:
            logger.error(f"Error fetching price for {token_address} on {date}: {e}")
            return None
    
    async def _fetch_position_events_alchemy(self, start_block: int, end_block: int) -> List[Dict]:
        """Fetch position events using Alchemy RPC"""
        try:
            # This would use Alchemy's enhanced APIs to get position events
            # Simplified implementation
            
            await asyncio.sleep(0.1)  # Simulate RPC call
            
            # Return mock events
            events = []
            for _ in range(np.random.randint(0, 10)):
                events.append({
                    'position_id': f"pos_{np.random.randint(1000, 9999)}",
                    'block_number': np.random.randint(start_block, end_block),
                    'owner': f"0x{''.join([f'{np.random.randint(0, 15):x}' for _ in range(40)])}",
                    'pool_address': f"0x{''.join([f'{np.random.randint(0, 15):x}' for _ in range(40)])}",
                    'tick_lower': np.random.randint(-887272, 887272),
                    'tick_upper': np.random.randint(-887272, 887272),
                    'liquidity': str(np.random.randint(1000000, 1000000000))
                })
            
            return events
            
        except Exception as e:
            logger.error(f"Error fetching position events: {e}")
            return []
    
    async def _store_positions_batch(self, positions: List[Dict]) -> int:
        """Store positions batch in database"""
        try:
            if not positions:
                return 0
            
            # Simplified storage (replace with actual database operations)
            await asyncio.sleep(0.1)  # Simulate database write
            
            logger.debug(f"Stored {len(positions)} positions in database")
            return len(positions)
            
        except Exception as e:
            logger.error(f"Error storing positions batch: {e}")
            return 0
    
    async def _store_price_data(self, token_address: str, date, price_data: Dict) -> bool:
        """Store price data in database"""
        try:
            # Simplified storage
            await asyncio.sleep(0.01)  # Simulate database write
            return True
            
        except Exception as e:
            logger.error(f"Error storing price data: {e}")
            return False
    
    async def _store_alchemy_positions(self, events: List[Dict]) -> int:
        """Store Alchemy position events in database"""
        try:
            if not events:
                return 0
            
            # Simplified storage
            await asyncio.sleep(0.05)  # Simulate database write
            return len(events)
            
        except Exception as e:
            logger.error(f"Error storing Alchemy positions: {e}")
            return 0
    
    async def _detect_data_gaps(self, source: str, max_gap_hours: int) -> List[Dict]:
        """Detect gaps in data for a specific source"""
        try:
            # This would analyze the database to find time gaps
            # Simplified implementation returning mock gaps
            
            gaps = []
            
            # Simulate finding some gaps
            if np.random.random() > 0.7:  # 30% chance of finding gaps
                gap_count = np.random.randint(1, 4)
                for i in range(gap_count):
                    gap_start = datetime.now() - timedelta(days=np.random.randint(1, 30))
                    gap_duration = timedelta(hours=np.random.randint(max_gap_hours, max_gap_hours * 3))
                    gap_end = gap_start + gap_duration
                    
                    gaps.append({
                        'start': gap_start,
                        'end': gap_end,
                        'duration_hours': gap_duration.total_seconds() / 3600
                    })
            
            return gaps
            
        except Exception as e:
            logger.error(f"Error detecting data gaps: {e}")
            return []
    
    async def _get_monitored_token_addresses(self) -> List[str]:
        """Get list of token addresses being monitored"""
        # This would query the database for unique token addresses
        # Simplified implementation
        return [
            "0xA0b86a33E6c4ffbb6d07C8c72E7c90Fc7C92A7F6",  # ETH
            "0xA0b86a33E6c4ffbb6d07C8c72E7c90Fc7C92A7F7",  # USDC
            "0xA0b86a33E6c4ffbb6d07C8c72E7c90Fc7C92A7F8",  # WBTC
        ]

# Demo function
async def demo_backfill_service():
    """Demonstrate the backfill service"""
    logger.info("🎯 Starting Backfill Service Demo")
    
    # Initialize service (with mock credentials)
    service = HistoricalBackfillService(
        database_url="postgresql://mock:mock@localhost/mock",
        alchemy_api_key="mock_key"
    )
    
    # Demo 1: Backfill recent Uniswap positions
    start_date = datetime.now() - timedelta(days=2)
    end_date = datetime.now() - timedelta(days=1)
    
    logger.info("📊 Demo: Backfilling Uniswap positions...")
    position_progress = await service.backfill_uniswap_positions(start_date, end_date, chunk_hours=4)
    logger.info(f"  ✅ Completed: {position_progress.records_processed} positions processed")
    
    # Demo 2: Backfill token prices
    logger.info("💰 Demo: Backfilling token prices...")
    token_addresses = ["0xETH", "0xUSDC", "0xWBTC"]
    price_progress = await service.backfill_token_prices(token_addresses, start_date, end_date)
    logger.info(f"  ✅ Completed: {price_progress.records_processed} price records processed")
    
    # Demo 3: Smart gap detection
    logger.info("🔍 Demo: Smart gap detection and filling...")
    gap_result = await service.smart_gap_detection_and_fill('graph_api', max_gap_hours=4)
    logger.info(f"  ✅ Completed: {gap_result['gaps_filled']} gaps filled, {gap_result['records_added']} records added")
    
    # Show final status
    status = await service.get_backfill_status()
    logger.info(f"📈 Final Status: {status['active_backfills']} active backfills")
    
    logger.info("🎉 Backfill Service Demo completed!")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run demo
    asyncio.run(demo_backfill_service())