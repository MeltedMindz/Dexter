#!/usr/bin/env python3
"""
DexBrain Data Ingestion Script

This script feeds historical closed liquidity positions to DexBrain
for machine learning model training and knowledge base updates.
"""

import asyncio
import sys
import os
from pathlib import Path
from typing import List, Dict, Any
import logging
from datetime import datetime

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from historical_position_fetcher import HistoricalPositionFetcher, ClosedPosition
from dexbrain.core import DexBrain
from dexbrain.models.knowledge_base import KnowledgeBase
from dexbrain.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DexBrainDataIngestion:
    """Handles ingestion of historical position data into DexBrain"""
    
    def __init__(self):
        self.fetcher = HistoricalPositionFetcher()
        self.knowledge_base = KnowledgeBase()
        self.stats = {
            'positions_fetched': 0,
            'positions_processed': 0,
            'insights_created': 0,
            'errors': 0
        }
    
    def position_to_insight(self, position: ClosedPosition) -> Dict[str, Any]:
        """Convert a ClosedPosition to a DexBrain insight format"""
        
        return {
            # Position identifiers
            'position_id': position.position_id,
            'pool_address': position.pool_address,
            'blockchain': 'base',
            'protocol': 'uniswap_v3',
            
            # Token information
            'token0_address': position.token0,
            'token1_address': position.token1,
            'fee_tier': position.fee_tier,
            
            # Position parameters
            'tick_lower': position.tick_lower,
            'tick_upper': position.tick_upper,
            'liquidity': position.liquidity,
            
            # Timing information
            'created_timestamp': position.created_at.timestamp(),
            'closed_timestamp': position.closed_at.timestamp(),
            'duration_hours': position.duration_hours,
            'duration_days': position.duration_hours / 24,
            
            # Financial metrics (key features for ML)
            'initial_value_usd': position.initial_usd_value,
            'final_value_usd': position.final_usd_value,
            'fees_earned_usd': position.total_fees_usd,
            'impermanent_loss_usd': position.impermanent_loss_usd,
            'net_pnl_usd': position.net_pnl_usd,
            'roi_percentage': position.roi_percentage,
            'apy': position.apy,
            
            # Token amounts
            'initial_token0_amount': position.initial_token0_amount,
            'initial_token1_amount': position.initial_token1_amount,
            'final_token0_amount': position.final_token0_amount,
            'final_token1_amount': position.final_token1_amount,
            'fees_token0': position.fees_earned_token0,
            'fees_token1': position.fees_earned_token1,
            
            # Market context
            'entry_price': position.entry_price,
            'exit_price': position.exit_price,
            'price_change_percentage': position.price_change_percentage,
            'volume_during_position': position.volume_during_position,
            'avg_volatility': position.avg_volatility,
            
            # Position management
            'close_reason': position.close_reason,
            'gas_costs_usd': position.gas_costs_usd,
            'rebalance_count': position.rebalance_count,
            'time_in_range_percentage': position.time_in_range_percentage,
            
            # Derived metrics for ML features
            'position_size_usd': position.initial_usd_value,
            'fee_yield_percentage': (position.total_fees_usd / position.initial_usd_value * 100) if position.initial_usd_value > 0 else 0,
            'total_return_percentage': (position.net_pnl_usd / position.initial_usd_value * 100) if position.initial_usd_value > 0 else 0,
            'fees_to_il_ratio': position.total_fees_usd / abs(position.impermanent_loss_usd) if position.impermanent_loss_usd != 0 else 0,
            'capital_efficiency': position.total_fees_usd / (position.duration_hours / 24) if position.duration_hours > 0 else 0,
            
            # Metadata
            'data_source': 'historical_fetch',
            'ingestion_timestamp': datetime.now().timestamp(),
            'data_quality_score': self.calculate_data_quality_score(position)
        }
    
    def calculate_data_quality_score(self, position: ClosedPosition) -> float:
        """Calculate a quality score for the position data (0-100)"""
        
        score = 100.0
        
        # Deduct points for missing or questionable data
        if position.duration_hours <= 0:
            score -= 20
        
        if position.initial_usd_value <= 0:
            score -= 20
        
        if position.total_fees_usd < 0:
            score -= 15
        
        if position.time_in_range_percentage == 0:
            score -= 10  # Likely missing data
        
        if position.volume_during_position == 0:
            score -= 10  # Likely missing data
        
        if position.avg_volatility == 0:
            score -= 10  # Likely missing data
        
        # Bonus for realistic metrics
        if 0 <= position.apy <= 1000:  # Reasonable APY range
            score += 5
        
        if position.duration_hours >= 24:  # Position lasted at least a day
            score += 5
        
        return max(0, min(100, score))
    
    async def process_positions_batch(
        self, 
        positions: List[ClosedPosition], 
        category: str = 'base_liquidity_positions'
    ) -> Dict[str, int]:
        """Process a batch of positions and store them in DexBrain"""
        
        batch_stats = {'processed': 0, 'insights_created': 0, 'errors': 0}
        
        for position in positions:
            try:
                # Convert to insight format
                insight = self.position_to_insight(position)
                
                # Filter out low-quality data
                if insight['data_quality_score'] < 50:
                    logger.warning(f"Skipping low-quality position {position.position_id} (score: {insight['data_quality_score']:.1f})")
                    continue
                
                # Store in knowledge base
                success = await self.knowledge_base.store_insight(
                    category=category,
                    insight=insight,
                    metadata={
                        'source': 'historical_position_fetcher',
                        'position_id': position.position_id,
                        'quality_score': insight['data_quality_score']
                    }
                )
                
                if success:
                    batch_stats['insights_created'] += 1
                    logger.debug(f"Stored insight for position {position.position_id}")
                else:
                    batch_stats['errors'] += 1
                    logger.error(f"Failed to store insight for position {position.position_id}")
                
                batch_stats['processed'] += 1
                
            except Exception as e:
                batch_stats['errors'] += 1
                logger.error(f"Error processing position {position.position_id}: {e}")
        
        return batch_stats
    
    async def run_historical_ingestion(
        self, 
        days_back: int = 30, 
        limit: int = 1000,
        min_position_value: float = 100.0
    ) -> Dict[str, Any]:
        """Run the complete historical data ingestion process"""
        
        logger.info(f"Starting historical data ingestion for last {days_back} days...")
        start_time = datetime.now()
        
        try:
            # Fetch historical positions
            async with self.fetcher:
                positions = await self.fetcher.fetch_closed_positions(
                    limit=limit, 
                    days_back=days_back
                )
            
            self.stats['positions_fetched'] = len(positions)
            logger.info(f"Fetched {len(positions)} closed positions")
            
            if not positions:
                logger.warning("No positions found to process")
                return self.stats
            
            # Filter positions by minimum value
            filtered_positions = [
                p for p in positions 
                if p.initial_usd_value >= min_position_value
            ]
            
            logger.info(f"Filtered to {len(filtered_positions)} positions >= ${min_position_value}")
            
            # Process in batches
            batch_size = 50
            total_batches = (len(filtered_positions) + batch_size - 1) // batch_size
            
            for i in range(0, len(filtered_positions), batch_size):
                batch = filtered_positions[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                
                logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} positions)")
                
                batch_stats = await self.process_positions_batch(batch)
                
                # Update overall stats
                self.stats['positions_processed'] += batch_stats['processed']
                self.stats['insights_created'] += batch_stats['insights_created']
                self.stats['errors'] += batch_stats['errors']
                
                logger.info(f"Batch {batch_num} complete: {batch_stats['insights_created']} insights created")
            
            # Calculate summary stats
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            self.stats['duration_seconds'] = duration
            self.stats['positions_per_second'] = self.stats['positions_processed'] / duration if duration > 0 else 0
            self.stats['success_rate'] = (self.stats['insights_created'] / self.stats['positions_processed'] * 100) if self.stats['positions_processed'] > 0 else 0
            
            logger.info(f"Ingestion complete! Stats: {self.stats}")
            
            # Trigger ML training if we have enough data
            if self.stats['insights_created'] >= 10:
                logger.info("Sufficient data available, triggering ML model training...")
                await self.trigger_ml_training()
            
            return self.stats
            
        except Exception as e:
            logger.error(f"Error during historical ingestion: {e}")
            self.stats['errors'] += 1
            raise
    
    async def trigger_ml_training(self):
        """Trigger DexBrain ML model training with the new data"""
        
        try:
            # Initialize DexBrain
            brain = DexBrain()
            
            # Train models with the new insights
            training_result = await brain.train_models('base_liquidity_positions')
            
            logger.info(f"ML training completed: {training_result}")
            
        except Exception as e:
            logger.error(f"Error during ML training: {e}")
    
    async def get_ingestion_status(self) -> Dict[str, Any]:
        """Get the current status of data ingestion"""
        
        # Get insight counts from knowledge base
        insight_count = await self.knowledge_base.get_insight_count('base_liquidity_positions')
        
        return {
            'current_stats': self.stats,
            'total_insights_in_kb': insight_count,
            'knowledge_base_status': 'ready' if insight_count > 0 else 'empty'
        }


async def main():
    """Main execution function"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Ingest historical liquidity position data into DexBrain')
    parser.add_argument('--days-back', type=int, default=7, help='Days back to fetch data (default: 7)')
    parser.add_argument('--limit', type=int, default=500, help='Maximum positions to fetch (default: 500)')
    parser.add_argument('--min-value', type=float, default=100.0, help='Minimum position value in USD (default: 100)')
    parser.add_argument('--status-only', action='store_true', help='Only show current status')
    
    args = parser.parse_args()
    
    # Initialize ingestion system
    ingestion = DexBrainDataIngestion()
    
    if args.status_only:
        # Show current status
        status = await ingestion.get_ingestion_status()
        print(f"\n=== DexBrain Data Ingestion Status ===")
        print(f"Total insights in knowledge base: {status['total_insights_in_kb']}")
        print(f"Knowledge base status: {status['knowledge_base_status']}")
        print(f"Last run stats: {status['current_stats']}")
        return
    
    # Run historical data ingestion
    print(f"\n=== Starting Historical Data Ingestion ===")
    print(f"Days back: {args.days_back}")
    print(f"Position limit: {args.limit}")
    print(f"Minimum position value: ${args.min_value}")
    print()
    
    try:
        result = await ingestion.run_historical_ingestion(
            days_back=args.days_back,
            limit=args.limit,
            min_position_value=args.min_value
        )
        
        print(f"\n=== Ingestion Complete ===")
        print(f"Positions fetched: {result['positions_fetched']}")
        print(f"Positions processed: {result['positions_processed']}")
        print(f"Insights created: {result['insights_created']}")
        print(f"Errors: {result['errors']}")
        print(f"Success rate: {result.get('success_rate', 0):.1f}%")
        print(f"Duration: {result.get('duration_seconds', 0):.1f} seconds")
        
    except Exception as e:
        print(f"\nError during ingestion: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)