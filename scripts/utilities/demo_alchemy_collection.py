#!/usr/bin/env python3
"""
Demo script to showcase Alchemy-based position data collection
Demonstrates direct on-chain data collection from Base network
"""

import asyncio
import logging
import json
import os
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Import our Alchemy collector
from backend.dexbrain.alchemy_position_collector import AlchemyPositionCollector

# Website logging
website_logger = logging.getLogger('website')
website_handler = logging.FileHandler('/var/log/dexter/liquidity.log')
website_handler.setFormatter(logging.Formatter('%(message)s'))
website_logger.addHandler(website_handler)
website_logger.setLevel(logging.INFO)

async def demo_alchemy_collection():
    """
    Demonstrate Alchemy-based position collection
    """
    logger.info("🚀 Starting Alchemy position collection demo...")
    
    # Log demo start
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "level": "INFO",
        "source": "AlchemyDemo",
        "message": "Alchemy position collection demo started",
        "details": {
            "network": "Base",
            "data_source": "Alchemy RPC",
            "features": ["Direct on-chain data", "Closed positions", "Event logs", "Block-level precision"]
        }
    }
    website_logger.info(json.dumps(log_data))
    
    # Initialize collector with API key
    alchemy_key = os.getenv('ALCHEMY_API_KEY', 'demo')
    
    try:
        collector = AlchemyPositionCollector(alchemy_key)
        
        # Collect closed positions from last 5000 blocks
        logger.info("📊 Collecting closed positions from last 5000 blocks...")
        closed_positions = await collector.collect_closed_positions(
            block_range=5000,
            limit=30
        )
        
        # Log collection results
        if closed_positions:
            logger.info(f"✅ Successfully collected {len(closed_positions)} closed positions")
            
            # Show sample data
            for i, position in enumerate(closed_positions[:5]):
                logger.info(f"\nPosition {i+1}:")
                logger.info(f"  Token ID: {position.token_id}")
                logger.info(f"  Pool: {position.token0[:10]}.../{position.token1[:10]}...")
                logger.info(f"  Fee Tier: {position.fee_tier}")
                logger.info(f"  Range: [{position.tick_lower}, {position.tick_upper}] (width: {position.range_width})")
                logger.info(f"  Tokens Owed: {position.tokens_owed0} / {position.tokens_owed1}")
                logger.info(f"  Closed at: Block {position.closed_at_block} ({position.closed_at_timestamp})")
            
            # Export to CSV
            csv_filename = f"alchemy_demo_positions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            await collector.export_to_csv(closed_positions, csv_filename)
            logger.info(f"📄 Exported positions to {csv_filename}")
            
            # Log success
            log_data = {
                "timestamp": datetime.now().isoformat(),
                "level": "SUCCESS",
                "source": "AlchemyDemo",
                "message": f"Successfully collected {len(closed_positions)} positions via Alchemy RPC",
                "details": {
                    "positions_collected": len(closed_positions),
                    "blocks_scanned": collector.collection_stats['blocks_scanned'],
                    "rpc_calls": collector.collection_stats['rpc_calls'],
                    "average_range_width": sum(p.range_width for p in closed_positions) / len(closed_positions),
                    "unique_pools": len(set((p.token0, p.token1, p.fee_tier) for p in closed_positions)),
                    "export_file": csv_filename
                }
            }
            website_logger.info(json.dumps(log_data))
            
        else:
            logger.warning("No closed positions found in the specified block range")
        
        # Show collection summary
        summary = collector.get_collection_summary()
        logger.info(f"\n📊 Collection Summary:")
        logger.info(f"  Total positions: {summary['total_positions']}")
        logger.info(f"  RPC calls made: {summary['collection_stats']['rpc_calls']}")
        logger.info(f"  Blocks scanned: {summary['collection_stats']['blocks_scanned']}")
        logger.info(f"  Error rate: {summary['error_rate']:.1%}")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "level": "ERROR",
            "source": "AlchemyDemo",
            "message": f"Demo failed: {e}",
            "details": {
                "error_type": type(e).__name__,
                "error_message": str(e)
            }
        }
        website_logger.info(json.dumps(log_data))

async def main():
    """Run the demo"""
    await demo_alchemy_collection()

if __name__ == "__main__":
    asyncio.run(main())