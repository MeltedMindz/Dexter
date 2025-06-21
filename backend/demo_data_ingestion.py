#!/usr/bin/env python3
"""
Demo script showing DexBrain data ingestion with mock closed positions
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta
import random

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from data_sources.historical_position_fetcher import ClosedPosition
from data_sources.dexbrain_data_ingestion import DexBrainDataIngestion


def create_mock_positions(count: int = 20) -> list[ClosedPosition]:
    """Create mock closed positions for demonstration"""
    
    positions = []
    base_time = datetime.now() - timedelta(days=7)
    
    # Common Base network pools
    pools = [
        {
            'pool_address': '0x4C36388bE6F416A29C8d8Eee81C771cE6bE14B18',  # WETH/USDC 0.05%
            'token0': '0x4200000000000000000000000000000000000006',  # WETH
            'token1': '0x833589fcd6edb6e08f4c7c32d4f71b54bda02913',  # USDC
            'fee_tier': 500
        },
        {
            'pool_address': '0xd0b53D9277642d899DF5C87A3966A349A798F224',  # WETH/USDC 0.30%
            'token0': '0x4200000000000000000000000000000000000006',  # WETH
            'token1': '0x833589fcd6edb6e08f4c7c32d4f71b54bda02913',  # USDC
            'fee_tier': 3000
        },
        {
            'pool_address': '0xc9034c3E7F58003E6ae0C8438e7c8f4598d5ACAA',  # WETH/USDbC 0.05%
            'token0': '0x4200000000000000000000000000000000000006',  # WETH
            'token1': '0xd9aaec86b65d86f6a7b5b1b0c42ffa531710b6ca',  # USDbC
            'fee_tier': 500
        }
    ]
    
    for i in range(count):
        pool = random.choice(pools)
        
        # Random position parameters
        duration_hours = random.uniform(12, 168)  # 12 hours to 1 week
        created_at = base_time + timedelta(hours=random.uniform(0, 120))
        closed_at = created_at + timedelta(hours=duration_hours)
        
        # Random financial metrics
        initial_value = random.uniform(500, 50000)  # $500 to $50k
        fee_yield = random.uniform(0.001, 0.05)  # 0.1% to 5% fees
        il_percentage = random.uniform(-0.02, 0.01)  # -2% to +1% IL
        
        fees_earned = initial_value * fee_yield
        il_amount = initial_value * il_percentage
        net_pnl = fees_earned + il_amount
        roi = (net_pnl / initial_value) * 100
        apy = roi * (365 * 24 / duration_hours)
        
        # Position range (simplified)
        price_range = random.uniform(0.8, 1.2)
        
        position = ClosedPosition(
            position_id=f"mock_{i}_{int(created_at.timestamp())}",
            owner=f"0x{''.join(random.choices('0123456789abcdef', k=40))}",
            pool_address=pool['pool_address'],
            token0=pool['token0'],
            token1=pool['token1'],
            fee_tier=pool['fee_tier'],
            tick_lower=random.randint(-200000, -1000),
            tick_upper=random.randint(1000, 200000),
            liquidity=random.randint(1000000, 100000000),
            
            created_at=created_at,
            closed_at=closed_at,
            duration_hours=duration_hours,
            
            initial_token0_amount=random.uniform(0.1, 10),
            initial_token1_amount=random.uniform(500, 5000),
            final_token0_amount=random.uniform(0.1, 10),
            final_token1_amount=random.uniform(500, 5000),
            fees_earned_token0=random.uniform(0.001, 0.1),
            fees_earned_token1=random.uniform(1, 50),
            
            initial_usd_value=initial_value,
            final_usd_value=initial_value + net_pnl - fees_earned,
            total_fees_usd=fees_earned,
            impermanent_loss_usd=il_amount,
            net_pnl_usd=net_pnl,
            apy=apy,
            roi_percentage=roi,
            
            entry_price=random.uniform(1500, 4000),
            exit_price=random.uniform(1500, 4000),
            price_change_percentage=random.uniform(-15, 15),
            volume_during_position=random.uniform(100000, 5000000),
            avg_volatility=random.uniform(0.1, 0.8),
            
            close_reason=random.choice(['manual', 'rebalance', 'stop_loss', 'target_reached']),
            gas_costs_usd=random.uniform(1, 50),
            rebalance_count=random.randint(0, 5),
            time_in_range_percentage=random.uniform(20, 95)
        )
        
        positions.append(position)
    
    return positions


async def demo_ingestion():
    """Demonstrate the data ingestion process with mock data"""
    
    print("=== DexBrain Data Ingestion Demo ===")
    print()
    
    # Create mock positions
    print("1. Creating mock closed liquidity positions...")
    positions = create_mock_positions(20)
    print(f"   ✓ Created {len(positions)} mock positions")
    
    # Show sample position
    sample = positions[0]
    print(f"\n2. Sample Position:")
    print(f"   Position ID: {sample.position_id}")
    print(f"   Pool: {sample.token0[:10]}.../{sample.token1[:10]}...")
    print(f"   Duration: {sample.duration_hours:.1f} hours")
    print(f"   Initial Value: ${sample.initial_usd_value:.2f}")
    print(f"   Fees Earned: ${sample.total_fees_usd:.2f}")
    print(f"   Net P&L: ${sample.net_pnl_usd:.2f}")
    print(f"   ROI: {sample.roi_percentage:.2f}%")
    print(f"   APY: {sample.apy:.2f}%")
    
    # Initialize ingestion system
    print("\n3. Initializing DexBrain ingestion system...")
    ingestion = DexBrainDataIngestion()
    print("   ✓ Ingestion system ready")
    
    # Process positions
    print("\n4. Processing positions into DexBrain insights...")
    batch_stats = await ingestion.process_positions_batch(positions, 'demo_base_positions')
    
    print(f"   ✓ Processed: {batch_stats['processed']}")
    print(f"   ✓ Insights created: {batch_stats['insights_created']}")
    print(f"   ✓ Errors: {batch_stats['errors']}")
    print(f"   ✓ Success rate: {(batch_stats['insights_created']/batch_stats['processed']*100):.1f}%")
    
    # Show insight example
    if batch_stats['insights_created'] > 0:
        print("\n5. Sample insight format:")
        insight = ingestion.position_to_insight(sample)
        key_fields = [
            'position_id', 'blockchain', 'protocol', 'fee_tier',
            'duration_hours', 'apy', 'roi_percentage', 'fee_yield_percentage',
            'data_quality_score'
        ]
        for field in key_fields:
            value = insight.get(field, 'N/A')
            print(f"   {field}: {value}")
    
    # Check knowledge base status
    print("\n6. Checking DexBrain knowledge base...")
    status = await ingestion.get_ingestion_status()
    print(f"   ✓ Total insights in KB: {status['total_insights_in_kb']}")
    print(f"   ✓ KB Status: {status['knowledge_base_status']}")
    
    # Trigger ML training if enough data
    if status['total_insights_in_kb'] >= 10:
        print("\n7. Triggering ML model training...")
        try:
            await ingestion.trigger_ml_training()
            print("   ✓ ML training completed successfully")
        except Exception as e:
            print(f"   ⚠ ML training failed: {e}")
    else:
        print("\n7. Skipping ML training (need ≥10 insights)")
    
    print("\n=== Demo Complete ===")
    print()
    print("Next steps for real data:")
    print("  1. Get The Graph API key for Base network")
    print("  2. Configure proper price data sources") 
    print("  3. Set up automatic ingestion schedule")
    print("  4. Monitor ML model performance")


if __name__ == "__main__":
    asyncio.run(demo_ingestion())