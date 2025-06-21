# Historical Liquidity Position Data Ingestion for DexBrain

## Overview

This system automatically fetches real closed liquidity positions from Base network and feeds them to DexBrain for machine learning model training. It uses multiple data sources to ensure comprehensive coverage of historical position data.

## Data Sources

### 1. The Graph Protocol Subgraphs
- **Primary Source**: Uniswap V3 subgraph on Base network
- **Endpoints**: 
  - Messari's standardized subgraph: `FUbEPQw1oMghy39fwWBFY5fE6MXPXZQtjncQy2cXdrNS`
  - Community subgraph: `HMuAwufqZ1YCRmzL2SfHTVkzZovC9VL2UAKhjvRqKiR1`
- **Data**: Closed positions with full lifecycle data
- **Rate Limits**: Public endpoints limited, API key recommended

### 2. Direct Blockchain Queries
- **Source**: Base network RPC via Alchemy
- **Purpose**: Supplement subgraph data with transaction details
- **Use Case**: Position creation events, gas costs, precise timing

### 3. Future Integrations
- Price history APIs for accurate USD calculations
- Volume and volatility data from DEX aggregators
- Additional DEX protocols (Aerodrome, etc.)

## Data Structure

### Closed Position Fields
Each closed position contains:

**Core Identifiers:**
- Position ID (NFT token ID)
- Pool address and token pair
- Owner address
- Fee tier and tick range

**Lifecycle Data:**
- Creation and closure timestamps
- Position duration
- Close reason (manual, auto-rebalance, etc.)

**Financial Metrics:**
- Initial and final token amounts
- Fees earned in both tokens
- USD values and conversions
- ROI, APY, and P&L calculations

**Performance Indicators:**
- Impermanent loss calculations
- Time in range percentage
- Rebalancing frequency
- Capital efficiency metrics

**Market Context:**
- Entry and exit prices
- Price movements during position
- Volume and volatility data
- Market conditions

## Data Quality Scoring

Each position receives a quality score (0-100) based on:

- **Complete Data**: All required fields present (+20 points)
- **Realistic Duration**: Position lasted reasonable time (+10 points)
- **Valid Metrics**: APY and ROI within expected ranges (+10 points)
- **Rich Context**: Price and volume data available (+10 points)

Positions below 50% quality score are filtered out.

## DexBrain Integration

### Knowledge Base Storage
Positions are stored as "insights" in the knowledge base with category `base_liquidity_positions`.

### ML Feature Engineering
Key features extracted for ML models:
- Position size and duration
- Fee yield vs impermanent loss
- Market volatility during position
- Capital efficiency metrics
- Time-based performance patterns

### Model Training Triggers
- Automatic training when >10 new insights added
- Scheduled retraining every 24 hours
- Manual training via DexBrain API

## Deployment Architecture

### VPS Setup
```
/opt/dexbrain/
├── data_sources/
│   ├── historical_position_fetcher.py  # Core fetching logic
│   └── dexbrain_data_ingestion.py     # DexBrain integration
├── data/
│   ├── historical/                    # Raw data cache
│   └── backups/                       # Data backups
└── venv/                             # Python environment
```

### Services
1. **dexbrain-ingestion.service**: One-shot ingestion runs
2. **dexbrain-ingestion.timer**: Automatic scheduling (every 6 hours)
3. **Manual script**: `/opt/dexbrain/run_ingestion.sh`

## Usage

### Automatic Operation
The system runs automatically every 6 hours, fetching the last 7 days of data.

### Manual Execution
```bash
# SSH to VPS
ssh root@5.78.71.231

# Run manual ingestion
/opt/dexbrain/run_ingestion.sh

# Custom parameters
cd /opt/dexbrain && source venv/bin/activate
python data_sources/dexbrain_data_ingestion.py --days-back 14 --limit 1000 --min-value 500

# Check status only
python data_sources/dexbrain_data_ingestion.py --status-only
```

### Monitoring
```bash
# Check service status
systemctl status dexbrain-ingestion.timer

# View logs
tail -f /opt/dexter-ai/dexbrain.log | grep -E "ingestion|position"

# Check DexBrain health
curl http://localhost:8080/health
```

## Configuration

### Environment Variables (.env)
```bash
# Data Sources
GRAPH_API_KEY=your_api_key_here           # Optional but recommended
BASE_RPC_URL=https://base-mainnet.g.alchemy.com/v2/your_key
ALCHEMY_API_KEY=your_alchemy_key

# Quality Filters
MIN_POSITION_VALUE_USD=100                # Filter small positions
MAX_DAYS_BACK=30                         # Historical data limit
DEFAULT_BATCH_SIZE=50                     # Processing batch size

# Automation
AUTO_INGESTION_ENABLED=true              # Enable automatic runs
INGESTION_INTERVAL_HOURS=6               # Run frequency
```

### Service Configuration
```bash
# Modify ingestion frequency
systemctl edit dexbrain-ingestion.timer

# Change ingestion parameters
systemctl edit dexbrain-ingestion.service
```

## API Integration

### Register Agent for Intelligence
```bash
curl -X POST http://5.78.71.231:8080/api/register \
  -H 'Content-Type: application/json' \
  -d '{"agent_id": "my-trading-agent"}'
```

### Get Historical Intelligence
```bash
curl -H "Authorization: Bearer dx_your_api_key" \
  "http://5.78.71.231:8080/api/intelligence?blockchain=base&limit=100"
```

### Submit New Position Data
```bash
curl -X POST http://5.78.71.231:8080/api/submit-data \
  -H "Authorization: Bearer dx_your_api_key" \
  -H 'Content-Type: application/json' \
  -d '{
    "blockchain": "base",
    "pool_address": "0x...",
    "performance_data": {...}
  }'
```

## Performance Optimization

### Batch Processing
- Processes positions in batches of 50
- Parallel processing where possible
- Memory-efficient streaming for large datasets

### Caching Strategy
- Raw position data cached locally
- Incremental updates to avoid re-processing
- Backup storage for data recovery

### Rate Limiting
- Respects The Graph API rate limits
- Exponential backoff on failures
- Multiple endpoint fallbacks

## Troubleshooting

### Common Issues

**No positions found:**
- Check Graph API key configuration
- Verify Base network subgraph availability
- Try different time ranges

**Low data quality scores:**
- Review quality scoring criteria
- Check for missing price data
- Verify USD conversion accuracy

**Ingestion failures:**
- Check DexBrain service status
- Verify database connectivity
- Review error logs

**ML training not triggered:**
- Ensure minimum 10 insights threshold
- Check training logs
- Verify model storage permissions

### Debug Commands
```bash
# Test Graph API connectivity
curl -X POST https://gateway.thegraph.com/api/[KEY]/subgraphs/id/HMuAwufqZ1YCRmzL2SfHTVkzZovC9VL2UAKhjvRqKiR1 \
  -H 'Content-Type: application/json' \
  -d '{"query": "{ positions(first: 1) { id } }"}'

# Test DexBrain integration
cd /opt/dexbrain && source venv/bin/activate
python -c "from dexbrain.core import DexBrain; print('DexBrain import OK')"

# Check knowledge base
python -c "
from dexbrain.models.knowledge_base import KnowledgeBase
kb = KnowledgeBase()
import asyncio
count = asyncio.run(kb.get_insight_count('base_liquidity_positions'))
print(f'Insights in KB: {count}')
"
```

## Future Enhancements

### Planned Features
1. **Multi-DEX Support**: Aerodrome, PancakeSwap, etc.
2. **Real-time Streaming**: Live position updates
3. **Advanced Analytics**: Cohort analysis, strategy backtesting
4. **Risk Modeling**: Position risk scoring and alerts
5. **Cross-chain Analysis**: Ethereum, Arbitrum, Optimism

### Data Enrichment
1. **Token Metadata**: Prices, market caps, social metrics
2. **Macro Context**: Market conditions, volatility regimes
3. **User Behavior**: Trading patterns, position strategies
4. **Protocol Metrics**: TVL trends, fee collection patterns

### ML Model Improvements
1. **Feature Engineering**: Time-series features, technical indicators
2. **Model Ensemble**: Multiple models for different scenarios
3. **Online Learning**: Continuous model updates
4. **Prediction Types**: APY forecasting, optimal ranges, exit timing