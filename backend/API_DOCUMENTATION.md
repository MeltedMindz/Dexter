# DexBrain Global Intelligence Network API

## Overview

The DexBrain API enables any developer to connect their liquidity management agents to the global intelligence network. All agents contribute performance data and benefit from collective insights, creating a shared knowledge base that improves DeFi efficiency for everyone.

## Base URL
```
https://api.dexteragent.com
```

## Authentication

All API endpoints (except registration) require an API key in the Authorization header:

```bash
Authorization: Bearer dx_your_api_key_here
```

## Rate Limits

- **Standard**: 100 requests per minute per API key
- **Quality Tier**: Rate limits may be adjusted based on data quality score

## Endpoints

### 1. Agent Registration

Register a new agent and receive an API key.

```http
POST /api/register
```

**Request Body:**
```json
{
  "agent_id": "my-trading-bot-v1",
  "metadata": {
    "version": "1.0.0",
    "risk_profile": "aggressive",
    "supported_blockchains": ["base", "ethereum"],
    "supported_dexs": ["uniswap_v3", "uniswap_v4"],
    "description": "High-frequency arbitrage bot"
  }
}
```

**Response:**
```json
{
  "api_key": "dx_abcd1234567890efgh...",
  "agent_id": "my-trading-bot-v1",
  "message": "Agent registered successfully"
}
```

### 2. Get Intelligence Feed

Retrieve market intelligence, predictions, and network insights.

```http
GET /api/intelligence?blockchain=base&limit=100
```

**Query Parameters:**
- `blockchain` (optional): Target blockchain ("base", "ethereum", "solana")
- `pool_address` (optional): Specific pool to get predictions for
- `category` (optional): Data category filter
- `limit` (optional): Max insights to return (default: 100, max: 1000)

**Response:**
```json
{
  "insights": [
    {
      "pool_address": "0x1234...",
      "blockchain": "base",
      "dex_protocol": "uniswap_v3",
      "total_liquidity": 1500000.0,
      "volume_24h": 750000.0,
      "fee_tier": 0.003,
      "apr": 12.5,
      "quality_score": 87.3,
      "timestamp": "2024-12-15T10:30:00Z"
    }
  ],
  "predictions": [
    {
      "pool_address": "0x1234...",
      "predicted_apr": 15.2,
      "confidence": 0.78,
      "data_points": 156
    }
  ],
  "network_stats": {
    "total_agents": 1247,
    "active_agents": 892,
    "total_volume": 12500000.0,
    "average_performance_score": 72.4
  },
  "benchmarks": {
    "median_apr": 8.7,
    "top_10_percent_score": 85.2,
    "median_win_rate": 0.67
  }
}
```

### 3. Submit Performance Data

Submit your agent's performance data to contribute to the network.

```http
POST /api/submit-data
```

**Request Body:**
```json
{
  "blockchain": "base",
  "dex_protocol": "uniswap_v3",
  "performance_data": {
    "pool_address": "0x1234...",
    "position_id": "pos_001",
    "total_return_usd": 1250.75,
    "total_return_percent": 12.5,
    "fees_earned_usd": 125.30,
    "impermanent_loss_usd": -45.20,
    "net_profit_usd": 1180.85,
    "apr": 15.2,
    "duration_hours": 168,
    "gas_costs_usd": 24.70,
    "slippage_percent": 0.15,
    "win": true,
    "liquidity_amount": 10000.0,
    "entry_price": 1850.50,
    "exit_price": 1920.75
  }
}
```

**Response:**
```json
{
  "status": "accepted",
  "message": "Data submitted successfully",
  "quality_report": {
    "overall_score": 87.3,
    "quality_scores": {
      "completeness": 95.0,
      "accuracy": 92.1,
      "consistency": 88.5,
      "timeliness": 94.2,
      "validity": 89.7
    },
    "is_accepted": true,
    "recommendations": [
      "Excellent data quality - keep up the good work!"
    ]
  },
  "timestamp": "2024-12-15T10:30:00Z"
}
```

### 4. Get Network Statistics

Public endpoint for network-wide statistics.

```http
GET /api/stats
```

**Response:**
```json
{
  "network_stats": {
    "total_agents": 1247,
    "active_agents": 892,
    "total_insights": 156789,
    "recent_submissions_24h": 2341,
    "supported_blockchains": ["base", "ethereum", "solana"],
    "categories": {
      "base_liquidity": 45678,
      "ethereum_liquidity": 67890,
      "solana_liquidity": 43221
    }
  },
  "timestamp": "2024-12-15T10:30:00Z"
}
```

### 5. List Network Agents

Get information about agents in the network.

```http
GET /api/agents
```

**Response:**
```json
{
  "agents": {
    "agent_001": {
      "agent_id": "agent_001",
      "created_at": "2024-11-01T10:00:00Z",
      "last_used": "2024-12-15T09:45:00Z",
      "request_count": 15678,
      "data_submissions": 892,
      "is_active": true
    }
  },
  "total_count": 1247,
  "timestamp": "2024-12-15T10:30:00Z"
}
```

## Data Schemas

### Standardized Position Data

```json
{
  "position_id": "unique_position_identifier",
  "agent_id": "your_agent_id",
  "pool": {
    "address": "0x1234...",
    "blockchain": "base",
    "dex_protocol": "uniswap_v3",
    "token0": {
      "address": "0xabcd...",
      "symbol": "WETH",
      "name": "Wrapped Ether",
      "decimals": 18
    },
    "token1": {
      "address": "0xefgh...",
      "symbol": "USDC",
      "name": "USD Coin",
      "decimals": 6
    },
    "fee_tier": 0.003,
    "total_liquidity_usd": 1500000.0,
    "volume_24h_usd": 750000.0
  },
  "status": "active",
  "liquidity_amount": 10000.0,
  "token0_amount": 5.4,
  "token1_amount": 9850.0,
  "position_value_usd": 10000.0,
  "entry_price": 1850.50,
  "current_price": 1920.75,
  "price_range_lower": 1800.0,
  "price_range_upper": 2000.0,
  "fees_earned_usd": 125.30,
  "impermanent_loss_usd": -45.20,
  "created_at": "2024-12-08T10:00:00Z",
  "updated_at": "2024-12-15T10:00:00Z"
}
```

### Performance Metrics

```json
{
  "position_id": "unique_position_identifier",
  "agent_id": "your_agent_id",
  "total_return_usd": 1250.75,
  "total_return_percent": 12.5,
  "fees_earned_usd": 125.30,
  "impermanent_loss_usd": -45.20,
  "net_profit_usd": 1180.85,
  "apr": 15.2,
  "duration_hours": 168,
  "gas_costs_usd": 24.70,
  "slippage_percent": 0.15,
  "win": true,
  "timestamp": "2024-12-15T10:00:00Z"
}
```

## Integration Examples

### Python Integration

```python
import requests
import json

class DexBrainClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.dexteragent.com"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def get_intelligence(self, blockchain="base", limit=100):
        """Get market intelligence and predictions"""
        response = requests.get(
            f"{self.base_url}/api/intelligence",
            headers=self.headers,
            params={"blockchain": blockchain, "limit": limit}
        )
        return response.json()
    
    def submit_performance(self, blockchain, dex_protocol, performance_data):
        """Submit performance data"""
        payload = {
            "blockchain": blockchain,
            "dex_protocol": dex_protocol,
            "performance_data": performance_data
        }
        response = requests.post(
            f"{self.base_url}/api/submit-data",
            headers=self.headers,
            json=payload
        )
        return response.json()

# Usage example
client = DexBrainClient("dx_your_api_key_here")

# Get intelligence
intelligence = client.get_intelligence(blockchain="base", limit=50)
print(f"Found {len(intelligence['insights'])} insights")

# Submit performance data
performance = {
    "pool_address": "0x1234...",
    "position_id": "pos_001",
    "total_return_usd": 1250.75,
    "apr": 15.2,
    "win": True
    # ... other metrics
}

result = client.submit_performance("base", "uniswap_v3", performance)
print(f"Submission status: {result['status']}")
```

### JavaScript/Node.js Integration

```javascript
class DexBrainClient {
    constructor(apiKey) {
        this.apiKey = apiKey;
        this.baseUrl = "https://api.dexteragent.com";
    }

    async getIntelligence(blockchain = "base", limit = 100) {
        const response = await fetch(
            `${this.baseUrl}/api/intelligence?blockchain=${blockchain}&limit=${limit}`,
            {
                headers: {
                    "Authorization": `Bearer ${this.apiKey}`,
                    "Content-Type": "application/json"
                }
            }
        );
        return response.json();
    }

    async submitPerformance(blockchain, dexProtocol, performanceData) {
        const response = await fetch(
            `${this.baseUrl}/api/submit-data`,
            {
                method: "POST",
                headers: {
                    "Authorization": `Bearer ${this.apiKey}`,
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    blockchain,
                    dex_protocol: dexProtocol,
                    performance_data: performanceData
                })
            }
        );
        return response.json();
    }
}

// Usage example
const client = new DexBrainClient("dx_your_api_key_here");

// Get intelligence
const intelligence = await client.getIntelligence("base", 50);
console.log(`Found ${intelligence.insights.length} insights`);

// Submit performance data
const performance = {
    pool_address: "0x1234...",
    position_id: "pos_001",
    total_return_usd: 1250.75,
    apr: 15.2,
    win: true
    // ... other metrics
};

const result = await client.submitPerformance("base", "uniswap_v3", performance);
console.log(`Submission status: ${result.status}`);
```

## Error Handling

### HTTP Status Codes

- `200` - Success
- `400` - Bad Request (invalid data)
- `401` - Unauthorized (invalid API key)
- `429` - Rate Limit Exceeded
- `500` - Internal Server Error

### Error Response Format

```json
{
  "error": "Invalid API key",
  "code": 401,
  "timestamp": "2024-12-15T10:30:00Z"
}
```

## Data Quality Guidelines

To ensure high data quality and maximize your agent's benefit from the network:

### Required Fields
- Always include all required position and performance fields
- Provide accurate timestamps in ISO format
- Use standard blockchain/DEX protocol names

### Best Practices
- Submit data promptly (within 1 hour of position changes)
- Include optional fields like price ranges when available
- Validate calculations before submission
- Use consistent position IDs across related submissions

### Quality Scoring
Your submissions are scored on:
- **Completeness** (95%+): All required fields present
- **Accuracy** (90%+): Reasonable values and calculations
- **Consistency** (90%+): Internal data consistency
- **Timeliness** (95%+): Recent and timely submissions
- **Validity** (95%+): Proper formats and business logic

Higher quality scores may result in:
- Increased rate limits
- Priority access to premium insights
- Higher weighting in network intelligence

## Support

- **Documentation**: [GitHub Repository](https://github.com/MeltedMindz/Dexter)
- **Issues**: [GitHub Issues](https://github.com/MeltedMindz/Dexter/issues)
- **Community**: [Discord](https://discord.gg/dexter) (coming soon)

---

*The DexBrain Global Intelligence Network: Making DeFi better for everyone through shared knowledge.*