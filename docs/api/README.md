# Dexter Protocol API Documentation

> **Comprehensive API reference for the Dexter Protocol backend services**

[![API Version](https://img.shields.io/badge/API-v2.0.0-blue)](https://api.dexteragent.com)
[![Status](https://img.shields.io/badge/Status-Production-green)](https://status.dexteragent.com)
[![Uptime](https://img.shields.io/badge/Uptime-99.9%25-brightgreen)](https://status.dexteragent.com)

## 🌐 API Overview

The Dexter Protocol API provides a comprehensive suite of endpoints for interacting with the AI-powered DeFi infrastructure. The API is organized into multiple services, each handling specific aspects of the protocol.

### Base URLs
- **Production**: `https://api.dexteragent.com`
- **Staging**: `https://staging-api.dexteragent.com`
- **DexBrain Hub**: `https://ai.dexteragent.com` (Port 8001)
- **Enhanced Alchemy**: `https://data.dexteragent.com` (Port 8002)

### Authentication
```bash
# API Key Authentication (Required for all endpoints)
curl -H "Authorization: Bearer YOUR_API_KEY" \
     -H "Content-Type: application/json" \
     https://api.dexteragent.com/v1/positions
```

## 🧠 DexBrain AI Services (Port 8001)

### ML Prediction Endpoints

#### Get ML Predictions
```http
POST /api/ml/predict
Content-Type: application/json
Authorization: Bearer YOUR_API_KEY

{
  "model_type": "fee_predictor",
  "position_data": {
    "token_id": "12345",
    "pool_address": "0x...",
    "tick_lower": -276320,
    "tick_upper": -276300,
    "liquidity": "1000000000000000000",
    "current_price": 2456.78
  }
}
```

**Response:**
```json
{
  "prediction": {
    "predicted_fees": 45.67,
    "confidence": 0.87,
    "recommended_action": "compound",
    "optimal_timing": "2024-12-20T14:30:00Z"
  },
  "model_info": {
    "model_type": "fee_predictor",
    "version": "v2.1.0",
    "last_trained": "2024-12-20T10:00:00Z",
    "accuracy": 0.92
  }
}
```

#### Available Model Types
- `fee_predictor` - Predicts position fee generation
- `range_optimizer` - Determines optimal position ranges
- `volatility_predictor` - Forecasts market volatility
- `yield_optimizer` - Optimizes position yields

### Service Analytics

#### Get Service Status
```http
GET /api/analytics/services
Authorization: Bearer YOUR_API_KEY
```

**Response:**
```json
{
  "services": {
    "position_harvester": {
      "status": "healthy",
      "uptime": "99.9%",
      "last_compound": "2024-12-20T13:45:00Z",
      "positions_managed": 1247
    },
    "vault_processor": {
      "status": "healthy",
      "uptime": "99.8%",
      "vaults_active": 89,
      "total_tvl": "12456789.12"
    },
    "market_analyzer": {
      "status": "healthy",
      "signals_generated": 156,
      "accuracy": 0.89
    },
    "ml_pipeline": {
      "status": "training",
      "models_trained": 4,
      "last_training": "2024-12-20T13:30:00Z"
    }
  }
}
```

### Intelligence Summary

#### Get AI Intelligence Summary
```http
GET /api/intelligence/summary
Authorization: Bearer YOUR_API_KEY
```

**Response:**
```json
{
  "summary": {
    "total_positions": 1247,
    "ai_managed_positions": 892,
    "total_fees_collected": "45678.90",
    "avg_performance": {
      "apr": 18.45,
      "success_rate": 0.94
    },
    "market_conditions": {
      "volatility": "medium",
      "trend": "bullish",
      "recommendation": "increase_allocation"
    },
    "recent_compounds": 45,
    "gas_efficiency": 0.87
  },
  "timestamp": "2024-12-20T14:00:00Z"
}
```

## 📊 Enhanced Alchemy Service (Port 8002)

### Blockchain Data Collection

#### Get Position Data
```http
GET /api/positions/{token_id}
Authorization: Bearer YOUR_API_KEY
```

**Response:**
```json
{
  "position": {
    "token_id": "12345",
    "owner": "0x742d35Cc6637C0532c2c1dD7A2b43F8E7b6b8D9f",
    "pool_address": "0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640",
    "token0": "WETH",
    "token1": "USDC",
    "fee": 3000,
    "tick_lower": -276320,
    "tick_upper": -276300,
    "liquidity": "1000000000000000000",
    "tokens_owed0": "0.0045",
    "tokens_owed1": "12.34",
    "is_ai_managed": true,
    "last_compound_time": 1703087400,
    "compound_count": 15
  },
  "performance": {
    "total_fees": 234.56,
    "total_rewards": 45.67,
    "apr": 18.45,
    "impermanent_loss": -2.3
  }
}
```

#### Get Pool Information
```http
GET /api/pools/{pool_address}
Authorization: Bearer YOUR_API_KEY
```

#### Batch Position Data
```http
POST /api/positions/batch
Content-Type: application/json
Authorization: Bearer YOUR_API_KEY

{
  "token_ids": ["12345", "67890", "11111"],
  "include_performance": true
}
```

### Data Quality Metrics

#### Get Data Quality Status
```http
GET /api/data/quality
Authorization: Bearer YOUR_API_KEY
```

**Response:**
```json
{
  "quality_metrics": {
    "completeness": 0.995,
    "accuracy": 0.992,
    "consistency": 0.988,
    "timeliness": 0.997
  },
  "data_sources": {
    "alchemy_api": "healthy",
    "uniswap_subgraph": "healthy",
    "direct_rpc": "healthy"
  },
  "processing_rate": "127 records/minute",
  "last_update": "2024-12-20T13:59:00Z"
}
```

## 🏦 Vault Management API

### Vault Operations

#### Create Vault
```http
POST /api/vaults
Content-Type: application/json
Authorization: Bearer YOUR_API_KEY

{
  "template_type": "ai_optimized",
  "name": "My AI Vault",
  "description": "AI-managed multi-range vault",
  "initial_deposit": "1000.0",
  "strategy_config": {
    "max_ranges": 5,
    "ai_enabled": true,
    "risk_level": "medium"
  }
}
```

**Response:**
```json
{
  "vault": {
    "address": "0x1234567890123456789012345678901234567890",
    "name": "My AI Vault",
    "template_type": "ai_optimized",
    "tvl": "1000.0",
    "share_token": "0xabcd...",
    "strategy": {
      "max_ranges": 5,
      "current_ranges": 0,
      "ai_enabled": true,
      "risk_level": "medium"
    }
  },
  "transaction_hash": "0xabc123...",
  "gas_used": 456789
}
```

#### Get Vault Details
```http
GET /api/vaults/{vault_address}
Authorization: Bearer YOUR_API_KEY
```

#### List User Vaults
```http
GET /api/vaults?owner={user_address}
Authorization: Bearer YOUR_API_KEY
```

### Position Management

#### Add Position to Vault
```http
POST /api/vaults/{vault_address}/positions
Content-Type: application/json
Authorization: Bearer YOUR_API_KEY

{
  "token_id": "12345",
  "range_type": "base",
  "allocation_percentage": 40.0
}
```

#### Remove Position from Vault
```http
DELETE /api/vaults/{vault_address}/positions/{token_id}
Authorization: Bearer YOUR_API_KEY
```

#### Compound Vault Positions
```http
POST /api/vaults/{vault_address}/compound
Content-Type: application/json
Authorization: Bearer YOUR_API_KEY

{
  "use_ai_optimization": true,
  "reward_conversion": "AI_OPTIMIZED",
  "do_swap": true
}
```

## 📈 Analytics & Performance API

### Performance Metrics

#### Get Position Performance
```http
GET /api/analytics/position/{token_id}/performance
Authorization: Bearer YOUR_API_KEY
```

**Response:**
```json
{
  "performance": {
    "current_value": 2456.78,
    "initial_value": 2000.00,
    "total_return": 456.78,
    "total_return_percentage": 22.84,
    "fees_collected": 234.56,
    "rewards_earned": 45.67,
    "impermanent_loss": -23.45,
    "sharpe_ratio": 2.1,
    "compound_frequency": "daily",
    "gas_efficiency": 0.92
  },
  "period": "30d",
  "last_updated": "2024-12-20T14:00:00Z"
}
```

#### Get Portfolio Analytics
```http
GET /api/analytics/portfolio/{user_address}
Authorization: Bearer YOUR_API_KEY
```

#### Historical Performance
```http
GET /api/analytics/position/{token_id}/history?period=30d&interval=1h
Authorization: Bearer YOUR_API_KEY
```

### Market Data

#### Get Market Regime
```http
GET /api/market/regime
Authorization: Bearer YOUR_API_KEY
```

**Response:**
```json
{
  "regime": {
    "trend": "bullish",
    "volatility": "medium",
    "volume_trend": "increasing",
    "confidence": 0.87
  },
  "indicators": {
    "rsi": 65.4,
    "bollinger_position": 0.75,
    "volume_ma_ratio": 1.23
  },
  "recommendation": {
    "action": "increase_allocation",
    "reason": "Strong bullish trend with moderate volatility",
    "confidence": 0.85
  }
}
```

## 🔐 Security & Compliance API

### Risk Assessment

#### Get Position Risk Analysis
```http
POST /api/risk/position
Content-Type: application/json
Authorization: Bearer YOUR_API_KEY

{
  "token_id": "12345",
  "analysis_type": "comprehensive"
}
```

**Response:**
```json
{
  "risk_analysis": {
    "overall_score": 7.2,
    "factors": {
      "impermanent_loss_risk": 6.5,
      "liquidity_risk": 4.2,
      "smart_contract_risk": 2.1,
      "market_risk": 8.9
    },
    "recommendations": [
      "Consider narrowing price range to reduce IL risk",
      "Monitor market volatility closely",
      "Enable AI optimization for dynamic management"
    ],
    "max_loss_estimate": "5.2%"
  }
}
```

### Compliance

#### Transaction Compliance Check
```http
POST /api/compliance/check
Content-Type: application/json
Authorization: Bearer YOUR_API_KEY

{
  "user_address": "0x742d35Cc6637C0532c2c1dD7A2b43F8E7b6b8D9f",
  "transaction_type": "compound",
  "amount": "1000.0"
}
```

## 🛠️ Utility Endpoints

### Health Checks

#### Service Health
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "2.0.0",
  "uptime": "99.9%",
  "dependencies": {
    "database": "healthy",
    "redis": "healthy",
    "blockchain_rpc": "healthy",
    "ml_services": "healthy"
  }
}
```

### Configuration

#### Get Network Configuration
```http
GET /api/config/network/{chain_id}
Authorization: Bearer YOUR_API_KEY
```

#### Get Fee Structure
```http
GET /api/config/fees
Authorization: Bearer YOUR_API_KEY
```

## 📊 Rate Limits & Error Handling

### Rate Limits
- **Free Tier**: 100 requests/hour
- **Premium**: 1,000 requests/hour
- **Institutional**: 10,000 requests/hour
- **VIP**: Unlimited

### Error Codes

| Code | Message | Description |
|------|---------|-------------|
| 400 | Bad Request | Invalid request parameters |
| 401 | Unauthorized | Invalid or missing API key |
| 403 | Forbidden | Insufficient permissions |
| 404 | Not Found | Resource not found |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Server error |
| 503 | Service Unavailable | Service temporarily unavailable |

### Error Response Format
```json
{
  "error": {
    "code": 400,
    "message": "Invalid token_id format",
    "details": "Token ID must be a positive integer",
    "request_id": "req_abc123",
    "timestamp": "2024-12-20T14:00:00Z"
  }
}
```

## 🔗 Webhooks

### Event Notifications

#### Compound Events
```json
{
  "event": "position_compounded",
  "data": {
    "token_id": "12345",
    "fees_collected": "45.67",
    "gas_used": 123456,
    "transaction_hash": "0xabc123..."
  },
  "timestamp": "2024-12-20T14:00:00Z"
}
```

#### AI Recommendations
```json
{
  "event": "ai_recommendation",
  "data": {
    "position_id": "12345",
    "recommendation": "rebalance_range",
    "confidence": 0.89,
    "reasoning": "High volatility detected, recommend wider range"
  }
}
```

### Webhook Configuration
```http
POST /api/webhooks
Content-Type: application/json
Authorization: Bearer YOUR_API_KEY

{
  "url": "https://your-app.com/webhooks/dexter",
  "events": ["position_compounded", "ai_recommendation"],
  "secret": "your_webhook_secret"
}
```

## 📚 SDK Examples

### JavaScript/TypeScript
```typescript
import { DexterAPI } from '@dexter/sdk'

const api = new DexterAPI({
  apiKey: 'your_api_key',
  network: 'mainnet'
})

// Get position data
const position = await api.positions.get('12345')

// Compound position with AI optimization
const result = await api.positions.compound('12345', {
  useAIOptimization: true,
  rewardConversion: 'AI_OPTIMIZED'
})
```

### Python
```python
from dexter_sdk import DexterClient

client = DexterClient(
    api_key='your_api_key',
    network='mainnet'
)

# Get ML prediction
prediction = client.ml.predict(
    model_type='fee_predictor',
    position_data={
        'token_id': '12345',
        'current_price': 2456.78
    }
)
```

## 📞 Support & Resources

### Getting Help
- **API Documentation**: [docs.dexteragent.com/api](https://docs.dexteragent.com/api)
- **SDK Documentation**: [docs.dexteragent.com/sdk](https://docs.dexteragent.com/sdk)
- **Status Page**: [status.dexteragent.com](https://status.dexteragent.com)
- **Support Email**: api-support@dexteragent.com
- **Discord**: [discord.gg/dexter](https://discord.gg/dexter)

### API Keys
Request API keys at: [dashboard.dexteragent.com/api](https://dashboard.dexteragent.com/api)

---

*Last Updated: December 2024 | API Version: 2.0.0*