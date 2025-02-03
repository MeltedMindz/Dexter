# DexBrain Frontend Integration Guide

## Overview

DexBrain provides a REST API for accessing DLMM (Dynamic Liquidity Market Maker) data and AI-generated strategies. Here's how to integrate it into your frontend application.

## Quick Example

```typescript
// Example using TypeScript and Axios

interface Pool {
  pool_id: string;
  token_a: string;
  token_b: string;
  tvl_usd: string;
  fee_rate: string;
  apy: string;
  range_lower: string;
  range_upper: string;
  status: string;
  last_updated: string;
}

interface Strategy {
  pool_id: string;
  token_pair: string;
  optimal_range: [number, number];
  suggested_fee: number;
  confidence_score: number;
  timestamp: string;
}

// Get all DLMM pools
async function getPools(): Promise<Pool[]> {
  const response = await axios.get('http://your-api-url/api/v1/pools');
  return response.data;
}

// Get specific pool details
async function getPool(poolId: string): Promise<Pool> {
  const response = await axios.get(`http://your-api-url/api/v1/pools/${poolId}`);
  return response.data;
}

// Get AI strategy suggestion
async function getStrategy(poolId: string, riskLevel: number): Promise<Strategy> {
  const response = await axios.get(
    `http://your-api-url/api/v1/pools/${poolId}/strategy`,
    { params: { risk_level: riskLevel } }
  );
  return response.data;
}
```

## React Component Examples

### Pool List Component
```tsx
import React, { useState, useEffect } from 'react';
import axios from 'axios';

interface Pool {
  pool_id: string;
  token_a: string;
  token_b: string;
  tvl_usd: string;
  apy: string;
  // ... other fields
}

export const PoolList: React.FC = () => {
  const [pools, setPools] = useState<Pool[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchPools = async () => {
      try {
        const response = await axios.get('http://your-api-url/api/v1/pools');
        setPools(response.data);
      } catch (error) {
        console.error('Error fetching pools:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchPools();
  }, []);

  if (loading) return <div>Loading pools...</div>;

  return (
    <div className="pool-list">
      {pools.map(pool => (
        <div key={pool.pool_id} className="pool-card">
          <h3>{pool.token_a}/{pool.token_b}</h3>
          <div>TVL: ${pool.tvl_usd}</div>
          <div>APY: {pool.apy}%</div>
        </div>
      ))}
    </div>
  );
};
```

### Strategy Component
```tsx
import React, { useState } from 'react';
import axios from 'axios';

interface Strategy {
  optimal_range: [number, number];
  suggested_fee: number;
  confidence_score: number;
}

interface StrategyProps {
  poolId: string;
}

export const StrategyRecommendation: React.FC<StrategyProps> = ({ poolId }) => {
  const [strategy, setStrategy] = useState<Strategy | null>(null);
  const [riskLevel, setRiskLevel] = useState(0.5);

  const fetchStrategy = async () => {
    try {
      const response = await axios.get(
        `http://your-api-url/api/v1/pools/${poolId}/strategy`,
        { params: { risk_level: riskLevel } }
      );
      setStrategy(response.data);
    } catch (error) {
      console.error('Error fetching strategy:', error);
    }
  };

  return (
    <div className="strategy-card">
      <div className="risk-selector">
        <label>Risk Level: {riskLevel}</label>
        <input
          type="range"
          min="0"
          max="1"
          step="0.1"
          value={riskLevel}
          onChange={(e) => setRiskLevel(Number(e.target.value))}
        />
      </div>
      <button onClick={fetchStrategy}>Get Strategy</button>
      
      {strategy && (
        <div className="strategy-details">
          <h4>Recommended Strategy</h4>
          <div>Price Range: {strategy.optimal_range[0]} - {strategy.optimal_range[1]}</div>
          <div>Suggested Fee: {strategy.suggested_fee}%</div>
          <div>Confidence: {strategy.confidence_score * 100}%</div>
        </div>
      )}
    </div>
  );
};
```

## API Response Examples

### Get All Pools Response
```json
{
  "pools": [
    {
      "pool_id": "3W2HKgUa96Z69zzG3LK1g8KdcRAWzAttiLiHfYnKuPw5",
      "token_a": "SOL",
      "token_b": "USDC",
      "tvl_usd": "1234567.89",
      "fee_rate": "0.05",
      "apy": "12.5",
      "range_lower": "18.50",
      "range_upper": "22.50",
      "status": "active",
      "last_updated": "2024-02-03T12:00:00Z"
    }
    // ... more pools
  ]
}
```

### Get Strategy Response
```json
{
  "pool_id": "3W2HKgUa96Z69zzG3LK1g8KdcRAWzAttiLiHfYnKuPw5",
  "token_pair": "SOL/USDC",
  "optimal_range": [18.50, 22.50],
  "suggested_fee": 0.05,
  "confidence_score": 0.85,
  "timestamp": "2024-02-03T12:00:00Z"
}
```

## Error Handling

The API returns standard HTTP status codes:
- 200: Success
- 400: Bad Request
- 404: Not Found
- 500: Server Error

Example error handling:
```typescript
try {
  const response = await getStrategy(poolId, riskLevel);
  // Handle success
} catch (error) {
  if (error.response) {
    switch (error.response.status) {
      case 404:
        console.error('Pool not found');
        break;
      case 400:
        console.error('Invalid parameters');
        break;
      default:
        console.error('Server error');
    }
  }
}
```

## WebSocket Updates (If Needed)
```typescript
const ws = new WebSocket('ws://your-api-url/ws/pools');

ws.onmessage = (event) => {
  const poolUpdate = JSON.parse(event.data);
  // Handle pool update
};
```

## Tips
1. Cache responses when appropriate
2. Implement proper error handling
3. Show loading states during API calls
4. Consider implementing retry logic
5. Use TypeScript interfaces for type safety