# DexBrain API Quick Start Guide

## 🚀 Get Started in 5 Minutes

### 1. Register Your Agent

First, register your agent to get an API key:

```bash
curl -X POST https://api.dexteragent.com/api/register \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "my-trading-bot-v1",
    "metadata": {
      "version": "1.0.0",
      "risk_profile": "aggressive",
      "supported_blockchains": ["base", "ethereum"],
      "supported_dexs": ["uniswap_v3"],
      "description": "My first DexBrain agent"
    }
  }'
```

**Response:**
```json
{
  "api_key": "dx_abcd1234567890efgh...",
  "agent_id": "my-trading-bot-v1",
  "message": "Agent registered successfully"
}
```

### 2. Get Market Intelligence

Use your API key to get market insights:

```bash
curl -X GET "https://api.dexteragent.com/api/intelligence?blockchain=base&limit=10" \
  -H "Authorization: Bearer dx_your_api_key_here"
```

### 3. Submit Performance Data

Share your trading performance to help the network:

```bash
curl -X POST https://api.dexteragent.com/api/submit-data \
  -H "Authorization: Bearer dx_your_api_key_here" \
  -H "Content-Type: application/json" \
  -d '{
    "blockchain": "base",
    "dex_protocol": "uniswap_v3",
    "performance_data": {
      "pool_address": "0x1234...",
      "position_id": "pos_001",
      "total_return_usd": 1250.75,
      "apr": 15.2,
      "win": true
    }
  }'
```

## 💻 Code Examples

### Python

```python
import requests

class DexBrainClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.dexteragent.com"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def get_intelligence(self, blockchain="base"):
        response = requests.get(
            f"{self.base_url}/api/intelligence",
            headers=self.headers,
            params={"blockchain": blockchain}
        )
        return response.json()

# Usage
client = DexBrainClient("dx_your_api_key_here")
intelligence = client.get_intelligence()
print(f"Found {len(intelligence['insights'])} insights")
```

### JavaScript

```javascript
class DexBrainClient {
    constructor(apiKey) {
        this.apiKey = apiKey;
        this.baseUrl = "https://api.dexteragent.com";
    }

    async getIntelligence(blockchain = "base") {
        const response = await fetch(
            `${this.baseUrl}/api/intelligence?blockchain=${blockchain}`,
            {
                headers: {
                    "Authorization": `Bearer ${this.apiKey}`,
                    "Content-Type": "application/json"
                }
            }
        );
        return response.json();
    }
}

// Usage
const client = new DexBrainClient("dx_your_api_key_here");
const intelligence = await client.getIntelligence();
console.log(`Found ${intelligence.insights.length} insights`);
```

## 🔗 Resources

- **Full Documentation**: [API_DOCUMENTATION.md](backend/API_DOCUMENTATION.md)
- **Interactive Docs**: [dexteragent.com/docs](https://dexteragent.com/docs)
- **GitHub Repository**: [github.com/MeltedMindz/Dexter](https://github.com/MeltedMindz/Dexter)
- **Live Brain Monitor**: [dexteragent.com](https://dexteragent.com) (See the "Window into the Brain")

## 📊 Data Quality

The network rewards high-quality data. Ensure your submissions:
- Include all required fields
- Use accurate timestamps
- Provide consistent position IDs
- Submit data promptly (within 1 hour)

Higher quality scores unlock:
- ⚡ Increased rate limits
- 🎯 Priority access to insights
- 🏆 Higher network influence

---

**Ready to connect?** Visit [dexteragent.com/docs](https://dexteragent.com/docs) for the interactive API explorer!