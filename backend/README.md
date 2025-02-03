# DexBrain

DexBrain is an AI-powered analytics and strategy engine for Dynamic Liquidity Market Makers (DLMMs). It provides real-time data access, strategy suggestions, and performance metrics for DLMM positions.

## Features

- **DLMM Analytics**: Real-time pool data and metrics
- **AI Strategy Generation**: Risk-adjusted position suggestions
- **Performance Tracking**: Historical data and metrics
- **Fast API Access**: Redis-cached endpoint responses

## Quick Start

### Prerequisites

- Python 3.9+
- PostgreSQL
- Redis
- Solana RPC access

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/dexbrain.git
cd dexbrain
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create `.env` file:
```bash
cp .env.example .env
# Edit .env with your configurations
```

5. Initialize the database:
```bash
python -m db.models
```

6. Start the server:
```bash
uvicorn app.api.routes:app --reload
```

### Configuration

Required environment variables:
```env
SOLANA_RPC_URL=https://api.mainnet-beta.solana.com
REDIS_URL=redis://localhost:6379
POSTGRES_URL=postgresql://user:pass@localhost:5432/dexbrain
LOG_LEVEL=INFO
```

## API Usage

### Get All Pools
```http
GET /api/v1/pools
```

Response:
```json
[
  {
    "pool_id": "string",
    "token_a": "string",
    "token_b": "string",
    "tvl_usd": "string",
    "fee_rate": "string",
    "apy": "string",
    "range_lower": "string",
    "range_upper": "string",
    "status": "string",
    "last_updated": "2024-02-03T12:00:00Z"
  }
]
```

### Get Pool Details
```http
GET /api/v1/pools/{pool_id}
```

### Get Strategy Suggestion
```http
GET /api/v1/pools/{pool_id}/strategy?risk_level=0.5
```

Response:
```json
{
  "pool_id": "string",
  "token_pair": "string",
  "optimal_range": [0.95, 1.05],
  "suggested_fee": 0.1,
  "confidence_score": 0.8,
  "timestamp": "2024-02-03T12:00:00Z"
}
```

## Project Structure

```
dexbrain/
├── app/
│   ├── api/          # FastAPI routes
│   ├── core/         # Core functionality
│   │   ├── agent.py  # AI strategy generation
│   │   └── metrics.py # Metrics collection
│   ├── cache/        # Redis caching
│   └── protocols/    # Protocol adapters
│       └── meteora/  # Meteora implementation
├── config/           # Configuration settings
│   └── settings.py
├── db/              # Database layer
│   └── models.py    # Database models
├── .env             # Environment variables
├── .env.example     # Example environment file
├── requirements.txt # Project dependencies
└── README.md       # Project documentation
```

## Development

### Running Tests
```bash
pytest
```

### Code Style
The project follows PEP 8 style guide. Format code using:
```bash
black .
```

## Architecture

### Components

1. **API Layer** (`app/api/`): FastAPI endpoints for data access
2. **Protocol Layer** (`app/protocols/`): DLMM data integration
3. **Strategy Layer** (`app/core/agent.py`): AI-based position suggestions
4. **Cache Layer** (`app/cache/`): Redis for performance
5. **Database Layer** (`db/`): Historical data storage

### Data Flow

1. Client requests pool data or strategy
2. System checks Redis cache
3. If not cached, fetches from Solana
4. Processes data and generates strategy
5. Caches result and returns response

## Performance Considerations

- Uses Redis caching for frequent requests
- Implements connection pooling
- Handles RPC rate limiting
- Caches strategies for 15 minutes
- Caches pool data for 5 minutes

## Error Handling

The API uses standard HTTP status codes:
- 200: Success
- 400: Bad Request
- 404: Not Found
- 500: Server Error

All errors include detailed messages in the response body.

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[MIT License](LICENSE)

## Support

For support, please open an issue in the GitHub repository.