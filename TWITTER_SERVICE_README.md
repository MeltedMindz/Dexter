# Twitter Service for Dexter AI

A robust Twitter posting service designed for the Dexter AI trading agent, compatible with Twitter API v2 and OAuth 1.0a authentication.

## Features

- ✅ **Twitter API v2 Compatible**: Uses the latest Twitter API with OAuth 1.0a authentication
- ✅ **CommonJS Module System**: Compatible with existing Node.js infrastructure
- ✅ **Rate Limiting**: Built-in rate limit handling with exponential backoff
- ✅ **Error Handling**: Comprehensive error handling with retry logic
- ✅ **Free Tier Optimized**: Designed for Twitter API v2 free tier (300 tweets per 15 minutes)
- ✅ **VPS Ready**: Optimized for deployment on `/opt/dexter-ai/` VPS structure

## Files Structure

```
twitter-service.cjs                 # Main Twitter service (deploy to VPS)
twitter-integration-example.js      # Usage examples and integration guide
deploy-twitter-service.sh          # VPS deployment script
TWITTER_SERVICE_README.md          # This documentation
```

## Quick Start

### 1. Environment Variables

Add these variables to your `.env` file:

```bash
# Twitter API Credentials (Required)
TWITTER_API_KEY=your_api_key_here
TWITTER_API_SECRET_KEY=your_api_secret_here
TWITTER_ACCESS_TOKEN=your_access_token_here
TWITTER_ACCESS_TOKEN_SECRET=your_access_token_secret_here
```

### 2. Deploy to VPS

```bash
# Deploy the Twitter service to your VPS
./deploy-twitter-service.sh
```

### 3. Test the Service

```bash
# SSH into your VPS and test
ssh root@5.78.71.231
cd /opt/dexter-ai
node test-twitter.js
```

## Usage Examples

### Basic Tweet Posting

```javascript
const TwitterService = require('./twitter-service.cjs');

async function postTweet() {
    const twitter = new TwitterService();
    
    // Test connection first
    await twitter.testConnection();
    
    // Post a tweet
    const result = await twitter.postTweet('🤖 Dexter AI is monitoring the markets!');
    console.log('Tweet posted:', result.data.id);
}
```

### Integration with Dexter AI

```javascript
const TwitterHelper = require('./twitter-helper.js');

class DexterAgent {
    constructor() {
        this.twitter = new TwitterHelper();
    }
    
    async start() {
        // Initialize Twitter
        await this.twitter.init();
        
        // Post market updates
        await this.postMarketUpdate();
    }
    
    async postMarketUpdate() {
        const marketData = {
            totalLiquidity: 1250000,
            activePositions: 12,
            performance24h: 2.34,
            topToken: 'USDC/ETH'
        };
        
        const message = `📊 Market Update
💧 Liquidity: $${this.formatNumber(marketData.totalLiquidity)}
🎯 Positions: ${marketData.activePositions}
📈 24h: +${marketData.performance24h}%
🔥 Top: ${marketData.topToken}

#DeFi #AI #Trading`;
        
        await this.twitter.postMarketUpdate(message);
    }
}
```

## API Reference

### TwitterService Class

#### Constructor
```javascript
new TwitterService()
```
Automatically loads credentials from environment variables.

#### Methods

##### `postTweet(message, options)`
Posts a tweet to Twitter.

**Parameters:**
- `message` (string): Tweet content (max 280 characters)
- `options` (object): Optional parameters
  - `reply_to` (string): Tweet ID to reply to
  - `media_ids` (array): Media IDs to attach

**Returns:** Promise\<Object> - Tweet response data

##### `testConnection()`
Tests the Twitter API connection and displays user info.

**Returns:** Promise\<boolean> - Connection success status

##### `searchTweets(query, options)`
Searches for tweets matching the query.

**Parameters:**
- `query` (string): Search query
- `options` (object): Optional parameters
  - `max_results` (number): Maximum results (default: 10)
  - `since_id` (string): Return tweets after this ID

**Returns:** Promise\<Array> - Array of tweet objects

##### `getRateLimitStatus()`
Returns current rate limit information.

**Returns:** Object with `remaining`, `resetTime`, and `minutesUntilReset`

## Rate Limiting

The service automatically handles Twitter's rate limits:

- **Free Tier Limit**: 300 tweets per 15 minutes
- **Automatic Retry**: Exponential backoff for temporary failures
- **Rate Limit Monitoring**: Tracks remaining requests and reset times

## Error Handling

The service includes comprehensive error handling for:

- **Authentication Errors**: Invalid credentials
- **Rate Limiting**: Automatic waiting and retry
- **Network Issues**: Connection timeouts and retries
- **API Errors**: Twitter API error responses

## Security Features

- **Credential Validation**: Validates all required environment variables
- **OAuth 1.0a**: Secure authentication with signature generation
- **Request Signing**: All requests are properly signed
- **No Credential Logging**: Sensitive data is never logged

## Twitter API v2 Free Tier Limitations

- **Tweet Posting**: 300 tweets per 15-minute window
- **Tweet Reading**: Public tweets and user data
- **Search**: Recent tweets (last 7 days)
- **User Operations**: Basic user information

## Deployment Architecture

```
VPS: 5.78.71.231
└── /opt/dexter-ai/
    ├── twitter-service.cjs         # Main service
    ├── twitter-helper.js           # Integration helper
    ├── test-twitter.js            # Test script
    └── .env                       # Environment variables
```

## Troubleshooting

### Common Issues

1. **"Missing required environment variables"**
   - Ensure all Twitter API credentials are set in `.env`
   - Check variable names match exactly

2. **"Twitter API Error (401)"**
   - Verify API credentials are correct
   - Ensure tokens have proper permissions

3. **"Rate limit reached"**
   - Service will automatically wait and retry
   - Check rate limit status with `getRateLimitStatus()`

4. **"Tweet too long"**
   - Free tier limit is 280 characters
   - Use `message.length` to check before posting

### Testing Commands

```bash
# Test connection only
node -e "const T = require('./twitter-service.cjs'); new T().testConnection()"

# Check rate limits
node -e "const T = require('./twitter-service.cjs'); console.log(new T().getRateLimitStatus())"

# Full integration test
node twitter-integration-example.js
```

## Integration with Eliza Framework

The service is designed to work with the existing Eliza framework:

```javascript
// In your Eliza plugin or character
const TwitterService = require('./twitter-service.cjs');

// Add to your agent's actions
const twitterAction = {
    name: 'POST_TWEET',
    similes: ['tweet', 'post', 'share'],
    description: 'Post a tweet to Twitter',
    handler: async (message, context) => {
        const twitter = new TwitterService();
        return await twitter.postTweet(message);
    }
};
```

## Support

For issues or questions:

1. Check the troubleshooting section above
2. Review the Twitter API v2 documentation
3. Test with the provided examples
4. Check VPS logs: `journalctl -u dexter-agent -f`

## License

Part of the Dexter AI project. See main project license for details.