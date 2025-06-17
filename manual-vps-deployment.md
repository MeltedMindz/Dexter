# Manual VPS Deployment Instructions

Since the VPS directory `/opt/dexter-ai/` needs to be created with proper permissions, here are the manual steps to deploy the Twitter service:

## Step 1: Connect to VPS and Create Directory

```bash
# Connect to your VPS
ssh root@5.78.71.231

# Create the application directory
mkdir -p /opt/dexter-ai
cd /opt/dexter-ai

# Verify directory creation
pwd
# Should output: /opt/dexter-ai
```

## Step 2: Copy Files to VPS

From your local machine, run these commands:

```bash
# Navigate to the Dexter project directory
cd /Users/melted/Documents/GitHub/Dexter

# Copy the Twitter service file
scp twitter-service.cjs root@5.78.71.231:/opt/dexter-ai/

# Copy the integration example
scp twitter-integration-example.js root@5.78.71.231:/opt/dexter-ai/

# Copy the README for reference
scp TWITTER_SERVICE_README.md root@5.78.71.231:/opt/dexter-ai/
```

## Step 3: Set Up Environment Variables on VPS

```bash
# SSH back into the VPS
ssh root@5.78.71.231
cd /opt/dexter-ai

# Create or update the .env file with Twitter credentials
nano .env

# Add these lines (replace with your actual credentials):
TWITTER_API_KEY=your_api_key_here
TWITTER_API_SECRET_KEY=your_api_secret_here
TWITTER_ACCESS_TOKEN=your_access_token_here
TWITTER_ACCESS_TOKEN_SECRET=your_access_token_secret_here
```

## Step 4: Create Helper Scripts on VPS

```bash
# Still on VPS, create test script
cat > /opt/dexter-ai/test-twitter.js << 'EOF'
// Simple test script for Twitter service
const TwitterService = require('./twitter-service.cjs');

async function testTwitter() {
    try {
        console.log('🤖 Testing Twitter service...');
        
        const twitter = new TwitterService();
        await twitter.testConnection();
        
        console.log('✅ Twitter service is working correctly');
        console.log('📊 Rate limit status:', twitter.getRateLimitStatus());
        
    } catch (error) {
        console.error('❌ Twitter test failed:', error.message);
        process.exit(1);
    }
}

testTwitter();
EOF

# Create Twitter helper for integration
cat > /opt/dexter-ai/twitter-helper.js << 'EOF'
// Twitter helper for Dexter AI agent
const TwitterService = require('./twitter-service.cjs');

class DexterTwitterHelper {
    constructor() {
        this.twitter = new TwitterService();
        this.initialized = false;
    }
    
    async init() {
        if (!this.initialized) {
            await this.twitter.testConnection();
            this.initialized = true;
            console.log('✅ Twitter helper initialized');
        }
    }
    
    async postMarketUpdate(message) {
        await this.init();
        return await this.twitter.postTweet(message);
    }
    
    async postTradingSignal(message) {
        await this.init();
        return await this.twitter.postTweet(message);
    }
    
    getRateLimitStatus() {
        return this.twitter.getRateLimitStatus();
    }
}

module.exports = DexterTwitterHelper;
EOF

# Make scripts executable
chmod +x test-twitter.js
chmod +x twitter-helper.js
```

## Step 5: Test the Installation

```bash
# Test the Twitter service
node test-twitter.js

# If successful, you should see:
# 🤖 Testing Twitter service...
# 🔍 Testing Twitter API connection...
# ✅ Connected as @YourUsername (Your Name)
# 📊 Rate limit status: 300 requests remaining
# ✅ Twitter service is working correctly
```

## Step 6: Verify File Structure

```bash
# Check that all files are in place
ls -la /opt/dexter-ai/

# You should see:
# twitter-service.cjs
# twitter-integration-example.js
# test-twitter.js
# twitter-helper.js
# .env
# TWITTER_SERVICE_README.md
```

## Step 7: Integration with Existing Agent

Once the files are deployed, you can integrate the Twitter service with your existing Dexter AI agent by adding this to your agent code:

```javascript
// In your main agent file
const TwitterHelper = require('/opt/dexter-ai/twitter-helper.js');

class DexterAgent {
    constructor() {
        this.twitter = new TwitterHelper();
    }
    
    async initialize() {
        // Initialize Twitter along with other services
        await this.twitter.init();
        console.log('🐦 Twitter service ready');
    }
    
    async postUpdate(message) {
        try {
            await this.twitter.postMarketUpdate(message);
            console.log('✅ Posted to Twitter');
        } catch (error) {
            console.error('❌ Twitter post failed:', error.message);
        }
    }
}
```

## Troubleshooting

If you encounter issues:

1. **Permission Errors**: Ensure you're running as root or have proper permissions
2. **Module Not Found**: Verify the file paths and that files were copied correctly
3. **Authentication Errors**: Double-check your Twitter API credentials in the .env file
4. **Network Issues**: Ensure the VPS has internet access

## Next Steps

After successful deployment:

1. Integrate the Twitter service with your main Dexter AI agent
2. Set up scheduled posts for market updates
3. Configure mention monitoring and auto-responses
4. Monitor the logs to ensure everything is working correctly

The Twitter service is now ready to use with your Dexter AI agent on the VPS!