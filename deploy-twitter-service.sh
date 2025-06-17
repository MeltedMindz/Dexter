#!/bin/bash

# Deploy Twitter Service to VPS
# This script copies the twitter-service.cjs file to the VPS and sets up the integration

set -e

VPS_HOST="5.78.71.231"
VPS_USER="root"
APP_DIR="/opt/dexter-ai"
TWITTER_SERVICE="twitter-service.cjs"

echo "🐦 Deploying Twitter Service to VPS..."
echo "📍 Target: ${VPS_USER}@${VPS_HOST}"
echo "📁 App Directory: ${APP_DIR}"
echo ""

# Check if we can connect to VPS
echo "🔗 Testing VPS connection..."
ssh -o ConnectTimeout=10 ${VPS_USER}@${VPS_HOST} "echo 'VPS connection successful'"

# Create the twitter service directory if it doesn't exist
echo "📁 Setting up Twitter service directory..."
ssh ${VPS_USER}@${VPS_HOST} "
  mkdir -p ${APP_DIR}
  cd ${APP_DIR}
  echo 'Directory ready'
"

# Copy the Twitter service file
echo "📦 Copying Twitter service file..."
scp ${TWITTER_SERVICE} ${VPS_USER}@${VPS_HOST}:${APP_DIR}/

# Copy the integration example (optional)
echo "📦 Copying integration example..."
scp twitter-integration-example.js ${VPS_USER}@${VPS_HOST}:${APP_DIR}/

# Test the Twitter service on VPS
echo "🧪 Testing Twitter service on VPS..."
ssh ${VPS_USER}@${VPS_HOST} "
  cd ${APP_DIR}
  
  # Install Node.js dependencies if needed (crypto, https, etc. are built-in)
  # The service uses only Node.js built-in modules for maximum compatibility
  
  # Test the service (this will validate credentials without posting)
  echo '🔍 Testing Twitter service...'
  node ${TWITTER_SERVICE}
  
  echo 'Twitter service deployed and tested successfully'
"

# Create a simple usage script for the VPS
echo "📝 Creating usage script..."
ssh ${VPS_USER}@${VPS_HOST} "
cat > ${APP_DIR}/test-twitter.js << 'EOF'
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

  chmod +x ${APP_DIR}/test-twitter.js
  echo 'Usage script created'
"

# Update the systemd service to include Twitter functionality (optional)
echo "⚙️ Creating Twitter service helper..."
ssh ${VPS_USER}@${VPS_HOST} "
cat > ${APP_DIR}/twitter-helper.js << 'EOF'
// Twitter helper for Dexter AI agent
// This can be imported and used by the main agent

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

  echo 'Twitter helper created'
"

echo ""
echo "✅ Twitter Service Deployment Complete!"
echo ""
echo "📁 Files deployed to: ${APP_DIR}/"
echo "  - twitter-service.cjs (main service)"
echo "  - twitter-integration-example.js (usage examples)"
echo "  - test-twitter.js (simple test script)"
echo "  - twitter-helper.js (helper for main agent)"
echo ""
echo "🧪 Test commands:"
echo "  ssh ${VPS_USER}@${VPS_HOST} 'cd ${APP_DIR} && node test-twitter.js'"
echo "  ssh ${VPS_USER}@${VPS_HOST} 'cd ${APP_DIR} && node twitter-integration-example.js'"
echo ""
echo "🔧 Integration:"
echo "  // In your main agent code:"
echo "  const TwitterHelper = require('./twitter-helper.js');"
echo "  const twitter = new TwitterHelper();"
echo "  await twitter.postMarketUpdate('Your market update message');"
echo ""
echo "⚠️  Required Environment Variables:"
echo "  - TWITTER_API_KEY"
echo "  - TWITTER_API_SECRET_KEY" 
echo "  - TWITTER_ACCESS_TOKEN"
echo "  - TWITTER_ACCESS_TOKEN_SECRET"
echo ""
echo "🐦 Ready to integrate with your Dexter AI agent!"