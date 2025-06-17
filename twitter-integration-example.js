/**
 * Example of how to integrate the Twitter service with the Dexter AI agent
 * This shows how to use the twitter-service.cjs module in your existing code
 */

// Import the Twitter service
const TwitterService = require('./twitter-service.cjs');

/**
 * Example integration class showing how to use Twitter service
 */
class DexterTwitterBot {
    constructor() {
        this.twitter = new TwitterService();
        this.isInitialized = false;
    }
    
    /**
     * Initialize the Twitter bot
     */
    async initialize() {
        try {
            console.log('🤖 Initializing Dexter Twitter Bot...');
            
            // Test Twitter connection
            const connected = await this.twitter.testConnection();
            if (!connected) {
                throw new Error('Failed to connect to Twitter API');
            }
            
            this.isInitialized = true;
            console.log('✅ Dexter Twitter Bot initialized successfully');
            
        } catch (error) {
            console.error('❌ Failed to initialize Twitter bot:', error.message);
            throw error;
        }
    }
    
    /**
     * Post a market update tweet
     */
    async postMarketUpdate(marketData) {
        if (!this.isInitialized) {
            throw new Error('Twitter bot not initialized');
        }
        
        try {
            // Format market data into tweet
            const tweet = this.formatMarketTweet(marketData);
            
            // Post the tweet
            const result = await this.twitter.postTweet(tweet);
            
            console.log('📊 Market update posted:', result.data.id);
            return result;
            
        } catch (error) {
            console.error('❌ Failed to post market update:', error.message);
            throw error;
        }
    }
    
    /**
     * Post a trading signal tweet
     */
    async postTradingSignal(signal) {
        if (!this.isInitialized) {
            throw new Error('Twitter bot not initialized');
        }
        
        try {
            const tweet = this.formatTradingSignal(signal);
            const result = await this.twitter.postTweet(tweet);
            
            console.log('📈 Trading signal posted:', result.data.id);
            return result;
            
        } catch (error) {
            console.error('❌ Failed to post trading signal:', error.message);
            throw error;
        }
    }
    
    /**
     * Post a general AI update
     */
    async postAIUpdate(message) {
        if (!this.isInitialized) {
            throw new Error('Twitter bot not initialized');
        }
        
        try {
            const tweet = `🤖 ${message}\n\n#DeFi #AI #Trading #Base`;
            const result = await this.twitter.postTweet(tweet);
            
            console.log('🤖 AI update posted:', result.data.id);
            return result;
            
        } catch (error) {
            console.error('❌ Failed to post AI update:', error.message);
            throw error;
        }
    }
    
    /**
     * Format market data into a tweet
     */
    formatMarketTweet(marketData) {
        const {
            totalLiquidity,
            activePositions,
            performance24h,
            topToken,
            timestamp
        } = marketData;
        
        const performanceEmoji = performance24h >= 0 ? '📈' : '📉';
        const performanceText = performance24h >= 0 ? '+' : '';
        
        return `${performanceEmoji} Dexter Market Update
        
💧 Total Liquidity: $${this.formatNumber(totalLiquidity)}
🎯 Active Positions: ${activePositions}
📊 24h Performance: ${performanceText}${performance24h.toFixed(2)}%
🔥 Top Token: ${topToken}

#DeFi #LiquidityManagement #Base #AI`;
    }
    
    /**
     * Format trading signal into a tweet
     */
    formatTradingSignal(signal) {
        const {
            action,
            token,
            confidence,
            reason,
            targetPrice
        } = signal;
        
        const actionEmoji = action === 'BUY' ? '🟢' : action === 'SELL' ? '🔴' : '🟡';
        
        return `${actionEmoji} Trading Signal Alert
        
🎯 Action: ${action} ${token}
📊 Confidence: ${confidence}%
💡 Reason: ${reason}
${targetPrice ? `🎯 Target: $${targetPrice}` : ''}

⚠️ Not financial advice - DYOR

#Trading #DeFi #AI #Signals`;
    }
    
    /**
     * Format large numbers for display
     */
    formatNumber(num) {
        if (num >= 1000000) {
            return (num / 1000000).toFixed(1) + 'M';
        } else if (num >= 1000) {
            return (num / 1000).toFixed(1) + 'K';
        }
        return num.toString();
    }
    
    /**
     * Monitor mentions and respond appropriately
     */
    async monitorMentions() {
        if (!this.isInitialized) {
            throw new Error('Twitter bot not initialized');
        }
        
        try {
            const mentions = await this.twitter.searchTweets('@YourTwitterHandle', {
                max_results: 10
            });
            
            console.log(`🔍 Found ${mentions.length} mentions`);
            
            for (const mention of mentions) {
                await this.handleMention(mention);
            }
            
        } catch (error) {
            console.error('❌ Failed to monitor mentions:', error.message);
        }
    }
    
    /**
     * Handle individual mention
     */
    async handleMention(mention) {
        try {
            // Simple auto-response logic
            const response = this.generateResponse(mention.text);
            
            if (response) {
                await this.twitter.postTweet(response, {
                    reply_to: mention.id
                });
                
                console.log('💬 Replied to mention:', mention.id);
            }
            
        } catch (error) {
            console.error('❌ Failed to handle mention:', error.message);
        }
    }
    
    /**
     * Generate appropriate response to mention
     */
    generateResponse(mentionText) {
        const text = mentionText.toLowerCase();
        
        if (text.includes('help') || text.includes('info')) {
            return '🤖 I\'m Dexter, an AI liquidity management system for DeFi. I help optimize trading positions and provide market insights. Visit our dashboard for more info!';
        }
        
        if (text.includes('price') || text.includes('market')) {
            return '📊 I analyze market conditions 24/7 to optimize liquidity positions. Check my recent tweets for the latest market updates!';
        }
        
        if (text.includes('thank')) {
            return '🙏 You\'re welcome! Happy to help with DeFi insights and liquidity optimization.';
        }
        
        // Default response for other mentions
        return '🤖 Thanks for the mention! I\'m here to help with DeFi and liquidity management. Feel free to ask questions!';
    }
    
    /**
     * Get current Twitter rate limit status
     */
    getRateLimitStatus() {
        return this.twitter.getRateLimitStatus();
    }
}

// Export the integration class
module.exports = DexterTwitterBot;

// Example usage
if (require.main === module) {
    async function runExample() {
        try {
            // Load environment variables
            try {
                require('dotenv').config();
            } catch (e) {
                console.log('Note: dotenv not available, using system environment variables');
            }
            
            const bot = new DexterTwitterBot();
            await bot.initialize();
            
            // Example market update
            const marketData = {
                totalLiquidity: 1250000,
                activePositions: 12,
                performance24h: 2.34,
                topToken: 'USDC/ETH',
                timestamp: new Date()
            };
            
            console.log('📊 Posting example market update...');
            console.log('⚠️  This is a test - the actual tweet posting is commented out');
            
            // Uncomment to actually post
            // await bot.postMarketUpdate(marketData);
            
            // Example trading signal
            const signal = {
                action: 'BUY',
                token: 'ETH',
                confidence: 85,
                reason: 'Strong support at $2,000 level',
                targetPrice: 2150
            };
            
            console.log('📈 Posting example trading signal...');
            // Uncomment to actually post
            // await bot.postTradingSignal(signal);
            
            // Check rate limits
            const rateLimits = bot.getRateLimitStatus();
            console.log('📊 Rate limit status:', rateLimits);
            
        } catch (error) {
            console.error('❌ Example failed:', error.message);
        }
    }
    
    runExample();
}