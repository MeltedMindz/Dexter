/**
 * Twitter Posting Service for Dexter AI
 * Compatible with Twitter API v2 and OAuth 1.0a authentication
 * Uses CommonJS module system for VPS compatibility
 */

const crypto = require('crypto');
const https = require('https');
const querystring = require('querystring');
const { URL } = require('url');

class TwitterService {
    constructor() {
        // Load credentials from environment variables
        this.apiKey = process.env.TWITTER_API_KEY;
        this.apiSecretKey = process.env.TWITTER_API_SECRET_KEY;
        this.accessToken = process.env.TWITTER_ACCESS_TOKEN;
        this.accessTokenSecret = process.env.TWITTER_ACCESS_TOKEN_SECRET;
        
        // Rate limiting configuration for FREE TIER
        this.rateLimitReset = null;
        this.remainingRequests = 17; // Twitter API v2 FREE TIER: Only 17 tweets per 24 hours!
        this.dailyPostCount = 0;
        this.dailyResetTime = null;
        this.retryDelay = 60000; // 1 minute base retry delay
        this.maxRetries = 3;
        
        // Validate credentials on initialization
        this.validateCredentials();
    }
    
    /**
     * Validate that all required credentials are present
     */
    validateCredentials() {
        const requiredVars = [
            'TWITTER_API_KEY',
            'TWITTER_API_SECRET_KEY', 
            'TWITTER_ACCESS_TOKEN',
            'TWITTER_ACCESS_TOKEN_SECRET'
        ];
        
        const missing = requiredVars.filter(varName => !process.env[varName]);
        
        if (missing.length > 0) {
            throw new Error(`Missing required environment variables: ${missing.join(', ')}`);
        }
        
        console.log('✅ Twitter credentials validated');
    }
    
    /**
     * Generate OAuth 1.0a signature for Twitter API requests
     */
    generateOAuthSignature(method, url, params) {
        // OAuth parameters
        const oauthParams = {
            oauth_consumer_key: this.apiKey,
            oauth_token: this.accessToken,
            oauth_signature_method: 'HMAC-SHA1',
            oauth_timestamp: Math.floor(Date.now() / 1000).toString(),
            oauth_nonce: crypto.randomBytes(16).toString('hex'),
            oauth_version: '1.0'
        };
        
        // Combine OAuth params with request params
        const allParams = { ...oauthParams, ...params };
        
        // Create parameter string
        const paramString = Object.keys(allParams)
            .sort()
            .map(key => `${encodeURIComponent(key)}=${encodeURIComponent(allParams[key])}`)
            .join('&');
        
        // Create signature base string
        const signatureBaseString = [
            method.toUpperCase(),
            encodeURIComponent(url),
            encodeURIComponent(paramString)
        ].join('&');
        
        // Create signing key
        const signingKey = [
            encodeURIComponent(this.apiSecretKey),
            encodeURIComponent(this.accessTokenSecret)
        ].join('&');
        
        // Generate signature
        const signature = crypto
            .createHmac('sha1', signingKey)
            .update(signatureBaseString)
            .digest('base64');
        
        oauthParams.oauth_signature = signature;
        
        return oauthParams;
    }
    
    /**
     * Create OAuth authorization header
     */
    createAuthHeader(method, url, params = {}) {
        const oauthParams = this.generateOAuthSignature(method, url, params);
        
        const authHeader = 'OAuth ' + Object.keys(oauthParams)
            .map(key => `${encodeURIComponent(key)}="${encodeURIComponent(oauthParams[key])}"`)
            .join(', ');
        
        return authHeader;
    }
    
    /**
     * Make an authenticated request to Twitter API
     */
    async makeRequest(method, endpoint, data = {}) {
        const url = `https://api.twitter.com/2/${endpoint}`;
        
        return new Promise((resolve, reject) => {
            const parsedUrl = new URL(url);
            
            // Create authorization header
            const authHeader = this.createAuthHeader(method, url, method === 'GET' ? data : {});
            
            // Prepare request options
            const options = {
                hostname: parsedUrl.hostname,
                path: parsedUrl.pathname + (method === 'GET' && Object.keys(data).length > 0 ? '?' + querystring.stringify(data) : ''),
                method: method,
                headers: {
                    'Authorization': authHeader,
                    'Content-Type': 'application/json',
                    'User-Agent': 'DexterAI/1.0'
                }
            };
            
            let postData = '';
            if (method === 'POST' && Object.keys(data).length > 0) {
                postData = JSON.stringify(data);
                options.headers['Content-Length'] = Buffer.byteLength(postData);
            }
            
            const req = https.request(options, (res) => {
                let responseData = '';
                
                res.on('data', (chunk) => {
                    responseData += chunk;
                });
                
                res.on('end', () => {
                    // Update rate limit information
                    this.updateRateLimitInfo(res.headers);
                    
                    try {
                        const parsed = JSON.parse(responseData);
                        
                        if (res.statusCode >= 200 && res.statusCode < 300) {
                            resolve(parsed);
                        } else {
                            reject(new Error(`Twitter API Error (${res.statusCode}): ${parsed.detail || parsed.title || responseData}`));
                        }
                    } catch (e) {
                        reject(new Error(`Invalid JSON response: ${responseData}`));
                    }
                });
            });
            
            req.on('error', (error) => {
                reject(new Error(`Request failed: ${error.message}`));
            });
            
            if (postData) {
                req.write(postData);
            }
            
            req.end();
        });
    }
    
    /**
     * Update rate limit information from response headers
     */
    updateRateLimitInfo(headers) {
        if (headers['x-rate-limit-remaining']) {
            // For FREE TIER, cap at 17 tweets per 24 hours regardless of API response
            const apiRemaining = parseInt(headers['x-rate-limit-remaining']);
            this.remainingRequests = Math.min(apiRemaining, 17);
        }
        
        if (headers['x-rate-limit-reset']) {
            this.rateLimitReset = parseInt(headers['x-rate-limit-reset']) * 1000; // Convert to milliseconds
        }
    }
    
    /**
     * Check if we're being rate limited and handle accordingly (FREE TIER: 24-hour reset)
     */
    async handleRateLimit() {
        const now = Date.now();
        
        // Reset daily count if 24 hours have passed
        if (!this.dailyResetTime || now >= this.dailyResetTime) {
            this.dailyPostCount = 0;
            this.remainingRequests = 17;
            this.dailyResetTime = now + (24 * 60 * 60 * 1000); // 24 hours from now
            console.log('🔄 Daily rate limit reset - 17 tweets available for next 24 hours');
        }
        
        if (this.remainingRequests <= 0) {
            const hoursUntilReset = Math.ceil((this.dailyResetTime - now) / (60 * 60 * 1000));
            console.log(`⏳ FREE TIER LIMIT REACHED: 17 tweets per 24 hours. ${hoursUntilReset} hours until reset.`);
            throw new Error(`Daily tweet limit reached (17/24 hours). Try again in ${hoursUntilReset} hours.`);
        }
        
        console.log(`📊 Free tier status: ${this.remainingRequests} tweets remaining in next 24 hours`);
    }
    
    /**
     * Post a tweet to Twitter
     * @param {string} message - The tweet content (max 280 characters for free tier)
     * @param {Object} options - Additional tweet options
     * @returns {Promise<Object>} - Tweet response data
     */
    async postTweet(message, options = {}) {
        try {
            // Validate message length (280 characters for free tier)
            if (message.length > 280) {
                throw new Error(`Tweet too long: ${message.length} characters (max 280)`);
            }
            
            // Check rate limits
            await this.handleRateLimit();
            
            // Prepare tweet data
            const tweetData = {
                text: message
            };
            
            // Add optional parameters if provided
            if (options.reply_to) {
                tweetData.reply = { in_reply_to_tweet_id: options.reply_to };
            }
            
            if (options.media_ids && options.media_ids.length > 0) {
                tweetData.media = { media_ids: options.media_ids };
            }
            
            console.log('🐦 Posting tweet:', message.substring(0, 50) + (message.length > 50 ? '...' : ''));
            
            // Make the API request with retry logic
            const response = await this.makeRequestWithRetry('POST', 'tweets', tweetData);
            
            // Update daily count after successful post
            this.dailyPostCount++;
            this.remainingRequests--;
            
            console.log(`✅ Tweet posted successfully: ${response.data.id}`);
            console.log(`📊 Remaining tweets today: ${this.remainingRequests}/17`);
            return response;
            
        } catch (error) {
            console.error('❌ Failed to post tweet:', error.message);
            throw error;
        }
    }
    
    /**
     * Make request with retry logic for handling temporary failures
     */
    async makeRequestWithRetry(method, endpoint, data, retryCount = 0) {
        try {
            return await this.makeRequest(method, endpoint, data);
        } catch (error) {
            if (retryCount < this.maxRetries) {
                const isRetryable = this.isRetryableError(error);
                
                if (isRetryable) {
                    const delay = this.retryDelay * Math.pow(2, retryCount); // Exponential backoff
                    console.log(`⚠️  Request failed, retrying in ${delay/1000}s... (${retryCount + 1}/${this.maxRetries})`);
                    
                    await new Promise(resolve => setTimeout(resolve, delay));
                    return this.makeRequestWithRetry(method, endpoint, data, retryCount + 1);
                }
            }
            
            throw error;
        }
    }
    
    /**
     * Determine if an error is retryable
     */
    isRetryableError(error) {
        const retryableStatuses = [429, 500, 502, 503, 504]; // Rate limit, server errors
        const errorMessage = error.message.toLowerCase();
        
        // Check for rate limit or server errors
        return retryableStatuses.some(status => errorMessage.includes(status.toString())) ||
               errorMessage.includes('timeout') ||
               errorMessage.includes('network') ||
               errorMessage.includes('econnreset');
    }
    
    /**
     * Get user information (for testing authentication)
     */
    async getUserInfo() {
        try {
            const response = await this.makeRequest('GET', 'users/me');
            return response.data;
        } catch (error) {
            console.error('❌ Failed to get user info:', error.message);
            throw error;
        }
    }
    
    /**
     * Search for tweets (for monitoring mentions, etc.)
     */
    async searchTweets(query, options = {}) {
        try {
            const params = {
                query: query,
                max_results: options.max_results || 10
            };
            
            if (options.since_id) {
                params.since_id = options.since_id;
            }
            
            const response = await this.makeRequest('GET', 'tweets/search/recent', params);
            return response.data || [];
        } catch (error) {
            console.error('❌ Failed to search tweets:', error.message);
            throw error;
        }
    }
    
    /**
     * Test the Twitter service connection
     */
    async testConnection() {
        try {
            console.log('🔍 Testing Twitter API connection...');
            const userInfo = await this.getUserInfo();
            console.log(`✅ Connected as @${userInfo.username} (${userInfo.name})`);
            const status = this.getRateLimitStatus();
            console.log(`📊 FREE TIER Status: ${status.remaining}/${status.dailyLimit} tweets remaining (${status.hoursUntilReset}h until reset)`);
            return true;
        } catch (error) {
            console.error('❌ Twitter connection test failed:', error.message);
            return false;
        }
    }
    
    /**
     * Get current rate limit status (FREE TIER: 24-hour cycle)
     */
    getRateLimitStatus() {
        const now = Date.now();
        const hoursUntilReset = this.dailyResetTime ? Math.ceil((this.dailyResetTime - now) / (60 * 60 * 1000)) : 24;
        
        return {
            remaining: this.remainingRequests,
            dailyLimit: 17,
            used: this.dailyPostCount,
            resetTime: this.dailyResetTime ? new Date(this.dailyResetTime) : null,
            hoursUntilReset: Math.max(hoursUntilReset, 0),
            tier: 'FREE (17 tweets/24 hours)'
        };
    }
}

// Export the TwitterService class
module.exports = TwitterService;

// Example usage (if run directly)
if (require.main === module) {
    async function testTwitterService() {
        try {
            // Load environment variables from .env file if available
            try {
                require('dotenv').config();
            } catch (e) {
                console.log('Note: dotenv not available, using system environment variables');
            }
            
            const twitter = new TwitterService();
            
            // Test connection
            const connected = await twitter.testConnection();
            if (!connected) {
                process.exit(1);
            }
            
            // Example tweet
            const testMessage = `🤖 Dexter AI is online and monitoring the markets! Current time: ${new Date().toISOString()}`;
            
            console.log('\n📝 Test tweet:', testMessage);
            console.log('⚠️  This is a test - uncomment the line below to actually post');
            
            // Uncomment the line below to actually post a test tweet
            // const result = await twitter.postTweet(testMessage);
            // console.log('Tweet posted:', result.data.id);
            
        } catch (error) {
            console.error('❌ Test failed:', error.message);
            process.exit(1);
        }
    }
    
    testTwitterService();
}