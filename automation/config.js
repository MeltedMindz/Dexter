require('dotenv').config();

/**
 * Configuration for Ultra-Frequent Keeper Service
 * Optimized for Base chain's low gas environment
 */

const config = {
    // Network Configuration
    network: {
        name: process.env.NETWORK || 'base-sepolia',
        rpcUrl: process.env.BASE_RPC_URL || 'https://sepolia.base.org',
        chainId: process.env.CHAIN_ID || 84532,
        gasLimit: {
            compound: 150000,      // ~150k gas per compound
            rebalance: 300000,     // ~300k gas per rebalance
            batchCompound: 2000000 // ~2M gas for batch (100 positions)
        }
    },
    
    // Wallet Configuration
    wallet: {
        privateKey: process.env.KEEPER_PRIVATE_KEY,
        minBalance: process.env.KEEPER_WALLET_MIN_BALANCE || '0.01', // ETH
        gasMultiplier: parseFloat(process.env.GAS_MULTIPLIER || '1.1'),
        maxGasPriceGwei: parseFloat(process.env.MAX_GAS_PRICE_GWEI || '10')
    },
    
    // Contract Addresses
    contracts: {
        dexterMVP: process.env.DEXTER_MVP_ADDRESS,
        compounder: process.env.COMPOUNDER_ADDRESS,
        rebalancer: process.env.REBALANCER_ADDRESS,
        positionManager: process.env.UNISWAP_NPM_ADDRESS || '0x03a520b32C04BF3bEEf7BF5d2E3eF30500000000'
    },
    
    // Automation Settings
    automation: {
        // Ultra-frequent compounding (5 minutes)
        compoundInterval: parseInt(process.env.COMPOUND_INTERVAL_SECONDS || '300'),
        compoundIntervalMs: parseInt(process.env.COMPOUND_INTERVAL_SECONDS || '300') * 1000,
        
        // Bin-based rebalancing (30 minutes)
        rebalanceInterval: parseInt(process.env.REBALANCE_INTERVAL_SECONDS || '1800'),
        rebalanceIntervalMs: parseInt(process.env.REBALANCE_INTERVAL_SECONDS || '1800') * 1000,
        
        // Batch processing
        maxBatchSize: parseInt(process.env.MAX_BATCH_SIZE || '100'),
        batchDelay: 1000, // 1 second between batches
        
        // Thresholds
        minCompoundThresholdUSD: parseFloat(process.env.MIN_COMPOUND_THRESHOLD_USD || '0.5'),
        maxBinsFromPrice: parseInt(process.env.MAX_BINS_FROM_PRICE || '5'),
        
        // Timing
        maxOperationTime: 60000, // 60 seconds max per operation
        retryAttempts: 3,
        retryDelay: 5000 // 5 seconds between retries
    },
    
    // Monitoring & Logging
    monitoring: {
        enableMetrics: process.env.ENABLE_METRICS === 'true',
        metricsInterval: parseInt(process.env.METRICS_INTERVAL_MINUTES || '60') * 60 * 1000,
        logLevel: process.env.LOG_LEVEL || 'info',
        
        // Health check endpoints
        healthCheckPort: process.env.HEALTH_CHECK_PORT || 3001,
        
        // Performance tracking
        trackGasUsage: true,
        trackOperationTiming: true,
        trackSuccess: true
    },
    
    // Cron Schedules (for node-cron)
    cron: {
        // Every 5 minutes for compounds
        compoundCheck: '*/5 * * * *',
        
        // Every 30 minutes for rebalances  
        rebalanceCheck: '*/30 * * * *',
        
        // Every hour for metrics reporting
        metricsReport: '0 * * * *',
        
        // Every 6 hours for health check
        healthCheck: '0 */6 * * *',
        
        // Daily cleanup at 2 AM
        dailyCleanup: '0 2 * * *'
    },
    
    // Base Chain Specific Optimizations
    baseOptimizations: {
        // Low gas prices enable aggressive automation
        enableUltraFrequent: true,
        
        // Base has consistent block times
        expectedBlockTime: 2, // 2 seconds
        
        // MEV protection (less needed on Base but still good practice)
        enableMevProtection: true,
        maxSlippage: 0.005, // 0.5%
        
        // Base-specific RPC optimizations
        useMulticall: true,
        batchRequests: true,
        parallelOperations: true
    },
    
    // Error Handling
    errorHandling: {
        // Retry failed operations
        enableRetry: true,
        maxRetries: 3,
        retryBackoff: 'exponential', // linear, exponential
        
        // Circuit breaker for repeated failures
        enableCircuitBreaker: true,
        failureThreshold: 5,
        resetTimeout: 300000, // 5 minutes
        
        // Emergency pause conditions
        emergencyPause: {
            highGasPrice: 50, // Gwei
            lowKeeperBalance: 0.005, // ETH
            highFailureRate: 0.5 // 50% failure rate
        }
    },
    
    // Development/Testing
    development: {
        isDevelopment: process.env.NODE_ENV === 'development',
        enableMockData: process.env.ENABLE_MOCK_DATA === 'true',
        verboseLogging: process.env.VERBOSE_LOGGING === 'true',
        
        // Test mode configurations
        testMode: {
            enabled: process.env.TEST_MODE === 'true',
            mockGasPrice: '0.1', // Gwei
            mockPositions: true,
            skipActualTransactions: true
        }
    },
    
    // Performance Targets (for monitoring)
    targets: {
        // Compound frequency (Base chain enables this)
        avgCompoundInterval: 300, // 5 minutes
        maxCompoundInterval: 900, // 15 minutes max
        
        // Success rates
        compoundSuccessRate: 0.95, // 95%
        rebalanceSuccessRate: 0.90, // 90%
        
        // Gas efficiency
        avgGasPerCompound: 150000,
        avgGasPerRebalance: 300000,
        
        // Timing
        avgOperationTime: 10000, // 10 seconds
        maxOperationTime: 60000   // 60 seconds
    }
};

// Validation
function validateConfig() {
    const required = [
        'wallet.privateKey',
        'contracts.dexterMVP',
        'contracts.compounder', 
        'contracts.rebalancer'
    ];
    
    for (const path of required) {
        const value = path.split('.').reduce((obj, key) => obj?.[key], config);
        if (!value) {
            throw new Error(`Missing required configuration: ${path}`);
        }
    }
    
    // Validate intervals
    if (config.automation.compoundInterval < 60) {
        console.warn('⚠️  Compound interval less than 1 minute may be too aggressive');
    }
    
    if (config.automation.maxBatchSize > 200) {
        console.warn('⚠️  Batch size over 200 may hit gas limits');
    }
    
    console.log('✅ Configuration validated successfully');
}

// Environment-specific overrides
if (process.env.NODE_ENV === 'production') {
    config.monitoring.logLevel = 'info';
    config.development.verboseLogging = false;
    config.errorHandling.enableCircuitBreaker = true;
}

if (process.env.NODE_ENV === 'development') {
    config.monitoring.logLevel = 'debug';
    config.development.verboseLogging = true;
    config.automation.retryAttempts = 1; // Faster testing
}

// Export configuration
module.exports = {
    config,
    validateConfig,
    
    // Helper functions
    isMainnet: () => config.network.chainId === 8453,
    isTestnet: () => config.network.chainId === 84532,
    isDevelopment: () => config.development.isDevelopment,
    
    // Get formatted addresses for logging
    getContractSummary: () => ({
        DexterMVP: config.contracts.dexterMVP,
        Compounder: config.contracts.compounder,
        Rebalancer: config.contracts.rebalancer,
        Network: config.network.name
    })
};