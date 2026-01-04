#!/usr/bin/env node
/**
 * Ultra-Frequent Keeper Service
 * Optimized for Base chain's low gas costs
 * Executes compounds every 5 minutes and rebalances every 30 minutes
 */

const { ethers } = require('ethers');
const cron = require('node-cron');

class UltraFrequentKeeper {
    constructor(config) {
        this.config = config;
        this.provider = new ethers.providers.JsonRpcProvider(config.rpcUrl);
        this.wallet = new ethers.Wallet(config.privateKey, this.provider);
        
        // Contract instances
        this.dexterMVP = new ethers.Contract(
            config.contracts.dexterMVP,
            require('./abi/DexterMVP.json'),
            this.wallet
        );
        
        this.compounder = new ethers.Contract(
            config.contracts.compounder,
            require('./abi/UltraFrequentCompounder.json'),
            this.wallet
        );
        
        this.rebalancer = new ethers.Contract(
            config.contracts.rebalancer,
            require('./abi/BinRebalancer.json'),
            this.wallet
        );
        
        // Timing intervals
        this.compoundInterval = 5 * 60 * 1000; // 5 minutes
        this.rebalanceInterval = 30 * 60 * 1000; // 30 minutes
        this.batchSize = 100; // Max positions per batch on Base
        
        // Performance tracking
        this.metrics = {
            totalCompounds: 0,
            totalRebalances: 0,
            totalGasUsed: 0,
            errors: 0,
            lastUpdate: Date.now()
        };
        
        console.log('🚀 Ultra-Frequent Keeper initialized for Base chain');
        console.log(`📊 Compound interval: ${this.compoundInterval / 1000}s`);
        console.log(`📊 Rebalance interval: ${this.rebalanceInterval / 1000}s`);
    }
    
    /**
     * Start the ultra-frequent automation service
     */
    async start() {
        console.log('🎯 Starting ultra-frequent automation service...');
        
        // Initial health check
        await this.healthCheck();
        
        // Schedule compound checks every 5 minutes
        cron.schedule('*/5 * * * *', async () => {
            await this.executeCompoundRound();
        });
        
        // Schedule rebalance checks every 30 minutes
        cron.schedule('*/30 * * * *', async () => {
            await this.executeRebalanceRound();
        });
        
        // Schedule metrics reporting every hour
        cron.schedule('0 * * * *', () => {
            this.reportMetrics();
        });
        
        console.log('✅ Ultra-frequent keeper service started successfully');
    }
    
    /**
     * Execute compound round - check and compound all eligible positions
     */
    async executeCompoundRound() {
        const startTime = Date.now();
        
        try {
            console.log('🔄 Starting compound round...');
            
            // Get all active positions
            const activePositions = await this.getAllActivePositions();
            console.log(`📋 Found ${activePositions.length} active positions`);
            
            if (activePositions.length === 0) {
                console.log('⏭️  No positions to check for compounding');
                return;
            }
            
            // Check which positions need compounding
            const positionsToCompound = [];
            
            for (const tokenId of activePositions) {
                try {
                    const needsCompound = await this.dexterMVP.shouldCompound(tokenId);
                    if (needsCompound) {
                        positionsToCompound.push(tokenId);
                    }
                } catch (error) {
                    console.warn(`⚠️  Error checking position ${tokenId}:`, error.message);
                }
            }
            
            console.log(`💰 Found ${positionsToCompound.length} positions ready for compounding`);
            
            if (positionsToCompound.length === 0) {
                console.log('⏭️  No positions need compounding at this time');
                return;
            }
            
            // Execute batch compounds
            await this.executeBatchCompounds(positionsToCompound);
            
            const duration = Date.now() - startTime;
            console.log(`✅ Compound round completed in ${duration}ms`);
            
        } catch (error) {
            console.error('❌ Error in compound round:', error);
            this.metrics.errors++;
        }
    }
    
    /**
     * Execute rebalance round - check and rebalance positions based on bin drift
     */
    async executeRebalanceRound() {
        const startTime = Date.now();
        
        try {
            console.log('⚖️  Starting rebalance round...');
            
            // Get all active positions
            const activePositions = await this.getAllActivePositions();
            console.log(`📋 Checking ${activePositions.length} positions for rebalancing`);
            
            const positionsToRebalance = [];
            
            for (const tokenId of activePositions) {
                try {
                    const needsRebalance = await this.dexterMVP.shouldRebalance(tokenId);
                    if (needsRebalance) {
                        // Get bin position data for logging
                        const binData = await this.dexterMVP.calculateBinPosition(tokenId);
                        console.log(`🎯 Position ${tokenId} needs rebalance: ${binData.binsFromPrice} bins from price`);
                        positionsToRebalance.push(tokenId);
                    }
                } catch (error) {
                    console.warn(`⚠️  Error checking rebalance for position ${tokenId}:`, error.message);
                }
            }
            
            console.log(`⚖️  Found ${positionsToRebalance.length} positions ready for rebalancing`);
            
            // Execute rebalances individually (more complex than compounds)
            for (const tokenId of positionsToRebalance) {
                try {
                    await this.executeRebalance(tokenId);
                } catch (error) {
                    console.error(`❌ Failed to rebalance position ${tokenId}:`, error.message);
                }
            }
            
            const duration = Date.now() - startTime;
            console.log(`✅ Rebalance round completed in ${duration}ms`);
            
        } catch (error) {
            console.error('❌ Error in rebalance round:', error);
            this.metrics.errors++;
        }
    }
    
    /**
     * Execute batch compounds with Base chain optimization
     */
    async executeBatchCompounds(positionIds) {
        console.log(`🔥 Executing batch compound for ${positionIds.length} positions`);
        
        // Split into batches to avoid gas limit issues
        const batches = this.chunkArray(positionIds, this.batchSize);
        
        for (let i = 0; i < batches.length; i++) {
            const batch = batches[i];
            console.log(`📦 Processing batch ${i + 1}/${batches.length} (${batch.length} positions)`);
            
            try {
                // Estimate gas for batch
                const gasEstimate = await this.compounder.estimateGas.batchCompound(batch);
                
                // Execute batch compound with Base-optimized gas settings
                const tx = await this.compounder.batchCompound(batch, {
                    gasLimit: Math.floor(gasEstimate.toNumber() * 1.2), // 20% buffer
                    gasPrice: await this.getOptimalGasPrice()
                });
                
                console.log(`📋 Batch compound transaction: ${tx.hash}`);
                
                // Wait for confirmation
                const receipt = await tx.wait();
                
                console.log(`✅ Batch compound confirmed! Gas used: ${receipt.gasUsed.toString()}`);
                
                // Update metrics
                this.metrics.totalCompounds += batch.length;
                this.metrics.totalGasUsed += receipt.gasUsed.toNumber();
                
                // Parse events for detailed logging
                this.parseCompoundEvents(receipt);
                
            } catch (error) {
                console.error(`❌ Batch compound failed for batch ${i + 1}:`, error.message);
                this.metrics.errors++;
            }
        }
    }
    
    /**
     * Execute individual rebalance
     */
    async executeRebalance(tokenId) {
        console.log(`⚖️  Executing rebalance for position ${tokenId}`);
        
        try {
            // Get current bin position
            const binData = await this.dexterMVP.calculateBinPosition(tokenId);
            console.log(`📊 Position ${tokenId}: ${binData.binsFromPrice} bins from price, inRange: ${binData.inRange}`);
            
            // Estimate gas
            const gasEstimate = await this.rebalancer.estimateGas.executeRebalance(tokenId);
            
            // Execute rebalance
            const tx = await this.rebalancer.executeRebalance(tokenId, {
                gasLimit: Math.floor(gasEstimate.toNumber() * 1.2),
                gasPrice: await this.getOptimalGasPrice()
            });
            
            console.log(`📋 Rebalance transaction: ${tx.hash}`);
            
            // Wait for confirmation
            const receipt = await tx.wait();
            
            console.log(`✅ Rebalance confirmed! Gas used: ${receipt.gasUsed.toString()}`);
            
            // Update metrics
            this.metrics.totalRebalances++;
            this.metrics.totalGasUsed += receipt.gasUsed.toNumber();
            
            // Parse rebalance events
            this.parseRebalanceEvents(receipt);
            
        } catch (error) {
            console.error(`❌ Rebalance failed for position ${tokenId}:`, error.message);
            throw error;
        }
    }
    
    /**
     * Get all active positions across all users
     */
    async getAllActivePositions() {
        try {
            // This would need to be implemented based on your indexing strategy
            // For MVP, we can query events or maintain a position registry
            
            // Placeholder implementation - in production you'd have an indexer
            const filter = this.dexterMVP.filters.PositionDeposited();
            const events = await this.dexterMVP.queryFilter(filter, -1000); // Last 1000 blocks
            
            const activePositions = new Set();
            
            // Add deposited positions
            for (const event of events) {
                activePositions.add(event.args.tokenId.toString());
            }
            
            // Remove withdrawn positions
            const withdrawFilter = this.dexterMVP.filters.PositionWithdrawn();
            const withdrawEvents = await this.dexterMVP.queryFilter(withdrawFilter, -1000);
            
            for (const event of withdrawEvents) {
                activePositions.delete(event.args.tokenId.toString());
            }
            
            return Array.from(activePositions);
            
        } catch (error) {
            console.error('❌ Error getting active positions:', error);
            return [];
        }
    }
    
    /**
     * Get optimal gas price for Base chain
     */
    async getOptimalGasPrice() {
        try {
            const gasPrice = await this.provider.getGasPrice();
            // Base chain typically has very low gas prices, so we can be aggressive
            return gasPrice.mul(110).div(100); // 10% above current
        } catch (error) {
            console.warn('⚠️  Failed to get gas price, using default');
            return ethers.utils.parseUnits('0.1', 'gwei'); // Very low default for Base
        }
    }
    
    /**
     * Parse compound events for detailed logging
     */
    parseCompoundEvents(receipt) {
        const compoundEvents = receipt.events?.filter(e => e.event === 'UltraFrequentCompound') || [];
        
        for (const event of compoundEvents) {
            const { tokenId, feesUSD, compoundNumber } = event.args;
            console.log(`💰 Position ${tokenId} compounded: $${ethers.utils.formatEther(feesUSD)} (compound #${compoundNumber})`);
        }
    }
    
    /**
     * Parse rebalance events for detailed logging
     */
    parseRebalanceEvents(receipt) {
        const rebalanceEvents = receipt.events?.filter(e => e.event === 'BinBasedRebalance') || [];
        
        for (const event of rebalanceEvents) {
            const { oldTokenId, newTokenId, binsFromPrice } = event.args;
            console.log(`⚖️  Position ${oldTokenId} → ${newTokenId} rebalanced (was ${binsFromPrice} bins from price)`);
        }
    }
    
    /**
     * Health check for the keeper service
     */
    async healthCheck() {
        try {
            console.log('🔍 Performing health check...');
            
            // Check network connection
            const blockNumber = await this.provider.getBlockNumber();
            console.log(`📊 Connected to Base, current block: ${blockNumber}`);
            
            // Check wallet balance
            const balance = await this.wallet.getBalance();
            const balanceETH = ethers.utils.formatEther(balance);
            console.log(`💰 Keeper wallet balance: ${balanceETH} ETH`);
            
            if (balance.lt(ethers.utils.parseEther('0.01'))) {
                console.warn('⚠️  Low keeper balance! Please fund the keeper wallet.');
            }
            
            // Check contract connections
            const isAuthorized = await this.dexterMVP.authorizedKeepers(this.wallet.address);
            console.log(`🔐 Keeper authorization: ${isAuthorized ? 'AUTHORIZED' : 'NOT AUTHORIZED'}`);
            
            if (!isAuthorized) {
                console.error('❌ Keeper is not authorized! Please authorize the keeper address.');
            }
            
            console.log('✅ Health check completed');
            
        } catch (error) {
            console.error('❌ Health check failed:', error);
            throw error;
        }
    }
    
    /**
     * Report performance metrics
     */
    reportMetrics() {
        const now = Date.now();
        const duration = (now - this.metrics.lastUpdate) / 1000 / 60; // minutes
        
        console.log('📊 ===== KEEPER PERFORMANCE METRICS =====');
        console.log(`⏱️  Time period: ${duration.toFixed(1)} minutes`);
        console.log(`💰 Total compounds: ${this.metrics.totalCompounds}`);
        console.log(`⚖️  Total rebalances: ${this.metrics.totalRebalances}`);
        console.log(`⛽ Total gas used: ${this.metrics.totalGasUsed.toLocaleString()}`);
        console.log(`❌ Errors: ${this.metrics.errors}`);
        
        if (this.metrics.totalCompounds > 0) {
            const avgGasPerCompound = this.metrics.totalGasUsed / this.metrics.totalCompounds;
            console.log(`📈 Avg gas per compound: ${avgGasPerCompound.toFixed(0)}`);
        }
        
        console.log('========================================');
        
        this.metrics.lastUpdate = now;
    }
    
    /**
     * Utility function to chunk array into smaller batches
     */
    chunkArray(array, chunkSize) {
        const chunks = [];
        for (let i = 0; i < array.length; i += chunkSize) {
            chunks.push(array.slice(i, i + chunkSize));
        }
        return chunks;
    }
}

// Configuration for Base chain
const config = {
    rpcUrl: process.env.BASE_RPC_URL || 'https://mainnet.base.org',
    privateKey: process.env.KEEPER_PRIVATE_KEY,
    contracts: {
        dexterMVP: process.env.DEXTER_MVP_ADDRESS,
        compounder: process.env.COMPOUNDER_ADDRESS,
        rebalancer: process.env.REBALANCER_ADDRESS
    }
};

// Start the keeper service
if (require.main === module) {
    const keeper = new UltraFrequentKeeper(config);
    
    // Handle graceful shutdown
    process.on('SIGINT', () => {
        console.log('🛑 Shutting down ultra-frequent keeper...');
        keeper.reportMetrics();
        process.exit(0);
    });
    
    // Start the service
    keeper.start().catch(error => {
        console.error('❌ Failed to start keeper service:', error);
        process.exit(1);
    });
}

module.exports = UltraFrequentKeeper;