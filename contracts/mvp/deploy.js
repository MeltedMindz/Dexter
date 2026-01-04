#!/usr/bin/env node
/**
 * Deployment script for Dexter MVP contracts on Base chain
 * Optimized for ultra-frequent automation with low gas costs
 */

const { ethers } = require('ethers');
require('dotenv').config();

class MVPDeployer {
    constructor() {
        this.provider = new ethers.providers.JsonRpcProvider(
            process.env.BASE_RPC_URL || 'https://sepolia.base.org'
        );
        this.wallet = new ethers.Wallet(process.env.DEPLOYER_PRIVATE_KEY, this.provider);
        this.deployedContracts = {};
        
        // Base chain contract addresses
        this.baseAddresses = {
            positionManager: '0x03a520b32C04BF3bEEf7BF5d2E3eF30500000000', // Base NPM
            factory: '0x33128a8fC17869897dcE68Ed026d694621f6FD00',        // Base Factory
            weth: '0x4200000000000000000000000000000000000006'             // Base WETH
        };
        
        console.log('🚀 MVP Deployer initialized for Base chain');
        console.log(`📊 Deployer address: ${this.wallet.address}`);
    }
    
    /**
     * Deploy all MVP contracts in sequence
     */
    async deployAll() {
        try {
            console.log('🎯 Starting MVP contract deployment...\n');
            
            // Check deployer balance
            await this.checkDeployerBalance();
            
            // 1. Deploy BinRebalancer
            await this.deployBinRebalancer();
            
            // 2. Deploy UltraFrequentCompounder
            await this.deployUltraFrequentCompounder();
            
            // 3. Deploy DexterMVP (main contract)
            await this.deployDexterMVP();
            
            // 4. Configure contracts
            await this.configureContracts();
            
            // 5. Verify deployments
            await this.verifyDeployments();
            
            // 6. Generate deployment summary
            this.generateDeploymentSummary();
            
            console.log('✅ MVP deployment completed successfully!');
            
        } catch (error) {
            console.error('❌ Deployment failed:', error);
            throw error;
        }
    }
    
    /**
     * Check deployer wallet balance
     */
    async checkDeployerBalance() {
        const balance = await this.wallet.getBalance();
        const balanceETH = ethers.utils.formatEther(balance);
        
        console.log(`💰 Deployer balance: ${balanceETH} ETH`);
        
        if (balance.lt(ethers.utils.parseEther('0.1'))) {
            throw new Error('⚠️  Insufficient balance for deployment. Need at least 0.1 ETH');
        }
    }
    
    /**
     * Deploy BinRebalancer contract
     */
    async deployBinRebalancer() {
        console.log('📦 Deploying BinRebalancer...');
        
        const BinRebalancer = await ethers.getContractFactory('BinRebalancer', this.wallet);
        
        const gasEstimate = await BinRebalancer.estimateGas.deploy(
            this.baseAddresses.positionManager,
            this.baseAddresses.factory
        );
        
        const binRebalancer = await BinRebalancer.deploy(
            this.baseAddresses.positionManager,
            this.baseAddresses.factory,
            {
                gasLimit: Math.floor(gasEstimate.toNumber() * 1.2),
                gasPrice: await this.getOptimalGasPrice()
            }
        );
        
        await binRebalancer.deployed();
        
        this.deployedContracts.binRebalancer = binRebalancer.address;
        console.log(`✅ BinRebalancer deployed: ${binRebalancer.address}`);
        console.log(`📊 Gas used: ${(await binRebalancer.deployTransaction.wait()).gasUsed.toString()}\n`);
    }
    
    /**
     * Deploy UltraFrequentCompounder contract
     */
    async deployUltraFrequentCompounder() {
        console.log('📦 Deploying UltraFrequentCompounder...');
        
        const UltraFrequentCompounder = await ethers.getContractFactory('UltraFrequentCompounder', this.wallet);
        
        const compounder = await UltraFrequentCompounder.deploy(
            this.baseAddresses.positionManager,
            {
                gasPrice: await this.getOptimalGasPrice()
            }
        );
        
        await compounder.deployed();
        
        this.deployedContracts.compounder = compounder.address;
        console.log(`✅ UltraFrequentCompounder deployed: ${compounder.address}`);
        console.log(`📊 Gas used: ${(await compounder.deployTransaction.wait()).gasUsed.toString()}\n`);
    }
    
    /**
     * Deploy DexterMVP main contract
     */
    async deployDexterMVP() {
        console.log('📦 Deploying DexterMVP...');
        
        const DexterMVP = await ethers.getContractFactory('DexterMVP', this.wallet);
        
        const dexterMVP = await DexterMVP.deploy(
            this.baseAddresses.positionManager,
            this.deployedContracts.compounder,
            this.deployedContracts.binRebalancer,
            this.baseAddresses.weth,
            {
                gasPrice: await this.getOptimalGasPrice()
            }
        );
        
        await dexterMVP.deployed();
        
        this.deployedContracts.dexterMVP = dexterMVP.address;
        console.log(`✅ DexterMVP deployed: ${dexterMVP.address}`);
        console.log(`📊 Gas used: ${(await dexterMVP.deployTransaction.wait()).gasUsed.toString()}\n`);
    }
    
    /**
     * Configure deployed contracts
     */
    async configureContracts() {
        console.log('⚙️  Configuring contracts...');
        
        const dexterMVP = new ethers.Contract(
            this.deployedContracts.dexterMVP,
            require('./abi/DexterMVP.json'),
            this.wallet
        );
        
        const compounder = new ethers.Contract(
            this.deployedContracts.compounder,
            require('./abi/UltraFrequentCompounder.json'),
            this.wallet
        );
        
        const rebalancer = new ethers.Contract(
            this.deployedContracts.binRebalancer,
            require('./abi/BinRebalancer.json'),
            this.wallet
        );
        
        // 1. Set DexterMVP as authorized caller for compounder
        console.log('🔐 Authorizing DexterMVP in UltraFrequentCompounder...');
        await compounder.setAuthorizedVault(this.deployedContracts.dexterMVP, true);
        
        // 2. Set DexterMVP as authorized caller for rebalancer
        console.log('🔐 Authorizing DexterMVP in BinRebalancer...');
        await rebalancer.setKeeperAuthorization(this.deployedContracts.dexterMVP, true);
        
        // 3. Configure ultra-frequent defaults in compounder
        console.log('⚡ Setting ultra-frequent defaults...');
        await compounder.setUltraFrequentDefaults();
        
        console.log('✅ Contract configuration completed\n');
    }
    
    /**
     * Verify all deployments
     */
    async verifyDeployments() {
        console.log('🔍 Verifying deployments...');
        
        for (const [name, address] of Object.entries(this.deployedContracts)) {
            const code = await this.provider.getCode(address);
            if (code === '0x') {
                throw new Error(`❌ ${name} deployment failed - no code at address`);
            }
            console.log(`✅ ${name} verified at ${address}`);
        }
        
        console.log('✅ All deployments verified\n');
    }
    
    /**
     * Generate deployment summary
     */
    generateDeploymentSummary() {
        const summary = {
            network: 'Base Sepolia Testnet',
            timestamp: new Date().toISOString(),
            deployer: this.wallet.address,
            contracts: this.deployedContracts,
            configuration: {
                compoundInterval: '5 minutes',
                rebalanceInterval: '30 minutes',
                minimumThreshold: '$0.50',
                maxBatchSize: 100
            }
        };
        
        console.log('📄 ===== DEPLOYMENT SUMMARY =====');
        console.log(`🌐 Network: ${summary.network}`);
        console.log(`⏰ Deployed at: ${summary.timestamp}`);
        console.log(`👤 Deployer: ${summary.deployer}`);
        console.log('\n📋 Contract Addresses:');
        
        for (const [name, address] of Object.entries(this.deployedContracts)) {
            console.log(`   ${name}: ${address}`);
        }
        
        console.log('\n⚙️  Configuration:');
        console.log(`   Compound Interval: ${summary.configuration.compoundInterval}`);
        console.log(`   Rebalance Interval: ${summary.configuration.rebalanceInterval}`);
        console.log(`   Minimum Threshold: ${summary.configuration.minimumThreshold}`);
        console.log(`   Max Batch Size: ${summary.configuration.maxBatchSize}`);
        
        console.log('\n🚀 Next Steps:');
        console.log('   1. Fund keeper wallet for automation');
        console.log('   2. Start keeper service with these addresses');
        console.log('   3. Test with small position first');
        console.log('   4. Monitor automation metrics');
        console.log('================================\n');
        
        // Save to file for CI/CD
        const fs = require('fs');
        fs.writeFileSync('./deployment-summary.json', JSON.stringify(summary, null, 2));
        console.log('📁 Deployment summary saved to deployment-summary.json');
    }
    
    /**
     * Get optimal gas price for Base
     */
    async getOptimalGasPrice() {
        try {
            const gasPrice = await this.provider.getGasPrice();
            return gasPrice.mul(110).div(100); // 10% above current
        } catch (error) {
            console.warn('⚠️  Failed to get gas price, using default');
            return ethers.utils.parseUnits('0.1', 'gwei');
        }
    }
}

// Deploy testnet contracts
async function deployTestnet() {
    const deployer = new MVPDeployer();
    await deployer.deployAll();
}

// Deploy mainnet contracts (with additional safety checks)
async function deployMainnet() {
    const deployer = new MVPDeployer();
    
    // Additional mainnet safety checks
    console.log('⚠️  MAINNET DEPLOYMENT - Performing additional safety checks...');
    
    const balance = await deployer.wallet.getBalance();
    if (balance.lt(ethers.utils.parseEther('0.5'))) {
        throw new Error('Insufficient balance for mainnet deployment');
    }
    
    // Confirm mainnet deployment
    console.log('🚨 This will deploy to MAINNET. Contracts will be immutable.');
    console.log('⏰ You have 10 seconds to cancel...');
    
    await new Promise(resolve => setTimeout(resolve, 10000));
    
    await deployer.deployAll();
}

// CLI interface
if (require.main === module) {
    const args = process.argv.slice(2);
    const network = args[0] || 'testnet';
    
    if (network === 'mainnet') {
        deployMainnet().catch(console.error);
    } else {
        deployTestnet().catch(console.error);
    }
}

module.exports = { MVPDeployer };