# Dexter MVP - Deployment Guide

## 🚀 Quick Start

Deploy the complete MVP system with one command:

```bash
# Deploy to testnet
./scripts/mvp-deploy-and-test.sh testnet

# Deploy to mainnet (requires confirmation)
./scripts/mvp-deploy-and-test.sh mainnet
```

## 📋 Prerequisites

### 1. Environment Setup
```bash
# Copy environment template
cp contracts/mvp/.env.example contracts/mvp/.env

# Edit with your configuration
nano contracts/mvp/.env
```

### 2. Required Environment Variables
```bash
# Blockchain access
BASE_RPC_URL=https://mainnet.base.org
DEPLOYER_PRIVATE_KEY=your_deployer_private_key
KEEPER_PRIVATE_KEY=your_keeper_private_key

# Contract verification
BASESCAN_API_KEY=your_basescan_api_key
```

### 3. Minimum Requirements
- **Node.js 16+**
- **0.1 ETH** for testnet deployment
- **0.5 ETH** for mainnet deployment
- **Base chain RPC access**

## 🔧 Manual Deployment Steps

### Step 1: Install Dependencies
```bash
cd contracts/mvp
npm install

cd ../../automation
npm install
```

### Step 2: Compile Contracts
```bash
cd contracts/mvp
npm run compile
```

### Step 3: Run Tests
```bash
# Unit tests
npm run test

# Gas optimization tests
npm run test:gas

# Coverage report
npm run coverage
```

### Step 4: Deploy Contracts
```bash
# Testnet
npm run deploy:testnet

# Mainnet
npm run deploy:mainnet
```

### Step 5: Configure Automation
```bash
cd ../../automation

# Update .env with deployed addresses
# (automatically done by deployment script)

# Test configuration
npm run health
```

### Step 6: Start Keeper Service
```bash
cd automation
npm start
```

## 📊 Deployment Verification

### Contract Verification
```bash
# Verify on BaseScan
cd contracts/mvp
npx hardhat verify --network base-mainnet CONTRACT_ADDRESS "CONSTRUCTOR_ARGS"
```

### Health Check
```bash
cd automation
npm run health
```

Expected output:
```
🔍 Performing health check...
📊 Connected to Base, current block: 12345678
💰 Keeper wallet balance: 0.1 ETH
🔐 Keeper authorization: AUTHORIZED
✅ Health check completed
```

## ⚙️ Configuration Options

### Ultra-Frequent Settings
```javascript
// In automation/config.js
automation: {
    compoundInterval: 300,        // 5 minutes
    rebalanceInterval: 1800,      // 30 minutes
    minCompoundThresholdUSD: 0.5, // $0.50 minimum
    maxBatchSize: 100             // Batch processing
}
```

### Concentration Levels
```solidity
// In contracts
enum ConcentrationLevel {
    ULTRA_TIGHT,    // 1-2 bins (0.05-0.1% range)
    VERY_TIGHT,     // 2-3 bins (0.1-0.15% range)
    TIGHT,          // 3-4 bins (0.15-0.2% range)
    MODERATE,       // 4-6 bins (0.2-0.3% range)
    WIDE            // 6-10 bins (0.3-0.5% range)
}
```

## 🎯 Post-Deployment Setup

### 1. Fund Keeper Wallet
```bash
# Send ETH to keeper address for gas
# Minimum: 0.01 ETH
# Recommended: 0.1 ETH for continuous operation
```

### 2. Test with Small Position
```javascript
// Example test position
const automationSettings = {
    autoCompoundEnabled: true,
    autoRebalanceEnabled: true,
    compoundThresholdUSD: ethers.utils.parseEther('2'),
    maxBinsFromPrice: 3,
    concentrationLevel: 5,
    lastCompoundTime: 0,
    lastRebalanceTime: 0
};

await dexterMVP.depositPosition(tokenId, automationSettings);
```

### 3. Monitor Performance
```bash
# Real-time logs
tail -f automation/automation.log

# Metrics dashboard
npm run metrics

# Health status
npm run health
```

## 📈 Expected Performance

### Base Chain Metrics
- **Compound Frequency:** 5-15 minutes
- **Rebalance Frequency:** 30 minutes - 2 hours
- **Gas Cost per Compound:** ~$0.05-0.10
- **Gas Cost per Rebalance:** ~$0.20-0.50
- **Capital Efficiency:** >95% time in optimal range

### Success Rates
- **Compound Success Rate:** >95%
- **Rebalance Success Rate:** >90%
- **Average Gas per Compound:** ~150k gas
- **Average Gas per Rebalance:** ~300k gas

## 🚨 Troubleshooting

### Common Issues

#### 1. "Not authorized" errors
```bash
# Check keeper authorization
await dexterMVP.authorizedKeepers(keeperAddress)

# Authorize if needed
await dexterMVP.setKeeperAuthorization(keeperAddress, true)
```

#### 2. "Insufficient balance" errors
```bash
# Check keeper balance
npm run health

# Fund keeper wallet if needed
```

#### 3. Gas estimation failures
```bash
# Check gas price
await provider.getGasPrice()

# Adjust gas multiplier in config.js
gasMultiplier: 1.2 // Increase if needed
```

### Monitoring Commands
```bash
# Service status
pm2 status dexter-keeper

# Real-time logs
pm2 logs dexter-keeper --lines 100

# Restart service
pm2 restart dexter-keeper

# Stop service
pm2 stop dexter-keeper
```

## 🔧 Advanced Configuration

### Custom Concentration Levels
```solidity
// Set custom concentration
await rebalancer.updateConcentrationMultiplier(
    ConcentrationLevel.TIGHT,
    3 // New multiplier
);
```

### Batch Size Optimization
```javascript
// Adjust batch size based on network conditions
const optimalBatchSize = await calculateOptimalBatchSize();
config.automation.maxBatchSize = optimalBatchSize;
```

### Emergency Procedures
```javascript
// Emergency pause
await dexterMVP.emergencyPause();

// Resume operations
await dexterMVP.unpause();
```

## 📞 Support & Monitoring

### Health Monitoring
```bash
# Automated health checks
crontab -e
# Add: */30 * * * * cd /path/to/automation && npm run health
```

### Key Metrics to Monitor
- **Compound success rate** (target: >95%)
- **Average gas per operation**
- **Position concentration maintenance**
- **Keeper wallet balance**
- **Failed transaction rate**

### Log Analysis
```bash
# Search for errors
grep "ERROR" automation/automation.log

# Check compound frequency
grep "compound" automation/automation.log | tail -20

# Monitor gas usage
grep "Gas used" automation/automation.log | tail -20
```

## 🎉 Success Checklist

- [ ] All contracts deployed successfully
- [ ] Keeper wallet funded and authorized
- [ ] Health check passes
- [ ] Test position compounds successfully
- [ ] Automation service running continuously
- [ ] Monitoring setup and working
- [ ] Gas costs within expected range
- [ ] Performance metrics being tracked

## 📁 File Structure

```
contracts/mvp/
├── DexterMVP.sol                 // Main vault contract
├── UltraFrequentCompounder.sol   // 5-minute compound logic
├── BinRebalancer.sol             // Bin-based rebalancing
├── deploy.js                     // Deployment script
├── hardhat.config.js            // Hardhat configuration
└── package.json                 // Dependencies

automation/
├── ultra-frequent-keeper.js     // Main keeper service
├── config.js                    // Configuration
└── package.json                 // Dependencies

scripts/
└── mvp-deploy-and-test.sh       // Complete deployment script
```

This MVP provides immediate ultra-frequent automation while maintaining compatibility for future AI integration.