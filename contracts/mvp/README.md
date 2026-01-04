# Dexter MVP - Ultra-Frequent Automation

## 🚀 **Overview**

The Dexter MVP contracts provide ultra-high frequency automation for Uniswap V3 positions, optimized for Base chain's low gas costs:

- **5-minute compound intervals** with $0.50 minimum thresholds
- **Bin-based rebalancing** to maintain concentrated liquidity 
- **Gas-optimized batch operations** for up to 100 positions
- **No AI dependencies** - pure deterministic automation

## 📁 **Contract Architecture**

```
contracts/mvp/
├── DexterMVP.sol                    // Main vault contract
├── UltraFrequentCompounder.sol      // 5-minute compound logic  
├── BinRebalancer.sol                // Bin-based position management
└── README.md                        // This documentation
```

## 🔧 **Key Features**

### **Ultra-Frequent Compounding**
- **Frequency**: Every 5 minutes OR when fees > threshold
- **Threshold**: As low as $0.50 (Base chain optimized)
- **Batch Size**: Up to 100 positions per transaction
- **Gas Limit**: ~150k gas per compound

### **Bin-Based Rebalancing**
- **Concentration Levels**: 1-10 scale (10 = ultra-tight)
- **Bin Drift Tolerance**: 1-5 bins before rebalance
- **Range Calculation**: Dynamic around current price
- **MEV Protection**: Built-in slippage controls

### **Base Chain Optimization**
- **Low Gas Costs**: ~$0.05-0.10 per compound
- **High Frequency**: Every 5-30 minutes based on activity
- **Batch Processing**: Multiple positions per transaction
- **Smart Triggering**: Fee + time based automation

## 🎛️ **Configuration Options**

### **Automation Settings**
```solidity
struct AutomationSettings {
    bool autoCompoundEnabled;        // Enable 5-min compounds
    bool autoRebalanceEnabled;       // Enable bin rebalancing
    uint256 compoundThresholdUSD;    // $0.50 - $10 range
    uint256 maxBinsFromPrice;        // 1-5 bins before rebalance
    uint256 concentrationLevel;      // 1-10 concentration scale
    uint256 lastCompoundTime;        // Tracking timestamp
    uint256 lastRebalanceTime;       // Tracking timestamp
}
```

### **Concentration Levels**
- **Level 1-2**: Wide range (0.5-1% width)
- **Level 3-4**: Moderate concentration (0.3-0.5% width)
- **Level 5-6**: Tight concentration (0.2-0.3% width)
- **Level 7-8**: Very tight (0.1-0.2% width)
- **Level 9-10**: Ultra-tight (0.05-0.1% width)

## 🚀 **Deployment Guide**

### **1. Deploy Core Contracts**
```bash
# Deploy main contracts
forge create DexterMVP --constructor-args $NPM $ROUTER $WETH
forge create UltraFrequentCompounder --constructor-args $NPM
forge create BinRebalancer --constructor-args $NPM $FACTORY
```

### **2. Configure Automation**
```solidity
// Authorize keeper
dexterMVP.setKeeperAuthorization(keeperAddress, true);

// Set ultra-frequent defaults
compounder.setUltraFrequentDefaults(tokenId);

// Configure bin settings
rebalancer.setConcentrationLevel(tokenId, ConcentrationLevel.TIGHT, 3);
```

### **3. Start Keeper Service**
```bash
cd automation/
npm install
npm start
```

## 📊 **Expected Performance**

### **Base Chain Metrics**
- **Compound Frequency**: 5-15 minutes
- **Rebalance Frequency**: 30 minutes - 2 hours
- **Gas Cost per Compound**: ~$0.05-0.10
- **Gas Cost per Rebalance**: ~$0.20-0.50
- **Capital Efficiency**: >95% time in optimal range

### **Fee Generation Enhancement**
- **Concentrated Positions**: 5-20x higher fee generation
- **Ultra-Frequent Compounding**: Maximizes compound effect
- **Bin-Based Management**: Maintains optimal concentration
- **Base Chain Advantage**: Sustainable high-frequency operations

## 🔧 **Integration Examples**

### **Deposit Position with Automation**
```solidity
AutomationSettings memory settings = AutomationSettings({
    autoCompoundEnabled: true,
    autoRebalanceEnabled: true,
    compoundThresholdUSD: 2e18,          // $2 minimum
    maxBinsFromPrice: 3,                 // Rebalance after 3 bins
    concentrationLevel: 7,               // Very tight concentration
    lastCompoundTime: block.timestamp,
    lastRebalanceTime: block.timestamp
});

dexterMVP.depositPosition(tokenId, settings);
```

### **Check Automation Status**
```solidity
// Check if position needs compounding
bool needsCompound = dexterMVP.shouldCompound(tokenId);

// Check if position needs rebalancing  
bool needsRebalance = dexterMVP.shouldRebalance(tokenId);

// Get current bin position
BinPosition memory binData = dexterMVP.calculateBinPosition(tokenId);
```

### **Execute Batch Operations**
```solidity
// Batch compound multiple positions
uint256[] memory tokenIds = [1, 2, 3, 4, 5];
compounder.batchCompound(tokenIds);

// Smart batch (only compound positions that need it)
compounder.smartBatchCompound(allTokenIds);
```

## ⚠️ **Important Notes**

### **Concentration Risk**
- Higher concentration = higher fees but higher IL risk
- Ultra-tight positions (level 9-10) require frequent rebalancing
- Monitor bin drift carefully for concentrated positions

### **Gas Optimization**
- Batch operations recommended for multiple positions
- Base chain enables sustainable high-frequency automation
- Monitor keeper wallet balance for continuous operation

### **Slippage Protection**
- Built-in MEV protection for rebalance operations
- Conservative slippage settings for automated trades
- Emergency pause functionality for unusual market conditions

## 🔄 **Migration to Full AI**

The MVP contracts are designed for seamless upgrade to full AI automation:

1. **Interface Compatibility**: Same interfaces as future AI contracts
2. **Settings Preservation**: Automation preferences maintained
3. **Performance Data**: Historical metrics for AI training
4. **Gradual Migration**: Switch individual positions as desired

## 📞 **Support & Monitoring**

### **Health Checks**
```bash
# Check keeper service health
npm run health

# View automation metrics
node -e "const keeper = require('./ultra-frequent-keeper.js'); keeper.reportMetrics();"
```

### **Key Metrics to Monitor**
- Compound success rate (target: >95%)
- Average gas per operation
- Position concentration maintenance
- Keeper wallet balance
- Failed transaction rate

This MVP system provides immediate value with ultra-frequent automation while maintaining compatibility for future AI integration.