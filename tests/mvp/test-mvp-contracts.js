const { expect } = require('chai');
const { ethers } = require('hardhat');
const { time } = require('@nomicfoundation/hardhat-network-helpers');

describe('Dexter MVP Contracts', function () {
    let owner, keeper, user1, user2;
    let dexterMVP, compounder, rebalancer;
    let mockNPM, mockFactory, mockPool;
    let token0, token1;
    
    const MOCK_TOKEN_ID = 1;
    const MOCK_FEE = 3000;
    const COMPOUND_INTERVAL = 5 * 60; // 5 minutes
    const REBALANCE_INTERVAL = 30 * 60; // 30 minutes
    
    beforeEach(async function () {
        [owner, keeper, user1, user2] = await ethers.getSigners();
        
        // Deploy mock tokens
        const MockERC20 = await ethers.getContractFactory('MockERC20');
        token0 = await MockERC20.deploy('Token0', 'T0', ethers.utils.parseEther('1000000'));
        token1 = await MockERC20.deploy('Token1', 'T1', ethers.utils.parseEther('1000000'));
        
        // Deploy mock Uniswap contracts
        const MockNPM = await ethers.getContractFactory('MockNonfungiblePositionManager');
        mockNPM = await MockNPM.deploy();
        
        const MockFactory = await ethers.getContractFactory('MockUniswapV3Factory');
        mockFactory = await MockFactory.deploy();
        
        const MockPool = await ethers.getContractFactory('MockUniswapV3Pool');
        mockPool = await MockPool.deploy();
        
        // Set up mock pool
        await mockFactory.setPool(token0.address, token1.address, MOCK_FEE, mockPool.address);
        await mockPool.setSlot0(0, 0, 0, 0, 0, 0, true); // Current tick = 0
        await mockPool.setTickSpacing(60);
        
        // Deploy MVP contracts
        const BinRebalancer = await ethers.getContractFactory('BinRebalancer');
        rebalancer = await BinRebalancer.deploy(mockNPM.address, mockFactory.address);
        
        const UltraFrequentCompounder = await ethers.getContractFactory('UltraFrequentCompounder');
        compounder = await UltraFrequentCompounder.deploy(mockNPM.address);
        
        const DexterMVP = await ethers.getContractFactory('DexterMVP');
        dexterMVP = await DexterMVP.deploy(
            mockNPM.address,
            compounder.address,
            rebalancer.address,
            token0.address // Using token0 as mock WETH
        );
        
        // Authorize contracts
        await compounder.setAuthorizedVault(dexterMVP.address, true);
        await rebalancer.setKeeperAuthorization(dexterMVP.address, true);
        await dexterMVP.setKeeperAuthorization(keeper.address, true);
        
        // Set up mock position in NPM
        await mockNPM.setPosition(
            MOCK_TOKEN_ID,
            owner.address,
            token0.address,
            token1.address,
            MOCK_FEE,
            -120, // tickLower
            120,  // tickUpper
            ethers.utils.parseEther('100'), // liquidity
            0, 0, 0, 0
        );
        
        // Mock collected fees
        await mockNPM.setCollectableAmounts(MOCK_TOKEN_ID, ethers.utils.parseEther('1'), ethers.utils.parseEther('1'));
    });
    
    describe('DexterMVP Contract', function () {
        it('Should deploy with correct configuration', async function () {
            expect(await dexterMVP.nonfungiblePositionManager()).to.equal(mockNPM.address);
            expect(await dexterMVP.compounder()).to.equal(compounder.address);
            expect(await dexterMVP.rebalancer()).to.equal(rebalancer.address);
        });
        
        it('Should deposit position with automation settings', async function () {
            const automationSettings = {
                autoCompoundEnabled: true,
                autoRebalanceEnabled: true,
                compoundThresholdUSD: ethers.utils.parseEther('2'),
                maxBinsFromPrice: 3,
                concentrationLevel: 7,
                lastCompoundTime: 0,
                lastRebalanceTime: 0
            };
            
            await dexterMVP.depositPosition(MOCK_TOKEN_ID, automationSettings);
            
            const settings = await dexterMVP.getPositionAutomation(MOCK_TOKEN_ID);
            expect(settings.autoCompoundEnabled).to.be.true;
            expect(settings.autoRebalanceEnabled).to.be.true;
            expect(settings.concentrationLevel).to.equal(7);
        });
        
        it('Should check if position needs compounding', async function () {
            const automationSettings = {
                autoCompoundEnabled: true,
                autoRebalanceEnabled: false,
                compoundThresholdUSD: ethers.utils.parseEther('0.5'), // $0.50 threshold
                maxBinsFromPrice: 3,
                concentrationLevel: 5,
                lastCompoundTime: 0,
                lastRebalanceTime: 0
            };
            
            await dexterMVP.depositPosition(MOCK_TOKEN_ID, automationSettings);
            
            // Mock sufficient fees
            await mockNPM.setCollectableAmounts(MOCK_TOKEN_ID, ethers.utils.parseEther('1'), ethers.utils.parseEther('1'));
            
            const needsCompound = await dexterMVP.shouldCompound(MOCK_TOKEN_ID);
            expect(needsCompound).to.be.true;
        });
        
        it('Should execute compound operation', async function () {
            const automationSettings = {
                autoCompoundEnabled: true,
                autoRebalanceEnabled: false,
                compoundThresholdUSD: ethers.utils.parseEther('0.5'),
                maxBinsFromPrice: 3,
                concentrationLevel: 5,
                lastCompoundTime: 0,
                lastRebalanceTime: 0
            };
            
            await dexterMVP.depositPosition(MOCK_TOKEN_ID, automationSettings);
            
            // Execute compound as keeper
            await expect(dexterMVP.connect(keeper).executeCompound(MOCK_TOKEN_ID))
                .to.emit(dexterMVP, 'PositionCompounded');
        });
    });
    
    describe('UltraFrequentCompounder Contract', function () {
        beforeEach(async function () {
            await compounder.setUltraFrequentDefaults();
        });
        
        it('Should set ultra-frequent defaults', async function () {
            const defaults = await compounder.getUltraFrequentDefaults();
            expect(defaults.compoundInterval).to.equal(COMPOUND_INTERVAL);
            expect(defaults.minThresholdUSD).to.equal(ethers.utils.parseEther('0.5'));
        });
        
        it('Should execute batch compound', async function () {
            // Set up multiple positions
            const tokenIds = [1, 2, 3];
            
            for (const tokenId of tokenIds) {
                await mockNPM.setPosition(
                    tokenId,
                    owner.address,
                    token0.address,
                    token1.address,
                    MOCK_FEE,
                    -120, 120,
                    ethers.utils.parseEther('100'),
                    0, 0, 0, 0
                );
                await mockNPM.setCollectableAmounts(tokenId, ethers.utils.parseEther('1'), ethers.utils.parseEther('1'));
            }
            
            await expect(compounder.batchCompound(tokenIds))
                .to.emit(compounder, 'BatchCompoundExecuted');
        });
        
        it('Should respect compound interval timing', async function () {
            await compounder.setPositionCompoundTime(MOCK_TOKEN_ID, await time.latest());
            
            // Should not be ready immediately
            expect(await compounder.canCompound(MOCK_TOKEN_ID)).to.be.false;
            
            // Advance time by compound interval
            await time.increase(COMPOUND_INTERVAL + 1);
            
            // Should be ready now
            expect(await compounder.canCompound(MOCK_TOKEN_ID)).to.be.true;
        });
        
        it('Should track compound metrics', async function () {
            await compounder.executeCompound(MOCK_TOKEN_ID, ethers.utils.parseEther('1'));
            
            const metrics = await compounder.getCompoundMetrics(MOCK_TOKEN_ID);
            expect(metrics.compoundCount).to.equal(1);
            expect(metrics.totalFeesCompounded).to.equal(ethers.utils.parseEther('1'));
        });
    });
    
    describe('BinRebalancer Contract', function () {
        beforeEach(async function () {
            // Set up bin settings for test position
            const binSettings = {
                maxBinsFromPrice: 3,
                level: 2, // TIGHT
                concentrationRatio: 0,
                maintainConcentration: true,
                lastRebalanceTime: 0,
                lastRebalanceTick: 0
            };
            
            await rebalancer.setBinSettings(MOCK_TOKEN_ID, binSettings);
        });
        
        it('Should calculate bin position correctly', async function () {
            const binPosition = await rebalancer.calculateBinPosition(MOCK_TOKEN_ID);
            
            expect(binPosition.currentTick).to.equal(0);
            expect(binPosition.positionTickLower).to.equal(-120);
            expect(binPosition.positionTickUpper).to.equal(120);
            expect(binPosition.inRange).to.be.true;
        });
        
        it('Should determine rebalance necessity', async function () {
            // Position is in range, should not need rebalance
            expect(await rebalancer.shouldRebalance(MOCK_TOKEN_ID)).to.be.false;
            
            // Move price out of range by changing current tick
            await mockPool.setSlot0(300, 0, 0, 0, 0, 0, true); // Move price far from position
            
            // Now should need rebalance
            expect(await rebalancer.shouldRebalance(MOCK_TOKEN_ID)).to.be.true;
        });
        
        it('Should calculate concentrated range', async function () {
            const currentTick = 0;
            const tickSpacing = 60;
            const level = 2; // TIGHT
            
            const [tickLower, tickUpper] = await rebalancer.calculateConcentratedRange(
                currentTick,
                tickSpacing,
                level,
                0
            );
            
            expect(tickLower).to.be.lt(currentTick);
            expect(tickUpper).to.be.gt(currentTick);
            expect(tickUpper - tickLower).to.be.gt(tickSpacing);
        });
        
        it('Should preview rebalance before execution', async function () {
            const [newTickLower, newTickUpper, binsFromPrice] = await rebalancer.previewRebalance(MOCK_TOKEN_ID);
            
            expect(newTickLower).to.be.a('number');
            expect(newTickUpper).to.be.a('number');
            expect(binsFromPrice).to.be.a('number');
        });
        
        it('Should update concentration levels', async function () {
            const newMultiplier = 5;
            await rebalancer.updateConcentrationMultiplier(2, newMultiplier); // TIGHT level
            
            // Verify the update
            const multiplier = await rebalancer.concentrationMultipliers(2);
            expect(multiplier).to.equal(newMultiplier);
        });
    });
    
    describe('Integration Tests', function () {
        it('Should execute full automation cycle', async function () {
            // 1. Deposit position with full automation
            const automationSettings = {
                autoCompoundEnabled: true,
                autoRebalanceEnabled: true,
                compoundThresholdUSD: ethers.utils.parseEther('0.5'),
                maxBinsFromPrice: 2,
                concentrationLevel: 7,
                lastCompoundTime: 0,
                lastRebalanceTime: 0
            };
            
            await dexterMVP.depositPosition(MOCK_TOKEN_ID, automationSettings);
            
            // 2. Advance time to trigger compound
            await time.increase(COMPOUND_INTERVAL + 1);
            
            // 3. Execute compound
            await expect(dexterMVP.connect(keeper).executeCompound(MOCK_TOKEN_ID))
                .to.emit(dexterMVP, 'PositionCompounded');
            
            // 4. Move price to trigger rebalance
            await mockPool.setSlot0(500, 0, 0, 0, 0, 0, true);
            
            // 5. Check rebalance necessity
            expect(await dexterMVP.shouldRebalance(MOCK_TOKEN_ID)).to.be.true;
            
            // 6. Execute rebalance
            await expect(dexterMVP.connect(keeper).executeRebalance(MOCK_TOKEN_ID))
                .to.emit(rebalancer, 'BinBasedRebalance');
        });
        
        it('Should handle batch operations efficiently', async function () {
            const tokenIds = [1, 2, 3, 4, 5];
            
            // Set up multiple positions
            for (const tokenId of tokenIds) {
                await mockNPM.setPosition(
                    tokenId,
                    owner.address,
                    token0.address,
                    token1.address,
                    MOCK_FEE,
                    -120, 120,
                    ethers.utils.parseEther('100'),
                    0, 0, 0, 0
                );
                
                const automationSettings = {
                    autoCompoundEnabled: true,
                    autoRebalanceEnabled: true,
                    compoundThresholdUSD: ethers.utils.parseEther('0.5'),
                    maxBinsFromPrice: 3,
                    concentrationLevel: 5,
                    lastCompoundTime: 0,
                    lastRebalanceTime: 0
                };
                
                await dexterMVP.depositPosition(tokenId, automationSettings);
            }
            
            // Execute batch compound
            await expect(compounder.batchCompound(tokenIds))
                .to.emit(compounder, 'BatchCompoundExecuted');
        });
        
        it('Should maintain position concentration over time', async function () {
            const automationSettings = {
                autoCompoundEnabled: true,
                autoRebalanceEnabled: true,
                compoundThresholdUSD: ethers.utils.parseEther('0.5'),
                maxBinsFromPrice: 2,
                concentrationLevel: 9, // Ultra-tight
                lastCompoundTime: 0,
                lastRebalanceTime: 0
            };
            
            await dexterMVP.depositPosition(MOCK_TOKEN_ID, automationSettings);
            
            // Simulate price movement
            await mockPool.setSlot0(200, 0, 0, 0, 0, 0, true);
            
            // Should trigger rebalance due to tight concentration
            expect(await dexterMVP.shouldRebalance(MOCK_TOKEN_ID)).to.be.true;
            
            // Execute rebalance to maintain concentration
            await dexterMVP.connect(keeper).executeRebalance(MOCK_TOKEN_ID);
            
            // Position should be back in optimal range
            const binData = await rebalancer.calculateBinPosition(MOCK_TOKEN_ID);
            expect(binData.binsFromPrice).to.be.lte(2);
        });
    });
    
    describe('Gas Optimization Tests', function () {
        it('Should optimize gas for batch compounds', async function () {
            const tokenIds = [1, 2, 3, 4, 5];
            
            // Set up positions
            for (const tokenId of tokenIds) {
                await mockNPM.setPosition(
                    tokenId,
                    owner.address,
                    token0.address,
                    token1.address,
                    MOCK_FEE,
                    -120, 120,
                    ethers.utils.parseEther('100'),
                    0, 0, 0, 0
                );
            }
            
            // Test gas estimation
            const gasEstimate = await compounder.estimateGas.batchCompound(tokenIds);
            expect(gasEstimate.toNumber()).to.be.lt(1000000); // Should be under 1M gas
        });
        
        it('Should track gas usage metrics', async function () {
            await compounder.executeCompound(MOCK_TOKEN_ID, ethers.utils.parseEther('1'));
            
            const metrics = await compounder.getGasMetrics();
            expect(metrics.totalGasUsed).to.be.gt(0);
        });
    });
    
    describe('Error Handling', function () {
        it('Should revert unauthorized compound attempts', async function () {
            await expect(dexterMVP.connect(user1).executeCompound(MOCK_TOKEN_ID))
                .to.be.revertedWith('Not authorized');
        });
        
        it('Should handle insufficient fees gracefully', async function () {
            const automationSettings = {
                autoCompoundEnabled: true,
                autoRebalanceEnabled: false,
                compoundThresholdUSD: ethers.utils.parseEther('10'), // High threshold
                maxBinsFromPrice: 3,
                concentrationLevel: 5,
                lastCompoundTime: 0,
                lastRebalanceTime: 0
            };
            
            await dexterMVP.depositPosition(MOCK_TOKEN_ID, automationSettings);
            
            // Should not need compound with low fees
            expect(await dexterMVP.shouldCompound(MOCK_TOKEN_ID)).to.be.false;
        });
        
        it('Should prevent rebalance when not needed', async function () {
            const automationSettings = {
                autoCompoundEnabled: false,
                autoRebalanceEnabled: true,
                compoundThresholdUSD: ethers.utils.parseEther('1'),
                maxBinsFromPrice: 5, // High tolerance
                concentrationLevel: 5,
                lastCompoundTime: 0,
                lastRebalanceTime: 0
            };
            
            await dexterMVP.depositPosition(MOCK_TOKEN_ID, automationSettings);
            
            // Position is in range, should not rebalance
            await expect(dexterMVP.connect(keeper).executeRebalance(MOCK_TOKEN_ID))
                .to.be.revertedWith('Rebalance not needed');
        });
    });
});