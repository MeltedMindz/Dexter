// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/security/Pausable.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/utils/math/Math.sol";
import "@uniswap/v3-core/contracts/interfaces/IUniswapV3Pool.sol";
import "@uniswap/v3-core/contracts/libraries/TickMath.sol";
import "@uniswap/v3-core/contracts/libraries/FullMath.sol";
import "@uniswap/v3-periphery/contracts/libraries/LiquidityAmounts.sol";
import "@uniswap/v3-core/contracts/interfaces/callback/IUniswapV3MintCallback.sol";

import "./IDexterVault.sol";
import "../core/DexterCompoundor.sol";

/// @title DexterVault - ERC4626 compliant vault for AI-powered Uniswap V3 position management
/// @notice Combines proven vault patterns with advanced AI optimization and risk management
contract DexterVault is 
    IDexterVault, 
    ERC20, 
    ReentrancyGuard, 
    Pausable, 
    Ownable,
    IUniswapV3MintCallback 
{
    using SafeERC20 for IERC20;
    using Math for uint256;

    // ============ CONSTANTS ============
    
    uint256 public constant PRECISION = 1e18;
    uint256 public constant MAX_BPS = 10000;
    uint256 public constant MAX_MANAGEMENT_FEE_BPS = 200; // 2%
    uint256 public constant MAX_PERFORMANCE_FEE_BPS = 2000; // 20%
    uint256 public constant MAX_AI_FEE_BPS = 100; // 1%
    uint256 public constant MAX_POSITION_RANGES = 10;
    uint32 public constant DEFAULT_TWAP_INTERVAL = 3600; // 1 hour
    uint256 public constant DEFAULT_MAX_DEVIATION = 500; // 5%

    // ============ STATE VARIABLES ============
    
    IUniswapV3Pool public immutable pool;
    IERC20 public immutable token0;
    IERC20 public immutable token1;
    uint24 public immutable fee;
    int24 public immutable tickSpacing;
    
    DexterCompoundor public immutable compoundor;
    address public immutable aiManager;
    
    VaultConfig public vaultConfig;
    FeeConfiguration public feeConfiguration;
    TWAPConfig public twapConfig;
    
    PositionRange[] public positionRanges;
    mapping(uint256 => bool) public activeRanges;
    
    VaultMetrics public metrics;
    
    bool public gammaMode;
    bool public mintCalled;
    
    uint256 public lastCompoundTimestamp;
    uint256 public lastRebalanceTimestamp;
    uint256 public totalFeesCollected0;
    uint256 public totalFeesCollected1;
    
    // AI-related storage
    mapping(bytes32 => uint256) public aiRecommendationScores;
    uint256 public lastAIUpdateTimestamp;
    
    // Emergency controls
    mapping(address => bool) public emergencyOperators;
    
    // ============ MODIFIERS ============
    
    modifier onlyAIManager() {
        require(msg.sender == aiManager || msg.sender == owner(), "Not AI manager");
        _;
    }
    
    modifier onlyEmergencyOperator() {
        require(emergencyOperators[msg.sender] || msg.sender == owner(), "Not emergency operator");
        _;
    }
    
    modifier validRange(uint256 rangeId) {
        require(rangeId < positionRanges.length && activeRanges[rangeId], "Invalid range");
        _;
    }
    
    modifier twapProtected() {
        if (twapConfig.enabled) {
            _validateTWAP();
        }
        _;
    }

    // ============ CONSTRUCTOR ============
    
    constructor(
        address _pool,
        address _compoundor,
        address _aiManager,
        string memory _name,
        string memory _symbol,
        VaultConfig memory _initialConfig
    ) ERC20(_name, _symbol) {
        require(_pool != address(0), "Invalid pool");
        require(_compoundor != address(0), "Invalid compoundor");
        require(_aiManager != address(0), "Invalid AI manager");
        
        pool = IUniswapV3Pool(_pool);
        token0 = IERC20(pool.token0());
        token1 = IERC20(pool.token1());
        fee = pool.fee();
        tickSpacing = pool.tickSpacing();
        
        compoundor = DexterCompoundor(_compoundor);
        aiManager = _aiManager;
        
        vaultConfig = _initialConfig;
        
        // Initialize default fee configuration
        feeConfiguration = FeeConfiguration({
            managementFeeBps: 50,        // 0.5%
            performanceFeeBps: 1000,     // 10%
            aiOptimizationFeeBps: 25,    // 0.25%
            feeRecipient: owner(),
            strategistShareBps: 2000,    // 20%
            strategist: owner()
        });
        
        // Initialize TWAP configuration
        twapConfig = TWAPConfig({
            enabled: true,
            interval: DEFAULT_TWAP_INTERVAL,
            maxDeviation: DEFAULT_MAX_DEVIATION,
            cooldownPeriod: 300 // 5 minutes
        });
        
        // Set emergency operators
        emergencyOperators[owner()] = true;
        emergencyOperators[_aiManager] = true;
        
        // Approve tokens for pool operations
        token0.safeApprove(_pool, type(uint256).max);
        token1.safeApprove(_pool, type(uint256).max);
    }

    // ============ ERC4626 IMPLEMENTATION ============
    
    function asset() public view override returns (address) {
        // Return the "primary" asset - could be token0 or a synthetic representation
        return address(token0);
    }
    
    function totalAssets() public view override returns (uint256) {
        (uint256 amount0, uint256 amount1) = getTotalAmounts();
        return _convertToAssetValue(amount0, amount1);
    }
    
    function convertToShares(uint256 assets) public view override returns (uint256) {
        return _convertToShares(assets, Math.Rounding.Down);
    }
    
    function convertToAssets(uint256 shares) public view override returns (uint256) {
        return _convertToAssets(shares, Math.Rounding.Down);
    }
    
    function maxDeposit(address) public view override returns (uint256) {
        if (paused()) return 0;
        return type(uint256).max; // No deposit limit by default
    }
    
    function maxMint(address receiver) public view override returns (uint256) {
        return convertToShares(maxDeposit(receiver));
    }
    
    function maxWithdraw(address owner) public view override returns (uint256) {
        return convertToAssets(balanceOf(owner));
    }
    
    function maxRedeem(address owner) public view override returns (uint256) {
        return balanceOf(owner);
    }
    
    function previewDeposit(uint256 assets) public view override returns (uint256) {
        return convertToShares(assets);
    }
    
    function previewMint(uint256 shares) public view override returns (uint256) {
        return convertToAssets(shares);
    }
    
    function previewWithdraw(uint256 assets) public view override returns (uint256) {
        return _convertToShares(assets, Math.Rounding.Up);
    }
    
    function previewRedeem(uint256 shares) public view override returns (uint256) {
        return convertToAssets(shares);
    }
    
    function deposit(uint256 assets, address receiver) 
        public 
        override 
        nonReentrant 
        whenNotPaused 
        twapProtected
        returns (uint256 shares) 
    {
        require(assets > 0, "Zero assets");
        require(receiver != address(0), "Zero receiver");
        
        // Validate deposit
        (bool valid, string memory reason) = validateDeposit(assets, receiver);
        require(valid, reason);
        
        // Calculate shares
        shares = previewDeposit(assets);
        require(shares > 0, "Zero shares");
        
        // Transfer tokens
        token0.safeTransferFrom(msg.sender, address(this), assets);
        
        // Mint shares
        _mint(receiver, shares);
        
        // Deploy liquidity if configured
        if (vaultConfig.positionType != PositionType.MANUAL) {
            _deployLiquidity();
        }
        
        // Update metrics
        _updateMetrics();
        
        emit Deposit(msg.sender, receiver, assets, shares);
    }
    
    function mint(uint256 shares, address receiver) 
        public 
        override 
        nonReentrant 
        whenNotPaused 
        returns (uint256 assets) 
    {
        require(shares > 0, "Zero shares");
        require(receiver != address(0), "Zero receiver");
        
        assets = previewMint(shares);
        require(assets > 0, "Zero assets");
        
        // Transfer tokens
        token0.safeTransferFrom(msg.sender, address(this), assets);
        
        // Mint shares
        _mint(receiver, shares);
        
        // Deploy liquidity
        if (vaultConfig.positionType != PositionType.MANUAL) {
            _deployLiquidity();
        }
        
        _updateMetrics();
        
        emit Deposit(msg.sender, receiver, assets, shares);
    }
    
    function withdraw(uint256 assets, address receiver, address owner) 
        public 
        override 
        nonReentrant 
        returns (uint256 shares) 
    {
        require(assets > 0, "Zero assets");
        require(receiver != address(0), "Zero receiver");
        
        shares = previewWithdraw(assets);
        require(shares > 0, "Zero shares");
        
        if (msg.sender != owner) {
            _spendAllowance(owner, msg.sender, shares);
        }
        
        // Burn shares
        _burn(owner, shares);
        
        // Withdraw liquidity
        _withdrawLiquidity(assets, receiver);
        
        _updateMetrics();
        
        emit Withdraw(msg.sender, receiver, owner, assets, shares);
    }
    
    function redeem(uint256 shares, address receiver, address owner) 
        public 
        override 
        nonReentrant 
        returns (uint256 assets) 
    {
        require(shares > 0, "Zero shares");
        require(receiver != address(0), "Zero receiver");
        
        if (msg.sender != owner) {
            _spendAllowance(owner, msg.sender, shares);
        }
        
        assets = previewRedeem(shares);
        require(assets > 0, "Zero assets");
        
        // Burn shares
        _burn(owner, shares);
        
        // Withdraw liquidity
        _withdrawLiquidity(assets, receiver);
        
        _updateMetrics();
        
        emit Withdraw(msg.sender, receiver, owner, assets, shares);
    }

    // ============ VAULT MANAGEMENT ============
    
    function configureVault(VaultConfig calldata config) external override onlyOwner {
        require(config.rebalanceThreshold <= MAX_BPS, "Invalid threshold");
        require(config.maxSlippageBps <= MAX_BPS, "Invalid slippage");
        
        StrategyMode oldMode = vaultConfig.mode;
        vaultConfig = config;
        
        emit StrategyModeChanged(oldMode, config.mode);
    }
    
    function setFeeConfiguration(FeeConfiguration calldata config) external override onlyOwner {
        require(config.managementFeeBps <= MAX_MANAGEMENT_FEE_BPS, "Management fee too high");
        require(config.performanceFeeBps <= MAX_PERFORMANCE_FEE_BPS, "Performance fee too high");
        require(config.aiOptimizationFeeBps <= MAX_AI_FEE_BPS, "AI fee too high");
        require(config.strategistShareBps <= MAX_BPS, "Strategist share too high");
        require(config.feeRecipient != address(0), "Zero fee recipient");
        require(config.strategist != address(0), "Zero strategist");
        
        feeConfiguration = config;
    }
    
    function setAIManagement(bool enabled) external override onlyOwner {
        vaultConfig.aiOptimizationEnabled = enabled;
        emit AIOptimizationEnabled(enabled);
    }
    
    function setAutoCompound(bool enabled, uint256 threshold) external override onlyOwner {
        vaultConfig.autoCompoundEnabled = enabled;
        vaultConfig.rebalanceThreshold = threshold;
    }
    
    function configureTWAPProtection(TWAPConfig calldata config) external override onlyOwner {
        require(config.interval >= 60, "TWAP interval too short");
        require(config.maxDeviation <= MAX_BPS, "Max deviation too high");
        
        twapConfig = config;
    }

    // ============ POSITION MANAGEMENT ============
    
    function addPositionRange(int24 tickLower, int24 tickUpper, uint256 allocation) 
        external 
        override 
        onlyOwner 
        returns (uint256 rangeId) 
    {
        require(positionRanges.length < MAX_POSITION_RANGES, "Too many ranges");
        require(tickLower < tickUpper, "Invalid tick range");
        require(tickLower % tickSpacing == 0, "Invalid tick lower");
        require(tickUpper % tickSpacing == 0, "Invalid tick upper");
        require(allocation <= MAX_BPS, "Invalid allocation");
        
        rangeId = positionRanges.length;
        
        positionRanges.push(PositionRange({
            tickLower: tickLower,
            tickUpper: tickUpper,
            allocation: allocation,
            isActive: true,
            liquidity: 0
        }));
        
        activeRanges[rangeId] = true;
        
        emit PositionRangeAdded(rangeId, tickLower, tickUpper, allocation);
    }
    
    function removePositionRange(uint256 rangeId) external override onlyOwner validRange(rangeId) {
        // Remove liquidity from this range first
        PositionRange storage range = positionRanges[rangeId];
        if (range.liquidity > 0) {
            _burnLiquidity(range.tickLower, range.tickUpper, uint128(range.liquidity));
        }
        
        activeRanges[rangeId] = false;
        range.isActive = false;
        
        emit PositionRangeRemoved(rangeId);
    }
    
    function updateAllocations(uint256[] calldata rangeIds, uint256[] calldata allocations) 
        external 
        override 
        onlyOwner 
    {
        require(rangeIds.length == allocations.length, "Length mismatch");
        
        uint256 totalAllocation = 0;
        for (uint256 i = 0; i < rangeIds.length; i++) {
            require(activeRanges[rangeIds[i]], "Inactive range");
            require(allocations[i] <= MAX_BPS, "Invalid allocation");
            
            positionRanges[rangeIds[i]].allocation = allocations[i];
            totalAllocation += allocations[i];
        }
        
        require(totalAllocation <= MAX_BPS, "Total allocation exceeds 100%");
        
        // Rebalance liquidity according to new allocations
        _rebalanceLiquidity();
    }
    
    function rebalancePositions() external override nonReentrant returns (bool success) {
        require(
            msg.sender == owner() || 
            msg.sender == aiManager || 
            vaultConfig.mode == StrategyMode.FULLY_AUTOMATED,
            "Not authorized"
        );
        
        try this._rebalanceLiquidity() {
            lastRebalanceTimestamp = block.timestamp;
            metrics.aiOptimizationCount++;
            success = true;
        } catch {
            success = false;
        }
    }
    
    function getPositionRanges() external view override returns (PositionRange[] memory) {
        uint256 activeCount = 0;
        for (uint256 i = 0; i < positionRanges.length; i++) {
            if (activeRanges[i]) activeCount++;
        }
        
        PositionRange[] memory active = new PositionRange[](activeCount);
        uint256 index = 0;
        
        for (uint256 i = 0; i < positionRanges.length; i++) {
            if (activeRanges[i]) {
                active[index] = positionRanges[i];
                index++;
            }
        }
        
        return active;
    }

    // ============ AI INTEGRATION ============
    
    function getAIRecommendation() 
        external 
        view 
        override 
        returns (
            StrategyMode recommendedMode,
            PositionRange[] memory recommendedRanges,
            uint256 confidenceScore
        ) 
    {
        // This would integrate with the AI service
        // For now, return current configuration
        recommendedMode = vaultConfig.mode;
        recommendedRanges = this.getPositionRanges();
        confidenceScore = 85; // Mock confidence score
    }
    
    function applyAIRecommendation(bytes calldata aiData) external override onlyAIManager {
        // Decode and apply AI recommendations
        // This would integrate with the AI service
        lastAIUpdateTimestamp = block.timestamp;
        metrics.aiOptimizationCount++;
    }
    
    function getHealthScore() 
        external 
        view 
        override 
        returns (uint256 healthScore, string memory analysis) 
    {
        // Calculate health score based on various factors
        uint256 utilizationScore = _calculateUtilizationScore();
        uint256 performanceScore = _calculatePerformanceScore();
        uint256 riskScore = _calculateRiskScore();
        
        healthScore = (utilizationScore + performanceScore + riskScore) / 3;
        analysis = _generateHealthAnalysis(healthScore);
    }

    // ============ COMPOUNDING ============
    
    function compound() external override nonReentrant returns (uint256 newLiquidity) {
        require(
            msg.sender == owner() || 
            msg.sender == aiManager ||
            vaultConfig.autoCompoundEnabled,
            "Not authorized"
        );
        
        // Collect fees from all positions
        (uint256 fees0, uint256 fees1) = _collectAllFees();
        
        if (fees0 > 0 || fees1 > 0) {
            // Distribute fees
            _distributeFees(fees0, fees1);
            
            // Compound remaining fees
            newLiquidity = _compoundFees(fees0, fees1);
            
            lastCompoundTimestamp = block.timestamp;
            metrics.successfulCompounds++;
            
            emit AutoCompoundTriggered(fees0, fees1, newLiquidity);
        }
    }
    
    function shouldAutoCompound() 
        external 
        view 
        override 
        returns (bool shouldCompound, uint256 estimatedGas) 
    {
        if (!vaultConfig.autoCompoundEnabled) return (false, 0);
        
        (uint256 fees0, uint256 fees1) = _getPendingFees();
        uint256 totalFeeValue = _convertToAssetValue(fees0, fees1);
        
        shouldCompound = totalFeeValue >= vaultConfig.rebalanceThreshold;
        estimatedGas = 200000; // Estimated gas cost
    }
    
    function analyzeCompoundOpportunity() 
        external 
        view 
        override 
        returns (
            uint256 availableFees0,
            uint256 availableFees1,
            uint256 estimatedNewLiquidity,
            uint256 expectedAPRIncrease
        ) 
    {
        (availableFees0, availableFees1) = _getPendingFees();
        estimatedNewLiquidity = _estimateNewLiquidity(availableFees0, availableFees1);
        expectedAPRIncrease = _calculateAPRIncrease(estimatedNewLiquidity);
    }

    // ============ ANALYTICS ============
    
    function getVaultMetrics() external view override returns (VaultMetrics memory) {
        VaultMetrics memory currentMetrics = metrics;
        currentMetrics.totalValueLocked = totalAssets();
        currentMetrics.apr = _calculateCurrentAPR();
        return currentMetrics;
    }
    
    function getPerformanceHistory(uint256 fromTimestamp, uint256 toTimestamp) 
        external 
        view 
        override 
        returns (uint256[] memory timestamps, uint256[] memory values) 
    {
        // This would integrate with historical data storage
        // For now, return empty arrays
        timestamps = new uint256[](0);
        values = new uint256[](0);
    }
    
    function benchmarkPerformance() 
        external 
        view 
        override 
        returns (
            uint256 vaultAPR,
            uint256 benchmarkAPR,
            int256 outperformance
        ) 
    {
        vaultAPR = _calculateCurrentAPR();
        benchmarkAPR = _getBenchmarkAPR();
        outperformance = int256(vaultAPR) - int256(benchmarkAPR);
    }

    // ============ RISK MANAGEMENT ============
    
    function validateDeposit(uint256 assets, address receiver) 
        public 
        view 
        override 
        returns (bool valid, string memory reason) 
    {
        if (paused()) return (false, "Vault paused");
        if (assets == 0) return (false, "Zero assets");
        if (receiver == address(0)) return (false, "Zero receiver");
        
        // Check TWAP if enabled
        if (twapConfig.enabled) {
            try this._checkTWAP() {
                // TWAP check passed
            } catch {
                return (false, "TWAP validation failed");
            }
        }
        
        return (true, "");
    }
    
    function validateWithdrawal(uint256 shares, address receiver, address owner) 
        external 
        view 
        override 
        returns (bool valid, string memory reason) 
    {
        if (shares == 0) return (false, "Zero shares");
        if (receiver == address(0)) return (false, "Zero receiver");
        if (balanceOf(owner) < shares) return (false, "Insufficient shares");
        
        return (true, "");
    }
    
    function getRiskAssessment() 
        external 
        view 
        override 
        returns (
            uint256 riskScore,
            string memory riskLevel,
            string[] memory riskFactors
        ) 
    {
        riskScore = _calculateRiskScore();
        
        if (riskScore >= 80) {
            riskLevel = "Low";
        } else if (riskScore >= 60) {
            riskLevel = "Medium";
        } else if (riskScore >= 40) {
            riskLevel = "High";
        } else {
            riskLevel = "Very High";
        }
        
        riskFactors = _identifyRiskFactors();
    }
    
    function emergencyPause(bool paused, string calldata reason) 
        external 
        override 
        onlyEmergencyOperator 
    {
        if (paused) {
            _pause();
        } else {
            _unpause();
        }
        
        emit EmergencyPause(paused, reason);
    }

    // ============ COMPATIBILITY ============
    
    function enableGammaMode(bool enabled) external override onlyOwner {
        gammaMode = enabled;
        
        if (enabled && positionRanges.length == 0) {
            // Initialize with default dual positions
            _initializeGammaPositions();
        }
    }
    
    function setDualPositionStrategy(
        int24 baseLower, 
        int24 baseUpper,
        int24 limitLower, 
        int24 limitUpper
    ) external override onlyOwner {
        require(gammaMode, "Gamma mode not enabled");
        require(baseLower < baseUpper, "Invalid base range");
        require(limitLower < limitUpper, "Invalid limit range");
        
        // Clear existing ranges
        for (uint256 i = 0; i < positionRanges.length; i++) {
            activeRanges[i] = false;
        }
        delete positionRanges;
        
        // Add base position (80% allocation)
        positionRanges.push(PositionRange({
            tickLower: baseLower,
            tickUpper: baseUpper,
            allocation: 8000,
            isActive: true,
            liquidity: 0
        }));
        activeRanges[0] = true;
        
        // Add limit position (20% allocation)
        positionRanges.push(PositionRange({
            tickLower: limitLower,
            tickUpper: limitUpper,
            allocation: 2000,
            isActive: true,
            liquidity: 0
        }));
        activeRanges[1] = true;
        
        emit PositionRangeAdded(0, baseLower, baseUpper, 8000);
        emit PositionRangeAdded(1, limitLower, limitUpper, 2000);
    }
    
    function migrateFromNFTPositions(uint256[] calldata tokenIds) 
        external 
        override 
        returns (uint256 shares) 
    {
        // This would integrate with the NFT position manager
        // For now, revert as not implemented
        revert("Not implemented");
    }
    
    function extractToNFTPositions(uint256 shares, uint256[] calldata desiredRanges) 
        external 
        override 
        returns (uint256[] memory tokenIds) 
    {
        // This would integrate with the NFT position manager
        // For now, revert as not implemented
        revert("Not implemented");
    }

    // ============ VIEW FUNCTIONS ============
    
    function getVaultConfig() external view override returns (VaultConfig memory) {
        return vaultConfig;
    }
    
    function getFeeConfiguration() external view override returns (FeeConfiguration memory) {
        return feeConfiguration;
    }
    
    function getTWAPConfig() external view override returns (TWAPConfig memory) {
        return twapConfig;
    }
    
    function isGammaMode() external view override returns (bool) {
        return gammaMode;
    }
    
    function getTokens() external view override returns (address, address, uint24) {
        return (address(token0), address(token1), fee);
    }
    
    function getTotalAmounts() public view override returns (uint256 amount0, uint256 amount1) {
        // Calculate total amounts across all position ranges
        for (uint256 i = 0; i < positionRanges.length; i++) {
            if (activeRanges[i] && positionRanges[i].liquidity > 0) {
                (uint256 rangeAmount0, uint256 rangeAmount1) = _getAmountsForRange(i);
                amount0 += rangeAmount0;
                amount1 += rangeAmount1;
            }
        }
        
        // Add idle balances
        amount0 += token0.balanceOf(address(this));
        amount1 += token1.balanceOf(address(this));
    }
    
    function isPaused() external view override returns (bool) {
        return paused();
    }

    // ============ UNISWAP V3 CALLBACK ============
    
    function uniswapV3MintCallback(
        uint256 amount0Owed,
        uint256 amount1Owed,
        bytes calldata
    ) external override {
        require(msg.sender == address(pool), "Invalid callback");
        require(mintCalled, "Callback not expected");
        
        mintCalled = false;
        
        if (amount0Owed > 0) token0.safeTransfer(msg.sender, amount0Owed);
        if (amount1Owed > 0) token1.safeTransfer(msg.sender, amount1Owed);
    }

    // ============ INTERNAL FUNCTIONS ============
    
    function _convertToShares(uint256 assets, Math.Rounding rounding) 
        internal 
        view 
        returns (uint256) 
    {
        uint256 supply = totalSupply();
        return supply == 0 ? assets : assets.mulDiv(supply, totalAssets(), rounding);
    }
    
    function _convertToAssets(uint256 shares, Math.Rounding rounding) 
        internal 
        view 
        returns (uint256) 
    {
        uint256 supply = totalSupply();
        return supply == 0 ? shares : shares.mulDiv(totalAssets(), supply, rounding);
    }
    
    function _convertToAssetValue(uint256 amount0, uint256 amount1) 
        internal 
        view 
        returns (uint256) 
    {
        // Simple conversion - could be enhanced with oracle pricing
        return amount0 + amount1; // Assuming 1:1 for simplicity
    }
    
    function _deployLiquidity() internal {
        uint256 balance0 = token0.balanceOf(address(this));
        uint256 balance1 = token1.balanceOf(address(this));
        
        if (balance0 == 0 && balance1 == 0) return;
        
        // Deploy liquidity according to current allocations
        for (uint256 i = 0; i < positionRanges.length; i++) {
            if (activeRanges[i]) {
                uint256 allocation = positionRanges[i].allocation;
                if (allocation > 0) {
                    uint256 amount0 = balance0 * allocation / MAX_BPS;
                    uint256 amount1 = balance1 * allocation / MAX_BPS;
                    
                    if (amount0 > 0 || amount1 > 0) {
                        _mintLiquidity(i, amount0, amount1);
                    }
                }
            }
        }
    }
    
    function _withdrawLiquidity(uint256 assets, address receiver) internal {
        // Calculate proportional withdrawal from all ranges
        uint256 totalAssets = totalAssets();
        
        for (uint256 i = 0; i < positionRanges.length; i++) {
            if (activeRanges[i] && positionRanges[i].liquidity > 0) {
                uint256 liquidity = positionRanges[i].liquidity * assets / totalAssets;
                if (liquidity > 0) {
                    _burnLiquidity(
                        positionRanges[i].tickLower,
                        positionRanges[i].tickUpper,
                        uint128(liquidity)
                    );
                }
            }
        }
        
        // Transfer available tokens
        uint256 available0 = token0.balanceOf(address(this));
        uint256 available1 = token1.balanceOf(address(this));
        
        if (available0 > 0) token0.safeTransfer(receiver, available0);
        if (available1 > 0) token1.safeTransfer(receiver, available1);
    }
    
    function _mintLiquidity(uint256 rangeId, uint256 amount0, uint256 amount1) internal {
        PositionRange storage range = positionRanges[rangeId];
        
        uint128 liquidity = LiquidityAmounts.getLiquidityForAmounts(
            _getSqrtRatioX96(),
            TickMath.getSqrtRatioAtTick(range.tickLower),
            TickMath.getSqrtRatioAtTick(range.tickUpper),
            amount0,
            amount1
        );
        
        if (liquidity > 0) {
            mintCalled = true;
            (uint256 actualAmount0, uint256 actualAmount1) = pool.mint(
                address(this),
                range.tickLower,
                range.tickUpper,
                liquidity,
                ""
            );
            
            range.liquidity += liquidity;
        }
    }
    
    function _burnLiquidity(int24 tickLower, int24 tickUpper, uint128 liquidity) internal {
        if (liquidity > 0) {
            pool.burn(tickLower, tickUpper, liquidity);
            pool.collect(
                address(this),
                tickLower,
                tickUpper,
                type(uint128).max,
                type(uint128).max
            );
        }
    }
    
    function _collectAllFees() internal returns (uint256 fees0, uint256 fees1) {
        for (uint256 i = 0; i < positionRanges.length; i++) {
            if (activeRanges[i] && positionRanges[i].liquidity > 0) {
                // Burn 0 liquidity to collect fees
                pool.burn(positionRanges[i].tickLower, positionRanges[i].tickUpper, 0);
                (uint256 collected0, uint256 collected1) = pool.collect(
                    address(this),
                    positionRanges[i].tickLower,
                    positionRanges[i].tickUpper,
                    type(uint128).max,
                    type(uint128).max
                );
                fees0 += collected0;
                fees1 += collected1;
            }
        }
    }
    
    function _distributeFees(uint256 fees0, uint256 fees1) internal {
        FeeConfiguration memory config = feeConfiguration;
        
        // Management fees
        uint256 mgmtFees0 = fees0 * config.managementFeeBps / MAX_BPS;
        uint256 mgmtFees1 = fees1 * config.managementFeeBps / MAX_BPS;
        
        // AI optimization fees
        uint256 aiFees0 = fees0 * config.aiOptimizationFeeBps / MAX_BPS;
        uint256 aiFees1 = fees1 * config.aiOptimizationFeeBps / MAX_BPS;
        
        // Strategist fees (from management fees)
        uint256 strategistFees0 = mgmtFees0 * config.strategistShareBps / MAX_BPS;
        uint256 strategistFees1 = mgmtFees1 * config.strategistShareBps / MAX_BPS;
        
        // Transfer fees
        if (mgmtFees0 > 0) {
            token0.safeTransfer(config.feeRecipient, mgmtFees0 - strategistFees0);
            if (strategistFees0 > 0) {
                token0.safeTransfer(config.strategist, strategistFees0);
            }
        }
        
        if (mgmtFees1 > 0) {
            token1.safeTransfer(config.feeRecipient, mgmtFees1 - strategistFees1);
            if (strategistFees1 > 0) {
                token1.safeTransfer(config.strategist, strategistFees1);
            }
        }
        
        if (aiFees0 > 0) token0.safeTransfer(aiManager, aiFees0);
        if (aiFees1 > 0) token1.safeTransfer(aiManager, aiFees1);
        
        emit FeeDistribution(mgmtFees0 + mgmtFees1, 0, aiFees0 + aiFees1);
    }
    
    function _compoundFees(uint256 fees0, uint256 fees1) internal returns (uint256 newLiquidity) {
        // After fee distribution, compound remaining fees
        uint256 remaining0 = token0.balanceOf(address(this));
        uint256 remaining1 = token1.balanceOf(address(this));
        
        if (remaining0 > 0 || remaining1 > 0) {
            _deployLiquidity();
            // Calculate new liquidity added
            newLiquidity = _estimateNewLiquidity(remaining0, remaining1);
        }
    }
    
    function _rebalanceLiquidity() internal {
        // Remove all liquidity
        for (uint256 i = 0; i < positionRanges.length; i++) {
            if (activeRanges[i] && positionRanges[i].liquidity > 0) {
                _burnLiquidity(
                    positionRanges[i].tickLower,
                    positionRanges[i].tickUpper,
                    uint128(positionRanges[i].liquidity)
                );
                positionRanges[i].liquidity = 0;
            }
        }
        
        // Re-deploy according to current allocations
        _deployLiquidity();
    }
    
    function _validateTWAP() internal view {
        if (!twapConfig.enabled) return;
        
        uint256 currentPrice = _getCurrentPrice();
        uint256 twapPrice = _getTWAPPrice();
        
        uint256 deviation = currentPrice * MAX_BPS / twapPrice;
        require(
            deviation <= twapConfig.maxDeviation + MAX_BPS &&
            deviation >= MAX_BPS - twapConfig.maxDeviation,
            "TWAP deviation"
        );
    }
    
    function _checkTWAP() external view {
        _validateTWAP();
    }
    
    function _getCurrentPrice() internal view returns (uint256) {
        (uint160 sqrtPriceX96, , , , , , ) = pool.slot0();
        return FullMath.mulDiv(uint256(sqrtPriceX96) ** 2, PRECISION, 2 ** 192);
    }
    
    function _getTWAPPrice() internal view returns (uint256) {
        uint32[] memory secondsAgos = new uint32[](2);
        secondsAgos[0] = twapConfig.interval;
        secondsAgos[1] = 0;
        
        (int56[] memory tickCumulatives, ) = pool.observe(secondsAgos);
        int24 avgTick = int24((tickCumulatives[1] - tickCumulatives[0]) / int32(twapConfig.interval));
        
        uint160 sqrtPriceX96 = TickMath.getSqrtRatioAtTick(avgTick);
        return FullMath.mulDiv(uint256(sqrtPriceX96) ** 2, PRECISION, 2 ** 192);
    }
    
    function _getSqrtRatioX96() internal view returns (uint160) {
        (uint160 sqrtPriceX96, , , , , , ) = pool.slot0();
        return sqrtPriceX96;
    }
    
    function _updateMetrics() internal {
        metrics.totalValueLocked = totalAssets();
        // Update other metrics as needed
    }
    
    function _getPendingFees() internal view returns (uint256 fees0, uint256 fees1) {
        // This would query pending fees from positions
        // Simplified implementation
        return (0, 0);
    }
    
    function _estimateNewLiquidity(uint256 amount0, uint256 amount1) internal view returns (uint256) {
        // Estimate new liquidity that would be created
        return amount0 + amount1; // Simplified
    }
    
    function _calculateAPRIncrease(uint256 newLiquidity) internal view returns (uint256) {
        // Calculate expected APR increase from compounding
        return newLiquidity * 100 / totalAssets(); // Simplified
    }
    
    function _calculateCurrentAPR() internal view returns (uint256) {
        // Calculate current APR based on fees and performance
        return 500; // 5% placeholder
    }
    
    function _getBenchmarkAPR() internal view returns (uint256) {
        // Get benchmark APR (e.g., from holding tokens)
        return 300; // 3% placeholder
    }
    
    function _calculateUtilizationScore() internal view returns (uint256) {
        // Calculate how well capital is utilized
        return 85; // 85% placeholder
    }
    
    function _calculatePerformanceScore() internal view returns (uint256) {
        // Calculate performance score
        return 90; // 90% placeholder
    }
    
    function _calculateRiskScore() internal view returns (uint256) {
        // Calculate risk score (higher = safer)
        return 75; // 75% placeholder
    }
    
    function _generateHealthAnalysis(uint256 score) internal pure returns (string memory) {
        if (score >= 80) return "Excellent health";
        if (score >= 60) return "Good health";
        if (score >= 40) return "Fair health";
        return "Poor health";
    }
    
    function _identifyRiskFactors() internal view returns (string[] memory factors) {
        factors = new string[](3);
        factors[0] = "Impermanent loss";
        factors[1] = "Market volatility";
        factors[2] = "Smart contract risk";
    }
    
    function _initializeGammaPositions() internal {
        // Initialize default dual positions for Gamma mode
        (, int24 currentTick, , , , , ) = pool.slot0();
        
        // Base position: wider range
        int24 baseLower = ((currentTick - 2000) / tickSpacing) * tickSpacing;
        int24 baseUpper = ((currentTick + 2000) / tickSpacing) * tickSpacing;
        
        // Limit position: narrower range
        int24 limitLower = ((currentTick - 200) / tickSpacing) * tickSpacing;
        int24 limitUpper = ((currentTick + 200) / tickSpacing) * tickSpacing;
        
        this.setDualPositionStrategy(baseLower, baseUpper, limitLower, limitUpper);
    }
    
    function _getAmountsForRange(uint256 rangeId) internal view returns (uint256 amount0, uint256 amount1) {
        PositionRange memory range = positionRanges[rangeId];
        if (range.liquidity > 0) {
            (amount0, amount1) = LiquidityAmounts.getAmountsForLiquidity(
                _getSqrtRatioX96(),
                TickMath.getSqrtRatioAtTick(range.tickLower),
                TickMath.getSqrtRatioAtTick(range.tickUpper),
                uint128(range.liquidity)
            );
        }
    }
}