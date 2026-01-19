// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@uniswap/v3-core/contracts/interfaces/IUniswapV3Pool.sol";
import "@uniswap/v3-core/contracts/interfaces/IUniswapV3Factory.sol";
import "./vendor/uniswap/libraries/TickMath.sol";
import "./vendor/uniswap/interfaces/INonfungiblePositionManager.sol";

/**
 * @title BinRebalancer
 * @notice Ultra-concentrated liquidity management with bin-based rebalancing
 * @dev Maintains positions within 1-5 bins of current price for maximum fee generation
 */
contract BinRebalancer is Ownable, ReentrancyGuard {
    
    // ============ CONSTANTS ============
    
    uint256 public constant MAX_BINS_FROM_PRICE = 5; // Maximum allowed bin drift
    uint256 public constant MIN_CONCENTRATION_LEVEL = 1; // Wide range
    uint256 public constant MAX_CONCENTRATION_LEVEL = 10; // Ultra-tight range
    uint256 public constant DEFAULT_CONCENTRATION = 5; // Balanced concentration
    
    // ============ ENUMS ============
    
    enum ConcentrationLevel {
        ULTRA_TIGHT,    // 1-2 bins (0.05-0.1% range)
        VERY_TIGHT,     // 2-3 bins (0.1-0.15% range)
        TIGHT,          // 3-4 bins (0.15-0.2% range)  
        MODERATE,       // 4-6 bins (0.2-0.3% range)
        WIDE            // 6-10 bins (0.3-0.5% range)
    }
    
    // ============ STRUCTS ============
    
    struct BinSettings {
        uint256 maxBinsFromPrice;        // Max bins before rebalance (1-5)
        ConcentrationLevel level;        // Concentration strategy
        uint256 concentrationRatio;      // Custom concentration (1-10)
        bool maintainConcentration;      // Enable bin-based rebalancing
        uint256 lastRebalanceTime;
        int24 lastRebalanceTick;         // Tick at last rebalance
    }
    
    struct BinPosition {
        int24 currentTick;               // Current pool price tick
        int24 positionTickLower;         // Position lower bound
        int24 positionTickUpper;         // Position upper bound
        uint256 ticksFromPrice;          // Ticks away from current price
        uint256 binsFromPrice;           // Bins away from current price
        bool inRange;                    // Is position currently in range
        int24 tickSpacing;               // Pool tick spacing
    }
    
    // ============ STATE VARIABLES ============
    
    INonfungiblePositionManager public immutable nonfungiblePositionManager;
    IUniswapV3Factory public immutable factory;
    
    mapping(uint256 => BinSettings) public positionBinSettings;
    mapping(address => bool) public authorizedKeepers;
    mapping(uint256 => uint256) public rebalanceCount;
    mapping(uint256 => uint256) public totalRebalanceCost;
    
    // Concentration level configurations
    mapping(ConcentrationLevel => uint256) public concentrationMultipliers;
    
    // ============ EVENTS ============
    
    event BinBasedRebalance(
        uint256 indexed oldTokenId,
        uint256 indexed newTokenId,
        uint256 binsFromPrice,
        int24 oldTickLower,
        int24 oldTickUpper,
        int24 newTickLower,
        int24 newTickUpper,
        ConcentrationLevel level
    );
    
    event BinSettingsUpdated(
        uint256 indexed tokenId,
        BinSettings settings
    );
    
    event ConcentrationLevelUpdated(
        ConcentrationLevel level,
        uint256 multiplier
    );
    
    // ============ MODIFIERS ============
    
    modifier onlyKeeper() {
        require(authorizedKeepers[msg.sender] || msg.sender == owner(), "Not authorized");
        _;
    }
    
    // ============ CONSTRUCTOR ============
    
    constructor(
        INonfungiblePositionManager _nonfungiblePositionManager,
        IUniswapV3Factory _factory
    ) {
        nonfungiblePositionManager = _nonfungiblePositionManager;
        factory = _factory;
        
        // Initialize concentration level multipliers
        _initializeConcentrationLevels();
    }
    
    // ============ BIN CALCULATION ============
    
    /**
     * @notice Calculate current bin position for a token
     * @param tokenId Position to analyze
     * @return binData Current bin position data
     */
    function calculateBinPosition(uint256 tokenId) public view returns (BinPosition memory) {
        (,, address token0, address token1, uint24 fee, int24 tickLower, int24 tickUpper,,,,,) = 
            nonfungiblePositionManager.positions(tokenId);
        
        IUniswapV3Pool pool = IUniswapV3Pool(factory.getPool(token0, token1, fee));
        require(address(pool) != address(0), "Pool does not exist");
        
        (, int24 currentTick,,,,,) = pool.slot0();
        int24 tickSpacing = pool.tickSpacing();
        
        uint256 ticksFromPrice = 0;
        bool inRange = currentTick >= tickLower && currentTick <= tickUpper;
        
        if (!inRange) {
            if (currentTick < tickLower) {
                ticksFromPrice = uint256(int256(tickLower - currentTick));
            } else {
                ticksFromPrice = uint256(int256(currentTick - tickUpper));
            }
        }
        
        uint256 binsFromPrice = ticksFromPrice / uint256(int256(tickSpacing));
        
        return BinPosition({
            currentTick: currentTick,
            positionTickLower: tickLower,
            positionTickUpper: tickUpper,
            ticksFromPrice: ticksFromPrice,
            binsFromPrice: binsFromPrice,
            inRange: inRange,
            tickSpacing: tickSpacing
        });
    }
    
    /**
     * @notice Check if position should be rebalanced based on bin drift
     * @param tokenId Position to check
     * @return shouldRebalance Whether position needs rebalancing
     */
    function shouldRebalance(uint256 tokenId) public view returns (bool) {
        BinSettings memory settings = positionBinSettings[tokenId];
        if (!settings.maintainConcentration) return false;
        
        BinPosition memory binData = calculateBinPosition(tokenId);
        
        // Rebalance if out of range or too many bins away
        return !binData.inRange || binData.binsFromPrice >= settings.maxBinsFromPrice;
    }
    
    // ============ REBALANCING ============
    
    /**
     * @notice Execute bin-based rebalance for concentrated liquidity
     * @param tokenId Position to rebalance
     */
    function executeRebalance(uint256 tokenId) external onlyKeeper nonReentrant {
        uint256 startGas = gasleft();
        require(shouldRebalance(tokenId), "Rebalance not needed");

        BinPosition memory binData = calculateBinPosition(tokenId);
        BinSettings storage settings = positionBinSettings[tokenId];
        
        // Calculate new concentrated range around current price
        (int24 newTickLower, int24 newTickUpper) = calculateConcentratedRange(
            binData.currentTick,
            binData.tickSpacing,
            settings.level,
            settings.concentrationRatio
        );
        
        // Close current position and collect liquidity
        (uint256 amount0, uint256 amount1) = _closePosition(tokenId);
        
        // Open new concentrated position
        uint256 newTokenId = _openConcentratedPosition(
            tokenId,
            newTickLower,
            newTickUpper,
            amount0,
            amount1
        );
        
        // Update tracking
        settings.lastRebalanceTime = block.timestamp;
        settings.lastRebalanceTick = binData.currentTick;
        rebalanceCount[tokenId]++;
        totalRebalanceCost[tokenId] += startGas - gasleft();

        emit BinBasedRebalance(
            tokenId,
            newTokenId,
            binData.binsFromPrice,
            binData.positionTickLower,
            binData.positionTickUpper,
            newTickLower,
            newTickUpper,
            settings.level
        );
    }
    
    /**
     * @notice Calculate new concentrated range based on concentration level
     * @param currentTick Current pool price tick
     * @param tickSpacing Pool tick spacing
     * @param level Concentration level
     * @param customRatio Custom concentration ratio (if level is custom)
     * @return tickLower New lower tick
     * @return tickUpper New upper tick
     */
    function calculateConcentratedRange(
        int24 currentTick,
        int24 tickSpacing,
        ConcentrationLevel level,
        uint256 customRatio
    ) public view returns (int24 tickLower, int24 tickUpper) {
        
        uint256 multiplier;
        if (customRatio > 0) {
            // Use custom ratio (1-10 scale)
            multiplier = 11 - customRatio; // Invert so 10 = tightest
        } else {
            // Use predefined concentration level
            multiplier = concentrationMultipliers[level];
        }
        
        // Calculate range based on concentration
        int24 halfRange = int24(tickSpacing * int256(multiplier));
        
        // Center around current price, aligned to tick spacing
        tickLower = ((currentTick - halfRange) / tickSpacing) * tickSpacing;
        tickUpper = ((currentTick + halfRange) / tickSpacing) * tickSpacing;
        
        // Ensure minimum range of one tick spacing
        if (tickUpper - tickLower < tickSpacing) {
            tickUpper = tickLower + tickSpacing;
        }
        
        // Ensure valid tick range
        require(tickLower < tickUpper, "Invalid tick range");
        require(tickLower >= TickMath.MIN_TICK, "Tick lower too low");
        require(tickUpper <= TickMath.MAX_TICK, "Tick upper too high");
    }
    
    // ============ CONFIGURATION ============
    
    /**
     * @notice Set bin settings for a position
     * @param tokenId Position to configure
     * @param settings Bin configuration
     */
    function setBinSettings(
        uint256 tokenId,
        BinSettings memory settings
    ) public onlyOwner {
        require(settings.maxBinsFromPrice >= 1 && settings.maxBinsFromPrice <= MAX_BINS_FROM_PRICE, "Invalid bin drift");
        require(settings.concentrationRatio <= MAX_CONCENTRATION_LEVEL, "Invalid concentration");
        
        // Set current time and tick for tracking
        settings.lastRebalanceTime = block.timestamp;
        
        // Get current tick for reference
        BinPosition memory binData = calculateBinPosition(tokenId);
        settings.lastRebalanceTick = binData.currentTick;
        
        positionBinSettings[tokenId] = settings;
        
        emit BinSettingsUpdated(tokenId, settings);
    }
    
    /**
     * @notice Set concentration level for a position using presets
     * @param tokenId Position to configure
     * @param level Concentration level preset
     * @param maxBins Maximum bins from price before rebalance
     */
    function setConcentrationLevel(
        uint256 tokenId,
        ConcentrationLevel level,
        uint256 maxBins
    ) external onlyOwner {
        require(maxBins >= 1 && maxBins <= MAX_BINS_FROM_PRICE, "Invalid bin drift");
        
        BinSettings memory settings = BinSettings({
            maxBinsFromPrice: maxBins,
            level: level,
            concentrationRatio: 0, // Use preset, not custom
            maintainConcentration: true,
            lastRebalanceTime: block.timestamp,
            lastRebalanceTick: 0
        });
        
        setBinSettings(tokenId, settings);
    }
    
    /**
     * @notice Update concentration level multiplier
     * @param level Concentration level to update
     * @param multiplier New multiplier value
     */
    function updateConcentrationMultiplier(
        ConcentrationLevel level,
        uint256 multiplier
    ) external onlyOwner {
        require(multiplier >= 1 && multiplier <= 10, "Invalid multiplier");
        concentrationMultipliers[level] = multiplier;
        
        emit ConcentrationLevelUpdated(level, multiplier);
    }
    
    /**
     * @notice Authorize keeper for rebalancing operations
     * @param keeper Address to authorize
     * @param authorized Authorization status
     */
    function setKeeperAuthorization(address keeper, bool authorized) external onlyOwner {
        authorizedKeepers[keeper] = authorized;
    }
    
    // ============ VIEW FUNCTIONS ============
    
    /**
     * @notice Get bin settings for a position
     * @param tokenId Position to query
     * @return settings Current bin settings
     */
    function getBinSettings(uint256 tokenId) external view returns (BinSettings memory) {
        return positionBinSettings[tokenId];
    }
    
    /**
     * @notice Get rebalance metrics for a position
     * @param tokenId Position to query
     * @return rebalanceCount_ Number of rebalances
     * @return totalCost Total rebalance cost
     * @return lastRebalance Last rebalance timestamp
     */
    function getRebalanceMetrics(uint256 tokenId) external view returns (
        uint256 rebalanceCount_,
        uint256 totalCost,
        uint256 lastRebalance
    ) {
        BinSettings memory settings = positionBinSettings[tokenId];
        return (
            rebalanceCount[tokenId],
            totalRebalanceCost[tokenId],
            settings.lastRebalanceTime
        );
    }
    
    /**
     * @notice Preview new range for a position
     * @param tokenId Position to preview
     * @return newTickLower Proposed lower tick
     * @return newTickUpper Proposed upper tick
     * @return binsFromPrice Current bins from price
     */
    function previewRebalance(uint256 tokenId) external view returns (
        int24 newTickLower,
        int24 newTickUpper,
        uint256 binsFromPrice
    ) {
        BinPosition memory binData = calculateBinPosition(tokenId);
        BinSettings memory settings = positionBinSettings[tokenId];
        
        (newTickLower, newTickUpper) = calculateConcentratedRange(
            binData.currentTick,
            binData.tickSpacing,
            settings.level,
            settings.concentrationRatio
        );
        
        return (newTickLower, newTickUpper, binData.binsFromPrice);
    }
    
    // ============ INTERNAL FUNCTIONS ============
    
    function _initializeConcentrationLevels() internal {
        concentrationMultipliers[ConcentrationLevel.ULTRA_TIGHT] = 1;  // Tightest
        concentrationMultipliers[ConcentrationLevel.VERY_TIGHT] = 2;
        concentrationMultipliers[ConcentrationLevel.TIGHT] = 3;
        concentrationMultipliers[ConcentrationLevel.MODERATE] = 5;
        concentrationMultipliers[ConcentrationLevel.WIDE] = 8;      // Widest
    }
    
    function _closePosition(uint256 tokenId) internal returns (uint256 amount0, uint256 amount1) {
        // Get position info
        (,, address token0, address token1, uint24 fee, int24 tickLower, int24 tickUpper, uint128 liquidity,,,,) = 
            nonfungiblePositionManager.positions(tokenId);
        
        // Decrease liquidity to 0
        if (liquidity > 0) {
            nonfungiblePositionManager.decreaseLiquidity(
                INonfungiblePositionManager.DecreaseLiquidityParams({
                    tokenId: tokenId,
                    liquidity: liquidity,
                    amount0Min: 0,
                    amount1Min: 0,
                    deadline: block.timestamp + 300
                })
            );
        }
        
        // Collect all tokens
        (amount0, amount1) = nonfungiblePositionManager.collect(
            INonfungiblePositionManager.CollectParams({
                tokenId: tokenId,
                recipient: address(this),
                amount0Max: type(uint128).max,
                amount1Max: type(uint128).max
            })
        );
    }
    
    function _openConcentratedPosition(
        uint256 oldTokenId,
        int24 tickLower,
        int24 tickUpper,
        uint256 amount0,
        uint256 amount1
    ) internal returns (uint256 newTokenId) {
        // Get token info from old position
        (,, address token0, address token1, uint24 fee,,,,,,,) = 
            nonfungiblePositionManager.positions(oldTokenId);
        
        // Mint new position with concentrated range
        (newTokenId,,,) = nonfungiblePositionManager.mint(
            INonfungiblePositionManager.MintParams({
                token0: token0,
                token1: token1,
                fee: fee,
                tickLower: tickLower,
                tickUpper: tickUpper,
                amount0Desired: amount0,
                amount1Desired: amount1,
                amount0Min: 0,
                amount1Min: 0,
                recipient: address(this),
                deadline: block.timestamp + 300
            })
        );
        
        // Copy bin settings to new position
        positionBinSettings[newTokenId] = positionBinSettings[oldTokenId];
        
        // Clear old position settings
        delete positionBinSettings[oldTokenId];
    }
}