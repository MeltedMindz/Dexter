// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/utils/math/Math.sol";
import "@uniswap/v3-core/contracts/interfaces/IUniswapV3Pool.sol";
import "@uniswap/v3-core/contracts/libraries/TickMath.sol";

import "../vaults/IDexterVault.sol";

/// @title StrategyManager - Hybrid strategy management combining Gamma-style manual control with AI optimization
/// @notice Enables seamless transitions between manual, AI-assisted, and fully automated strategies
contract StrategyManager is Ownable, ReentrancyGuard {
    using Math for uint256;

    // ============ ENUMS ============
    
    enum StrategyType {
        GAMMA_CONSERVATIVE,    // Gamma-style dual position with wide ranges
        GAMMA_BALANCED,        // Gamma-style with moderate ranges
        GAMMA_AGGRESSIVE,      // Gamma-style with narrow ranges
        AI_CONSERVATIVE,       // AI-managed conservative approach
        AI_BALANCED,           // AI-managed balanced approach
        AI_AGGRESSIVE,         // AI-managed aggressive approach
        HYBRID_MANUAL_AI,      // Manual base + AI limit positions
        HYBRID_AI_MANUAL,      // AI base + manual limit positions
        CUSTOM                 // Fully custom strategy
    }
    
    enum RebalanceReason {
        SCHEDULED,            // Regular scheduled rebalance
        PRICE_MOVEMENT,       // Price moved outside range
        PERFORMANCE_TRIGGER,  // Performance metrics triggered rebalance
        AI_RECOMMENDATION,    // AI recommended rebalance
        MANUAL_REQUEST,       // User manually requested
        EMERGENCY             // Emergency rebalance
    }
    
    enum StrategyStatus {
        ACTIVE,
        PAUSED,
        MIGRATING,
        DEPRECATED
    }

    // ============ STRUCTS ============
    
    struct StrategyConfig {
        StrategyType strategyType;
        StrategyStatus status;
        bool aiEnabled;
        bool autoRebalanceEnabled;
        uint256 rebalanceThreshold;     // Price movement threshold for rebalance (bps)
        uint256 performanceThreshold;   // Performance threshold for rebalance
        uint32 rebalanceInterval;       // Minimum time between rebalances
        uint256 maxSlippage;            // Maximum slippage tolerance (bps)
        int24 baseRangeWidth;           // Width of base position (ticks)
        int24 limitRangeWidth;          // Width of limit position (ticks)
        uint256 baseAllocation;         // Allocation to base position (bps)
        uint256 limitAllocation;        // Allocation to limit position (bps)
        address customImplementation;   // Custom strategy contract
    }
    
    struct PositionRange {
        int24 tickLower;
        int24 tickUpper;
        uint256 allocation;             // Percentage allocation (bps)
        bool isActive;
        uint256 liquidity;
        uint256 lastRebalance;
        string name;                    // Human-readable name
    }
    
    struct StrategyMetrics {
        uint256 totalRebalances;
        uint256 successfulRebalances;
        uint256 avgGasCost;
        uint256 totalFeesEarned;
        uint256 impermanentLoss;
        uint256 sharpeRatio;
        uint256 maxDrawdown;
        uint256 winRate;               // Percentage of profitable rebalances
        uint256 lastPerformanceUpdate;
    }
    
    struct RebalanceEvent {
        uint256 timestamp;
        RebalanceReason reason;
        int24 oldTickLower;
        int24 oldTickUpper;
        int24 newTickLower;
        int24 newTickUpper;
        uint256 gasCost;
        int256 pnl;                    // Profit/loss from rebalance
        bool success;
    }
    
    struct AIRecommendation {
        StrategyType recommendedStrategy;
        PositionRange[] recommendedRanges;
        uint256 confidenceScore;       // 0-10000 (0-100%)
        uint256 expectedAPR;
        uint256 expectedVolatility;
        string reasoning;
        uint256 timestamp;
        bool applied;
    }
    
    struct StrategyTransition {
        StrategyType fromStrategy;
        StrategyType toStrategy;
        uint256 startTime;
        uint256 estimatedDuration;
        uint256 progress;              // 0-10000 (0-100%)
        bool isActive;
        string reason;
    }

    // ============ STATE VARIABLES ============
    
    mapping(address => StrategyConfig) public vaultStrategies;
    mapping(address => StrategyMetrics) public strategyMetrics;
    mapping(address => PositionRange[]) public vaultRanges;
    mapping(address => RebalanceEvent[]) public rebalanceHistory;
    mapping(address => AIRecommendation) public latestAIRecommendations;
    mapping(address => StrategyTransition) public activeTransitions;
    
    mapping(StrategyType => StrategyConfig) public defaultConfigs;
    mapping(address => bool) public authorizedVaults;
    mapping(address => bool) public authorizedStrategists;
    mapping(address => uint256) public lastRebalanceTime;
    
    address public aiOracle;
    address public performanceOracle;
    uint256 public globalRebalanceInterval = 1 hours;
    uint256 public emergencyRebalanceThreshold = 2000; // 20%
    bool public globalPause;

    // ============ EVENTS ============
    
    event StrategyConfigured(address indexed vault, StrategyType strategyType, StrategyConfig config);
    event StrategyTransitionStarted(address indexed vault, StrategyType from, StrategyType to, string reason);
    event StrategyTransitionCompleted(address indexed vault, StrategyType newStrategy);
    event RebalanceExecuted(address indexed vault, RebalanceReason reason, bool success, uint256 gasCost);
    event AIRecommendationReceived(address indexed vault, StrategyType recommended, uint256 confidence);
    event PositionRangeUpdated(address indexed vault, uint256 rangeIndex, int24 tickLower, int24 tickUpper);
    event PerformanceMetricsUpdated(address indexed vault, uint256 sharpeRatio, uint256 winRate);
    event EmergencyRebalanceTriggered(address indexed vault, uint256 priceDeviation);

    // ============ MODIFIERS ============
    
    modifier onlyAuthorizedVault() {
        require(authorizedVaults[msg.sender], "Unauthorized vault");
        _;
    }
    
    modifier onlyStrategist() {
        require(authorizedStrategists[msg.sender] || msg.sender == owner(), "Not authorized strategist");
        _;
    }
    
    modifier notPaused() {
        require(!globalPause, "Globally paused");
        _;
    }
    
    modifier validStrategy(StrategyType strategy) {
        require(strategy <= StrategyType.CUSTOM, "Invalid strategy type");
        _;
    }

    // ============ CONSTRUCTOR ============
    
    constructor(address _aiOracle, address _performanceOracle) {
        aiOracle = _aiOracle;
        performanceOracle = _performanceOracle;
        
        // Initialize default strategy configurations
        _initializeDefaultConfigs();
        
        // Set owner as authorized strategist
        authorizedStrategists[owner()] = true;
    }

    // ============ VAULT AUTHORIZATION ============
    
    function authorizeVault(address vault, bool authorized) external onlyOwner {
        authorizedVaults[vault] = authorized;
        
        if (authorized && vaultStrategies[vault].strategyType == StrategyType(0)) {
            // Initialize with default strategy
            _initializeVaultStrategy(vault, StrategyType.GAMMA_BALANCED);
        }
    }
    
    function setStrategist(address strategist, bool authorized) external onlyOwner {
        authorizedStrategists[strategist] = authorized;
    }

    // ============ STRATEGY MANAGEMENT ============
    
    function configureStrategy(
        address vault,
        StrategyType strategyType,
        StrategyConfig calldata config
    ) 
        external 
        onlyStrategist 
        validStrategy(strategyType) 
    {
        require(authorizedVaults[vault], "Vault not authorized");
        require(config.baseAllocation + config.limitAllocation <= 10000, "Invalid allocations");
        require(config.maxSlippage <= 1000, "Slippage too high"); // Max 10%
        
        vaultStrategies[vault] = config;
        vaultStrategies[vault].strategyType = strategyType;
        
        // Initialize position ranges based on strategy type
        _initializePositionRanges(vault, strategyType, config);
        
        emit StrategyConfigured(vault, strategyType, config);
    }
    
    function initiateStrategyTransition(
        address vault,
        StrategyType newStrategy,
        string calldata reason
    ) 
        external 
        onlyStrategist 
        validStrategy(newStrategy) 
    {
        require(!activeTransitions[vault].isActive, "Transition already active");
        
        StrategyType currentStrategy = vaultStrategies[vault].strategyType;
        require(currentStrategy != newStrategy, "Same strategy");
        
        activeTransitions[vault] = StrategyTransition({
            fromStrategy: currentStrategy,
            toStrategy: newStrategy,
            startTime: block.timestamp,
            estimatedDuration: _estimateTransitionDuration(currentStrategy, newStrategy),
            progress: 0,
            isActive: true,
            reason: reason
        });
        
        emit StrategyTransitionStarted(vault, currentStrategy, newStrategy, reason);
        
        // Start transition process
        _executeStrategyTransition(vault);
    }
    
    function applyAIRecommendation(address vault) external onlyStrategist {
        AIRecommendation storage recommendation = latestAIRecommendations[vault];
        require(recommendation.timestamp > 0, "No AI recommendation");
        require(!recommendation.applied, "Already applied");
        require(recommendation.confidenceScore >= 7000, "Confidence too low"); // 70%
        
        // Apply the recommendation
        if (recommendation.recommendedStrategy != vaultStrategies[vault].strategyType) {
            initiateStrategyTransition(vault, recommendation.recommendedStrategy, "AI Recommendation");
        }
        
        // Update position ranges
        _updateRangesFromAI(vault, recommendation.recommendedRanges);
        
        recommendation.applied = true;
    }

    // ============ REBALANCING ============
    
    function executeRebalance(
        address vault,
        RebalanceReason reason
    ) 
        external 
        onlyStrategist 
        nonReentrant 
        notPaused 
        returns (bool success) 
    {
        require(authorizedVaults[vault], "Vault not authorized");
        require(_canRebalance(vault), "Rebalance not allowed");
        
        uint256 gasBefore = gasleft();
        
        try this._performRebalance(vault, reason) {
            success = true;
        } catch Error(string memory errorReason) {
            success = false;
            // Log error but don't revert
        }
        
        uint256 gasUsed = gasBefore - gasleft();
        
        // Record rebalance event
        _recordRebalanceEvent(vault, reason, success, gasUsed);
        
        // Update metrics
        _updateStrategyMetrics(vault, success, gasUsed);
        
        lastRebalanceTime[vault] = block.timestamp;
        
        emit RebalanceExecuted(vault, reason, success, gasUsed);
    }
    
    function autoRebalanceCheck(address vault) 
        external 
        view 
        returns (bool shouldRebalance, RebalanceReason reason) 
    {
        StrategyConfig memory config = vaultStrategies[vault];
        
        if (!config.autoRebalanceEnabled || !_canRebalance(vault)) {
            return (false, RebalanceReason.SCHEDULED);
        }
        
        // Check price movement
        if (_checkPriceMovement(vault, config.rebalanceThreshold)) {
            return (true, RebalanceReason.PRICE_MOVEMENT);
        }
        
        // Check performance triggers
        if (_checkPerformanceTriggers(vault, config.performanceThreshold)) {
            return (true, RebalanceReason.PERFORMANCE_TRIGGER);
        }
        
        // Check scheduled rebalance
        if (block.timestamp >= lastRebalanceTime[vault] + config.rebalanceInterval) {
            return (true, RebalanceReason.SCHEDULED);
        }
        
        // Check AI recommendations
        if (_checkAIRebalanceRecommendation(vault)) {
            return (true, RebalanceReason.AI_RECOMMENDATION);
        }
        
        return (false, RebalanceReason.SCHEDULED);
    }
    
    function emergencyRebalance(address vault, string calldata reason) 
        external 
        onlyStrategist 
    {
        require(authorizedVaults[vault], "Vault not authorized");
        
        // Force rebalance regardless of cooldowns
        this.executeRebalance(vault, RebalanceReason.EMERGENCY);
        
        emit EmergencyRebalanceTriggered(vault, emergencyRebalanceThreshold);
    }

    // ============ AI INTEGRATION ============
    
    function updateAIRecommendation(
        address vault,
        AIRecommendation calldata recommendation
    ) 
        external 
        onlyOwner 
    {
        require(recommendation.confidenceScore <= 10000, "Invalid confidence");
        
        latestAIRecommendations[vault] = recommendation;
        
        emit AIRecommendationReceived(vault, recommendation.recommendedStrategy, recommendation.confidenceScore);
        
        // Auto-apply if confidence is very high and auto-mode is enabled
        StrategyConfig memory config = vaultStrategies[vault];
        if (config.aiEnabled && recommendation.confidenceScore >= 9000) { // 90%
            try this.applyAIRecommendation(vault) {
                // AI recommendation applied successfully
            } catch {
                // Failed to apply, but don't revert
            }
        }
    }
    
    function setAIOracle(address _aiOracle) external onlyOwner {
        aiOracle = _aiOracle;
    }

    // ============ POSITION RANGE MANAGEMENT ============
    
    function updatePositionRange(
        address vault,
        uint256 rangeIndex,
        int24 tickLower,
        int24 tickUpper,
        uint256 allocation
    ) 
        external 
        onlyStrategist 
    {
        require(rangeIndex < vaultRanges[vault].length, "Invalid range index");
        require(tickLower < tickUpper, "Invalid tick range");
        require(allocation <= 10000, "Invalid allocation");
        
        PositionRange storage range = vaultRanges[vault][rangeIndex];
        range.tickLower = tickLower;
        range.tickUpper = tickUpper;
        range.allocation = allocation;
        range.lastRebalance = block.timestamp;
        
        emit PositionRangeUpdated(vault, rangeIndex, tickLower, tickUpper);
    }
    
    function addPositionRange(
        address vault,
        int24 tickLower,
        int24 tickUpper,
        uint256 allocation,
        string calldata name
    ) 
        external 
        onlyStrategist 
        returns (uint256 rangeIndex) 
    {
        require(vaultRanges[vault].length < 10, "Too many ranges");
        require(tickLower < tickUpper, "Invalid tick range");
        require(allocation <= 10000, "Invalid allocation");
        
        rangeIndex = vaultRanges[vault].length;
        
        vaultRanges[vault].push(PositionRange({
            tickLower: tickLower,
            tickUpper: tickUpper,
            allocation: allocation,
            isActive: true,
            liquidity: 0,
            lastRebalance: block.timestamp,
            name: name
        }));
        
        emit PositionRangeUpdated(vault, rangeIndex, tickLower, tickUpper);
    }
    
    function removePositionRange(address vault, uint256 rangeIndex) 
        external 
        onlyStrategist 
    {
        require(rangeIndex < vaultRanges[vault].length, "Invalid range index");
        require(vaultRanges[vault].length > 1, "Cannot remove last range");
        
        vaultRanges[vault][rangeIndex].isActive = false;
    }

    // ============ PERFORMANCE TRACKING ============
    
    function updatePerformanceMetrics(
        address vault,
        uint256 sharpeRatio,
        uint256 winRate,
        uint256 totalFees,
        uint256 impermanentLoss
    ) 
        external 
        onlyOwner 
    {
        StrategyMetrics storage metrics = strategyMetrics[vault];
        metrics.sharpeRatio = sharpeRatio;
        metrics.winRate = winRate;
        metrics.totalFeesEarned = totalFees;
        metrics.impermanentLoss = impermanentLoss;
        metrics.lastPerformanceUpdate = block.timestamp;
        
        emit PerformanceMetricsUpdated(vault, sharpeRatio, winRate);
    }
    
    function getStrategyPerformance(address vault) 
        external 
        view 
        returns (
            uint256 totalRebalances,
            uint256 successRate,
            uint256 avgGasCost,
            uint256 sharpeRatio,
            uint256 winRate
        ) 
    {
        StrategyMetrics memory metrics = strategyMetrics[vault];
        
        totalRebalances = metrics.totalRebalances;
        successRate = totalRebalances > 0 ? 
            (metrics.successfulRebalances * 10000) / totalRebalances : 0;
        avgGasCost = metrics.avgGasCost;
        sharpeRatio = metrics.sharpeRatio;
        winRate = metrics.winRate;
    }

    // ============ VIEW FUNCTIONS ============
    
    function getVaultStrategy(address vault) 
        external 
        view 
        returns (StrategyConfig memory) 
    {
        return vaultStrategies[vault];
    }
    
    function getVaultRanges(address vault) 
        external 
        view 
        returns (PositionRange[] memory) 
    {
        return vaultRanges[vault];
    }
    
    function getRebalanceHistory(address vault, uint256 limit) 
        external 
        view 
        returns (RebalanceEvent[] memory) 
    {
        uint256 length = rebalanceHistory[vault].length;
        uint256 returnLength = length > limit ? limit : length;
        
        RebalanceEvent[] memory events = new RebalanceEvent[](returnLength);
        
        for (uint256 i = 0; i < returnLength; i++) {
            events[i] = rebalanceHistory[vault][length - 1 - i]; // Most recent first
        }
        
        return events;
    }
    
    function getActiveTransition(address vault) 
        external 
        view 
        returns (StrategyTransition memory) 
    {
        return activeTransitions[vault];
    }
    
    function getLatestAIRecommendation(address vault) 
        external 
        view 
        returns (AIRecommendation memory) 
    {
        return latestAIRecommendations[vault];
    }

    // ============ ADMIN FUNCTIONS ============
    
    function setGlobalPause(bool paused) external onlyOwner {
        globalPause = paused;
    }
    
    function setGlobalRebalanceInterval(uint256 interval) external onlyOwner {
        require(interval >= 1 minutes, "Interval too short");
        globalRebalanceInterval = interval;
    }
    
    function setEmergencyThreshold(uint256 threshold) external onlyOwner {
        require(threshold <= 5000, "Threshold too high"); // Max 50%
        emergencyRebalanceThreshold = threshold;
    }

    // ============ INTERNAL FUNCTIONS ============
    
    function _initializeDefaultConfigs() internal {
        // Gamma Conservative
        defaultConfigs[StrategyType.GAMMA_CONSERVATIVE] = StrategyConfig({
            strategyType: StrategyType.GAMMA_CONSERVATIVE,
            status: StrategyStatus.ACTIVE,
            aiEnabled: false,
            autoRebalanceEnabled: true,
            rebalanceThreshold: 1000,  // 10%
            performanceThreshold: 500, // 5%
            rebalanceInterval: 24 hours,
            maxSlippage: 100,          // 1%
            baseRangeWidth: 4000,      // Wide range
            limitRangeWidth: 400,      // Narrow range
            baseAllocation: 8000,      // 80%
            limitAllocation: 2000,     // 20%
            customImplementation: address(0)
        });
        
        // AI Aggressive
        defaultConfigs[StrategyType.AI_AGGRESSIVE] = StrategyConfig({
            strategyType: StrategyType.AI_AGGRESSIVE,
            status: StrategyStatus.ACTIVE,
            aiEnabled: true,
            autoRebalanceEnabled: true,
            rebalanceThreshold: 200,   // 2%
            performanceThreshold: 100, // 1%
            rebalanceInterval: 1 hours,
            maxSlippage: 300,          // 3%
            baseRangeWidth: 1000,      // Narrow range
            limitRangeWidth: 200,      // Very narrow range
            baseAllocation: 6000,      // 60%
            limitAllocation: 4000,     // 40%
            customImplementation: address(0)
        });
        
        // Add other default configurations...
    }
    
    function _initializeVaultStrategy(address vault, StrategyType strategyType) internal {
        vaultStrategies[vault] = defaultConfigs[strategyType];
        _initializePositionRanges(vault, strategyType, defaultConfigs[strategyType]);
    }
    
    function _initializePositionRanges(
        address vault,
        StrategyType strategyType,
        StrategyConfig memory config
    ) internal {
        // Clear existing ranges
        delete vaultRanges[vault];
        
        if (strategyType == StrategyType.GAMMA_CONSERVATIVE ||
            strategyType == StrategyType.GAMMA_BALANCED ||
            strategyType == StrategyType.GAMMA_AGGRESSIVE) {
            
            // Gamma-style dual positions
            _addGammaStyleRanges(vault, config);
            
        } else if (strategyType >= StrategyType.AI_CONSERVATIVE &&
                   strategyType <= StrategyType.AI_AGGRESSIVE) {
            
            // AI-managed ranges
            _addAIStyleRanges(vault, config);
            
        } else if (strategyType == StrategyType.HYBRID_MANUAL_AI ||
                   strategyType == StrategyType.HYBRID_AI_MANUAL) {
            
            // Hybrid ranges
            _addHybridRanges(vault, config);
        }
    }
    
    function _addGammaStyleRanges(address vault, StrategyConfig memory config) internal {
        // Get current tick
        IUniswapV3Pool pool = IUniswapV3Pool(IDexterVault(vault).pool());
        (, int24 currentTick, , , , , ) = pool.slot0();
        int24 tickSpacing = pool.tickSpacing();
        
        // Base position (wider range)
        int24 baseLower = ((currentTick - config.baseRangeWidth / 2) / tickSpacing) * tickSpacing;
        int24 baseUpper = ((currentTick + config.baseRangeWidth / 2) / tickSpacing) * tickSpacing;
        
        vaultRanges[vault].push(PositionRange({
            tickLower: baseLower,
            tickUpper: baseUpper,
            allocation: config.baseAllocation,
            isActive: true,
            liquidity: 0,
            lastRebalance: block.timestamp,
            name: "Base Position"
        }));
        
        // Limit position (narrower range)
        int24 limitLower = ((currentTick - config.limitRangeWidth / 2) / tickSpacing) * tickSpacing;
        int24 limitUpper = ((currentTick + config.limitRangeWidth / 2) / tickSpacing) * tickSpacing;
        
        vaultRanges[vault].push(PositionRange({
            tickLower: limitLower,
            tickUpper: limitUpper,
            allocation: config.limitAllocation,
            isActive: true,
            liquidity: 0,
            lastRebalance: block.timestamp,
            name: "Limit Position"
        }));
    }
    
    function _addAIStyleRanges(address vault, StrategyConfig memory config) internal {
        // AI-determined ranges would be more dynamic
        // For now, use a single adaptive range
        IUniswapV3Pool pool = IUniswapV3Pool(IDexterVault(vault).pool());
        (, int24 currentTick, , , , , ) = pool.slot0();
        int24 tickSpacing = pool.tickSpacing();
        
        int24 rangeWidth = config.baseRangeWidth;
        int24 tickLower = ((currentTick - rangeWidth / 2) / tickSpacing) * tickSpacing;
        int24 tickUpper = ((currentTick + rangeWidth / 2) / tickSpacing) * tickSpacing;
        
        vaultRanges[vault].push(PositionRange({
            tickLower: tickLower,
            tickUpper: tickUpper,
            allocation: 10000, // 100%
            isActive: true,
            liquidity: 0,
            lastRebalance: block.timestamp,
            name: "AI Managed Position"
        }));
    }
    
    function _addHybridRanges(address vault, StrategyConfig memory config) internal {
        // Combination of manual and AI ranges
        _addGammaStyleRanges(vault, config);
        
        // Add AI-optimized range
        IUniswapV3Pool pool = IUniswapV3Pool(IDexterVault(vault).pool());
        (, int24 currentTick, , , , , ) = pool.slot0();
        int24 tickSpacing = pool.tickSpacing();
        
        int24 aiRangeWidth = config.limitRangeWidth / 2;
        int24 aiLower = ((currentTick - aiRangeWidth / 2) / tickSpacing) * tickSpacing;
        int24 aiUpper = ((currentTick + aiRangeWidth / 2) / tickSpacing) * tickSpacing;
        
        vaultRanges[vault].push(PositionRange({
            tickLower: aiLower,
            tickUpper: aiUpper,
            allocation: 1000, // 10%
            isActive: true,
            liquidity: 0,
            lastRebalance: block.timestamp,
            name: "AI Optimization Range"
        }));
    }
    
    function _canRebalance(address vault) internal view returns (bool) {
        StrategyConfig memory config = vaultStrategies[vault];
        
        if (config.status != StrategyStatus.ACTIVE) return false;
        if (globalPause) return false;
        
        uint256 timeSinceLastRebalance = block.timestamp - lastRebalanceTime[vault];
        if (timeSinceLastRebalance < globalRebalanceInterval) return false;
        
        return true;
    }
    
    function _performRebalance(address vault, RebalanceReason reason) external {
        // This would be called internally and would interface with the vault
        // to actually execute the rebalance
        require(msg.sender == address(this), "Internal only");
        
        // Implementation would depend on vault interface
        // For now, assume success
    }
    
    function _executeStrategyTransition(address vault) internal {
        // Implementation for gradual strategy transition
        StrategyTransition storage transition = activeTransitions[vault];
        
        // For now, complete immediately
        transition.progress = 10000; // 100%
        transition.isActive = false;
        
        // Apply new strategy
        StrategyConfig memory newConfig = defaultConfigs[transition.toStrategy];
        vaultStrategies[vault] = newConfig;
        _initializePositionRanges(vault, transition.toStrategy, newConfig);
        
        emit StrategyTransitionCompleted(vault, transition.toStrategy);
    }
    
    function _updateRangesFromAI(address vault, PositionRange[] memory aiRanges) internal {
        // Update ranges based on AI recommendations
        for (uint256 i = 0; i < aiRanges.length && i < vaultRanges[vault].length; i++) {
            vaultRanges[vault][i].tickLower = aiRanges[i].tickLower;
            vaultRanges[vault][i].tickUpper = aiRanges[i].tickUpper;
            vaultRanges[vault][i].allocation = aiRanges[i].allocation;
            vaultRanges[vault][i].lastRebalance = block.timestamp;
        }
    }
    
    function _recordRebalanceEvent(
        address vault,
        RebalanceReason reason,
        bool success,
        uint256 gasUsed
    ) internal {
        rebalanceHistory[vault].push(RebalanceEvent({
            timestamp: block.timestamp,
            reason: reason,
            oldTickLower: 0, // Would be populated with actual values
            oldTickUpper: 0,
            newTickLower: 0,
            newTickUpper: 0,
            gasCost: gasUsed,
            pnl: 0, // Would be calculated
            success: success
        }));
    }
    
    function _updateStrategyMetrics(address vault, bool success, uint256 gasUsed) internal {
        StrategyMetrics storage metrics = strategyMetrics[vault];
        
        metrics.totalRebalances++;
        if (success) {
            metrics.successfulRebalances++;
        }
        
        // Update average gas cost
        if (metrics.totalRebalances == 1) {
            metrics.avgGasCost = gasUsed;
        } else {
            metrics.avgGasCost = (metrics.avgGasCost * (metrics.totalRebalances - 1) + gasUsed) / metrics.totalRebalances;
        }
    }
    
    function _checkPriceMovement(address vault, uint256 threshold) internal view returns (bool) {
        // Check if price has moved beyond threshold
        // Implementation would check current price vs range bounds
        return false; // Placeholder
    }
    
    function _checkPerformanceTriggers(address vault, uint256 threshold) internal view returns (bool) {
        // Check performance-based triggers
        return false; // Placeholder
    }
    
    function _checkAIRebalanceRecommendation(address vault) internal view returns (bool) {
        // Check if AI recommends rebalance
        AIRecommendation memory rec = latestAIRecommendations[vault];
        return rec.timestamp > lastRebalanceTime[vault] && 
               rec.confidenceScore >= 8000 && // 80%
               !rec.applied;
    }
    
    function _estimateTransitionDuration(StrategyType from, StrategyType to) 
        internal 
        pure 
        returns (uint256) 
    {
        // Estimate transition duration based on strategy types
        return 1 hours; // Default 1 hour
    }
}