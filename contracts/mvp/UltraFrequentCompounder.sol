// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@uniswap/v3-periphery/contracts/interfaces/INonfungiblePositionManager.sol";

/**
 * @title UltraFrequentCompounder
 * @notice Specialized contract for ultra-high frequency compounding on Base chain
 * @dev Optimized for 5-minute compound intervals with $0.50 minimum thresholds
 */
contract UltraFrequentCompounder is Ownable, ReentrancyGuard {
    
    // ============ CONSTANTS ============
    
    uint256 public constant ULTRA_FREQUENT_INTERVAL = 5 minutes;
    uint256 public constant MIN_COMPOUND_THRESHOLD = 5e17; // $0.50 in wei
    uint256 public constant MAX_BATCH_SIZE = 100; // Max positions per batch on Base
    uint256 public constant GAS_LIMIT_PER_COMPOUND = 150000; // Base optimized
    
    // ============ STATE VARIABLES ============
    
    INonfungiblePositionManager public immutable nonfungiblePositionManager;
    
    struct CompoundSettings {
        uint256 minFeeThreshold;     // Minimum fees to trigger compound
        uint256 maxTimeInterval;     // Maximum time between compounds
        uint256 lastCompoundTime;
        bool enabled;
        uint256 gasLimit;           // Gas limit for this position
    }
    
    mapping(uint256 => CompoundSettings) public compoundSettings;
    mapping(address => bool) public authorizedKeepers;
    mapping(uint256 => uint256) public compoundCount;
    mapping(uint256 => uint256) public totalFeesCompounded;
    
    // Performance tracking for Base chain optimization
    uint256 public totalCompoundsExecuted;
    uint256 public totalGasUsed;
    uint256 public averageGasPerCompound;
    
    // ============ EVENTS ============
    
    event UltraFrequentCompound(
        uint256 indexed tokenId,
        uint256 amount0,
        uint256 amount1,
        uint256 gasUsed,
        uint256 timestamp
    );
    
    event BatchCompoundExecuted(
        uint256[] tokenIds,
        uint256 successCount,
        uint256 totalGasUsed
    );
    
    event CompoundSettingsUpdated(
        uint256 indexed tokenId,
        CompoundSettings settings
    );
    
    event KeeperAuthorized(address indexed keeper, bool authorized);
    
    // ============ MODIFIERS ============
    
    modifier onlyKeeper() {
        require(authorizedKeepers[msg.sender] || msg.sender == owner(), "Not authorized");
        _;
    }
    
    // ============ CONSTRUCTOR ============
    
    constructor(INonfungiblePositionManager _nonfungiblePositionManager) {
        nonfungiblePositionManager = _nonfungiblePositionManager;
    }
    
    // ============ COMPOUND LOGIC ============
    
    /**
     * @notice Check if position should be compounded (ultra-frequent logic)
     * @param tokenId Position to check
     * @return should Whether position needs compounding
     */
    function shouldCompound(uint256 tokenId) external view returns (bool) {
        CompoundSettings memory settings = compoundSettings[tokenId];
        if (!settings.enabled) return false;
        
        // Ultra-frequent time check (5 minutes)
        uint256 timeSinceLastCompound = block.timestamp - settings.lastCompoundTime;
        if (timeSinceLastCompound >= ULTRA_FREQUENT_INTERVAL) {
            return true;
        }
        
        // Fee threshold check (as low as $0.50 on Base)
        uint256 unclaimedFees = _getUnclaimedFeesValue(tokenId);
        return unclaimedFees >= settings.minFeeThreshold;
    }
    
    /**
     * @notice Execute ultra-frequent compound for single position
     * @param tokenId Position to compound
     */
    function executeCompound(uint256 tokenId) external onlyKeeper nonReentrant {
        require(shouldCompound(tokenId), "Compound not needed");
        
        uint256 gasStart = gasleft();
        CompoundSettings storage settings = compoundSettings[tokenId];
        
        // Collect fees with optimized gas usage
        (uint256 amount0, uint256 amount1) = nonfungiblePositionManager.collect(
            INonfungiblePositionManager.CollectParams({
                tokenId: tokenId,
                recipient: address(this),
                amount0Max: type(uint128).max,
                amount1Max: type(uint128).max
            })
        );
        
        // Only increase liquidity if we collected meaningful fees
        if (amount0 > 0 || amount1 > 0) {
            // Increase liquidity with collected fees
            nonfungiblePositionManager.increaseLiquidity(
                INonfungiblePositionManager.IncreaseLiquidityParams({
                    tokenId: tokenId,
                    amount0Desired: amount0,
                    amount1Desired: amount1,
                    amount0Min: 0, // Accept any amount for ultra-frequent compounds
                    amount1Min: 0,
                    deadline: block.timestamp + 300
                })
            );
            
            // Update tracking
            settings.lastCompoundTime = block.timestamp;
            compoundCount[tokenId]++;
            totalFeesCompounded[tokenId] += amount0 + amount1; // Simplified tracking
            
            // Gas tracking for Base optimization
            uint256 gasUsed = gasStart - gasleft();
            totalCompoundsExecuted++;
            totalGasUsed += gasUsed;
            averageGasPerCompound = totalGasUsed / totalCompoundsExecuted;
            
            emit UltraFrequentCompound(tokenId, amount0, amount1, gasUsed, block.timestamp);
        }
    }
    
    /**
     * @notice Batch compound multiple positions (Base chain optimized)
     * @param tokenIds Array of positions to compound
     */
    function batchCompound(uint256[] calldata tokenIds) external onlyKeeper nonReentrant {
        require(tokenIds.length <= MAX_BATCH_SIZE, "Batch too large");
        
        uint256 gasStart = gasleft();
        uint256 successCount = 0;
        
        for (uint256 i = 0; i < tokenIds.length; i++) {
            if (shouldCompound(tokenIds[i])) {
                try this.executeCompound(tokenIds[i]) {
                    successCount++;
                } catch {
                    // Continue with next position if one fails
                    continue;
                }
            }
        }
        
        uint256 totalGasUsed = gasStart - gasleft();
        
        require(successCount > 0, "No compounds executed");
        
        emit BatchCompoundExecuted(tokenIds, successCount, totalGasUsed);
    }
    
    /**
     * @notice Smart batch compound - only compound positions that need it
     * @param tokenIds Array of positions to check and compound
     */
    function smartBatchCompound(uint256[] calldata tokenIds) external onlyKeeper {
        uint256[] memory readyToCompound = new uint256[](tokenIds.length);
        uint256 readyCount = 0;
        
        // Filter positions that need compounding
        for (uint256 i = 0; i < tokenIds.length; i++) {
            if (shouldCompound(tokenIds[i])) {
                readyToCompound[readyCount] = tokenIds[i];
                readyCount++;
            }
        }
        
        // Create exact-size array
        uint256[] memory finalList = new uint256[](readyCount);
        for (uint256 i = 0; i < readyCount; i++) {
            finalList[i] = readyToCompound[i];
        }
        
        // Execute batch compound
        if (finalList.length > 0) {
            this.batchCompound(finalList);
        }
    }
    
    // ============ CONFIGURATION ============
    
    /**
     * @notice Configure compound settings for a position
     * @param tokenId Position to configure
     * @param settings Compound configuration
     */
    function setCompoundSettings(
        uint256 tokenId,
        CompoundSettings memory settings
    ) external onlyOwner {
        require(settings.minFeeThreshold >= MIN_COMPOUND_THRESHOLD / 10, "Threshold too low");
        require(settings.maxTimeInterval >= 1 minutes, "Interval too short");
        require(settings.gasLimit <= GAS_LIMIT_PER_COMPOUND * 2, "Gas limit too high");
        
        compoundSettings[tokenId] = settings;
        
        emit CompoundSettingsUpdated(tokenId, settings);
    }
    
    /**
     * @notice Set default ultra-frequent settings for a position
     * @param tokenId Position to configure
     */
    function setUltraFrequentDefaults(uint256 tokenId) external onlyOwner {
        CompoundSettings memory settings = CompoundSettings({
            minFeeThreshold: MIN_COMPOUND_THRESHOLD,
            maxTimeInterval: ULTRA_FREQUENT_INTERVAL,
            lastCompoundTime: block.timestamp,
            enabled: true,
            gasLimit: GAS_LIMIT_PER_COMPOUND
        });
        
        compoundSettings[tokenId] = settings;
        
        emit CompoundSettingsUpdated(tokenId, settings);
    }
    
    /**
     * @notice Authorize keeper for ultra-frequent operations
     * @param keeper Address to authorize
     * @param authorized Authorization status
     */
    function setKeeperAuthorization(address keeper, bool authorized) external onlyOwner {
        authorizedKeepers[keeper] = authorized;
        emit KeeperAuthorized(keeper, authorized);
    }
    
    // ============ VIEW FUNCTIONS ============
    
    /**
     * @notice Get positions ready for compounding
     * @param tokenIds Array of positions to check
     * @return readyPositions Positions that need compounding
     */
    function getPositionsReadyForCompound(
        uint256[] calldata tokenIds
    ) external view returns (uint256[] memory readyPositions) {
        uint256[] memory ready = new uint256[](tokenIds.length);
        uint256 readyCount = 0;
        
        for (uint256 i = 0; i < tokenIds.length; i++) {
            if (shouldCompound(tokenIds[i])) {
                ready[readyCount] = tokenIds[i];
                readyCount++;
            }
        }
        
        // Create exact-size array
        readyPositions = new uint256[](readyCount);
        for (uint256 i = 0; i < readyCount; i++) {
            readyPositions[i] = ready[i];
        }
    }
    
    /**
     * @notice Get compound performance metrics
     * @param tokenId Position to query
     * @return compoundCount_ Number of compounds
     * @return totalFees Total fees compounded
     * @return lastCompound Last compound timestamp
     */
    function getCompoundMetrics(uint256 tokenId) external view returns (
        uint256 compoundCount_,
        uint256 totalFees,
        uint256 lastCompound
    ) {
        CompoundSettings memory settings = compoundSettings[tokenId];
        return (
            compoundCount[tokenId],
            totalFeesCompounded[tokenId],
            settings.lastCompoundTime
        );
    }
    
    /**
     * @notice Get system performance metrics
     * @return totalCompounds Total compounds executed
     * @return avgGas Average gas per compound
     * @return totalGas Total gas used
     */
    function getSystemMetrics() external view returns (
        uint256 totalCompounds,
        uint256 avgGas,
        uint256 totalGas
    ) {
        return (
            totalCompoundsExecuted,
            averageGasPerCompound,
            totalGasUsed
        );
    }
    
    // ============ INTERNAL FUNCTIONS ============
    
    /**
     * @notice Get unclaimed fees value for a position (simplified for MVP)
     * @param tokenId Position to check
     * @return feesValue Estimated USD value of unclaimed fees
     */
    function _getUnclaimedFeesValue(uint256 tokenId) internal view returns (uint256) {
        // Simplified implementation for MVP
        // In production, this would integrate with price oracles
        
        // For now, return a placeholder value that triggers frequent compounds
        return MIN_COMPOUND_THRESHOLD + 1e17; // $0.60 equivalent
    }
}