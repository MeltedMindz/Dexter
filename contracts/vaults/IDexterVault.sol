// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/interfaces/IERC4626.sol";
import "@openzeppelin/contracts/token/ERC721/IERC721.sol";

/// @title IDexterVault - Enhanced ERC4626 vault interface for Dexter Protocol
/// @notice Combines ERC4626 standard with Dexter-specific features for AI-powered position management
interface IDexterVault is IERC4626 {
    
    // ============ ENUMS ============
    
    enum StrategyMode {
        MANUAL,          // User-controlled like Gamma
        AI_ASSISTED,     // AI recommendations with user approval
        FULLY_AUTOMATED  // Full AI management
    }
    
    enum PositionType {
        SINGLE_RANGE,    // Traditional single position
        DUAL_POSITION,   // Gamma-style base + limit positions
        MULTI_RANGE,     // Multiple position ranges
        AI_OPTIMIZED     // AI-determined optimal structure
    }
    
    // ============ STRUCTS ============
    
    struct VaultConfig {
        StrategyMode mode;
        PositionType positionType;
        bool aiOptimizationEnabled;
        bool autoCompoundEnabled;
        uint256 rebalanceThreshold;
        uint256 maxSlippageBps;
    }
    
    struct FeeConfiguration {
        uint256 managementFeeBps;     // Annual management fee (0-200 bps)
        uint256 performanceFeeBps;    // Performance fee on gains (0-2000 bps)
        uint256 aiOptimizationFeeBps; // Fee for AI management (0-100 bps)
        address feeRecipient;         // Protocol treasury
        uint256 strategistShareBps;   // Share for strategy provider (0-5000 bps)
        address strategist;           // Strategy provider address
    }
    
    struct PositionRange {
        int24 tickLower;
        int24 tickUpper;
        uint256 allocation;           // Percentage allocation (bps)
        bool isActive;
        uint256 liquidity;
    }
    
    struct VaultMetrics {
        uint256 totalValueLocked;
        uint256 totalFees24h;
        uint256 impermanentLoss;
        uint256 apr;
        uint256 sharpeRatio;
        uint256 maxDrawdown;
        uint256 successfulCompounds;
        uint256 aiOptimizationCount;
    }
    
    struct TWAPConfig {
        bool enabled;
        uint32 interval;              // TWAP interval in seconds
        uint256 maxDeviation;         // Maximum price deviation (bps)
        uint256 cooldownPeriod;       // Cooldown between large operations
    }
    
    // ============ EVENTS ============
    
    event StrategyModeChanged(StrategyMode oldMode, StrategyMode newMode);
    event PositionRebalanced(uint256 indexed positionId, int24 newLower, int24 newUpper);
    event AIOptimizationEnabled(bool enabled);
    event AutoCompoundTriggered(uint256 fees0, uint256 fees1, uint256 newLiquidity);
    event FeeDistribution(uint256 managementFees, uint256 performanceFees, uint256 aiFees);
    event PositionRangeAdded(uint256 indexed rangeId, int24 tickLower, int24 tickUpper, uint256 allocation);
    event PositionRangeRemoved(uint256 indexed rangeId);
    event TWAPValidationFailed(uint256 currentPrice, uint256 twapPrice, uint256 deviation);
    event EmergencyPause(bool paused, string reason);
    
    // ============ VAULT MANAGEMENT ============
    
    /// @notice Configure vault strategy and behavior
    function configureVault(VaultConfig calldata config) external;
    
    /// @notice Set fee configuration for the vault
    function setFeeConfiguration(FeeConfiguration calldata feeConfig) external;
    
    /// @notice Enable or disable AI management for the vault
    function setAIManagement(bool enabled) external;
    
    /// @notice Configure auto-compounding parameters
    function setAutoCompound(bool enabled, uint256 threshold) external;
    
    /// @notice Set TWAP protection parameters
    function configureTWAPProtection(TWAPConfig calldata twapConfig) external;
    
    // ============ POSITION MANAGEMENT ============
    
    /// @notice Add a new position range to the vault
    function addPositionRange(int24 tickLower, int24 tickUpper, uint256 allocation) external returns (uint256 rangeId);
    
    /// @notice Remove a position range from the vault
    function removePositionRange(uint256 rangeId) external;
    
    /// @notice Update allocation percentages across ranges
    function updateAllocations(uint256[] calldata rangeIds, uint256[] calldata allocations) external;
    
    /// @notice Manually trigger position rebalancing
    function rebalancePositions() external returns (bool success);
    
    /// @notice Get all active position ranges
    function getPositionRanges() external view returns (PositionRange[] memory);
    
    // ============ AI INTEGRATION ============
    
    /// @notice Get AI recommendation for position optimization
    function getAIRecommendation() external view returns (
        StrategyMode recommendedMode,
        PositionRange[] memory recommendedRanges,
        uint256 confidenceScore
    );
    
    /// @notice Apply AI-recommended strategy changes
    function applyAIRecommendation(bytes calldata aiData) external;
    
    /// @notice Get vault health score from AI analysis
    function getHealthScore() external view returns (uint256 healthScore, string memory analysis);
    
    // ============ COMPOUNDING ============
    
    /// @notice Manually compound accumulated fees
    function compound() external returns (uint256 newLiquidity);
    
    /// @notice Check if auto-compound conditions are met
    function shouldAutoCompound() external view returns (bool shouldCompound, uint256 estimatedGas);
    
    /// @notice Get compound opportunity analysis
    function analyzeCompoundOpportunity() external view returns (
        uint256 availableFees0,
        uint256 availableFees1,
        uint256 estimatedNewLiquidity,
        uint256 expectedAPRIncrease
    );
    
    // ============ ANALYTICS ============
    
    /// @notice Get comprehensive vault metrics
    function getVaultMetrics() external view returns (VaultMetrics memory);
    
    /// @notice Get historical performance data
    function getPerformanceHistory(uint256 fromTimestamp, uint256 toTimestamp) 
        external view returns (uint256[] memory timestamps, uint256[] memory values);
    
    /// @notice Compare performance against benchmark
    function benchmarkPerformance() external view returns (
        uint256 vaultAPR,
        uint256 benchmarkAPR,
        int256 outperformance
    );
    
    // ============ RISK MANAGEMENT ============
    
    /// @notice Validate deposit against TWAP and other risk parameters
    function validateDeposit(uint256 assets, address receiver) external view returns (bool valid, string memory reason);
    
    /// @notice Validate withdrawal against risk parameters
    function validateWithdrawal(uint256 shares, address receiver, address owner) 
        external view returns (bool valid, string memory reason);
    
    /// @notice Get current risk assessment
    function getRiskAssessment() external view returns (
        uint256 riskScore,
        string memory riskLevel,
        string[] memory riskFactors
    );
    
    /// @notice Emergency pause functionality
    function emergencyPause(bool paused, string calldata reason) external;
    
    // ============ COMPATIBILITY ============
    
    /// @notice Enable Gamma-style dual position mode
    function enableGammaMode(bool enabled) external;
    
    /// @notice Set dual position configuration (Gamma-style)
    function setDualPositionStrategy(
        int24 baseLower, 
        int24 baseUpper,
        int24 limitLower, 
        int24 limitUpper
    ) external;
    
    /// @notice Migrate from individual NFT positions to vault shares
    function migrateFromNFTPositions(uint256[] calldata tokenIds) 
        external returns (uint256 shares);
    
    /// @notice Extract individual NFT positions from vault
    function extractToNFTPositions(uint256 shares, uint256[] calldata desiredRanges) 
        external returns (uint256[] memory tokenIds);
    
    // ============ VIEW FUNCTIONS ============
    
    /// @notice Get vault configuration
    function getVaultConfig() external view returns (VaultConfig memory);
    
    /// @notice Get fee configuration
    function getFeeConfiguration() external view returns (FeeConfiguration memory);
    
    /// @notice Get TWAP configuration
    function getTWAPConfig() external view returns (TWAPConfig memory);
    
    /// @notice Check if vault is in Gamma compatibility mode
    function isGammaMode() external view returns (bool);
    
    /// @notice Get underlying Uniswap V3 pool
    function pool() external view returns (address);
    
    /// @notice Get token addresses
    function getTokens() external view returns (address token0, address token1, uint24 fee);
    
    /// @notice Get total amounts in underlying tokens
    function getTotalAmounts() external view returns (uint256 amount0, uint256 amount1);
    
    /// @notice Preview shares for deposit amount
    function previewDeposit(uint256 assets) external view override returns (uint256 shares);
    
    /// @notice Preview assets for share redemption
    function previewRedeem(uint256 shares) external view override returns (uint256 assets);
    
    /// @notice Get maximum deposit amount
    function maxDeposit(address receiver) external view override returns (uint256 maxAssets);
    
    /// @notice Get maximum withdrawal amount
    function maxWithdraw(address owner) external view override returns (uint256 maxAssets);

    /// @notice Get vault's pause status
    function isPaused() external view returns (bool);
}