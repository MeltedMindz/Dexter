// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@uniswap/v3-periphery/contracts/interfaces/INonfungiblePositionManager.sol";
import "@uniswap/v3-core/contracts/interfaces/IUniswapV3Factory.sol";
import "@uniswap/v3-periphery/contracts/interfaces/ISwapRouter.sol";

/**
 * @title IDexterLiquidityManager
 * @notice Interface for Dexter's AI-powered liquidity management system
 * @dev Handles auto-compounding, fee collection, and profit distribution
 */
interface IDexterLiquidityManager {
    
    /// @notice Emitted when a position is deposited
    event PositionDeposited(address indexed owner, uint256 indexed tokenId);
    
    /// @notice Emitted when a position is withdrawn
    event PositionWithdrawn(address indexed owner, uint256 indexed tokenId, address to);
    
    /// @notice Emitted when auto-compound occurs
    event AutoCompounded(
        uint256 indexed tokenId,
        uint256 amount0Added,
        uint256 amount1Added,
        uint256 profit0,
        uint256 profit1,
        uint256 feeCollected
    );
    
    /// @notice Emitted when fees are distributed to stakers
    event FeesDistributed(uint256 amount, uint256 totalStaked);
    
    /// @notice Emitted when reward parameters are updated
    event RewardParametersUpdated(uint256 profitFeeRate, uint256 compoundThreshold);
    
    /// @notice Position information
    struct Position {
        address owner;
        uint256 depositedAt;
        uint256 lastCompoundedAt;
        uint256 totalProfit0;
        uint256 totalProfit1;
        bool isActive;
    }
    
    /// @notice Auto-compound parameters
    struct AutoCompoundParams {
        uint256 tokenId;
        bool convertToken0ToToken1;
        bool convertToken1ToToken0;
        uint256 deadline;
        uint256 minAmountOut;
    }
    
    /// @notice Fee configuration (8% on profits only)
    function profitFeeRate() external view returns (uint256);
    
    /// @notice Minimum profit threshold for auto-compounding
    function compoundThreshold() external view returns (uint256);
    
    /// @notice DEX token address for staking rewards
    function dexToken() external view returns (address);
    
    /// @notice Total fees collected for distribution
    function totalFeesCollected() external view returns (uint256);
    
    /// @notice Uniswap V3 factory
    function factory() external view returns (IUniswapV3Factory);
    
    /// @notice Uniswap V3 position manager
    function positionManager() external view returns (INonfungiblePositionManager);
    
    /// @notice Uniswap V3 swap router
    function swapRouter() external view returns (ISwapRouter);
    
    /**
     * @notice Deposits a Uniswap V3 position NFT
     * @param tokenId The ID of the position NFT
     */
    function depositPosition(uint256 tokenId) external;
    
    /**
     * @notice Withdraws a position NFT
     * @param tokenId The ID of the position NFT
     * @param to The address to send the NFT to
     */
    function withdrawPosition(uint256 tokenId, address to) external;
    
    /**
     * @notice Auto-compounds a position
     * @param params Auto-compound parameters
     * @return amount0Added Amount of token0 added to position
     * @return amount1Added Amount of token1 added to position
     * @return feeCollected Fee collected (8% of profit)
     */
    function autoCompound(AutoCompoundParams calldata params) 
        external 
        returns (uint256 amount0Added, uint256 amount1Added, uint256 feeCollected);
    
    /**
     * @notice Batch auto-compounds multiple positions
     * @param params Array of auto-compound parameters
     */
    function batchAutoCompound(AutoCompoundParams[] calldata params) external;
    
    /**
     * @notice Gets position information
     * @param tokenId The position NFT ID
     * @return Position information
     */
    function getPosition(uint256 tokenId) external view returns (Position memory);
    
    /**
     * @notice Checks if a position is profitable enough to compound
     * @param tokenId The position NFT ID
     * @return isProfitable Whether the position meets compound threshold
     * @return profit0 Estimated profit in token0
     * @return profit1 Estimated profit in token1
     */
    function checkCompoundProfitability(uint256 tokenId) 
        external 
        view 
        returns (bool isProfitable, uint256 profit0, uint256 profit1);
    
    /**
     * @notice Updates fee parameters (owner only)
     * @param _profitFeeRate New profit fee rate (max 800 = 8%)
     * @param _compoundThreshold New minimum profit threshold
     */
    function updateRewardParameters(uint256 _profitFeeRate, uint256 _compoundThreshold) external;
    
    /**
     * @notice Emergency withdraw for owner
     * @param tokenId Position to withdraw
     */
    function emergencyWithdraw(uint256 tokenId) external;
}