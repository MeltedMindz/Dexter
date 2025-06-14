// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/**
 * @title IFeeDispenser
 * @notice Manages the collection and distribution of protocol fees in WETH
 * @dev Collects fees from LiquidityManager, converts to WETH, and distributes to stakers
 */
interface IFeeDispenser {
    
    /// @notice Emitted when fees are deposited
    event FeesDeposited(address indexed token, uint256 amount, uint256 wethAmount);
    
    /// @notice Emitted when fees are converted to WETH
    event FeesConverted(address indexed token, uint256 amountIn, uint256 wethOut);
    
    /// @notice Emitted when WETH is distributed to stakers
    event FeesDistributed(uint256 wethAmount, uint256 totalStaked);
    
    /// @notice Emitted when distribution threshold is updated
    event ThresholdUpdated(uint256 newThreshold);
    
    /// @notice WETH token address
    function WETH() external view returns (address);
    
    /// @notice DEX staking contract address
    function stakingContract() external view returns (address);
    
    /// @notice Minimum WETH balance to trigger distribution
    function distributionThreshold() external view returns (uint256);
    
    /// @notice Current WETH balance pending distribution
    function pendingDistribution() external view returns (uint256);
    
    /// @notice Swap router for token conversions
    function swapRouter() external view returns (address);
    
    /**
     * @notice Deposits fees from liquidity manager
     * @param token The token being deposited
     * @param amount The amount of tokens
     * @dev Only callable by authorized LiquidityManager
     */
    function depositFees(address token, uint256 amount) external;
    
    /**
     * @notice Converts accumulated tokens to WETH
     * @param tokens Array of token addresses to convert
     * @param minAmountsOut Minimum WETH amounts expected (slippage protection)
     */
    function convertToWETH(
        address[] calldata tokens,
        uint256[] calldata minAmountsOut
    ) external;
    
    /**
     * @notice Distributes WETH to stakers if threshold is met
     * @dev Can be called by anyone when threshold is reached
     * @return distributed Amount of WETH distributed
     */
    function distribute() external returns (uint256 distributed);
    
    /**
     * @notice Force distribution regardless of threshold (owner only)
     * @dev Emergency function for manual distribution
     */
    function forceDistribute() external returns (uint256 distributed);
    
    /**
     * @notice Updates the distribution threshold (owner only)
     * @param newThreshold New WETH threshold amount
     */
    function updateThreshold(uint256 newThreshold) external;
    
    /**
     * @notice Gets pending fees for a specific token
     * @param token Token address to check
     * @return balance Current balance of that token
     */
    function pendingFees(address token) external view returns (uint256 balance);
    
    /**
     * @notice Rescue stuck tokens (owner only)
     * @param token Token to rescue
     * @param amount Amount to rescue
     */
    function rescueTokens(address token, uint256 amount) external;
}