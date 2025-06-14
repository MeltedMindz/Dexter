// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/**
 * @title IDEXStaking
 * @notice Interface for DEX token staking and WETH reward distribution
 * @dev Handles staking of DEX tokens and distribution of protocol fees in WETH
 */
interface IDEXStaking {
    
    /// @notice Emitted when tokens are staked
    event Staked(address indexed user, uint256 amount);
    
    /// @notice Emitted when tokens are unstaked
    event Unstaked(address indexed user, uint256 amount);
    
    /// @notice Emitted when WETH rewards are claimed
    event RewardsClaimed(address indexed user, uint256 wethAmount);
    
    /// @notice Emitted when WETH fees are added to the reward pool
    event WETHRewardsAdded(uint256 amount, uint256 newRewardPerShare);
    
    /// @notice Staker information
    struct StakeInfo {
        uint256 stakedAmount;
        uint256 rewardDebt;
        uint256 pendingRewards;
        uint256 lastClaimTime;
        uint256 totalClaimed;
    }
    
    /// @notice DEX token address
    function dexToken() external view returns (address);
    
    /// @notice WETH token address (reward token)
    function WETH() external view returns (address);
    
    /// @notice Total amount of DEX tokens staked
    function totalStaked() external view returns (uint256);
    
    /// @notice Accumulated WETH rewards per share (scaled by 1e12)
    function accWETHPerShare() external view returns (uint256);
    
    /// @notice Total WETH distributed to stakers
    function totalWETHDistributed() external view returns (uint256);
    
    /// @notice Minimum stake amount
    function minStakeAmount() external view returns (uint256);
    
    /**
     * @notice Stakes DEX tokens
     * @param amount Amount of DEX tokens to stake
     */
    function stake(uint256 amount) external;
    
    /**
     * @notice Unstakes DEX tokens
     * @param amount Amount of DEX tokens to unstake
     */
    function unstake(uint256 amount) external;
    
    /**
     * @notice Claims accumulated WETH rewards
     * @return reward Amount of WETH rewards claimed
     */
    function claimRewards() external returns (uint256 reward);
    
    /**
     * @notice Gets pending WETH rewards for a user
     * @param user Address to check
     * @return pending Amount of pending WETH rewards
     */
    function pendingRewards(address user) external view returns (uint256 pending);
    
    /**
     * @notice Gets stake information for a user
     * @param user Address to check
     * @return info Stake information
     */
    function getStakeInfo(address user) external view returns (StakeInfo memory info);
    
    /**
     * @notice Receives WETH from fee dispenser and updates rewards
     * @param wethAmount Amount of WETH to distribute
     * @dev Only callable by authorized FeeDispenser
     */
    function receiveWETHRewards(uint256 wethAmount) external;
    
    /**
     * @notice Emergency withdraw without rewards (safety mechanism)
     */
    function emergencyWithdraw() external;
    
    /**
     * @notice Updates minimum stake amount (owner only)
     * @param newMinStake New minimum stake amount
     */
    function updateMinStake(uint256 newMinStake) external;
}