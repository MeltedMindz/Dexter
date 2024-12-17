// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract DexterStakeManager is ReentrancyGuard, Ownable {
    // Risk levels matching our Python implementation
    enum RiskLevel { CONSERVATIVE, AGGRESSIVE, HYPER_AGGRESSIVE }
    
    struct UserStake {
        uint256 amount;
        RiskLevel riskLevel;
        uint256 stakingTimestamp;
        uint256 lastRewardTimestamp;
        uint256 accumulatedRewards;
    }
    
    // Mapping of user address to their stake details
    mapping(address => UserStake) public userStakes;
    
    // Total staked amount per risk level
    mapping(RiskLevel => uint256) public totalStaked;
    
    // Performance tracking
    struct PoolPerformance {
        uint256 totalFees;
        uint256 averageAPY;
        uint256 averageDPY;
        uint256 lastUpdateTimestamp;
    }
    
    mapping(RiskLevel => PoolPerformance) public riskPoolPerformance;
    
    // Events
    event Staked(address indexed user, uint256 amount, RiskLevel riskLevel);
    event Withdrawn(address indexed user, uint256 amount);
    event RewardsDistributed(address indexed user, uint256 amount);
    
    constructor() {
        // Initialize performance tracking
        for (uint i = 0; i <= uint(RiskLevel.HYPER_AGGRESSIVE); i++) {
            riskPoolPerformance[RiskLevel(i)] = PoolPerformance({
                totalFees: 0,
                averageAPY: 0,
                averageDPY: 0,
                lastUpdateTimestamp: block.timestamp
            });
        }
    }
    
    // Stake tokens with specified risk level
    function stake(uint256 amount, RiskLevel riskLevel) external nonReentrant {
        require(amount > 0, "Amount must be greater than 0");
        
        // Transfer tokens from user
        IERC20(stakingToken).transferFrom(msg.sender, address(this), amount);
        
        // Update user stake
        UserStake storage userStake = userStakes[msg.sender];
        if (userStake.amount > 0) {
            // If user has existing stake, distribute pending rewards first
            _distributeRewards(msg.sender);
        }
        
        userStake.amount += amount;
        userStake.riskLevel = riskLevel;
        userStake.stakingTimestamp = block.timestamp;
        userStake.lastRewardTimestamp = block.timestamp;
        
        // Update total staked for risk level
        totalStaked[riskLevel] += amount;
        
        emit Staked(msg.sender, amount, riskLevel);
    }
    
    // Calculate and distribute rewards
    function _distributeRewards(address user) internal {
        UserStake storage userStake = userStakes[user];
        require(userStake.amount > 0, "No stake found");
        
        uint256 timeElapsed = block.timestamp - userStake.lastRewardTimestamp;
        if (timeElapsed > 0) {
            // Calculate rewards based on risk level and performance
            uint256 rewards = _calculateRewards(
                userStake.amount,
                userStake.riskLevel,
                timeElapsed
            );
            
            userStake.accumulatedRewards += rewards;
            userStake.lastRewardTimestamp = block.timestamp;
            
            emit RewardsDistributed(user, rewards);
        }
    }
    
    // Calculate rewards based on pool performance
    function _calculateRewards(
        uint256 amount,
        RiskLevel riskLevel,
        uint256 timeElapsed
    ) internal view returns (uint256) {
        PoolPerformance storage performance = riskPoolPerformance[riskLevel];
        
        // Calculate daily reward rate based on DPY
        uint256 dailyRate = performance.averageDPY;
        
        // Convert to per-second rate and calculate rewards
        uint256 rewardRate = dailyRate / 86400; // seconds in a day
        return (amount * rewardRate * timeElapsed) / 1e18;
    }
}