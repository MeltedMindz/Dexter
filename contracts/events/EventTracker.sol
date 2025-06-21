// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/utils/Multicall.sol";

/// @title EventTracker
/// @notice Comprehensive event tracking and analytics for Dexter Protocol
/// @dev Centralized event emission with detailed metadata and performance tracking
contract EventTracker is Ownable, ReentrancyGuard, Multicall {
    
    struct PositionEvent {
        uint256 timestamp;
        uint256 blockNumber;
        address user;
        uint256 tokenId;
        string eventType;
        bytes eventData;
        uint256 gasUsed;
        uint256 gasPrice;
        bytes32 transactionHash;
    }
    
    struct PerformanceMetrics {
        uint256 totalGasUsed;
        uint256 averageGasPrice;
        uint256 totalTransactions;
        uint256 successfulOperations;
        uint256 failedOperations;
        uint256 totalFeesCollected;
        uint256 totalValueCompounded;
        uint256 aiOptimizationCount;
        uint256 emergencyActionsCount;
    }
    
    struct UserMetrics {
        uint256 totalPositions;
        uint256 totalCompounds;
        uint256 totalFeesEarned;
        uint256 totalGasSpent;
        uint256 averageAPR;
        uint256 aiManagedPositions;
        uint256 lastActivityTimestamp;
        bool isActiveUser;
    }
    
    // Event storage
    mapping(uint256 => PositionEvent[]) public positionEvents; // tokenId => events
    mapping(address => PositionEvent[]) public userEvents; // user => events
    mapping(string => PositionEvent[]) public eventsByType; // eventType => events
    
    // Performance tracking
    PerformanceMetrics public globalMetrics;
    mapping(address => UserMetrics) public userMetrics;
    mapping(uint256 => uint256) public dailyMetrics; // day => metric value
    
    // Event filtering and pagination
    mapping(bytes32 => uint256) public eventCounts;
    uint256 public totalEventCount;
    
    // Analytics configuration
    uint256 public metricsRetentionDays = 365;
    bool public detailedTrackingEnabled = true;
    mapping(address => bool) public authorizedTrackers;
    
    // Event definitions
    event PositionDeposited(
        address indexed user,
        uint256 indexed tokenId,
        bool aiManaged,
        uint256 initialValue,
        uint256 timestamp
    );
    
    event PositionWithdrawn(
        address indexed user,
        uint256 indexed tokenId,
        uint256 finalValue,
        uint256 totalFeesEarned,
        uint256 timestamp
    );
    
    event CompoundExecuted(
        address indexed user,
        uint256 indexed tokenId,
        uint256 feesCompounded,
        uint256 gasUsed,
        uint256 gasPrice,
        bool aiOptimized,
        string strategy,
        uint256 timestamp
    );
    
    event AIOptimizationPerformed(
        address indexed user,
        uint256 indexed tokenId,
        string optimizationType,
        uint256 oldValue,
        uint256 newValue,
        uint256 improvementPercent,
        uint256 timestamp
    );
    
    event LiquidationExecuted(
        address indexed borrower,
        address indexed liquidator,
        uint256 indexed tokenId,
        uint256 debtRepaid,
        uint256 collateralLiquidated,
        uint256 liquidationBonus,
        uint256 timestamp
    );
    
    event EmergencyActionExecuted(
        address indexed admin,
        string actionType,
        address indexed target,
        bytes actionData,
        uint256 timestamp
    );
    
    event PerformanceSnapshot(
        uint256 timestamp,
        uint256 totalValueLocked,
        uint256 totalPositions,
        uint256 averageAPR,
        uint256 totalFeesGenerated,
        uint256 aiOptimizationRate
    );
    
    event UserMilestone(
        address indexed user,
        string milestoneType,
        uint256 value,
        uint256 timestamp
    );
    
    modifier onlyAuthorizedTracker() {
        require(authorizedTrackers[msg.sender] || msg.sender == owner(), "Unauthorized tracker");
        _;
    }
    
    constructor() {
        authorizedTrackers[msg.sender] = true;
        globalMetrics.totalTransactions = 0;
    }
    
    /// @notice Track position deposit event
    function trackPositionDeposit(
        address user,
        uint256 tokenId,
        bool aiManaged,
        uint256 initialValue
    ) external onlyAuthorizedTracker {
        _recordEvent(
            user,
            tokenId,
            "POSITION_DEPOSITED",
            abi.encode(aiManaged, initialValue),
            tx.gasprice
        );
        
        // Update user metrics
        userMetrics[user].totalPositions++;
        if (aiManaged) {
            userMetrics[user].aiManagedPositions++;
        }
        userMetrics[user].lastActivityTimestamp = block.timestamp;
        userMetrics[user].isActiveUser = true;
        
        emit PositionDeposited(user, tokenId, aiManaged, initialValue, block.timestamp);
    }
    
    /// @notice Track position withdrawal event
    function trackPositionWithdrawal(
        address user,
        uint256 tokenId,
        uint256 finalValue,
        uint256 totalFeesEarned
    ) external onlyAuthorizedTracker {
        _recordEvent(
            user,
            tokenId,
            "POSITION_WITHDRAWN",
            abi.encode(finalValue, totalFeesEarned),
            tx.gasprice
        );
        
        // Update user metrics
        userMetrics[user].totalPositions--;
        userMetrics[user].totalFeesEarned += totalFeesEarned;
        userMetrics[user].lastActivityTimestamp = block.timestamp;
        
        // Update global metrics
        globalMetrics.totalFeesCollected += totalFeesEarned;
        
        emit PositionWithdrawn(user, tokenId, finalValue, totalFeesEarned, block.timestamp);
    }
    
    /// @notice Track compound execution event
    function trackCompoundExecution(
        address user,
        uint256 tokenId,
        uint256 feesCompounded,
        uint256 gasUsed,
        bool aiOptimized,
        string calldata strategy
    ) external onlyAuthorizedTracker {
        _recordEvent(
            user,
            tokenId,
            "COMPOUND_EXECUTED",
            abi.encode(feesCompounded, gasUsed, aiOptimized, strategy),
            tx.gasprice
        );
        
        // Update user metrics
        userMetrics[user].totalCompounds++;
        userMetrics[user].totalGasSpent += gasUsed * tx.gasprice;
        userMetrics[user].lastActivityTimestamp = block.timestamp;
        
        // Update global metrics
        globalMetrics.totalGasUsed += gasUsed;
        globalMetrics.totalValueCompounded += feesCompounded;
        globalMetrics.successfulOperations++;
        
        if (aiOptimized) {
            globalMetrics.aiOptimizationCount++;
        }
        
        emit CompoundExecuted(
            user,
            tokenId,
            feesCompounded,
            gasUsed,
            tx.gasprice,
            aiOptimized,
            strategy,
            block.timestamp
        );
        
        // Check for user milestones
        _checkUserMilestones(user);
    }
    
    /// @notice Track AI optimization event
    function trackAIOptimization(
        address user,
        uint256 tokenId,
        string calldata optimizationType,
        uint256 oldValue,
        uint256 newValue
    ) external onlyAuthorizedTracker {
        uint256 improvementPercent = newValue > oldValue ? 
            ((newValue - oldValue) * 10000) / oldValue : 0;
        
        _recordEvent(
            user,
            tokenId,
            "AI_OPTIMIZATION",
            abi.encode(optimizationType, oldValue, newValue, improvementPercent),
            tx.gasprice
        );
        
        globalMetrics.aiOptimizationCount++;
        
        emit AIOptimizationPerformed(
            user,
            tokenId,
            optimizationType,
            oldValue,
            newValue,
            improvementPercent,
            block.timestamp
        );
    }
    
    /// @notice Track liquidation event
    function trackLiquidation(
        address borrower,
        address liquidator,
        uint256 tokenId,
        uint256 debtRepaid,
        uint256 collateralLiquidated,
        uint256 liquidationBonus
    ) external onlyAuthorizedTracker {
        _recordEvent(
            borrower,
            tokenId,
            "LIQUIDATION",
            abi.encode(liquidator, debtRepaid, collateralLiquidated, liquidationBonus),
            tx.gasprice
        );
        
        emit LiquidationExecuted(
            borrower,
            liquidator,
            tokenId,
            debtRepaid,
            collateralLiquidated,
            liquidationBonus,
            block.timestamp
        );
    }
    
    /// @notice Track emergency action event
    function trackEmergencyAction(
        address admin,
        string calldata actionType,
        address target,
        bytes calldata actionData
    ) external onlyAuthorizedTracker {
        _recordEvent(
            admin,
            0, // No specific token ID for emergency actions
            "EMERGENCY_ACTION",
            abi.encode(actionType, target, actionData),
            tx.gasprice
        );
        
        globalMetrics.emergencyActionsCount++;
        
        emit EmergencyActionExecuted(admin, actionType, target, actionData, block.timestamp);
    }
    
    /// @notice Generate performance snapshot
    function generatePerformanceSnapshot(
        uint256 totalValueLocked,
        uint256 totalPositions,
        uint256 averageAPR,
        uint256 totalFeesGenerated
    ) external onlyAuthorizedTracker {
        uint256 aiOptimizationRate = globalMetrics.totalTransactions > 0 ?
            (globalMetrics.aiOptimizationCount * 10000) / globalMetrics.totalTransactions : 0;
        
        emit PerformanceSnapshot(
            block.timestamp,
            totalValueLocked,
            totalPositions,
            averageAPR,
            totalFeesGenerated,
            aiOptimizationRate
        );
        
        // Store daily metrics
        uint256 day = block.timestamp / 86400;
        dailyMetrics[day] = totalValueLocked;
    }
    
    /// @notice Get events for a specific position
    function getPositionEvents(
        uint256 tokenId,
        uint256 offset,
        uint256 limit
    ) external view returns (PositionEvent[] memory events) {
        PositionEvent[] storage allEvents = positionEvents[tokenId];
        
        if (offset >= allEvents.length) {
            return new PositionEvent[](0);
        }
        
        uint256 end = offset + limit;
        if (end > allEvents.length) {
            end = allEvents.length;
        }
        
        events = new PositionEvent[](end - offset);
        for (uint256 i = offset; i < end; i++) {
            events[i - offset] = allEvents[i];
        }
    }
    
    /// @notice Get events for a specific user
    function getUserEvents(
        address user,
        uint256 offset,
        uint256 limit
    ) external view returns (PositionEvent[] memory events) {
        PositionEvent[] storage allEvents = userEvents[user];
        
        if (offset >= allEvents.length) {
            return new PositionEvent[](0);
        }
        
        uint256 end = offset + limit;
        if (end > allEvents.length) {
            end = allEvents.length;
        }
        
        events = new PositionEvent[](end - offset);
        for (uint256 i = offset; i < end; i++) {
            events[i - offset] = allEvents[i];
        }
    }
    
    /// @notice Get events by type
    function getEventsByType(
        string calldata eventType,
        uint256 offset,
        uint256 limit
    ) external view returns (PositionEvent[] memory events) {
        PositionEvent[] storage allEvents = eventsByType[eventType];
        
        if (offset >= allEvents.length) {
            return new PositionEvent[](0);
        }
        
        uint256 end = offset + limit;
        if (end > allEvents.length) {
            end = allEvents.length;
        }
        
        events = new PositionEvent[](end - offset);
        for (uint256 i = offset; i < end; i++) {
            events[i - offset] = allEvents[i];
        }
    }
    
    /// @notice Get user performance metrics
    function getUserMetrics(address user) external view returns (UserMetrics memory) {
        return userMetrics[user];
    }
    
    /// @notice Get global performance metrics
    function getGlobalMetrics() external view returns (PerformanceMetrics memory) {
        return globalMetrics;
    }
    
    /// @notice Get daily metrics for a range
    function getDailyMetrics(uint256 startDay, uint256 endDay) 
        external 
        view 
        returns (uint256[] memory metrics) 
    {
        require(endDay >= startDay, "Invalid date range");
        
        uint256 length = endDay - startDay + 1;
        metrics = new uint256[](length);
        
        for (uint256 i = 0; i < length; i++) {
            metrics[i] = dailyMetrics[startDay + i];
        }
    }
    
    /// @notice Calculate user APR
    function calculateUserAPR(address user) external view returns (uint256 apr) {
        UserMetrics memory metrics = userMetrics[user];
        
        if (metrics.totalPositions == 0 || metrics.lastActivityTimestamp == 0) {
            return 0;
        }
        
        uint256 timeActive = block.timestamp - (block.timestamp - metrics.lastActivityTimestamp);
        if (timeActive == 0) {
            return 0;
        }
        
        // Simplified APR calculation
        uint256 yearlyFees = (metrics.totalFeesEarned * 365 days) / timeActive;
        uint256 avgPositionValue = 10000; // Simplified - would calculate actual average
        
        apr = (yearlyFees * 10000) / avgPositionValue; // APR in basis points
    }
    
    /// @notice Internal function to record events
    function _recordEvent(
        address user,
        uint256 tokenId,
        string memory eventType,
        bytes memory eventData,
        uint256 gasPrice
    ) internal {
        if (!detailedTrackingEnabled) {
            return;
        }
        
        PositionEvent memory newEvent = PositionEvent({
            timestamp: block.timestamp,
            blockNumber: block.number,
            user: user,
            tokenId: tokenId,
            eventType: eventType,
            eventData: eventData,
            gasUsed: gasleft(),
            gasPrice: gasPrice,
            transactionHash: blockhash(block.number - 1) // Simplified
        });
        
        // Store event in different indexes
        positionEvents[tokenId].push(newEvent);
        userEvents[user].push(newEvent);
        eventsByType[eventType].push(newEvent);
        
        // Update counters
        totalEventCount++;
        bytes32 eventTypeHash = keccak256(abi.encodePacked(eventType));
        eventCounts[eventTypeHash]++;
        
        // Update global metrics
        globalMetrics.totalTransactions++;
        globalMetrics.averageGasPrice = 
            (globalMetrics.averageGasPrice + gasPrice) / 2;
    }
    
    /// @notice Check and emit user milestones
    function _checkUserMilestones(address user) internal {
        UserMetrics memory metrics = userMetrics[user];
        
        // Check compound milestones
        if (metrics.totalCompounds == 10) {
            emit UserMilestone(user, "FIRST_10_COMPOUNDS", 10, block.timestamp);
        } else if (metrics.totalCompounds == 100) {
            emit UserMilestone(user, "FIRST_100_COMPOUNDS", 100, block.timestamp);
        }
        
        // Check fee milestones
        if (metrics.totalFeesEarned >= 1000 * 1e18) {
            emit UserMilestone(user, "FEES_1000", metrics.totalFeesEarned, block.timestamp);
        }
        
        // Check position milestones
        if (metrics.totalPositions >= 10) {
            emit UserMilestone(user, "POSITIONS_10", metrics.totalPositions, block.timestamp);
        }
    }
    
    /// @notice Authorize event tracker
    function authorizeTracker(address tracker, bool authorized) external onlyOwner {
        authorizedTrackers[tracker] = authorized;
    }
    
    /// @notice Set metrics retention period
    function setMetricsRetention(uint256 days) external onlyOwner {
        require(days >= 30 && days <= 1095, "Invalid retention period");
        metricsRetentionDays = days;
    }
    
    /// @notice Toggle detailed tracking
    function toggleDetailedTracking(bool enabled) external onlyOwner {
        detailedTrackingEnabled = enabled;
    }
    
    /// @notice Clean old events (gas-intensive, use carefully)
    function cleanOldEvents(uint256 cutoffTimestamp) external onlyOwner {
        // Implementation would remove events older than cutoff
        // This is gas-intensive and would need careful implementation
    }
}