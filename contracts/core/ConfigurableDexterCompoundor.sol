// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/utils/Multicall.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import "@openzeppelin/contracts/token/ERC721/IERC721Receiver.sol";

import "@uniswap/v3-core/contracts/interfaces/IUniswapV3Pool.sol";
import "@uniswap/v3-core/contracts/libraries/TickMath.sol";
import "@uniswap/v3-periphery/contracts/libraries/LiquidityAmounts.sol";
import "@uniswap/v3-periphery/contracts/interfaces/INonfungiblePositionManager.sol";
import "@uniswap/v3-periphery/contracts/interfaces/ISwapRouter.sol";
import "../libraries/TWAPOracle.sol";
import "../governance/EmergencyAdmin.sol";
import "../security/AdvancedSecurityGuard.sol";
import "../utils/ConfigurationManager.sol";

/**
 * @title ConfigurableDexterCompoundor
 * @notice AI-powered auto-compounding system with configurable parameters
 * @dev Enhanced version of DexterCompoundor with configurable constants via ConfigurationManager
 */
contract ConfigurableDexterCompoundor is IERC721Receiver, AdvancedSecurityGuard, Multicall {
    using SafeERC20 for IERC20;

    // ============ IMMUTABLE DEPENDENCIES ============
    
    // Core addresses (immutable for gas efficiency)
    address public immutable weth;
    IUniswapV3Factory public immutable factory;
    INonfungiblePositionManager public immutable nonfungiblePositionManager;
    ISwapRouter public immutable swapRouter;
    ConfigurationManager public immutable configManager;
    
    // ============ MUTABLE CONFIGURATION REFERENCES ============
    
    // Configuration cache (updated periodically for gas efficiency)
    struct CachedConfig {
        uint64 totalRewardX64;
        uint64 compounderRewardX64;
        uint64 aiOptimizerRewardX64;
        uint32 maxPositionsPerAddress;
        uint256 maxGasPerCompound;
        uint256 dailyGasLimit;
        uint256 minOperationInterval;
        uint32 maxTWAPTickDifference;
        uint32 TWAPSeconds;
        bool twapProtectionEnabled;
        bool gasLimitingEnabled;
        uint256 lastUpdate;
    }
    
    CachedConfig private cachedConfig;
    uint256 public constant CONFIG_UPDATE_INTERVAL = 1 hours;
    
    // ============ STATE VARIABLES ============
    
    // AI and Emergency Management
    address public aiAgent;
    EmergencyAdmin public emergencyAdmin;
    bool public aiOptimizationEnabled = true;
    bool public emergencyPaused = false;
    
    // Position tracking
    mapping(uint256 => address) public ownerOf;
    mapping(address => uint256[]) public accountTokens;
    mapping(address => mapping(address => uint256)) public accountBalances;
    mapping(uint256 => bool) public aiManagedPositions;
    mapping(uint256 => uint256) public lastCompoundTime;
    mapping(uint256 => uint256) public compoundCount;
    
    // Gas safety tracking (enhanced with daily limits)
    struct AccountGasTracking {
        uint256 positionCount;
        uint256 lastOperationTimestamp;
        uint256 gasUsedToday;
        uint256 dailyResetTime;
    }
    
    mapping(address => AccountGasTracking) public accountGasTracking;
    
    // Fee tier tracking
    mapping(address => string) public userFeeTiers;
    
    // ============ EVENTS ============
    
    // Core operation events
    event TokenDeposited(address indexed account, uint256 indexed tokenId, bool aiManaged);
    event TokenWithdrawn(address indexed account, address indexed to, uint256 indexed tokenId);
    event AutoCompounded(
        address indexed account,
        uint256 indexed tokenId,
        uint256 amountAdded0,
        uint256 amountAdded1,
        uint256 reward0,
        uint256 reward1,
        address token0,
        address token1,
        bool aiOptimized
    );
    event BalanceWithdrawn(address indexed account, address indexed token, address indexed to, uint256 amount);
    
    // Configuration events
    event ConfigurationCacheUpdated(uint256 timestamp, address updatedBy);
    event UserFeeTierAssigned(address indexed user, string tierName, address indexed assignedBy);
    event DynamicParameterAdjusted(string parameter, uint256 oldValue, uint256 newValue, address indexed adjustedBy);
    
    // Enhanced safety events
    event GasUsageTracked(address indexed account, uint256 gasUsed, uint256 dailyTotal, uint256 remainingAllowance);
    event DailyGasLimitExceeded(address indexed account, uint256 attemptedGas, uint256 dailyLimit);
    event OperationRateLimited(address indexed account, uint256 timeSinceLastOp, uint256 minInterval);
    event PositionLimitReached(address indexed account, uint256 currentCount, uint256 maxAllowed);
    
    // AI and emergency events
    event AIAgentUpdated(address indexed oldAgent, address indexed newAgent);
    event AIOptimizationToggled(bool enabled);
    event EmergencyPaused(address indexed by);
    event EmergencyUnpaused(address indexed by);
    event TWAPValidationOverridden(address indexed aiAgent, uint256 indexed tokenId, string reason);

    // ============ ENUMS ============
    
    enum RewardConversion { NONE, TOKEN_0, TOKEN_1, AI_OPTIMIZED }

    // ============ STRUCTS ============
    
    struct AutoCompoundParams {
        uint256 tokenId;
        RewardConversion rewardConversion;
        bool withdrawReward;
        bool doSwap;
        bool useAIOptimization;
    }

    struct DecreaseLiquidityAndCollectParams {
        uint256 tokenId;
        uint128 liquidity;
        uint256 amount0Min;
        uint256 amount1Min;
        uint256 deadline;
        address recipient;
    }

    // ============ CONSTRUCTOR ============

    constructor(
        address _weth,
        IUniswapV3Factory _factory,
        INonfungiblePositionManager _nonfungiblePositionManager,
        ISwapRouter _swapRouter,
        ConfigurationManager _configManager,
        EmergencyAdmin _emergencyAdmin
    ) {
        weth = _weth;
        factory = _factory;
        nonfungiblePositionManager = _nonfungiblePositionManager;
        swapRouter = _swapRouter;
        configManager = _configManager;
        emergencyAdmin = _emergencyAdmin;
        
        // Initialize configuration cache
        _updateConfigurationCache();
    }

    // ============ CONFIGURATION MANAGEMENT ============

    /**
     * @notice Update configuration cache from ConfigurationManager
     * @dev Should be called periodically to refresh cached values
     */
    function updateConfigurationCache() external {
        require(
            block.timestamp >= cachedConfig.lastUpdate + CONFIG_UPDATE_INTERVAL || 
            msg.sender == owner(),
            "Update too frequent"
        );
        
        _updateConfigurationCache();
    }
    
    /**
     * @notice Internal function to update configuration cache
     */
    function _updateConfigurationCache() internal {
        ConfigurationManager.ProtocolConfig memory protocolConfig = configManager.getProtocolConfig();
        ConfigurationManager.TWAPConfig memory twapConfig = configManager.getTWAPConfig();
        
        // Calculate fee structure based on default tier
        ConfigurationManager.FeeTier memory defaultTier = configManager.getFeeConfigForUser(address(0));
        
        cachedConfig = CachedConfig({
            totalRewardX64: protocolConfig.maxRewardX64,
            compounderRewardX64: uint64(protocolConfig.maxRewardX64 / 2),
            aiOptimizerRewardX64: uint64(protocolConfig.maxRewardX64 / 4),
            maxPositionsPerAddress: protocolConfig.maxPositionsPerAddress,
            maxGasPerCompound: protocolConfig.maxGasPerCompound,
            dailyGasLimit: protocolConfig.dailyGasLimit,
            minOperationInterval: protocolConfig.minOperationInterval,
            maxTWAPTickDifference: twapConfig.maxTWAPTickDifference,
            TWAPSeconds: twapConfig.TWAPSeconds,
            twapProtectionEnabled: twapConfig.twapProtectionEnabled,
            gasLimitingEnabled: protocolConfig.gasLimitingEnabled,
            lastUpdate: block.timestamp
        });
        
        emit ConfigurationCacheUpdated(block.timestamp, msg.sender);
    }

    /**
     * @notice Assign user to specific fee tier
     * @param user User address
     * @param tierName Fee tier name
     */
    function assignUserFeeTier(address user, string calldata tierName) external onlyOwner {
        userFeeTiers[user] = tierName;
        emit UserFeeTierAssigned(user, tierName, msg.sender);
    }

    // ============ ENHANCED GETTERS WITH CONFIGURATION ============

    /**
     * @notice Get current fee structure for user
     * @param user User address
     * @return compounderReward Compounder reward in X64 format
     * @return aiOptimizerReward AI optimizer reward in X64 format
     */
    function getUserFeeStructure(address user) external view returns (uint64 compounderReward, uint64 aiOptimizerReward) {
        ConfigurationManager.FeeTier memory tier = _getUserFeeTier(user);
        
        // Convert basis points to X64 format
        uint256 Q64 = 2**64;
        compounderReward = uint64((tier.managementFeeBps * Q64) / 10000);
        aiOptimizerReward = uint64((tier.performanceFeeBps * Q64) / 20000); // Half of performance fee
    }

    /**
     * @notice Get current operational limits for user
     * @param user User address
     * @return maxPositions Maximum positions allowed
     * @return remainingGasToday Remaining gas allowance today
     * @return nextOperationTime Earliest next operation time
     */
    function getUserLimits(address user) external view returns (
        uint256 maxPositions,
        uint256 remainingGasToday,
        uint256 nextOperationTime
    ) {
        _ensureConfigurationFresh();
        
        maxPositions = cachedConfig.maxPositionsPerAddress;
        
        AccountGasTracking memory gasTracking = accountGasTracking[user];
        
        if (block.timestamp >= gasTracking.dailyResetTime + 1 days) {
            remainingGasToday = cachedConfig.dailyGasLimit;
        } else {
            remainingGasToday = gasTracking.gasUsedToday >= cachedConfig.dailyGasLimit ? 
                0 : cachedConfig.dailyGasLimit - gasTracking.gasUsedToday;
        }
        
        nextOperationTime = gasTracking.lastOperationTimestamp + cachedConfig.minOperationInterval;
    }

    /**
     * @notice Check if operation is allowed for user
     * @param user User address
     * @param estimatedGas Estimated gas for operation
     * @return allowed Whether operation is allowed
     * @return reason Reason if not allowed
     */
    function checkOperationAllowed(address user, uint256 estimatedGas) external view returns (bool allowed, string memory reason) {
        _ensureConfigurationFresh();
        
        if (!cachedConfig.gasLimitingEnabled) {
            return (true, "");
        }
        
        AccountGasTracking memory gasTracking = accountGasTracking[user];
        
        // Check position count limit
        if (gasTracking.positionCount >= cachedConfig.maxPositionsPerAddress) {
            return (false, "Position limit exceeded");
        }
        
        // Check operation interval
        if (block.timestamp < gasTracking.lastOperationTimestamp + cachedConfig.minOperationInterval) {
            return (false, "Operation too frequent");
        }
        
        // Check daily gas limit
        uint256 dailyGasUsed = gasTracking.gasUsedToday;
        if (block.timestamp >= gasTracking.dailyResetTime + 1 days) {
            dailyGasUsed = 0; // Reset for new day
        }
        
        if (dailyGasUsed + estimatedGas > cachedConfig.dailyGasLimit) {
            return (false, "Daily gas limit exceeded");
        }
        
        return (true, "");
    }

    // ============ ENHANCED POSITION MANAGEMENT ============

    /**
     * @notice Handle NFT deposits with enhanced tracking
     */
    function onERC721Received(
        address,
        address from,
        uint256 tokenId,
        bytes calldata data
    ) external override nonReentrant returns (bytes4) {
        require(msg.sender == address(nonfungiblePositionManager), "Not authorized NFT contract");
        
        _ensureConfigurationFresh();
        
        bool enableAIManagement = data.length > 0 && data[0] == 0x01;
        
        _addTokenWithLimits(tokenId, from, enableAIManagement);
        
        emit TokenDeposited(from, tokenId, enableAIManagement);
        return this.onERC721Received.selector;
    }

    /**
     * @notice Enhanced token addition with limit checking
     */
    function _addTokenWithLimits(uint256 tokenId, address account, bool enableAIManagement) internal {
        AccountGasTracking storage gasTracking = accountGasTracking[account];
        
        // Check position limit
        require(gasTracking.positionCount < cachedConfig.maxPositionsPerAddress, "Position limit exceeded");
        
        // Add token
        ownerOf[tokenId] = account;
        accountTokens[account].push(tokenId);
        gasTracking.positionCount = accountTokens[account].length;
        
        // Set AI management if enabled
        if (enableAIManagement && aiOptimizationEnabled) {
            aiManagedPositions[tokenId] = true;
        }
        
        emit PositionCountUpdated(account, gasTracking.positionCount, cachedConfig.maxPositionsPerAddress, block.timestamp);
    }

    // ============ ENHANCED AUTO-COMPOUND WITH CONFIGURATION ============

    /**
     * @notice Auto-compound with configurable parameters and enhanced safety
     */
    function autoCompound(AutoCompoundParams calldata params) 
        external 
        nonReentrant 
        notInEmergency
        returns (uint256 reward0, uint256 reward1, uint256 compounded0, uint256 compounded1) 
    {
        uint256 startGas = gasleft();
        uint256 tokenId = params.tokenId;
        
        require(ownerOf[tokenId] != address(0), "Invalid token");
        
        _ensureConfigurationFresh();
        
        // Enhanced authorization check
        address tokenOwner = ownerOf[tokenId];
        require(tokenOwner == msg.sender || msg.sender == aiAgent, "Not authorized");
        
        // Gas safety checks with configurable limits
        if (cachedConfig.gasLimitingEnabled && msg.sender != aiAgent) {
            _checkAndUpdateGasUsage(tokenOwner, startGas);
        }

        // Check if AI optimization is requested and authorized
        bool useAI = params.useAIOptimization && 
                    aiOptimizationEnabled && 
                    (msg.sender == aiAgent || aiManagedPositions[tokenId]);

        // Get position info
        (, , address token0, address token1, uint24 fee, int24 tickLower, int24 tickUpper, 
         uint128 liquidity, , , uint128 tokensOwed0, uint128 tokensOwed1) = 
            nonfungiblePositionManager.positions(tokenId);

        require(liquidity > 0, "No liquidity");
        require(tokensOwed0 > 0 || tokensOwed1 > 0, "No fees to compound");

        // TWAP validation with AI override capability
        if (!useAI && cachedConfig.twapProtectionEnabled) {
            _validateTWAP(token0, token1, fee);
        } else if (useAI && !cachedConfig.twapProtectionEnabled) {
            emit TWAPValidationOverridden(msg.sender, tokenId, "AI agent compound with TWAP disabled");
        }

        // Collect fees
        INonfungiblePositionManager.CollectParams memory collectParams = INonfungiblePositionManager.CollectParams({
            tokenId: tokenId,
            recipient: address(this),
            amount0Max: type(uint128).max,
            amount1Max: type(uint128).max
        });

        (uint256 collected0, uint256 collected1) = nonfungiblePositionManager.collect(collectParams);

        // Calculate rewards using user's fee tier
        ConfigurationManager.FeeTier memory feeTier = _getUserFeeTier(tokenOwner);
        (reward0, reward1) = _calculateRewardsWithTier(collected0, collected1, feeTier, useAI);

        // Amounts available for compounding
        uint256 amount0 = collected0 - reward0;
        uint256 amount1 = collected1 - reward1;

        // Add liquidity back to position
        if (amount0 > 0 || amount1 > 0) {
            INonfungiblePositionManager.IncreaseLiquidityParams memory increaseLiquidityParams = 
                INonfungiblePositionManager.IncreaseLiquidityParams({
                    tokenId: tokenId,
                    amount0Desired: amount0,
                    amount1Desired: amount1,
                    amount0Min: useAI ? 0 : (amount0 * 95) / 100,
                    amount1Min: useAI ? 0 : (amount1 * 95) / 100,
                    deadline: block.timestamp + 300
                });

            IERC20(token0).safeApprove(address(nonfungiblePositionManager), amount0);
            IERC20(token1).safeApprove(address(nonfungiblePositionManager), amount1);

            (, compounded0, compounded1) = nonfungiblePositionManager.increaseLiquidity(increaseLiquidityParams);

            // Reset approvals
            IERC20(token0).safeApprove(address(nonfungiblePositionManager), 0);
            IERC20(token1).safeApprove(address(nonfungiblePositionManager), 0);

            // Add any leftover amounts to account balances
            if (amount0 > compounded0) {
                accountBalances[tokenOwner][token0] += (amount0 - compounded0);
            }
            if (amount1 > compounded1) {
                accountBalances[tokenOwner][token1] += (amount1 - compounded1);
            }
        }

        // Transfer rewards to compounder
        if (reward0 > 0) {
            if (params.withdrawReward) {
                IERC20(token0).safeTransfer(msg.sender, reward0);
            } else {
                accountBalances[msg.sender][token0] += reward0;
            }
        }
        if (reward1 > 0) {
            if (params.withdrawReward) {
                IERC20(token1).safeTransfer(msg.sender, reward1);
            } else {
                accountBalances[msg.sender][token1] += reward1;
            }
        }

        // Update position stats
        lastCompoundTime[tokenId] = block.timestamp;
        compoundCount[tokenId]++;

        // Track final gas usage
        uint256 gasUsed = startGas - gasleft();
        emit GasUsageTracked(tokenOwner, gasUsed, accountGasTracking[tokenOwner].gasUsedToday + gasUsed, 
                           cachedConfig.dailyGasLimit > accountGasTracking[tokenOwner].gasUsedToday + gasUsed ? 
                           cachedConfig.dailyGasLimit - accountGasTracking[tokenOwner].gasUsedToday - gasUsed : 0);

        emit AutoCompounded(
            tokenOwner,
            tokenId,
            compounded0,
            compounded1,
            reward0,
            reward1,
            token0,
            token1,
            useAI
        );
    }

    // ============ INTERNAL HELPER FUNCTIONS ============

    /**
     * @notice Ensure configuration is fresh
     */
    function _ensureConfigurationFresh() internal view {
        require(
            block.timestamp < cachedConfig.lastUpdate + CONFIG_UPDATE_INTERVAL,
            "Configuration outdated - call updateConfigurationCache()"
        );
    }

    /**
     * @notice Get user's fee tier configuration
     */
    function _getUserFeeTier(address user) internal view returns (ConfigurationManager.FeeTier memory) {
        string memory tierName = userFeeTiers[user];
        if (bytes(tierName).length == 0) {
            return configManager.getFeeConfigForUser(user); // Use default from config manager
        }
        return configManager.getFeeConfigForUser(user);
    }

    /**
     * @notice Calculate rewards based on user's fee tier
     */
    function _calculateRewardsWithTier(
        uint256 amount0,
        uint256 amount1,
        ConfigurationManager.FeeTier memory feeTier,
        bool useAI
    ) internal pure returns (uint256 reward0, uint256 reward1) {
        uint256 rewardBps = useAI ? feeTier.performanceFeeBps : feeTier.managementFeeBps;
        
        reward0 = (amount0 * rewardBps) / 10000;
        reward1 = (amount1 * rewardBps) / 10000;
    }

    /**
     * @notice Enhanced gas usage checking and tracking
     */
    function _checkAndUpdateGasUsage(address account, uint256 startGas) internal {
        AccountGasTracking storage gasTracking = accountGasTracking[account];
        
        // Check operation frequency
        require(
            block.timestamp >= gasTracking.lastOperationTimestamp + cachedConfig.minOperationInterval,
            "Operation too frequent"
        );
        
        // Check estimated gas
        require(startGas >= cachedConfig.maxGasPerCompound + 50000, "Insufficient gas");
        
        // Reset daily counter if needed
        if (block.timestamp >= gasTracking.dailyResetTime + 1 days) {
            gasTracking.gasUsedToday = 0;
            gasTracking.dailyResetTime = block.timestamp;
        }
        
        // Check daily gas limit
        require(
            gasTracking.gasUsedToday + cachedConfig.maxGasPerCompound <= cachedConfig.dailyGasLimit,
            "Daily gas limit exceeded"
        );
        
        // Update tracking
        gasTracking.lastOperationTimestamp = block.timestamp;
    }

    /**
     * @notice Validate TWAP using configurable parameters
     */
    function _validateTWAP(address token0, address token1, uint24 fee) internal view {
        address pool = factory.getPool(token0, token1, fee);
        require(pool != address(0), "Pool not found");

        IUniswapV3Pool poolContract = IUniswapV3Pool(pool);
        
        // Use configurable TWAP parameters
        (bool success,) = poolContract.verifyTWAP(
            cachedConfig.TWAPSeconds,
            int24(cachedConfig.maxTWAPTickDifference),
            false
        );
        
        require(success, "TWAP validation failed");
    }

    // ============ ADMIN FUNCTIONS ============

    /**
     * @notice Set AI agent address (only owner)
     */
    function setAIAgent(address _aiAgent) external onlyOwner {
        address oldAgent = aiAgent;
        aiAgent = _aiAgent;
        emit AIAgentUpdated(oldAgent, _aiAgent);
    }

    /**
     * @notice Toggle AI optimization (only owner)
     */
    function toggleAIOptimization(bool _enabled) external onlyOwner {
        aiOptimizationEnabled = _enabled;
        emit AIOptimizationToggled(_enabled);
    }

    /**
     * @notice Emergency pause function
     */
    function emergencyPause() external {
        require(
            address(emergencyAdmin) != address(0) && 
            emergencyAdmin.hasRole(emergencyAdmin.EMERGENCY_ADMIN_ROLE(), msg.sender),
            "Not emergency admin"
        );
        emergencyPaused = true;
        emit EmergencyPaused(msg.sender);
    }

    /**
     * @notice Emergency unpause function
     */
    function emergencyUnpause() external onlyOwner {
        emergencyPaused = false;
        emit EmergencyUnpaused(msg.sender);
    }

    // ============ LEGACY COMPATIBILITY ============

    // Standard view functions for backward compatibility
    function totalRewardX64() external view returns (uint64) {
        return cachedConfig.totalRewardX64;
    }
    
    function maxTWAPTickDifference() external view returns (uint32) {
        return cachedConfig.maxTWAPTickDifference;
    }
    
    function TWAPSeconds() external view returns (uint32) {
        return cachedConfig.TWAPSeconds;
    }

    // ============ REQUIRED INTERFACE IMPLEMENTATIONS ============

    function onERC721Received(address, address, uint256, bytes calldata) external pure override returns (bytes4) {
        return IERC721Receiver.onERC721Received.selector;
    }

    /// @notice Modifier to check emergency status
    modifier notInEmergency() {
        require(
            !emergencyPaused && 
            (address(emergencyAdmin) == address(0) || !emergencyAdmin.isEmergencyPaused(address(this))),
            "Contract is in emergency mode"
        );
        _;
    }

    // Placeholder events for compatibility
    event PositionCountUpdated(address indexed account, uint256 newCount, uint256 maxAllowed, uint256 timestamp);
}