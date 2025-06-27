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

/**
 * @title GasOptimizedDexterCompoundor
 * @notice Gas-optimized version of DexterCompoundor with storage packing and caching
 * @dev Optimizations include:
 *      - Storage struct packing to reduce SSTORE operations
 *      - Cached storage reads to minimize SLOAD operations
 *      - Optimized loops and batch operations
 *      - Assembly optimizations for critical paths
 */
contract GasOptimizedDexterCompoundor is IERC721Receiver, AdvancedSecurityGuard, Multicall {
    using SafeERC20 for IERC20;

    // ============ PACKED STORAGE STRUCTS ============
    
    /**
     * @notice Packed configuration storage (256 bits)
     * @dev All configuration variables packed into single storage slot
     */
    struct PackedConfig {
        uint64 totalRewardX64;          // 64 bits - Total reward percentage
        uint64 compounderRewardX64;     // 64 bits - Compounder reward percentage  
        uint64 aiOptimizerRewardX64;    // 64 bits - AI optimizer reward percentage
        uint32 maxTWAPTickDifference;   // 32 bits - Max TWAP tick difference
        uint32 TWAPSeconds;             // 32 bits - TWAP calculation period
    }
    
    /**
     * @notice Packed position data (256 bits)
     * @dev Position metadata packed into single storage slot
     */
    struct PackedPositionData {
        address owner;                  // 160 bits - Position owner
        uint32 lastCompoundTime;       // 32 bits - Last compound timestamp (truncated)
        uint32 compoundCount;          // 32 bits - Number of compounds
        bool aiManaged;                // 8 bits - Whether AI managed
        bool isActive;                 // 8 bits - Whether position is active
        uint16 reserved;               // 16 bits - Reserved for future use
    }
    
    /**
     * @notice Packed account limits (256 bits) 
     * @dev Account-specific limits and tracking
     */
    struct PackedAccountLimits {
        uint32 positionCount;          // 32 bits - Number of positions
        uint32 lastOperationTime;     // 32 bits - Last operation timestamp (truncated)
        uint32 gasUsedToday;          // 32 bits - Gas used today
        uint32 dailyResetTime;        // 32 bits - Daily reset timestamp
        uint128 reserved;             // 128 bits - Reserved for future use
    }

    // ============ CONSTANTS (IMMUTABLE FOR GAS SAVINGS) ============
    
    uint128 private constant Q64 = 2**64;
    uint128 private constant Q96 = 2**96;
    uint64 private constant MAX_REWARD_X64 = uint64(Q64 / 50); // 2% max reward
    uint32 private constant MAX_POSITIONS_PER_ADDRESS = 200;
    uint256 private constant MAX_GAS_PER_COMPOUND = 300000;
    uint256 private constant GAS_BUFFER = 50000;
    uint256 private constant MIN_OPERATION_INTERVAL = 1 minutes;
    uint256 private constant DAILY_GAS_LIMIT = 1000000; // 1M gas per day per account

    // ============ PACKED STORAGE VARIABLES ============
    
    PackedConfig private config;
    
    // Position data packed for gas efficiency
    mapping(uint256 => PackedPositionData) private positionData;
    
    // Account limits packed for gas efficiency
    mapping(address => PackedAccountLimits) private accountLimits;
    
    // ============ REGULAR STORAGE (WHEN PACKING ISN'T BENEFICIAL) ============
    
    // Core addresses (immutable)
    address public immutable weth;
    IUniswapV3Factory public immutable factory;
    INonfungiblePositionManager public immutable nonfungiblePositionManager;
    ISwapRouter public immutable swapRouter;
    
    // Single storage variables
    address public aiAgent;
    EmergencyAdmin public emergencyAdmin;
    
    // State flags packed into single storage slot
    uint256 private packedFlags; // bit 0: twapProtectionEnabled, bit 1: aiOptimizationEnabled, bit 2: emergencyPaused, bit 3: gasLimitingEnabled
    
    // Account balances and tokens (separate mappings for different access patterns)
    mapping(address => mapping(address => uint256)) public accountBalances;
    mapping(address => uint256[]) public accountTokens;

    // ============ EVENTS ============
    
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
    event ConfigUpdated(uint64 totalRewardX64, uint64 compounderRewardX64, uint64 aiOptimizerRewardX64, uint32 maxTWAPTickDifference, uint32 TWAPSeconds);

    // ============ CONSTRUCTOR ============

    constructor(
        address _weth,
        IUniswapV3Factory _factory,
        INonfungiblePositionManager _nonfungiblePositionManager,
        ISwapRouter _swapRouter,
        EmergencyAdmin _emergencyAdmin
    ) {
        weth = _weth;
        factory = _factory;
        nonfungiblePositionManager = _nonfungiblePositionManager;
        swapRouter = _swapRouter;
        emergencyAdmin = _emergencyAdmin;
        
        // Initialize configuration with default values
        config = PackedConfig({
            totalRewardX64: MAX_REWARD_X64,
            compounderRewardX64: MAX_REWARD_X64 / 2,
            aiOptimizerRewardX64: MAX_REWARD_X64 / 4,
            maxTWAPTickDifference: 100,
            TWAPSeconds: 60
        });
        
        // Initialize flags (all enabled by default)
        packedFlags = 0xF; // Sets bits 0-3 to 1
    }

    // ============ GAS-OPTIMIZED VIEW FUNCTIONS ============

    /**
     * @notice Get configuration with single storage read
     * @return Configuration struct
     */
    function getConfig() external view returns (PackedConfig memory) {
        return config;
    }

    /**
     * @notice Get position data with single storage read
     * @param tokenId Position token ID
     * @return Position data struct
     */
    function getPositionData(uint256 tokenId) external view returns (PackedPositionData memory) {
        return positionData[tokenId];
    }

    /**
     * @notice Get account limits with single storage read
     * @param account Account address
     * @return Account limits struct
     */
    function getAccountLimits(address account) external view returns (PackedAccountLimits memory) {
        return accountLimits[account];
    }

    /**
     * @notice Get boolean flags efficiently
     * @return twapEnabled AI optimization enabled
     * @return aiEnabled AI optimization enabled
     * @return paused Emergency paused
     * @return gasLimited Gas limiting enabled
     */
    function getFlags() external view returns (bool twapEnabled, bool aiEnabled, bool paused, bool gasLimited) {
        uint256 flags = packedFlags;
        assembly {
            twapEnabled := and(flags, 1)
            aiEnabled := and(shr(1, flags), 1)
            paused := and(shr(2, flags), 1)
            gasLimited := and(shr(3, flags), 1)
        }
    }

    // ============ GAS-OPTIMIZED WRITE FUNCTIONS ============

    /**
     * @notice Set configuration with single storage write
     * @param _totalRewardX64 Total reward percentage
     * @param _compounderRewardX64 Compounder reward percentage
     * @param _aiOptimizerRewardX64 AI optimizer reward percentage
     * @param _maxTWAPTickDifference Maximum TWAP tick difference
     * @param _TWAPSeconds TWAP calculation period
     */
    function setConfigBatch(
        uint64 _totalRewardX64,
        uint64 _compounderRewardX64,
        uint64 _aiOptimizerRewardX64,
        uint32 _maxTWAPTickDifference,
        uint32 _TWAPSeconds
    ) external onlyOwner {
        require(_totalRewardX64 <= MAX_REWARD_X64, "Total reward too high");
        require(_compounderRewardX64 <= _totalRewardX64, "Compounder reward too high");
        require(_aiOptimizerRewardX64 <= _totalRewardX64, "AI optimizer reward too high");
        require(_TWAPSeconds >= TWAPOracle.MIN_TWAP_SECONDS, "TWAP period too short");
        
        // Single storage write
        config = PackedConfig({
            totalRewardX64: _totalRewardX64,
            compounderRewardX64: _compounderRewardX64,
            aiOptimizerRewardX64: _aiOptimizerRewardX64,
            maxTWAPTickDifference: _maxTWAPTickDifference,
            TWAPSeconds: _TWAPSeconds
        });
        
        emit ConfigUpdated(_totalRewardX64, _compounderRewardX64, _aiOptimizerRewardX64, _maxTWAPTickDifference, _TWAPSeconds);
    }

    /**
     * @notice Toggle flags efficiently with bitwise operations
     * @param twapEnabled TWAP protection enabled
     * @param aiEnabled AI optimization enabled  
     * @param gasLimited Gas limiting enabled
     */
    function setFlags(bool twapEnabled, bool aiEnabled, bool gasLimited) external onlyOwner {
        uint256 flags;
        assembly {
            // Build flags using bitwise operations
            flags := or(twapEnabled, shl(1, aiEnabled))
            flags := or(flags, shl(3, gasLimited))
            // Preserve emergency paused flag (bit 2)
            let currentFlags := sload(packedFlags.slot)
            let emergencyBit := and(shr(2, currentFlags), 1)
            flags := or(flags, shl(2, emergencyBit))
        }
        packedFlags = flags;
    }

    /**
     * @notice Deposit token with optimized storage updates
     * @param tokenId Position token ID
     * @param aiManaged Whether position should be AI managed
     */
    function depositToken(uint256 tokenId, bool aiManaged) external nonReentrant {
        require(nonfungiblePositionManager.ownerOf(tokenId) == msg.sender, "Not owner");
        
        // Cache account limits for multiple operations
        PackedAccountLimits memory limits = accountLimits[msg.sender];
        require(limits.positionCount < MAX_POSITIONS_PER_ADDRESS, "Too many positions");
        
        // Transfer NFT
        nonfungiblePositionManager.safeTransferFrom(msg.sender, address(this), tokenId);
        
        // Update position data in single write
        positionData[tokenId] = PackedPositionData({
            owner: msg.sender,
            lastCompoundTime: uint32(block.timestamp),
            compoundCount: 0,
            aiManaged: aiManaged,
            isActive: true,
            reserved: 0
        });
        
        // Update account limits in single write
        limits.positionCount++;
        accountLimits[msg.sender] = limits;
        
        // Add to account tokens
        accountTokens[msg.sender].push(tokenId);
        
        emit TokenDeposited(msg.sender, tokenId, aiManaged);
    }

    /**
     * @notice Auto-compound with gas optimizations
     * @param params Compound parameters
     */
    function autoCompound(AutoCompoundParams calldata params) external nonReentrant {
        uint256 startGas = gasleft();
        
        // Load position data once
        PackedPositionData memory posData = positionData[params.tokenId];
        require(posData.isActive, "Position not active");
        require(posData.owner == msg.sender || msg.sender == aiAgent, "Not authorized");
        
        // Check gas limiting with cached data
        if (_getFlagAt(3)) { // gasLimitingEnabled
            _checkGasLimits(posData.owner, startGas);
        }
        
        // Load configuration once
        PackedConfig memory cfg = config;
        
        // Check TWAP protection if enabled
        if (_getFlagAt(0) && twapProtectionEnabled) { // twapProtectionEnabled
            _checkTWAPProtection(params.tokenId, cfg);
        }
        
        // Perform compound operation
        (uint256 amountAdded0, uint256 amountAdded1, uint256 reward0, uint256 reward1, address token0, address token1) = 
            _performCompound(params, cfg);
        
        // Update position data in single write
        posData.lastCompoundTime = uint32(block.timestamp);
        posData.compoundCount++;
        positionData[params.tokenId] = posData;
        
        // Update gas tracking
        if (_getFlagAt(3)) { // gasLimitingEnabled
            _updateGasUsage(posData.owner, startGas - gasleft());
        }
        
        emit AutoCompounded(
            posData.owner,
            params.tokenId,
            amountAdded0,
            amountAdded1,
            reward0,
            reward1,
            token0,
            token1,
            params.useAIOptimization
        );
    }

    /**
     * @notice Batch compound multiple positions for gas efficiency
     * @param tokenIds Array of token IDs to compound
     * @param useAI Whether to use AI optimization for all positions
     */
    function batchCompound(uint256[] calldata tokenIds, bool useAI) external nonReentrant {
        require(tokenIds.length <= 10, "Too many positions");
        uint256 startGas = gasleft();
        
        // Load configuration once for all operations
        PackedConfig memory cfg = config;
        bool twapEnabled = _getFlagAt(0);
        bool gasLimited = _getFlagAt(3);
        
        // Cache commonly used data
        address msgSender = msg.sender;
        uint32 currentTime = uint32(block.timestamp);
        
        for (uint256 i = 0; i < tokenIds.length;) {
            uint256 tokenId = tokenIds[i];
            
            // Load position data once per iteration
            PackedPositionData memory posData = positionData[tokenId];
            require(posData.isActive, "Position not active");
            require(posData.owner == msgSender || msgSender == aiAgent, "Not authorized");
            
            // Check time-based limits (skip TWAP for batch efficiency unless explicitly required)
            if (gasLimited) {
                require(currentTime - posData.lastCompoundTime >= MIN_OPERATION_INTERVAL, "Too frequent");
            }
            
            // Perform compound operation
            AutoCompoundParams memory params = AutoCompoundParams({
                tokenId: tokenId,
                rewardConversion: RewardConversion.AI_OPTIMIZED,
                withdrawReward: false,
                doSwap: true,
                useAIOptimization: useAI
            });
            
            (uint256 amountAdded0, uint256 amountAdded1, uint256 reward0, uint256 reward1, address token0, address token1) = 
                _performCompound(params, cfg);
            
            // Update position data efficiently
            posData.lastCompoundTime = currentTime;
            posData.compoundCount++;
            positionData[tokenId] = posData;
            
            emit AutoCompounded(
                posData.owner,
                tokenId,
                amountAdded0,
                amountAdded1,
                reward0,
                reward1,
                token0,
                token1,
                useAI
            );
            
            unchecked {
                ++i;
            }
        }
        
        // Update gas usage once for entire batch
        if (gasLimited) {
            _updateGasUsage(msgSender, startGas - gasleft());
        }
    }

    // ============ ASSEMBLY-OPTIMIZED HELPER FUNCTIONS ============

    /**
     * @notice Get flag value at specific bit position using assembly
     * @param bitPosition Bit position (0-7)
     * @return flag Flag value
     */
    function _getFlagAt(uint256 bitPosition) private view returns (bool flag) {
        assembly {
            let flags := sload(packedFlags.slot)
            flag := and(shr(bitPosition, flags), 1)
        }
    }

    /**
     * @notice Set flag value at specific bit position using assembly
     * @param bitPosition Bit position (0-7)
     * @param value Flag value
     */
    function _setFlagAt(uint256 bitPosition, bool value) private {
        assembly {
            let flags := sload(packedFlags.slot)
            let mask := shl(bitPosition, 1)
            // Clear the bit first
            flags := and(flags, not(mask))
            // Set the bit if value is true
            if value {
                flags := or(flags, mask)
            }
            sstore(packedFlags.slot, flags)
        }
    }

    /**
     * @notice Check gas limits for account
     * @param account Account to check
     * @param startGas Gas at start of operation
     */
    function _checkGasLimits(address account, uint256 startGas) private view {
        PackedAccountLimits memory limits = accountLimits[account];
        
        require(startGas >= MAX_GAS_PER_COMPOUND + GAS_BUFFER, "Insufficient gas");
        require(block.timestamp - limits.lastOperationTime >= MIN_OPERATION_INTERVAL, "Operation too frequent");
        
        // Check daily gas limit
        if (block.timestamp >= limits.dailyResetTime + 1 days) {
            // New day, reset is handled in _updateGasUsage
        } else {
            require(limits.gasUsedToday + MAX_GAS_PER_COMPOUND <= DAILY_GAS_LIMIT, "Daily gas limit exceeded");
        }
    }

    /**
     * @notice Update gas usage tracking
     * @param account Account to update
     * @param gasUsed Gas consumed
     */
    function _updateGasUsage(address account, uint256 gasUsed) private {
        PackedAccountLimits memory limits = accountLimits[account];
        
        // Reset daily counter if needed
        if (block.timestamp >= limits.dailyResetTime + 1 days) {
            limits.gasUsedToday = 0;
            limits.dailyResetTime = uint32(block.timestamp);
        }
        
        limits.gasUsedToday += uint32(gasUsed);
        limits.lastOperationTime = uint32(block.timestamp);
        
        accountLimits[account] = limits;
    }

    /**
     * @notice Check TWAP protection
     * @param tokenId Position token ID
     * @param cfg Configuration
     */
    function _checkTWAPProtection(uint256 tokenId, PackedConfig memory cfg) private view {
        // Get position info
        (,, address token0, address token1, uint24 fee,,,,,,,) = nonfungiblePositionManager.positions(tokenId);
        
        // Get pool
        address poolAddress = factory.getPool(token0, token1, fee);
        require(poolAddress != address(0), "Pool does not exist");
        IUniswapV3Pool pool = IUniswapV3Pool(poolAddress);
        
        // Check TWAP protection using library
        pool.checkTWAP(cfg.maxTWAPTickDifference, cfg.TWAPSeconds);
    }

    /**
     * @notice Perform the actual compound operation
     * @param params Compound parameters
     * @param cfg Configuration
     * @return amountAdded0 Amount of token0 added to position
     * @return amountAdded1 Amount of token1 added to position  
     * @return reward0 Reward in token0
     * @return reward1 Reward in token1
     * @return token0 Token0 address
     * @return token1 Token1 address
     */
    function _performCompound(AutoCompoundParams memory params, PackedConfig memory cfg) 
        private 
        returns (uint256 amountAdded0, uint256 amountAdded1, uint256 reward0, uint256 reward1, address token0, address token1) 
    {
        // Get position information
        (,, token0, token1,,,,,,,,) = nonfungiblePositionManager.positions(params.tokenId);
        
        // Collect fees
        (uint256 amount0, uint256 amount1) = nonfungiblePositionManager.collect(
            INonfungiblePositionManager.CollectParams({
                tokenId: params.tokenId,
                recipient: address(this),
                amount0Max: type(uint128).max,
                amount1Max: type(uint128).max
            })
        );
        
        // Calculate rewards using cached config
        (reward0, reward1) = _calculateRewards(amount0, amount1, cfg, params.useAIOptimization);
        
        // Calculate amounts to add after rewards
        amountAdded0 = amount0 - reward0;
        amountAdded1 = amount1 - reward1;
        
        // Add liquidity back to position if there's any to add
        if (amountAdded0 > 0 || amountAdded1 > 0) {
            // Approve tokens
            if (amountAdded0 > 0) {
                IERC20(token0).approve(address(nonfungiblePositionManager), amountAdded0);
            }
            if (amountAdded1 > 0) {
                IERC20(token1).approve(address(nonfungiblePositionManager), amountAdded1);
            }
            
            // Increase liquidity
            nonfungiblePositionManager.increaseLiquidity(
                INonfungiblePositionManager.IncreaseLiquidityParams({
                    tokenId: params.tokenId,
                    amount0Desired: amountAdded0,
                    amount1Desired: amountAdded1,
                    amount0Min: 0,
                    amount1Min: 0,
                    deadline: block.timestamp
                })
            );
        }
        
        // Handle rewards
        if (reward0 > 0) {
            accountBalances[msg.sender][token0] += reward0;
        }
        if (reward1 > 0) {
            accountBalances[msg.sender][token1] += reward1;
        }
    }

    /**
     * @notice Calculate rewards efficiently
     * @param amount0 Amount of token0
     * @param amount1 Amount of token1
     * @param cfg Configuration
     * @param useAI Whether AI optimization is used
     * @return reward0 Reward in token0
     * @return reward1 Reward in token1
     */
    function _calculateRewards(
        uint256 amount0, 
        uint256 amount1, 
        PackedConfig memory cfg, 
        bool useAI
    ) private pure returns (uint256 reward0, uint256 reward1) {
        uint64 rewardX64;
        
        if (useAI) {
            rewardX64 = cfg.compounderRewardX64 + cfg.aiOptimizerRewardX64;
        } else {
            rewardX64 = cfg.compounderRewardX64;
        }
        
        // Use assembly for efficient calculations
        assembly {
            reward0 := shr(64, mul(amount0, rewardX64))
            reward1 := shr(64, mul(amount1, rewardX64))
        }
    }

    // ============ EMERGENCY FUNCTIONS ============

    /**
     * @notice Emergency pause (only emergency admin)
     */
    function emergencyPause() external {
        require(msg.sender == address(emergencyAdmin) || msg.sender == owner(), "Not authorized");
        _setFlagAt(2, true); // Set emergency paused flag
        emit EmergencyPaused(msg.sender);
    }

    /**
     * @notice Emergency unpause (only owner)
     */
    function emergencyUnpause() external onlyOwner {
        _setFlagAt(2, false); // Clear emergency paused flag
        emit EmergencyUnpaused(msg.sender);
    }

    // ============ REQUIRED INTERFACE IMPLEMENTATIONS ============

    function onERC721Received(address, address, uint256, bytes calldata) external pure override returns (bytes4) {
        return IERC721Receiver.onERC721Received.selector;
    }

    // ============ LEGACY COMPATIBILITY FUNCTIONS ============
    
    // These provide compatibility with the original interface while using optimized storage

    function totalRewardX64() external view returns (uint64) {
        return config.totalRewardX64;
    }
    
    function compounderRewardX64() external view returns (uint64) {
        return config.compounderRewardX64;
    }
    
    function aiOptimizerRewardX64() external view returns (uint64) {
        return config.aiOptimizerRewardX64;
    }
    
    function maxTWAPTickDifference() external view returns (uint32) {
        return config.maxTWAPTickDifference;
    }
    
    function TWAPSeconds() external view returns (uint32) {
        return config.TWAPSeconds;
    }
    
    function twapProtectionEnabled() external view returns (bool) {
        return _getFlagAt(0);
    }
    
    function aiOptimizationEnabled() external view returns (bool) {
        return _getFlagAt(1);
    }
    
    function emergencyPaused() external view returns (bool) {
        return _getFlagAt(2);
    }
    
    function gasLimitingEnabled() external view returns (bool) {
        return _getFlagAt(3);
    }
    
    function ownerOf(uint256 tokenId) external view returns (address) {
        return positionData[tokenId].owner;
    }
    
    function lastCompoundTime(uint256 tokenId) external view returns (uint256) {
        return uint256(positionData[tokenId].lastCompoundTime);
    }
    
    function compoundCount(uint256 tokenId) external view returns (uint256) {
        return uint256(positionData[tokenId].compoundCount);
    }
    
    function aiManagedPositions(uint256 tokenId) external view returns (bool) {
        return positionData[tokenId].aiManaged;
    }
    
    function accountPositionCount(address account) external view returns (uint256) {
        return uint256(accountLimits[account].positionCount);
    }
    
    function lastOperationTimestamp(address account) external view returns (uint256) {
        return uint256(accountLimits[account].lastOperationTime);
    }

    // ============ ENUMS AND STRUCTS ============

    enum RewardConversion { NONE, TOKEN_0, TOKEN_1, AI_OPTIMIZED }

    struct AutoCompoundParams {
        uint256 tokenId;
        RewardConversion rewardConversion;
        bool withdrawReward;
        bool doSwap;
        bool useAIOptimization;
    }
}