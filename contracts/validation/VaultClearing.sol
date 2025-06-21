// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/Pausable.sol";
import "@openzeppelin/contracts/utils/math/Math.sol";
import "@uniswap/v3-core/contracts/interfaces/IUniswapV3Pool.sol";
import "@uniswap/v3-core/contracts/libraries/TickMath.sol";
import "@uniswap/v3-core/contracts/libraries/FullMath.sol";

import "../vaults/IDexterVault.sol";

/// @title VaultClearing - TWAP validation and MEV protection for Dexter vaults
/// @notice Gamma-inspired clearing system with enhanced AI integration and risk management
contract VaultClearing is Ownable, Pausable {
    using Math for uint256;

    // ============ CONSTANTS ============
    
    uint256 public constant MAX_BPS = 10000;
    uint256 public constant PRECISION = 1e18;
    uint32 public constant MIN_TWAP_INTERVAL = 60;      // 1 minute minimum
    uint32 public constant MAX_TWAP_INTERVAL = 86400;   // 24 hours maximum
    uint256 public constant MAX_DEVIATION = 2000;       // 20% maximum deviation
    uint256 public constant MAX_DEPOSIT_DELTA = 15000;  // 150% maximum delta
    
    // ============ ENUMS ============
    
    enum ValidationResult {
        APPROVED,
        REJECTED_TWAP,
        REJECTED_RATIO,
        REJECTED_LIMIT,
        REJECTED_COOLDOWN,
        REJECTED_PAUSE
    }
    
    enum OperationType {
        DEPOSIT,
        WITHDRAW,
        COMPOUND,
        REBALANCE,
        EMERGENCY
    }

    // ============ STRUCTS ============
    
    struct VaultConfig {
        bool enabled;                    // Whether validation is enabled for this vault
        bool twapCheckEnabled;           // Enable TWAP validation
        bool ratioCheckEnabled;          // Enable deposit ratio validation
        bool customConfig;               // Has custom configuration
        uint32 twapInterval;             // TWAP interval in seconds
        uint256 maxDeviation;            // Maximum price deviation (bps)
        uint256 depositDelta;            // Deposit ratio tolerance (bps)
        uint256 maxDepositAmount;        // Maximum single deposit amount
        uint256 maxWithdrawAmount;       // Maximum single withdrawal amount
        uint256 cooldownPeriod;          // Cooldown between large operations
        uint256 dailyOperationLimit;     // Maximum operations per day
        address customValidator;         // Custom validation contract (optional)
    }
    
    struct UserState {
        uint256 lastOperationTime;      // Timestamp of last operation
        uint256 dailyOperationCount;    // Operations count for current day
        uint256 lastDayReset;           // Last day count was reset
        uint256 totalVolume24h;         // Total volume in last 24 hours
        mapping(OperationType => uint256) operationCounts;
        bool isWhitelisted;             // Whitelisted for reduced validation
        bool isBlacklisted;             // Blacklisted from operations
    }
    
    struct TWAPData {
        uint256 currentPrice;
        uint256 twapPrice;
        uint256 deviation;
        bool isValid;
        uint32 interval;
        uint256 timestamp;
    }
    
    struct ValidationContext {
        address vault;
        address user;
        OperationType opType;
        uint256 amount0;
        uint256 amount1;
        uint256 shares;
        bytes additionalData;
    }
    
    struct RatioValidation {
        uint256 expectedRatio;
        uint256 actualRatio;
        uint256 tolerance;
        bool isValid;
        string reason;
    }

    // ============ STATE VARIABLES ============
    
    mapping(address => VaultConfig) public vaultConfigs;
    mapping(address => UserState) public userStates;
    mapping(address => bool) public authorizedVaults;
    mapping(address => mapping(address => bool)) public whitelistedUsers;
    mapping(address => uint256) public lastTWAPUpdate;
    
    // Global settings
    bool public globalPause;
    uint32 public defaultTwapInterval = 3600;           // 1 hour
    uint256 public defaultMaxDeviation = 500;           // 5%
    uint256 public defaultDepositDelta = 10010;         // 0.1%
    uint256 public emergencyDeviation = 1000;           // 10% emergency threshold
    
    // MEV protection
    mapping(bytes32 => uint256) public operationHashes;
    uint256 public mevProtectionDelay = 1;               // 1 block delay
    mapping(address => uint256) public lastBlockOperated;
    
    // AI integration
    address public aiOracle;
    mapping(address => uint256) public aiValidationScores;
    uint256 public minAIScore = 7000;                    // 70% minimum AI approval
    
    // Emergency controls
    mapping(address => bool) public emergencyOperators;
    uint256 public emergencyActivatedAt;
    bool public emergencyMode;

    // ============ EVENTS ============
    
    event VaultConfigured(address indexed vault, VaultConfig config);
    event ValidationResult(address indexed vault, address indexed user, OperationType opType, ValidationResult result);
    event TWAPValidation(address indexed vault, uint256 currentPrice, uint256 twapPrice, uint256 deviation, bool valid);
    event RatioValidation(address indexed vault, address indexed user, uint256 expectedRatio, uint256 actualRatio, bool valid);
    event UserWhitelisted(address indexed vault, address indexed user, bool whitelisted);
    event EmergencyActivated(address indexed activator, string reason);
    event EmergencyDeactivated(address indexed deactivator);
    event MEVProtectionTriggered(address indexed vault, address indexed user, bytes32 opHash);
    event AIValidationScore(address indexed vault, address indexed user, uint256 score, bool approved);

    // ============ MODIFIERS ============
    
    modifier onlyAuthorizedVault() {
        require(authorizedVaults[msg.sender], "Unauthorized vault");
        _;
    }
    
    modifier onlyEmergencyOperator() {
        require(emergencyOperators[msg.sender] || msg.sender == owner(), "Not emergency operator");
        _;
    }
    
    modifier notInEmergencyMode() {
        require(!emergencyMode || emergencyOperators[msg.sender], "Emergency mode active");
        _;
    }
    
    modifier validVault(address vault) {
        require(vault != address(0) && authorizedVaults[vault], "Invalid vault");
        _;
    }

    // ============ CONSTRUCTOR ============
    
    constructor(address _aiOracle) {
        aiOracle = _aiOracle;
        emergencyOperators[owner()] = true;
    }

    // ============ VAULT AUTHORIZATION ============
    
    function authorizeVault(address vault, bool authorized) external onlyOwner {
        authorizedVaults[vault] = authorized;
        
        if (authorized && !vaultConfigs[vault].enabled) {
            // Initialize with default configuration
            _initializeVaultConfig(vault);
        }
    }
    
    function configureVault(address vault, VaultConfig calldata config) 
        external 
        onlyOwner 
        validVault(vault) 
    {
        require(config.twapInterval >= MIN_TWAP_INTERVAL, "TWAP interval too short");
        require(config.twapInterval <= MAX_TWAP_INTERVAL, "TWAP interval too long");
        require(config.maxDeviation <= MAX_DEVIATION, "Max deviation too high");
        require(config.depositDelta <= MAX_DEPOSIT_DELTA, "Deposit delta too high");
        
        vaultConfigs[vault] = config;
        emit VaultConfigured(vault, config);
    }

    // ============ MAIN VALIDATION FUNCTIONS ============
    
    function validateOperation(ValidationContext calldata context) 
        external 
        view 
        returns (ValidationResult result, string memory reason) 
    {
        // Check global conditions first
        if (globalPause) {
            return (ValidationResult.REJECTED_PAUSE, "Global pause active");
        }
        
        if (!authorizedVaults[context.vault]) {
            return (ValidationResult.REJECTED_PAUSE, "Vault not authorized");
        }
        
        VaultConfig memory config = vaultConfigs[context.vault];
        if (!config.enabled) {
            return (ValidationResult.APPROVED, "Validation disabled");
        }
        
        // Check user state
        (bool userValid, string memory userReason) = _validateUserState(context);
        if (!userValid) {
            return (ValidationResult.REJECTED_LIMIT, userReason);
        }
        
        // MEV protection
        if (!_validateMEVProtection(context)) {
            return (ValidationResult.REJECTED_COOLDOWN, "MEV protection triggered");
        }
        
        // TWAP validation
        if (config.twapCheckEnabled) {
            (bool twapValid, string memory twapReason) = _validateTWAP(context.vault, config);
            if (!twapValid) {
                return (ValidationResult.REJECTED_TWAP, twapReason);
            }
        }
        
        // Ratio validation for deposits
        if (config.ratioCheckEnabled && context.opType == OperationType.DEPOSIT) {
            (bool ratioValid, string memory ratioReason) = _validateDepositRatio(context, config);
            if (!ratioValid) {
                return (ValidationResult.REJECTED_RATIO, ratioReason);
            }
        }
        
        // AI validation (if available)
        if (aiOracle != address(0)) {
            (bool aiValid, string memory aiReason) = _validateWithAI(context);
            if (!aiValid) {
                return (ValidationResult.REJECTED_TWAP, aiReason); // Using TWAP rejection for AI
            }
        }
        
        return (ValidationResult.APPROVED, "Validation passed");
    }
    
    function preValidateDeposit(
        address vault,
        address user,
        uint256 amount0,
        uint256 amount1
    ) 
        external 
        view 
        returns (bool valid, string memory reason) 
    {
        ValidationContext memory context = ValidationContext({
            vault: vault,
            user: user,
            opType: OperationType.DEPOSIT,
            amount0: amount0,
            amount1: amount1,
            shares: 0,
            additionalData: ""
        });
        
        (ValidationResult result, string memory resultReason) = this.validateOperation(context);
        valid = (result == ValidationResult.APPROVED);
        reason = resultReason;
    }
    
    function preValidateWithdraw(
        address vault,
        address user,
        uint256 shares
    ) 
        external 
        view 
        returns (bool valid, string memory reason) 
    {
        ValidationContext memory context = ValidationContext({
            vault: vault,
            user: user,
            opType: OperationType.WITHDRAW,
            amount0: 0,
            amount1: 0,
            shares: shares,
            additionalData: ""
        });
        
        (ValidationResult result, string memory resultReason) = this.validateOperation(context);
        valid = (result == ValidationResult.APPROVED);
        reason = resultReason;
    }

    // ============ TWAP VALIDATION ============
    
    function checkTWAP(address vault) 
        external 
        view 
        returns (TWAPData memory twapData) 
    {
        VaultConfig memory config = vaultConfigs[vault];
        IUniswapV3Pool pool = IUniswapV3Pool(IDexterVault(vault).pool());
        
        // Get current price
        (uint160 sqrtPriceX96, , , , , , ) = pool.slot0();
        twapData.currentPrice = _sqrtPriceToPrice(sqrtPriceX96);
        
        // Get TWAP price
        uint32 interval = config.customConfig ? config.twapInterval : defaultTwapInterval;
        twapData.twapPrice = _getTWAPPrice(pool, interval);
        twapData.interval = interval;
        
        // Calculate deviation
        if (twapData.twapPrice > 0) {
            twapData.deviation = _calculateDeviation(twapData.currentPrice, twapData.twapPrice);
            
            uint256 maxDev = config.customConfig ? config.maxDeviation : defaultMaxDeviation;
            twapData.isValid = twapData.deviation <= maxDev;
        }
        
        twapData.timestamp = block.timestamp;
    }
    
    function forceTWAPUpdate(address vault) external onlyEmergencyOperator {
        lastTWAPUpdate[vault] = block.timestamp;
    }

    // ============ RATIO VALIDATION ============
    
    function validateDepositRatio(
        address vault,
        uint256 deposit0,
        uint256 deposit1
    ) 
        external 
        view 
        returns (RatioValidation memory validation) 
    {
        require(deposit0 > 0 && deposit1 > 0, "Zero deposits not allowed");
        
        VaultConfig memory config = vaultConfigs[vault];
        
        // Get vault's current token amounts
        (uint256 total0, uint256 total1) = IDexterVault(vault).getTotalAmounts();
        
        if (total0 == 0 || total1 == 0) {
            // First deposit - any ratio allowed
            validation.isValid = true;
            validation.reason = "First deposit";
            return validation;
        }
        
        // Calculate expected ratio
        validation.expectedRatio = total0 * PRECISION / total1;
        validation.actualRatio = deposit0 * PRECISION / deposit1;
        
        uint256 depositDelta = config.customConfig ? config.depositDelta : defaultDepositDelta;
        validation.tolerance = depositDelta;
        
        // Check if actual ratio is within tolerance
        uint256 minRatio = validation.expectedRatio * (MAX_BPS - depositDelta) / MAX_BPS;
        uint256 maxRatio = validation.expectedRatio * (MAX_BPS + depositDelta) / MAX_BPS;
        
        validation.isValid = (validation.actualRatio >= minRatio && validation.actualRatio <= maxRatio);
        
        if (!validation.isValid) {
            validation.reason = "Deposit ratio outside tolerance";
        } else {
            validation.reason = "Ratio validation passed";
        }
    }

    // ============ USER MANAGEMENT ============
    
    function setUserWhitelist(address vault, address user, bool whitelisted) 
        external 
        onlyOwner 
        validVault(vault) 
    {
        whitelistedUsers[vault][user] = whitelisted;
        userStates[user].isWhitelisted = whitelisted;
        emit UserWhitelisted(vault, user, whitelisted);
    }
    
    function setUserBlacklist(address user, bool blacklisted) external onlyOwner {
        userStates[user].isBlacklisted = blacklisted;
    }
    
    function getUserOperationLimits(address vault, address user) 
        external 
        view 
        returns (
            uint256 dailyRemaining,
            uint256 cooldownRemaining,
            bool canOperate
        ) 
    {
        VaultConfig memory config = vaultConfigs[vault];
        UserState storage state = userStates[user];
        
        // Check daily limits
        if (block.timestamp >= state.lastDayReset + 1 days) {
            dailyRemaining = config.dailyOperationLimit;
        } else {
            dailyRemaining = config.dailyOperationLimit > state.dailyOperationCount ? 
                config.dailyOperationLimit - state.dailyOperationCount : 0;
        }
        
        // Check cooldown
        uint256 timeSinceLastOp = block.timestamp - state.lastOperationTime;
        cooldownRemaining = timeSinceLastOp >= config.cooldownPeriod ? 
            0 : config.cooldownPeriod - timeSinceLastOp;
        
        canOperate = dailyRemaining > 0 && cooldownRemaining == 0 && !state.isBlacklisted;
    }

    // ============ EMERGENCY CONTROLS ============
    
    function activateEmergency(string calldata reason) external onlyEmergencyOperator {
        emergencyMode = true;
        emergencyActivatedAt = block.timestamp;
        emit EmergencyActivated(msg.sender, reason);
    }
    
    function deactivateEmergency() external onlyEmergencyOperator {
        emergencyMode = false;
        emergencyActivatedAt = 0;
        emit EmergencyDeactivated(msg.sender);
    }
    
    function setEmergencyOperator(address operator, bool authorized) external onlyOwner {
        emergencyOperators[operator] = authorized;
    }
    
    function emergencyBypassValidation(address vault, address user, OperationType opType) 
        external 
        onlyEmergencyOperator 
    {
        require(emergencyMode, "Not in emergency mode");
        
        // Log the bypass
        emit ValidationResult(vault, user, opType, ValidationResult.APPROVED);
    }

    // ============ AI INTEGRATION ============
    
    function setAIOracle(address _aiOracle) external onlyOwner {
        aiOracle = _aiOracle;
    }
    
    function updateAIValidationScore(address vault, address user, uint256 score) 
        external 
        onlyOwner 
    {
        require(score <= MAX_BPS, "Invalid score");
        aiValidationScores[_getAIKey(vault, user)] = score;
        
        bool approved = score >= minAIScore;
        emit AIValidationScore(vault, user, score, approved);
    }
    
    function setMinAIScore(uint256 score) external onlyOwner {
        require(score <= MAX_BPS, "Invalid score");
        minAIScore = score;
    }

    // ============ MEV PROTECTION ============
    
    function setMEVProtectionDelay(uint256 blocks) external onlyOwner {
        require(blocks <= 10, "Delay too long");
        mevProtectionDelay = blocks;
    }
    
    function registerOperation(
        address vault,
        address user,
        OperationType opType,
        bytes32 opHash
    ) 
        external 
        onlyAuthorizedVault 
    {
        operationHashes[opHash] = block.number;
        lastBlockOperated[user] = block.number;
        
        // Update user state
        UserState storage state = userStates[user];
        state.lastOperationTime = block.timestamp;
        
        // Reset daily count if needed
        if (block.timestamp >= state.lastDayReset + 1 days) {
            state.dailyOperationCount = 0;
            state.lastDayReset = block.timestamp;
        }
        
        state.dailyOperationCount++;
        state.operationCounts[opType]++;
        
        emit ValidationResult(vault, user, opType, ValidationResult.APPROVED);
    }

    // ============ GLOBAL SETTINGS ============
    
    function setGlobalPause(bool paused) external onlyOwner {
        globalPause = paused;
    }
    
    function setDefaultTwapInterval(uint32 interval) external onlyOwner {
        require(interval >= MIN_TWAP_INTERVAL && interval <= MAX_TWAP_INTERVAL, "Invalid interval");
        defaultTwapInterval = interval;
    }
    
    function setDefaultMaxDeviation(uint256 deviation) external onlyOwner {
        require(deviation <= MAX_DEVIATION, "Deviation too high");
        defaultMaxDeviation = deviation;
    }
    
    function setDefaultDepositDelta(uint256 delta) external onlyOwner {
        require(delta <= MAX_DEPOSIT_DELTA, "Delta too high");
        defaultDepositDelta = delta;
    }

    // ============ VIEW FUNCTIONS ============
    
    function getVaultConfig(address vault) external view returns (VaultConfig memory) {
        return vaultConfigs[vault];
    }
    
    function getUserState(address user) 
        external 
        view 
        returns (
            uint256 lastOperationTime,
            uint256 dailyOperationCount,
            uint256 totalVolume24h,
            bool isWhitelisted,
            bool isBlacklisted
        ) 
    {
        UserState storage state = userStates[user];
        return (
            state.lastOperationTime,
            state.dailyOperationCount,
            state.totalVolume24h,
            state.isWhitelisted,
            state.isBlacklisted
        );
    }
    
    function isOperationAllowed(
        address vault,
        address user,
        OperationType opType
    ) 
        external 
        view 
        returns (bool allowed, string memory reason) 
    {
        ValidationContext memory context = ValidationContext({
            vault: vault,
            user: user,
            opType: opType,
            amount0: 0,
            amount1: 0,
            shares: 0,
            additionalData: ""
        });
        
        (ValidationResult result, string memory resultReason) = this.validateOperation(context);
        allowed = (result == ValidationResult.APPROVED);
        reason = resultReason;
    }

    // ============ INTERNAL FUNCTIONS ============
    
    function _initializeVaultConfig(address vault) internal {
        vaultConfigs[vault] = VaultConfig({
            enabled: true,
            twapCheckEnabled: true,
            ratioCheckEnabled: true,
            customConfig: false,
            twapInterval: defaultTwapInterval,
            maxDeviation: defaultMaxDeviation,
            depositDelta: defaultDepositDelta,
            maxDepositAmount: type(uint256).max,
            maxWithdrawAmount: type(uint256).max,
            cooldownPeriod: 60, // 1 minute default
            dailyOperationLimit: 100,
            customValidator: address(0)
        });
    }
    
    function _validateUserState(ValidationContext memory context) 
        internal 
        view 
        returns (bool valid, string memory reason) 
    {
        UserState storage state = userStates[context.user];
        
        if (state.isBlacklisted) {
            return (false, "User blacklisted");
        }
        
        if (state.isWhitelisted) {
            return (true, "User whitelisted");
        }
        
        VaultConfig memory config = vaultConfigs[context.vault];
        
        // Check daily operation limits
        uint256 dailyCount = state.dailyOperationCount;
        if (block.timestamp >= state.lastDayReset + 1 days) {
            dailyCount = 0; // Would be reset
        }
        
        if (dailyCount >= config.dailyOperationLimit) {
            return (false, "Daily operation limit exceeded");
        }
        
        // Check cooldown period
        if (block.timestamp < state.lastOperationTime + config.cooldownPeriod) {
            return (false, "Cooldown period active");
        }
        
        // Check amount limits
        if (context.opType == OperationType.DEPOSIT) {
            uint256 totalDeposit = context.amount0 + context.amount1; // Simplified
            if (totalDeposit > config.maxDepositAmount) {
                return (false, "Deposit amount too large");
            }
        }
        
        return (true, "User validation passed");
    }
    
    function _validateMEVProtection(ValidationContext memory context) 
        internal 
        view 
        returns (bool) 
    {
        if (mevProtectionDelay == 0) return true;
        
        // Check if user operated in recent blocks
        uint256 lastBlock = lastBlockOperated[context.user];
        if (lastBlock > 0 && block.number < lastBlock + mevProtectionDelay) {
            return false;
        }
        
        return true;
    }
    
    function _validateTWAP(address vault, VaultConfig memory config) 
        internal 
        view 
        returns (bool valid, string memory reason) 
    {
        try this.checkTWAP(vault) returns (TWAPData memory twapData) {
            if (twapData.isValid) {
                return (true, "TWAP validation passed");
            } else {
                return (false, "TWAP deviation too high");
            }
        } catch {
            return (false, "TWAP calculation failed");
        }
    }
    
    function _validateDepositRatio(ValidationContext memory context, VaultConfig memory config) 
        internal 
        view 
        returns (bool valid, string memory reason) 
    {
        if (context.amount0 == 0 || context.amount1 == 0) {
            return (false, "Single-sided deposits not allowed");
        }
        
        try this.validateDepositRatio(context.vault, context.amount0, context.amount1) 
            returns (RatioValidation memory validation) {
            return (validation.isValid, validation.reason);
        } catch {
            return (false, "Ratio validation failed");
        }
    }
    
    function _validateWithAI(ValidationContext memory context) 
        internal 
        view 
        returns (bool valid, string memory reason) 
    {
        uint256 score = aiValidationScores[_getAIKey(context.vault, context.user)];
        
        if (score >= minAIScore) {
            return (true, "AI validation passed");
        } else {
            return (false, "AI validation score too low");
        }
    }
    
    function _getTWAPPrice(IUniswapV3Pool pool, uint32 interval) 
        internal 
        view 
        returns (uint256) 
    {
        if (interval == 0) {
            // Return current price
            (uint160 sqrtPriceX96, , , , , , ) = pool.slot0();
            return _sqrtPriceToPrice(sqrtPriceX96);
        }
        
        uint32[] memory secondsAgos = new uint32[](2);
        secondsAgos[0] = interval;
        secondsAgos[1] = 0;
        
        try pool.observe(secondsAgos) returns (int56[] memory tickCumulatives, uint160[] memory) {
            int24 avgTick = int24((tickCumulatives[1] - tickCumulatives[0]) / int32(interval));
            uint160 sqrtPriceX96 = TickMath.getSqrtRatioAtTick(avgTick);
            return _sqrtPriceToPrice(sqrtPriceX96);
        } catch {
            // Fallback to current price
            (uint160 sqrtPriceX96, , , , , , ) = pool.slot0();
            return _sqrtPriceToPrice(sqrtPriceX96);
        }
    }
    
    function _sqrtPriceToPrice(uint160 sqrtPriceX96) internal pure returns (uint256) {
        return FullMath.mulDiv(uint256(sqrtPriceX96) * uint256(sqrtPriceX96), PRECISION, 2 ** 192);
    }
    
    function _calculateDeviation(uint256 currentPrice, uint256 twapPrice) 
        internal 
        pure 
        returns (uint256) 
    {
        if (twapPrice == 0) return type(uint256).max;
        
        uint256 diff = currentPrice > twapPrice ? currentPrice - twapPrice : twapPrice - currentPrice;
        return diff * MAX_BPS / twapPrice;
    }
    
    function _getAIKey(address vault, address user) internal pure returns (uint256) {
        return uint256(keccak256(abi.encodePacked(vault, user)));
    }
}