// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/access/AccessControl.sol";

/**
 * @title ConfigurationManager
 * @notice Centralized configuration management for Dexter Protocol
 * @dev Replaces hardcoded constants with configurable parameters with proper bounds and governance
 */
contract ConfigurationManager is Ownable, AccessControl {
    
    // ============ ROLES ============
    
    bytes32 public constant GOVERNANCE_ROLE = keccak256("GOVERNANCE_ROLE");
    bytes32 public constant AI_OPTIMIZER_ROLE = keccak256("AI_OPTIMIZER_ROLE");
    bytes32 public constant EMERGENCY_ROLE = keccak256("EMERGENCY_ROLE");
    bytes32 public constant CONFIG_ADMIN_ROLE = keccak256("CONFIG_ADMIN_ROLE");
    
    // ============ CONFIGURATION STRUCTS ============
    
    struct ProtocolConfig {
        uint64 maxRewardX64;              // Maximum reward percentage (Q64 format)
        uint32 maxPositionsPerAddress;    // Maximum positions per user
        uint256 maxGasPerCompound;        // Maximum gas per compound operation
        uint256 dailyGasLimit;            // Daily gas limit per account
        uint256 minOperationInterval;     // Minimum time between operations
        uint256 gasBuffer;                // Gas buffer for safety checks
        bool gasLimitingEnabled;          // Whether gas limiting is active
    }
    
    struct TWAPConfig {
        uint32 maxTWAPTickDifference;     // Maximum TWAP tick difference
        uint32 TWAPSeconds;               // TWAP observation period
        uint32 minTWAPSeconds;            // Minimum TWAP period
        uint32 maxTWAPSeconds;            // Maximum TWAP period
        bool twapProtectionEnabled;       // Whether TWAP protection is active
        uint32 emergencyTWAPOverride;     // Emergency TWAP override period
    }
    
    struct RangeConfig {
        uint256 maxRanges;                // Maximum ranges per vault
        uint256 minRangeWidth;            // Minimum range width in ticks
        uint256 maxRangeWidth;            // Maximum range width in ticks
        uint256 minAllocation;            // Minimum allocation in basis points
        uint256 maxAllocationDeviation;   // Maximum deviation from target allocation
        uint256 rebalanceInterval;        // Default rebalance interval
        uint256 emergencyRebalanceThreshold; // Emergency rebalance threshold
    }
    
    struct FeeConfig {
        uint256 managementFeeBps;         // Management fee in basis points
        uint256 performanceFeeBps;        // Performance fee in basis points
        uint256 maxManagementFee;         // Maximum management fee
        uint256 maxPerformanceFee;        // Maximum performance fee
        address feeRecipient;             // Fee recipient address
        bool dynamicFeesEnabled;          // Whether dynamic fees are enabled
    }
    
    struct FeeTier {
        string tierName;                  // Tier name (Retail, Premium, etc.)
        uint256 managementFeeBps;         // Management fee for this tier
        uint256 performanceFeeBps;        // Performance fee for this tier
        uint256 volumeThreshold;          // Volume threshold for this tier
        uint256 rebatePercentage;         // Rebate percentage
        bool isActive;                    // Whether tier is active
    }
    
    struct MLConfig {
        uint256 maxConfidence;            // Maximum confidence level (10000 = 100%)
        uint256 minPredictionAge;         // Minimum prediction age in seconds
        uint256 maxPredictionAge;         // Maximum prediction age in seconds
        uint256 regimeUpdateThreshold;    // Regime update confidence threshold
        uint256 consensusThreshold;       // Consensus agreement threshold
        uint256 minConsensusProviders;    // Minimum providers for consensus
        uint256 disagreementThreshold;    // Disagreement alert threshold
        bool manualOverrideEnabled;       // Whether manual override is enabled
    }
    
    struct HookConfig {
        uint256 maxPriceHistory;          // Maximum price history points
        uint256 mlUpdateInterval;         // ML update interval in seconds
        uint256 emergencyVolatilityThreshold; // Emergency volatility threshold
        uint24 minFeeBP;                  // Minimum fee in basis points
        uint24 maxFeeBP;                  // Maximum fee in basis points
        uint256 volatilityDecayRate;      // Volatility decay rate
        bool emergencyModeEnabled;        // Whether emergency mode is enabled
    }
    
    struct MarketRegimeThresholds {
        uint256 stableVolatilityMax;      // Maximum volatility for stable regime
        uint256 volatileVolatilityMin;    // Minimum volatility for volatile regime
        uint256 crisisVolatilityMin;      // Minimum volatility for crisis regime
        uint256 trendingVolumeMin;        // Minimum volume for trending regime
        uint256 regimeConfidenceThreshold; // Confidence threshold for regime change
    }
    
    // ============ STORAGE ============
    
    // Network-specific configurations
    mapping(uint256 => ProtocolConfig) public networkConfigs;
    mapping(uint256 => TWAPConfig) public twapConfigs;
    mapping(uint256 => RangeConfig) public rangeConfigs;
    mapping(uint256 => MLConfig) public mlConfigs;
    mapping(uint256 => HookConfig) public hookConfigs;
    mapping(uint256 => MarketRegimeThresholds) public regimeThresholds;
    
    // Fee configurations
    mapping(string => FeeTier) public feeTiers;
    mapping(address => string) public userTiers;
    FeeConfig public defaultFeeConfig;
    
    // Dynamic parameters (can be adjusted by AI)
    mapping(bytes32 => uint256) public dynamicParameters;
    mapping(bytes32 => uint256) public parameterBounds; // min << 128 | max
    
    // Configuration history for audit trail
    mapping(bytes32 => ConfigUpdate[]) public configHistory;
    
    struct ConfigUpdate {
        uint256 timestamp;
        address updatedBy;
        string parameter;
        uint256 oldValue;
        uint256 newValue;
        string reason;
    }
    
    // ============ EVENTS ============
    
    event ConfigurationUpdated(
        string indexed category,
        string indexed parameter,
        uint256 oldValue,
        uint256 newValue,
        address indexed updatedBy,
        string reason
    );
    
    event NetworkConfigUpdated(uint256 indexed networkId, string configType, address indexed updatedBy);
    event FeeTierUpdated(string indexed tierName, address indexed updatedBy);
    event UserTierAssigned(address indexed user, string tierName, address indexed assignedBy);
    event DynamicParameterUpdated(bytes32 indexed parameterKey, uint256 oldValue, uint256 newValue, address indexed updatedBy);
    event ConfigurationLocked(string parameter, uint256 lockDuration, address indexed lockedBy);
    event EmergencyConfigOverride(string parameter, uint256 oldValue, uint256 newValue, address indexed overriddenBy, string reason);
    
    // ============ CONSTRUCTOR ============
    
    constructor() {
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(GOVERNANCE_ROLE, msg.sender);
        _grantRole(CONFIG_ADMIN_ROLE, msg.sender);
        
        // Initialize default configurations
        _initializeDefaultConfigs();
    }
    
    // ============ CONFIGURATION GETTERS ============
    
    /**
     * @notice Get protocol configuration for current network
     * @return config Protocol configuration
     */
    function getProtocolConfig() external view returns (ProtocolConfig memory config) {
        return getProtocolConfig(block.chainid);
    }
    
    /**
     * @notice Get protocol configuration for specific network
     * @param networkId Network chain ID
     * @return config Protocol configuration
     */
    function getProtocolConfig(uint256 networkId) public view returns (ProtocolConfig memory config) {
        config = networkConfigs[networkId];
        
        // Return default if network not configured
        if (config.maxRewardX64 == 0) {
            config = networkConfigs[0]; // Default network config
        }
    }
    
    /**
     * @notice Get TWAP configuration for current network
     * @return config TWAP configuration
     */
    function getTWAPConfig() external view returns (TWAPConfig memory config) {
        return getTWAPConfig(block.chainid);
    }
    
    /**
     * @notice Get TWAP configuration for specific network
     * @param networkId Network chain ID
     * @return config TWAP configuration
     */
    function getTWAPConfig(uint256 networkId) public view returns (TWAPConfig memory config) {
        config = twapConfigs[networkId];
        
        if (config.TWAPSeconds == 0) {
            config = twapConfigs[0]; // Default config
        }
    }
    
    /**
     * @notice Get range configuration for current network
     * @return config Range configuration
     */
    function getRangeConfig() external view returns (RangeConfig memory config) {
        return getRangeConfig(block.chainid);
    }
    
    /**
     * @notice Get range configuration for specific network
     * @param networkId Network chain ID
     * @return config Range configuration
     */
    function getRangeConfig(uint256 networkId) public view returns (RangeConfig memory config) {
        config = rangeConfigs[networkId];
        
        if (config.maxRanges == 0) {
            config = rangeConfigs[0]; // Default config
        }
    }
    
    /**
     * @notice Get fee configuration for user
     * @param user User address
     * @return config Fee configuration for user's tier
     */
    function getFeeConfigForUser(address user) external view returns (FeeTier memory config) {
        string memory tierName = userTiers[user];
        
        if (bytes(tierName).length == 0) {
            tierName = "RETAIL"; // Default tier
        }
        
        return feeTiers[tierName];
    }
    
    /**
     * @notice Get ML configuration for current network
     * @return config ML configuration
     */
    function getMLConfig() external view returns (MLConfig memory config) {
        return getMLConfig(block.chainid);
    }
    
    /**
     * @notice Get ML configuration for specific network
     * @param networkId Network chain ID
     * @return config ML configuration
     */
    function getMLConfig(uint256 networkId) public view returns (MLConfig memory config) {
        config = mlConfigs[networkId];
        
        if (config.maxConfidence == 0) {
            config = mlConfigs[0]; // Default config
        }
    }
    
    /**
     * @notice Get dynamic parameter value
     * @param key Parameter key
     * @return value Parameter value
     */
    function getDynamicParameter(string memory key) external view returns (uint256 value) {
        bytes32 keyHash = keccak256(abi.encodePacked(key));
        return dynamicParameters[keyHash];
    }
    
    // ============ CONFIGURATION SETTERS ============
    
    /**
     * @notice Update protocol configuration
     * @param networkId Network chain ID (0 for default)
     * @param config New protocol configuration
     * @param reason Reason for update
     */
    function updateProtocolConfig(
        uint256 networkId,
        ProtocolConfig calldata config,
        string calldata reason
    ) external onlyRole(GOVERNANCE_ROLE) {
        _validateProtocolConfig(config);
        
        ProtocolConfig memory oldConfig = networkConfigs[networkId];
        networkConfigs[networkId] = config;
        
        _recordConfigUpdate("ProtocolConfig", "maxRewardX64", oldConfig.maxRewardX64, config.maxRewardX64, reason);
        _recordConfigUpdate("ProtocolConfig", "maxPositionsPerAddress", oldConfig.maxPositionsPerAddress, config.maxPositionsPerAddress, reason);
        _recordConfigUpdate("ProtocolConfig", "maxGasPerCompound", oldConfig.maxGasPerCompound, config.maxGasPerCompound, reason);
        
        emit NetworkConfigUpdated(networkId, "ProtocolConfig", msg.sender);
    }
    
    /**
     * @notice Update TWAP configuration
     * @param networkId Network chain ID (0 for default)
     * @param config New TWAP configuration
     * @param reason Reason for update
     */
    function updateTWAPConfig(
        uint256 networkId,
        TWAPConfig calldata config,
        string calldata reason
    ) external onlyRole(GOVERNANCE_ROLE) {
        _validateTWAPConfig(config);
        
        TWAPConfig memory oldConfig = twapConfigs[networkId];
        twapConfigs[networkId] = config;
        
        _recordConfigUpdate("TWAPConfig", "maxTWAPTickDifference", oldConfig.maxTWAPTickDifference, config.maxTWAPTickDifference, reason);
        _recordConfigUpdate("TWAPConfig", "TWAPSeconds", oldConfig.TWAPSeconds, config.TWAPSeconds, reason);
        
        emit NetworkConfigUpdated(networkId, "TWAPConfig", msg.sender);
    }
    
    /**
     * @notice Update range configuration
     * @param networkId Network chain ID (0 for default)
     * @param config New range configuration
     * @param reason Reason for update
     */
    function updateRangeConfig(
        uint256 networkId,
        RangeConfig calldata config,
        string calldata reason
    ) external onlyRole(GOVERNANCE_ROLE) {
        _validateRangeConfig(config);
        
        RangeConfig memory oldConfig = rangeConfigs[networkId];
        rangeConfigs[networkId] = config;
        
        _recordConfigUpdate("RangeConfig", "maxRanges", oldConfig.maxRanges, config.maxRanges, reason);
        _recordConfigUpdate("RangeConfig", "rebalanceInterval", oldConfig.rebalanceInterval, config.rebalanceInterval, reason);
        
        emit NetworkConfigUpdated(networkId, "RangeConfig", msg.sender);
    }
    
    /**
     * @notice Create or update fee tier
     * @param tierName Tier name
     * @param tier Fee tier configuration
     * @param reason Reason for update
     */
    function updateFeeTier(
        string calldata tierName,
        FeeTier calldata tier,
        string calldata reason
    ) external onlyRole(GOVERNANCE_ROLE) {
        _validateFeeTier(tier);
        
        feeTiers[tierName] = tier;
        
        emit FeeTierUpdated(tierName, msg.sender);
        emit ConfigurationUpdated("FeeTier", tierName, 0, tier.managementFeeBps, msg.sender, reason);
    }
    
    /**
     * @notice Assign user to fee tier
     * @param user User address
     * @param tierName Tier name
     */
    function assignUserTier(address user, string calldata tierName) external onlyRole(CONFIG_ADMIN_ROLE) {
        require(feeTiers[tierName].isActive, "Tier not active");
        
        userTiers[user] = tierName;
        
        emit UserTierAssigned(user, tierName, msg.sender);
    }
    
    /**
     * @notice Update dynamic parameter (can be called by AI)
     * @param key Parameter key
     * @param value New value
     * @param reason Reason for update
     */
    function updateDynamicParameter(
        string calldata key,
        uint256 value,
        string calldata reason
    ) external onlyRole(AI_OPTIMIZER_ROLE) {
        bytes32 keyHash = keccak256(abi.encodePacked(key));
        
        // Check bounds
        uint256 bounds = parameterBounds[keyHash];
        if (bounds > 0) {
            uint256 min = bounds >> 128;
            uint256 max = bounds & type(uint128).max;
            require(value >= min && value <= max, "Value out of bounds");
        }
        
        uint256 oldValue = dynamicParameters[keyHash];
        dynamicParameters[keyHash] = value;
        
        emit DynamicParameterUpdated(keyHash, oldValue, value, msg.sender);
        emit ConfigurationUpdated("Dynamic", key, oldValue, value, msg.sender, reason);
    }
    
    /**
     * @notice Set bounds for dynamic parameter
     * @param key Parameter key
     * @param min Minimum value
     * @param max Maximum value
     */
    function setParameterBounds(
        string calldata key,
        uint256 min,
        uint256 max
    ) external onlyRole(GOVERNANCE_ROLE) {
        require(min <= max, "Invalid bounds");
        
        bytes32 keyHash = keccak256(abi.encodePacked(key));
        parameterBounds[keyHash] = (min << 128) | max;
    }
    
    // ============ EMERGENCY FUNCTIONS ============
    
    /**
     * @notice Emergency configuration override
     * @param parameter Parameter name
     * @param value New value
     * @param reason Emergency reason
     */
    function emergencyOverride(
        string calldata parameter,
        uint256 value,
        string calldata reason
    ) external onlyRole(EMERGENCY_ROLE) {
        bytes32 keyHash = keccak256(abi.encodePacked(parameter));
        uint256 oldValue = dynamicParameters[keyHash];
        
        dynamicParameters[keyHash] = value;
        
        emit EmergencyConfigOverride(parameter, oldValue, value, msg.sender, reason);
    }
    
    // ============ VALIDATION FUNCTIONS ============
    
    function _validateProtocolConfig(ProtocolConfig calldata config) internal pure {
        require(config.maxRewardX64 <= type(uint64).max / 20, "Reward too high"); // Max 5%
        require(config.maxPositionsPerAddress >= 10 && config.maxPositionsPerAddress <= 1000, "Invalid position limit");
        require(config.maxGasPerCompound >= 100000 && config.maxGasPerCompound <= 1000000, "Invalid gas limit");
        require(config.minOperationInterval >= 10 && config.minOperationInterval <= 3600, "Invalid interval");
    }
    
    function _validateTWAPConfig(TWAPConfig calldata config) internal pure {
        require(config.TWAPSeconds >= 30 && config.TWAPSeconds <= 3600, "Invalid TWAP period");
        require(config.maxTWAPTickDifference >= 10 && config.maxTWAPTickDifference <= 1000, "Invalid tick difference");
        require(config.minTWAPSeconds <= config.maxTWAPSeconds, "Invalid TWAP bounds");
    }
    
    function _validateRangeConfig(RangeConfig calldata config) internal pure {
        require(config.maxRanges >= 1 && config.maxRanges <= 50, "Invalid range count");
        require(config.minRangeWidth >= 10, "Range too narrow");
        require(config.maxRangeWidth <= 1000000, "Range too wide");
        require(config.minAllocation >= 10 && config.minAllocation <= 10000, "Invalid allocation");
    }
    
    function _validateFeeTier(FeeTier calldata tier) internal pure {
        require(tier.managementFeeBps <= 500, "Management fee too high"); // Max 5%
        require(tier.performanceFeeBps <= 2000, "Performance fee too high"); // Max 20%
        require(tier.rebatePercentage <= 10000, "Invalid rebate");
    }
    
    // ============ INTERNAL FUNCTIONS ============
    
    function _recordConfigUpdate(
        string memory category,
        string memory parameter,
        uint256 oldValue,
        uint256 newValue,
        string memory reason
    ) internal {
        bytes32 key = keccak256(abi.encodePacked(category, parameter));
        
        configHistory[key].push(ConfigUpdate({
            timestamp: block.timestamp,
            updatedBy: msg.sender,
            parameter: parameter,
            oldValue: oldValue,
            newValue: newValue,
            reason: reason
        }));
        
        emit ConfigurationUpdated(category, parameter, oldValue, newValue, msg.sender, reason);
    }
    
    function _initializeDefaultConfigs() internal {
        // Default protocol config (network ID 0)
        networkConfigs[0] = ProtocolConfig({
            maxRewardX64: uint64(2**64 / 50),  // 2%
            maxPositionsPerAddress: 200,
            maxGasPerCompound: 300000,
            dailyGasLimit: 1000000,
            minOperationInterval: 60,
            gasBuffer: 50000,
            gasLimitingEnabled: true
        });
        
        // Default TWAP config
        twapConfigs[0] = TWAPConfig({
            maxTWAPTickDifference: 100,
            TWAPSeconds: 60,
            minTWAPSeconds: 30,
            maxTWAPSeconds: 600,
            twapProtectionEnabled: true,
            emergencyTWAPOverride: 30
        });
        
        // Default range config
        rangeConfigs[0] = RangeConfig({
            maxRanges: 20,
            minRangeWidth: 60,
            maxRangeWidth: 887272,
            minAllocation: 50,
            maxAllocationDeviation: 500,
            rebalanceInterval: 3600,
            emergencyRebalanceThreshold: 2000
        });
        
        // Default fee tiers
        feeTiers["RETAIL"] = FeeTier({
            tierName: "RETAIL",
            managementFeeBps: 25,    // 0.25%
            performanceFeeBps: 1000, // 10%
            volumeThreshold: 0,
            rebatePercentage: 0,
            isActive: true
        });
        
        feeTiers["PREMIUM"] = FeeTier({
            tierName: "PREMIUM",
            managementFeeBps: 20,    // 0.20%
            performanceFeeBps: 800,  // 8%
            volumeThreshold: 100000e18,
            rebatePercentage: 500,   // 5%
            isActive: true
        });
        
        feeTiers["INSTITUTIONAL"] = FeeTier({
            tierName: "INSTITUTIONAL",
            managementFeeBps: 15,    // 0.15%
            performanceFeeBps: 500,  // 5%
            volumeThreshold: 1000000e18,
            rebatePercentage: 1000,  // 10%
            isActive: true
        });
        
        feeTiers["VIP"] = FeeTier({
            tierName: "VIP",
            managementFeeBps: 10,    // 0.10%
            performanceFeeBps: 300,  // 3%
            volumeThreshold: 10000000e18,
            rebatePercentage: 1500,  // 15%
            isActive: true
        });
    }
}