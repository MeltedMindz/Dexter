// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@chainlink/contracts/src/v0.8/interfaces/AggregatorV3Interface.sol";
import "@uniswap/v3-core/contracts/interfaces/IUniswapV3Pool.sol";
import "@uniswap/v3-core/contracts/libraries/TickMath.sol";
import "@uniswap/v3-periphery/contracts/libraries/OracleLibrary.sol";

/**
 * @title MultiOracleValidator
 * @notice Comprehensive price validation using multiple oracle sources with consensus mechanism
 * @dev Integrates Chainlink, Uniswap TWAP, and instant prices with AI override capabilities
 */
contract MultiOracleValidator is Ownable, ReentrancyGuard {

    // ========================================
    // ORACLE CONFIGURATION
    // ========================================

    struct OracleConfig {
        AggregatorV3Interface chainlinkFeed;
        IUniswapV3Pool uniswapPool;
        address token0;
        address token1;
        uint32 twapPeriod;
        uint256 maxDeviation; // Basis points (10000 = 100%)
        uint256 stalenessThreshold; // Seconds
        bool isActive;
        uint256 weight; // Weight in consensus (0-10000)
    }

    struct PriceData {
        uint256 price;
        uint256 timestamp;
        uint256 confidence; // 0-10000 (0-100%)
        string source;
        bool isValid;
    }

    struct ConsensusResult {
        uint256 consensusPrice;
        uint256 confidence;
        uint256 deviation;
        uint256 participatingOracles;
        bool isValid;
        string reason;
    }

    mapping(address => OracleConfig) public oracleConfigs;
    mapping(address => PriceData) public latestPrices;
    mapping(address => mapping(string => PriceData)) public historicalPrices;
    
    // Consensus parameters
    uint256 public constant MIN_ORACLES_FOR_CONSENSUS = 2;
    uint256 public constant MAX_PRICE_DEVIATION = 500; // 5% max deviation
    uint256 public constant MIN_CONSENSUS_CONFIDENCE = 8000; // 80% minimum confidence
    uint256 public constant CHAINLINK_WEIGHT = 4000; // 40%
    uint256 public constant TWAP_WEIGHT = 3500; // 35%
    uint256 public constant INSTANT_WEIGHT = 2500; // 25%

    // Emergency override
    address public aiOracleOverride;
    mapping(address => bool) public emergencyOverrideActive;
    mapping(address => uint256) public overridePrices;

    // ========================================
    // CIRCUIT BREAKER
    // ========================================

    struct CircuitBreaker {
        uint256 failureCount;
        uint256 lastFailureTime;
        bool isOpen;
        uint256 resetTime;
    }

    mapping(address => mapping(string => CircuitBreaker)) public circuitBreakers;
    uint256 public constant FAILURE_THRESHOLD = 5;
    uint256 public constant CIRCUIT_RESET_TIME = 1 hours;

    // ========================================
    // INITIALIZATION
    // ========================================

    constructor(address _aiOracleOverride) {
        aiOracleOverride = _aiOracleOverride;
    }

    function configureOracle(
        address token,
        address chainlinkFeed,
        address uniswapPool,
        address token0,
        address token1,
        uint32 twapPeriod,
        uint256 maxDeviation,
        uint256 stalenessThreshold
    ) external onlyOwner {
        require(token != address(0), "Invalid token address");
        require(twapPeriod >= 60 && twapPeriod <= 3600, "TWAP period must be 1-60 minutes");
        require(maxDeviation <= 2000, "Max deviation cannot exceed 20%");

        oracleConfigs[token] = OracleConfig({
            chainlinkFeed: AggregatorV3Interface(chainlinkFeed),
            uniswapPool: IUniswapV3Pool(uniswapPool),
            token0: token0,
            token1: token1,
            twapPeriod: twapPeriod,
            maxDeviation: maxDeviation,
            stalenessThreshold: stalenessThreshold,
            isActive: true,
            weight: 10000 // Full weight by default
        });

        emit OracleConfigured(token, chainlinkFeed, uniswapPool, twapPeriod);
    }

    // ========================================
    // PRICE FETCHING
    // ========================================

    function getChainlinkPrice(address token) public view returns (PriceData memory) {
        OracleConfig memory config = oracleConfigs[token];
        if (address(config.chainlinkFeed) == address(0)) {
            return PriceData(0, 0, 0, "chainlink", false);
        }

        try config.chainlinkFeed.latestRoundData() returns (
            uint80 roundId,
            int256 price,
            uint256 startedAt,
            uint256 updatedAt,
            uint80 answeredInRound
        ) {
            // Validate price data
            if (price <= 0) {
                return PriceData(0, 0, 0, "chainlink", false);
            }

            // Check staleness
            if (block.timestamp - updatedAt > config.stalenessThreshold) {
                return PriceData(0, 0, 0, "chainlink", false);
            }

            // Calculate confidence based on recency and round completion
            uint256 confidence = 10000; // Start with 100%
            
            // Reduce confidence for older data
            uint256 age = block.timestamp - updatedAt;
            if (age > 300) { // > 5 minutes
                confidence = confidence * (3600 - age) / 3600; // Linear decrease over 1 hour
            }

            // Ensure minimum confidence
            if (confidence < 5000) confidence = 5000; // Min 50%

            return PriceData(
                uint256(price),
                updatedAt,
                confidence,
                "chainlink",
                true
            );

        } catch {
            return PriceData(0, 0, 0, "chainlink", false);
        }
    }

    function getTWAPPrice(address token) public view returns (PriceData memory) {
        OracleConfig memory config = oracleConfigs[token];
        if (address(config.uniswapPool) == address(0)) {
            return PriceData(0, 0, 0, "twap", false);
        }

        try this._getTWAPPriceInternal(config.uniswapPool, config.twapPeriod) returns (
            uint256 price,
            bool isValid
        ) {
            if (!isValid) {
                return PriceData(0, 0, 0, "twap", false);
            }

            // TWAP confidence is generally high for established pools
            uint256 confidence = 9000; // 90% base confidence

            // Check pool liquidity for confidence adjustment
            try config.uniswapPool.liquidity() returns (uint128 liquidity) {
                if (liquidity < 1e18) {
                    confidence = 6000; // Lower confidence for low liquidity
                } else if (liquidity > 1e20) {
                    confidence = 9500; // Higher confidence for high liquidity
                }
            } catch {
                confidence = 7000; // Default to moderate confidence if can't read liquidity
            }

            return PriceData(
                price,
                block.timestamp,
                confidence,
                "twap",
                true
            );

        } catch {
            return PriceData(0, 0, 0, "twap", false);
        }
    }

    function _getTWAPPriceInternal(
        IUniswapV3Pool pool,
        uint32 period
    ) external view returns (uint256 price, bool isValid) {
        try pool.observe(_buildSecondsArray(period)) returns (
            int56[] memory tickCumulatives,
            uint160[] memory
        ) {
            int24 avgTick = int24((tickCumulatives[1] - tickCumulatives[0]) / int32(period));
            uint160 sqrtPriceX96 = TickMath.getSqrtRatioAtTick(avgTick);
            
            // Convert to price (simplified - would need proper decimals handling in production)
            price = (uint256(sqrtPriceX96) ** 2 * 1e18) >> (96 * 2);
            isValid = true;
            
        } catch {
            price = 0;
            isValid = false;
        }
    }

    function getInstantPrice(address token) public view returns (PriceData memory) {
        OracleConfig memory config = oracleConfigs[token];
        if (address(config.uniswapPool) == address(0)) {
            return PriceData(0, 0, 0, "instant", false);
        }

        try config.uniswapPool.slot0() returns (
            uint160 sqrtPriceX96,
            int24,
            uint16,
            uint16,
            uint16,
            uint8,
            bool
        ) {
            // Convert to price
            uint256 price = (uint256(sqrtPriceX96) ** 2 * 1e18) >> (96 * 2);
            
            // Instant price has moderate confidence due to manipulation risk
            uint256 confidence = 7000; // 70% base confidence
            
            return PriceData(
                price,
                block.timestamp,
                confidence,
                "instant",
                true
            );

        } catch {
            return PriceData(0, 0, 0, "instant", false);
        }
    }

    // ========================================
    // CONSENSUS MECHANISM
    // ========================================

    function getValidatedPrice(address token) external view returns (ConsensusResult memory) {
        // Check for emergency override first
        if (emergencyOverrideActive[token]) {
            return ConsensusResult(
                overridePrices[token],
                10000, // 100% confidence for overrides
                0,
                1,
                true,
                "Emergency override active"
            );
        }

        // Get prices from all sources
        PriceData memory chainlinkPrice = getChainlinkPrice(token);
        PriceData memory twapPrice = getTWAPPrice(token);
        PriceData memory instantPrice = getInstantPrice(token);

        // Filter valid prices
        PriceData[] memory validPrices = new PriceData[](3);
        uint256 validCount = 0;
        uint256 totalWeight = 0;

        if (chainlinkPrice.isValid && !_isCircuitBreakerOpen(token, "chainlink")) {
            validPrices[validCount] = chainlinkPrice;
            validCount++;
            totalWeight += CHAINLINK_WEIGHT;
        }

        if (twapPrice.isValid && !_isCircuitBreakerOpen(token, "twap")) {
            validPrices[validCount] = twapPrice;
            validCount++;
            totalWeight += TWAP_WEIGHT;
        }

        if (instantPrice.isValid && !_isCircuitBreakerOpen(token, "instant")) {
            validPrices[validCount] = instantPrice;
            validCount++;
            totalWeight += INSTANT_WEIGHT;
        }

        // Check minimum oracle requirement
        if (validCount < MIN_ORACLES_FOR_CONSENSUS) {
            return ConsensusResult(
                0,
                0,
                0,
                validCount,
                false,
                "Insufficient valid oracles"
            );
        }

        // Calculate weighted consensus
        return _calculateWeightedConsensus(validPrices, validCount, totalWeight);
    }

    function _calculateWeightedConsensus(
        PriceData[] memory prices,
        uint256 validCount,
        uint256 totalWeight
    ) internal pure returns (ConsensusResult memory) {
        if (validCount == 0) {
            return ConsensusResult(0, 0, 0, 0, false, "No valid prices");
        }

        // Calculate weighted average
        uint256 weightedSum = 0;
        uint256 confidenceSum = 0;

        for (uint256 i = 0; i < validCount; i++) {
            uint256 weight;
            if (keccak256(bytes(prices[i].source)) == keccak256(bytes("chainlink"))) {
                weight = CHAINLINK_WEIGHT;
            } else if (keccak256(bytes(prices[i].source)) == keccak256(bytes("twap"))) {
                weight = TWAP_WEIGHT;
            } else {
                weight = INSTANT_WEIGHT;
            }

            weightedSum += prices[i].price * weight;
            confidenceSum += prices[i].confidence * weight;
        }

        uint256 consensusPrice = weightedSum / totalWeight;
        uint256 averageConfidence = confidenceSum / totalWeight;

        // Calculate maximum deviation from consensus
        uint256 maxDeviation = 0;
        for (uint256 i = 0; i < validCount; i++) {
            uint256 deviation = prices[i].price > consensusPrice 
                ? ((prices[i].price - consensusPrice) * 10000) / consensusPrice
                : ((consensusPrice - prices[i].price) * 10000) / consensusPrice;
            
            if (deviation > maxDeviation) {
                maxDeviation = deviation;
            }
        }

        // Validate consensus
        bool isValid = maxDeviation <= MAX_PRICE_DEVIATION && 
                      averageConfidence >= MIN_CONSENSUS_CONFIDENCE;

        string memory reason = "";
        if (maxDeviation > MAX_PRICE_DEVIATION) {
            reason = "Price deviation too high";
        } else if (averageConfidence < MIN_CONSENSUS_CONFIDENCE) {
            reason = "Confidence too low";
        } else {
            reason = "Consensus valid";
        }

        return ConsensusResult(
            consensusPrice,
            averageConfidence,
            maxDeviation,
            validCount,
            isValid,
            reason
        );
    }

    // ========================================
    // CIRCUIT BREAKER FUNCTIONS
    // ========================================

    function _isCircuitBreakerOpen(address token, string memory source) internal view returns (bool) {
        CircuitBreaker memory breaker = circuitBreakers[token][source];
        
        if (!breaker.isOpen) return false;
        
        return block.timestamp < breaker.resetTime;
    }

    function _recordOracleFailure(address token, string memory source) internal {
        CircuitBreaker storage breaker = circuitBreakers[token][source];
        
        breaker.failureCount++;
        breaker.lastFailureTime = block.timestamp;
        
        if (breaker.failureCount >= FAILURE_THRESHOLD) {
            breaker.isOpen = true;
            breaker.resetTime = block.timestamp + CIRCUIT_RESET_TIME;
            
            emit CircuitBreakerTripped(token, source, breaker.failureCount);
        }
    }

    function _recordOracleSuccess(address token, string memory source) internal {
        CircuitBreaker storage breaker = circuitBreakers[token][source];
        
        if (breaker.isOpen && block.timestamp >= breaker.resetTime) {
            breaker.isOpen = false;
            breaker.failureCount = 0;
            
            emit CircuitBreakerReset(token, source);
        }
    }

    // ========================================
    // AI OVERRIDE FUNCTIONS
    // ========================================

    modifier onlyAIOracle() {
        require(msg.sender == aiOracleOverride, "Only AI oracle can override");
        _;
    }

    function setEmergencyOverride(
        address token,
        uint256 price,
        string calldata reason
    ) external onlyAIOracle {
        require(price > 0, "Invalid override price");
        
        emergencyOverrideActive[token] = true;
        overridePrices[token] = price;
        
        emit EmergencyOverrideActivated(token, price, reason);
    }

    function clearEmergencyOverride(address token) external onlyAIOracle {
        emergencyOverrideActive[token] = false;
        overridePrices[token] = 0;
        
        emit EmergencyOverrideCleared(token);
    }

    function updateAIOracle(address newAIOracle) external onlyOwner {
        require(newAIOracle != address(0), "Invalid AI oracle address");
        
        address oldOracle = aiOracleOverride;
        aiOracleOverride = newAIOracle;
        
        emit AIOracleUpdated(oldOracle, newAIOracle);
    }

    // ========================================
    // UTILITY FUNCTIONS
    // ========================================

    function _buildSecondsArray(uint32 period) internal pure returns (uint32[] memory) {
        uint32[] memory secondsAgos = new uint32[](2);
        secondsAgos[0] = period;
        secondsAgos[1] = 0;
        return secondsAgos;
    }

    function updateOracleWeights(
        uint256 chainlinkWeight,
        uint256 twapWeight,
        uint256 instantWeight
    ) external onlyOwner {
        require(
            chainlinkWeight + twapWeight + instantWeight == 10000,
            "Weights must sum to 10000"
        );
        
        // These would be storage variables in a more complete implementation
        emit OracleWeightsUpdated(chainlinkWeight, twapWeight, instantWeight);
    }

    // ========================================
    // VIEW FUNCTIONS
    // ========================================

    function getAllPrices(address token) external view returns (
        PriceData memory chainlink,
        PriceData memory twap,
        PriceData memory instant,
        ConsensusResult memory consensus
    ) {
        chainlink = getChainlinkPrice(token);
        twap = getTWAPPrice(token);
        instant = getInstantPrice(token);
        consensus = this.getValidatedPrice(token);
    }

    function getOracleHealth(address token) external view returns (
        bool chainlinkHealthy,
        bool twapHealthy,
        bool instantHealthy,
        uint256 chainlinkFailures,
        uint256 twapFailures,
        uint256 instantFailures
    ) {
        chainlinkHealthy = !_isCircuitBreakerOpen(token, "chainlink");
        twapHealthy = !_isCircuitBreakerOpen(token, "twap");
        instantHealthy = !_isCircuitBreakerOpen(token, "instant");
        
        chainlinkFailures = circuitBreakers[token]["chainlink"].failureCount;
        twapFailures = circuitBreakers[token]["twap"].failureCount;
        instantFailures = circuitBreakers[token]["instant"].failureCount;
    }

    function isEmergencyOverrideActive(address token) external view returns (bool) {
        return emergencyOverrideActive[token];
    }

    // ========================================
    // EVENTS
    // ========================================

    event OracleConfigured(
        address indexed token,
        address chainlinkFeed,
        address uniswapPool,
        uint32 twapPeriod
    );

    event CircuitBreakerTripped(
        address indexed token,
        string indexed source,
        uint256 failureCount
    );

    event CircuitBreakerReset(
        address indexed token,
        string indexed source
    );

    event EmergencyOverrideActivated(
        address indexed token,
        uint256 price,
        string reason
    );

    event EmergencyOverrideCleared(address indexed token);

    event AIOracleUpdated(address indexed oldOracle, address indexed newOracle);

    event OracleWeightsUpdated(
        uint256 chainlinkWeight,
        uint256 twapWeight,
        uint256 instantWeight
    );

    event PriceValidationFailed(
        address indexed token,
        string reason,
        uint256 deviation,
        uint256 confidence
    );
}