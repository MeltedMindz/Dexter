// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@chainlink/contracts/src/v0.8/interfaces/AggregatorV3Interface.sol";
import "@uniswap/v3-core/contracts/interfaces/IUniswapV3Pool.sol";
import "../libraries/TWAPOracle.sol";

/// @title PriceAggregator
/// @notice Multi-oracle price validation with redundancy and anomaly detection
/// @dev Integrates Chainlink, Uniswap TWAP, and AI anomaly detection
contract PriceAggregator is Ownable, ReentrancyGuard {
    using TWAPOracle for IUniswapV3Pool;
    
    struct OracleConfig {
        address chainlinkFeed;
        address uniswapPool;
        uint32 twapPeriod;
        uint256 maxDeviationBps; // Maximum deviation in basis points
        uint256 stalePriceThreshold; // Seconds after which price is stale
        bool isActive;
    }
    
    struct PriceData {
        uint256 price;
        uint256 timestamp;
        uint256 confidence; // 0-100, higher is more confident
        string source;
    }
    
    // Token pair => Oracle configuration
    mapping(bytes32 => OracleConfig) public oracleConfigs;
    
    // Emergency price feeds (fallback)
    mapping(bytes32 => uint256) public emergencyPrices;
    mapping(bytes32 => uint256) public emergencyPriceTimestamps;
    
    // AI anomaly detection
    address public aiDetectionContract;
    bool public aiDetectionEnabled = true;
    
    // Circuit breaker
    mapping(bytes32 => bool) public circuitBreakerTripped;
    mapping(bytes32 => uint256) public lastValidPrice;
    mapping(bytes32 => uint256) public lastValidTimestamp;
    
    uint256 public constant MIN_CONFIDENCE_THRESHOLD = 60; // Minimum 60% confidence
    uint256 public constant MAX_PRICE_AGE = 1 hours; // Maximum price age
    uint256 public constant EMERGENCY_PRICE_VALIDITY = 6 hours;
    
    event OracleConfigured(
        bytes32 indexed pairHash,
        address indexed chainlinkFeed,
        address indexed uniswapPool
    );
    
    event PriceAnomalyDetected(
        bytes32 indexed pairHash,
        uint256 chainlinkPrice,
        uint256 uniswapPrice,
        uint256 deviation
    );
    
    event CircuitBreakerTripped(bytes32 indexed pairHash, string reason);
    event CircuitBreakerReset(bytes32 indexed pairHash);
    event EmergencyPriceSet(bytes32 indexed pairHash, uint256 price);
    event AIDetectionToggled(bool enabled);
    
    error PriceDeviationTooHigh();
    error StalePriceData();
    error CircuitBreakerActive();
    error InsufficientConfidence();
    error NoValidPrice();
    error InvalidOracle();
    
    constructor() {}
    
    /// @notice Configure oracle for a token pair
    /// @param token0 First token address
    /// @param token1 Second token address
    /// @param chainlinkFeed Chainlink price feed address
    /// @param uniswapPool Uniswap V3 pool address
    /// @param twapPeriod TWAP calculation period
    /// @param maxDeviationBps Maximum allowed deviation between oracles
    /// @param stalePriceThreshold Time after which price is considered stale
    function configureOracle(
        address token0,
        address token1,
        address chainlinkFeed,
        address uniswapPool,
        uint32 twapPeriod,
        uint256 maxDeviationBps,
        uint256 stalePriceThreshold
    ) external onlyOwner {
        bytes32 pairHash = keccak256(abi.encodePacked(token0, token1));
        
        oracleConfigs[pairHash] = OracleConfig({
            chainlinkFeed: chainlinkFeed,
            uniswapPool: uniswapPool,
            twapPeriod: twapPeriod,
            maxDeviationBps: maxDeviationBps,
            stalePriceThreshold: stalePriceThreshold,
            isActive: true
        });
        
        emit OracleConfigured(pairHash, chainlinkFeed, uniswapPool);
    }
    
    /// @notice Get validated price with multi-oracle consensus
    /// @param token0 First token address
    /// @param token1 Second token address
    /// @return price Validated price
    /// @return confidence Confidence level (0-100)
    /// @return isValid Whether price passed all validations
    function getValidatedPrice(address token0, address token1)
        external
        view
        returns (uint256 price, uint256 confidence, bool isValid)
    {
        bytes32 pairHash = keccak256(abi.encodePacked(token0, token1));
        OracleConfig memory config = oracleConfigs[pairHash];
        
        if (!config.isActive) {
            return _getEmergencyPrice(pairHash);
        }
        
        if (circuitBreakerTripped[pairHash]) {
            return (lastValidPrice[pairHash], 50, false);
        }
        
        // Get prices from multiple sources
        PriceData[] memory prices = new PriceData[](3);
        uint256 validPrices = 0;
        
        // Chainlink price
        if (config.chainlinkFeed != address(0)) {
            (uint256 chainlinkPrice, uint256 chainlinkConfidence, bool chainlinkValid) = 
                _getChainlinkPrice(config.chainlinkFeed, config.stalePriceThreshold);
            if (chainlinkValid) {
                prices[validPrices] = PriceData({
                    price: chainlinkPrice,
                    timestamp: block.timestamp,
                    confidence: chainlinkConfidence,
                    source: "Chainlink"
                });
                validPrices++;
            }
        }
        
        // Uniswap TWAP price
        if (config.uniswapPool != address(0)) {
            (uint256 uniswapPrice, uint256 uniswapConfidence, bool uniswapValid) = 
                _getUniswapPrice(config.uniswapPool, config.twapPeriod);
            if (uniswapValid) {
                prices[validPrices] = PriceData({
                    price: uniswapPrice,
                    timestamp: block.timestamp,
                    confidence: uniswapConfidence,
                    source: "Uniswap"
                });
                validPrices++;
            }
        }
        
        // AI detection price (if available)
        if (aiDetectionEnabled && aiDetectionContract != address(0)) {
            (uint256 aiPrice, uint256 aiConfidence, bool aiValid) = 
                _getAIPrice(pairHash);
            if (aiValid) {
                prices[validPrices] = PriceData({
                    price: aiPrice,
                    timestamp: block.timestamp,
                    confidence: aiConfidence,
                    source: "AI"
                });
                validPrices++;
            }
        }
        
        if (validPrices == 0) {
            return _getEmergencyPrice(pairHash);
        }
        
        // Validate price consensus
        (price, confidence, isValid) = _validatePriceConsensus(
            prices,
            validPrices,
            config.maxDeviationBps
        );
        
        if (!isValid) {
            return _getEmergencyPrice(pairHash);
        }
    }
    
    /// @notice Get Chainlink price
    function _getChainlinkPrice(address feedAddress, uint256 staleThreshold)
        internal
        view
        returns (uint256 price, uint256 confidence, bool isValid)
    {
        try AggregatorV3Interface(feedAddress).latestRoundData() returns (
            uint80 roundId,
            int256 answer,
            uint256 startedAt,
            uint256 updatedAt,
            uint80 answeredInRound
        ) {
            if (answer <= 0) {
                return (0, 0, false);
            }
            
            if (block.timestamp > updatedAt + staleThreshold) {
                return (0, 0, false);
            }
            
            // Calculate confidence based on recency
            uint256 age = block.timestamp - updatedAt;
            confidence = age < staleThreshold / 4 ? 95 : 80;
            
            price = uint256(answer);
            isValid = true;
            
        } catch {
            return (0, 0, false);
        }
    }
    
    /// @notice Get Uniswap TWAP price
    function _getUniswapPrice(address poolAddress, uint32 twapPeriod)
        internal
        view
        returns (uint256 price, uint256 confidence, bool isValid)
    {
        try IUniswapV3Pool(poolAddress).verifyTWAP(
            twapPeriod,
            100, // 1% max difference
            false
        ) returns (bool success, int24 twapTick) {
            if (!success) {
                return (0, 0, false);
            }
            
            price = TWAPOracle.getQuoteAtTick(twapTick, 1e18);
            confidence = 85; // TWAP generally reliable but slightly lower than Chainlink
            isValid = true;
            
        } catch {
            return (0, 0, false);
        }
    }
    
    /// @notice Get AI-based price (placeholder for future implementation)
    function _getAIPrice(bytes32 pairHash)
        internal
        view
        returns (uint256 price, uint256 confidence, bool isValid)
    {
        // Placeholder for AI price detection
        // Would integrate with AI detection contract
        return (0, 0, false);
    }
    
    /// @notice Validate price consensus among oracles
    function _validatePriceConsensus(
        PriceData[] memory prices,
        uint256 validCount,
        uint256 maxDeviationBps
    ) internal pure returns (uint256 price, uint256 confidence, bool isValid) {
        if (validCount == 0) {
            return (0, 0, false);
        }
        
        if (validCount == 1) {
            return (prices[0].price, prices[0].confidence, prices[0].confidence >= MIN_CONFIDENCE_THRESHOLD);
        }
        
        // Calculate weighted average
        uint256 totalWeightedPrice = 0;
        uint256 totalWeight = 0;
        uint256 minConfidence = 100;
        
        for (uint256 i = 0; i < validCount; i++) {
            uint256 weight = prices[i].confidence;
            totalWeightedPrice += prices[i].price * weight;
            totalWeight += weight;
            
            if (prices[i].confidence < minConfidence) {
                minConfidence = prices[i].confidence;
            }
        }
        
        price = totalWeightedPrice / totalWeight;
        
        // Check for outliers
        for (uint256 i = 0; i < validCount; i++) {
            uint256 deviation = price > prices[i].price 
                ? ((price - prices[i].price) * 10000) / price
                : ((prices[i].price - price) * 10000) / price;
                
            if (deviation > maxDeviationBps) {
                // Outlier detected, reduce confidence
                minConfidence = minConfidence > 20 ? minConfidence - 20 : 0;
            }
        }
        
        confidence = minConfidence;
        isValid = confidence >= MIN_CONFIDENCE_THRESHOLD;
    }
    
    /// @notice Get emergency price fallback
    function _getEmergencyPrice(bytes32 pairHash)
        internal
        view
        returns (uint256 price, uint256 confidence, bool isValid)
    {
        uint256 emergencyPrice = emergencyPrices[pairHash];
        uint256 emergencyTimestamp = emergencyPriceTimestamps[pairHash];
        
        if (emergencyPrice == 0 || 
            block.timestamp > emergencyTimestamp + EMERGENCY_PRICE_VALIDITY) {
            return (0, 0, false);
        }
        
        return (emergencyPrice, 30, true); // Low confidence for emergency prices
    }
    
    /// @notice Set emergency price (only owner or emergency admin)
    /// @param token0 First token address
    /// @param token1 Second token address
    /// @param price Emergency price to set
    function setEmergencyPrice(address token0, address token1, uint256 price) 
        external 
        onlyOwner 
    {
        bytes32 pairHash = keccak256(abi.encodePacked(token0, token1));
        
        emergencyPrices[pairHash] = price;
        emergencyPriceTimestamps[pairHash] = block.timestamp;
        
        emit EmergencyPriceSet(pairHash, price);
    }
    
    /// @notice Trip circuit breaker for a pair
    /// @param token0 First token address
    /// @param token1 Second token address
    /// @param reason Reason for tripping circuit breaker
    function tripCircuitBreaker(address token0, address token1, string calldata reason) 
        external 
        onlyOwner 
    {
        bytes32 pairHash = keccak256(abi.encodePacked(token0, token1));
        
        circuitBreakerTripped[pairHash] = true;
        
        emit CircuitBreakerTripped(pairHash, reason);
    }
    
    /// @notice Reset circuit breaker for a pair
    /// @param token0 First token address
    /// @param token1 Second token address
    function resetCircuitBreaker(address token0, address token1) 
        external 
        onlyOwner 
    {
        bytes32 pairHash = keccak256(abi.encodePacked(token0, token1));
        
        circuitBreakerTripped[pairHash] = false;
        
        emit CircuitBreakerReset(pairHash);
    }
    
    /// @notice Set AI detection contract
    /// @param _aiDetectionContract Address of AI detection contract
    function setAIDetectionContract(address _aiDetectionContract) external onlyOwner {
        aiDetectionContract = _aiDetectionContract;
    }
    
    /// @notice Toggle AI detection
    /// @param _enabled Whether AI detection is enabled
    function toggleAIDetection(bool _enabled) external onlyOwner {
        aiDetectionEnabled = _enabled;
        emit AIDetectionToggled(_enabled);
    }
    
    /// @notice Check if price feed is healthy
    /// @param token0 First token address
    /// @param token1 Second token address
    /// @return isHealthy Whether the price feed is functioning properly
    /// @return lastUpdate Timestamp of last successful price update
    function checkPriceFeedHealth(address token0, address token1)
        external
        view
        returns (bool isHealthy, uint256 lastUpdate)
    {
        bytes32 pairHash = keccak256(abi.encodePacked(token0, token1));
        
        if (circuitBreakerTripped[pairHash]) {
            return (false, lastValidTimestamp[pairHash]);
        }
        
        (uint256 price, uint256 confidence, bool isValid) = this.getValidatedPrice(token0, token1);
        
        isHealthy = isValid && confidence >= MIN_CONFIDENCE_THRESHOLD;
        lastUpdate = block.timestamp;
    }
    
    /// @notice Batch check multiple price feeds
    /// @param token0s Array of first token addresses
    /// @param token1s Array of second token addresses
    /// @return healthStatuses Array of health statuses
    function batchCheckPriceFeeds(address[] calldata token0s, address[] calldata token1s)
        external
        view
        returns (bool[] memory healthStatuses)
    {
        require(token0s.length == token1s.length, "Array length mismatch");
        
        healthStatuses = new bool[](token0s.length);
        
        for (uint256 i = 0; i < token0s.length; i++) {
            (bool isHealthy, ) = this.checkPriceFeedHealth(token0s[i], token1s[i]);
            healthStatuses[i] = isHealthy;
        }
    }
}