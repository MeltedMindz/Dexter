// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@uniswap/v3-core/contracts/interfaces/IUniswapV3Pool.sol";
import "@uniswap/v3-core/contracts/libraries/TickMath.sol";
import "@uniswap/v3-periphery/contracts/libraries/OracleLibrary.sol";

/**
 * @title SandwichProtection
 * @notice Advanced protection against sandwich attacks and price manipulation
 * @dev Implements multiple layers of protection including price impact validation,
 *      TWAP comparison, and dynamic slippage protection
 */
contract SandwichProtection is Ownable {
    
    // ========================================
    // PRICE IMPACT VALIDATION
    // ========================================
    
    struct PriceValidation {
        uint256 expectedAmountOut;
        uint256 actualAmountOut;
        uint256 priceImpact; // Basis points (10000 = 100%)
        uint256 maxAllowedImpact; // Basis points
        bool isValid;
    }
    
    // Maximum allowed price impact per operation type
    mapping(bytes4 => uint256) public maxPriceImpact; // Function selector => max impact in basis points
    
    // Default maximum price impacts
    uint256 public constant DEFAULT_MAX_IMPACT = 300; // 3%
    uint256 public constant LARGE_TRADE_MAX_IMPACT = 100; // 1% for large trades
    uint256 public constant EMERGENCY_MAX_IMPACT = 50; // 0.5% during emergencies
    
    modifier validatePriceImpact(
        address tokenIn,
        address tokenOut,
        uint256 amountIn,
        uint256 expectedAmountOut,
        address pool
    ) {
        PriceValidation memory validation = _validatePriceImpact(
            tokenIn,
            tokenOut,
            amountIn,
            expectedAmountOut,
            pool
        );
        
        require(validation.isValid, "SandwichProtection: Excessive price impact");
        _;
    }
    
    function _validatePriceImpact(
        address tokenIn,
        address tokenOut,
        uint256 amountIn,
        uint256 expectedAmountOut,
        address pool
    ) internal view returns (PriceValidation memory) {
        // Get theoretical amount out without price impact
        uint256 theoreticalAmountOut = _getTheoreticalAmountOut(
            tokenIn,
            tokenOut,
            amountIn,
            pool
        );
        
        // Calculate price impact
        uint256 priceImpact = 0;
        if (theoreticalAmountOut > expectedAmountOut) {
            priceImpact = ((theoreticalAmountOut - expectedAmountOut) * 10000) / theoreticalAmountOut;
        }
        
        // Get maximum allowed impact for this function
        bytes4 functionSelector = msg.sig;
        uint256 maxAllowed = maxPriceImpact[functionSelector];
        if (maxAllowed == 0) {
            maxAllowed = DEFAULT_MAX_IMPACT;
        }
        
        // Adjust for trade size
        if (amountIn > _getLargeTradeThreshold(tokenIn)) {
            maxAllowed = LARGE_TRADE_MAX_IMPACT;
        }
        
        return PriceValidation({
            expectedAmountOut: expectedAmountOut,
            actualAmountOut: theoreticalAmountOut,
            priceImpact: priceImpact,
            maxAllowedImpact: maxAllowed,
            isValid: priceImpact <= maxAllowed
        });
    }
    
    function _getTheoreticalAmountOut(
        address tokenIn,
        address tokenOut,
        uint256 amountIn,
        address pool
    ) internal view returns (uint256) {
        IUniswapV3Pool uniswapPool = IUniswapV3Pool(pool);
        (uint160 sqrtPriceX96,,,,,,) = uniswapPool.slot0();
        
        // Calculate theoretical output using current price
        // This is simplified - production would use exact math
        uint256 price = _sqrtPriceToPrice(sqrtPriceX96);
        return (amountIn * price) / 1e18;
    }
    
    function _sqrtPriceToPrice(uint160 sqrtPriceX96) internal pure returns (uint256) {
        return (uint256(sqrtPriceX96) ** 2 * 1e18) >> (96 * 2);
    }
    
    function _getLargeTradeThreshold(address token) internal pure returns (uint256) {
        // This would typically be configurable per token
        return 100 ether; // 100 tokens as default threshold
    }
    
    // ========================================
    // TWAP PROTECTION
    // ========================================
    
    struct TWAPValidation {
        uint256 currentPrice;
        uint256 twapPrice;
        uint256 deviation; // Basis points
        uint256 maxDeviation; // Basis points
        bool isValid;
    }
    
    uint256 public constant MAX_TWAP_DEVIATION = 500; // 5%
    uint32 public constant TWAP_PERIOD = 300; // 5 minutes
    
    modifier validateTWAP(address pool) {
        TWAPValidation memory validation = _validateTWAP(pool);
        require(validation.isValid, "SandwichProtection: Price deviation from TWAP too high");
        _;
    }
    
    function _validateTWAP(address pool) internal view returns (TWAPValidation memory) {
        IUniswapV3Pool uniswapPool = IUniswapV3Pool(pool);
        
        // Get current price
        (uint160 sqrtPriceX96,,,,,,) = uniswapPool.slot0();
        uint256 currentPrice = _sqrtPriceToPrice(sqrtPriceX96);
        
        // Get TWAP price
        uint32[] memory secondsAgos = new uint32[](2);
        secondsAgos[0] = TWAP_PERIOD;
        secondsAgos[1] = 0;
        
        (int56[] memory tickCumulatives,) = uniswapPool.observe(secondsAgos);
        int24 avgTick = int24((tickCumulatives[1] - tickCumulatives[0]) / int32(TWAP_PERIOD));
        uint160 twapSqrtPriceX96 = TickMath.getSqrtRatioAtTick(avgTick);
        uint256 twapPrice = _sqrtPriceToPrice(twapSqrtPriceX96);
        
        // Calculate deviation
        uint256 deviation = 0;
        if (currentPrice > twapPrice) {
            deviation = ((currentPrice - twapPrice) * 10000) / twapPrice;
        } else {
            deviation = ((twapPrice - currentPrice) * 10000) / currentPrice;
        }
        
        return TWAPValidation({
            currentPrice: currentPrice,
            twapPrice: twapPrice,
            deviation: deviation,
            maxDeviation: MAX_TWAP_DEVIATION,
            isValid: deviation <= MAX_TWAP_DEVIATION
        });
    }
    
    // ========================================
    // DYNAMIC SLIPPAGE PROTECTION
    // ========================================
    
    struct SlippageConfig {
        uint256 baseSlippage; // Base slippage in basis points
        uint256 volatilityMultiplier; // Multiplier based on volatility
        uint256 liquidityAdjustment; // Adjustment based on liquidity
        uint256 finalSlippage; // Final calculated slippage
    }
    
    function calculateDynamicSlippage(
        address pool,
        uint256 tradeSize
    ) public view returns (SlippageConfig memory) {
        IUniswapV3Pool uniswapPool = IUniswapV3Pool(pool);
        
        // Base slippage (0.5%)
        uint256 baseSlippage = 50;
        
        // Adjust for volatility
        uint256 volatility = _calculateVolatility(pool);
        uint256 volatilityMultiplier = 10000 + (volatility * 2); // 2x volatility impact
        
        // Adjust for liquidity
        uint128 liquidity = uniswapPool.liquidity();
        uint256 liquidityAdjustment = 10000;
        if (liquidity < 1e18) {
            liquidityAdjustment = 15000; // 50% increase for low liquidity
        } else if (liquidity > 1e20) {
            liquidityAdjustment = 8000; // 20% decrease for high liquidity
        }
        
        // Adjust for trade size
        uint256 sizeMultiplier = 10000;
        uint256 poolTVL = _estimatePoolTVL(pool);
        uint256 tradePercentage = (tradeSize * 10000) / poolTVL;
        
        if (tradePercentage > 1000) { // > 10% of pool
            sizeMultiplier = 20000; // 100% increase
        } else if (tradePercentage > 500) { // > 5% of pool
            sizeMultiplier = 15000; // 50% increase
        }
        
        // Calculate final slippage
        uint256 finalSlippage = baseSlippage;
        finalSlippage = (finalSlippage * volatilityMultiplier) / 10000;
        finalSlippage = (finalSlippage * liquidityAdjustment) / 10000;
        finalSlippage = (finalSlippage * sizeMultiplier) / 10000;
        
        // Cap at maximum
        if (finalSlippage > 1000) { // Max 10%
            finalSlippage = 1000;
        }
        
        return SlippageConfig({
            baseSlippage: baseSlippage,
            volatilityMultiplier: volatilityMultiplier,
            liquidityAdjustment: liquidityAdjustment,
            finalSlippage: finalSlippage
        });
    }
    
    function _calculateVolatility(address pool) internal view returns (uint256) {
        // Simplified volatility calculation
        // In production, would use more sophisticated methods
        try this._getVolatilityFromOracle(pool) returns (uint256 vol) {
            return vol;
        } catch {
            return 100; // Default 1% volatility
        }
    }
    
    function _getVolatilityFromOracle(address pool) external view returns (uint256) {
        IUniswapV3Pool uniswapPool = IUniswapV3Pool(pool);
        
        // Get price observations for volatility calculation
        uint32[] memory secondsAgos = new uint32[](5);
        secondsAgos[0] = 3600; // 1 hour ago
        secondsAgos[1] = 2700; // 45 min ago
        secondsAgos[2] = 1800; // 30 min ago
        secondsAgos[3] = 900;  // 15 min ago
        secondsAgos[4] = 0;    // Now
        
        (int56[] memory tickCumulatives,) = uniswapPool.observe(secondsAgos);
        
        // Calculate price changes and volatility
        uint256 totalVariance = 0;
        for (uint i = 1; i < tickCumulatives.length; i++) {
            int24 tick1 = int24((tickCumulatives[i-1] - tickCumulatives[i]) / 900);
            int24 tick2 = int24((tickCumulatives[i] - tickCumulatives[i]) / 900);
            
            uint256 priceChange = tick1 > tick2 ? 
                uint256(int256(tick1 - tick2)) : 
                uint256(int256(tick2 - tick1));
            
            totalVariance += priceChange ** 2;
        }
        
        return totalVariance / (tickCumulatives.length - 1);
    }
    
    function _estimatePoolTVL(address pool) internal view returns (uint256) {
        // Simplified TVL estimation
        // In production, would use more accurate calculation
        IUniswapV3Pool uniswapPool = IUniswapV3Pool(pool);
        uint128 liquidity = uniswapPool.liquidity();
        
        // Rough approximation: liquidity * 2 (for both tokens)
        return uint256(liquidity) * 2;
    }
    
    // ========================================
    // MEMPOOL MONITORING
    // ========================================
    
    struct MempoolInfo {
        uint256 pendingTransactions;
        uint256 avgGasPrice;
        bool highActivity;
        uint256 suspiciousActivity;
    }
    
    mapping(address => MempoolInfo) public mempoolData;
    uint256 public constant HIGH_ACTIVITY_THRESHOLD = 50;
    
    function updateMempoolData(
        address pool,
        uint256 pendingTxs,
        uint256 avgGas,
        uint256 suspiciousCount
    ) external onlyOwner {
        mempoolData[pool] = MempoolInfo({
            pendingTransactions: pendingTxs,
            avgGasPrice: avgGas,
            highActivity: pendingTxs > HIGH_ACTIVITY_THRESHOLD,
            suspiciousActivity: suspiciousCount
        });
    }
    
    modifier checkMempoolActivity(address pool) {
        MempoolInfo memory info = mempoolData[pool];
        
        if (info.highActivity) {
            require(
                info.suspiciousActivity < 10,
                "SandwichProtection: High suspicious mempool activity"
            );
        }
        _;
    }
    
    // ========================================
    // COMPREHENSIVE PROTECTION
    // ========================================
    
    modifier fullSandwichProtection(
        address tokenIn,
        address tokenOut,
        uint256 amountIn,
        uint256 expectedAmountOut,
        address pool
    ) {
        // 1. Validate price impact
        PriceValidation memory priceValidation = _validatePriceImpact(
            tokenIn,
            tokenOut,
            amountIn,
            expectedAmountOut,
            pool
        );
        require(priceValidation.isValid, "Price impact too high");
        
        // 2. Validate TWAP deviation
        TWAPValidation memory twapValidation = _validateTWAP(pool);
        require(twapValidation.isValid, "Price deviation from TWAP too high");
        
        // 3. Check mempool activity
        MempoolInfo memory mempoolInfo = mempoolData[pool];
        require(
            !mempoolInfo.highActivity || mempoolInfo.suspiciousActivity < 5,
            "High risk mempool conditions"
        );
        
        // 4. Dynamic slippage check
        SlippageConfig memory slippageConfig = calculateDynamicSlippage(pool, amountIn);
        uint256 minAmountOut = (expectedAmountOut * (10000 - slippageConfig.finalSlippage)) / 10000;
        require(
            expectedAmountOut >= minAmountOut,
            "Expected output below dynamic slippage threshold"
        );
        
        _;
    }
    
    // ========================================
    // CONFIGURATION
    // ========================================
    
    function setMaxPriceImpact(bytes4 functionSelector, uint256 maxImpact) external onlyOwner {
        require(maxImpact <= 1000, "Max impact cannot exceed 10%");
        maxPriceImpact[functionSelector] = maxImpact;
        emit MaxPriceImpactUpdated(functionSelector, maxImpact);
    }
    
    function setMultipleFunctionImpacts(
        bytes4[] calldata selectors,
        uint256[] calldata impacts
    ) external onlyOwner {
        require(selectors.length == impacts.length, "Array length mismatch");
        
        for (uint i = 0; i < selectors.length; i++) {
            require(impacts[i] <= 1000, "Max impact cannot exceed 10%");
            maxPriceImpact[selectors[i]] = impacts[i];
            emit MaxPriceImpactUpdated(selectors[i], impacts[i]);
        }
    }
    
    // ========================================
    // EVENTS
    // ========================================
    
    event MaxPriceImpactUpdated(bytes4 indexed functionSelector, uint256 maxImpact);
    event SandwichAttackDetected(
        address indexed pool,
        address indexed user,
        uint256 priceImpact,
        uint256 twapDeviation
    );
    event MempoolActivityAlert(address indexed pool, uint256 suspiciousCount);
    
    // ========================================
    // VIEW FUNCTIONS
    // ========================================
    
    function validateSwap(
        address tokenIn,
        address tokenOut,
        uint256 amountIn,
        uint256 expectedAmountOut,
        address pool
    ) external view returns (
        bool isValid,
        string memory reason,
        uint256 recommendedSlippage
    ) {
        // Check price impact
        PriceValidation memory priceValidation = _validatePriceImpact(
            tokenIn,
            tokenOut,
            amountIn,
            expectedAmountOut,
            pool
        );
        
        if (!priceValidation.isValid) {
            return (false, "Excessive price impact", 0);
        }
        
        // Check TWAP
        TWAPValidation memory twapValidation = _validateTWAP(pool);
        if (!twapValidation.isValid) {
            return (false, "Price deviation from TWAP too high", 0);
        }
        
        // Calculate recommended slippage
        SlippageConfig memory slippageConfig = calculateDynamicSlippage(pool, amountIn);
        
        return (true, "Swap looks safe", slippageConfig.finalSlippage);
    }
    
    function getPoolRiskMetrics(address pool) external view returns (
        uint256 currentVolatility,
        uint256 liquidityDepth,
        uint256 twapDeviation,
        bool isHighRisk
    ) {
        currentVolatility = _calculateVolatility(pool);
        
        IUniswapV3Pool uniswapPool = IUniswapV3Pool(pool);
        liquidityDepth = uint256(uniswapPool.liquidity());
        
        TWAPValidation memory twapValidation = _validateTWAP(pool);
        twapDeviation = twapValidation.deviation;
        
        isHighRisk = currentVolatility > 500 || // > 5% volatility
                     liquidityDepth < 1e18 || // Low liquidity
                     twapDeviation > 200; // > 2% TWAP deviation
        
        return (currentVolatility, liquidityDepth, twapDeviation, isHighRisk);
    }
}