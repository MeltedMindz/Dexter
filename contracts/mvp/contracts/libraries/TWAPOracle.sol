// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@uniswap/v3-core/contracts/interfaces/IUniswapV3Pool.sol";
import "../vendor/uniswap/libraries/TickMath.sol";
import "../vendor/uniswap/libraries/FullMath.sol";

/// @title TWAP Oracle Library
/// @notice Provides TWAP protection against MEV attacks
/// @dev Based on Revert Finance's battle-tested TWAP implementation
library TWAPOracle {
    uint32 public constant MIN_TWAP_SECONDS = 60; // Minimum 60 seconds TWAP period
    uint32 public constant DEFAULT_TWAP_SECONDS = 60;
    int24 public constant MAX_TICK_DIFFERENCE = 100; // ~1% price difference
    
    error TWAPCheckFailed();
    error InvalidTWAPPeriod();
    
    /// @notice Verifies that the current tick is within acceptable range of TWAP
    /// @param pool The Uniswap V3 pool to check
    /// @param twapPeriod The period over which to calculate TWAP
    /// @param maxTickDifference Maximum allowed tick difference
    /// @param acceptAiOverride Whether to accept AI override in case of failure
    /// @return success Whether the check passed
    /// @return twapTick The calculated TWAP tick
    function verifyTWAP(
        IUniswapV3Pool pool,
        uint32 twapPeriod,
        int24 maxTickDifference,
        bool acceptAiOverride
    ) internal view returns (bool success, int24 twapTick) {
        if (twapPeriod < MIN_TWAP_SECONDS) {
            revert InvalidTWAPPeriod();
        }
        
        // Get current tick
        (, int24 currentTick, , , , , ) = pool.slot0();
        
        // Calculate TWAP tick
        twapTick = getTWAPTick(pool, twapPeriod);
        
        // Calculate tick difference
        int24 tickDifference = currentTick > twapTick 
            ? currentTick - twapTick 
            : twapTick - currentTick;
            
        // Check if within acceptable range
        success = tickDifference <= maxTickDifference;
        
        // If failed and AI override is not accepted, revert
        if (!success && !acceptAiOverride) {
            revert TWAPCheckFailed();
        }
    }
    
    /// @notice Calculates the TWAP tick for a given period
    /// @param pool The Uniswap V3 pool
    /// @param twapPeriod The period over which to calculate TWAP
    /// @return twapTick The time-weighted average tick
    function getTWAPTick(
        IUniswapV3Pool pool,
        uint32 twapPeriod
    ) internal view returns (int24 twapTick) {
        uint32[] memory secondsAgos = new uint32[](2);
        secondsAgos[0] = twapPeriod;
        secondsAgos[1] = 0;
        
        (int56[] memory tickCumulatives, ) = pool.observe(secondsAgos);
        
        int56 tickCumulativesDelta = tickCumulatives[1] - tickCumulatives[0];
        
        twapTick = int24(tickCumulativesDelta / int56(uint56(twapPeriod)));
        
        // Always round to negative infinity
        if (tickCumulativesDelta < 0 && (tickCumulativesDelta % int56(uint56(twapPeriod)) != 0)) {
            twapTick--;
        }
    }
    
    /// @notice Gets the current and TWAP prices for verification
    /// @param pool The Uniswap V3 pool
    /// @param twapPeriod The period over which to calculate TWAP
    /// @return currentPrice The current price
    /// @return twapPrice The TWAP price
    function getPrices(
        IUniswapV3Pool pool,
        uint32 twapPeriod
    ) internal view returns (uint256 currentPrice, uint256 twapPrice) {
        (, int24 currentTick, , , , , ) = pool.slot0();
        int24 twapTick = getTWAPTick(pool, twapPeriod);
        
        currentPrice = getQuoteAtTick(currentTick, 1e18);
        twapPrice = getQuoteAtTick(twapTick, 1e18);
    }
    
    /// @notice Converts a tick to a price quote
    /// @param tick The tick to convert
    /// @param baseAmount The amount of base token
    /// @return quoteAmount The equivalent amount of quote token
    function getQuoteAtTick(
        int24 tick,
        uint128 baseAmount
    ) internal pure returns (uint256 quoteAmount) {
        uint160 sqrtRatioX96 = TickMath.getSqrtRatioAtTick(tick);
        
        // Calculate amount1 based on amount0
        if (sqrtRatioX96 <= type(uint128).max) {
            uint256 ratioX192 = uint256(sqrtRatioX96) * sqrtRatioX96;
            quoteAmount = FullMath.mulDiv(ratioX192, baseAmount, 1 << 192);
        } else {
            uint256 ratioX128 = FullMath.mulDiv(sqrtRatioX96, sqrtRatioX96, 1 << 64);
            quoteAmount = FullMath.mulDiv(ratioX128, baseAmount, 1 << 128);
        }
    }
    
    /// @notice Validates that price movement is within acceptable bounds
    /// @param currentPrice Current price
    /// @param referencePrice Reference price (e.g., TWAP)
    /// @param maxDeviationBps Maximum deviation in basis points (1% = 100)
    /// @return isValid Whether the price is within bounds
    function validatePriceDeviation(
        uint256 currentPrice,
        uint256 referencePrice,
        uint256 maxDeviationBps
    ) internal pure returns (bool isValid) {
        uint256 priceDiff = currentPrice > referencePrice 
            ? currentPrice - referencePrice 
            : referencePrice - currentPrice;
            
        uint256 maxDiff = (referencePrice * maxDeviationBps) / 10000;
        
        isValid = priceDiff <= maxDiff;
    }
    
    /// @notice Enhanced price validation with multiple oracle support
    /// @param pool The Uniswap V3 pool
    /// @param twapPeriod TWAP calculation period
    /// @param chainlinkPrice Optional Chainlink price for comparison
    /// @param maxDeviationBps Maximum allowed deviation
    /// @return isValid Whether all validations passed
    /// @return confidence Confidence level (0-100)
    function validateWithMultiOracle(
        IUniswapV3Pool pool,
        uint32 twapPeriod,
        uint256 chainlinkPrice,
        uint256 maxDeviationBps
    ) internal view returns (bool isValid, uint256 confidence) {
        // Get current and TWAP prices
        (uint256 currentPrice, uint256 twapPrice) = getPrices(pool, twapPeriod);
        
        // Base validation with TWAP
        bool twapValid = validatePriceDeviation(currentPrice, twapPrice, maxDeviationBps);
        
        if (!twapValid) {
            return (false, 0);
        }
        
        confidence = 70; // Base confidence for TWAP validation
        
        // Additional validation with Chainlink if provided
        if (chainlinkPrice > 0) {
            bool chainlinkValid = validatePriceDeviation(currentPrice, chainlinkPrice, maxDeviationBps);
            bool twapChainlinkValid = validatePriceDeviation(twapPrice, chainlinkPrice, maxDeviationBps * 2);
            
            if (chainlinkValid && twapChainlinkValid) {
                confidence = 95; // High confidence when all oracles agree
            } else if (chainlinkValid || twapChainlinkValid) {
                confidence = 80; // Medium confidence with partial agreement
            } else {
                confidence = 40; // Low confidence with disagreement
            }
        }
        
        isValid = confidence >= 60; // Minimum confidence threshold
    }
}