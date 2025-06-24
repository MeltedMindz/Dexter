// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {FixedPoint96} from "@uniswap/v4-core/src/libraries/FixedPoint96.sol";
import {TickMath} from "@uniswap/v4-core/src/libraries/TickMath.sol";

/**
 * @title DexterMath
 * @notice Mathematical utilities for Dexter Protocol's AI-powered liquidity management
 * @dev Provides calculations for volatility, capital efficiency, and optimal positioning
 */
library DexterMath {
    
    uint256 internal constant Q96 = 0x1000000000000000000000000;
    uint256 internal constant BP_DENOMINATOR = 10000;
    uint256 internal constant VOLATILITY_PRECISION = 1e18;
    uint256 internal constant SECONDS_PER_DAY = 86400;
    
    /**
     * @notice Calculate volatility from price movements
     * @param priceHistory Array of recent prices
     * @param timeWeights Time weights for each price (most recent = highest weight)
     * @return volatility Annualized volatility in basis points
     */
    function calculateVolatility(
        uint256[] memory priceHistory,
        uint256[] memory timeWeights
    ) internal pure returns (uint256 volatility) {
        require(priceHistory.length >= 2, "DexterMath: Insufficient price data");
        require(priceHistory.length == timeWeights.length, "DexterMath: Array length mismatch");
        
        uint256 weightedMean = 0;
        uint256 totalWeight = 0;
        uint256 n = priceHistory.length;
        
        // Calculate weighted mean of log returns
        for (uint256 i = 1; i < n; i++) {
            require(priceHistory[i] > 0 && priceHistory[i-1] > 0, "DexterMath: Invalid price");
            
            // Calculate log return: ln(P_t / P_{t-1})
            uint256 logReturn = _ln(priceHistory[i] * VOLATILITY_PRECISION / priceHistory[i-1]);
            weightedMean += logReturn * timeWeights[i];
            totalWeight += timeWeights[i];
        }
        
        if (totalWeight == 0) return 0;
        weightedMean = weightedMean / totalWeight;
        
        // Calculate weighted variance
        uint256 weightedVariance = 0;
        for (uint256 i = 1; i < n; i++) {
            uint256 logReturn = _ln(priceHistory[i] * VOLATILITY_PRECISION / priceHistory[i-1]);
            uint256 deviation = logReturn > weightedMean ? 
                logReturn - weightedMean : 
                weightedMean - logReturn;
            weightedVariance += (deviation * deviation * timeWeights[i]) / totalWeight;
        }
        
        // Annualize volatility (assuming daily data)
        volatility = _sqrt(weightedVariance * SECONDS_PER_DAY) * BP_DENOMINATOR / VOLATILITY_PRECISION;
    }
    
    /**
     * @notice Calculate capital efficiency for a position
     * @param tickLower Lower tick boundary
     * @param tickUpper Upper tick boundary
     * @param currentTick Current pool tick
     * @param liquidity Position liquidity
     * @return efficiency Capital efficiency ratio (0-10000 bp)
     */
    function calculateCapitalEfficiency(
        int24 tickLower,
        int24 tickUpper,
        int24 currentTick,
        uint128 liquidity
    ) internal pure returns (uint256 efficiency) {
        require(tickUpper > tickLower, "DexterMath: Invalid tick range");
        
        // If position is out of range, efficiency is 0
        if (currentTick < tickLower || currentTick >= tickUpper) {
            return 0;
        }
        
        // Calculate range width
        uint256 rangeWidth = uint256(uint24(tickUpper - tickLower));
        
        // Calculate distance from center of range
        int24 rangeMidpoint = (tickLower + tickUpper) / 2;
        uint256 distanceFromCenter = currentTick > rangeMidpoint ? 
            uint256(uint24(currentTick - rangeMidpoint)) :
            uint256(uint24(rangeMidpoint - currentTick));
        
        // Efficiency decreases as position moves away from center and range widens
        uint256 centerEfficiency = rangeWidth > 0 ? 
            (rangeWidth - (distanceFromCenter * 2)) * BP_DENOMINATOR / rangeWidth : 0;
        
        // Apply liquidity density bonus
        uint256 liquidityBonus = liquidity > 1e12 ? 
            _min(1000, liquidity / 1e12) : 0; // Up to 10% bonus for high liquidity
        
        efficiency = _min(BP_DENOMINATOR, centerEfficiency + liquidityBonus);
    }
    
    /**
     * @notice Calculate optimal tick spacing based on volatility
     * @param volatility Current volatility in basis points
     * @param targetConcentration Desired liquidity concentration (0-10000 bp)
     * @return optimalSpacing Recommended tick spacing
     */
    function calculateOptimalTickSpacing(
        uint256 volatility,
        uint256 targetConcentration
    ) internal pure returns (int24 optimalSpacing) {
        require(targetConcentration <= BP_DENOMINATOR, "DexterMath: Invalid concentration");
        
        // Base spacing proportional to volatility
        uint256 baseSpacing = volatility * 2; // 2 ticks per 1bp volatility
        
        // Adjust for target concentration
        uint256 concentrationMultiplier = BP_DENOMINATOR * BP_DENOMINATOR / 
            _max(targetConcentration, 100); // Minimum 1% to avoid division issues
        
        uint256 adjustedSpacing = baseSpacing * concentrationMultiplier / BP_DENOMINATOR;
        
        // Clamp to reasonable bounds (10 to 1000 ticks)
        optimalSpacing = int24(int256(_clamp(adjustedSpacing, 10, 1000)));
    }
    
    /**
     * @notice Calculate optimal fee based on volatility and market conditions
     * @param volatility Current volatility in basis points
     * @param volume24h 24-hour trading volume
     * @param liquidity Total pool liquidity
     * @return optimalFee Recommended fee in basis points
     */
    function calculateOptimalFee(
        uint256 volatility,
        uint256 volume24h,
        uint256 liquidity
    ) internal pure returns (uint24 optimalFee) {
        // Base fee proportional to volatility
        uint256 baseFee = volatility / 10; // 0.1% fee for 10% volatility
        
        // Adjust for volume/liquidity ratio (higher ratio = lower fees to encourage volume)
        uint256 volumeRatio = liquidity > 0 ? volume24h * BP_DENOMINATOR / liquidity : 0;
        uint256 volumeAdjustment = volumeRatio > 1000 ? 
            1000 - _min(500, volumeRatio / 10) : // Reduce fee by up to 50bp for high volume
            1000; // No adjustment for low volume
        
        uint256 adjustedFee = baseFee * volumeAdjustment / 1000;
        
        // Clamp to reasonable bounds (1bp to 100bp)
        optimalFee = uint24(_clamp(adjustedFee, 1, 100));
    }
    
    /**
     * @notice Check if position should be rebalanced based on price movement
     * @param tickLower Lower tick boundary
     * @param tickUpper Upper tick boundary  
     * @param currentTick Current pool tick
     * @param thresholdBp Rebalance threshold in basis points
     * @return shouldRebalance Whether position should be rebalanced
     */
    function shouldRebalance(
        int24 tickLower,
        int24 tickUpper,
        int24 currentTick,
        uint256 thresholdBp
    ) internal pure returns (bool) {
        // Check if out of range
        if (currentTick < tickLower || currentTick >= tickUpper) {
            return true;
        }
        
        // Check if close to range boundaries
        int24 rangeWidth = tickUpper - tickLower;
        int24 threshold = int24(int256(int256(rangeWidth) * int256(thresholdBp) / int256(BP_DENOMINATOR)));
        
        int24 distanceToLower = currentTick - tickLower;
        int24 distanceToUpper = tickUpper - currentTick;
        
        return distanceToLower <= threshold || distanceToUpper <= threshold;
    }
    
    // Internal utility functions
    function _ln(uint256 x) private pure returns (uint256) {
        // Simplified natural logarithm approximation
        // This is a basic implementation - production would use a more accurate library
        require(x > 0, "DexterMath: ln of zero");
        if (x == VOLATILITY_PRECISION) return 0;
        
        uint256 result = 0;
        while (x >= 2 * VOLATILITY_PRECISION) {
            result += 693147180559945309; // ln(2) * 1e18
            x = x / 2;
        }
        
        // Taylor series approximation for ln(1+y) where y = x - 1
        uint256 y = x - VOLATILITY_PRECISION;
        uint256 yPower = y;
        result += yPower; // y
        
        yPower = yPower * y / VOLATILITY_PRECISION;
        result -= yPower / 2; // -y^2/2
        
        yPower = yPower * y / VOLATILITY_PRECISION;
        result += yPower / 3; // y^3/3
        
        return result;
    }
    
    function _sqrt(uint256 x) private pure returns (uint256) {
        if (x == 0) return 0;
        uint256 z = (x + 1) / 2;
        uint256 y = x;
        while (z < y) {
            y = z;
            z = (x / z + z) / 2;
        }
        return y;
    }
    
    function _min(uint256 a, uint256 b) private pure returns (uint256) {
        return a < b ? a : b;
    }
    
    function _max(uint256 a, uint256 b) private pure returns (uint256) {
        return a > b ? a : b;
    }
    
    function _clamp(uint256 value, uint256 min, uint256 max) private pure returns (uint256) {
        return value < min ? min : (value > max ? max : value);
    }
}