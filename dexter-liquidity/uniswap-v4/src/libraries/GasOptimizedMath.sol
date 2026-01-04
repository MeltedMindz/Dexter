// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title GasOptimizedMath
 * @notice Assembly-optimized mathematical operations for V4 hooks
 * @dev Provides maximum gas efficiency for critical calculations
 */
library GasOptimizedMath {
    
    error DivisionByZero();
    error Overflow();
    
    /**
     * @notice Assembly-optimized volatility calculation
     * @param prices Array of recent prices (max 32 elements for gas efficiency)
     * @return volatility Calculated volatility in basis points
     */
    function calculateVolatility(uint256[32] memory prices, uint256 length) 
        internal 
        pure 
        returns (uint256 volatility) 
    {
        if (length < 2) return 1000; // Default 10% volatility
        
        assembly {
            let sum := 0
            let sumSquared := 0
            let count := 0
            
            // Calculate returns and their squares
            for { let i := 1 } lt(i, length) { i := add(i, 1) } {
                let current := mload(add(prices, mul(i, 0x20)))
                let previous := mload(add(prices, mul(sub(i, 1), 0x20)))
                
                if gt(previous, 0) {
                    // Calculate return: (current - previous) / previous * 10000
                    let returnVal := 0
                    if gt(current, previous) {
                        returnVal := div(mul(sub(current, previous), 10000), previous)
                    }
                    if lt(current, previous) {
                        returnVal := div(mul(sub(previous, current), 10000), previous)
                    }
                    
                    sum := add(sum, returnVal)
                    sumSquared := add(sumSquared, mul(returnVal, returnVal))
                    count := add(count, 1)
                }
            }
            
            if gt(count, 0) {
                // Calculate variance: E[X²] - E[X]²
                let mean := div(sum, count)
                let meanSquared := mul(mean, mean)
                let variance := sub(div(sumSquared, count), meanSquared)
                
                // Approximate square root using Newton's method (simplified)
                volatility := sqrt(variance)
                
                // Scale to annualized volatility (approximate)
                volatility := mul(volatility, 15) // Rough scaling factor
            }
        }
        
        // Clamp volatility to reasonable bounds
        if (volatility < 50) volatility = 50;   // Min 0.5%
        if (volatility > 10000) volatility = 10000; // Max 100%
    }
    
    /**
     * @notice Assembly-optimized square root using Newton's method
     * @param x Input value
     * @return result Square root of x
     */
    function sqrt(uint256 x) internal pure returns (uint256 result) {
        if (x == 0) return 0;
        
        assembly {
            // Initial guess: x / 2
            result := div(x, 2)
            
            // Newton's method: result = (result + x/result) / 2
            // Perform 8 iterations for good precision
            for { let i := 0 } lt(i, 8) { i := add(i, 1) } {
                result := div(add(result, div(x, result)), 2)
            }
        }
    }
    
    /**
     * @notice Gas-optimized fee calculation with clamping
     * @param volatility Current volatility in basis points
     * @param baseMultiplier Base multiplier for fee calculation
     * @return fee Optimal fee in basis points
     */
    function calculateOptimalFee(uint256 volatility, uint256 baseMultiplier) 
        internal 
        pure 
        returns (uint256 fee) 
    {
        assembly {
            // fee = volatility * baseMultiplier / 1000
            fee := div(mul(volatility, baseMultiplier), 1000)
            
            // Clamp to [1, 10000]
            if lt(fee, 1) { fee := 1 }
            if gt(fee, 10000) { fee := 10000 }
        }
    }
    
    /**
     * @notice Assembly-optimized TWAP calculation
     * @param prices Price array
     * @param weights Time weights array
     * @param length Array length
     * @return twap Time-weighted average price
     */
    function calculateTWAP(
        uint256[32] memory prices, 
        uint256[32] memory weights, 
        uint256 length
    ) internal pure returns (uint256 twap) {
        if (length == 0) return 0;
        
        assembly {
            let weightedSum := 0
            let totalWeight := 0
            
            for { let i := 0 } lt(i, length) { i := add(i, 1) } {
                let price := mload(add(prices, mul(i, 0x20)))
                let weight := mload(add(weights, mul(i, 0x20)))
                
                weightedSum := add(weightedSum, mul(price, weight))
                totalWeight := add(totalWeight, weight)
            }
            
            if gt(totalWeight, 0) {
                twap := div(weightedSum, totalWeight)
            }
        }
    }
    
    /**
     * @notice Gas-optimized price impact calculation
     * @param amountIn Input amount
     * @param reserveIn Input reserve
     * @param reserveOut Output reserve
     * @return impact Price impact in basis points
     */
    function calculatePriceImpact(
        uint256 amountIn,
        uint256 reserveIn,
        uint256 reserveOut
    ) internal pure returns (uint256 impact) {
        if (reserveIn == 0 || reserveOut == 0) return 10000; // 100% impact
        
        assembly {
            // Calculate output amount using constant product formula
            let numerator := mul(amountIn, reserveOut)
            let denominator := add(reserveIn, amountIn)
            let amountOut := div(numerator, denominator)
            
            // Calculate expected output without impact
            let expectedOut := div(mul(amountIn, reserveOut), reserveIn)
            
            // Price impact = (expectedOut - amountOut) / expectedOut * 10000
            if gt(expectedOut, 0) {
                let diff := sub(expectedOut, amountOut)
                impact := div(mul(diff, 10000), expectedOut)
            }
            
            // Cap at 100%
            if gt(impact, 10000) { impact := 10000 }
        }
    }
    
    /**
     * @notice Assembly-optimized position range optimization
     * @param currentTick Current pool tick
     * @param volatility Current volatility
     * @param targetConcentration Target liquidity concentration (0-10000)
     * @return lowerTick Optimal lower tick
     * @return upperTick Optimal upper tick
     */
    function calculateOptimalRange(
        int24 currentTick,
        uint256 volatility,
        uint256 targetConcentration
    ) internal pure returns (int24 lowerTick, int24 upperTick) {
        assembly {
            // Calculate range width based on volatility and target concentration
            // Higher volatility = wider range, higher concentration = narrower range
            let baseWidth := mul(volatility, 60) // 60 ticks per 1% volatility
            let concentrationFactor := div(10000, add(targetConcentration, 1))
            let rangeWidth := div(mul(baseWidth, concentrationFactor), 10000)
            
            // Ensure minimum range width
            if lt(rangeWidth, 60) { rangeWidth := 60 }
            
            // Calculate symmetric range around current tick
            let halfRange := div(rangeWidth, 2)
            lowerTick := sub(currentTick, halfRange)
            upperTick := add(currentTick, halfRange)
            
            // Align to tick spacing (assuming 60 tick spacing)
            let tickSpacing := 60
            lowerTick := mul(div(lowerTick, tickSpacing), tickSpacing)
            upperTick := mul(div(add(upperTick, sub(tickSpacing, 1)), tickSpacing), tickSpacing)
        }
    }
    
    /**
     * @notice Gas-optimized capital efficiency calculation
     * @param tickLower Lower tick of position
     * @param tickUpper Upper tick of position
     * @param currentTick Current pool tick
     * @return efficiency Capital efficiency score (0-10000)
     */
    function calculateCapitalEfficiency(
        int24 tickLower,
        int24 tickUpper,
        int24 currentTick
    ) internal pure returns (uint256 efficiency) {
        assembly {
            let rangeWidth := sub(tickUpper, tickLower)
            if eq(rangeWidth, 0) {
                efficiency := 0
                leave
            }
            
            // Check if current tick is in range
            let inRange := and(gte(currentTick, tickLower), lte(currentTick, tickUpper))
            
            if inRange {
                // Calculate distance from center
                let center := div(add(tickLower, tickUpper), 2)
                let distance := 0
                if gt(currentTick, center) {
                    distance := sub(currentTick, center)
                }
                if lt(currentTick, center) {
                    distance := sub(center, currentTick)
                }
                
                // Efficiency decreases with distance from center and range width
                let maxDistance := div(rangeWidth, 2)
                let centerScore := sub(10000, div(mul(distance, 10000), maxDistance))
                let widthScore := div(60000, rangeWidth) // Narrower is better
                
                efficiency := div(add(centerScore, widthScore), 2)
                if gt(efficiency, 10000) { efficiency := 10000 }
            }
        }
    }
    
    /**
     * @notice Assembly-optimized exponential moving average
     * @param currentValue Current value
     * @param previousEMA Previous EMA value
     * @param alpha Smoothing factor (0-10000)
     * @return newEMA Updated EMA value
     */
    function calculateEMA(
        uint256 currentValue,
        uint256 previousEMA,
        uint256 alpha
    ) internal pure returns (uint256 newEMA) {
        assembly {
            // EMA = alpha * current + (1 - alpha) * previous
            let alphaCurrent := div(mul(alpha, currentValue), 10000)
            let oneMinusAlpha := sub(10000, alpha)
            let betaPrevious := div(mul(oneMinusAlpha, previousEMA), 10000)
            
            newEMA := add(alphaCurrent, betaPrevious)
        }
    }
    
    /**
     * @notice Assembly-optimized logarithm base 2 (for tick calculations)
     * @param x Input value (scaled by 1e18)
     * @return result Log2(x) scaled by 1e18
     */
    function log2(uint256 x) internal pure returns (uint256 result) {
        if (x == 0) revert DivisionByZero();
        
        assembly {
            let input := x
            result := 0
            
            // Find the highest bit set
            for { } gt(input, 1) { } {
                input := shr(1, input)
                result := add(result, 1000000000000000000) // Add 1e18
            }
            
            // Fine-tune with fractional part (simplified)
            // This is a rough approximation for gas efficiency
        }
    }
    
    /**
     * @notice Assembly-optimized absolute difference
     * @param a First value
     * @param b Second value
     * @return diff Absolute difference
     */
    function absDiff(uint256 a, uint256 b) internal pure returns (uint256 diff) {
        assembly {
            if gt(a, b) {
                diff := sub(a, b)
            }
            if gte(b, a) {
                diff := sub(b, a)
            }
        }
    }
    
    /**
     * @notice Assembly-optimized percentage calculation
     * @param part Part value
     * @param whole Whole value
     * @return percentage Percentage in basis points
     */
    function calculatePercentage(uint256 part, uint256 whole) 
        internal 
        pure 
        returns (uint256 percentage) 
    {
        if (whole == 0) revert DivisionByZero();
        
        assembly {
            percentage := div(mul(part, 10000), whole)
            if gt(percentage, 10000) { percentage := 10000 }
        }
    }
    
    /**
     * @notice Assembly-optimized min function
     */
    function min(uint256 a, uint256 b) internal pure returns (uint256 result) {
        assembly {
            result := a
            if lt(b, a) { result := b }
        }
    }
    
    /**
     * @notice Assembly-optimized max function
     */
    function max(uint256 a, uint256 b) internal pure returns (uint256 result) {
        assembly {
            result := a
            if gt(b, a) { result := b }
        }
    }
    
    /**
     * @notice Assembly-optimized safe addition with overflow check
     */
    function safeAdd(uint256 a, uint256 b) internal pure returns (uint256 result) {
        assembly {
            result := add(a, b)
            if lt(result, a) {
                mstore(0x00, 0x4e487b71) // Error selector for Panic
                mstore(0x04, 0x11) // Arithmetic overflow
                revert(0x00, 0x24)
            }
        }
    }
    
    /**
     * @notice Assembly-optimized safe subtraction with underflow check
     */
    function safeSub(uint256 a, uint256 b) internal pure returns (uint256 result) {
        assembly {
            if lt(a, b) {
                mstore(0x00, 0x4e487b71) // Error selector for Panic
                mstore(0x04, 0x11) // Arithmetic underflow
                revert(0x00, 0x24)
            }
            result := sub(a, b)
        }
    }
}