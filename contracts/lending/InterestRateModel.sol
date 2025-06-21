// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/access/Ownable.sol";

/// @title InterestRateModel
/// @notice Dynamic interest rate model for DexterVault
/// @dev Implements utilization-based interest rates with AI optimization
contract InterestRateModel is Ownable {
    
    struct RateParams {
        uint256 baseRateX96; // Base interest rate (96-bit fixed point)
        uint256 multiplierX96; // Rate multiplier based on utilization
        uint256 jumpMultiplierX96; // Jump multiplier after kink
        uint256 kinkX96; // Utilization rate where jump multiplier kicks in
        uint256 reserveFactorX96; // Reserve factor for protocol
    }
    
    struct UtilizationData {
        uint256 totalBorrows;
        uint256 totalReserves;
        uint256 totalSupply;
        uint256 timestamp;
    }
    
    // Rate parameters
    RateParams public rateParams;
    
    // Historical utilization for AI optimization
    mapping(uint256 => UtilizationData) public utilizationHistory;
    uint256 public currentHistoryIndex;
    uint256 public constant HISTORY_LENGTH = 100;
    
    // AI optimization
    address public aiRateOptimizer;
    bool public aiOptimizationEnabled = true;
    uint256 public lastAIOptimization;
    uint256 public constant AI_OPTIMIZATION_COOLDOWN = 1 hours;
    
    // Constants
    uint256 constant Q96 = 2**96;
    uint256 constant BLOCKS_PER_YEAR = 2102400; // Assuming 15 second blocks
    uint256 constant MAX_UTILIZATION_RATE = Q96; // 100%
    uint256 constant MAX_INTEREST_RATE = Q96 * 100 / 100; // 100% APR max
    
    // Events
    event RateParamsUpdated(
        uint256 baseRate,
        uint256 multiplier,
        uint256 jumpMultiplier,
        uint256 kink
    );
    
    event AIOptimization(
        uint256 oldBaseRate,
        uint256 newBaseRate,
        uint256 utilizationRate,
        string reason
    );
    
    event UtilizationRecorded(
        uint256 indexed timestamp,
        uint256 utilizationRate,
        uint256 borrowRate,
        uint256 supplyRate
    );
    
    error InvalidRateParams();
    error AIOptimizationTooFrequent();
    error UnauthorizedAIAccess();
    
    constructor() {
        // Initialize with conservative defaults
        rateParams = RateParams({
            baseRateX96: Q96 * 2 / 100, // 2% base rate
            multiplierX96: Q96 * 15 / 100, // 15% multiplier
            jumpMultiplierX96: Q96 * 150 / 100, // 150% jump multiplier
            kinkX96: Q96 * 80 / 100, // 80% kink point
            reserveFactorX96: Q96 * 10 / 100 // 10% reserve factor
        });
    }
    
    /// @notice Calculate current interest rates
    /// @param totalBorrows Total amount borrowed
    /// @param totalReserves Total reserves
    /// @param totalSupply Total supply in the vault
    /// @return borrowRate Current borrow rate (annual)
    /// @return supplyRate Current supply rate (annual)
    /// @return utilizationRate Current utilization rate
    function getInterestRates(
        uint256 totalBorrows,
        uint256 totalReserves,
        uint256 totalSupply
    ) external view returns (uint256 borrowRate, uint256 supplyRate, uint256 utilizationRate) {
        
        utilizationRate = _calculateUtilizationRate(totalBorrows, totalSupply);
        borrowRate = _calculateBorrowRate(utilizationRate);
        supplyRate = _calculateSupplyRate(borrowRate, utilizationRate, rateParams.reserveFactorX96);
    }
    
    /// @notice Calculate utilization rate
    /// @param totalBorrows Total borrowed amount
    /// @param totalSupply Total supplied amount
    /// @return utilizationRate Utilization rate (96-bit fixed point)
    function _calculateUtilizationRate(
        uint256 totalBorrows,
        uint256 totalSupply
    ) internal pure returns (uint256 utilizationRate) {
        if (totalSupply == 0) {
            return 0;
        }
        
        utilizationRate = (totalBorrows * Q96) / totalSupply;
        
        // Cap at 100% utilization
        if (utilizationRate > MAX_UTILIZATION_RATE) {
            utilizationRate = MAX_UTILIZATION_RATE;
        }
    }
    
    /// @notice Calculate borrow rate based on utilization
    /// @param utilizationRate Current utilization rate
    /// @return borrowRate Annual borrow rate
    function _calculateBorrowRate(uint256 utilizationRate) internal view returns (uint256 borrowRate) {
        if (utilizationRate <= rateParams.kinkX96) {
            // Below kink: baseRate + utilizationRate * multiplier
            borrowRate = rateParams.baseRateX96 + 
                (utilizationRate * rateParams.multiplierX96) / Q96;
        } else {
            // Above kink: baseRate + kink * multiplier + (utilizationRate - kink) * jumpMultiplier
            uint256 baseAndKinkRate = rateParams.baseRateX96 + 
                (rateParams.kinkX96 * rateParams.multiplierX96) / Q96;
            
            uint256 excessUtilization = utilizationRate - rateParams.kinkX96;
            uint256 jumpRate = (excessUtilization * rateParams.jumpMultiplierX96) / Q96;
            
            borrowRate = baseAndKinkRate + jumpRate;
        }
        
        // Cap at maximum rate
        if (borrowRate > MAX_INTEREST_RATE) {
            borrowRate = MAX_INTEREST_RATE;
        }
    }
    
    /// @notice Calculate supply rate
    /// @param borrowRate Current borrow rate
    /// @param utilizationRate Current utilization rate
    /// @param reserveFactorX96 Reserve factor
    /// @return supplyRate Annual supply rate
    function _calculateSupplyRate(
        uint256 borrowRate,
        uint256 utilizationRate,
        uint256 reserveFactorX96
    ) internal pure returns (uint256 supplyRate) {
        // supplyRate = borrowRate * utilizationRate * (1 - reserveFactor)
        uint256 oneMinusReserveFactor = Q96 - reserveFactorX96;
        supplyRate = (borrowRate * utilizationRate * oneMinusReserveFactor) / (Q96 * Q96);
    }
    
    /// @notice Update rate parameters (only owner)
    /// @param baseRateX96 New base rate
    /// @param multiplierX96 New multiplier
    /// @param jumpMultiplierX96 New jump multiplier
    /// @param kinkX96 New kink point
    /// @param reserveFactorX96 New reserve factor
    function updateRateParams(
        uint256 baseRateX96,
        uint256 multiplierX96,
        uint256 jumpMultiplierX96,
        uint256 kinkX96,
        uint256 reserveFactorX96
    ) external onlyOwner {
        // Validate parameters
        if (kinkX96 > Q96 || reserveFactorX96 > Q96) {
            revert InvalidRateParams();
        }
        
        // Test that max rate is reasonable
        uint256 maxRate = baseRateX96 + (kinkX96 * multiplierX96) / Q96 + 
            ((Q96 - kinkX96) * jumpMultiplierX96) / Q96;
        if (maxRate > MAX_INTEREST_RATE) {
            revert InvalidRateParams();
        }
        
        rateParams = RateParams({
            baseRateX96: baseRateX96,
            multiplierX96: multiplierX96,
            jumpMultiplierX96: jumpMultiplierX96,
            kinkX96: kinkX96,
            reserveFactorX96: reserveFactorX96
        });
        
        emit RateParamsUpdated(baseRateX96, multiplierX96, jumpMultiplierX96, kinkX96);
    }
    
    /// @notice Record utilization data for AI analysis
    /// @param totalBorrows Current total borrows
    /// @param totalReserves Current total reserves
    /// @param totalSupply Current total supply
    function recordUtilization(
        uint256 totalBorrows,
        uint256 totalReserves,
        uint256 totalSupply
    ) external {
        uint256 utilizationRate = _calculateUtilizationRate(totalBorrows, totalSupply);
        uint256 borrowRate = _calculateBorrowRate(utilizationRate);
        uint256 supplyRate = _calculateSupplyRate(borrowRate, utilizationRate, rateParams.reserveFactorX96);
        
        // Store in circular buffer
        utilizationHistory[currentHistoryIndex] = UtilizationData({
            totalBorrows: totalBorrows,
            totalReserves: totalReserves,
            totalSupply: totalSupply,
            timestamp: block.timestamp
        });
        
        currentHistoryIndex = (currentHistoryIndex + 1) % HISTORY_LENGTH;
        
        emit UtilizationRecorded(block.timestamp, utilizationRate, borrowRate, supplyRate);
    }
    
    /// @notice AI-driven rate optimization
    /// @param newBaseRateX96 New base rate suggested by AI
    /// @param reason Reason for the optimization
    function optimizeRates(uint256 newBaseRateX96, string calldata reason) external {
        if (msg.sender != aiRateOptimizer) {
            revert UnauthorizedAIAccess();
        }
        
        if (!aiOptimizationEnabled) {
            return;
        }
        
        if (block.timestamp < lastAIOptimization + AI_OPTIMIZATION_COOLDOWN) {
            revert AIOptimizationTooFrequent();
        }
        
        // Validate new rate is reasonable
        if (newBaseRateX96 > Q96 * 20 / 100) { // Max 20% base rate
            return;
        }
        
        uint256 oldBaseRate = rateParams.baseRateX96;
        rateParams.baseRateX96 = newBaseRateX96;
        lastAIOptimization = block.timestamp;
        
        emit AIOptimization(oldBaseRate, newBaseRateX96, 0, reason);
    }
    
    /// @notice Get historical utilization data
    /// @param lookback Number of periods to look back
    /// @return timestamps Array of timestamps
    /// @return utilizationRates Array of utilization rates
    /// @return borrowRates Array of borrow rates
    function getUtilizationHistory(uint256 lookback) 
        external 
        view 
        returns (
            uint256[] memory timestamps,
            uint256[] memory utilizationRates,
            uint256[] memory borrowRates
        ) 
    {
        if (lookback > HISTORY_LENGTH) {
            lookback = HISTORY_LENGTH;
        }
        
        timestamps = new uint256[](lookback);
        utilizationRates = new uint256[](lookback);
        borrowRates = new uint256[](lookback);
        
        uint256 startIndex = currentHistoryIndex >= lookback ? 
            currentHistoryIndex - lookback : 
            HISTORY_LENGTH + currentHistoryIndex - lookback;
        
        for (uint256 i = 0; i < lookback; i++) {
            uint256 index = (startIndex + i) % HISTORY_LENGTH;
            UtilizationData memory data = utilizationHistory[index];
            
            timestamps[i] = data.timestamp;
            
            if (data.totalSupply > 0) {
                utilizationRates[i] = _calculateUtilizationRate(data.totalBorrows, data.totalSupply);
                borrowRates[i] = _calculateBorrowRate(utilizationRates[i]);
            }
        }
    }
    
    /// @notice Calculate APY from APR
    /// @param aprX96 Annual percentage rate (96-bit fixed point)
    /// @param compoundingFrequency Number of compounding periods per year
    /// @return apyX96 Annual percentage yield
    function calculateAPY(uint256 aprX96, uint256 compoundingFrequency) 
        external 
        pure 
        returns (uint256 apyX96) 
    {
        if (compoundingFrequency == 0) {
            return aprX96;
        }
        
        // APY = (1 + APR/n)^n - 1, where n is compounding frequency
        uint256 ratePerPeriod = aprX96 / compoundingFrequency;
        uint256 onePlusRate = Q96 + ratePerPeriod;
        
        // Calculate (1 + rate)^n using approximation for gas efficiency
        uint256 compounded = _power(onePlusRate, compoundingFrequency);
        
        apyX96 = compounded - Q96;
    }
    
    /// @notice Calculate power using binary exponentiation (simplified)
    function _power(uint256 base, uint256 exponent) internal pure returns (uint256 result) {
        result = Q96;
        uint256 b = base;
        
        while (exponent > 0) {
            if (exponent & 1 == 1) {
                result = (result * b) / Q96;
            }
            b = (b * b) / Q96;
            exponent >>= 1;
        }
    }
    
    /// @notice Get current rate parameters
    /// @return baseRate Current base rate
    /// @return multiplier Current multiplier
    /// @return jumpMultiplier Current jump multiplier
    /// @return kink Current kink point
    /// @return reserveFactor Current reserve factor
    function getCurrentRateParams() 
        external 
        view 
        returns (
            uint256 baseRate,
            uint256 multiplier,
            uint256 jumpMultiplier,
            uint256 kink,
            uint256 reserveFactor
        ) 
    {
        return (
            rateParams.baseRateX96,
            rateParams.multiplierX96,
            rateParams.jumpMultiplierX96,
            rateParams.kinkX96,
            rateParams.reserveFactorX96
        );
    }
    
    /// @notice Set AI rate optimizer
    /// @param _aiRateOptimizer Address of AI optimizer contract
    function setAIRateOptimizer(address _aiRateOptimizer) external onlyOwner {
        aiRateOptimizer = _aiRateOptimizer;
    }
    
    /// @notice Toggle AI optimization
    /// @param _enabled Whether AI optimization is enabled
    function toggleAIOptimization(bool _enabled) external onlyOwner {
        aiOptimizationEnabled = _enabled;
    }
    
    /// @notice Emergency rate adjustment (only owner)
    /// @param newBaseRateX96 Emergency base rate
    function emergencyRateAdjustment(uint256 newBaseRateX96) external onlyOwner {
        require(newBaseRateX96 <= Q96 * 50 / 100, "Rate too high"); // Max 50% emergency rate
        
        uint256 oldRate = rateParams.baseRateX96;
        rateParams.baseRateX96 = newBaseRateX96;
        
        emit AIOptimization(oldRate, newBaseRateX96, 0, "Emergency adjustment");
    }
}