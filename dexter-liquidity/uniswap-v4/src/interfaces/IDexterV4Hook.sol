// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {IHooks} from "@uniswap/v4-core/src/interfaces/IHooks.sol";
import {PoolKey} from "@uniswap/v4-core/src/types/PoolKey.sol";
import {BeforeSwapDelta, BeforeSwapDeltaLibrary} from "@uniswap/v4-core/src/types/BeforeSwapDelta.sol";
import {BalanceDelta} from "@uniswap/v4-core/src/types/BalanceDelta.sol";

/**
 * @title IDexterV4Hook
 * @notice Interface for Dexter Protocol's AI-powered Uniswap V4 hooks
 * @dev Extends IHooks with Dexter-specific functionality for ML-driven liquidity management
 */
interface IDexterV4Hook is IHooks {
    
    // Events
    event VolatilityUpdated(bytes32 indexed poolId, uint256 newVolatility, uint256 timestamp);
    event FeeAdjusted(bytes32 indexed poolId, uint24 oldFee, uint24 newFee, string reason);
    event MLPredictionReceived(bytes32 indexed poolId, uint8 regime, uint256 confidence, uint256 timestamp);
    event PositionRebalanceTriggered(bytes32 indexed poolId, address indexed owner, int24 oldLower, int24 oldUpper, int24 newLower, int24 newUpper);
    event EmergencyModeActivated(bytes32 indexed poolId, string reason);
    
    // Errors
    error InvalidVolatility();
    error FeeTooHigh();
    error FeeTooLow(); 
    error UnauthorizedCaller();
    error PoolNotSupported();
    error EmergencyModeActive();
    error MLServiceUnavailable();
    
    // Market Regime Detection
    enum MarketRegime {
        STABLE,       // Low volatility, tight spreads
        TRENDING_UP,  // Consistent upward price movement
        TRENDING_DOWN,// Consistent downward price movement
        RANGING,      // Sideways price action
        VOLATILE,     // High volatility, wide spreads
        CRISIS        // Extreme volatility, emergency mode
    }
    
    // ML Prediction Data
    struct MLPrediction {
        MarketRegime regime;
        uint256 confidence;        // Confidence score (0-10000 basis points)
        uint256 predictedVolatility;
        uint256 optimalFee;
        uint256 timestamp;
        bool isValid;
    }
    
    // Pool State
    struct PoolState {
        uint256 currentVolatility;
        uint24 currentFee;
        MarketRegime currentRegime;
        uint256 lastMLUpdate;
        uint256 lastRebalance;
        bool emergencyMode;
        uint256 totalSwapVolume;
        uint256 avgPositionSize;
    }
    
    /**
     * @notice Get current market regime for a pool
     * @param poolKey The pool identifier
     * @return regime Current market regime
     * @return confidence Confidence level (0-10000 bp)
     */
    function getMarketRegime(PoolKey calldata poolKey) 
        external 
        view 
        returns (MarketRegime regime, uint256 confidence);
    
    /**
     * @notice Get current pool state including volatility and fees
     * @param poolKey The pool identifier
     * @return state Current pool state
     */
    function getPoolState(PoolKey calldata poolKey) 
        external 
        view 
        returns (PoolState memory state);
    
    /**
     * @notice Update ML prediction for a pool (called by authorized ML service)
     * @param poolKey The pool identifier
     * @param prediction New ML prediction data
     */
    function updateMLPrediction(PoolKey calldata poolKey, MLPrediction calldata prediction) 
        external;
    
    /**
     * @notice Trigger emergency mode for a pool
     * @param poolKey The pool identifier
     * @param reason Reason for emergency activation
     */
    function activateEmergencyMode(PoolKey calldata poolKey, string memory reason) 
        external;
    
    /**
     * @notice Calculate optimal fee based on current market conditions
     * @param poolKey The pool identifier
     * @param baseVolatility Current volatility measure
     * @return optimalFee Recommended fee in basis points
     */
    function calculateOptimalFee(PoolKey calldata poolKey, uint256 baseVolatility) 
        external 
        view 
        returns (uint24 optimalFee);
    
    /**
     * @notice Check if position should be rebalanced
     * @param poolKey The pool identifier
     * @param tickLower Lower tick of position
     * @param tickUpper Upper tick of position
     * @return shouldRebalance Whether rebalancing is recommended
     * @return newLower Recommended new lower tick
     * @return newUpper Recommended new upper tick
     */
    function shouldRebalancePosition(
        PoolKey calldata poolKey,
        int24 tickLower,
        int24 tickUpper
    ) external view returns (bool shouldRebalance, int24 newLower, int24 newUpper);
    
    /**
     * @notice Get real-time capital efficiency for a position
     * @param poolKey The pool identifier
     * @param tickLower Lower tick of position
     * @param tickUpper Upper tick of position
     * @return efficiency Capital efficiency ratio (0-10000 bp)
     */
    function getCapitalEfficiency(
        PoolKey calldata poolKey,
        int24 tickLower,
        int24 tickUpper
    ) external view returns (uint256 efficiency);
}