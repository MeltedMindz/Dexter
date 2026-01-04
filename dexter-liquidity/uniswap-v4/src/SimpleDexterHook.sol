// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {IHooks} from "@uniswap/v4-core/src/interfaces/IHooks.sol";
import {IPoolManager} from "@uniswap/v4-core/src/interfaces/IPoolManager.sol";
import {PoolKey} from "@uniswap/v4-core/src/types/PoolKey.sol";
import {BeforeSwapDelta, BeforeSwapDeltaLibrary} from "@uniswap/v4-core/src/types/BeforeSwapDelta.sol";
import {BalanceDelta} from "@uniswap/v4-core/src/types/BalanceDelta.sol";
import {PoolId, PoolIdLibrary} from "@uniswap/v4-core/src/types/PoolId.sol";
import {Currency} from "@uniswap/v4-core/src/types/Currency.sol";
import {ReentrancyGuard} from "@openzeppelin/contracts/security/ReentrancyGuard.sol";

import {IDexterV4Hook} from "./interfaces/IDexterV4Hook.sol";
import {DexterMath} from "./libraries/DexterMath.sol";

/**
 * @title SimpleDexterHook
 * @notice Simplified AI-powered Uniswap V4 hook for dynamic fee management
 * @dev Implementation with reentrancy protection and secure state management
 */
contract SimpleDexterHook is IHooks, IDexterV4Hook, ReentrancyGuard {
    using PoolIdLibrary for PoolKey;
    using BeforeSwapDeltaLibrary for BeforeSwapDelta;
    
    // State variables
    mapping(PoolId => PoolState) public poolStates;
    mapping(PoolId => MLPrediction) public mlPredictions;
    mapping(PoolId => uint256[]) public priceHistory;
    mapping(PoolId => uint256[]) public timeWeights;
    mapping(address => bool) public authorizedMLServices;
    
    IPoolManager public immutable poolManager;
    address public immutable dexBrainService;
    address public owner;
    
    // Configuration
    uint256 public constant MAX_PRICE_HISTORY = 100;
    uint256 public constant ML_UPDATE_INTERVAL = 300; // 5 minutes
    uint256 public constant EMERGENCY_VOLATILITY_THRESHOLD = 5000; // 50% volatility
    uint256 public constant MIN_FEE_BP = 1; // 0.01%
    uint256 public constant MAX_FEE_BP = 10000; // 100%
    
    modifier onlyOwner() {
        require(msg.sender == owner, "SimpleDexterHook: Not owner");
        _;
    }
    
    modifier onlyAuthorizedML() {
        require(authorizedMLServices[msg.sender], "SimpleDexterHook: Not authorized ML service");
        _;
    }
    
    constructor(
        IPoolManager _poolManager,
        address _dexBrainService,
        address _owner
    ) {
        poolManager = _poolManager;
        dexBrainService = _dexBrainService;
        owner = _owner;
        authorizedMLServices[_dexBrainService] = true;
    }
    
    // IHooks implementation
    function beforeInitialize(
        address,
        PoolKey calldata key,
        uint160
    ) external override returns (bytes4) {
        PoolId poolId = key.toId();
        
        poolStates[poolId] = PoolState({
            currentVolatility: 1000, // 10% default
            currentFee: key.fee,
            currentRegime: MarketRegime.STABLE,
            lastMLUpdate: block.timestamp,
            lastRebalance: block.timestamp,
            emergencyMode: false,
            totalSwapVolume: 0,
            avgPositionSize: 0
        });
        
        emit VolatilityUpdated(bytes32(PoolId.unwrap(poolId)), 1000, block.timestamp);
        
        return IHooks.beforeInitialize.selector;
    }

    function afterInitialize(
        address,
        PoolKey calldata,
        uint160,
        int24
    ) external override returns (bytes4) {
        return IHooks.afterInitialize.selector;
    }

    function beforeAddLiquidity(
        address,
        PoolKey calldata key,
        IPoolManager.ModifyLiquidityParams calldata,
        bytes calldata
    ) external override returns (bytes4) {
        PoolId poolId = key.toId();
        PoolState storage state = poolStates[poolId];
        
        // Emergency mode check
        if (state.emergencyMode) {
            revert EmergencyModeActive();
        }
        
        return IHooks.beforeAddLiquidity.selector;
    }

    function afterAddLiquidity(
        address,
        PoolKey calldata,
        IPoolManager.ModifyLiquidityParams calldata,
        BalanceDelta,
        BalanceDelta,
        bytes calldata
    ) external override returns (bytes4, BalanceDelta) {
        return (IHooks.afterAddLiquidity.selector, BalanceDelta.wrap(0));
    }

    function beforeRemoveLiquidity(
        address,
        PoolKey calldata,
        IPoolManager.ModifyLiquidityParams calldata,
        bytes calldata
    ) external override returns (bytes4) {
        return IHooks.beforeRemoveLiquidity.selector;
    }

    function afterRemoveLiquidity(
        address,
        PoolKey calldata,
        IPoolManager.ModifyLiquidityParams calldata,
        BalanceDelta,
        BalanceDelta,
        bytes calldata
    ) external override returns (bytes4, BalanceDelta) {
        return (IHooks.afterRemoveLiquidity.selector, BalanceDelta.wrap(0));
    }

    function beforeSwap(
        address,
        PoolKey calldata key,
        IPoolManager.SwapParams calldata,
        bytes calldata
    ) external override nonReentrant returns (bytes4, BeforeSwapDelta, uint24) {
        PoolId poolId = key.toId();
        
        // CHECKS: Validate input and current state
        _validateSwapConditions(poolId);
        
        // EFFECTS: Calculate new values without modifying state yet
        (uint256 newVolatility, uint24 optimalFee, bool emergencyTriggered) = _calculateSwapParameters(poolId);
        
        // EFFECTS: Update state only after all calculations are complete
        _updatePoolState(poolId, newVolatility, optimalFee, emergencyTriggered);
        
        // INTERACTIONS: Emit events (safe after state updates)
        emit VolatilityUpdated(bytes32(PoolId.unwrap(poolId)), newVolatility, block.timestamp);
        
        if (emergencyTriggered) {
            emit EmergencyModeActivated(bytes32(PoolId.unwrap(poolId)), "High volatility detected");
            revert EmergencyModeActive();
        }
        
        return (IHooks.beforeSwap.selector, BeforeSwapDeltaLibrary.ZERO_DELTA, optimalFee);
    }

    function afterSwap(
        address,
        PoolKey calldata key,
        IPoolManager.SwapParams calldata params,
        BalanceDelta,
        bytes calldata
    ) external override returns (bytes4, int128) {
        PoolId poolId = key.toId();
        PoolState storage state = poolStates[poolId];
        
        // Update swap volume
        uint256 swapAmount = params.amountSpecified > 0 ? 
            uint256(params.amountSpecified) : 
            uint256(-params.amountSpecified);
        state.totalSwapVolume += swapAmount;
        
        return (IHooks.afterSwap.selector, 0);
    }

    function beforeDonate(
        address,
        PoolKey calldata,
        uint256,
        uint256,
        bytes calldata
    ) external override returns (bytes4) {
        return IHooks.beforeDonate.selector;
    }

    function afterDonate(
        address,
        PoolKey calldata,
        uint256,
        uint256,
        bytes calldata
    ) external override returns (bytes4) {
        return IHooks.afterDonate.selector;
    }
    
    // IDexterV4Hook implementation
    function getMarketRegime(PoolKey calldata key) 
        external 
        view 
        override 
        returns (MarketRegime regime, uint256 confidence) 
    {
        PoolId poolId = key.toId();
        PoolState memory state = poolStates[poolId];
        
        if (state.currentVolatility < 500) { // < 5%
            return (MarketRegime.STABLE, 8000);
        } else if (state.currentVolatility > 2000) { // > 20%
            return (MarketRegime.VOLATILE, 7000);
        } else {
            return (MarketRegime.RANGING, 6000);
        }
    }
    
    function getPoolState(PoolKey calldata key) 
        external 
        view 
        override 
        returns (PoolState memory state) 
    {
        PoolId poolId = key.toId();
        return poolStates[poolId];
    }
    
    function updateMLPrediction(PoolKey calldata key, MLPrediction calldata prediction) 
        external 
        override 
        onlyAuthorizedML 
    {
        PoolId poolId = key.toId();
        mlPredictions[poolId] = prediction;
        
        emit MLPredictionReceived(
            bytes32(PoolId.unwrap(poolId)),
            uint8(prediction.regime),
            prediction.confidence,
            prediction.timestamp
        );
        
        // Update pool state based on ML prediction
        PoolState storage state = poolStates[poolId];
        state.currentRegime = prediction.regime;
        
        // Activate emergency mode if crisis detected
        if (prediction.regime == MarketRegime.CRISIS && prediction.confidence > 8000) {
            string memory reason = "ML crisis detection";
            activateEmergencyMode(key, reason);
        }
    }
    
    function activateEmergencyMode(PoolKey calldata key, string memory reason) 
        public 
        override 
    {
        require(
            msg.sender == owner || authorizedMLServices[msg.sender],
            "SimpleDexterHook: Not authorized"
        );
        
        PoolId poolId = key.toId();
        poolStates[poolId].emergencyMode = true;
        
        emit EmergencyModeActivated(bytes32(PoolId.unwrap(poolId)), reason);
    }
    
    function calculateOptimalFee(PoolKey calldata, uint256 baseVolatility) 
        external 
        pure
        override 
        returns (uint24 optimalFee) 
    {
        return _calculateOptimalFeeSimple(baseVolatility);
    }
    
    function shouldRebalancePosition(
        PoolKey calldata,
        int24 tickLower,
        int24 tickUpper
    ) external pure override returns (bool shouldRebalance, int24 newLower, int24 newUpper) {
        // Simplified rebalancing logic
        int24 range = tickUpper - tickLower;
        shouldRebalance = range < 120; // Narrow range threshold
        
        if (shouldRebalance) {
            int24 center = (tickLower + tickUpper) / 2;
            newLower = center - 300; // Wider range
            newUpper = center + 300;
        }
    }
    
    function getCapitalEfficiency(
        PoolKey calldata,
        int24 tickLower,
        int24 tickUpper
    ) external pure override returns (uint256 efficiency) {
        // Simplified capital efficiency calculation
        uint256 rangeWidth = uint256(uint24(tickUpper - tickLower));
        efficiency = rangeWidth > 0 ? 10000 / rangeWidth : 0; // Inverse of range width
        if (efficiency > 10000) efficiency = 10000; // Cap at 100%
    }
    
    // Internal helper functions
    function _calculateSimpleVolatility(PoolId poolId) internal view returns (uint256) {
        uint256[] memory history = priceHistory[poolId];
        if (history.length < 2) {
            return poolStates[poolId].currentVolatility; // Return current if insufficient data
        }
        
        // Simple volatility calculation based on recent price changes
        uint256 totalChange = 0;
        for (uint256 i = 1; i < history.length; i++) {
            uint256 change = history[i] > history[i-1] ? 
                history[i] - history[i-1] : 
                history[i-1] - history[i];
            totalChange += change * 10000 / history[i-1]; // Percentage change in bp
        }
        
        return totalChange / (history.length - 1); // Average percentage change
    }
    
    function _calculateOptimalFeeSimple(uint256 volatility) internal pure returns (uint24) {
        // Base fee proportional to volatility
        uint256 baseFee = volatility / 10; // 0.1% fee for 10% volatility
        
        // Clamp to reasonable bounds
        if (baseFee < MIN_FEE_BP) return uint24(MIN_FEE_BP);
        if (baseFee > MAX_FEE_BP) return uint24(MAX_FEE_BP);
        
        return uint24(baseFee);
    }
    
    // Admin functions
    function setMLServiceAuthorization(address service, bool authorized) external onlyOwner {
        authorizedMLServices[service] = authorized;
    }
    
    function deactivateEmergencyMode(PoolKey calldata key) external onlyOwner {
        PoolId poolId = key.toId();
        poolStates[poolId].emergencyMode = false;
    }
    
    function transferOwnership(address newOwner) external onlyOwner {
        require(newOwner != address(0), "SimpleDexterHook: Zero address");
        owner = newOwner;
    }

    // ============ REENTRANCY PROTECTION HELPERS ============
    
    /**
     * @notice Validates swap conditions (CHECKS phase)
     * @param poolId Pool identifier
     */
    function _validateSwapConditions(PoolId poolId) internal view {
        // Check if pool is in emergency mode
        if (poolStates[poolId].emergencyMode) {
            revert EmergencyModeActive();
        }
        
        // Additional validation checks can be added here
        // e.g., pool exists, authorized caller, etc.
    }
    
    /**
     * @notice Calculates swap parameters without state changes (EFFECTS phase - read-only)
     * @param poolId Pool identifier
     * @return newVolatility Calculated volatility
     * @return optimalFee Calculated optimal fee
     * @return emergencyTriggered Whether emergency mode should be triggered
     */
    function _calculateSwapParameters(PoolId poolId) 
        internal 
        view 
        returns (uint256 newVolatility, uint24 optimalFee, bool emergencyTriggered) 
    {
        // Gas-optimized volatility calculation
        assembly {
            // Load current volatility from storage
            let stateSlot := keccak256(add(poolId, poolStates.slot))
            let currentVol := sload(stateSlot) // currentVolatility is first field
            
            // Simple volatility update (gas-optimized)
            newVolatility := currentVol
            
            // Calculate optimal fee: fee = volatility / 10, clamped to [1, 10000]
            optimalFee := div(newVolatility, 10)
            if lt(optimalFee, 1) { optimalFee := 1 }
            if gt(optimalFee, 10000) { optimalFee := 10000 }
        }
        
        // Check if emergency mode should be triggered
        emergencyTriggered = newVolatility > EMERGENCY_VOLATILITY_THRESHOLD;
    }
    
    /**
     * @notice Updates pool state (EFFECTS phase - state modifications)
     * @param poolId Pool identifier
     * @param newVolatility New volatility value
     * @param optimalFee New optimal fee
     * @param emergencyTriggered Whether to activate emergency mode
     */
    function _updatePoolState(
        PoolId poolId, 
        uint256 newVolatility, 
        uint24 optimalFee, 
        bool emergencyTriggered
    ) internal {
        // Update pool state in a single operation to minimize gas and ensure atomicity
        PoolState storage state = poolStates[poolId];
        
        // Use assembly for gas-optimized state updates
        assembly {
            // Get storage slot for the pool state
            let stateSlot := keccak256(add(poolId, poolStates.slot))
            
            // Update volatility and fee in storage
            sstore(stateSlot, newVolatility) // Update currentVolatility
            sstore(add(stateSlot, 1), optimalFee) // Update currentFee
        }
        
        // Update emergency mode if triggered (separate from assembly for clarity)
        if (emergencyTriggered) {
            state.emergencyMode = true;
        }
    }
    
    /**
     * @notice Additional reentrancy protection for ML prediction updates
     * @param key Pool key
     * @param prediction ML prediction data
     */
    function updateMLPrediction(PoolKey calldata key, MLPrediction calldata prediction) 
        external 
        override 
        onlyAuthorizedML 
        nonReentrant 
    {
        PoolId poolId = key.toId();
        
        // CHECKS: Validate prediction data
        require(prediction.confidence <= 10000, "Invalid confidence");
        require(prediction.timestamp <= block.timestamp, "Future timestamp");
        require(prediction.timestamp > block.timestamp - ML_UPDATE_INTERVAL, "Stale prediction");
        
        // EFFECTS: Update ML prediction state
        mlPredictions[poolId] = prediction;
        
        // INTERACTIONS: Emit events
        emit MLPredictionReceived(
            bytes32(PoolId.unwrap(poolId)),
            uint8(prediction.regime),
            prediction.confidence,
            prediction.timestamp
        );
        
        // Activate emergency mode if crisis detected with high confidence
        if (prediction.regime == MarketRegime.CRISIS && prediction.confidence > 8000) {
            poolStates[poolId].emergencyMode = true;
            emit EmergencyModeActivated(bytes32(PoolId.unwrap(poolId)), "ML crisis detection");
        }
    }
    
    /**
     * @notice Enhanced emergency mode activation with reentrancy protection
     * @param key Pool key
     * @param reason Reason for activation
     */
    function activateEmergencyMode(PoolKey calldata key, string memory reason) 
        public 
        override 
        nonReentrant 
    {
        require(
            msg.sender == owner || authorizedMLServices[msg.sender],
            "SimpleDexterHook: Not authorized"
        );
        
        PoolId poolId = key.toId();
        
        // CHECKS: Validate current state
        require(!poolStates[poolId].emergencyMode, "Emergency mode already active");
        
        // EFFECTS: Update state
        poolStates[poolId].emergencyMode = true;
        
        // INTERACTIONS: Emit events
        emit EmergencyModeActivated(bytes32(PoolId.unwrap(poolId)), reason);
    }
}