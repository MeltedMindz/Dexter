// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {BaseHook} from "@uniswap/v4-periphery/src/utils/BaseHook.sol";
import {IPoolManager} from "@uniswap/v4-core/src/interfaces/IPoolManager.sol";
import {Hooks} from "@uniswap/v4-core/src/libraries/Hooks.sol";
import {PoolKey} from "@uniswap/v4-core/src/types/PoolKey.sol";
import {BeforeSwapDelta, BeforeSwapDeltaLibrary} from "@uniswap/v4-core/src/types/BeforeSwapDelta.sol";
import {BalanceDelta} from "@uniswap/v4-core/src/types/BalanceDelta.sol";
import {PoolId, PoolIdLibrary} from "@uniswap/v4-core/src/types/PoolId.sol";
import {Currency} from "@uniswap/v4-core/src/types/Currency.sol";

import {IDexterV4Hook} from "./interfaces/IDexterV4Hook.sol";
import {DexterMath} from "./libraries/DexterMath.sol";

/**
 * @title DexterV4Hook
 * @notice AI-powered Uniswap V4 hook for dynamic fee management and position optimization
 * @dev Implements real-time volatility analysis, ML-driven fee adjustments, and position rebalancing
 */
contract DexterV4Hook is BaseHook, IDexterV4Hook {
    using PoolIdLibrary for PoolKey;
    using BeforeSwapDeltaLibrary for BeforeSwapDelta;
    
    // State variables
    mapping(PoolId => PoolState) public poolStates;
    mapping(PoolId => MLPrediction) public mlPredictions;
    mapping(PoolId => uint256[]) public priceHistory;
    mapping(PoolId => uint256[]) public timeWeights;
    mapping(address => bool) public authorizedMLServices;
    
    address public immutable dexBrainService;
    address public owner;
    
    // Configuration
    uint256 public constant MAX_PRICE_HISTORY = 100;
    uint256 public constant ML_UPDATE_INTERVAL = 300; // 5 minutes
    uint256 public constant EMERGENCY_VOLATILITY_THRESHOLD = 5000; // 50% volatility
    uint256 public constant MIN_FEE_BP = 1; // 0.01%
    uint256 public constant MAX_FEE_BP = 10000; // 100%
    
    modifier onlyOwner() {
        require(msg.sender == owner, "DexterV4Hook: Not owner");
        _;
    }
    
    modifier onlyAuthorizedML() {
        require(authorizedMLServices[msg.sender], "DexterV4Hook: Not authorized ML service");
        _;
    }
    
    constructor(
        IPoolManager _poolManager,
        address _dexBrainService,
        address _owner
    ) BaseHook(_poolManager) {
        dexBrainService = _dexBrainService;
        owner = _owner;
        authorizedMLServices[_dexBrainService] = true;
    }
    
    /**
     * @notice Returns the hook permissions required
     */
    function getHookPermissions() public pure override returns (Hooks.Permissions memory) {
        return Hooks.Permissions({
            beforeInitialize: true,
            afterInitialize: false,
            beforeAddLiquidity: true,
            afterAddLiquidity: false,
            beforeRemoveLiquidity: true,
            afterRemoveLiquidity: false,
            beforeSwap: true,
            afterSwap: true,
            beforeDonate: false,
            afterDonate: false,
            beforeSwapReturnDelta: false,
            afterSwapReturnDelta: false,
            afterAddLiquidityReturnDelta: false,
            afterRemoveLiquidityReturnDelta: false
        });
    }
    
    /**
     * @notice Initialize pool with default state
     */
    function beforeInitialize(
        address,
        PoolKey calldata key,
        uint160,
        bytes calldata
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
        
        return BaseHook.beforeInitialize.selector;
    }
    
    /**
     * @notice Pre-swap hook: Update volatility and adjust fees if needed
     */
    function beforeSwap(
        address,
        PoolKey calldata key,
        IPoolManager.SwapParams calldata params,
        bytes calldata
    ) external override returns (bytes4, BeforeSwapDelta, uint24) {
        PoolId poolId = key.toId();
        PoolState storage state = poolStates[poolId];
        
        // Emergency mode check
        if (state.emergencyMode) {
            revert EmergencyModeActive();
        }
        
        // Update price history for volatility calculation
        _updatePriceHistory(key);
        
        // Calculate current volatility
        uint256 newVolatility = _calculateCurrentVolatility(poolId);
        
        // Check for emergency conditions
        if (newVolatility > EMERGENCY_VOLATILITY_THRESHOLD) {
            state.emergencyMode = true;
            emit EmergencyModeActivated(bytes32(PoolId.unwrap(poolId)), "High volatility detected");
            revert EmergencyModeActive();
        }
        
        // Update volatility if significant change
        if (_volatilityChangeSignificant(state.currentVolatility, newVolatility)) {
            state.currentVolatility = newVolatility;
            emit VolatilityUpdated(bytes32(PoolId.unwrap(poolId)), newVolatility, block.timestamp);
        }
        
        // Calculate optimal fee based on current conditions
        uint24 optimalFee = _calculateOptimalFeeInternal(key, newVolatility);
        
        // Adjust fee if significantly different
        if (_feeChangeSignificant(state.currentFee, optimalFee)) {
            uint24 oldFee = state.currentFee;
            state.currentFee = optimalFee;
            emit FeeAdjusted(
                bytes32(PoolId.unwrap(poolId)), 
                oldFee, 
                optimalFee, 
                "Volatility-based adjustment"
            );
        }
        
        return (BaseHook.beforeSwap.selector, BeforeSwapDeltaLibrary.ZERO_DELTA, state.currentFee);
    }
    
    /**
     * @notice Post-swap hook: Update statistics and trigger ML updates
     */
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
        
        // Trigger ML update if interval has passed
        if (block.timestamp >= state.lastMLUpdate + ML_UPDATE_INTERVAL) {
            _triggerMLUpdate(key);
            state.lastMLUpdate = block.timestamp;
        }
        
        return (BaseHook.afterSwap.selector, 0);
    }
    
    /**
     * @notice Pre-liquidity hook: Check for rebalancing opportunities
     */
    function beforeAddLiquidity(
        address,
        PoolKey calldata key,
        IPoolManager.ModifyLiquidityParams calldata params,
        bytes calldata
    ) external override returns (bytes4) {
        PoolId poolId = key.toId();
        PoolState storage state = poolStates[poolId];
        
        // Emergency mode check
        if (state.emergencyMode) {
            revert EmergencyModeActive();
        }
        
        // Check if position should be rebalanced
        (bool shouldRebalance, int24 newLower, int24 newUpper) = 
            shouldRebalancePosition(key, params.tickLower, params.tickUpper);
        
        if (shouldRebalance) {
            emit PositionRebalanceTriggered(
                bytes32(PoolId.unwrap(poolId)),
                msg.sender,
                params.tickLower,
                params.tickUpper,
                newLower,
                newUpper
            );
        }
        
        return BaseHook.beforeAddLiquidity.selector;
    }
    
    /**
     * @notice Pre-remove liquidity hook: Update position statistics
     */
    function beforeRemoveLiquidity(
        address,
        PoolKey calldata key,
        IPoolManager.ModifyLiquidityParams calldata,
        bytes calldata
    ) external override returns (bytes4) {
        PoolId poolId = key.toId();
        PoolState storage state = poolStates[poolId];
        
        // Emergency mode allows liquidity removal
        state.lastRebalance = block.timestamp;
        
        return BaseHook.beforeRemoveLiquidity.selector;
    }
    
    // IDexterV4Hook Implementation
    
    function getMarketRegime(PoolKey calldata key) 
        external 
        view 
        override 
        returns (MarketRegime regime, uint256 confidence) 
    {
        PoolId poolId = key.toId();
        MLPrediction memory prediction = mlPredictions[poolId];
        
        if (prediction.isValid && block.timestamp <= prediction.timestamp + ML_UPDATE_INTERVAL) {
            return (prediction.regime, prediction.confidence);
        }
        
        // Fallback to volatility-based regime detection
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
            activateEmergencyMode(key, "ML crisis detection");
        }
    }
    
    function activateEmergencyMode(PoolKey calldata key, string calldata reason) 
        public 
        override 
    {
        require(
            msg.sender == owner || authorizedMLServices[msg.sender],
            "DexterV4Hook: Not authorized"
        );
        
        PoolId poolId = key.toId();
        poolStates[poolId].emergencyMode = true;
        
        emit EmergencyModeActivated(bytes32(PoolId.unwrap(poolId)), reason);
    }
    
    function calculateOptimalFee(PoolKey calldata key, uint256 baseVolatility) 
        external 
        view 
        override 
        returns (uint24 optimalFee) 
    {
        return _calculateOptimalFeeInternal(key, baseVolatility);
    }
    
    function shouldRebalancePosition(
        PoolKey calldata key,
        int24 tickLower,
        int24 tickUpper
    ) public view override returns (bool shouldRebalance, int24 newLower, int24 newUpper) {
        PoolId poolId = key.toId();
        PoolState memory state = poolStates[poolId];
        
        // Get current tick from pool manager
        (, int24 currentTick,) = poolManager.getSlot0(poolId);
        
        // Check if rebalancing needed based on volatility threshold
        uint256 thresholdBp = state.currentVolatility / 10; // 1/10th of volatility as threshold
        shouldRebalance = DexterMath.shouldRebalance(tickLower, tickUpper, currentTick, thresholdBp);
        
        if (shouldRebalance) {
            // Calculate optimal new range
            int24 optimalSpacing = DexterMath.calculateOptimalTickSpacing(
                state.currentVolatility,
                9500 // 95% target concentration
            );
            
            newLower = currentTick - optimalSpacing / 2;
            newUpper = currentTick + optimalSpacing / 2;
        }
    }
    
    function getCapitalEfficiency(
        PoolKey calldata key,
        int24 tickLower,
        int24 tickUpper
    ) external view override returns (uint256 efficiency) {
        PoolId poolId = key.toId();
        
        // Get current tick from pool manager
        (, int24 currentTick,) = poolManager.getSlot0(poolId);
        
        // Use arbitrary liquidity for calculation (actual implementation would get real liquidity)
        uint128 liquidity = 1e18;
        
        return DexterMath.calculateCapitalEfficiency(tickLower, tickUpper, currentTick, liquidity);
    }
    
    // Internal functions
    
    function _updatePriceHistory(PoolKey calldata key) internal {
        PoolId poolId = key.toId();
        
        // Get current price from pool manager
        (uint160 sqrtPriceX96,,) = poolManager.getSlot0(poolId);
        uint256 currentPrice = _sqrtPriceX96ToPrice(sqrtPriceX96);
        
        // Add to price history
        uint256[] storage history = priceHistory[poolId];
        uint256[] storage weights = timeWeights[poolId];
        
        history.push(currentPrice);
        weights.push(block.timestamp);
        
        // Maintain maximum history size
        if (history.length > MAX_PRICE_HISTORY) {
            // Remove oldest entries
            for (uint256 i = 0; i < history.length - 1; i++) {
                history[i] = history[i + 1];
                weights[i] = weights[i + 1];
            }
            history.pop();
            weights.pop();
        }
    }
    
    function _calculateCurrentVolatility(PoolId poolId) internal view returns (uint256) {
        uint256[] memory history = priceHistory[poolId];
        uint256[] memory weights = timeWeights[poolId];
        
        if (history.length < 2) {
            return poolStates[poolId].currentVolatility; // Return current if insufficient data
        }
        
        // Create time-weighted array (more recent = higher weight)
        uint256[] memory timeWeights = new uint256[](weights.length);
        uint256 currentTime = block.timestamp;
        
        for (uint256 i = 0; i < weights.length; i++) {
            // Weight decreases exponentially with time
            uint256 age = currentTime - weights[i];
            timeWeights[i] = age < 3600 ? 100 - (age * 100 / 3600) : 1; // 1 hour decay
        }
        
        return DexterMath.calculateVolatility(history, timeWeights);
    }
    
    function _calculateOptimalFeeInternal(PoolKey calldata key, uint256 volatility) 
        internal 
        view 
        returns (uint24) 
    {
        PoolId poolId = key.toId();
        PoolState memory state = poolStates[poolId];
        
        // Get pool liquidity (simplified - real implementation would get accurate data)
        uint128 liquidity = 1e18; // Placeholder
        
        uint24 optimalFee = DexterMath.calculateOptimalFee(
            volatility,
            state.totalSwapVolume,
            liquidity
        );
        
        // Apply bounds
        if (optimalFee < MIN_FEE_BP) optimalFee = uint24(MIN_FEE_BP);
        if (optimalFee > MAX_FEE_BP) optimalFee = uint24(MAX_FEE_BP);
        
        return optimalFee;
    }
    
    function _volatilityChangeSignificant(uint256 oldVol, uint256 newVol) 
        internal 
        pure 
        returns (bool) 
    {
        uint256 threshold = 100; // 1% change threshold
        uint256 change = oldVol > newVol ? oldVol - newVol : newVol - oldVol;
        return change * 10000 / oldVol > threshold;
    }
    
    function _feeChangeSignificant(uint24 oldFee, uint24 newFee) 
        internal 
        pure 
        returns (bool) 
    {
        uint256 threshold = 10; // 10% change threshold
        uint256 change = oldFee > newFee ? oldFee - newFee : newFee - oldFee;
        return oldFee > 0 && (change * 100 / oldFee > threshold);
    }
    
    function _triggerMLUpdate(PoolKey calldata key) internal {
        // This would trigger an off-chain ML service update
        // Implementation would use events or direct service calls
        PoolId poolId = key.toId();
        emit VolatilityUpdated(
            bytes32(PoolId.unwrap(poolId)), 
            poolStates[poolId].currentVolatility, 
            block.timestamp
        );
    }
    
    function _sqrtPriceX96ToPrice(uint160 sqrtPriceX96) internal pure returns (uint256) {
        // Convert sqrt price to actual price
        // Simplified implementation - production would handle token decimals
        return uint256(sqrtPriceX96) * uint256(sqrtPriceX96) / (2**192);
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
        require(newOwner != address(0), "DexterV4Hook: Zero address");
        owner = newOwner;
    }
}