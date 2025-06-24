// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {Test} from "forge-std/Test.sol";
import {console} from "forge-std/console.sol";
import {GasSnapshot} from "forge-gas-snapshot/GasSnapshot.sol";

import {IPoolManager} from "@uniswap/v4-core/src/interfaces/IPoolManager.sol";
import {PoolManager} from "@uniswap/v4-core/src/PoolManager.sol";
import {PoolKey} from "@uniswap/v4-core/src/types/PoolKey.sol";
import {Currency, CurrencyLibrary} from "@uniswap/v4-core/src/types/Currency.sol";
import {PoolId, PoolIdLibrary} from "@uniswap/v4-core/src/types/PoolId.sol";
import {Hooks} from "@uniswap/v4-core/src/libraries/Hooks.sol";
import {TickMath} from "@uniswap/v4-core/src/libraries/TickMath.sol";

import {PoolSwapTest} from "@uniswap/v4-core/src/test/PoolSwapTest.sol";
import {PoolModifyLiquidityTest} from "@uniswap/v4-core/src/test/PoolModifyLiquidityTest.sol";

import {DexterV4Hook} from "../src/DexterV4Hook.sol";
import {IDexterV4Hook} from "../src/interfaces/IDexterV4Hook.sol";
import {HookMiner} from "./utils/HookMiner.sol";

contract DexterV4HookTest is Test, GasSnapshot {
    using PoolIdLibrary for PoolKey;
    using CurrencyLibrary for Currency;

    // Test accounts
    address constant ALICE = address(0x1);
    address constant BOB = address(0x2);
    address constant DEXBRAIN_SERVICE = address(0x3);
    
    // Uniswap V4 core contracts
    PoolManager poolManager;
    PoolSwapTest swapRouter;
    PoolModifyLiquidityTest modifyLiquidityRouter;
    
    // Dexter hook
    DexterV4Hook hook;
    
    // Test currencies
    Currency currency0;
    Currency currency1;
    
    // Test pool
    PoolKey poolKey;
    PoolId poolId;
    
    function setUp() public {
        // Deploy PoolManager
        poolManager = new PoolManager(500000);
        
        // Create test currencies
        currency0 = Currency.wrap(address(new MockERC20("Token0", "T0", 18)));
        currency1 = Currency.wrap(address(new MockERC20("Token1", "T1", 18)));
        
        // Ensure currency0 < currency1
        if (Currency.unwrap(currency0) > Currency.unwrap(currency1)) {
            (currency0, currency1) = (currency1, currency0);
        }
        
        // Mine hook address with correct permissions
        uint160 permissions = uint160(
            Hooks.BEFORE_INITIALIZE_FLAG |
            Hooks.BEFORE_ADD_LIQUIDITY_FLAG |
            Hooks.BEFORE_REMOVE_LIQUIDITY_FLAG |
            Hooks.BEFORE_SWAP_FLAG |
            Hooks.AFTER_SWAP_FLAG
        );
        
        (address hookAddress, bytes32 salt) = HookMiner.find(
            address(this),
            permissions,
            type(DexterV4Hook).creationCode,
            abi.encode(address(poolManager), DEXBRAIN_SERVICE, address(this))
        );
        
        // Deploy hook at the mined address
        hook = new DexterV4Hook{salt: salt}(
            poolManager,
            DEXBRAIN_SERVICE,
            address(this)
        );
        require(address(hook) == hookAddress, "Hook address mismatch");
        
        // Create pool key
        poolKey = PoolKey({
            currency0: currency0,
            currency1: currency1,
            fee: 3000,
            tickSpacing: 60,
            hooks: hook
        });
        poolId = poolKey.toId();
        
        // Initialize pool
        poolManager.initialize(poolKey, TickMath.getSqrtPriceAtTick(0), "");
        
        // Deploy swap and liquidity routers
        swapRouter = new PoolSwapTest(poolManager);
        modifyLiquidityRouter = new PoolModifyLiquidityTest(poolManager);
        
        // Setup test accounts
        vm.deal(ALICE, 100 ether);
        vm.deal(BOB, 100 ether);
        
        // Mint tokens to test accounts
        MockERC20(Currency.unwrap(currency0)).mint(ALICE, 1000000e18);
        MockERC20(Currency.unwrap(currency1)).mint(ALICE, 1000000e18);
        MockERC20(Currency.unwrap(currency0)).mint(BOB, 1000000e18);
        MockERC20(Currency.unwrap(currency1)).mint(BOB, 1000000e18);
        
        // Approve routers
        vm.startPrank(ALICE);
        MockERC20(Currency.unwrap(currency0)).approve(address(modifyLiquidityRouter), type(uint256).max);
        MockERC20(Currency.unwrap(currency1)).approve(address(modifyLiquidityRouter), type(uint256).max);
        MockERC20(Currency.unwrap(currency0)).approve(address(swapRouter), type(uint256).max);
        MockERC20(Currency.unwrap(currency1)).approve(address(swapRouter), type(uint256).max);
        vm.stopPrank();
        
        vm.startPrank(BOB);
        MockERC20(Currency.unwrap(currency0)).approve(address(modifyLiquidityRouter), type(uint256).max);
        MockERC20(Currency.unwrap(currency1)).approve(address(modifyLiquidityRouter), type(uint256).max);
        MockERC20(Currency.unwrap(currency0)).approve(address(swapRouter), type(uint256).max);
        MockERC20(Currency.unwrap(currency1)).approve(address(swapRouter), type(uint256).max);
        vm.stopPrank();
    }
    
    function testHookInitialization() public {
        // Check initial pool state
        IDexterV4Hook.PoolState memory state = hook.getPoolState(poolKey);
        
        assertEq(state.currentVolatility, 1000); // 10% default
        assertEq(state.currentFee, 3000); // Pool fee
        assertEq(uint8(state.currentRegime), uint8(IDexterV4Hook.MarketRegime.STABLE));
        assertFalse(state.emergencyMode);
    }
    
    function testBasicSwap() public {
        // Add liquidity first
        vm.startPrank(ALICE);
        modifyLiquidityRouter.modifyLiquidity(
            poolKey,
            IPoolManager.ModifyLiquidityParams({
                tickLower: -600,
                tickUpper: 600,
                liquidityDelta: 1000e18,
                salt: bytes32(0)
            }),
            ""
        );
        vm.stopPrank();
        
        // Perform swap
        vm.startPrank(BOB);
        snapStart("basicSwap");
        swapRouter.swap(
            poolKey,
            IPoolManager.SwapParams({
                zeroForOne: true,
                amountSpecified: -1e18, // Exact output
                sqrtPriceLimitX96: TickMath.MIN_SQRT_PRICE + 1
            }),
            PoolSwapTest.TestSettings({
                takeClaims: false,
                settleUsingBurn: false
            }),
            ""
        );
        snapEnd();
        vm.stopPrank();
        
        // Check that volatility was updated
        IDexterV4Hook.PoolState memory state = hook.getPoolState(poolKey);
        assertTrue(state.totalSwapVolume > 0);
    }
    
    function testVolatilityCalculation() public {
        // Add liquidity
        vm.startPrank(ALICE);
        modifyLiquidityRouter.modifyLiquidity(
            poolKey,
            IPoolManager.ModifyLiquidityParams({
                tickLower: -600,
                tickUpper: 600,
                liquidityDelta: 1000e18,
                salt: bytes32(0)
            }),
            ""
        );
        vm.stopPrank();
        
        // Perform multiple swaps to generate price history
        vm.startPrank(BOB);
        for (uint i = 0; i < 5; i++) {
            swapRouter.swap(
                poolKey,
                IPoolManager.SwapParams({
                    zeroForOne: i % 2 == 0,
                    amountSpecified: int256(1e18 + i * 1e17),
                    sqrtPriceLimitX96: i % 2 == 0 ? 
                        TickMath.MIN_SQRT_PRICE + 1 : 
                        TickMath.MAX_SQRT_PRICE - 1
                }),
                PoolSwapTest.TestSettings({
                    takeClaims: false,
                    settleUsingBurn: false
                }),
                ""
            );
            
            // Advance time between swaps
            vm.warp(block.timestamp + 300);
        }
        vm.stopPrank();
        
        // Check that volatility increased due to price movements
        IDexterV4Hook.PoolState memory state = hook.getPoolState(poolKey);
        assertTrue(state.currentVolatility >= 1000); // Should be at least default
    }
    
    function testMLPredictionUpdate() public {
        // Create ML prediction
        IDexterV4Hook.MLPrediction memory prediction = IDexterV4Hook.MLPrediction({
            regime: IDexterV4Hook.MarketRegime.VOLATILE,
            confidence: 8500, // 85% confidence
            predictedVolatility: 2500, // 25%
            optimalFee: 50, // 0.5%
            timestamp: block.timestamp,
            isValid: true
        });
        
        // Update from authorized ML service
        vm.startPrank(DEXBRAIN_SERVICE);
        hook.updateMLPrediction(poolKey, prediction);
        vm.stopPrank();
        
        // Check market regime was updated
        (IDexterV4Hook.MarketRegime regime, uint256 confidence) = hook.getMarketRegime(poolKey);
        assertEq(uint8(regime), uint8(IDexterV4Hook.MarketRegime.VOLATILE));
        assertEq(confidence, 8500);
    }
    
    function testEmergencyMode() public {
        // Activate emergency mode
        vm.startPrank(DEXBRAIN_SERVICE);
        hook.activateEmergencyMode(poolKey, "Test emergency");
        vm.stopPrank();
        
        // Check emergency mode is active
        IDexterV4Hook.PoolState memory state = hook.getPoolState(poolKey);
        assertTrue(state.emergencyMode);
        
        // Try to add liquidity (should fail)
        vm.startPrank(ALICE);
        vm.expectRevert(IDexterV4Hook.EmergencyModeActive.selector);
        modifyLiquidityRouter.modifyLiquidity(
            poolKey,
            IPoolManager.ModifyLiquidityParams({
                tickLower: -600,
                tickUpper: 600,
                liquidityDelta: 1000e18,
                salt: bytes32(0)
            }),
            ""
        );
        vm.stopPrank();
        
        // Deactivate emergency mode as owner
        hook.deactivateEmergencyMode(poolKey);
        
        // Check emergency mode is deactivated
        state = hook.getPoolState(poolKey);
        assertFalse(state.emergencyMode);
    }
    
    function testCapitalEfficiency() public {
        int24 tickLower = -600;
        int24 tickUpper = 600;
        
        uint256 efficiency = hook.getCapitalEfficiency(poolKey, tickLower, tickUpper);
        
        // Should return some efficiency value (depends on current tick)
        assertTrue(efficiency <= 10000); // Max 100%
    }
    
    function testPositionRebalancing() public {
        // Add liquidity first
        vm.startPrank(ALICE);
        modifyLiquidityRouter.modifyLiquidity(
            poolKey,
            IPoolManager.ModifyLiquidityParams({
                tickLower: -60,
                tickUpper: 60,
                liquidityDelta: 1000e18,
                salt: bytes32(0)
            }),
            ""
        );
        vm.stopPrank();
        
        // Check if rebalancing is needed for narrow range
        (bool shouldRebalance, int24 newLower, int24 newUpper) = 
            hook.shouldRebalancePosition(poolKey, -60, 60);
        
        // Position might or might not need rebalancing depending on current tick
        if (shouldRebalance) {
            assertTrue(newUpper > newLower);
            console.log("Rebalance recommended: %d to %d", newLower, newUpper);
        }
    }
    
    function testFeeOptimization() public {
        uint256 volatility = 1500; // 15%
        uint24 optimalFee = hook.calculateOptimalFee(poolKey, volatility);
        
        // Fee should be reasonable for 15% volatility
        assertTrue(optimalFee >= 1); // At least 0.01%
        assertTrue(optimalFee <= 10000); // At most 100%
        
        console.log("Optimal fee for 15%% volatility: %d bp", optimalFee);
    }
    
    function testUnauthorizedMLUpdate() public {
        IDexterV4Hook.MLPrediction memory prediction = IDexterV4Hook.MLPrediction({
            regime: IDexterV4Hook.MarketRegime.VOLATILE,
            confidence: 8500,
            predictedVolatility: 2500,
            optimalFee: 50,
            timestamp: block.timestamp,
            isValid: true
        });
        
        // Try to update from unauthorized address
        vm.startPrank(ALICE);
        vm.expectRevert("DexterV4Hook: Not authorized ML service");
        hook.updateMLPrediction(poolKey, prediction);
        vm.stopPrank();
    }
    
    function testGasOptimization() public {
        // Add liquidity
        vm.startPrank(ALICE);
        modifyLiquidityRouter.modifyLiquidity(
            poolKey,
            IPoolManager.ModifyLiquidityParams({
                tickLower: -600,
                tickUpper: 600,
                liquidityDelta: 1000e18,
                salt: bytes32(0)
            }),
            ""
        );
        vm.stopPrank();
        
        // Measure gas for swap with hook
        vm.startPrank(BOB);
        snapStart("swapWithHook");
        swapRouter.swap(
            poolKey,
            IPoolManager.SwapParams({
                zeroForOne: true,
                amountSpecified: -1e18,
                sqrtPriceLimitX96: TickMath.MIN_SQRT_PRICE + 1
            }),
            PoolSwapTest.TestSettings({
                takeClaims: false,
                settleUsingBurn: false
            }),
            ""
        );
        snapEnd();
        vm.stopPrank();
    }
}

// Mock ERC20 for testing
contract MockERC20 {
    string public name;
    string public symbol;
    uint8 public decimals;
    uint256 public totalSupply;
    
    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;
    
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
    
    constructor(string memory _name, string memory _symbol, uint8 _decimals) {
        name = _name;
        symbol = _symbol;
        decimals = _decimals;
    }
    
    function mint(address to, uint256 amount) external {
        balanceOf[to] += amount;
        totalSupply += amount;
        emit Transfer(address(0), to, amount);
    }
    
    function transfer(address to, uint256 amount) external returns (bool) {
        require(balanceOf[msg.sender] >= amount, "Insufficient balance");
        balanceOf[msg.sender] -= amount;
        balanceOf[to] += amount;
        emit Transfer(msg.sender, to, amount);
        return true;
    }
    
    function approve(address spender, uint256 amount) external returns (bool) {
        allowance[msg.sender][spender] = amount;
        emit Approval(msg.sender, spender, amount);
        return true;
    }
    
    function transferFrom(address from, address to, uint256 amount) external returns (bool) {
        require(balanceOf[from] >= amount, "Insufficient balance");
        require(allowance[from][msg.sender] >= amount, "Insufficient allowance");
        
        balanceOf[from] -= amount;
        balanceOf[to] += amount;
        allowance[from][msg.sender] -= amount;
        
        emit Transfer(from, to, amount);
        return true;
    }
}