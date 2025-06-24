// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {Test} from "forge-std/Test.sol";
import {console} from "forge-std/console.sol";

import {IPoolManager} from "@uniswap/v4-core/src/interfaces/IPoolManager.sol";
import {PoolManager} from "@uniswap/v4-core/src/PoolManager.sol";
import {PoolKey} from "@uniswap/v4-core/src/types/PoolKey.sol";
import {Currency, CurrencyLibrary} from "@uniswap/v4-core/src/types/Currency.sol";
import {PoolId, PoolIdLibrary} from "@uniswap/v4-core/src/types/PoolId.sol";

import {SimpleDexterHook} from "../src/SimpleDexterHook.sol";
import {IDexterV4Hook} from "../src/interfaces/IDexterV4Hook.sol";

contract SimpleDexterHookTest is Test {
    using PoolIdLibrary for PoolKey;
    using CurrencyLibrary for Currency;

    // Test accounts
    address constant ALICE = address(0x1);
    address constant BOB = address(0x2);
    address constant DEXBRAIN_SERVICE = address(0x3);
    
    // Uniswap V4 core contracts
    PoolManager poolManager;
    
    // Dexter hook
    SimpleDexterHook hook;
    
    // Test currencies
    Currency currency0;
    Currency currency1;
    
    // Test pool
    PoolKey poolKey;
    PoolId poolId;
    
    function setUp() public {
        // Deploy PoolManager
        poolManager = new PoolManager();
        
        // Create test currencies
        currency0 = Currency.wrap(address(new MockERC20("Token0", "T0", 18)));
        currency1 = Currency.wrap(address(new MockERC20("Token1", "T1", 18)));
        
        // Ensure currency0 < currency1
        if (Currency.unwrap(currency0) > Currency.unwrap(currency1)) {
            (currency0, currency1) = (currency1, currency0);
        }
        
        // Deploy hook
        hook = new SimpleDexterHook(
            poolManager,
            DEXBRAIN_SERVICE,
            address(this)
        );
        
        // Create pool key
        poolKey = PoolKey({
            currency0: currency0,
            currency1: currency1,
            fee: 3000,
            tickSpacing: 60,
            hooks: hook
        });
        poolId = poolKey.toId();
    }
    
    function testHookInitialization() public {
        // Test that hook was deployed correctly
        assertEq(address(hook.poolManager()), address(poolManager));
        assertEq(hook.dexBrainService(), DEXBRAIN_SERVICE);
        assertEq(hook.owner(), address(this));
        assertTrue(hook.authorizedMLServices(DEXBRAIN_SERVICE));
    }
    
    function testBeforeInitialize() public {
        // Test pool initialization
        bytes4 selector = hook.beforeInitialize(
            address(this),
            poolKey,
            0
        );
        
        assertEq(selector, SimpleDexterHook.beforeInitialize.selector);
        
        // Check pool state was created
        IDexterV4Hook.PoolState memory state = hook.getPoolState(poolKey);
        assertEq(state.currentVolatility, 1000); // 10% default
        assertEq(state.currentFee, 3000); // Pool fee
        assertEq(uint8(state.currentRegime), uint8(IDexterV4Hook.MarketRegime.STABLE));
        assertFalse(state.emergencyMode);
    }
    
    function testMLPredictionUpdate() public {
        // Initialize pool first
        hook.beforeInitialize(address(this), poolKey, 0);
        
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
        assertEq(confidence, 7000); // Based on volatility, not ML prediction
    }
    
    function testEmergencyMode() public {
        // Initialize pool first
        hook.beforeInitialize(address(this), poolKey, 0);
        
        // Activate emergency mode
        vm.startPrank(DEXBRAIN_SERVICE);
        hook.activateEmergencyMode(poolKey, "Test emergency");
        vm.stopPrank();
        
        // Check emergency mode is active
        IDexterV4Hook.PoolState memory state = hook.getPoolState(poolKey);
        assertTrue(state.emergencyMode);
        
        // Try to add liquidity (should fail)
        vm.expectRevert(IDexterV4Hook.EmergencyModeActive.selector);
        hook.beforeAddLiquidity(
            ALICE,
            poolKey,
            IPoolManager.ModifyLiquidityParams({
                tickLower: -600,
                tickUpper: 600,
                liquidityDelta: 1000e18,
                salt: bytes32(0)
            }),
            ""
        );
        
        // Deactivate emergency mode as owner
        hook.deactivateEmergencyMode(poolKey);
        
        // Check emergency mode is deactivated
        state = hook.getPoolState(poolKey);
        assertFalse(state.emergencyMode);
    }
    
    function testFeeOptimization() public {
        uint256 volatility = 1500; // 15%
        uint24 optimalFee = hook.calculateOptimalFee(poolKey, volatility);
        
        // Fee should be reasonable for 15% volatility
        assertTrue(optimalFee >= 1); // At least 0.01%
        assertTrue(optimalFee <= 10000); // At most 100%
        
        console.log("Optimal fee for 15%% volatility: %d bp", optimalFee);
    }
    
    function testPositionRebalancing() public {
        // Check if rebalancing is needed for narrow range
        (bool shouldRebalance, int24 newLower, int24 newUpper) = 
            hook.shouldRebalancePosition(poolKey, -60, 60);
        
        // Narrow range should trigger rebalancing
        assertTrue(shouldRebalance);
        assertTrue(newUpper > newLower);
        console.log("Rebalance recommended: %d to %d", newLower, newUpper);
    }
    
    function testCapitalEfficiency() public {
        int24 tickLower = -600;
        int24 tickUpper = 600;
        
        uint256 efficiency = hook.getCapitalEfficiency(poolKey, tickLower, tickUpper);
        
        // Should return some efficiency value
        assertTrue(efficiency <= 10000); // Max 100%
        console.log("Capital efficiency: %d bp", efficiency);
    }
    
    function testUnauthorizedMLUpdate() public {
        // Initialize pool first
        hook.beforeInitialize(address(this), poolKey, 0);
        
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
        vm.expectRevert("SimpleDexterHook: Not authorized ML service");
        hook.updateMLPrediction(poolKey, prediction);
        vm.stopPrank();
    }
    
    function testOwnershipTransfer() public {
        address newOwner = address(0x999);
        
        // Transfer ownership
        hook.transferOwnership(newOwner);
        assertEq(hook.owner(), newOwner);
        
        // Only new owner can transfer again
        vm.startPrank(newOwner);
        hook.transferOwnership(address(this));
        vm.stopPrank();
        
        assertEq(hook.owner(), address(this));
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